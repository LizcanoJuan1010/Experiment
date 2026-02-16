"""
Vision Aphasia Test — BLIP Image Captioning with Concept Ablation
==================================================================
Loads BLIP (Salesforce/blip-image-captioning-base), asks it to describe
an image, then ablates the "tool" concept direction from the text decoder
and observes how the caption changes.

Sweeps all factor combinations: layers, techniques, alphas.

Usage:
    python vision_aphasia_test.py

Output:
    results/vision_aphasia_test_results.json
"""

import torch
import json
import os
from PIL import Image
from functools import partial
from itertools import product
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Salesforce/blip-image-captioning-base"
RESULTS_DIR = "results"

# Images
TOOL_IMAGE = "images/tools/hammer.png"       # Target — tools concept
CONTROL_IMAGE = "images/cow/image.png"        # Control — animal (no tools)

# Questions to ask the model
QUESTIONS = [
    "What is in this image?",
    "Describe this image in detail.",
    "What objects do you see?",
]

# Ablation factors
ABLATION_LAYERS = [3, 6, 9, 11]
TECHNIQUES = ["subtraction", "projection"]
ALPHAS = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0]

MAX_NEW_TOKENS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CAV extraction from BLIP
# ---------------------------------------------------------------------------
def extract_vision_cav(model, processor, tool_img_path, ctrl_img_path, layer_idx):
    """Extract a concept direction from BLIP's text decoder.

    Feeds both images with the same prompt, captures activations at the
    specified decoder layer, and returns the normalized difference vector.
    """
    prompt = "What is in this image?"

    def get_activation(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(img, text=prompt, return_tensors="pt").to(DEVICE)
        activation = {}

        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                activation["act"] = output[0].detach()
            else:
                activation["act"] = output.detach()
            return output

        layer_module = model.text_decoder.bert.encoder.layer[layer_idx].output
        handle = layer_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)

        handle.remove()
        # Mean pool over sequence tokens -> [768]
        return activation["act"].mean(dim=1).squeeze()

    tool_vec = get_activation(tool_img_path)
    ctrl_vec = get_activation(ctrl_img_path)

    diff = tool_vec - ctrl_vec
    cav = diff / diff.norm()
    return cav


# ---------------------------------------------------------------------------
# Ablation hooks for HuggingFace models
# ---------------------------------------------------------------------------
def subtraction_hook(module, input, output, cav, alpha):
    """Subtract scaled CAV from hidden states."""
    if isinstance(output, tuple):
        hidden = output[0]
        hidden = hidden - alpha * cav
        return (hidden,) + output[1:]
    else:
        return output - alpha * cav


def projection_hook(module, input, output, cav, alpha):
    """Project out the CAV component from hidden states."""
    if isinstance(output, tuple):
        hidden = output[0]
        dot = torch.einsum("bsd,d->bs", hidden, cav)
        proj = torch.einsum("bs,d->bsd", dot, cav)
        hidden = hidden - alpha * proj
        return (hidden,) + output[1:]
    else:
        dot = torch.einsum("bsd,d->bs", output, cav)
        proj = torch.einsum("bs,d->bsd", dot, cav)
        return output - alpha * proj


HOOK_FNS = {
    "subtraction": subtraction_hook,
    "projection": projection_hook,
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_caption(model, processor, img_path, question, max_new_tokens=50):
    """Generate a caption/answer for an image."""
    img = Image.open(img_path).convert("RGB")
    inputs = processor(img, text=question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full = processor.decode(out[0], skip_special_tokens=True)
    # Remove the question prefix if present
    if full.lower().startswith(question.lower()):
        return full[len(question):].strip()
    return full.strip()


def generate_with_ablation(model, processor, img_path, question, cavs, layers,
                           technique, alpha, max_new_tokens=50):
    """Generate with ablation hooks on specified decoder layers."""
    hook_fn_cls = HOOK_FNS[technique]
    handles = []

    for layer_idx in layers:
        cav = cavs[layer_idx]
        layer_module = model.text_decoder.bert.encoder.layer[layer_idx].output
        fn = partial(hook_fn_cls, cav=cav, alpha=alpha)
        h = layer_module.register_forward_hook(fn)
        handles.append(h)

    try:
        result = generate_caption(model, processor, img_path, question, max_new_tokens)
    finally:
        for h in handles:
            h.remove()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70

    print(f"\n{sep}")
    print("  VISION APHASIA TEST — BLIP Factor Sweep")
    print(f"  Model: {MODEL_ID}")
    print(f"  Tool image: {TOOL_IMAGE}")
    print(f"  Control image: {CONTROL_IMAGE}")
    print(f"  Layers: {ABLATION_LAYERS}")
    print(f"  Techniques: {TECHNIQUES}")
    print(f"  Alphas: {ALPHAS}")
    print(f"{sep}")

    # Load model
    print(f"\nLoading BLIP on {DEVICE}...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}!")

    # Extract CAVs at each layer
    print("\nExtracting concept vectors (tool vs control)...")
    cavs = {}
    for layer_idx in ABLATION_LAYERS:
        cav = extract_vision_cav(model, processor, TOOL_IMAGE, CONTROL_IMAGE, layer_idx)
        cavs[layer_idx] = cav
        print(f"  Layer {layer_idx}: CAV extracted (norm={cav.norm():.4f})")

    # Generate baselines (no ablation)
    print("\nGenerating baselines...")
    baselines = {}
    for img_name, img_path in [("hammer", TOOL_IMAGE), ("cow", CONTROL_IMAGE)]:
        baselines[img_name] = {}
        for q in QUESTIONS:
            caption = generate_caption(model, processor, img_path, q, MAX_NEW_TOKENS)
            baselines[img_name][q] = caption
            print(f"  {img_name} | {q[:30]}... -> {caption[:60]}")

    # Build layer configs
    layer_configs = []
    for l in ABLATION_LAYERS:
        layer_configs.append(([l], f"layer_{l}"))
    layer_configs.append((ABLATION_LAYERS, "all_layers"))

    # Count conditions
    n_configs = len(layer_configs) * len(TECHNIQUES) * len([a for a in ALPHAS if a > 0])
    n_total = n_configs * 2 * len(QUESTIONS)  # 2 images x N questions
    print(f"\nRunning {n_configs} ablation configs x 2 images x {len(QUESTIONS)} questions = {n_total} generations...")

    # Sweep
    all_results = []
    config_num = 0

    for (layers, layer_label), technique in product(layer_configs, TECHNIQUES):
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue  # baseline already captured

            config_num += 1
            config_label = f"{layer_label} | {technique} | alpha={alpha}"
            print(f"  [{config_num}/{n_configs}] {config_label}")

            for img_name, img_path in [("hammer", TOOL_IMAGE), ("cow", CONTROL_IMAGE)]:
                role = "target" if img_name == "hammer" else "control"

                for question in QUESTIONS:
                    ablated = generate_with_ablation(
                        model, processor, img_path, question,
                        cavs, layers, technique, alpha, MAX_NEW_TOKENS
                    )

                    baseline_text = baselines[img_name][question]

                    all_results.append({
                        "layers": layers,
                        "layer_label": layer_label,
                        "technique": technique,
                        "alpha": alpha,
                        "image": img_name,
                        "image_path": img_path,
                        "role": role,
                        "question": question,
                        "baseline": baseline_text,
                        "ablated": ablated,
                        "changed": baseline_text != ablated,
                    })

    # Save JSON
    output = {
        "config": {
            "model": MODEL_ID,
            "concept": "tools (hammer vs cow)",
            "tool_image": TOOL_IMAGE,
            "control_image": CONTROL_IMAGE,
            "layers": ABLATION_LAYERS,
            "techniques": TECHNIQUES,
            "alphas": ALPHAS,
            "questions": QUESTIONS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_results": len(all_results),
        },
        "baselines": baselines,
        "results": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "vision_aphasia_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{sep}")
    print("  SUMMARY: Which conditions changed the HAMMER caption?")
    print(f"{sep}")
    print(f"  {'Layers':<14} {'Technique':<13} {'Alpha':>6}  {'Hammer':>8} {'Cow':>8}")
    print(f"  {'-'*14} {'-'*13} {'-'*6}  {'-'*8} {'-'*8}")

    for (layers, layer_label), technique in product(layer_configs, TECHNIQUES):
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue

            hammer_changed = any(
                r["changed"] for r in all_results
                if r["layers"] == layers and r["technique"] == technique
                and r["alpha"] == alpha and r["image"] == "hammer"
            )
            cow_changed = any(
                r["changed"] for r in all_results
                if r["layers"] == layers and r["technique"] == technique
                and r["alpha"] == alpha and r["image"] == "cow"
            )

            h = "YES" if hammer_changed else "no"
            c = "YES" if cow_changed else "no"
            print(f"  {layer_label:<14} {technique:<13} {alpha:>6.1f}  {h:>8} {c:>8}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
