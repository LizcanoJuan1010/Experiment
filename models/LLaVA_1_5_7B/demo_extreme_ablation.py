"""
Extreme Ablation Demo — tabby_cat Concept Erasure
==================================================
Interactive demonstration of catastrophic concept ablation using
extreme intensities (α=10, 15, 20) with the tabby_cat CAV.

Tests projection-based ablation across layers to observe semantic
collapse and hallucination patterns.

Usage:
    python demo_extreme_ablation.py

Output: results_extreme_ablation/
        ├── demo_results.json
        ├── summary_report.txt
        └── figures/ (optional)
"""

import torch
import json
import os
import numpy as np
from datetime import datetime
from PIL import Image

import experiment_config as cfg
from model_loader import load_llava_model
from pyvene_utils import AblationContext, _prepare_inputs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONCEPT = "tabby_cat"
CAV_METHOD = "svm"  # Statistically significant per TCAV tests
TECHNIQUE = "projection"  # More effective than subtraction

# Layers to test (skip layer 3 - no CAV available)
LAYERS_CONFIG = {
    "L16": [16],           # Mid-level (50% depth)
    "L27": [27],           # Deep semantic (84% depth) - most dramatic
    "multi": [16, 24, 27], # Compound multilayer effect (all available CAV layers)
}

# Extreme ablation intensities
ALPHAS = [0.0, 6.0, 10.0, 15.0, 20.0]

# Test images (subset of 50 available)
N_TEST_IMAGES = 5

# Questions to probe concept degradation
QUESTIONS = {
    "Q1_identity": "USER: <image>\nWhat animal is shown in this image?\nASSISTANT:",
    "Q2_description": "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    "Q3_granular": "USER: <image>\nWhat breed or type is this?\nASSISTANT:",
    "Q4_forced_choice": "USER: <image>\nIs this a dog or a cat? Answer with one word.\nASSISTANT:",
}

# Output directory (NEW - won't overwrite existing results)
RESULTS_DIR = "results_extreme_ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

# Generation parameters
GEN_MAX_NEW_TOKENS = 50
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def load_cav(concept, method, layer):
    """Load CAV tensor from disk."""
    path = os.path.join(cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CAV not found: {path}")
    return torch.load(path, weights_only=True)


def load_test_images(concept, n_images):
    """Load n random test images for the concept."""
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    # Get test split images (not used for CAV training)
    test_images = data["test_split"][concept]

    if len(test_images) < n_images:
        print(f"  WARNING: Only {len(test_images)} test images available, using all")
        n_images = len(test_images)

    # Select first n images (could randomize if desired)
    selected = test_images[:n_images]

    return selected


def generate_with_ablation(model, processor, image_path, prompt,
                           cav=None, alpha=0.0, layers=None, technique="projection"):
    """
    Generate text with optional CAV ablation.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_path: Path to image file
        prompt: Text prompt (should end with "ASSISTANT:")
        cav: CAV tensor [d_model] or None for baseline
        alpha: Ablation intensity
        layers: List of layer indices
        technique: "subtraction" or "projection"

    Returns:
        str: Generated text
    """
    device = next(model.parameters()).device

    # Prepare inputs
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate with or without ablation
    if cav is not None and alpha > 0.0 and layers is not None:
        with AblationContext(model, cav, alpha, layers, technique):
            outputs = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                do_sample=True,
            )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            do_sample=True,
        )

    # Decode response (skip prompt)
    prompt_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return response.strip()


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Extreme Ablation Demo — {CONCEPT.upper()}")
    print(f"  Concept: {CONCEPT}")
    print(f"  CAV Method: {CAV_METHOD}")
    print(f"  Technique: {TECHNIQUE}")
    print(f"  Layers: {list(LAYERS_CONFIG.keys())}")
    print(f"  Alphas: {ALPHAS}")
    print(f"{sep}\n")

    # Load model
    print("Loading LLaVA-1.5 7B (4-bit quantized)...")
    model, processor = load_llava_model()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")

    # Load test images
    print(f"\nLoading {N_TEST_IMAGES} test images for {CONCEPT}...")
    test_images = load_test_images(CONCEPT, N_TEST_IMAGES)
    print(f"  Loaded {len(test_images)} images")

    # Load CAVs for each layer
    print(f"\nLoading CAVs...")
    cavs = {}
    for layer_name, layer_list in LAYERS_CONFIG.items():
        if layer_name == "multi":
            # For multilayer, load each layer's CAV
            cavs[layer_name] = {}
            for L in layer_list:
                cav = load_cav(CONCEPT, CAV_METHOD, L)
                cavs[layer_name][L] = cav
                print(f"  {layer_name} layer {L}: {cav.shape}")
        else:
            # Single layer
            L = layer_list[0]
            cavs[layer_name] = load_cav(CONCEPT, CAV_METHOD, L)
            print(f"  {layer_name}: {cavs[layer_name].shape}")

    # Run experiments
    print(f"\n{sep}")
    print("STARTING ABLATION EXPERIMENTS")
    print(f"{sep}\n")

    results = {
        "metadata": {
            "concept": CONCEPT,
            "cav_method": CAV_METHOD,
            "technique": TECHNIQUE,
            "layers_config": {k: v for k, v in LAYERS_CONFIG.items()},
            "alphas": ALPHAS,
            "n_test_images": len(test_images),
            "timestamp": datetime.now().isoformat(),
            "model": cfg.MODEL_NAME,
        },
        "experiments": []
    }

    total_conditions = len(test_images) * len(LAYERS_CONFIG) * len(ALPHAS) * len(QUESTIONS)
    condition_num = 0

    for img_idx, image_data in enumerate(test_images):
        image_path = image_data["image_path"]
        print(f"\n{'─' * 70}")
        print(f"IMAGE {img_idx + 1}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'─' * 70}")

        for layer_name, layer_spec in LAYERS_CONFIG.items():
            print(f"\n  Layer Config: {layer_name} {layer_spec}")

            for alpha in ALPHAS:
                alpha_label = "BASELINE" if alpha == 0.0 else f"α={alpha}"
                print(f"\n    {alpha_label}")

                for q_name, prompt in QUESTIONS.items():
                    condition_num += 1
                    progress = f"[{condition_num}/{total_conditions}]"

                    # Get CAV for this layer config
                    if layer_name == "multi":
                        # For multilayer, use first layer's CAV (could average if desired)
                        cav = cavs[layer_name][layer_spec[0]]
                    else:
                        cav = cavs[layer_name]

                    # Generate response
                    try:
                        response = generate_with_ablation(
                            model, processor, image_path, prompt,
                            cav=cav if alpha > 0.0 else None,
                            alpha=alpha,
                            layers=layer_spec,
                            technique=TECHNIQUE,
                        )

                        print(f"      {progress} {q_name}: {response[:60]}...")

                        results["experiments"].append({
                            "image_idx": img_idx,
                            "image_path": image_path,
                            "layer_config": layer_name,
                            "layers": layer_spec,
                            "alpha": alpha,
                            "question": q_name,
                            "prompt": prompt,
                            "response": response,
                            "response_length": len(response),
                        })

                    except Exception as e:
                        print(f"      {progress} {q_name}: ERROR - {str(e)}")
                        results["experiments"].append({
                            "image_idx": img_idx,
                            "image_path": image_path,
                            "layer_config": layer_name,
                            "layers": layer_spec,
                            "alpha": alpha,
                            "question": q_name,
                            "prompt": prompt,
                            "response": f"ERROR: {str(e)}",
                            "response_length": 0,
                        })

    # Save results
    results_path = os.path.join(RESULTS_DIR, "demo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{sep}")
    print(f"RESULTS SAVED: {results_path}")
    print(f"{sep}\n")

    # Generate summary report
    generate_summary_report(results)

    print(f"\nExperiment complete!")
    print(f"  Total conditions tested: {total_conditions}")
    print(f"  Output directory: {RESULTS_DIR}/")
    print(f"    - demo_results.json (full data)")
    print(f"    - summary_report.txt (human-readable)")


def generate_summary_report(results):
    """Generate human-readable summary report."""
    report_path = os.path.join(RESULTS_DIR, "summary_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXTREME ABLATION DEMO - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Concept: {results['metadata']['concept']}\n")
        f.write(f"CAV Method: {results['metadata']['cav_method']}\n")
        f.write(f"Technique: {results['metadata']['technique']}\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"\nTotal Experiments: {len(results['experiments'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS BY ALPHA INTENSITY\n")
        f.write("=" * 80 + "\n\n")

        # Group by alpha and analyze
        for alpha in ALPHAS:
            alpha_label = "BASELINE (No Ablation)" if alpha == 0.0 else f"α = {alpha}"
            f.write(f"\n{alpha_label}\n")
            f.write("-" * 80 + "\n")

            alpha_exps = [e for e in results["experiments"] if e["alpha"] == alpha]

            # Sample responses for Q1 (identity question)
            q1_responses = [e for e in alpha_exps if e["question"] == "Q1_identity"]
            if q1_responses:
                f.write("\nSample Responses (Q1: What animal is shown?):\n")
                for i, exp in enumerate(q1_responses[:3]):  # Show first 3
                    f.write(f"  Image {exp['image_idx']+1}, {exp['layer_config']}: {exp['response']}\n")

            # Average response length
            avg_len = np.mean([e["response_length"] for e in alpha_exps])
            f.write(f"\nAverage Response Length: {avg_len:.1f} characters\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("LAYER COMPARISON (α=20 only)\n")
        f.write("=" * 80 + "\n\n")

        extreme_exps = [e for e in results["experiments"] if e["alpha"] == 20.0]
        for layer_name in LAYERS_CONFIG.keys():
            layer_exps = [e for e in extreme_exps if e["layer_config"] == layer_name]
            if layer_exps:
                f.write(f"\n{layer_name.upper()}:\n")
                f.write("-" * 80 + "\n")
                # Show Q1 responses for this layer
                q1 = [e for e in layer_exps if e["question"] == "Q1_identity"]
                for exp in q1[:2]:
                    f.write(f"  {exp['response']}\n")

    print(f"Summary report saved: {report_path}")


if __name__ == "__main__":
    main()
