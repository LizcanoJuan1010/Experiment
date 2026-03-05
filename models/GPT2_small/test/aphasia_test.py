"""
Qualitative Aphasia Test — Full Factor Sweep
==============================================
Sweeps ALL factor combinations (method, layer, technique, alpha)
to show how each affects generated text when ablating TIME.

Uses 2 fixed prompts (1 time, 1 control) with greedy decoding
so differences are purely from ablation, not sampling noise.

Usage:
    python aphasia_test.py

Output:
    results/aphasia_test_results.json
"""

import torch
import json
import os
from functools import partial
from itertools import product
from transformer_lens import HookedTransformer

import experiment_config as cfg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = cfg.MODEL_NAME
CAV_DIR = cfg.CAV_DIR
RESULTS_DIR = cfg.RESULTS_DIR

# Factors to sweep
CAV_METHODS = ["mean_diff", "svm"]
SINGLE_LAYERS = [6, 9, 10]
MULTI_LAYERS = [6, 9, 10]              # All three combined
TECHNIQUES = ["subtraction", "projection"]
ALPHAS = [1.0, 3.0, 5.0, 8.0, 10.0]

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0                       # Greedy = deterministic

ABLATED_CONCEPT = "time"

# Fixed prompts: 1 time (target), 1 place (control), 1 tools (control)
TEST_PROMPTS = {
    "time":  "Yesterday morning, I woke up early and",
    "place": "The city of Paris is famous for",
    "tools": "He picked up the hammer and began to",
}


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def subtraction_hook(resid_post, hook, cav, alpha):
    """Direct subtraction: resid = resid - alpha * cav."""
    return resid_post - alpha * cav


def projection_hook(resid_post, hook, cav, alpha):
    """Project out the CAV component: resid = resid - alpha * (resid . cav) * cav."""
    dot = torch.einsum("bsd,d->bs", resid_post, cav)
    proj = torch.einsum("bs,d->bsd", dot, cav)
    return resid_post - alpha * proj


HOOK_FNS = {
    "subtraction": subtraction_hook,
    "projection": projection_hook,
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_text(model, prompt):
    """Generate text completion. Returns only the continuation."""
    input_ids = model.to_tokens(prompt)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
    full_text = model.to_string(output_ids[0])
    return full_text[len(prompt):].strip()


def generate_with_ablation(model, prompt, cavs_by_layer, layers, technique, alpha):
    """Generate with ablation hooks on specified layers."""
    hook_fn_cls = HOOK_FNS[technique]
    hooks = []
    for layer in layers:
        cav = cavs_by_layer[layer]
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(hook_fn_cls, cav=cav, alpha=alpha)
        hooks.append((hook_name, hook_fn))

    with model.hooks(fwd_hooks=hooks):
        return generate_text(model, prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70

    print(f"\n{sep}")
    print("  APHASIA TEST — FULL FACTOR SWEEP")
    print(f"  Concept: {ABLATED_CONCEPT} | Greedy decoding (temp=0)")
    print(f"  Methods: {CAV_METHODS}")
    print(f"  Layers: {SINGLE_LAYERS} (single) + {MULTI_LAYERS} (multi)")
    print(f"  Techniques: {TECHNIQUES}")
    print(f"  Alphas: {ALPHAS}")
    print(f"{sep}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    device = model.cfg.device

    # Load ALL CAVs into a nested dict: cavs[method][layer] = tensor
    cavs = {}
    for method in CAV_METHODS:
        cavs[method] = {}
        for layer in SINGLE_LAYERS:
            path = os.path.join(CAV_DIR, f"{ABLATED_CONCEPT}_{method}_layer{layer}.pt")
            cavs[method][layer] = torch.load(path).to(device)
            print(f"  Loaded: {path}")

    # Generate baselines (no ablation)
    print(f"\nGenerating baselines...")
    baselines = {}
    for category, prompt in TEST_PROMPTS.items():
        baselines[category] = generate_text(model, prompt)
        print(f"  {category}: done")

    # Build layer configs to test
    layer_configs = []
    for l in SINGLE_LAYERS:
        layer_configs.append(([l], f"layer_{l}"))
    layer_configs.append((MULTI_LAYERS, f"layers_{'_'.join(str(l) for l in MULTI_LAYERS)}"))

    # Count total conditions
    total = len(CAV_METHODS) * len(layer_configs) * len(TECHNIQUES) * len(ALPHAS) * len(TEST_PROMPTS)
    print(f"\nRunning {total} conditions...")

    # Sweep all factors
    all_results = []
    condition_num = 0

    for method, (layers, layer_label), technique, alpha in product(
        CAV_METHODS, layer_configs, TECHNIQUES, ALPHAS
    ):
        condition_num += 1
        condition_label = f"{method} | {layer_label} | {technique} | alpha={alpha}"
        print(f"  [{condition_num}/{total // len(TEST_PROMPTS)}] {condition_label}")

        for category, prompt in TEST_PROMPTS.items():
            ablated_text = generate_with_ablation(
                model, prompt, cavs[method], layers, technique, alpha
            )

            all_results.append({
                "method": method,
                "layers": layers,
                "layer_label": layer_label,
                "technique": technique,
                "alpha": alpha,
                "category": category,
                "role": "target" if category == "time" else "control",
                "prompt": prompt,
                "baseline": baselines[category],
                "ablated": ablated_text,
                "changed": baselines[category] != ablated_text,
            })

    # Save JSON
    output = {
        "config": {
            "model": MODEL_NAME,
            "ablated_concept": ABLATED_CONCEPT,
            "methods": CAV_METHODS,
            "single_layers": SINGLE_LAYERS,
            "multi_layers": MULTI_LAYERS,
            "techniques": TECHNIQUES,
            "alphas": ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "total_conditions": len(all_results),
        },
        "baselines": {cat: {"prompt": p, "text": baselines[cat]}
                      for cat, p in TEST_PROMPTS.items()},
        "results": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "aphasia_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{sep}")
    print("  SUMMARY: Which conditions changed the TIME prompt?")
    print(f"{sep}")
    print(f"  {'Method':<10} {'Layers':<16} {'Technique':<13} {'Alpha':>6}  {'Time':>6} {'Place':>6} {'Tools':>6}")
    print(f"  {'-'*10} {'-'*16} {'-'*13} {'-'*6}  {'-'*6} {'-'*6} {'-'*6}")

    for method, (layers, layer_label), technique, alpha in product(
        CAV_METHODS, layer_configs, TECHNIQUES, ALPHAS
    ):
        row = {r["category"]: r["changed"]
               for r in all_results
               if r["method"] == method and r["layers"] == layers
               and r["technique"] == technique and r["alpha"] == alpha}

        time_ch = "YES" if row.get("time") else "no"
        place_ch = "YES" if row.get("place") else "no"
        tools_ch = "YES" if row.get("tools") else "no"

        print(f"  {method:<10} {layer_label:<16} {technique:<13} {alpha:>6.1f}  {time_ch:>6} {place_ch:>6} {tools_ch:>6}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
