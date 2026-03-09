"""
Qualitative Aphasia Test — Feature Steering Multi-Capa
=======================================================
Tests how text generation changes when ablating concepts using
un-normalized steering vectors with multi-layer intervention.

Ablates each concept (time, place, tools) separately with the best
conditions from the factorial experiment:
  - Modes: single_L3, multi_early (L3-5), multi_all (L0-11)
  - Alphas: 1.0, 4.0
  - Centering: centered
  - Direction: ablation (subtract) + 1 amplification control

Uses greedy decoding (temp=0) so differences are purely from
the intervention, not sampling noise.

Usage:
    python aphasia_test_steering.py

Input:  cavs_steering/*.pt (from experiment_multilayer.py)
Output: results_multilayer/aphasia_test_results.json
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
STEERING_DIR = "cavs_steering"
RESULTS_DIR = "results_multilayer"

CONCEPTS = cfg.CONCEPTS

N_LAYERS = 12
MODES = {
    "single_L3": [3],
    "multi_early": [3, 4, 5],
    "multi_all": list(range(N_LAYERS)),
}

ALPHAS = [1.0, 4.0]
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0  # Greedy = deterministic

# Test prompts per concept: 1 target + 2 controls
TEST_PROMPTS = {
    "time": {
        "target":  "Yesterday morning, I woke up early and",
        "control_place": "The city of Paris is famous for",
        "control_tools": "He picked up the hammer and began to",
    },
    "place": {
        "target":  "The city of Paris is famous for",
        "control_time": "Yesterday morning, I woke up early and",
        "control_tools": "He picked up the hammer and began to",
    },
    "tools": {
        "target":  "He picked up the hammer and began to",
        "control_time": "Yesterday morning, I woke up early and",
        "control_place": "The city of Paris is famous for",
    },
}


# ---------------------------------------------------------------------------
# Load Steering Vectors
# ---------------------------------------------------------------------------
def load_steering_vec(concept, layer, centered=True):
    if centered:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_centered_layer{layer}.pt"
        )
    else:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_layer{layer}.pt"
        )
    return torch.load(path)


def load_global_mean(concept, layer):
    path = os.path.join(
        STEERING_DIR, f"{concept}_mean_layer{layer}.pt"
    )
    return torch.load(path)


# ---------------------------------------------------------------------------
# Steering Hook
# ---------------------------------------------------------------------------
def ablation_steering_hook(
    resid_post, hook, steering_vec, global_mean, alpha, use_centering,
    direction=-1
):
    """Feature steering: act' = act + direction * α * steering_vec."""
    if use_centering:
        centered = resid_post - global_mean
        sv_unit = steering_vec / steering_vec.norm()
        dot = torch.einsum("bsd,d->bs", centered, sv_unit)
        proj = torch.einsum("bs,d->bsd", dot, sv_unit)
        centered = centered + direction * alpha * proj
        return centered + global_mean
    else:
        return resid_post + direction * alpha * steering_vec


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_text(model, prompt):
    """Generate text with greedy decoding. Returns only the continuation."""
    input_ids = model.to_tokens(prompt)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
    full_text = model.to_string(output_ids[0])
    return full_text[len(prompt):].strip()


def generate_with_steering(model, prompt, concept, layers, alpha,
                           centered=True, direction=-1):
    """Generate with steering hooks on specified layers."""
    device = model.cfg.device
    hooks = []

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        sv = load_steering_vec(concept, layer, centered=centered).to(device)
        gm = load_global_mean(concept, layer).to(device)
        hook_fn = partial(
            ablation_steering_hook,
            steering_vec=sv,
            global_mean=gm,
            alpha=alpha,
            use_centering=centered,
            direction=direction,
        )
        hooks.append((hook_name, hook_fn))

    with model.hooks(fwd_hooks=hooks):
        return generate_text(model, prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70

    print(f"\n{sep}")
    print("  APHASIA TEST — FEATURE STEERING MULTI-CAPA")
    print(f"  Concepts: {CONCEPTS}")
    print(f"  Modes: {list(MODES.keys())}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Greedy decoding (temp=0)")
    print(f"{sep}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    # Verify steering vectors exist
    for concept in CONCEPTS:
        for layer in range(N_LAYERS):
            for suffix in ["steer_centered", "steer", "mean"]:
                path = os.path.join(
                    STEERING_DIR, f"{concept}_{suffix}_layer{layer}.pt"
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing: {path}\n"
                        f"Run experiment_multilayer.py first to extract vectors."
                    )
    print("  All steering vectors found.")

    # Generate baselines (no intervention)
    print(f"\nGenerating baselines...")
    all_prompts = {}
    for concept, prompts in TEST_PROMPTS.items():
        for label, prompt in prompts.items():
            all_prompts[prompt] = None

    baselines = {}
    for prompt in all_prompts:
        baselines[prompt] = generate_text(model, prompt)
        print(f"  \"{prompt[:40]}...\" → done")

    # Run conditions
    all_results = []
    condition_num = 0

    # Ablation conditions: 3 concepts × 3 modes × 2 alphas = 18
    # + Amplification controls: 3 concepts × 1 (multi_all, α=4) = 3
    # Each condition tests target + controls = 3 prompts
    total_conditions = len(CONCEPTS) * len(MODES) * len(ALPHAS) + len(CONCEPTS)
    print(f"\nRunning {total_conditions} conditions "
          f"({total_conditions * 3} text generations)...")

    for concept in CONCEPTS:
        prompts = TEST_PROMPTS[concept]

        print(f"\n  --- Ablating: {concept.upper()} ---")

        for mode_name, layers in MODES.items():
            for alpha in ALPHAS:
                condition_num += 1
                label = (f"[{condition_num}/{total_conditions}] "
                         f"{concept}/{mode_name}/α={alpha}/ABLATE")
                print(f"  {label}")

                for prompt_label, prompt in prompts.items():
                    ablated_text = generate_with_steering(
                        model, prompt, concept, layers, alpha,
                        centered=True, direction=-1,
                    )
                    changed = baselines[prompt] != ablated_text
                    role = "target" if prompt_label == "target" else "control"

                    all_results.append({
                        "ablated_concept": concept,
                        "mode": mode_name,
                        "n_layers": len(layers),
                        "alpha": alpha,
                        "direction": "ablation",
                        "centering": "centered",
                        "prompt_label": prompt_label,
                        "role": role,
                        "prompt": prompt,
                        "baseline": baselines[prompt],
                        "generated": ablated_text,
                        "changed": changed,
                    })

                    flag = "CHANGED" if changed else "same"
                    print(f"    {prompt_label}: {flag}")

        # Amplification control: multi_all, α=4.0
        condition_num += 1
        label = (f"[{condition_num}/{total_conditions}] "
                 f"{concept}/multi_all/α=4.0/AMPLIFY")
        print(f"  {label}")

        for prompt_label, prompt in prompts.items():
            amplified_text = generate_with_steering(
                model, prompt, concept, MODES["multi_all"], 4.0,
                centered=True, direction=+1,
            )
            changed = baselines[prompt] != amplified_text
            role = "target" if prompt_label == "target" else "control"

            all_results.append({
                "ablated_concept": concept,
                "mode": "multi_all",
                "n_layers": N_LAYERS,
                "alpha": 4.0,
                "direction": "amplification",
                "centering": "centered",
                "prompt_label": prompt_label,
                "role": role,
                "prompt": prompt,
                "baseline": baselines[prompt],
                "generated": amplified_text,
                "changed": changed,
            })

            flag = "CHANGED" if changed else "same"
            print(f"    {prompt_label}: {flag}")

    # Save results
    output = {
        "config": {
            "model": MODEL_NAME,
            "concepts": CONCEPTS,
            "modes": {k: v for k, v in MODES.items()},
            "alphas": ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "steering_dir": STEERING_DIR,
            "total_conditions": len(all_results),
            "method": "feature_steering_negative (un-normalized mean diff)",
        },
        "baselines": {
            prompt: {"prompt": prompt, "text": baselines[prompt]}
            for prompt in baselines
        },
        "results": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "aphasia_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{sep}")
    print("  SUMMARY: Text changes per condition")
    print(f"{sep}")

    for concept in CONCEPTS:
        print(f"\n  Ablated concept: {concept.upper()}")
        print(f"  {'Mode':<15} {'Alpha':>6} {'Dir':<8}  "
              f"{'Target':>7} {'Ctrl1':>7} {'Ctrl2':>7}")
        print(f"  {'-'*15} {'-'*6} {'-'*8}  {'-'*7} {'-'*7} {'-'*7}")

        concept_results = [r for r in all_results
                           if r["ablated_concept"] == concept]

        # Group by condition
        conditions = {}
        for r in concept_results:
            key = (r["mode"], r["alpha"], r["direction"])
            if key not in conditions:
                conditions[key] = {}
            conditions[key][r["prompt_label"]] = r["changed"]

        for (mode, alpha, direction), prompts_changed in conditions.items():
            dir_label = "ABLATE" if direction == "ablation" else "AMPLFY"
            vals = []
            for pl in TEST_PROMPTS[concept]:
                ch = prompts_changed.get(pl, False)
                vals.append("YES" if ch else "no")

            print(f"  {mode:<15} {alpha:>6.1f} {dir_label:<8}  "
                  f"{vals[0]:>7} {vals[1]:>7} {vals[2]:>7}")

    # Show actual text changes for best conditions
    print(f"\n{sep}")
    print("  GENERATED TEXT COMPARISONS (best ablation conditions)")
    print(f"{sep}")

    for concept in CONCEPTS:
        print(f"\n  === {concept.upper()} ablation (multi_all, α=4.0) ===")

        for r in all_results:
            if (r["ablated_concept"] == concept and
                    r["mode"] == "multi_all" and
                    r["alpha"] == 4.0 and
                    r["direction"] == "ablation"):
                print(f"\n  [{r['prompt_label']}] \"{r['prompt']}\"")
                print(f"    BASELINE:  {r['baseline'][:120]}")
                print(f"    ABLATED:   {r['generated'][:120]}")
                if r["changed"]:
                    print(f"    >> CHANGED")
                else:
                    print(f"    >> (same)")

    # Amplification examples
    print(f"\n{sep}")
    print("  AMPLIFICATION CONTROLS (multi_all, α=4.0)")
    print(f"{sep}")

    for concept in CONCEPTS:
        print(f"\n  === {concept.upper()} amplification ===")
        for r in all_results:
            if (r["ablated_concept"] == concept and
                    r["direction"] == "amplification" and
                    r["role"] == "target"):
                print(f"  [{r['prompt_label']}] \"{r['prompt']}\"")
                print(f"    BASELINE:   {r['baseline'][:120]}")
                print(f"    AMPLIFIED:  {r['generated'][:120]}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
