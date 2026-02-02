"""
Experiment Runner — Full Factorial Design
==========================================
Runs ablation experiments across all factor combinations:
  - 3 Concepts (time, place, tools)
  - 2 Techniques (subtraction, projection)
  - 2 Layers (6=middle, 10=late)
  - 4 Intensities (0.0, 1.5, 3.5, 6.0)
  - 2 CAV Methods (mean_diff, svm)

Measures R1 (accuracy) and R2 (similarity) for each condition.
Also runs cross-concept specificity test (Test 12).

Usage (inside Docker):
    python experiment_runner.py

Input:  experiment_data.json, cavs/*.pt
Output: results/results_factorial.csv
        results/results_specificity.csv
        results/results_baseline.json
"""

import torch
import json
import os
import pandas as pd
from functools import partial
from transformer_lens import HookedTransformer
from metrics import evaluate_r1, evaluate_r2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2-small"
DATA_FILE = "experiment_data.json"
CAV_DIR = "cavs"
RESULTS_DIR = "results"

CONCEPTS = ["time", "place", "tools"]
TECHNIQUES = ["subtraction", "projection"]
LAYERS = [6, 10]
INTENSITIES = [0.0, 1.5, 3.5, 6.0]
CAV_METHODS = ["mean_diff", "svm"]


# ---------------------------------------------------------------------------
# Ablation Hook
# ---------------------------------------------------------------------------
def ablation_hook(resid_post, hook, cav, alpha, technique):
    """Modify residual stream activations to ablate a concept."""
    if technique == "subtraction":
        resid_post -= alpha * cav
    elif technique == "projection":
        dot_product = torch.einsum("bsd,d->bs", resid_post, cav)
        proj = torch.einsum("bs,d->bsd", dot_product, cav)
        resid_post -= alpha * proj
    return resid_post


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_cav(concept, method, layer):
    """Load a CAV tensor from disk."""
    path = os.path.join(CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CAV not found: {path}")
    return torch.load(path)


def run_condition(model, cav, layer, alpha, technique, r1_data, r2_data):
    """Run one experimental condition and return R1, R2 scores."""
    if alpha == 0.0:
        # Baseline: no ablation
        r1_acc, r1_items = evaluate_r1(model, r1_data)
        r2_sim, r2_items = evaluate_r2(model, r2_data)
    else:
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(
            ablation_hook, cav=cav, alpha=alpha, technique=technique
        )
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            r1_acc, r1_items = evaluate_r1(model, r1_data)
            r2_sim, r2_items = evaluate_r2(model, r2_data)

    return r1_acc, r2_sim, r1_items, r2_items


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Experiment Runner — Full Factorial Design")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    # ===================================================================
    # PHASE 1: BASELINE (no ablation)
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 1: BASELINE")
    print(f"{'='*50}")

    baselines = {}
    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]

        r1_acc, r1_items = evaluate_r1(model, r1_data)
        r2_sim, r2_items = evaluate_r2(model, r2_data)

        baselines[concept] = {
            "r1_accuracy": r1_acc,
            "r2_similarity": r2_sim,
            "r1_per_item": r1_items,
            "r2_per_pair": r2_items,
        }
        print(f"  {concept.upper()}: R1={r1_acc:.3f}, R2={r2_sim:.4f}")

    # Save baselines
    baseline_path = os.path.join(RESULTS_DIR, "results_baseline.json")
    # Convert for JSON serialization
    baseline_save = {}
    for c in CONCEPTS:
        baseline_save[c] = {
            "r1_accuracy": baselines[c]["r1_accuracy"],
            "r2_similarity": baselines[c]["r2_similarity"],
        }
    with open(baseline_path, "w") as f:
        json.dump(baseline_save, f, indent=2)
    print(f"  Saved: {baseline_path}")

    # Check baseline validity (Test 10)
    for concept in CONCEPTS:
        r1 = baselines[concept]["r1_accuracy"]
        r2 = baselines[concept]["r2_similarity"]
        r1_ok = "PASS" if r1 >= 0.50 else "FAIL"
        r2_ok = "PASS" if r2 >= 0.60 else "FAIL"
        print(f"  Validity {concept.upper()}: "
              f"R1={r1:.3f} [{r1_ok}], R2={r2:.4f} [{r2_ok}]")

    # ===================================================================
    # PHASE 2: FACTORIAL EXPERIMENT
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL EXPERIMENT")
    print(f"{'='*50}")

    total_conditions = (
        len(CONCEPTS) * len(CAV_METHODS) * len(LAYERS) *
        len(TECHNIQUES) * (len(INTENSITIES) - 1)  # skip alpha=0
    )
    print(f"  Total conditions: {total_conditions}")

    results_rows = []
    condition_num = 0

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]
        r1_base = baselines[concept]["r1_accuracy"]
        r2_base = baselines[concept]["r2_similarity"]

        for cav_method in CAV_METHODS:
            for layer in LAYERS:
                cav = load_cav(concept, cav_method, layer)
                cav = cav.to(model.cfg.device)

                for technique in TECHNIQUES:
                    for alpha in INTENSITIES:
                        if alpha == 0.0:
                            continue

                        condition_num += 1
                        label = (f"[{condition_num}/{total_conditions}] "
                                 f"{concept}/{cav_method}/L{layer}/"
                                 f"{technique}/a={alpha}")
                        print(f"  {label}")

                        r1_acc, r2_sim, _, _ = run_condition(
                            model, cav, layer, alpha, technique,
                            r1_data, r2_data
                        )

                        delta_r1 = r1_base - r1_acc
                        delta_r2 = r2_base - r2_sim

                        results_rows.append({
                            "concept": concept,
                            "cav_method": cav_method,
                            "layer": layer,
                            "technique": technique,
                            "alpha": alpha,
                            "r1_accuracy": r1_acc,
                            "r2_similarity": r2_sim,
                            "r1_baseline": r1_base,
                            "r2_baseline": r2_base,
                            "delta_r1": delta_r1,
                            "delta_r2": delta_r2,
                        })

                        print(f"    R1={r1_acc:.3f} (Δ={delta_r1:+.3f}), "
                              f"R2={r2_sim:.4f} (Δ={delta_r2:+.4f})")

    # Save factorial results
    df = pd.DataFrame(results_rows)
    factorial_path = os.path.join(RESULTS_DIR, "results_factorial.csv")
    df.to_csv(factorial_path, index=False)
    print(f"\n  Saved: {factorial_path} ({len(df)} rows)")

    # ===================================================================
    # PHASE 3: SPECIFICITY TEST (Test 12)
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY TEST")
    print(f"{'='*50}")
    print("  Ablating each concept's CAV and measuring ALL benchmarks.")

    spec_rows = []
    # Use default settings: mean_diff, layer 6, subtraction, alpha=3.5
    spec_method = "mean_diff"
    spec_layer = 6
    spec_technique = "subtraction"
    spec_alpha = 3.5

    for ablated_concept in CONCEPTS:
        cav = load_cav(ablated_concept, spec_method, spec_layer)
        cav = cav.to(model.cfg.device)

        for measured_concept in CONCEPTS:
            r1_data = data["r1_benchmark"][measured_concept]
            r2_data = data["r2_benchmark"][measured_concept]

            r1_acc, r2_sim, _, _ = run_condition(
                model, cav, spec_layer, spec_alpha, spec_technique,
                r1_data, r2_data
            )

            r1_base = baselines[measured_concept]["r1_accuracy"]
            r2_base = baselines[measured_concept]["r2_similarity"]

            on_target = ablated_concept == measured_concept
            tag = "ON-TARGET" if on_target else "off-target"

            spec_rows.append({
                "ablated_concept": ablated_concept,
                "measured_concept": measured_concept,
                "on_target": on_target,
                "r1_accuracy": r1_acc,
                "r2_similarity": r2_sim,
                "delta_r1": r1_base - r1_acc,
                "delta_r2": r2_base - r2_sim,
            })

            print(f"  Ablate {ablated_concept.upper()} → "
                  f"Measure {measured_concept.upper()} [{tag}]: "
                  f"ΔR1={r1_base - r1_acc:+.3f}, "
                  f"ΔR2={r2_base - r2_sim:+.4f}")

    # Save specificity
    df_spec = pd.DataFrame(spec_rows)
    spec_path = os.path.join(RESULTS_DIR, "results_specificity.csv")
    df_spec.to_csv(spec_path, index=False)
    print(f"\n  Saved: {spec_path}")

    # Specificity ratio
    on_target = df_spec[df_spec["on_target"]]
    off_target = df_spec[~df_spec["on_target"]]
    mean_on_r1 = on_target["delta_r1"].mean()
    mean_off_r1 = off_target["delta_r1"].mean()
    mean_on_r2 = on_target["delta_r2"].mean()
    mean_off_r2 = off_target["delta_r2"].mean()

    if mean_off_r1 > 0:
        ratio_r1 = mean_on_r1 / mean_off_r1
    else:
        ratio_r1 = float("inf") if mean_on_r1 > 0 else 0.0

    if mean_off_r2 > 0:
        ratio_r2 = mean_on_r2 / mean_off_r2
    else:
        ratio_r2 = float("inf") if mean_on_r2 > 0 else 0.0

    print(f"\n  Specificity Ratios:")
    print(f"    R1 on/off = {ratio_r1:.2f} "
          f"[{'PASS' if ratio_r1 > 2.0 else 'FAIL'}]")
    print(f"    R2 on/off = {ratio_r2:.2f} "
          f"[{'PASS' if ratio_r2 > 2.0 else 'FAIL'}]")

    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*50}")
    print(f"  Factorial: {factorial_path}")
    print(f"  Specificity: {spec_path}")
    print(f"  Baseline: {baseline_path}")


if __name__ == "__main__":
    main()
