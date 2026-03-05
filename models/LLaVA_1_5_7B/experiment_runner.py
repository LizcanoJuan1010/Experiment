"""
Experiment Runner — Full Factorial Design (LLaVA-1.5 7B)
=========================================================
Runs ablation experiments across all factor combinations:
  - N Concepts (pilot: golden_retriever, labrador_retriever, tabby_cat)
  - 2 Techniques (subtraction, projection)
  - 2 Layers (16=50%, 27=84%)
  - 4 Intensities (0.0, 1.5, 3.5, 6.0)
  - 2 CAV Methods (mean_diff, svm)

Measures R1 (VQA accuracy), R2a (cross-modal similarity),
and R2b (output representation disruption).
Also runs cross-concept specificity test.

Usage:
    python experiment_runner.py

Input:  data/experiment_data.json, cavs/*.pt
Output: results/results_factorial.csv
        results/results_specificity.csv
        results/results_baseline.json
"""

import torch
import json
import os
import numpy as np
import pandas as pd

import experiment_config as cfg
from model_loader import load_llava_model
from metrics import evaluate_r1, evaluate_r2a, evaluate_r2b, compute_r2b_baselines
from pyvene_utils import collect_activations_at_layer, pool_activations, AblationContext

# Configuration
CONCEPTS = cfg.CONCEPTS
TECHNIQUES = cfg.TECHNIQUES
LAYERS = cfg.EXPERIMENT_LAYERS
INTENSITIES = cfg.INTENSITIES
CAV_METHODS = cfg.CAV_METHODS
RESULTS_DIR = cfg.RESULTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_cav(concept, method, layer):
    """Load a CAV tensor from disk."""
    path = os.path.join(cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CAV not found: {path}")
    return torch.load(path, weights_only=True)


def measure_concept_removal(model, processor, cav, layer, alpha, technique,
                            probe_image, probe_prompt):
    """
    Measure how much of the concept direction was removed by ablation.

    Returns dict with dot_pre, dot_post, removal_ratio.
    """
    cav_np = cav.cpu().numpy()

    # Pre-ablation activation
    act_pre = collect_activations_at_layer(
        model, processor, probe_image, probe_prompt, layer,
    )
    from PIL import Image
    img = Image.open(probe_image).convert("RGB")
    inputs = processor(text=probe_prompt, images=img, return_tensors="pt")
    pooled_pre = pool_activations(act_pre, inputs["input_ids"], processor).numpy()
    dot_pre = float(np.dot(pooled_pre, cav_np))

    # Post-ablation activation
    with AblationContext(model, cav, alpha, [layer], technique):
        act_post = collect_activations_at_layer(
            model, processor, probe_image, probe_prompt, layer,
        )
    pooled_post = pool_activations(act_post, inputs["input_ids"], processor).numpy()
    dot_post = float(np.dot(pooled_post, cav_np))

    if abs(dot_pre) > 1e-8:
        removal_ratio = 1.0 - (dot_post / dot_pre)
    else:
        removal_ratio = 0.0

    return {
        "dot_pre": round(dot_pre, 6),
        "dot_post": round(dot_post, 6),
        "removal_ratio": round(removal_ratio, 4),
        "warning": removal_ratio < 0.5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 60
    pilot_str = " [PILOT]" if cfg.PILOT_MODE else ""
    print(f"\n{sep}")
    print(f"  Experiment Runner — LLaVA-1.5 7B Full Factorial{pilot_str}")
    print(f"  Concepts: {CONCEPTS}")
    print(f"  Image-only ablation: {cfg.IMAGE_ONLY_ABLATION}")
    print(f"{sep}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print(f"\n  Loading data from {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"  Loading {cfg.MODEL_NAME} (4-bit)...")
    model, processor = load_llava_model()

    image_tokens_only = cfg.IMAGE_ONLY_ABLATION

    # ===================================================================
    # PHASE 1: BASELINE
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 1: BASELINE")
    print(f"{'='*50}")

    baselines = {}
    r2b_baseline_cache = {}  # {concept: [baseline_embeddings]}

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]

        # R1 baseline
        r1_acc, r1_items = evaluate_r1(model, processor, r1_data)

        # R2a baseline (cross-modal similarity without ablation)
        r2a_sim, r2a_items = evaluate_r2a(model, processor, r2_data)

        # R2b baselines: pre-compute and cache
        r2b_baselines_emb = compute_r2b_baselines(model, processor, r2_data)
        r2b_baseline_cache[concept] = r2b_baselines_emb

        # R2b at baseline should be ~1.0 (comparing with itself)
        r2b_sim, r2b_items = evaluate_r2b(
            model, processor, r2_data, r2b_baselines_emb
        )

        baselines[concept] = {
            "r1_accuracy": r1_acc,
            "r2a_similarity": r2a_sim,
            "r2b_similarity": r2b_sim,
        }
        print(f"  {concept.upper()}: R1={r1_acc:.3f}, "
              f"R2a={r2a_sim:.4f}, R2b={r2b_sim:.4f}")

    # Save baselines
    baseline_path = os.path.join(RESULTS_DIR, "results_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Saved: {baseline_path}")

    # Check validity
    all_zero = True
    for concept in CONCEPTS:
        r1 = baselines[concept]["r1_accuracy"]
        r2a = baselines[concept]["r2a_similarity"]
        r2b = baselines[concept]["r2b_similarity"]
        if r1 > 0 or r2a > 0 or r2b > 0:
            all_zero = False
        r1_ok = "PASS" if r1 >= cfg.R1_BASELINE_THRESHOLD else "FAIL"
        r2a_ok = "PASS" if r2a >= cfg.R2A_BASELINE_THRESHOLD else "FAIL"
        r2b_ok = "PASS" if r2b >= cfg.R2B_BASELINE_THRESHOLD else "FAIL"
        print(f"  Validity {concept.upper()}: "
              f"R1={r1:.3f} [{r1_ok}], "
              f"R2a={r2a:.4f} [{r2a_ok}], "
              f"R2b={r2b:.4f} [{r2b_ok}]")

    if all_zero:
        print(f"\n  FATAL: All baselines are 0.000 — no benchmark data!")
        print(f"  This means experiment_data.json has empty r1/r2 benchmarks.")
        print(f"  Re-run data_gen_multimodal.py with HuggingFace auth.")
        raise SystemExit(1)

    # ===================================================================
    # PHASE 2: FACTORIAL EXPERIMENT
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL EXPERIMENT")
    print(f"{'='*50}")

    total_conditions = (
        len(CONCEPTS) * len(CAV_METHODS) * len(LAYERS) *
        len(TECHNIQUES) * (len(INTENSITIES) - 1)
    )
    print(f"  Total conditions: {total_conditions}")

    results_rows = []
    condition_num = 0

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]
        r1_base = baselines[concept]["r1_accuracy"]
        r2a_base = baselines[concept]["r2a_similarity"]
        r2b_base = baselines[concept]["r2b_similarity"]

        for cav_method in CAV_METHODS:
            for layer in LAYERS:
                cav = load_cav(concept, cav_method, layer)

                for technique in TECHNIQUES:
                    for alpha in INTENSITIES:
                        if alpha == 0.0:
                            continue

                        condition_num += 1
                        label = (f"[{condition_num}/{total_conditions}] "
                                 f"{concept}/{cav_method}/L{layer}/"
                                 f"{technique}/a={alpha}")
                        print(f"  {label}")

                        # R1 with ablation
                        r1_acc, _ = evaluate_r1(
                            model, processor, r1_data,
                            cav=cav, alpha=alpha,
                            layers=[layer], technique=technique,
                            image_tokens_only=image_tokens_only,
                        )

                        # R2a with ablation
                        r2a_sim, _ = evaluate_r2a(
                            model, processor, r2_data,
                            cav=cav, alpha=alpha,
                            layers_ablation=[layer], technique=technique,
                            image_tokens_only=image_tokens_only,
                        )

                        # R2b with ablation (using cached baselines)
                        r2b_sim, _ = evaluate_r2b(
                            model, processor, r2_data,
                            r2b_baseline_cache[concept],
                            cav=cav, alpha=alpha,
                            layers_ablation=[layer], technique=technique,
                            image_tokens_only=image_tokens_only,
                        )

                        delta_r1 = r1_base - r1_acc
                        delta_r2a = r2a_base - r2a_sim
                        delta_r2b = r2b_base - r2b_sim

                        results_rows.append({
                            "concept": concept,
                            "cav_method": cav_method,
                            "layer": layer,
                            "technique": technique,
                            "alpha": alpha,
                            "r1_accuracy": r1_acc,
                            "r2a_similarity": r2a_sim,
                            "r2b_similarity": r2b_sim,
                            "r1_baseline": r1_base,
                            "r2a_baseline": r2a_base,
                            "r2b_baseline": r2b_base,
                            "delta_r1": delta_r1,
                            "delta_r2a": delta_r2a,
                            "delta_r2b": delta_r2b,
                        })

                        print(f"    R1={r1_acc:.3f} (D={delta_r1:+.3f}), "
                              f"R2a={r2a_sim:.4f} (D={delta_r2a:+.4f}), "
                              f"R2b={r2b_sim:.4f} (D={delta_r2b:+.4f})")

    # Save factorial results
    df = pd.DataFrame(results_rows)
    factorial_path = os.path.join(RESULTS_DIR, "results_factorial.csv")
    df.to_csv(factorial_path, index=False)
    print(f"\n  Saved: {factorial_path} ({len(df)} rows)")

    # ===================================================================
    # PHASE 3: SPECIFICITY TEST
    # ===================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY TEST")
    print(f"{'='*50}")

    spec_rows = []
    spec_layer = cfg.SPECIFICITY_LAYER
    spec_method = cfg.SPECIFICITY_METHOD
    spec_technique = cfg.SPECIFICITY_TECHNIQUE
    spec_alpha = cfg.SPECIFICITY_ALPHA

    for ablated_concept in CONCEPTS:
        cav = load_cav(ablated_concept, spec_method, spec_layer)

        for measured_concept in CONCEPTS:
            r1_data = data["r1_benchmark"][measured_concept]
            r2_data = data["r2_benchmark"][measured_concept]

            r1_acc, _ = evaluate_r1(
                model, processor, r1_data,
                cav=cav, alpha=spec_alpha,
                layers=[spec_layer], technique=spec_technique,
                image_tokens_only=image_tokens_only,
            )
            r2a_sim, _ = evaluate_r2a(
                model, processor, r2_data,
                cav=cav, alpha=spec_alpha,
                layers_ablation=[spec_layer], technique=spec_technique,
                image_tokens_only=image_tokens_only,
            )
            r2b_sim, _ = evaluate_r2b(
                model, processor, r2_data,
                r2b_baseline_cache[measured_concept],
                cav=cav, alpha=spec_alpha,
                layers_ablation=[spec_layer], technique=spec_technique,
                image_tokens_only=image_tokens_only,
            )

            r1_base = baselines[measured_concept]["r1_accuracy"]
            r2a_base = baselines[measured_concept]["r2a_similarity"]
            r2b_base = baselines[measured_concept]["r2b_similarity"]
            on_target = ablated_concept == measured_concept
            tag = "ON-TARGET" if on_target else "off-target"

            spec_rows.append({
                "ablated_concept": ablated_concept,
                "measured_concept": measured_concept,
                "on_target": on_target,
                "r1_accuracy": r1_acc,
                "r2a_similarity": r2a_sim,
                "r2b_similarity": r2b_sim,
                "delta_r1": r1_base - r1_acc,
                "delta_r2a": r2a_base - r2a_sim,
                "delta_r2b": r2b_base - r2b_sim,
            })

            print(f"  Ablate {ablated_concept.upper()} -> "
                  f"Measure {measured_concept.upper()} [{tag}]: "
                  f"DR1={r1_base - r1_acc:+.3f}, "
                  f"DR2a={r2a_base - r2a_sim:+.4f}, "
                  f"DR2b={r2b_base - r2b_sim:+.4f}")

    # Save specificity
    df_spec = pd.DataFrame(spec_rows)
    spec_path = os.path.join(RESULTS_DIR, "results_specificity.csv")
    df_spec.to_csv(spec_path, index=False)
    print(f"\n  Saved: {spec_path}")

    # Specificity ratio
    on = df_spec[df_spec["on_target"]]
    off = df_spec[~df_spec["on_target"]]
    mean_on_r1 = on["delta_r1"].mean()
    mean_off_r1 = off["delta_r1"].mean()

    if mean_off_r1 > 0:
        ratio_r1 = mean_on_r1 / mean_off_r1
    else:
        ratio_r1 = float("inf") if mean_on_r1 > 0 else 0.0

    print(f"\n  Specificity Ratio R1: {ratio_r1:.2f} "
          f"[{'PASS' if ratio_r1 > cfg.SPECIFICITY_RATIO_THRESHOLD else 'FAIL'}]")

    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*50}")
    print(f"  Factorial: {factorial_path}")
    print(f"  Specificity: {spec_path}")
    print(f"  Baseline: {baseline_path}")


if __name__ == "__main__":
    main()
