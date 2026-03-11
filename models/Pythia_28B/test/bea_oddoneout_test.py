"""
BEA Odd-One-Out Test — Semantic Intruder Detection with Ablation (Pythia 2.8B)
=====================================================================
Simulates the BEA Odd-One-Out neuropsychological test on
Pythia 2.8B with concept ablation via CAV projection.

Concepts, layers, methods, and intensities are all read from
experiment_config.py — no hardcoded values.

Design:
  Factorial: N concepts × M CAV methods × 4 layer conditions × K alphas
  Layer conditions: EXTRACTION_LAYERS (individual) + all layers (0-31)
  For each ablation condition, measures accuracy on ALL concept benchmarks
  (on-target + off-target) to test specificity.

Input:  Benchmark_construction/output/bea_oddoneout_benchmark.json
        cavs/*.pt (pre-trained CAVs)
Output: results/bea_oddoneout_test_results.json

Usage:
    python -m test.bea_oddoneout_test
"""

import torch
import json
import os
import sys
import random
from functools import partial
from datetime import datetime

# Ensure parent directory is in path
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
from experiment_runner import ablation_hook, load_cav
from metrics import evaluate_bea_oddoneout

# ---------------------------------------------------------------------------
# Configuration — all from experiment_config.py
# ---------------------------------------------------------------------------
CONCEPTS = cfg.CONCEPTS
CAV_METHODS = cfg.CAV_METHODS
SINGLE_LAYERS = cfg.EXTRACTION_LAYERS
ALL_LAYERS = cfg.EXPERIMENT_LAYERS_ALL
ALPHAS = cfg.INTENSITIES

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "Benchmark_construction", "output", "bea_oddoneout_benchmark.json",
)
RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, "bea_oddoneout_test_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate_with_hooks(model, dataset, fwd_hooks):
    """Evaluate BEA odd-one-out benchmark with ablation hooks active."""
    if not fwd_hooks:
        return evaluate_bea_oddoneout(model, dataset)
    with model.hooks(fwd_hooks=fwd_hooks):
        return evaluate_bea_oddoneout(model, dataset)


def build_single_layer_hooks(model, concept, cav_method, layer, alpha):
    """Build hook list for single-layer ablation."""
    cav = load_cav(concept, cav_method, layer)
    cav = cav.to(model.cfg.device)
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique="projection")
    return [(hook_name, hook_fn)]


def build_multi_layer_hooks(model, concept, cav_method, layers, alpha):
    """Build hook list for multi-layer ablation (all layers simultaneously)."""
    fwd_hooks = []
    for layer in layers:
        try:
            cav = load_cav(concept, cav_method, layer)
        except FileNotFoundError:
            print(f"    WARNING: CAV not found for {concept}/{cav_method}/L{layer}, skipping layer")
            continue
        cav = cav.to(model.cfg.device)
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique="projection")
        fwd_hooks.append((hook_name, hook_fn))
    return fwd_hooks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BEA Odd-One-Out Test — Semantic Intruder Detection (Pythia 2.8B)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load benchmark
    print(f"\nLoading benchmark: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # OddOneOut benchmark uses 3 broad categories: "time", "place", "tools"
    category_mapping = {
        "tools": ["lock", "toothbrush", "violin", "candle", "fire", "snowman"],
        "place": ["king", "knight", "soldier"],
        "time": ["musician", "baker", "barber"],
    }

    concept_to_category = {}
    for cat, concepts in category_mapping.items():
        for c in concepts:
            concept_to_category[c] = cat

    available = [c for c in CONCEPTS if c in concept_to_category]
    missing = [c for c in CONCEPTS if c not in concept_to_category]
    if missing:
        print(f"  WARNING: {len(missing)} concepts not in category mapping: {missing}")
    assert available, "No concepts from cfg.CONCEPTS found in mapping!"

    items_by_concept = {}
    for c in available:
        cat = concept_to_category[c]
        items_by_concept[c] = benchmark["items"].get(cat, [])

    for c in available:
        print(f"  {c} (category: {concept_to_category[c]}): {len(items_by_concept[c])} items")

    # Load model
    print(f"\nLoading model: {cfg.MODEL_NAME}...")
    model = cfg.load_model()

    print(f"\n  Concepts in benchmark: {len(available)}")

    # =================================================================
    # PHASE 1: BASELINE (no ablation)
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 1: BASELINE (healthy brain)")
    print(f"{'='*50}")

    baselines = {}
    correct_items_by_concept = {}

    for concept in available:
        acc, per_item = evaluate_bea_oddoneout(model, items_by_concept[concept])
        baselines[concept] = {
            "accuracy": acc,
            "n_items": len(items_by_concept[concept]),
            "per_item": per_item,
        }

        correct_indices = [i for i, pi in enumerate(per_item) if pi["is_correct"]]
        correct_items = [items_by_concept[concept][i] for i in correct_indices]
        correct_items_by_concept[concept] = correct_items

        print(f"  {concept.upper()}: accuracy={acc:.3f} "
              f"({len(correct_items)}/{len(items_by_concept[concept])} correct)")

    # =================================================================
    # BASELINE FILTERING
    # =================================================================
    MIN_CORRECT = 5
    all_concepts_baselines = list(available)
    available = [c for c in available
                 if len(correct_items_by_concept[c]) >= MIN_CORRECT]

    filtered_out = [c for c in all_concepts_baselines if c not in available]
    if filtered_out:
        print(f"\n  Filtered out {len(filtered_out)} concepts with <{MIN_CORRECT} "
              f"correct items: {filtered_out}")
    print(f"  Concepts for ablation: {len(available)}/{len(all_concepts_baselines)}")
    print(f"  Strategy: test only baseline-correct items (baseline=1.0 by construction)")
    assert available, f"No concepts have >= {MIN_CORRECT} correct baseline items!"

    for c in available:
        print(f"    {c}: {len(correct_items_by_concept[c])} correct items for ablation")

    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_layers", ALL_LAYERS))

    N_OFF_TARGET = min(3, len(available) - 1)

    total_conditions = (
        len(available) * len(CAV_METHODS) * len(layer_conditions) * len(ALPHAS)
    )
    evals_per_condition = 1 + N_OFF_TARGET
    print(f"  Ablation conditions: {total_conditions}")
    print(f"  Off-target sample per condition: {N_OFF_TARGET}")
    print(f"  Total evaluations: {total_conditions * evals_per_condition}")

    # =================================================================
    # PHASE 2: FACTORIAL ABLATION
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL ABLATION")
    print(f"{'='*50}")

    results = []
    condition_num = 0

    for ablated_concept in available:
        for cav_method in CAV_METHODS:
            for layer_label, layers in layer_conditions:
                for alpha in ALPHAS:
                    condition_num += 1
                    print(f"\n  [{condition_num}/{total_conditions}] "
                          f"Ablate {ablated_concept}/{cav_method}/{layer_label}/α={alpha}")

                    if len(layers) == 1:
                        fwd_hooks = build_single_layer_hooks(
                            model, ablated_concept, cav_method, layers[0], alpha
                        )
                    else:
                        fwd_hooks = build_multi_layer_hooks(
                            model, ablated_concept, cav_method, layers, alpha
                        )

                    if not fwd_hooks:
                        print(f"    SKIP: no valid hooks")
                        continue

                    off_candidates = [c for c in available if c != ablated_concept]
                    random.seed(hash((ablated_concept, cav_method, layer_label, alpha)))
                    off_sample = random.sample(off_candidates, N_OFF_TARGET) if off_candidates else []
                    measure_concepts = [ablated_concept] + off_sample

                    for measured_concept in measure_concepts:
                        on_target = ablated_concept == measured_concept
                        acc, per_item = evaluate_with_hooks(
                            model, correct_items_by_concept[measured_concept], fwd_hooks
                        )
                        baseline_acc = 1.0
                        delta = baseline_acc - acc

                        tag = "ON-TARGET" if on_target else "off-target"
                        print(f"    → {measured_concept.upper()} [{tag}]: "
                              f"acc={acc:.3f} (Δ={delta:+.3f})")

                        results.append({
                            "ablated_concept": ablated_concept,
                            "measured_concept": measured_concept,
                            "on_target": on_target,
                            "cav_method": cav_method,
                            "layer": layer_label,
                            "alpha": alpha,
                            "accuracy": acc,
                            "baseline_accuracy": baseline_acc,
                            "delta_accuracy": delta,
                            "per_item": per_item,
                        })

    # =================================================================
    # PHASE 3: SPECIFICITY ANALYSIS
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY ANALYSIS")
    print(f"{'='*50}")

    on_target_deltas = [r["delta_accuracy"] for r in results if r["on_target"]]
    off_target_deltas = [r["delta_accuracy"] for r in results if not r["on_target"]]

    mean_on = sum(on_target_deltas) / len(on_target_deltas) if on_target_deltas else 0
    mean_off = sum(off_target_deltas) / len(off_target_deltas) if off_target_deltas else 0

    print(f"  Mean on-target Δaccuracy:  {mean_on:+.4f}")
    print(f"  Mean off-target Δaccuracy: {mean_off:+.4f}")

    if mean_off > 0:
        ratio = mean_on / mean_off
    else:
        ratio = float("inf") if mean_on > 0 else 0.0
    print(f"  Specificity ratio (on/off): {ratio:.2f}")

    for ablated in available:
        concept_results = [r for r in results if r["ablated_concept"] == ablated]
        on_deltas = [r["delta_accuracy"] for r in concept_results if r["on_target"]]
        off_deltas = [r["delta_accuracy"] for r in concept_results if not r["on_target"]]
        mean_on_c = sum(on_deltas) / len(on_deltas) if on_deltas else 0
        mean_off_c = sum(off_deltas) / len(off_deltas) if off_deltas else 0
        print(f"\n  Ablate {ablated.upper()}:")
        print(f"    On-target Δ:  {mean_on_c:+.4f}")
        print(f"    Off-target Δ: {mean_off_c:+.4f}")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    output = {
        "metadata": {
            "test": "BEA Odd-One-Out (Semantic Intruder Detection)",
            "strategy": "baseline-correct-only",
            "strategy_description": "Ablation tested only on items the model "
                                    "answers correctly at baseline (baseline=1.0). "
                                    "Any accuracy drop is directly caused by ablation.",
            "model": cfg.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "layers_single": SINGLE_LAYERS,
            "layers_all": list(ALL_LAYERS),
            "alphas": ALPHAS,
            "cav_methods": CAV_METHODS,
            "concepts_configured": list(CONCEPTS),
            "concepts_evaluated": available,
            "concepts_filtered_out": filtered_out,
            "min_correct_items": MIN_CORRECT,
        },
        "baselines": {
            c: {
                "accuracy": baselines[c]["accuracy"],
                "n_items": baselines[c]["n_items"],
                "n_correct": len(correct_items_by_concept.get(c, [])),
                "per_item": baselines[c]["per_item"],
            }
            for c in all_concepts_baselines
        },
        "results": results,
        "specificity": {
            "mean_on_target_delta": mean_on,
            "mean_off_target_delta": mean_off,
            "specificity_ratio": ratio,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    print("BEA ODD-ONE-OUT TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
