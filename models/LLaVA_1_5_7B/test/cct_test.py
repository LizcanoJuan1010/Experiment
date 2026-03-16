"""
CCT Test — Visual Concept Classification Test (LLaVA-1.5 7B)
=============================================================
Visual analog of the Camel and Cactus Test (CCT) / Pyramids & Palm Trees Test.

Given an IMAGE of a concept (e.g., a dog), the model must identify which
category label it belongs to via multiple-choice VQA
(e.g., "dog / car / flower / furniture").

CAV ablation removes the concept's visual representation from specified
residual-stream layers and measures the resulting accuracy drop.

Design (factorial):
    N concepts
    × 2 CAV methods         (mean_diff, svm)
    × L layer conditions    (each extraction layer individually + all-layers)
    × K alphas              (INTENSITIES — skipping 0.0 baseline)
    × 2 ablation modes      (all_tokens / image_tokens_only)

For each condition:
  - ON-TARGET:  ablate concept C, measure R1 accuracy on concept C images
  - OFF-TARGET: ablate concept C, measure R1 on N_OFF_TARGET other concepts
                → specificity_ratio = mean(on_delta) / mean(off_delta)

Strategy — "baseline-correct-only":
    Ablation is evaluated ONLY on images the model correctly classifies at
    baseline (no hooks). This fixes baseline_accuracy = 1.0 by construction,
    so every observed drop is unambiguously caused by the ablation.

Ablation modes:
    all_tokens        — ablation hooks fire on ALL token positions (image + text).
                        Stronger intervention; removes concept from multimodal
                        integration.
    image_tokens_only — hooks fire only on the 576 CLIP image-token positions.
                        Simulates pure visual-concept agnosia: the model's
                        text-token processing is unaffected.

Input:
    data/experiment_data.json       (r1_benchmark section)
    cavs/{concept}_{method}_layer{L}.pt

Output:
    results/cct_test_results.json

Usage (from models/LLaVA_1_5_7B/):
    python -m test.cct_test
"""

import json
import os
import random
import sys
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# Path bootstrap — allow both `python -m test.cct_test` and direct execution
# ---------------------------------------------------------------------------
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
from experiment_runner import load_cav
from metrics import evaluate_r1
from model_loader import load_llava_model
from pyvene_utils import create_ablation_hooks

# ---------------------------------------------------------------------------
# Configuration (all values from experiment_config.py)
# ---------------------------------------------------------------------------
CONCEPTS     = cfg.CONCEPTS
CAV_METHODS  = cfg.CAV_METHODS
SINGLE_LAYERS = cfg.EXTRACTION_LAYERS            # [16, 24, 27] — tested one at a time
ALL_LAYERS    = list(cfg.EXTRACTION_LAYERS)      # all extraction layers simultaneously
ALPHAS        = [a for a in cfg.INTENSITIES if a > 0.0]   # skip 0.0 (that's baseline)

RESULTS_DIR  = cfg.RESULTS_DIR
OUTPUT_FILE  = os.path.join(RESULTS_DIR, "cct_test_results.json")

# Ablation modes: (label, image_tokens_only flag)
ABLATION_MODES = [
    ("all_tokens",        False),   # modify all token positions
    ("image_tokens_only", True),    # modify only the 576 CLIP image-token positions
]

# Minimum number of baseline-correct items required to include a concept
# (lower than text-CCT's 5 because the pilot R1 benchmark has ~15 items/concept)
MIN_CORRECT = 3

# Off-target concepts sampled per ablation condition
N_OFF_TARGET = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_cav_context(model, cavs_by_layer, alpha, technique, image_tokens_only):
    """
    Register one ablation hook per layer, each with its own per-layer CAV.

    This replicates the text-CCT's build_multi_layer_hooks() pattern:
    in the all-layers condition each layer carries the CAV that was
    trained specifically for that layer's activation space.

    Returns list of hook handles — caller is responsible for removal.
    """
    handles = []
    for layer_idx, cav in cavs_by_layer.items():
        h = create_ablation_hooks(
            model, cav, alpha, [layer_idx], technique, image_tokens_only,
        )
        handles.extend(h)
    return handles


def evaluate_r1_with_multi_cav(model, processor, items, cavs_by_layer,
                                alpha, technique, image_tokens_only):
    """
    Evaluate R1 accuracy while multiple per-layer CAV hooks are active.

    PyTorch hooks registered by _multi_cav_context() fire automatically
    during every forward pass, so evaluate_r1() is called without any
    cav/alpha arguments — the intervention is already in place.
    """
    handles = _multi_cav_context(
        model, cavs_by_layer, alpha, technique, image_tokens_only,
    )
    try:
        acc, per_item = evaluate_r1(model, processor, items)
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_image_token_mask"):
            del model._image_token_mask
    return acc, per_item


def run_r1(model, processor, items, cav, layers, alpha, technique, image_tokens_only):
    """
    Unified R1 evaluation: single-layer (via evaluate_r1 API) or
    multi-layer with per-layer CAVs (via evaluate_r1_with_multi_cav).

    Args:
        cav:    For single-layer, a single Tensor [d_model].
                For multi-layer, a dict {layer_idx: Tensor}.
        layers: List of layer indices (1 element → single-layer).
    """
    if isinstance(cav, dict):
        # Multi-layer: each layer has its own CAV
        return evaluate_r1_with_multi_cav(
            model, processor, items, cav, alpha, technique, image_tokens_only,
        )
    else:
        # Single-layer: delegate to evaluate_r1
        return evaluate_r1(
            model, processor, items,
            cav=cav, alpha=alpha,
            layers=layers, technique=technique,
            image_tokens_only=image_tokens_only,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  CCT Test — Visual Concept Classification Test (LLaVA-1.5 7B)")
    print(f"  Concepts       : {CONCEPTS}")
    print(f"  CAV methods    : {CAV_METHODS}")
    print(f"  Single layers  : {SINGLE_LAYERS}")
    print(f"  All-layers cond: {ALL_LAYERS}")
    print(f"  Alphas         : {ALPHAS}")
    print(f"  Ablation modes : {[m for m, _ in ABLATION_MODES]}")
    print(f"  Min correct    : {MIN_CORRECT}")
    print(f"{sep}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load benchmark data
    # ------------------------------------------------------------------
    print(f"\n  Loading data from {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    r1_data_by_concept = data.get("r1_benchmark", {})

    available = [
        c for c in CONCEPTS
        if c in r1_data_by_concept and len(r1_data_by_concept[c]) > 0
    ]
    missing = [c for c in CONCEPTS if c not in available]
    if missing:
        print(f"  WARNING: {len(missing)} concept(s) absent from r1_benchmark: {missing}")
    assert available, (
        "No concepts found in r1_benchmark — run data_gen_multimodal.py first."
    )

    for c in available:
        print(f"  {c}: {len(r1_data_by_concept[c])} R1 items")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n  Loading {cfg.MODEL_NAME} (4-bit)...")
    model, processor = load_llava_model()
    print(f"  Model loaded.\n")

    # ==================================================================
    # PHASE 1 — BASELINE (no ablation)
    # ==================================================================
    print(f"{'='*50}")
    print("PHASE 1: BASELINE (no ablation)")
    print(f"{'='*50}")

    baselines = {}
    correct_items_by_concept = {}

    for concept in available:
        items = r1_data_by_concept[concept]
        acc, per_item = evaluate_r1(model, processor, items)

        correct_idx   = [i for i, pi in enumerate(per_item) if pi["is_correct"]]
        correct_items = [items[i] for i in correct_idx]

        baselines[concept] = {
            "accuracy": acc,
            "n_items":  len(items),
            "per_item": per_item,
        }
        correct_items_by_concept[concept] = correct_items

        print(f"  {concept.upper():>20s}: acc={acc:.3f}  "
              f"({len(correct_items)}/{len(items)} correct)")

    # ------------------------------------------------------------------
    # Filter: require MIN_CORRECT baseline-correct items
    # ------------------------------------------------------------------
    all_available = list(available)
    available = [c for c in available
                 if len(correct_items_by_concept[c]) >= MIN_CORRECT]
    filtered_out = [c for c in all_available if c not in available]

    if filtered_out:
        print(f"\n  Filtered out {len(filtered_out)} concept(s) with "
              f"< {MIN_CORRECT} correct items: {filtered_out}")
    print(f"  Concepts for ablation: {len(available)} / {len(all_available)}")
    print(f"  Strategy: baseline-correct-only (baseline_accuracy = 1.0 by construction)")
    assert available, f"No concept has >= {MIN_CORRECT} correctly classified images!"

    for c in available:
        print(f"    {c}: {len(correct_items_by_concept[c])} correct items")

    # ==================================================================
    # PHASE 2 — FACTORIAL ABLATION
    # ==================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL ABLATION")
    print(f"{'='*50}")

    # Layer conditions: each individual extraction layer + all-layers combined
    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_extraction_layers", ALL_LAYERS))

    n_off = min(N_OFF_TARGET, len(available) - 1)

    total_conditions = (
        len(available) * len(CAV_METHODS) * len(layer_conditions)
        * len(ALPHAS) * len(ABLATION_MODES)
    )
    print(f"  Ablation conditions   : {total_conditions}")
    print(f"  Off-target per cond   : {n_off}")
    print(f"  Total R1 evaluations  : {total_conditions * (1 + n_off)}\n")

    results      = []
    condition_num = 0

    for ablated_concept in available:
        for cav_method in CAV_METHODS:

            for layer_label, layers in layer_conditions:
                is_multi = len(layers) > 1

                # ---- Load CAV(s) ----------------------------------------
                if is_multi:
                    # Per-layer CAVs for the all-layers condition
                    cavs_by_layer = {}
                    for l in layers:
                        try:
                            cavs_by_layer[l] = load_cav(ablated_concept, cav_method, l)
                        except FileNotFoundError:
                            print(f"  WARNING: CAV missing — "
                                  f"{ablated_concept}/{cav_method}/L{l}, skipping layer")
                    if not cavs_by_layer:
                        print(f"  SKIP {ablated_concept}/{cav_method}/"
                              f"{layer_label}: no CAVs found")
                        continue
                    cav_arg = cavs_by_layer   # dict → multi-layer path in run_r1()
                else:
                    layer = layers[0]
                    try:
                        cav_arg = load_cav(ablated_concept, cav_method, layer)
                    except FileNotFoundError:
                        print(f"  SKIP {ablated_concept}/{cav_method}/L{layer}: "
                              f"CAV not found")
                        continue
                # ---------------------------------------------------------

                for alpha in ALPHAS:
                    # Deterministic off-target sample (reproducible across runs)
                    off_candidates = [c for c in available if c != ablated_concept]
                    random.seed(hash((ablated_concept, cav_method, layer_label, alpha)))
                    off_sample = (random.sample(off_candidates, n_off)
                                  if off_candidates else [])
                    measure_concepts = [ablated_concept] + off_sample

                    for mode_label, image_only in ABLATION_MODES:
                        condition_num += 1
                        print(f"  [{condition_num}/{total_conditions}]  "
                              f"Ablate {ablated_concept}/{cav_method}/"
                              f"{layer_label}/a={alpha}/{mode_label}")

                        for measured_concept in measure_concepts:
                            on_target = measured_concept == ablated_concept
                            tag = "ON-TARGET" if on_target else "off-target"

                            acc, per_item = run_r1(
                                model, processor,
                                correct_items_by_concept[measured_concept],
                                cav=cav_arg,
                                layers=layers,
                                alpha=alpha,
                                technique="projection",
                                image_tokens_only=image_only,
                            )

                            # baseline_accuracy = 1.0 by construction (correct items only)
                            delta = 1.0 - acc

                            print(f"    >> {measured_concept.upper():>20s} [{tag}]: "
                                  f"acc={acc:.3f}  d={delta:+.3f}")

                            results.append({
                                "ablated_concept":   ablated_concept,
                                "measured_concept":  measured_concept,
                                "on_target":         on_target,
                                "cav_method":        cav_method,
                                "layer":             layer_label,
                                "alpha":             alpha,
                                "ablation_mode":     mode_label,
                                "accuracy":          acc,
                                "baseline_accuracy": 1.0,
                                "delta_accuracy":    delta,
                                "per_item":          per_item,
                            })

    # ==================================================================
    # PHASE 3 — SPECIFICITY ANALYSIS
    # ==================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY ANALYSIS")
    print(f"{'='*50}")

    specificity = {}

    for mode_label, _ in ABLATION_MODES:
        mode_res = [r for r in results if r["ablation_mode"] == mode_label]

        on_deltas  = [r["delta_accuracy"] for r in mode_res if r["on_target"]]
        off_deltas = [r["delta_accuracy"] for r in mode_res if not r["on_target"]]

        mean_on  = sum(on_deltas)  / len(on_deltas)  if on_deltas  else 0.0
        mean_off = sum(off_deltas) / len(off_deltas) if off_deltas else 0.0

        if mean_off > 0:
            ratio = mean_on / mean_off
        else:
            ratio = float("inf") if mean_on > 0 else 0.0

        status = "PASS" if ratio >= cfg.SPECIFICITY_RATIO_THRESHOLD else "FAIL"

        print(f"\n  ── Mode: {mode_label} ──")
        print(f"    Mean on-target  dacc : {mean_on:+.4f}")
        print(f"    Mean off-target dacc : {mean_off:+.4f}")
        print(f"    Specificity ratio    : {ratio:.2f}  [{status}]")

        # Per-concept breakdown
        per_concept = {}
        for ablated in available:
            c_res  = [r for r in mode_res if r["ablated_concept"] == ablated]
            on_d   = [r["delta_accuracy"] for r in c_res if r["on_target"]]
            off_d  = [r["delta_accuracy"] for r in c_res if not r["on_target"]]
            m_on   = sum(on_d)  / len(on_d)  if on_d  else 0.0
            m_off  = sum(off_d) / len(off_d) if off_d else 0.0
            r_c    = (m_on / m_off) if m_off > 0 else (float("inf") if m_on > 0 else 0.0)

            print(f"    {ablated.upper():>20s}: "
                  f"on={m_on:+.4f}  off={m_off:+.4f}  ratio={r_c:.2f}")

            per_concept[ablated] = {
                "mean_on_target_delta":  m_on,
                "mean_off_target_delta": m_off,
                "specificity_ratio":     r_c,
            }

        specificity[mode_label] = {
            "mean_on_target_delta":  mean_on,
            "mean_off_target_delta": mean_off,
            "specificity_ratio":     ratio,
            "pass":                  ratio >= cfg.SPECIFICITY_RATIO_THRESHOLD,
            "per_concept":           per_concept,
        }

    # ---- Cross-mode comparison (image_tokens_only vs all_tokens) -----
    if len(ABLATION_MODES) == 2:
        labels = [m for m, _ in ABLATION_MODES]
        r0 = specificity[labels[0]]["specificity_ratio"]
        r1_v = specificity[labels[1]]["specificity_ratio"]
        more_specific = labels[1] if r1_v > r0 else labels[0]
        print(f"\n  Mode comparison:")
        print(f"    {labels[0]}: ratio={r0:.2f}")
        print(f"    {labels[1]}: ratio={r1_v:.2f}")
        print(f"    >> '{more_specific}' is more selective")

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    output = {
        "metadata": {
            "test":        "CCT — Visual Concept Classification Test",
            "model":       cfg.MODEL_NAME,
            "timestamp":   datetime.now().isoformat(),
            "strategy":    "baseline-correct-only",
            "strategy_description": (
                "Ablation is evaluated only on images the model correctly "
                "classifies at baseline (no hooks). "
                "baseline_accuracy = 1.0 by construction; every accuracy "
                "drop is directly caused by the CAV ablation."
            ),
            "technique":                "projection",
            "layers_single":            list(SINGLE_LAYERS),
            "layers_all_cond":          ALL_LAYERS,
            "alphas":                   ALPHAS,
            "cav_methods":              CAV_METHODS,
            "ablation_modes":           [m for m, _ in ABLATION_MODES],
            "n_off_target":             n_off,
            "min_correct_items":        MIN_CORRECT,
            "n_image_tokens":           cfg.N_IMAGE_TOKENS,
            "token_position_cav":       cfg.TOKEN_POSITION,
            "concepts_configured":      list(CONCEPTS),
            "concepts_evaluated":       list(available),
            "concepts_filtered_out":    filtered_out,
        },
        "baselines": {
            c: {
                "accuracy":  baselines[c]["accuracy"],
                "n_items":   baselines[c]["n_items"],
                "n_correct": len(correct_items_by_concept.get(c, [])),
                "per_item":  baselines[c]["per_item"],
            }
            for c in all_available
        },
        "results":     results,
        "specificity": specificity,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"\n{'='*60}")
    print("  CCT TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
