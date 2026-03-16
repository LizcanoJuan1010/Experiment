"""
SFA Test — Semantic Feature Analysis Test (LLaVA-1.5 7B)
=========================================================
Visual analog of the Semantic Feature Analysis aphasia assessment
(McKenna & Warrington 1980; Patterson & Hodges 1992).

Given an IMAGE of a concept (e.g., a dog), the model is asked binary yes/no
questions across five semantic dimensions:

    category    — "Is this an animal?"
    function    — "Is this used for companionship?"
    perceptual  — "Is this typically furry?"
    structural  — "Does this have four legs?"
    associative — "Is this typically found in homes?"

Each dimension provides one TRUE feature (expected "yes") and one FOIL
(expected "no"). After CAV ablation, per-dimension accuracy drops reveal
WHICH semantic features are lost — e.g., "functional knowledge of 'dog'
collapses while perceptual knowledge is preserved." This dimension-specific
dissociation mirrors clinical SFA findings in semantic dementia patients.

Design (factorial, identical structure to cct_test.py):
    N concepts
    × 2 CAV methods         (mean_diff, svm)
    × L layer conditions    (each extraction layer individually + all-layers)
    × K alphas              (INTENSITIES — skipping 0.0 baseline)
    × 2 ablation modes      (all_tokens / image_tokens_only)

For each condition:
    ON-TARGET  — ablate concept C, evaluate SFA for concept C images
    OFF-TARGET — ablate concept C, evaluate SFA for N_OFF_TARGET other concepts

Strategy — "baseline-correct-only":
    Ablation evaluated only on (image, question) pairs the model answers
    correctly at baseline. baseline_accuracy = 1.0 by construction.

Scoring:
    prompt = SFA_PROMPT_TEMPLATE.format(question=question)
    loss_yes = sequence_loss(prompt + " yes")
    loss_no  = sequence_loss(prompt + " no")
    model_answer = argmin(loss_yes, loss_no)

Input:
    data/experiment_data.json       (r1_benchmark section provides images)
    cavs/{concept}_{method}_layer{L}.pt

Output:
    results/sfa_test_results.json

Usage (from models/LLaVA_1_5_7B/):
    python -m test.sfa_test
"""

import json
import os
import random
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
from experiment_runner import load_cav
from metrics import evaluate_sfa
from model_loader import load_llava_model
from pyvene_utils import create_ablation_hooks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONCEPTS      = cfg.CONCEPTS
CAV_METHODS   = cfg.CAV_METHODS
SINGLE_LAYERS = cfg.EXTRACTION_LAYERS           # [16, 24, 27] — tested one at a time
ALL_LAYERS    = list(cfg.EXTRACTION_LAYERS)     # all extraction layers simultaneously
ALPHAS        = [a for a in cfg.INTENSITIES if a > 0.0]

RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sfa_test_results.json")

DIMENSIONS = ["category", "function", "perceptual", "structural", "associative"]

ABLATION_MODES = [
    ("all_tokens",        False),   # ablate all token positions
    ("image_tokens_only", True),    # ablate only the 576 CLIP image-token positions
]

MIN_CORRECT  = 3    # min baseline-correct (image, question) pairs per concept
N_OFF_TARGET = 2    # off-target concepts sampled per ablation condition


# ---------------------------------------------------------------------------
# Multi-layer CAV helpers (mirrored from cct_test.py)
# ---------------------------------------------------------------------------

def _multi_cav_context(model, cavs_by_layer, alpha, technique, image_tokens_only):
    """
    Register one ablation hook per layer, each with its own per-layer CAV.
    Returns list of handles — caller must remove them.
    """
    handles = []
    for layer_idx, cav in cavs_by_layer.items():
        h = create_ablation_hooks(
            model, cav, alpha, [layer_idx], technique, image_tokens_only,
        )
        handles.extend(h)
    return handles


def evaluate_sfa_with_multi_cav(model, processor, items, cavs_by_layer,
                                 alpha, technique, image_tokens_only):
    """
    Evaluate SFA while multiple per-layer CAV hooks are active simultaneously.

    PyTorch hooks fire automatically during every forward pass, so
    evaluate_sfa() is called without ablation parameters — the intervention
    is already in place via the registered hooks.
    """
    handles = _multi_cav_context(
        model, cavs_by_layer, alpha, technique, image_tokens_only,
    )
    try:
        acc, acc_by_dim, per_item = evaluate_sfa(model, processor, items)
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_image_token_mask"):
            del model._image_token_mask
    return acc, acc_by_dim, per_item


def run_sfa(model, processor, items, cav, layers, alpha, technique, image_tokens_only):
    """
    Unified SFA evaluation:
        cav is dict  → multi-layer (one CAV per layer)
        cav is Tensor → single-layer via evaluate_sfa() API
    """
    if isinstance(cav, dict):
        return evaluate_sfa_with_multi_cav(
            model, processor, items, cav, alpha, technique, image_tokens_only,
        )
    return evaluate_sfa(
        model, processor, items,
        cav=cav, alpha=alpha,
        layers=layers, technique="projection",
        image_tokens_only=image_tokens_only,
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_sfa_items(r1_images, feature_bank):
    """
    Expand to the cross-product: every R1 image × every feature question.

    Args:
        r1_images:    list of R1 benchmark dicts (must have "image_path" key)
        feature_bank: list of (question, expected_answer, dimension) tuples

    Returns:
        Flat list of sfa_item dicts ready for evaluate_sfa().
    """
    items = []
    for img_item in r1_images:
        for question, expected_answer, dimension in feature_bank:
            items.append({
                "image_path":      img_item["image_path"],
                "question":        question,
                "expected_answer": expected_answer,
                "dimension":       dimension,
            })
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SFA Test — Semantic Feature Analysis Test (LLaVA-1.5 7B)")
    print(f"  Concepts       : {CONCEPTS}")
    print(f"  CAV methods    : {CAV_METHODS}")
    print(f"  Single layers  : {SINGLE_LAYERS}")
    print(f"  All-layers cond: {ALL_LAYERS}")
    print(f"  Alphas         : {ALPHAS}")
    print(f"  Ablation modes : {[m for m, _ in ABLATION_MODES]}")
    print(f"  Dimensions     : {DIMENSIONS}")
    print(f"  Min correct    : {MIN_CORRECT}")
    print(f"{sep}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate feature bank and load benchmark images
    # ------------------------------------------------------------------
    assert cfg.SFA_FEATURE_BANK, (
        "SFA_FEATURE_BANK is empty. "
        "Ensure PILOT_MODE=True in experiment_config.py or define a full bank."
    )

    print(f"\n  Loading data from {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    r1_data_by_concept = data.get("r1_benchmark", {})

    available = [
        c for c in CONCEPTS
        if c in r1_data_by_concept and len(r1_data_by_concept[c]) > 0
        and c in cfg.SFA_FEATURE_BANK
    ]
    missing = [c for c in CONCEPTS if c not in available]
    if missing:
        print(f"  WARNING: {len(missing)} concept(s) missing from r1_benchmark "
              f"or SFA_FEATURE_BANK: {missing}")
    assert available, (
        "No concepts ready for SFA — run data_gen_multimodal.py and ensure "
        "SFA_FEATURE_BANK is populated."
    )

    # Validate dimension names in feature banks
    for c in available:
        unknown = [dim for _, _, dim in cfg.SFA_FEATURE_BANK[c]
                   if dim not in DIMENSIONS]
        assert not unknown, (
            f"Unknown dimension(s) in SFA_FEATURE_BANK['{c}']: {unknown}. "
            f"Valid: {DIMENSIONS}"
        )

    for c in available:
        n_r1   = len(r1_data_by_concept[c])
        n_feat = len(cfg.SFA_FEATURE_BANK[c])
        print(f"  {c}: {n_r1} R1 images × {n_feat} questions = {n_r1 * n_feat} SFA items")

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
    sfa_items_by_concept = {}        # keep originals for JSON output

    for concept in available:
        feature_bank = cfg.SFA_FEATURE_BANK[concept]
        sfa_items = build_sfa_items(r1_data_by_concept[concept], feature_bank)
        sfa_items_by_concept[concept] = sfa_items

        acc, acc_by_dim, per_item = evaluate_sfa(model, processor, sfa_items)

        # Baseline-correct-only filter at (image_path, question) pair level
        correct_items = [sfa_items[i]
                         for i, p in enumerate(per_item) if p["is_correct"]]

        baselines[concept] = {
            "accuracy":              acc,
            "accuracy_by_dimension": acc_by_dim,
            "n_items_total":         len(sfa_items),
            "n_items_correct":       len(correct_items),
            "per_item":              per_item,
        }
        correct_items_by_concept[concept] = correct_items

        print(f"\n  {concept.upper()}:")
        print(f"    Overall accuracy : {acc:.3f}  "
              f"({len(correct_items)}/{len(sfa_items)} correct)")
        for dim in DIMENSIONS:
            d_acc = acc_by_dim.get(dim, 0.0)
            dim_total = sum(1 for it in sfa_items if it["dimension"] == dim)
            dim_ok    = sum(1 for it, p in zip(sfa_items, per_item)
                           if it["dimension"] == dim and p["is_correct"])
            print(f"    {dim:<12s}: {d_acc:.3f}  ({dim_ok}/{dim_total})")

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
    print(f"\n  Concepts for ablation: {len(available)} / {len(all_available)}")
    print(f"  Strategy: baseline-correct-only "
          f"(baseline_accuracy = 1.0 by construction)")
    assert available, f"No concept has ≥ {MIN_CORRECT} correctly answered SFA items!"

    for c in available:
        print(f"    {c}: {len(correct_items_by_concept[c])} correct items")

    # ==================================================================
    # PHASE 2 — FACTORIAL ABLATION
    # ==================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL ABLATION")
    print(f"{'='*50}")

    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_extraction_layers", ALL_LAYERS))

    n_off = min(N_OFF_TARGET, len(available) - 1)

    total_conditions = (
        len(available) * len(CAV_METHODS) * len(layer_conditions)
        * len(ALPHAS) * len(ABLATION_MODES)
    )
    print(f"  Ablation conditions   : {total_conditions}")
    print(f"  Off-target per cond   : {n_off}")
    print(f"  Total SFA evaluations : {total_conditions * (1 + n_off)}\n")

    results      = []
    condition_num = 0

    for ablated_concept in available:
        for cav_method in CAV_METHODS:
            for layer_label, layers in layer_conditions:
                is_multi = len(layers) > 1

                # ---- Load CAV(s) ----------------------------------------
                if is_multi:
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
                    cav_arg = cavs_by_layer
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
                    # Deterministic off-target sample
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

                            acc, acc_by_dim, per_item = run_sfa(
                                model, processor,
                                correct_items_by_concept[measured_concept],
                                cav=cav_arg,
                                layers=layers,
                                alpha=alpha,
                                technique="projection",
                                image_tokens_only=image_only,
                            )

                            # baseline_accuracy = 1.0 by construction
                            delta = 1.0 - acc
                            delta_by_dim = {
                                d: 1.0 - acc_by_dim.get(d, 0.0)
                                for d in DIMENSIONS
                            }

                            dim_summary = "  ".join(
                                f"{d[:3]}={delta_by_dim[d]:+.2f}"
                                for d in DIMENSIONS
                            )
                            print(f"    → {measured_concept.upper():>20s} [{tag}]: "
                                  f"acc={acc:.3f}  Δ={delta:+.3f}  "
                                  f"[{dim_summary}]")

                            results.append({
                                "ablated_concept":        ablated_concept,
                                "measured_concept":       measured_concept,
                                "on_target":              on_target,
                                "cav_method":             cav_method,
                                "layer":                  layer_label,
                                "alpha":                  alpha,
                                "ablation_mode":          mode_label,
                                "accuracy":               acc,
                                "baseline_accuracy":      1.0,
                                "delta_accuracy":         delta,
                                "accuracy_by_dimension":  acc_by_dim,
                                "delta_by_dimension":     delta_by_dim,
                                "per_item":               per_item,
                            })

    # ==================================================================
    # PHASE 3 — SPECIFICITY + DIMENSION DISSOCIATION
    # ==================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY + DIMENSION DISSOCIATION")
    print(f"{'='*50}")

    specificity          = {}
    dimension_dissociation = {}

    for mode_label, _ in ABLATION_MODES:
        mode_res = [r for r in results if r["ablation_mode"] == mode_label]

        # ---- Specificity (same as cct_test.py) -----------------------
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
        print(f"    Mean on-target  Δacc : {mean_on:+.4f}")
        print(f"    Mean off-target Δacc : {mean_off:+.4f}")
        print(f"    Specificity ratio    : {ratio:.2f}  [{status}]")

        per_concept_spec = {}
        for ablated in available:
            c_res  = [r for r in mode_res if r["ablated_concept"] == ablated]
            on_d   = [r["delta_accuracy"] for r in c_res if r["on_target"]]
            off_d  = [r["delta_accuracy"] for r in c_res if not r["on_target"]]
            m_on   = sum(on_d)  / len(on_d)  if on_d  else 0.0
            m_off  = sum(off_d) / len(off_d) if off_d else 0.0
            r_c    = (m_on / m_off) if m_off > 0 else (
                float("inf") if m_on > 0 else 0.0
            )
            print(f"    {ablated.upper():>20s}: on={m_on:+.4f}  "
                  f"off={m_off:+.4f}  ratio={r_c:.2f}")
            per_concept_spec[ablated] = {
                "mean_on_target_delta":  m_on,
                "mean_off_target_delta": m_off,
                "specificity_ratio":     r_c,
            }

        specificity[mode_label] = {
            "mean_on_target_delta":  mean_on,
            "mean_off_target_delta": mean_off,
            "specificity_ratio":     ratio,
            "pass":                  ratio >= cfg.SPECIFICITY_RATIO_THRESHOLD,
            "per_concept":           per_concept_spec,
        }

        # ---- Dimension dissociation (new vs CCT) ---------------------
        print(f"\n  Dimension dissociation [{mode_label}]:")
        dissoc_per_concept = {}

        for ablated in available:
            on_res = [r for r in mode_res
                      if r["ablated_concept"] == ablated and r["on_target"]]
            if not on_res:
                continue

            delta_by_dim = {}
            for dim in DIMENSIONS:
                dim_deltas = [r["delta_by_dimension"].get(dim, 0.0)
                              for r in on_res]
                delta_by_dim[dim] = (
                    sum(dim_deltas) / len(dim_deltas) if dim_deltas else 0.0
                )

            # Warn about zero-count dimensions (should not happen if data is valid)
            zero_dims = [d for d in DIMENSIONS
                         if all(r["delta_by_dimension"].get(d) is None
                                for r in on_res)]
            if zero_dims:
                print(f"    WARNING: {ablated}: no data for dimensions {zero_dims}")

            # Rank by sensitivity (highest delta = most affected)
            dimension_ranking = sorted(
                DIMENSIONS, key=lambda d: delta_by_dim[d], reverse=True
            )
            most_affected  = dimension_ranking[0]
            least_affected = dimension_ranking[-1]

            print(f"    {ablated.upper()}:")
            for dim in dimension_ranking:
                marker = " ← most" if dim == most_affected else (
                    " ← least" if dim == least_affected else "")
                print(f"      {dim:<12s}: Δ={delta_by_dim[dim]:+.4f}{marker}")

            dissoc_per_concept[ablated] = {
                "delta_by_dimension":    delta_by_dim,
                "dimension_ranking":     dimension_ranking,
                "most_affected_dimension":  most_affected,
                "least_affected_dimension": least_affected,
            }

        dimension_dissociation[mode_label] = dissoc_per_concept

    # ---- Cross-mode comparison ----------------------------------------
    if len(ABLATION_MODES) == 2:
        labels = [m for m, _ in ABLATION_MODES]
        r0 = specificity[labels[0]]["specificity_ratio"]
        r1 = specificity[labels[1]]["specificity_ratio"]
        more_specific = labels[1] if r1 > r0 else labels[0]
        print(f"\n  Mode comparison:")
        print(f"    {labels[0]}: ratio={r0:.2f}")
        print(f"    {labels[1]}: ratio={r1:.2f}")
        print(f"    → '{more_specific}' is more selective")

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    output = {
        "metadata": {
            "test":        "SFA — Semantic Feature Analysis Test",
            "model":       cfg.MODEL_NAME,
            "timestamp":   datetime.now().isoformat(),
            "strategy":    "baseline-correct-only",
            "strategy_description": (
                "Ablation is evaluated only on (image, question) pairs the model "
                "answers correctly at baseline (no hooks). "
                "baseline_accuracy = 1.0 by construction; every drop is directly "
                "caused by the CAV ablation."
            ),
            "technique":             "projection",
            "layers_single":         list(SINGLE_LAYERS),
            "layers_all_cond":       ALL_LAYERS,
            "alphas":                ALPHAS,
            "cav_methods":           CAV_METHODS,
            "ablation_modes":        [m for m, _ in ABLATION_MODES],
            "dimensions":            DIMENSIONS,
            "feature_bank":          {
                c: cfg.SFA_FEATURE_BANK[c] for c in all_available
            },
            "n_off_target":          n_off,
            "min_correct_items":     MIN_CORRECT,
            "n_image_tokens":        cfg.N_IMAGE_TOKENS,
            "token_position_cav":    cfg.TOKEN_POSITION,
            "concepts_configured":   list(CONCEPTS),
            "concepts_evaluated":    list(available),
            "concepts_filtered_out": filtered_out,
        },
        "baselines": {
            c: {
                "accuracy":              baselines[c]["accuracy"],
                "accuracy_by_dimension": baselines[c]["accuracy_by_dimension"],
                "n_items_total":         baselines[c]["n_items_total"],
                "n_items_correct":       baselines[c]["n_items_correct"],
                "per_item":              baselines[c]["per_item"],
            }
            for c in all_available
        },
        "results":               results,
        "specificity":           specificity,
        "dimension_dissociation": dimension_dissociation,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved → {OUTPUT_FILE}")
    print(f"\n{'='*60}")
    print("  SFA TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
