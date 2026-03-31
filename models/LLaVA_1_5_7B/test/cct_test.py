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

Features:
    - Progressive checkpoint saving (resilient to OOM crashes)
    - Automatic resume from checkpoint on restart
    - Memory cleanup (gc + torch.cuda.empty_cache) per layer condition
    - Dual output: console + log file

Input:
    data/experiment_data.json       (r1_benchmark section)
    cavs/{concept}_{method}_layer{L}.pt

Output:
    results/cct_test_results.json
    results/cct_test_log.txt

Usage (from models/LLaVA_1_5_7B/):
    python -m test.cct_test
"""

import gc
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
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "cct_checkpoint.json")
CRASH_LOG_FILE  = os.path.join(RESULTS_DIR, "cct_crash_keys.json")
LOG_FILE        = os.path.join(RESULTS_DIR, "cct_test_log.txt")
CHECKPOINT_EVERY = 1   # save checkpoint every N conditions (frequent due to GPU crashes)
MAX_RETRIES      = 2   # skip evaluation after this many crashes on the same key

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
# Logging helper — prints to console AND appends to log file
# ---------------------------------------------------------------------------
_log_file_handle = None


def log(msg=""):
    """Print to console and append to log file."""
    print(msg)
    if _log_file_handle is not None:
        _log_file_handle.write(msg + "\n")
        _log_file_handle.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_cav_context(model, cavs_by_layer, alpha, technique, image_tokens_only):
    """
    Register one ablation hook per layer, each with its own per-layer CAV.
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
    Unified R1 evaluation: single-layer or multi-layer with per-layer CAVs.
    """
    if isinstance(cav, dict):
        return evaluate_r1_with_multi_cav(
            model, processor, items, cav, alpha, technique, image_tokens_only,
        )
    else:
        return evaluate_r1(
            model, processor, items,
            cav=cav, alpha=alpha,
            layers=layers, technique=technique,
            image_tokens_only=image_tokens_only,
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers (progressive save / resume on OOM)
# ---------------------------------------------------------------------------

def make_condition_key(ablated, cav_method, layer_label, alpha, mode_label, measured):
    """Deterministic string key that uniquely identifies one R1 evaluation."""
    return f"{ablated}|{cav_method}|{layer_label}|{alpha}|{mode_label}|{measured}"


def save_checkpoint(baselines, results, completed_keys, available, filtered_out):
    """Atomically write checkpoint to disk (temp file + rename, fsync).
    Keeps a .bak copy so a crash during write doesn't corrupt both files."""
    data = {
        "baselines":      baselines,
        "results":        results,
        "completed_keys": list(completed_keys),
        "available":      available,
        "filtered_out":   filtered_out,
        "timestamp":      datetime.now().isoformat(),
    }
    tmp = CHECKPOINT_FILE + ".tmp"
    bak = CHECKPOINT_FILE + ".bak"
    # Write to temp file with fsync
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    # Keep backup of current checkpoint before replacing
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.replace(CHECKPOINT_FILE, bak)
        except OSError:
            pass
    os.replace(tmp, CHECKPOINT_FILE)


def load_checkpoint():
    """Load checkpoint if it exists; fall back to .bak if main is corrupt."""
    for path in [CHECKPOINT_FILE, CHECKPOINT_FILE + ".bak"]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if path != CHECKPOINT_FILE:
                print(f"  WARNING: Main checkpoint corrupt, loaded from backup")
            return data
        except (json.JSONDecodeError, IOError):
            continue
    return None


def load_crash_log():
    """Load crash attempt counts per key. Returns dict {key: count}."""
    if not os.path.exists(CRASH_LOG_FILE):
        return {}
    try:
        with open(CRASH_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_crash_log(crash_counts):
    """Save crash attempt counts (with fsync to survive SIGILL)."""
    with open(CRASH_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(crash_counts, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def record_pending_key(key):
    """Record that we're about to attempt this key. If we crash, the next
    run will see it in the crash log and increment its retry count."""
    crash_counts = load_crash_log()
    crash_counts[key] = crash_counts.get(key, 0) + 1
    save_crash_log(crash_counts)


def clear_pending_key(key):
    """Remove a key from crash log after successful completion."""
    crash_counts = load_crash_log()
    if key in crash_counts:
        del crash_counts[key]
        save_crash_log(crash_counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _log_file_handle

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log_file_handle = open(LOG_FILE, "a", encoding="utf-8")
    log(f"\n{'='*60}")
    log(f"  CCT Test started at {datetime.now().isoformat()}")
    log(f"{'='*60}")

    sep = "=" * 60
    log(f"\n{sep}")
    log("  CCT Test — Visual Concept Classification Test (LLaVA-1.5 7B)")
    log(f"  Concepts       : {CONCEPTS}")
    log(f"  CAV methods    : {CAV_METHODS}")
    log(f"  Single layers  : {SINGLE_LAYERS}")
    log(f"  All-layers cond: {ALL_LAYERS}")
    log(f"  Alphas         : {ALPHAS}")
    log(f"  Ablation modes : {[m for m, _ in ABLATION_MODES]}")
    log(f"  Min correct    : {MIN_CORRECT}")
    log(f"{sep}")

    # ------------------------------------------------------------------
    # Load benchmark data
    # ------------------------------------------------------------------
    log(f"\n  Loading data from {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    r1_data_by_concept = data.get("r1_benchmark", {})

    available = [
        c for c in CONCEPTS
        if c in r1_data_by_concept and len(r1_data_by_concept[c]) > 0
    ]
    missing = [c for c in CONCEPTS if c not in available]
    if missing:
        log(f"  WARNING: {len(missing)} concept(s) absent from r1_benchmark: {missing}")
    assert available, (
        "No concepts found in r1_benchmark — run data_gen_multimodal.py first."
    )

    for c in available:
        log(f"  {c}: {len(r1_data_by_concept[c])} R1 items")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    log(f"\n  Loading {cfg.MODEL_NAME} (4-bit)...")
    model, processor = load_llava_model()
    log(f"  Model loaded.\n")

    # ==================================================================
    # CHECK FOR EXISTING CHECKPOINT
    # ==================================================================
    checkpoint = load_checkpoint()
    resuming = False

    if checkpoint and set(checkpoint.get("available", [])).issubset(set(available)):
        ckpt_available = checkpoint["available"]
        if all(c in available for c in ckpt_available):
            resuming = True
            log(f"\n  RESUMING from checkpoint "
                f"({len(checkpoint['completed_keys'])} evaluations completed)")

    # ==================================================================
    # PHASE 1 — BASELINE (no ablation)
    # ==================================================================
    log(f"\n{'='*50}")
    log("PHASE 1: BASELINE (no ablation)")
    log(f"{'='*50}")

    baselines = {}
    correct_items_by_concept = {}

    if resuming and "baselines" in checkpoint:
        # Restore baselines from checkpoint — rebuild correct_items
        baselines = checkpoint["baselines"]
        for concept in available:
            items = r1_data_by_concept[concept]
            if concept in baselines and "per_item" in baselines[concept]:
                per_item = baselines[concept]["per_item"]
                correct_idx = [i for i, pi in enumerate(per_item) if pi["is_correct"]]
                correct_items = [items[i] for i in correct_idx]
                correct_items_by_concept[concept] = correct_items
                log(f"  {concept.upper():>20s}: {len(correct_items)} correct items [from checkpoint]")
            else:
                # Concept not in checkpoint — evaluate fresh
                acc, per_item = evaluate_r1(model, processor, items)
                correct_idx = [i for i, pi in enumerate(per_item) if pi["is_correct"]]
                correct_items = [items[i] for i in correct_idx]
                baselines[concept] = {
                    "accuracy": acc,
                    "n_items":  len(items),
                    "per_item": per_item,
                }
                correct_items_by_concept[concept] = correct_items
                log(f"  {concept.upper():>20s}: acc={acc:.3f}  "
                    f"({len(correct_items)}/{len(items)} correct)")
    else:
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

            log(f"  {concept.upper():>20s}: acc={acc:.3f}  "
                f"({len(correct_items)}/{len(items)} correct)")

    # ------------------------------------------------------------------
    # Filter: require MIN_CORRECT baseline-correct items
    # ------------------------------------------------------------------
    all_available = list(available)
    available = [c for c in available
                 if len(correct_items_by_concept.get(c, [])) >= MIN_CORRECT]
    filtered_out = [c for c in all_available if c not in available]

    if filtered_out:
        log(f"\n  Filtered out {len(filtered_out)} concept(s) with "
            f"< {MIN_CORRECT} correct items: {filtered_out}")
    log(f"  Concepts for ablation: {len(available)} / {len(all_available)}")
    log(f"  Strategy: baseline-correct-only (baseline_accuracy = 1.0 by construction)")
    assert available, f"No concept has >= {MIN_CORRECT} correctly classified images!"

    for c in available:
        log(f"    {c}: {len(correct_items_by_concept[c])} correct items")

    # ==================================================================
    # PHASE 2 — FACTORIAL ABLATION
    # ==================================================================
    log(f"\n{'='*50}")
    log("PHASE 2: FACTORIAL ABLATION")
    log(f"{'='*50}")

    # Layer conditions: each individual extraction layer + all-layers combined
    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_extraction_layers", ALL_LAYERS))

    n_off = min(N_OFF_TARGET, len(available) - 1)

    total_conditions = (
        len(available) * len(CAV_METHODS) * len(layer_conditions)
        * len(ALPHAS) * len(ABLATION_MODES)
    )
    log(f"  Ablation conditions   : {total_conditions}")
    log(f"  Off-target per cond   : {n_off}")
    log(f"  Total R1 evaluations  : {total_conditions * (1 + n_off)}\n")

    # Load existing progress from checkpoint
    completed_keys = set(checkpoint["completed_keys"]) if resuming else set()
    results        = list(checkpoint["results"])        if resuming else []
    condition_num  = 0

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
                            log(f"  WARNING: CAV missing — "
                                f"{ablated_concept}/{cav_method}/L{l}, skipping layer")
                    if not cavs_by_layer:
                        log(f"  SKIP {ablated_concept}/{cav_method}/"
                            f"{layer_label}: no CAVs found")
                        continue
                    cav_arg = cavs_by_layer
                else:
                    layer = layers[0]
                    try:
                        cav_arg = load_cav(ablated_concept, cav_method, layer)
                    except FileNotFoundError:
                        log(f"  SKIP {ablated_concept}/{cav_method}/L{layer}: "
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
                        log(f"  [{condition_num}/{total_conditions}]  "
                            f"Ablate {ablated_concept}/{cav_method}/"
                            f"{layer_label}/a={alpha}/{mode_label}")

                        for measured_concept in measure_concepts:
                            key = make_condition_key(
                                ablated_concept, cav_method, layer_label,
                                alpha, mode_label, measured_concept,
                            )
                            if key in completed_keys:
                                log(f"    >> {measured_concept.upper():>20s} [CACHED]")
                                continue

                            # Check crash log — skip if this key crashed too many times
                            crash_counts = load_crash_log()
                            if crash_counts.get(key, 0) >= MAX_RETRIES:
                                log(f"    >> {measured_concept.upper():>20s} "
                                    f"[SKIPPED — crashed {crash_counts[key]}x]")
                                completed_keys.add(key)
                                continue

                            on_target = measured_concept == ablated_concept
                            tag = "ON-TARGET" if on_target else "off-target"

                            # Save checkpoint + record pending key BEFORE the forward pass.
                            # If SIGILL kills the process, the crash log will have this key
                            # and the next restart will know it crashed.
                            record_pending_key(key)
                            save_checkpoint(baselines, results, completed_keys,
                                            available, filtered_out)

                            try:
                                acc, per_item = run_r1(
                                    model, processor,
                                    correct_items_by_concept[measured_concept],
                                    cav=cav_arg,
                                    layers=layers,
                                    alpha=alpha,
                                    technique="projection",
                                    image_tokens_only=image_only,
                                )
                            except Exception as e:
                                log(f"    >> {measured_concept.upper():>20s} [{tag}]: "
                                    f"ERROR — {type(e).__name__}: {e}")
                                completed_keys.add(key)
                                gc.collect()
                                torch.cuda.empty_cache()
                                continue

                            # Success — clear from crash log
                            clear_pending_key(key)

                            # baseline_accuracy = 1.0 by construction (correct items only)
                            delta = 1.0 - acc

                            log(f"    >> {measured_concept.upper():>20s} [{tag}]: "
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
                            completed_keys.add(key)

                        # -- Periodic checkpoint save --
                        if condition_num % CHECKPOINT_EVERY == 0:
                            save_checkpoint(baselines, results, completed_keys,
                                            available, filtered_out)
                            log(f"    [CHECKPOINT] Saved at condition "
                                f"{condition_num}/{total_conditions}")

                # -- Memory cleanup after each layer condition --
                del cav_arg
                gc.collect()
                torch.cuda.empty_cache()

    # Final checkpoint after Phase 2
    save_checkpoint(baselines, results, completed_keys, available, filtered_out)
    log(f"  [CHECKPOINT] Phase 2 complete — {len(results)} results saved")

    # ==================================================================
    # PHASE 3 — SPECIFICITY ANALYSIS
    # ==================================================================
    log(f"\n{'='*50}")
    log("PHASE 3: SPECIFICITY ANALYSIS")
    log(f"{'='*50}")

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

        log(f"\n  ── Mode: {mode_label} ──")
        log(f"    Mean on-target  dacc : {mean_on:+.4f}")
        log(f"    Mean off-target dacc : {mean_off:+.4f}")
        log(f"    Specificity ratio    : {ratio:.2f}  [{status}]")

        # Per-concept breakdown
        per_concept = {}
        for ablated in available:
            c_res  = [r for r in mode_res if r["ablated_concept"] == ablated]
            on_d   = [r["delta_accuracy"] for r in c_res if r["on_target"]]
            off_d  = [r["delta_accuracy"] for r in c_res if not r["on_target"]]
            m_on   = sum(on_d)  / len(on_d)  if on_d  else 0.0
            m_off  = sum(off_d) / len(off_d) if off_d else 0.0
            r_c    = (m_on / m_off) if m_off > 0 else (float("inf") if m_on > 0 else 0.0)

            log(f"    {ablated.upper():>20s}: "
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
        log(f"\n  Mode comparison:")
        log(f"    {labels[0]}: ratio={r0:.2f}")
        log(f"    {labels[1]}: ratio={r1_v:.2f}")
        log(f"    >> '{more_specific}' is more selective")

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

    log(f"\n  Saved: {OUTPUT_FILE}")

    # Remove checkpoint and crash log after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log(f"  Checkpoint removed: {CHECKPOINT_FILE}")
    if os.path.exists(CRASH_LOG_FILE):
        os.remove(CRASH_LOG_FILE)
        log(f"  Crash log removed: {CRASH_LOG_FILE}")

    log(f"\n{'='*60}")
    log("  CCT TEST COMPLETE")
    log(f"{'='*60}")

    if _log_file_handle is not None:
        _log_file_handle.close()


if __name__ == "__main__":
    main()
