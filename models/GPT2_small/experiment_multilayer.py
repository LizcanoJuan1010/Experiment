"""
Multi-Layer Ablation via Feature Steering Negativo (SVM CAVs)
==============================================================
Concept ablation using un-normalized SVM CAV vectors with
multi-layer intervention.

Key differences from previous experiments:
  - SVM CAVs are NOT normalized (natural scale of clf.coef_[0])
  - Multi-layer intervention (hooks on multiple layers simultaneously)
  - Mean-centering option (RepE-style)
  - Amplification control to validate vector direction
  - Full SVM validation (cross-validation accuracy, Cohen's d)

Design:
  36 ablation conditions (3 concepts × 3 modes × 2 alphas × 2 centering)
  + 3 amplification controls (multi-all, α=4.0, centered)
  = 39 total conditions

References:
  - Kim et al. 2018 — TCAV: SVM-based CAV extraction
  - Turner et al. 2023 — CAA: un-normalized vectors + multi-layer
  - Zou et al. 2023 — RepE: mean-centering
  - Belrose et al. 2023 — LEACE: multi-layer erasure

Usage:
    python experiment_multilayer.py

Input:  experiment_data.json, model gpt2-small
Output: cavs_steering/*.pt
        results_multilayer/results_factorial.csv
        results_multilayer/results_baseline.json
        results_multilayer/extraction_report.json
"""

import torch
import json
import os
import warnings
import numpy as np
import pandas as pd
from functools import partial
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformer_lens import HookedTransformer

import experiment_config as cfg
from cav_extraction import collect_activations, cohens_d_activations
from metrics import evaluate_r1, evaluate_r2


# ---------------------------------------------------------------------------
# Experiment Parameters
# ---------------------------------------------------------------------------
CONCEPTS = cfg.CONCEPTS
N_LAYERS = 12  # GPT-2-small has 12 layers (0-11)
ALL_LAYERS = list(range(N_LAYERS))

# Intervention modes
MODES = {
    "single_L3": [3],
    "multi_early": [3, 4, 5],
    "multi_all": list(range(N_LAYERS)),
}

ALPHAS = [1.0, 4.0]
CENTERING_OPTIONS = [True, False]

STEERING_DIR = "cavs_steering"
STEERING_MEANDIFF_DIR = "cavs_steering_meandiff"
RESULTS_DIR = "results_multilayer"
DATA_FILE = cfg.DATA_FILE
MODEL_NAME = cfg.MODEL_NAME


# ---------------------------------------------------------------------------
# Phase 0: Extract SVM CAVs (un-normalized) for all layers
# ---------------------------------------------------------------------------
def extract_svm_raw(pos_acts, neg_acts):
    """
    Train LinearSVC and return the raw coefficient vector (NOT normalized).

    This preserves the natural scale of the SVM decision boundary normal,
    which encodes the magnitude of the concept's separability.

    Also returns cross-validation metrics for validation.
    """
    X = torch.cat([pos_acts, neg_acts], dim=0).numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    skf = StratifiedKFold(
        n_splits=cfg.SVM_CV_FOLDS, shuffle=True,
        random_state=cfg.SVM_RANDOM_STATE,
    )

    # Hyperparameter tuning
    best_C = cfg.SVM_C
    if getattr(cfg, "SVM_TUNE_C", False):
        candidates = getattr(cfg, "SVM_C_CANDIDATES", [0.01, 0.1, 1.0, 10.0])
        best_score = -1
        for c_val in candidates:
            scores = cross_val_score(
                LinearSVC(C=c_val, max_iter=cfg.SVM_MAX_ITER),
                X, y, cv=skf, scoring="accuracy",
            )
            if float(scores.mean()) > best_score:
                best_score = float(scores.mean())
                best_C = c_val

    # Train final SVM
    clf = LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER)
    clf.fit(X, y)

    # Convergence check
    converged = True
    if hasattr(clf, "n_iter_") and clf.n_iter_ >= cfg.SVM_MAX_ITER:
        warnings.warn(f"SVM did not converge (max_iter={cfg.SVM_MAX_ITER})")
        converged = False

    # RAW coefficient vector — NOT normalized
    cav_raw = torch.tensor(clf.coef_[0], dtype=torch.float32)
    raw_norm = float(cav_raw.norm())

    # Cross-validation accuracy
    cv_scores = cross_val_score(
        LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER),
        X, y, cv=skf, scoring="accuracy",
    )

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "train_accuracy": float(clf.score(X, y)),
        "C_used": best_C,
        "converged": converged,
        "raw_norm": raw_norm,
    }

    return cav_raw, metrics


def extract_steering_vectors(model, data):
    """
    Extract un-normalized SVM CAVs for all concepts and layers.

    For each concept and layer:
    1. Collect pos/neg activations
    2. Train SVM → raw coef (NOT normalized)
    3. Compute centered version: center activations, retrain SVM
    4. Save: raw CAV, centered CAV, global mean
    5. Report: CV accuracy, Cohen's d, norms

    Returns extraction report with validation metrics.
    """
    os.makedirs(STEERING_DIR, exist_ok=True)
    report = {}

    for concept in CONCEPTS:
        pos_sentences = data["cav_training"][concept]["positive"]
        neg_sentences = data["cav_training"][concept]["negative"]

        print(f"\n  Concept: {concept} "
              f"(pos={len(pos_sentences)}, neg={len(neg_sentences)})")

        for layer in ALL_LAYERS:
            steer_path = os.path.join(
                STEERING_DIR, f"{concept}_steer_layer{layer}.pt"
            )
            centered_path = os.path.join(
                STEERING_DIR, f"{concept}_steer_centered_layer{layer}.pt"
            )
            mean_path = os.path.join(
                STEERING_DIR, f"{concept}_mean_layer{layer}.pt"
            )

            # Skip if all files exist
            if (os.path.exists(steer_path) and
                    os.path.exists(centered_path) and
                    os.path.exists(mean_path)):
                print(f"    SKIP layer {layer}: files exist")
                sv = torch.load(steer_path)
                sv_c = torch.load(centered_path)
                report[f"{concept}_layer{layer}"] = {
                    "svm_raw_norm": float(sv.norm()),
                    "svm_centered_norm": float(sv_c.norm()),
                    "skipped": True,
                }
                continue

            print(f"    Layer {layer}: collecting activations...")
            pos_acts = collect_activations(model, pos_sentences, layer)
            neg_acts = collect_activations(model, neg_sentences, layer)

            # --- Raw SVM CAV (un-normalized) ---
            print(f"      Training SVM (raw)...")
            cav_raw, metrics_raw = extract_svm_raw(pos_acts, neg_acts)

            # --- Global mean for centering ---
            all_acts = torch.cat([pos_acts, neg_acts], dim=0)
            global_mean = all_acts.mean(dim=0)

            # --- Centered SVM CAV ---
            print(f"      Training SVM (centered)...")
            pos_centered = pos_acts - global_mean
            neg_centered = neg_acts - global_mean
            cav_centered, metrics_centered = extract_svm_raw(
                pos_centered, neg_centered
            )

            # --- Cohen's d (using raw CAV direction) ---
            cav_unit = cav_raw / cav_raw.norm()
            d_result = cohens_d_activations(pos_acts, neg_acts, cav_unit)

            # Compute activation norms for reference
            mean_act_norm = float(all_acts.norm(dim=1).mean())

            # Save
            torch.save(cav_raw, steer_path)
            torch.save(cav_centered, centered_path)
            torch.save(global_mean, mean_path)

            report[f"{concept}_layer{layer}"] = {
                "n_pos": len(pos_acts),
                "n_neg": len(neg_acts),
                "svm_raw_norm": float(cav_raw.norm()),
                "svm_centered_norm": float(cav_centered.norm()),
                "mean_act_norm": mean_act_norm,
                "ratio_raw_to_act": float(cav_raw.norm()) / mean_act_norm,
                "cv_accuracy_raw": metrics_raw["cv_accuracy_mean"],
                "cv_accuracy_centered": metrics_centered["cv_accuracy_mean"],
                "cv_std_raw": metrics_raw["cv_accuracy_std"],
                "cohens_d": d_result["cohens_d"],
                "effect_size": d_result["interpretation"],
                "C_used": metrics_raw["C_used"],
                "converged": metrics_raw["converged"],
            }

            print(f"      ||svm_raw||={float(cav_raw.norm()):.2f}, "
                  f"||svm_centered||={float(cav_centered.norm()):.2f}, "
                  f"||act||={mean_act_norm:.2f}")
            print(f"      CV acc: raw={metrics_raw['cv_accuracy_mean']:.3f}, "
                  f"centered={metrics_centered['cv_accuracy_mean']:.3f}, "
                  f"Cohen's d={d_result['cohens_d']:.2f} "
                  f"({d_result['interpretation']})")

    return report


# ---------------------------------------------------------------------------
# Phase 0b: Extract Mean-Diff CAVs (un-normalized) for all layers
# ---------------------------------------------------------------------------
def extract_meandiff_raw(pos_acts, neg_acts):
    """
    Mean difference vector (NOT normalized) for feature steering.

    Analogous to extract_svm_raw but uses centroid difference instead of
    SVM decision boundary normal. Preserves natural scale.
    """
    cav_raw = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    raw_norm = float(cav_raw.norm())
    return cav_raw, {"raw_norm": raw_norm}


def extract_meandiff_steering_vectors(model, data):
    """
    Extract un-normalized mean-diff vectors for all concepts and layers.

    Saves to STEERING_MEANDIFF_DIR for use by experiment_sweep.py.
    """
    os.makedirs(STEERING_MEANDIFF_DIR, exist_ok=True)
    report = {}

    for concept in CONCEPTS:
        pos_sentences = data["cav_training"][concept]["positive"]
        neg_sentences = data["cav_training"][concept]["negative"]

        print(f"\n  Concept: {concept} "
              f"(pos={len(pos_sentences)}, neg={len(neg_sentences)})")

        for layer in ALL_LAYERS:
            steer_path = os.path.join(
                STEERING_MEANDIFF_DIR, f"{concept}_steer_layer{layer}.pt"
            )

            if os.path.exists(steer_path):
                sv = torch.load(steer_path, weights_only=True)
                print(f"    SKIP layer {layer}: file exists")
                report[f"{concept}_layer{layer}"] = {
                    "meandiff_raw_norm": float(sv.norm()),
                    "skipped": True,
                }
                continue

            print(f"    Layer {layer}: collecting activations...")
            pos_acts = collect_activations(model, pos_sentences, layer)
            neg_acts = collect_activations(model, neg_sentences, layer)

            cav_raw, metrics = extract_meandiff_raw(pos_acts, neg_acts)

            # Compute activation norms for reference
            all_acts = torch.cat([pos_acts, neg_acts], dim=0)
            mean_act_norm = float(all_acts.norm(dim=1).mean())

            torch.save(cav_raw, steer_path)

            report[f"{concept}_layer{layer}"] = {
                "n_pos": len(pos_acts),
                "n_neg": len(neg_acts),
                "meandiff_raw_norm": metrics["raw_norm"],
                "mean_act_norm": mean_act_norm,
                "ratio_raw_to_act": metrics["raw_norm"] / mean_act_norm,
            }

            print(f"      ||meandiff||={metrics['raw_norm']:.2f}, "
                  f"||act||={mean_act_norm:.2f}")

    return report


def load_steering_vec(concept, layer, centered=False):
    """Load a steering vector from disk."""
    if centered:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_centered_layer{layer}.pt"
        )
    else:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_layer{layer}.pt"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering vector not found: {path}")
    return torch.load(path)


def load_global_mean(concept, layer):
    """Load the global mean vector for centering."""
    path = os.path.join(
        STEERING_DIR, f"{concept}_mean_layer{layer}.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global mean not found: {path}")
    return torch.load(path)


# ---------------------------------------------------------------------------
# Ablation Hook (Feature Steering Negativo)
# ---------------------------------------------------------------------------
def ablation_steering_hook(
    resid_post, hook, steering_vec, global_mean, alpha, use_centering,
    direction=-1
):
    """
    Concept ablation via negative feature steering.

    Two modes of intervention:

    Raw (use_centering=False):
        act' = act + direction * α * steering_vec
        Direct addition/subtraction of the un-normalized steering vector.
        The vector's natural SVM norm encodes concept separability.

    Centered projection (use_centering=True):
        centered = act - global_mean
        proj = (centered · sv_unit) * sv_unit   # concept component
        centered' = centered + direction * α * proj
        act' = centered' + global_mean
        Surgical: only removes/amplifies the concept component in centered
        activation space. More precise than raw steering.

    Args:
        resid_post: Residual stream activations [batch, seq, d_model]
        hook: TransformerLens hook object
        steering_vec: Un-normalized steering vector [d_model]
        global_mean: Global mean of activations [d_model]
        alpha: Scaling factor (1.0 = 1x the concept component magnitude)
        use_centering: Whether to use centered projection (True) or raw steering (False)
        direction: -1 for ablation (subtract), +1 for amplification (add)
    """
    if use_centering:
        # RepE-style: project and operate in centered space
        centered = resid_post - global_mean
        # Normalize steering vec to get concept direction
        sv_unit = steering_vec / steering_vec.norm()
        # Project centered activations onto concept direction
        dot = torch.einsum("bsd,d->bs", centered, sv_unit)
        proj = torch.einsum("bs,d->bsd", dot, sv_unit)
        # Remove (direction=-1) or amplify (direction=+1) the concept component
        centered = centered + direction * alpha * proj
        return centered + global_mean
    else:
        # Direct steering: add/subtract raw vector (preserves SVM norm)
        return resid_post + direction * alpha * steering_vec


# ---------------------------------------------------------------------------
# Concept Removal Measurement
# ---------------------------------------------------------------------------
def measure_concept_removal_multilayer(model, concept, layers, alpha, centered,
                                       direction, probe_text=None):
    """
    Measure how much concept signal was removed by the multi-layer intervention.

    For each intervened layer, computes dot product of the residual stream
    with the steering vector's unit direction before and after ablation.

    Returns:
        dict with per-layer removal ratios and aggregate removal ratio.
    """
    if probe_text is None:
        probe_texts = getattr(cfg, "CONCEPT_PROBE_TEXTS", {})
        probe_text = probe_texts.get(concept, "The time was running out.")
    device = model.cfg.device
    token_method = getattr(cfg, "TOKEN_POSITION", "mean")

    # Collect pre-intervention activations
    pre_dots = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        sv = load_steering_vec(concept, layer, centered=centered).to(device)
        sv_unit = (sv / sv.norm()).cpu().numpy()

        with torch.no_grad():
            _, cache = model.run_with_cache(probe_text, names_filter=[hook_name])
            if token_method == "mean":
                act = cache[hook_name][0].mean(dim=0).cpu().numpy()
            else:
                act = cache[hook_name][0, -1, :].cpu().numpy()
        pre_dots[layer] = float(act @ sv_unit)

    # Collect post-intervention activations
    fwd_hooks = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        sv = load_steering_vec(concept, layer, centered=centered).to(device)
        gm = load_global_mean(concept, layer).to(device)
        hook_fn = partial(
            ablation_steering_hook,
            steering_vec=sv, global_mean=gm,
            alpha=alpha, use_centering=centered, direction=direction,
        )
        fwd_hooks.append((hook_name, hook_fn))

    post_dots = {}
    with model.hooks(fwd_hooks=fwd_hooks):
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            sv = load_steering_vec(concept, layer, centered=centered).to(device)
            sv_unit = (sv / sv.norm()).cpu().numpy()

            with torch.no_grad():
                _, cache = model.run_with_cache(probe_text, names_filter=[hook_name])
                if token_method == "mean":
                    act = cache[hook_name][0].mean(dim=0).cpu().numpy()
                else:
                    act = cache[hook_name][0, -1, :].cpu().numpy()
            post_dots[layer] = float(act @ sv_unit)

    # Compute removal ratios
    removals = {}
    for layer in layers:
        if abs(pre_dots[layer]) > 1e-8:
            ratio = 1.0 - (post_dots[layer] / pre_dots[layer])
        else:
            ratio = 0.0
        removals[layer] = round(ratio, 4)

    # Aggregate: mean removal across intervened layers
    mean_removal = sum(removals.values()) / len(removals) if removals else 0.0

    return {
        "per_layer": removals,
        "mean_removal": round(mean_removal, 4),
        "warning": mean_removal < 0.5,
    }


# ---------------------------------------------------------------------------
# Run One Condition
# ---------------------------------------------------------------------------
def run_condition(model, concept, layers, alpha, centered, direction,
                  r1_data, r2_data):
    """
    Run one experimental condition with hooks on specified layers.

    Returns R1 accuracy, R2 similarity.
    """
    device = model.cfg.device

    # Build list of hooks
    fwd_hooks = []
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
        fwd_hooks.append((hook_name, hook_fn))

    # Evaluate with hooks active
    with model.hooks(fwd_hooks=fwd_hooks):
        r1_acc, r1_items = evaluate_r1(model, r1_data)
        r2_sim, r2_items = evaluate_r2(model, r2_data)

    return r1_acc, r2_sim


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Multi-Layer Ablation — SVM CAVs (un-normalized)")
    print("=" * 60)
    print(f"  Concepts:  {CONCEPTS}")
    print(f"  Layers:    0-11 ({N_LAYERS} total)")
    print(f"  Modes:     {list(MODES.keys())}")
    print(f"  Alphas:    {ALPHAS}")
    print(f"  Centering: {CENTERING_OPTIONS}")
    print(f"  Vector:    SVM coef (raw, NOT unit-normalized)")
    print(f"  Ablation conditions: 36 + 3 amplification controls = 39")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    # =================================================================
    # PHASE 0: EXTRACT STEERING VECTORS
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 0: SVM CAV EXTRACTION (un-normalized)")
    print(f"{'='*50}")

    extraction_report = extract_steering_vectors(model, data)

    report_path = os.path.join(RESULTS_DIR, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(extraction_report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    # Print norm summary
    print(f"\n  SVM CAV norms (raw / centered / act_mean / CV accuracy):")
    for concept in CONCEPTS:
        for layer in [0, 3, 5, 8, 11]:
            key = f"{concept}_layer{layer}"
            if key in extraction_report:
                r = extraction_report[key]
                cv = r.get('cv_accuracy_raw', 'N/A')
                cv_str = f"{cv:.3f}" if isinstance(cv, float) else cv
                print(f"    {concept}/L{layer}: "
                      f"||svm||={r['svm_raw_norm']:.2f}, "
                      f"||svm_c||={r['svm_centered_norm']:.2f}, "
                      f"||act||={r.get('mean_act_norm', 'N/A')}, "
                      f"CV={cv_str}")

    # =================================================================
    # PHASE 0b: EXTRACT MEAN-DIFF STEERING VECTORS
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 0b: MEAN-DIFF CAV EXTRACTION (un-normalized)")
    print(f"{'='*50}")

    meandiff_report = extract_meandiff_steering_vectors(model, data)

    meandiff_report_path = os.path.join(RESULTS_DIR, "extraction_report_meandiff.json")
    with open(meandiff_report_path, "w") as f:
        json.dump(meandiff_report, f, indent=2)
    print(f"\n  Saved: {meandiff_report_path}")

    # =================================================================
    # PHASE 1: BASELINE
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 1: BASELINE (no intervention)")
    print(f"{'='*50}")

    baselines = {}
    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]

        r1_acc, _ = evaluate_r1(model, r1_data)
        r2_sim, _ = evaluate_r2(model, r2_data)

        baselines[concept] = {
            "r1_accuracy": r1_acc,
            "r2_similarity": r2_sim,
        }
        print(f"  {concept.upper()}: R1={r1_acc:.3f}, R2={r2_sim:.4f}")

    baseline_path = os.path.join(RESULTS_DIR, "results_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Saved: {baseline_path}")

    # =================================================================
    # PHASE 2: FACTORIAL ABLATION (36 conditions)
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 2: ABLATION EXPERIMENT (feature steering negativo)")
    print(f"{'='*50}")

    results_rows = []
    condition_num = 0
    total_ablation = len(CONCEPTS) * len(MODES) * len(ALPHAS) * len(CENTERING_OPTIONS)
    total_all = total_ablation + len(CONCEPTS)  # + amplification controls
    print(f"  Total conditions: {total_all} "
          f"({total_ablation} ablation + {len(CONCEPTS)} amplification)")

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]
        r1_base = baselines[concept]["r1_accuracy"]
        r2_base = baselines[concept]["r2_similarity"]

        for mode_name, layers in MODES.items():
            for alpha in ALPHAS:
                for centered in CENTERING_OPTIONS:
                    condition_num += 1
                    centering_label = "centered" if centered else "raw"
                    label = (f"[{condition_num}/{total_all}] "
                             f"{concept}/{mode_name}/α={alpha}/"
                             f"{centering_label}/ABLATE")
                    print(f"  {label}")

                    r1_acc, r2_sim = run_condition(
                        model, concept, layers, alpha, centered,
                        direction=-1,  # ablation (subtract)
                        r1_data=r1_data, r2_data=r2_data,
                    )

                    delta_r1 = r1_base - r1_acc
                    delta_r2 = r2_base - r2_sim

                    # Measure concept removal effectiveness
                    removal = measure_concept_removal_multilayer(
                        model, concept, layers, alpha, centered,
                        direction=-1,
                    )
                    warn_tag = " [WEAK REMOVAL]" if removal["warning"] else ""

                    results_rows.append({
                        "concept": concept,
                        "mode": mode_name,
                        "n_layers": len(layers),
                        "alpha": alpha,
                        "centering": centering_label,
                        "direction": "ablation",
                        "r1_accuracy": r1_acc,
                        "r2_similarity": r2_sim,
                        "r1_baseline": r1_base,
                        "r2_baseline": r2_base,
                        "delta_r1": delta_r1,
                        "delta_r2": delta_r2,
                        "mean_removal_ratio": removal["mean_removal"],
                    })

                    print(f"    R1={r1_acc:.3f} (Δ={delta_r1:+.3f}), "
                          f"R2={r2_sim:.4f} (Δ={delta_r2:+.4f}), "
                          f"removal={removal['mean_removal']:.2f}"
                          f"{warn_tag}")

    # =================================================================
    # PHASE 3: AMPLIFICATION CONTROLS (3 conditions)
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: AMPLIFICATION CONTROLS (+α, validate direction)")
    print(f"{'='*50}")

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]
        r1_base = baselines[concept]["r1_accuracy"]
        r2_base = baselines[concept]["r2_similarity"]

        condition_num += 1
        label = (f"[{condition_num}/{total_all}] "
                 f"{concept}/multi_all/α=4.0/centered/AMPLIFY")
        print(f"  {label}")

        r1_acc, r2_sim = run_condition(
            model, concept, MODES["multi_all"], 4.0, True,
            direction=+1,  # amplification (add)
            r1_data=r1_data, r2_data=r2_data,
        )

        delta_r1 = r1_base - r1_acc
        delta_r2 = r2_base - r2_sim

        results_rows.append({
            "concept": concept,
            "mode": "multi_all",
            "n_layers": N_LAYERS,
            "alpha": 4.0,
            "centering": "centered",
            "direction": "amplification",
            "r1_accuracy": r1_acc,
            "r2_similarity": r2_sim,
            "r1_baseline": r1_base,
            "r2_baseline": r2_base,
            "delta_r1": delta_r1,
            "delta_r2": delta_r2,
        })

        print(f"    R1={r1_acc:.3f} (Δ={delta_r1:+.3f}), "
              f"R2={r2_sim:.4f} (Δ={delta_r2:+.4f})")
        if delta_r1 < 0:
            print(f"    ✓ R1 INCREASED → vector direction validated")
        else:
            print(f"    ✗ R1 did not increase → check vector direction")

    # Save all results
    df = pd.DataFrame(results_rows)
    factorial_path = os.path.join(RESULTS_DIR, "results_factorial.csv")
    df.to_csv(factorial_path, index=False)
    print(f"\n  Saved: {factorial_path} ({len(df)} rows)")

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE — SUMMARY")
    print(f"{'='*60}")

    # Filter ablation only for summary statistics
    df_abl = df[df["direction"] == "ablation"]
    df_amp = df[df["direction"] == "amplification"]

    # --- Effect of mode (single vs multi) ---
    print(f"\n  Effect of intervention mode (mean ΔR1 across all conditions):")
    for mode_name in MODES:
        mode_df = df_abl[df_abl["mode"] == mode_name]
        mean_dr1 = mode_df["delta_r1"].mean()
        mean_dr2 = mode_df["delta_r2"].mean()
        print(f"    {mode_name:15s}: ΔR1={mean_dr1:+.4f}, ΔR2={mean_dr2:+.4f}")

    # --- Effect of centering ---
    print(f"\n  Effect of centering (mean ΔR1):")
    for c_label in ["centered", "raw"]:
        c_df = df_abl[df_abl["centering"] == c_label]
        mean_dr1 = c_df["delta_r1"].mean()
        mean_dr2 = c_df["delta_r2"].mean()
        print(f"    {c_label:15s}: ΔR1={mean_dr1:+.4f}, ΔR2={mean_dr2:+.4f}")

    # --- Effect of alpha ---
    print(f"\n  Effect of alpha (mean ΔR1):")
    for alpha in ALPHAS:
        a_df = df_abl[df_abl["alpha"] == alpha]
        mean_dr1 = a_df["delta_r1"].mean()
        mean_dr2 = a_df["delta_r2"].mean()
        print(f"    α={alpha:5.1f}:         ΔR1={mean_dr1:+.4f}, ΔR2={mean_dr2:+.4f}")

    # --- Per concept summary ---
    print(f"\n  Per concept (best ablation condition):")
    for concept in CONCEPTS:
        concept_df = df_abl[df_abl["concept"] == concept]
        best_idx = concept_df["delta_r1"].idxmax()
        best = concept_df.loc[best_idx]
        print(f"    {concept.upper():8s}: ΔR1={best['delta_r1']:+.4f} "
              f"(mode={best['mode']}, α={best['alpha']}, "
              f"{best['centering']})")

    # --- Amplification controls ---
    print(f"\n  Amplification controls (direction validation):")
    for _, row in df_amp.iterrows():
        status = "✓ VALID" if row["delta_r1"] < 0 else "✗ INVALID"
        print(f"    {row['concept'].upper():8s}: ΔR1={row['delta_r1']:+.4f} "
              f"[{status}]")

    # --- Best overall vs previous experiments ---
    best_overall = df_abl.loc[df_abl["delta_r1"].idxmax()]
    print(f"\n  BEST ABLATION CONDITION:")
    print(f"    {best_overall['concept']}/{best_overall['mode']}/"
          f"α={best_overall['alpha']}/{best_overall['centering']}")
    print(f"    ΔR1={best_overall['delta_r1']:+.4f}, "
          f"ΔR2={best_overall['delta_r2']:+.4f}")
    print(f"    (Previous best was ~0.07 with single-layer normalized CAVs)")

    print(f"\n  Output files:")
    print(f"    {factorial_path} ({len(df)} rows)")
    print(f"    {baseline_path}")
    print(f"    {report_path}")


if __name__ == "__main__":
    main()
