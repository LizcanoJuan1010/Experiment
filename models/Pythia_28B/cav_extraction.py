"""
CAV Extraction (Dual Method)
============================
Extracts Concept Activation Vectors using two methods:
  1. Mean Difference: CAV = normalize(mean_pos - mean_neg)
  2. SVM: CAV = normal vector of LinearSVC decision boundary

Extracts at multiple layers for hypothesis H1B (middle vs late layers).

Usage (inside Docker):
    python cav_extraction.py

Input:  experiment_data.json
Output: cavs/{concept}_{method}_layer{L}.pt  (one file per CAV)
        cavs/extraction_report.json           (quality metrics)
"""

import torch
import numpy as np
import json
import os
import warnings
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import stats
from transformer_lens import HookedTransformer

import experiment_config as cfg

# ---------------------------------------------------------------------------
# Configuration (from experiment_config.py)
# ---------------------------------------------------------------------------
MODEL_NAME = cfg.MODEL_NAME
DATA_FILE = cfg.DATA_FILE
OUTPUT_DIR = cfg.CAV_DIR
LAYERS = cfg.EXTRACTION_LAYERS
CONCEPTS = cfg.CONCEPTS
METHODS = cfg.CAV_METHODS


# ---------------------------------------------------------------------------
# Activation Collection
# ---------------------------------------------------------------------------
def collect_activations(model, sentences, layer, concept_words=None):
    """
    Collect per-sentence activations for a given layer.

    Token position strategy controlled by cfg.TOKEN_POSITION:
      - "mean":         Mean pooling over all tokens (default, recommended).
                        Avoids last-token positional bias.
      - "last":         Last token only (original GPT-2 convention).
      - "concept_word": Activation at the concept word's token position
                        (requires concept_words list parallel to sentences).

    Returns a tensor of shape (N, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []
    token_method = getattr(cfg, "TOKEN_POSITION", "mean")

    with torch.no_grad():
        for i, sent in enumerate(tqdm(sentences, desc=f"  Layer {layer}", leave=False)):
            _, cache = model.run_with_cache(sent, names_filter=[hook_name])
            act = cache[hook_name]  # [1, seq_len, d_model]
            assert act.ndim == 3, f"Expected 3D tensor, got shape {act.shape}"

            if token_method == "mean":
                # Mean pooling over all tokens (Reimers & Gurevych, 2019)
                pooled = act[0].mean(dim=0)  # [d_model]
            elif token_method == "concept_word" and concept_words is not None:
                # Find token position of concept word
                word = concept_words[i]
                tokens = model.to_tokens(sent)[0]
                token_strs = [model.to_string(t) for t in tokens]
                word_lower = word.lower().strip()
                pos = None
                for j, ts in enumerate(token_strs):
                    if word_lower in ts.lower().strip():
                        pos = j
                        break
                if pos is not None:
                    pooled = act[0, pos, :]
                else:
                    # Fallback to mean if concept word not found
                    pooled = act[0].mean(dim=0)
            else:
                # "last" or fallback
                pooled = act[0, -1, :]

            activations.append(pooled.cpu())

    return torch.stack(activations)  # [N, d_model]


# ---------------------------------------------------------------------------
# CAV Extraction Methods
# ---------------------------------------------------------------------------
def extract_cav_mean_diff(pos_acts, neg_acts):
    """
    Method 1: Mean Difference.
    CAV = normalize(mean(pos) - mean(neg))
    """
    # Compute in float32 for numerical stability (float16 activations
    # can produce norms that deviate from 1.0 after division).
    mean_pos = pos_acts.float().mean(dim=0)
    mean_neg = neg_acts.float().mean(dim=0)
    cav = mean_pos - mean_neg
    norm = cav.norm()
    assert norm > 1e-8, "Mean difference vector has near-zero norm"
    cav = cav / norm
    assert abs(float(cav.norm()) - 1.0) < 1e-5, "CAV normalization failed"
    return cav, {}


def extract_cav_svm(pos_acts, neg_acts):
    """
    Method 2: Linear SVM (original TCAV, Kim et al. 2018).
    CAV = normal vector to the SVM decision boundary.

    Improvements over naive implementation:
      - Optional hyperparameter tuning (grid search over C)
      - Convergence verification
      - Cross-validation accuracy as quality metric
    """
    X = torch.cat([pos_acts, neg_acts], dim=0).float().numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    skf = StratifiedKFold(
        n_splits=cfg.SVM_CV_FOLDS, shuffle=True,
        random_state=cfg.SVM_RANDOM_STATE,
    )

    # Hyperparameter tuning if enabled
    best_C = cfg.SVM_C
    tuning_info = {}
    if getattr(cfg, "SVM_TUNE_C", False):
        candidates = getattr(cfg, "SVM_C_CANDIDATES", [0.01, 0.1, 1.0, 10.0])
        best_score = -1
        c_scores = {}
        for c_val in candidates:
            scores = cross_val_score(
                LinearSVC(C=c_val, max_iter=cfg.SVM_MAX_ITER),
                X, y, cv=skf, scoring="accuracy",
            )
            mean_score = float(scores.mean())
            c_scores[str(c_val)] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_C = c_val
        tuning_info = {"best_C": best_C, "c_scores": c_scores}

    # Train final SVM with best C
    clf = LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER)
    clf.fit(X, y)

    # Convergence check
    converged = True
    if hasattr(clf, "n_iter_"):
        if clf.n_iter_ >= cfg.SVM_MAX_ITER:
            warnings.warn(
                f"SVM did not converge (n_iter={clf.n_iter_} >= max_iter={cfg.SVM_MAX_ITER}). "
                f"Consider increasing SVM_MAX_ITER."
            )
            converged = False

    # CAV = coefficient vector (normal to decision boundary)
    cav_np = clf.coef_[0]
    norm = np.linalg.norm(cav_np)
    assert norm > 1e-8, "SVM coefficient vector has near-zero norm"
    cav_np = cav_np / norm
    cav = torch.tensor(cav_np, dtype=torch.float32)

    # Verify unit norm
    assert abs(float(cav.norm()) - 1.0) < 1e-5, "CAV normalization failed"

    # Cross-validation accuracy (quality metric) with best C
    cv_scores = cross_val_score(
        LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER),
        X, y, cv=skf, scoring="accuracy",
    )

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_scores": [float(s) for s in cv_scores],
        "train_accuracy": float(clf.score(X, y)),
        "C_used": best_C,
        "converged": converged,
        **tuning_info,
    }

    return cav, metrics


def cohens_d_activations(pos_acts, neg_acts, cav):
    """
    Effect size for positive vs negative activations projected onto CAV.

    Computes:
      - Cohen's d (pooled std) or Hedges' g (bias-corrected for small N)
      - Levene's test for homogeneity of variances
      - Independent t-test (or Welch's if variances unequal)
      - 95% CI for the effect size

    Interpretation (Cohen, 1988):
      |d| < 0.2: negligible | 0.2-0.5: small | 0.5-0.8: medium | >= 0.8: large
    """
    cav_np = cav.float().numpy() if isinstance(cav, torch.Tensor) else cav
    pos_np = pos_acts.float().numpy() if isinstance(pos_acts, torch.Tensor) else pos_acts
    neg_np = neg_acts.float().numpy() if isinstance(neg_acts, torch.Tensor) else neg_acts

    pos_proj = pos_np @ cav_np
    neg_proj = neg_np @ cav_np

    mean_pos = float(pos_proj.mean())
    mean_neg = float(neg_proj.mean())
    n1, n2 = len(pos_proj), len(neg_proj)

    var1 = float(np.var(pos_proj, ddof=1))
    var2 = float(np.var(neg_proj, ddof=1))
    pooled_std = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))

    d = (mean_pos - mean_neg) / pooled_std if pooled_std > 0 else 0.0

    # Hedges' g: bias correction for small samples (Hedges, 1981)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = d * correction

    # Levene's test for equal variances
    levene_stat, levene_p = stats.levene(pos_proj, neg_proj)
    equal_var = levene_p > 0.05

    # t-test (Welch's if variances unequal)
    t_stat, t_p = stats.ttest_ind(pos_proj, neg_proj, equal_var=equal_var)

    # 95% CI for Cohen's d (Hedges & Olkin, 1985)
    se_d = float(np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2))))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d

    abs_d = abs(d)
    if abs_d >= 0.8:
        interpretation = "large"
    elif abs_d >= 0.5:
        interpretation = "medium"
    elif abs_d >= 0.2:
        interpretation = "small"
    else:
        interpretation = "negligible"

    return {
        "cohens_d": round(d, 4),
        "hedges_g": round(hedges_g, 4),
        "mean_pos_projection": round(mean_pos, 4),
        "mean_neg_projection": round(mean_neg, 4),
        "pooled_std": round(pooled_std, 4),
        "t_statistic": round(float(t_stat), 4),
        "t_p_value": round(float(t_p), 6),
        "levene_p": round(float(levene_p), 4),
        "equal_variances": equal_var,
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Layer Diagnostic Probing (FIX-A3)
# ---------------------------------------------------------------------------
def run_layer_diagnostic(model, pos_sentences, neg_sentences, n_layers=None):
    """
    Probe all layers to empirically identify which encode concept information.

    Trains a LogisticRegression classifier at each layer and ranks them
    by cross-validated accuracy. This replaces arbitrary layer selection
    with data-driven evidence.

    Returns:
        dict with per-layer accuracy and ranking
    """
    if n_layers is None:
        n_layers = getattr(cfg, "N_LAYERS_TOTAL", 12)

    # Use a subset to keep this fast
    max_per_class = 50
    pos_sub = pos_sentences[:max_per_class]
    neg_sub = neg_sentences[:max_per_class]

    layer_scores = {}
    for layer in range(n_layers):
        pos_acts = collect_activations(model, pos_sub, layer)
        neg_acts = collect_activations(model, neg_sub, layer)

        X = torch.cat([pos_acts, neg_acts], dim=0).float().numpy()
        y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SVM_RANDOM_STATE)
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, solver="lbfgs"),
            X, y, cv=skf, scoring="accuracy",
        )
        layer_scores[layer] = float(scores.mean())

    # Rank layers by accuracy (descending)
    ranking = sorted(layer_scores.items(), key=lambda x: -x[1])
    top_layers = [layer for layer, _ in ranking[:3]]

    return {
        "layer_accuracies": {str(k): round(v, 4) for k, v in layer_scores.items()},
        "ranking": [{"layer": l, "accuracy": round(s, 4)} for l, s in ranking],
        "top_3_layers": top_layers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CAV Extraction (Dual Method, Multiple Layers)")
    print("=" * 60)

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    model = cfg.load_model()
    d_model = model.cfg.d_model
    print(f"  d_model = {d_model}")

    report = {
        "model": MODEL_NAME,
        "d_model": d_model,
        "layers": LAYERS,
        "concepts": CONCEPTS,
        "methods": METHODS,
        "results": {},
    }

    # --- Layer Diagnostic (if enabled) ---
    if getattr(cfg, "AUTO_LAYER_SELECTION", False):
        print(f"\n{'='*40}")
        print("LAYER DIAGNOSTIC PROBING")
        print(f"{'='*40}")
        # Use first concept for diagnostic
        first_concept = CONCEPTS[0]
        diag_pos = data["cav_training"][first_concept]["positive"]
        diag_neg = data["cav_training"][first_concept]["negative"]
        print(f"  Running diagnostic with concept '{first_concept}'...")
        layer_diag = run_layer_diagnostic(model, diag_pos, diag_neg)
        report["layer_diagnostic"] = layer_diag
        print(f"  Top 3 layers: {layer_diag['top_3_layers']}")
        for entry in layer_diag["ranking"]:
            marker = " <--" if entry["layer"] in LAYERS else ""
            print(f"    Layer {entry['layer']:2d}: {entry['accuracy']:.3f}{marker}")

    # --- Extract CAVs ---
    for concept in CONCEPTS:
        # Skip logic: check if ALL files for this concept already exist
        all_exist = True
        for layer in LAYERS:
            for method in METHODS:
                fpath = os.path.join(OUTPUT_DIR, f"{concept}_{method}_layer{layer}.pt")
                if not os.path.exists(fpath):
                    all_exist = False
                    break
            if not all_exist:
                break
        if all_exist:
            print(f"\n  SKIP: {concept.upper()} — all CAV files already exist")
            continue

        print(f"\n{'='*40}")
        print(f"CONCEPT: {concept.upper()}")
        print(f"{'='*40}")

        pos_sentences = data["cav_training"][concept]["positive"]
        neg_sentences = data["cav_training"][concept]["negative"]
        print(f"  Positive: {len(pos_sentences)}, Negative: {len(neg_sentences)}")

        report["results"][concept] = {}

        for layer in LAYERS:
            print(f"\n  --- Layer {layer} ---")

            # Collect activations
            print(f"  Collecting positive activations...")
            pos_acts = collect_activations(model, pos_sentences, layer)
            print(f"  Collecting negative activations...")
            neg_acts = collect_activations(model, neg_sentences, layer)

            report["results"][concept][f"layer_{layer}"] = {
                "n_pos": len(pos_acts),
                "n_neg": len(neg_acts),
            }

            # Save activations for downstream statistical tests
            if cfg.SAVE_ACTIVATIONS:
                acts_path = os.path.join(
                    OUTPUT_DIR, f"{concept}_acts_layer{layer}.pt"
                )
                torch.save({"pos": pos_acts, "neg": neg_acts}, acts_path)
                print(f"  Saved activations: {acts_path}")

            for method in METHODS:
                print(f"  Extracting CAV: {method}...")

                if method == "mean_diff":
                    cav, metrics = extract_cav_mean_diff(pos_acts, neg_acts)
                elif method == "svm":
                    cav, metrics = extract_cav_svm(pos_acts, neg_acts)

                # Cohen's d effect size (Cohen, 1988)
                d_result = cohens_d_activations(pos_acts, neg_acts, cav)
                metrics["cohens_d"] = d_result["cohens_d"]
                metrics["effect_size"] = d_result["interpretation"]
                print(f"    Cohen's d: {d_result['cohens_d']:.3f} "
                      f"({d_result['interpretation']})")

                # Save CAV
                filename = f"{concept}_{method}_layer{layer}.pt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                torch.save(cav, filepath)
                print(f"    Saved: {filepath}")

                # Report
                report["results"][concept][f"layer_{layer}"][method] = {
                    "file": filename,
                    "cav_norm": float(cav.norm()),
                    **metrics,
                }

                if method == "svm" and metrics:
                    acc = metrics["cv_accuracy_mean"]
                    std = metrics["cv_accuracy_std"]
                    status = "OK" if acc >= cfg.SVM_CV_ACCURACY_THRESHOLD else "WARNING"
                    print(f"    SVM CV accuracy: {acc:.3f} +/- {std:.3f} [{status}]")

    # --- Cross-concept cosine similarity ---
    print(f"\n{'='*40}")
    print("CROSS-CONCEPT COSINE SIMILARITY")
    print(f"{'='*40}")

    report["cross_concept_cosine"] = {}

    for layer in LAYERS:
        print(f"\n  Layer {layer}:")
        report["cross_concept_cosine"][f"layer_{layer}"] = {}

        for method in METHODS:
            cavs = {}
            for concept in CONCEPTS:
                filepath = os.path.join(
                    OUTPUT_DIR, f"{concept}_{method}_layer{layer}.pt"
                )
                cavs[concept] = torch.load(filepath, weights_only=True)

            pairs = [(CONCEPTS[i], CONCEPTS[j])
                     for i in range(len(CONCEPTS))
                     for j in range(i + 1, len(CONCEPTS))]
            print(f"    {method}:")
            for c1, c2 in pairs:
                cos = torch.nn.functional.cosine_similarity(
                    cavs[c1].unsqueeze(0), cavs[c2].unsqueeze(0)
                ).item()
                threshold = cfg.CROSS_CONCEPT_COSINE_THRESHOLD
                status = "OK" if abs(cos) < threshold else f"WARNING (|cos| >= {threshold})"
                print(f"      {c1}-{c2}: {cos:.4f} [{status}]")

                key = f"{method}_{c1}_{c2}"
                report["cross_concept_cosine"][f"layer_{layer}"][key] = cos

    # --- Cross-method cosine similarity ---
    print(f"\n{'='*40}")
    print("CROSS-METHOD COSINE (mean_diff vs svm)")
    print(f"{'='*40}")

    report["cross_method_cosine"] = {}

    for concept in CONCEPTS:
        for layer in LAYERS:
            cav_md = torch.load(
                os.path.join(OUTPUT_DIR, f"{concept}_mean_diff_layer{layer}.pt"),
                weights_only=True,
            )
            cav_svm = torch.load(
                os.path.join(OUTPUT_DIR, f"{concept}_svm_layer{layer}.pt"),
                weights_only=True,
            )
            cos = torch.nn.functional.cosine_similarity(
                cav_md.unsqueeze(0), cav_svm.unsqueeze(0)
            ).item()
            key = f"{concept}_layer{layer}"
            report["cross_method_cosine"][key] = cos
            print(f"  {concept.upper()} layer {layer}: {cos:.4f}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
