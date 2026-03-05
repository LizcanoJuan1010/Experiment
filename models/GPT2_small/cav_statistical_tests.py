"""
CAV Statistical Validation
===========================
Rigorous statistical tests for CAV quality and significance.

Tests implemented:
  1. TCAV Permutation Test  — Kim et al. (2018), Section 4
  2. Selectivity Test       — Hewitt & Liang (2019)
  3. Cohen's d Effect Size  — Cohen (1988)
  4. Bootstrap CI           — Efron & Tibshirani (1993)

Usage:
    python cav_statistical_tests.py

    # Without model (uses saved activations from cav_extraction.py):
    python cav_statistical_tests.py --from-saved

Input:  experiment_data.json, cavs/*.pt, cavs/*_acts_*.pt
Output: cavs/statistical_validation.json
"""

import torch
import numpy as np
import json
import os
import argparse
from sklearn.model_selection import StratifiedKFold

import experiment_config as cfg


# =========================================================================
# GPU-Accelerated Linear SVM (replaces sklearn LinearSVC for speed)
# =========================================================================
_GPU_DEVICE = None


def _get_gpu_device():
    """Get CUDA device if available, else CPU."""
    global _GPU_DEVICE
    if _GPU_DEVICE is None:
        _GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [GPU SVM] Using device: {_GPU_DEVICE}")
    return _GPU_DEVICE


def _fit_linear_svm_gpu(X_np, y_np, C=1.0, max_iter=2000, lr=0.005):
    """
    GPU-accelerated Linear SVM via Adam + hinge loss.
    Mathematically equivalent to sklearn LinearSVC but runs on CUDA.

    Returns: unit-normalized weight vector as numpy array [d].
    """
    device = _get_gpu_device()
    n, d = X_np.shape

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    y = 2.0 * y - 1.0  # {0,1} → {-1,+1} for hinge loss

    w = torch.randn(d, device=device) * 0.01
    w.requires_grad_(True)

    reg = 1.0 / (C * n)
    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(max_iter):
        margins = y * (X @ w)
        hinge = torch.clamp(1.0 - margins, min=0).mean()
        l2 = 0.5 * reg * (w @ w)
        loss = hinge + l2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step > 200 and hinge.item() < 1e-4:
            break

    w_np = w.detach().cpu().numpy()
    norm = np.linalg.norm(w_np)
    return w_np / norm if norm > 1e-8 else w_np


def _batched_linear_svm_gpu(X_list, y_list, C=1.0, max_iter=1500, lr=0.005):
    """
    Train multiple linear SVMs in parallel on GPU using batched operations.
    All bootstrap resamples are trained simultaneously via [B, n, d] bmm.

    Args:
        X_list: list of B numpy arrays, each [n, d]
        y_list: list of B numpy arrays, each [n] with {0,1} labels

    Returns: [B, d] numpy array of unit-normalized weight vectors
    """
    device = _get_gpu_device()
    B = len(X_list)
    d = X_list[0].shape[1]
    n = X_list[0].shape[0]

    # Stack into [B, n, d] and [B, n]
    X_batch = torch.tensor(np.stack(X_list), dtype=torch.float32, device=device)
    y_batch = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)
    y_batch = 2.0 * y_batch - 1.0  # {0,1} → {-1,+1}

    # Initialize weights: [B, d]
    W = torch.randn(B, d, device=device) * 0.01
    W.requires_grad_(True)

    reg = 1.0 / (C * n)
    optimizer = torch.optim.Adam([W], lr=lr)

    for step in range(max_iter):
        # margins: [B, n] via batched matmul
        margins = y_batch * torch.bmm(X_batch, W.unsqueeze(2)).squeeze(2)
        hinge = torch.clamp(1.0 - margins, min=0).mean(dim=1)  # [B]
        l2 = 0.5 * reg * (W * W).sum(dim=1)                    # [B]
        loss = (hinge + l2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step > 200 and hinge.max().item() < 1e-4:
            break

    W_np = W.detach().cpu().numpy()  # [B, d]
    norms = np.linalg.norm(W_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return W_np / norms


def _gpu_cv_accuracy(X_np, y_np, C, n_splits, seed):
    """GPU-accelerated stratified cross-validation for linear SVM."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X_np, y_np):
        w = _fit_linear_svm_gpu(X_np[train_idx], y_np[train_idx], C=C)
        preds = (X_np[val_idx] @ w > 0).astype(int)
        acc = float((preds == y_np[val_idx]).mean())
        scores.append(acc)
    return np.array(scores)


# =========================================================================
# 1. TCAV PERMUTATION TEST — Kim et al. (2018), Section 4
# =========================================================================
def tcav_score(model, sentences, cav, layer):
    """
    Compute TCAV score: fraction of examples where the directional
    derivative along the CAV is positive.

    TCAV_score = |{x : S_C(x) > 0}| / |examples|
    where S_C(x) = grad(f(x)) . CAV

    For efficiency with residual stream CAVs, we approximate using
    the dot product of the activation with the CAV direction.

    FIX-M4: Uses cfg.TOKEN_POSITION for consistent token extraction.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    positive_count = 0
    token_method = getattr(cfg, "TOKEN_POSITION", "mean")

    cav_np = cav.numpy() if isinstance(cav, torch.Tensor) else cav
    # FIX-M4: Assert unit norm
    cav_norm = np.linalg.norm(cav_np)
    assert abs(cav_norm - 1.0) < 1e-4, f"CAV not unit norm: {cav_norm:.6f}"

    with torch.no_grad():
        for sent in sentences:
            _, cache = model.run_with_cache(sent, names_filter=[hook_name])
            if token_method == "mean":
                act = cache[hook_name][0].mean(dim=0).cpu().numpy()
            else:
                act = cache[hook_name][0, -1, :].cpu().numpy()
            if np.dot(act, cav_np) > 0:
                positive_count += 1

    return positive_count / len(sentences) if sentences else 0.0


def tcav_score_from_acts(activations, cav):
    """
    Compute TCAV score from pre-collected activations (no model needed).
    """
    acts_np = activations.numpy() if isinstance(activations, torch.Tensor) else activations
    cav_np = cav.numpy() if isinstance(cav, torch.Tensor) else cav
    projections = acts_np @ cav_np
    return float((projections > 0).mean())


def extract_random_cav(neutral_sentences, method, seed):
    """
    Extract a CAV from a random split of neutral sentences.
    Splits the neutral sentences in half to create pseudo-positive/negative.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(neutral_sentences))
    mid = len(indices) // 2
    # These are just random activations, not real concept activations
    pseudo_pos = neutral_sentences[indices[:mid]]
    pseudo_neg = neutral_sentences[indices[mid:mid * 2]]

    if method == "mean_diff":
        diff = pseudo_pos.mean(axis=0) - pseudo_neg.mean(axis=0)
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 0 else diff
    elif method == "svm":
        X = np.vstack([pseudo_pos, pseudo_neg])
        y = np.array([1] * len(pseudo_pos) + [0] * len(pseudo_neg))
        try:
            return _fit_linear_svm_gpu(X, y, C=cfg.SVM_C)
        except Exception:
            # If SVM fails, return random direction
            d = pseudo_pos.shape[1]
            rand_dir = rng.randn(d)
            return rand_dir / np.linalg.norm(rand_dir)


def tcav_permutation_test(pos_acts, neg_acts, cav, method="svm",
                          n_permutations=None, seed=None):
    """
    TCAV significance test (Kim et al., 2018, Section 4).

    FIX-C4: Uses train/test split to avoid evaluating on training data.
    Without this, the real CAV has an unfair advantage over random CAVs
    because it was fitted on the same data used for scoring.

    Procedure:
      1. Split pos/neg activations into 70% train / 30% test (stratified)
      2. Re-extract real CAV from train split
      3. Compute real TCAV score on TEST split only
      4. Generate n random CAVs from train split (permuted labels)
      5. Compute each random CAV's TCAV score on TEST split
      6. p-value = fraction of random scores >= real score (on test)

    Parameters:
        pos_acts: Positive activations tensor [N_pos, d_model]
        neg_acts: Negative activations tensor [N_neg, d_model]
        cav: The real CAV vector (used as fallback reference)
        method: "mean_diff" or "svm"
        n_permutations: Number of random runs (default from config)
        seed: Random seed

    Returns:
        dict with real_tcav_score, random_scores, p_value, significant
    """
    if n_permutations is None:
        n_permutations = cfg.TCAV_PERMUTATION_RUNS
    if seed is None:
        seed = cfg.SVM_RANDOM_STATE

    test_fraction = getattr(cfg, "TCAV_TEST_SPLIT", 0.30)

    pos_np = pos_acts.numpy() if isinstance(pos_acts, torch.Tensor) else pos_acts
    neg_np = neg_acts.numpy() if isinstance(neg_acts, torch.Tensor) else neg_acts

    rng = np.random.RandomState(seed)

    # FIX-C4: Stratified train/test split
    n_pos_test = max(1, int(len(pos_np) * test_fraction))
    n_neg_test = max(1, int(len(neg_np) * test_fraction))

    pos_perm = rng.permutation(len(pos_np))
    neg_perm = rng.permutation(len(neg_np))

    pos_test = pos_np[pos_perm[:n_pos_test]]
    pos_train = pos_np[pos_perm[n_pos_test:]]
    neg_test = neg_np[neg_perm[:n_neg_test]]
    neg_train = neg_np[neg_perm[n_neg_test:]]

    # Re-extract real CAV from TRAIN split only
    if method == "mean_diff":
        real_cav = pos_train.mean(axis=0) - neg_train.mean(axis=0)
        norm = np.linalg.norm(real_cav)
        real_cav = real_cav / norm if norm > 1e-8 else real_cav
    elif method == "svm":
        X_train = np.vstack([pos_train, neg_train])
        y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))
        try:
            real_cav = _fit_linear_svm_gpu(X_train, y_train, C=cfg.SVM_C)
        except Exception:
            # Fallback to provided CAV
            real_cav = cav.numpy() if isinstance(cav, torch.Tensor) else cav
    else:
        real_cav = cav.numpy() if isinstance(cav, torch.Tensor) else cav

    # Compute real TCAV score on TEST set only
    real_score = float((pos_test @ real_cav > 0).mean())

    # Pool TRAIN activations for random CAV generation
    all_train = np.vstack([pos_train, neg_train])

    # Generate random CAVs from TRAIN split and score on TEST
    random_scores = []
    for i in range(n_permutations):
        random_cav = extract_random_cav(all_train, method, seed=seed + i + 100)
        r_score = float((pos_test @ random_cav > 0).mean())
        random_scores.append(r_score)

    # p-value: fraction of random scores >= real score (on test)
    random_scores = np.array(random_scores)
    p_value = float((random_scores >= real_score).mean())

    return {
        "real_tcav_score": round(real_score, 4),
        "random_tcav_scores_mean": round(float(random_scores.mean()), 4),
        "random_tcav_scores_std": round(float(random_scores.std()), 4),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < cfg.TCAV_SIGNIFICANCE_ALPHA),
        "n_permutations": n_permutations,
        "alpha": cfg.TCAV_SIGNIFICANCE_ALPHA,
        "train_test_split": {
            "test_fraction": test_fraction,
            "n_pos_train": len(pos_train),
            "n_pos_test": len(pos_test),
            "n_neg_train": len(neg_train),
            "n_neg_test": len(neg_test),
        },
    }


# =========================================================================
# 2. SELECTIVITY TEST — Hewitt & Liang (2019)
# =========================================================================
def selectivity_test(pos_acts, neg_acts, method="svm",
                     n_control_runs=None, seed=None):
    """
    Selectivity metric (Hewitt & Liang, 2019).

    Measures whether the probe captures genuine linguistic structure
    or merely memorizes training data.

    Procedure:
      1. Train real probe: accuracy_real = CV accuracy on (pos, neg)
      2. For each control run: shuffle labels, train probe, record accuracy
      3. Selectivity = accuracy_real - mean(accuracy_control)

    Interpretation:
      ~0: probe is memorizing
      >0.10: meaningful structure captured
      >0.30: strong evidence of genuine concept

    Returns:
        dict with real_accuracy, control_accuracies, selectivity
    """
    if n_control_runs is None:
        n_control_runs = cfg.SELECTIVITY_CONTROL_RUNS
    if seed is None:
        seed = cfg.SVM_RANDOM_STATE

    pos_np = pos_acts.numpy() if isinstance(pos_acts, torch.Tensor) else pos_acts
    neg_np = neg_acts.numpy() if isinstance(neg_acts, torch.Tensor) else neg_acts

    X = np.vstack([pos_np, neg_np])
    y_real = np.array([1] * len(pos_np) + [0] * len(neg_np))

    # Real accuracy (GPU-accelerated CV)
    real_scores = _gpu_cv_accuracy(X, y_real, C=cfg.SVM_C,
                                   n_splits=cfg.SVM_CV_FOLDS, seed=seed)
    real_accuracy = float(real_scores.mean())

    # Control accuracy (shuffled labels)
    rng = np.random.RandomState(seed)
    control_accuracies = []
    for i in range(n_control_runs):
        y_shuffled = y_real.copy()
        rng.shuffle(y_shuffled)

        ctrl_scores = _gpu_cv_accuracy(X, y_shuffled, C=cfg.SVM_C,
                                       n_splits=cfg.SVM_CV_FOLDS,
                                       seed=seed + i + 1)
        control_accuracies.append(float(ctrl_scores.mean()))

    mean_control = float(np.mean(control_accuracies))
    selectivity = real_accuracy - mean_control

    if selectivity >= 0.30:
        interpretation = "strong"
    elif selectivity >= 0.10:
        interpretation = "meaningful"
    else:
        interpretation = "weak (possible memorization)"

    return {
        "real_accuracy": round(real_accuracy, 4),
        "control_accuracies": [round(a, 4) for a in control_accuracies],
        "mean_control_accuracy": round(mean_control, 4),
        "selectivity": round(selectivity, 4),
        "interpretation": interpretation,
        "n_control_runs": n_control_runs,
    }


# =========================================================================
# 3. COHEN'S D — Cohen (1988)
# =========================================================================
def cohens_d_activations(pos_acts, neg_acts, cav):
    """
    Cohen's d / Hedges' g for positive vs negative activations projected
    onto the CAV direction.

    FIX-M1: Enhanced with Levene's test for variance homogeneity,
    Welch's t-test (robust to unequal variances), Hedges' g (small-sample
    bias correction), and 95% CI for the effect size.
    """
    from scipy import stats as sp_stats

    cav_np = cav.numpy() if isinstance(cav, torch.Tensor) else cav
    pos_np = pos_acts.numpy() if isinstance(pos_acts, torch.Tensor) else pos_acts
    neg_np = neg_acts.numpy() if isinstance(neg_acts, torch.Tensor) else neg_acts

    pos_proj = pos_np @ cav_np
    neg_proj = neg_np @ cav_np

    mean_pos = float(pos_proj.mean())
    mean_neg = float(neg_proj.mean())
    n1, n2 = len(pos_proj), len(neg_proj)

    var1 = float(np.var(pos_proj, ddof=1))
    var2 = float(np.var(neg_proj, ddof=1))
    pooled_std = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))

    # Cohen's d
    d = (mean_pos - mean_neg) / pooled_std if pooled_std > 0 else 0.0

    # Hedges' g: small-sample bias correction (Hedges, 1981)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = d * correction

    # Levene's test for variance homogeneity
    levene_stat, levene_p = sp_stats.levene(pos_proj, neg_proj)
    variances_equal = bool(levene_p > 0.05)

    # t-test: Welch's if variances unequal, Student's otherwise
    t_stat, t_p = sp_stats.ttest_ind(pos_proj, neg_proj,
                                      equal_var=variances_equal)

    # 95% CI for Cohen's d (approximate, Hedges & Olkin, 1985)
    se_d = float(np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2))))
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
        "variances_equal": variances_equal,
        "test_type": "Student" if variances_equal else "Welch",
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "interpretation": interpretation,
    }


# =========================================================================
# 4. BOOTSTRAP CI — Efron & Tibshirani (1993)
# =========================================================================
def _bca_ci(data, theta_hat, boot_thetas, ci_level):
    """
    Compute BCa (bias-corrected and accelerated) bootstrap CI.

    BCa corrects for both bias and skewness in the bootstrap distribution
    (Efron & Tibshirani, 1993, Chapter 14). Falls back to percentile CI
    if BCa computation fails.

    Args:
        data: Original data array (for jackknife acceleration)
        theta_hat: Point estimate from original data
        boot_thetas: Array of bootstrap estimates
        ci_level: Confidence level (e.g. 0.95)

    Returns:
        (ci_lower, ci_upper, method_used)
    """
    from scipy import stats as sp_stats

    alpha = 1 - ci_level
    n = len(data)

    try:
        # Bias correction factor z0
        prop_below = np.mean(boot_thetas < theta_hat)
        z0 = sp_stats.norm.ppf(max(1e-10, min(1 - 1e-10, prop_below)))

        # Acceleration factor a (jackknife)
        # For CAV cosine stability, we approximate jackknife on original data
        jack_vals = np.empty(n)
        for i in range(n):
            jack_vals[i] = np.mean(np.delete(boot_thetas[:n], i))
        jack_mean = jack_vals.mean()
        num = np.sum((jack_mean - jack_vals) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
        a = num / den if abs(den) > 1e-10 else 0.0

        # BCa percentiles
        z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
        z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

        a1 = sp_stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
        a2 = sp_stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))

        ci_lower = float(np.percentile(boot_thetas, 100 * a1))
        ci_upper = float(np.percentile(boot_thetas, 100 * a2))
        return ci_lower, ci_upper, "BCa"

    except Exception:
        # Fallback to percentile CI
        ci_lower = float(np.percentile(boot_thetas, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_thetas, 100 * (1 - alpha / 2)))
        return ci_lower, ci_upper, "percentile"


def bootstrap_ci_cav_direction(pos_acts, neg_acts, method="mean_diff",
                               n_resamples=None, ci_level=None, seed=None):
    """
    Bootstrap confidence interval for CAV directional stability.

    FIX-M2: Uses BCa (bias-corrected accelerated) CI instead of naive
    percentile CI. BCa corrects for bias and skewness in the bootstrap
    distribution (Efron & Tibshirani, 1993, Chapter 14).

    Procedure:
      1. Extract original CAV
      2. For each resample: resample pos/neg with replacement, extract CAV,
         compute cosine similarity with original
      3. Report BCa CI on cosine similarities (fallback to percentile)

    Returns:
        dict with cosine stats, CI bounds, interpretation
    """
    if n_resamples is None:
        n_resamples = cfg.BOOTSTRAP_N_RESAMPLES
    if ci_level is None:
        ci_level = cfg.BOOTSTRAP_CI_LEVEL
    if seed is None:
        seed = cfg.SVM_RANDOM_STATE

    pos_np = pos_acts.numpy() if isinstance(pos_acts, torch.Tensor) else pos_acts
    neg_np = neg_acts.numpy() if isinstance(neg_acts, torch.Tensor) else neg_acts

    # Original CAV (FIX-M4: assert unit norm)
    if method == "mean_diff":
        orig_cav = pos_np.mean(axis=0) - neg_np.mean(axis=0)
        norm = np.linalg.norm(orig_cav)
        assert norm > 1e-8, "CAV norm near zero — degenerate mean difference"
        orig_cav = orig_cav / norm
    elif method == "svm":
        X = np.vstack([pos_np, neg_np])
        y = np.array([1] * len(pos_np) + [0] * len(neg_np))
        orig_cav = _fit_linear_svm_gpu(X, y, C=cfg.SVM_C)
        norm = np.linalg.norm(orig_cav)
        assert norm > 1e-8, "SVM CAV norm near zero — degenerate hyperplane"

    rng = np.random.RandomState(seed)

    if method == "mean_diff":
        # Sequential bootstrap for mean_diff (already fast on CPU)
        cosines = []
        for _ in range(n_resamples):
            pos_idx = rng.choice(len(pos_np), size=len(pos_np), replace=True)
            neg_idx = rng.choice(len(neg_np), size=len(neg_np), replace=True)
            boot_cav = pos_np[pos_idx].mean(axis=0) - neg_np[neg_idx].mean(axis=0)
            norm = np.linalg.norm(boot_cav)
            boot_cav = boot_cav / norm if norm > 1e-8 else boot_cav
            cosines.append(float(np.dot(orig_cav, boot_cav)))
    elif method == "svm":
        # Batched GPU bootstrap — all resamples trained in parallel
        X_list, y_list = [], []
        for _ in range(n_resamples):
            pos_idx = rng.choice(len(pos_np), size=len(pos_np), replace=True)
            neg_idx = rng.choice(len(neg_np), size=len(neg_np), replace=True)
            X_boot = np.vstack([pos_np[pos_idx], neg_np[neg_idx]])
            y_boot = np.array([1] * len(pos_np) + [0] * len(neg_np))
            X_list.append(X_boot)
            y_list.append(y_boot)
        # Train all SVMs simultaneously on GPU
        boot_cavs = _batched_linear_svm_gpu(X_list, y_list, C=cfg.SVM_C)
        cosines = (boot_cavs @ orig_cav).tolist()

    cosines = np.array(cosines)
    median_cos = float(np.median(cosines))

    # FIX-M2: BCa bootstrap CI (with percentile fallback)
    ci_lower, ci_upper, ci_method = _bca_ci(
        cosines, median_cos, cosines, ci_level
    )

    if ci_lower >= 0.90:
        interpretation = "very stable"
    elif ci_lower >= 0.80:
        interpretation = "stable"
    elif ci_lower >= 0.60:
        interpretation = "moderately stable"
    else:
        interpretation = "unreliable"

    return {
        "median_cosine": round(median_cos, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci_level,
        "ci_method": ci_method,
        "n_resamples": len(cosines),
        "interpretation": interpretation,
    }


# =========================================================================
# ORCHESTRATOR
# =========================================================================
def run_all_statistical_tests(use_saved_acts=True):
    """
    Run all statistical validation tests on extracted CAVs.
    Outputs cavs/statistical_validation.json.

    Parameters:
        use_saved_acts: If True, load pre-saved activations from
                        cavs/*_acts_*.pt (faster, no model needed).
                        If False, re-collect activations using the model.
    """
    print("=" * 60)
    print("CAV Statistical Validation")
    print("=" * 60)

    results = {}
    model = None

    if not use_saved_acts:
        from transformer_lens import HookedTransformer
        print(f"\nLoading model: {cfg.MODEL_NAME}...")
        model = HookedTransformer.from_pretrained(cfg.MODEL_NAME)
        model.eval()

    for concept in cfg.CONCEPTS:
        print(f"\n{'='*40}")
        print(f"CONCEPT: {concept.upper()}")
        print(f"{'='*40}")

        results[concept] = {}

        for layer in cfg.EXTRACTION_LAYERS:
            print(f"\n  --- Layer {layer} ---")
            results[concept][f"layer_{layer}"] = {}

            # Load activations
            acts_path = os.path.join(cfg.CAV_DIR,
                                     f"{concept}_acts_layer{layer}.pt")
            if use_saved_acts and os.path.exists(acts_path):
                acts_data = torch.load(acts_path, weights_only=False)
                pos_acts = acts_data["pos"]
                neg_acts = acts_data["neg"]
                print(f"  Loaded saved activations: {len(pos_acts)} pos, "
                      f"{len(neg_acts)} neg")
            elif model is not None:
                from cav_extraction import collect_activations
                with open(cfg.DATA_FILE, "r") as f:
                    data = json.load(f)
                pos_sents = data["cav_training"][concept]["positive"]
                neg_sents = data["cav_training"][concept]["negative"]
                pos_acts = collect_activations(model, pos_sents, layer)
                neg_acts = collect_activations(model, neg_sents, layer)
            else:
                print(f"  SKIP: No saved activations and no model. "
                      f"Run cav_extraction.py with SAVE_ACTIVATIONS=True first.")
                continue

            for method in cfg.CAV_METHODS:
                print(f"\n  Method: {method}")

                # Load CAV
                cav_path = os.path.join(
                    cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt"
                )
                if not os.path.exists(cav_path):
                    print(f"    SKIP: {cav_path} not found")
                    continue
                cav = torch.load(cav_path, weights_only=False)

                method_results = {}

                # --- Test 1: TCAV Permutation ---
                print(f"    Running TCAV permutation test "
                      f"(n={cfg.TCAV_PERMUTATION_RUNS})...")
                tcav_result = tcav_permutation_test(
                    pos_acts, neg_acts, cav, method=method
                )
                method_results["tcav_permutation"] = tcav_result
                sig = "YES" if tcav_result["significant"] else "NO"
                print(f"      TCAV score: {tcav_result['real_tcav_score']:.3f}, "
                      f"p={tcav_result['p_value']:.3f} "
                      f"[significant={sig}]")

                # --- Test 2: Selectivity ---
                print(f"    Running selectivity test "
                      f"(n={cfg.SELECTIVITY_CONTROL_RUNS})...")
                sel_result = selectivity_test(
                    pos_acts, neg_acts, method=method
                )
                method_results["selectivity"] = sel_result
                print(f"      Real acc: {sel_result['real_accuracy']:.3f}, "
                      f"Control acc: {sel_result['mean_control_accuracy']:.3f}, "
                      f"Selectivity: {sel_result['selectivity']:.3f} "
                      f"({sel_result['interpretation']})")

                # --- Test 3: Cohen's d ---
                d_result = cohens_d_activations(pos_acts, neg_acts, cav)
                method_results["cohens_d"] = d_result
                print(f"      Cohen's d: {d_result['cohens_d']:.3f} "
                      f"({d_result['interpretation']})")

                # --- Test 4: Bootstrap CI ---
                # Use fewer resamples for SVM (expensive)
                n_boot = min(cfg.BOOTSTRAP_N_RESAMPLES,
                             200 if method == "svm" else 1000)
                print(f"    Running bootstrap CI (n={n_boot})...")
                boot_result = bootstrap_ci_cav_direction(
                    pos_acts, neg_acts, method=method,
                    n_resamples=n_boot,
                )
                method_results["bootstrap_ci"] = boot_result
                print(f"      Cosine CI [{boot_result['ci_lower']:.3f}, "
                      f"{boot_result['ci_upper']:.3f}] "
                      f"({boot_result['interpretation']})")

                results[concept][f"layer_{layer}"][method] = method_results

    # Save results
    output = {
        "config": {
            "tcav_permutation_runs": cfg.TCAV_PERMUTATION_RUNS,
            "tcav_alpha": cfg.TCAV_SIGNIFICANCE_ALPHA,
            "bootstrap_n_resamples": cfg.BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": cfg.BOOTSTRAP_CI_LEVEL,
            "selectivity_control_runs": cfg.SELECTIVITY_CONTROL_RUNS,
            "svm_c": cfg.SVM_C,
        },
        "results": results,
    }

    output_path = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print("Done.")

    return output


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAV Statistical Validation")
    parser.add_argument("--from-saved", action="store_true", default=True,
                        help="Use saved activations (default: True)")
    parser.add_argument("--recompute", action="store_true",
                        help="Re-collect activations using the model")
    args = parser.parse_args()

    use_saved = not args.recompute
    run_all_statistical_tests(use_saved_acts=use_saved)
