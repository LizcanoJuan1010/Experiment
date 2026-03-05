"""
CAV Extraction — LLaVA-1.5 7B via pyvene
==========================================
Extracts Concept Activation Vectors at multiple layers using:
  - Mean difference method
  - Linear SVM method (TCAV, Kim et al. 2018)

Uses pyvene_utils for activation collection from the Vicuna-7B
language model inside LLaVA-1.5.

Usage:
    python cav_extraction.py

Input:  data/experiment_data.json
Output: cavs/*.pt, cavs/extraction_report.json
"""

import torch
import json
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

import experiment_config as cfg
from model_loader import load_llava_model
from pyvene_utils import collect_activations_batch

# Configuration
CONCEPTS = cfg.CONCEPTS
LAYERS = cfg.EXTRACTION_LAYERS
METHODS = cfg.CAV_METHODS
OUTPUT_DIR = cfg.CAV_DIR


# ---------------------------------------------------------------------------
# CAV Extraction Methods (reused from Pythia — operate on tensors)
# ---------------------------------------------------------------------------
def extract_cav_mean_diff(pos_acts, neg_acts):
    """
    Extract CAV via mean difference: CAV = normalize(mean(pos) - mean(neg)).

    Args:
        pos_acts: numpy array [N_pos, d_model]
        neg_acts: numpy array [N_neg, d_model]

    Returns:
        cav: numpy array [d_model] (unit norm)
        info: dict with metadata
    """
    diff = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm > 1e-8:
        cav = diff / norm
    else:
        cav = diff
    return cav, {"norm_before_normalize": float(norm)}


def extract_cav_svm(pos_acts, neg_acts):
    """
    Extract CAV via Linear SVM (TCAV method).

    Args:
        pos_acts: numpy array [N_pos, d_model]
        neg_acts: numpy array [N_neg, d_model]

    Returns:
        cav: numpy array [d_model] (unit norm)
        info: dict with cv_accuracy, training_accuracy, etc.
    """
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    # Hyperparameter tuning
    best_c = 1.0
    best_score = 0.0

    if cfg.SVM_TUNE_C:
        for c in cfg.SVM_C_VALUES:
            clf = LinearSVC(C=c, max_iter=cfg.SVM_MAX_ITER, random_state=cfg.RANDOM_SEED)
            try:
                cv = StratifiedKFold(n_splits=cfg.SVM_CV_FOLDS, shuffle=True,
                                     random_state=cfg.RANDOM_SEED)
                scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_c = c
            except Exception:
                continue

    # Train final SVM
    clf = LinearSVC(C=best_c, max_iter=cfg.SVM_MAX_ITER, random_state=cfg.RANDOM_SEED)
    clf.fit(X, y)

    # Cross-validation accuracy
    cv = StratifiedKFold(n_splits=cfg.SVM_CV_FOLDS, shuffle=True,
                         random_state=cfg.RANDOM_SEED)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    # Extract and normalize CAV
    cav = clf.coef_[0]
    norm = np.linalg.norm(cav)
    if norm > 1e-8:
        cav = cav / norm

    info = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "training_accuracy": float(clf.score(X, y)),
        "best_C": best_c,
        "coef_norm": float(norm),
        "n_iter": int(getattr(clf, "n_iter_", 0)),
    }

    return cav, info


def cohens_d(pos_acts, neg_acts, cav):
    """Compute Cohen's d effect size along CAV direction."""
    pos_proj = pos_acts @ cav
    neg_proj = neg_acts @ cav

    mean_diff = pos_proj.mean() - neg_proj.mean()
    pooled_std = np.sqrt(
        ((len(pos_proj) - 1) * pos_proj.std() ** 2 +
         (len(neg_proj) - 1) * neg_proj.std() ** 2) /
        (len(pos_proj) + len(neg_proj) - 2)
    )

    if pooled_std > 1e-8:
        return float(mean_diff / pooled_std)
    return 0.0


# ---------------------------------------------------------------------------
# Layer Diagnostic Probing
# ---------------------------------------------------------------------------
def run_layer_diagnostic(model, processor, pos_pairs, neg_pairs):
    """
    Probe all layers with LogisticRegression to find best concept layers.

    IMPORTANT: With n=20 samples per class and d=4096 dimensions, a linear
    classifier can trivially achieve 100% accuracy because the problem is
    underdetermined (n << d). To get meaningful results, we:
      1. Apply PCA to reduce dimensionality to min(n_total, 50) components
      2. Use 5-fold stratified CV with L2-regularized LogisticRegression

    The PCA reduction ensures the probe accuracy reflects genuine concept
    separability rather than high-dimensional overfitting.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    print("\n  Layer Diagnostic Probing (all 32 layers)...")
    print("    (Using PCA reduction to avoid n << d overfitting)")
    layer_scores = {}

    for layer in range(cfg.N_LAYERS_TOTAL):
        pos_acts = collect_activations_batch(
            model, processor, pos_pairs[:20], layer,
            show_progress=False,
        ).numpy()
        neg_acts = collect_activations_batch(
            model, processor, neg_pairs[:20], layer,
            show_progress=False,
        ).numpy()

        X = np.vstack([pos_acts, neg_acts])
        y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

        # PCA to avoid trivial separation: reduce to min(n_samples, 50) components
        n_components = min(len(X) - 2, 50)  # -2 to leave room for CV folds
        pipe = Pipeline([
            ("pca", PCA(n_components=n_components, random_state=cfg.RANDOM_SEED)),
            ("clf", LogisticRegression(
                max_iter=5000, C=1.0, random_state=cfg.RANDOM_SEED,
            )),
        ])
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=cfg.RANDOM_SEED)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            layer_scores[layer] = float(scores.mean())
        except Exception:
            layer_scores[layer] = 0.5

        print(f"    Layer {layer:2d}: {layer_scores[layer]:.3f}")

    # Top layers
    sorted_layers = sorted(layer_scores.items(), key=lambda x: -x[1])
    top_k = sorted_layers[:cfg.LAYER_VALIDITY_TOP_K]
    print(f"\n  Top {cfg.LAYER_VALIDITY_TOP_K} layers: "
          f"{[(l, f'{s:.3f}') for l, s in top_k]}")

    return layer_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  CAV Extraction — LLaVA-1.5 7B via pyvene")
    print(f"  Concepts: {CONCEPTS}")
    print(f"  Layers: {LAYERS}")
    print(f"  Methods: {METHODS}")
    print(f"{sep}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"\n  Loading data from {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"  Loading {cfg.MODEL_NAME} (4-bit)...")
    model, processor = load_llava_model()
    print(f"  Model loaded.")

    report = {
        "model": cfg.MODEL_NAME,
        "layers": LAYERS,
        "methods": METHODS,
        "concepts": CONCEPTS,
        "cavs": {},
        "layer_diagnostic": {},
    }

    # Optional: Layer diagnostic with first concept
    first_concept = CONCEPTS[0]
    first_pos = data["cav_training"][first_concept]["positive"]
    first_neg = data["cav_training"][first_concept]["negative"]
    if len(first_pos) > 0 and len(first_neg) > 0:
        report["layer_diagnostic"] = run_layer_diagnostic(
            model, processor, first_pos, first_neg,
        )

    # Validate training data exists
    empty_concepts = [c for c in CONCEPTS
                      if not data["cav_training"][c]["positive"]
                      or not data["cav_training"][c]["negative"]]
    if empty_concepts:
        print(f"\n  FATAL: No training data for: {empty_concepts}")
        print(f"  Run data_gen_multimodal.py first (with HuggingFace auth).")
        raise SystemExit(1)

    # Extract CAVs
    for concept in CONCEPTS:
        pos_pairs = data["cav_training"][concept]["positive"]
        neg_pairs = data["cav_training"][concept]["negative"]

        print(f"\n  === {concept.upper()} ===")
        print(f"    {len(pos_pairs)} positive, {len(neg_pairs)} negative pairs")

        for layer in LAYERS:
            print(f"\n    Layer {layer}:")

            # Collect activations
            print(f"      Collecting positive activations...")
            pos_acts = collect_activations_batch(
                model, processor, pos_pairs, layer,
            ).numpy()

            print(f"      Collecting negative activations...")
            neg_acts = collect_activations_batch(
                model, processor, neg_pairs, layer,
            ).numpy()

            print(f"      Shapes: pos={pos_acts.shape}, neg={neg_acts.shape}")

            # Save raw activations for downstream statistical tests
            if cfg.SAVE_ACTIVATIONS:
                acts_path = os.path.join(
                    OUTPUT_DIR, f"{concept}_acts_layer{layer}.pt"
                )
                torch.save({
                    "pos": torch.tensor(pos_acts, dtype=torch.float32),
                    "neg": torch.tensor(neg_acts, dtype=torch.float32),
                }, acts_path)
                print(f"      Saved activations: {acts_path}")

            for method in METHODS:
                print(f"      Method: {method}")

                if method == "mean_diff":
                    cav, info = extract_cav_mean_diff(pos_acts, neg_acts)
                elif method == "svm":
                    cav, info = extract_cav_svm(pos_acts, neg_acts)
                else:
                    continue

                # Effect size
                d = cohens_d(pos_acts, neg_acts, cav)
                info["cohens_d"] = d

                # Save CAV
                cav_tensor = torch.tensor(cav, dtype=torch.float32)
                filename = f"{concept}_{method}_layer{layer}.pt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                torch.save(cav_tensor, filepath)

                key = f"{concept}_{method}_layer{layer}"
                report["cavs"][key] = info
                print(f"        Saved: {filename}")
                if method == "svm":
                    print(f"        CV accuracy: {info['cv_accuracy_mean']:.3f} "
                          f"(+/- {info['cv_accuracy_std']:.3f})")
                print(f"        Cohen's d: {d:.3f}")

    # Cross-concept cosine similarity
    print(f"\n{sep}")
    print("  CROSS-CONCEPT COSINE SIMILARITY")
    print(f"{sep}")

    report["cross_concept_cosine"] = {}
    for layer in LAYERS:
        report["cross_concept_cosine"][f"layer_{layer}"] = {}
        for method in METHODS:
            cavs = {}
            for concept in CONCEPTS:
                filepath = os.path.join(
                    OUTPUT_DIR, f"{concept}_{method}_layer{layer}.pt"
                )
                if os.path.exists(filepath):
                    cavs[concept] = torch.load(filepath, weights_only=True)

            pairs = [(CONCEPTS[i], CONCEPTS[j])
                     for i in range(len(CONCEPTS))
                     for j in range(i + 1, len(CONCEPTS))]

            print(f"    Layer {layer}, {method}:")
            for c1, c2 in pairs:
                if c1 in cavs and c2 in cavs:
                    cos = torch.nn.functional.cosine_similarity(
                        cavs[c1].unsqueeze(0), cavs[c2].unsqueeze(0)
                    ).item()
                    status = "OK" if abs(cos) < cfg.CAV_OVERLAP_THRESHOLD else "WARNING"
                    print(f"      {c1}-{c2}: {cos:.4f} [{status}]")
                    report["cross_concept_cosine"][f"layer_{layer}"][
                        f"{method}_{c1}_{c2}"
                    ] = cos

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
