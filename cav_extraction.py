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
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2-small"
DATA_FILE = "experiment_data.json"
OUTPUT_DIR = "cavs"

# Layers to extract CAVs from:
#   Middle (6) = semantic knowledge hypothesis
#   Late (9,10) = syntactic composition / token selection
LAYERS = [6, 9, 10]

CONCEPTS = ["time", "place", "tools"]
METHODS = ["mean_diff", "svm"]


# ---------------------------------------------------------------------------
# Activation Collection
# ---------------------------------------------------------------------------
def collect_activations(model, sentences, layer):
    """
    Collect per-sentence activations at the last token position
    for a given layer. Returns a tensor of shape (N, d_model).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    with torch.no_grad():
        for sent in tqdm(sentences, desc=f"  Layer {layer}", leave=False):
            _, cache = model.run_with_cache(sent, names_filter=[hook_name])
            act = cache[hook_name]       # [1, seq_len, d_model]
            last_tok = act[0, -1, :]     # [d_model]
            activations.append(last_tok.cpu())

    return torch.stack(activations)  # [N, d_model]


# ---------------------------------------------------------------------------
# CAV Extraction Methods
# ---------------------------------------------------------------------------
def extract_cav_mean_diff(pos_acts, neg_acts):
    """
    Method 1: Mean Difference.
    CAV = normalize(mean(pos) - mean(neg))
    """
    mean_pos = pos_acts.mean(dim=0)
    mean_neg = neg_acts.mean(dim=0)
    cav = mean_pos - mean_neg
    cav = cav / cav.norm()
    return cav, {}


def extract_cav_svm(pos_acts, neg_acts):
    """
    Method 2: Linear SVM (original TCAV, Kim et al. 2018).
    CAV = normal vector to the SVM decision boundary.
    Also reports cross-validation accuracy as quality metric.
    """
    X = torch.cat([pos_acts, neg_acts], dim=0).numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    # Train SVM
    clf = LinearSVC(C=1.0, max_iter=10000)
    clf.fit(X, y)

    # CAV = coefficient vector (normal to decision boundary)
    cav_np = clf.coef_[0]
    cav_np = cav_np / np.linalg.norm(cav_np)
    cav = torch.tensor(cav_np, dtype=torch.float32)

    # Cross-validation accuracy (quality metric)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LinearSVC(C=1.0, max_iter=10000), X, y, cv=skf, scoring="accuracy"
    )

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_scores": [float(s) for s in cv_scores],
        "train_accuracy": float(clf.score(X, y)),
    }

    return cav, metrics


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
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
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

    # --- Extract CAVs ---
    for concept in CONCEPTS:
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

            for method in METHODS:
                print(f"  Extracting CAV: {method}...")

                if method == "mean_diff":
                    cav, metrics = extract_cav_mean_diff(pos_acts, neg_acts)
                elif method == "svm":
                    cav, metrics = extract_cav_svm(pos_acts, neg_acts)

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
                    status = "OK" if acc >= 0.70 else "WARNING (< 0.70)"
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
                cavs[concept] = torch.load(filepath)

            pairs = [
                ("time", "place"),
                ("time", "tools"),
                ("place", "tools"),
            ]
            print(f"    {method}:")
            for c1, c2 in pairs:
                cos = torch.nn.functional.cosine_similarity(
                    cavs[c1].unsqueeze(0), cavs[c2].unsqueeze(0)
                ).item()
                status = "OK" if abs(cos) < 0.3 else "WARNING (|cos| >= 0.3)"
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
                os.path.join(OUTPUT_DIR, f"{concept}_mean_diff_layer{layer}.pt")
            )
            cav_svm = torch.load(
                os.path.join(OUTPUT_DIR, f"{concept}_svm_layer{layer}.pt")
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
