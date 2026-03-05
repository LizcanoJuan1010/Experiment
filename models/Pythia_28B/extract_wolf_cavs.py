"""
Wolf CAV Extraction
===================
Extracts Concept Activation Vectors for the "wolf" concept at the
standard extraction layers, using both mean_diff and SVM methods.

Reuses the extraction functions from cav_extraction.py.

Usage:
    python extract_wolf_cavs.py

Prerequisite: experiment_data.json must contain wolf training data
              (run gen_wolf_data.py first).
Output: cavs/wolf_mean_diff_layer{L}.pt
        cavs/wolf_svm_layer{L}.pt
        cavs/wolf_acts_layer{L}.pt  (if SAVE_ACTIVATIONS is True)
"""

import torch
import json
import os

import experiment_config as cfg
from cav_extraction import (
    collect_activations,
    extract_cav_mean_diff,
    extract_cav_svm,
    cohens_d_activations,
)

CONCEPT = "wolf"
LAYERS = cfg.EXTRACTION_LAYERS    # [16, 24, 27]
METHODS = cfg.CAV_METHODS         # ["mean_diff", "svm"]
OUTPUT_DIR = cfg.CAV_DIR          # "cavs"


def main():
    print("=" * 60)
    print(f"CAV Extraction -- {CONCEPT.upper()}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load experiment data
    print(f"\n  Loading {cfg.DATA_FILE}...")
    with open(cfg.DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if CONCEPT not in data.get("cav_training", {}):
        print(f"\n  ERROR: '{CONCEPT}' not found in {cfg.DATA_FILE}")
        print(f"  Run 'python gen_wolf_data.py' first.")
        return

    pos_sentences = data["cav_training"][CONCEPT]["positive"]
    neg_sentences = data["cav_training"][CONCEPT]["negative"]
    print(f"  Positive sentences: {len(pos_sentences)}")
    print(f"  Negative sentences: {len(neg_sentences)}")

    # Load model
    print(f"\n  Loading model: {cfg.MODEL_NAME}...")
    model = cfg.load_model()

    extraction_report = {}

    for layer in LAYERS:
        print(f"\n{'=' * 50}")
        print(f"  Layer {layer}")
        print(f"{'=' * 50}")

        # Collect activations
        print(f"  Collecting positive activations...")
        pos_acts = collect_activations(model, pos_sentences, layer)
        print(f"  Collecting negative activations...")
        neg_acts = collect_activations(model, neg_sentences, layer)
        print(f"  Shapes: pos={pos_acts.shape}, neg={neg_acts.shape}")

        # Save activations
        if getattr(cfg, "SAVE_ACTIVATIONS", True):
            acts_path = os.path.join(OUTPUT_DIR, f"{CONCEPT}_acts_layer{layer}.pt")
            torch.save({"pos": pos_acts, "neg": neg_acts}, acts_path)
            print(f"  Saved activations: {acts_path}")

        layer_report = {}

        for method in METHODS:
            print(f"\n  --- {method} ---")

            if method == "mean_diff":
                cav, metrics = extract_cav_mean_diff(pos_acts, neg_acts)
            elif method == "svm":
                cav, metrics = extract_cav_svm(pos_acts, neg_acts)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Effect size
            d_result = cohens_d_activations(pos_acts, neg_acts, cav)
            print(f"    Cohen's d: {d_result['cohens_d']:.3f} "
                  f"({d_result['interpretation']})")
            if method == "svm" and "cv_accuracy_mean" in metrics:
                print(f"    SVM CV accuracy: {metrics['cv_accuracy_mean']:.3f} "
                      f"(std={metrics['cv_accuracy_std']:.3f})")

            # Save CAV
            filename = f"{CONCEPT}_{method}_layer{layer}.pt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            torch.save(cav, filepath)
            print(f"    Saved: {filepath}")

            layer_report[method] = {
                "cohens_d": d_result["cohens_d"],
                "interpretation": d_result["interpretation"],
                **metrics,
            }

        extraction_report[f"layer_{layer}"] = layer_report

    # Save report
    report_path = os.path.join(OUTPUT_DIR, f"{CONCEPT}_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(extraction_report, f, indent=2)
    print(f"\n  Extraction report: {report_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for layer in LAYERS:
        lr = extraction_report[f"layer_{layer}"]
        for method in METHODS:
            mr = lr[method]
            d = mr["cohens_d"]
            interp = mr["interpretation"]
            print(f"  L{layer}/{method}: Cohen's d = {d:.3f} ({interp})")
    print("\nDone.")


if __name__ == "__main__":
    main()
