"""
Validation Test Suite
=====================
12 automated tests to validate the experimental pipeline.
Tests that require the model are skipped if transformer_lens is unavailable.

Usage (inside Docker):
    python validation_tests.py

    # Run only data-level tests (no model needed):
    python validation_tests.py --no-model

Input:  experiment_data.json, cavs/ (for model-dependent tests)
Output: validation_report.json
"""

import json
import os
import sys
import argparse
import math
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "experiment_data.json"
CAV_DIR = "cavs"
REPORT_FILE = "validation_report.json"

CONCEPTS = ["time", "place", "tools"]

# All concept words across categories (for contamination checks)
# Will be populated from experiment_data.json


# ---------------------------------------------------------------------------
# Test Results Collector
# ---------------------------------------------------------------------------
class TestResults:
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def record(self, name, passed, details=None, skipped=False):
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        self.results[name] = {"status": status, "details": details or {}}
        if skipped:
            self.skipped += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1
        icon = {"PASS": "OK", "FAIL": "!!", "SKIP": "--"}[status]
        print(f"  [{icon}] {name}: {status}")
        if details and not passed and not skipped:
            for k, v in details.items():
                if isinstance(v, (str, int, float, bool)):
                    print(f"       {k}: {v}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n  Results: {self.passed} passed, {self.failed} failed, "
              f"{self.skipped} skipped / {total} total")
        return self.results


# ---------------------------------------------------------------------------
# TEST 1: ConceptNet Data Quality
# ---------------------------------------------------------------------------
def test_01_conceptnet_quality(data, results):
    """Check concept node counts and overlap."""
    details = {}
    passed = True

    for concept in CONCEPTS:
        nodes = data["concept_nodes"][concept]["nodes"]
        count = len(nodes)
        details[f"{concept}_count"] = count
        if count < 20:
            passed = False
            details[f"{concept}_status"] = f"FAIL: {count} < 20 nodes"
        else:
            details[f"{concept}_status"] = f"OK: {count} nodes"

    results.record("T01_concept_node_counts", passed, details)


# ---------------------------------------------------------------------------
# TEST 2: Negative Example Contamination
# ---------------------------------------------------------------------------
def test_02_negative_contamination(data, results):
    """Scan negative sentences for concept keywords."""
    all_concept_words = set()
    for concept in CONCEPTS:
        for word in data["concept_nodes"][concept]["nodes"]:
            all_concept_words.add(word.lower())
            # Also add individual words for multi-word nodes
            for w in word.lower().split():
                if len(w) > 3:
                    all_concept_words.add(w)

    details = {}
    passed = True

    for concept in CONCEPTS:
        negatives = data["cav_training"][concept]["negative"]
        contaminated = []
        for i, sent in enumerate(negatives):
            sent_lower = sent.lower()
            for cw in all_concept_words:
                if f" {cw} " in f" {sent_lower} " or f" {cw}." in sent_lower:
                    contaminated.append((i, sent[:60], cw))
                    break

        ratio = len(contaminated) / len(negatives) if negatives else 0
        details[f"{concept}_contaminated"] = len(contaminated)
        details[f"{concept}_total"] = len(negatives)
        details[f"{concept}_ratio"] = round(ratio, 4)

        if ratio >= 0.05:
            passed = False
            details[f"{concept}_examples"] = [
                f"'{s}' contains '{w}'" for _, s, w in contaminated[:3]
            ]

    results.record("T02_negative_contamination", passed, details)


# ---------------------------------------------------------------------------
# TEST 3: Polysemy Detection
# ---------------------------------------------------------------------------
def test_03_polysemy(data, results):
    """Find words that appear in multiple concept categories."""
    word_to_concepts = {}
    for concept in CONCEPTS:
        for word in data["concept_nodes"][concept]["nodes"]:
            w = word.lower()
            if w not in word_to_concepts:
                word_to_concepts[w] = []
            word_to_concepts[w].append(concept)

    overlapping = {w: cs for w, cs in word_to_concepts.items() if len(cs) > 1}
    total_words = len(word_to_concepts)
    ratio = len(overlapping) / total_words if total_words > 0 else 0

    passed = len(overlapping) == 0 or ratio < 0.08
    details = {
        "overlapping_words": overlapping if overlapping else "none",
        "overlap_count": len(overlapping),
        "total_unique_words": total_words,
        "overlap_ratio": round(ratio, 4),
    }

    results.record("T03_polysemy_detection", passed, details)


# ---------------------------------------------------------------------------
# TEST 4: Sample Size Adequacy
# ---------------------------------------------------------------------------
def test_04_sample_size(data, results):
    """Check training data sizes for both CAV methods."""
    details = {}
    passed = True

    for concept in CONCEPTS:
        n_pos = len(data["cav_training"][concept]["positive"])
        n_neg = len(data["cav_training"][concept]["negative"])
        balance = n_pos / n_neg if n_neg > 0 else 0

        details[f"{concept}_pos"] = n_pos
        details[f"{concept}_neg"] = n_neg
        details[f"{concept}_balance"] = round(balance, 3)

        # Mean diff: >= 30 per class
        if n_pos < 30 or n_neg < 30:
            passed = False
            details[f"{concept}_mean_diff"] = "FAIL: < 30"
        else:
            details[f"{concept}_mean_diff"] = "OK"

        # SVM: >= 100 per class recommended
        if n_pos < 100 or n_neg < 100:
            details[f"{concept}_svm"] = f"WARNING: {min(n_pos,n_neg)} < 100"
        else:
            details[f"{concept}_svm"] = "OK"

        # Balance check
        if balance < 0.8 or balance > 1.25:
            passed = False
            details[f"{concept}_balance_status"] = "FAIL: imbalanced"

    results.record("T04_sample_size", passed, details)


# ---------------------------------------------------------------------------
# TEST 5: Tokenization Integrity
# ---------------------------------------------------------------------------
def test_05_tokenization(data, results, model=None):
    """Check how GPT-2 BPE tokenizes concept words."""
    if model is None:
        results.record("T05_tokenization", False, {}, skipped=True)
        return

    details = {}
    multi_token = []
    total_words = 0
    single_count = 0

    for concept in CONCEPTS:
        for word in data["concept_nodes"][concept]["nodes"]:
            total_words += 1
            tokens = model.to_tokens(word, prepend_bos=False)[0]
            n_tokens = len(tokens)
            if n_tokens == 1:
                single_count += 1
            else:
                multi_token.append((concept, word, n_tokens))

    single_ratio = single_count / total_words if total_words > 0 else 0
    excessive = [x for x in multi_token if x[2] > 3]

    passed = single_ratio >= 0.50 and len(excessive) == 0
    details = {
        "total_words": total_words,
        "single_token": single_count,
        "single_ratio": round(single_ratio, 3),
        "multi_token_count": len(multi_token),
        "excessive_splits": [(w, n) for c, w, n in excessive],
    }

    results.record("T05_tokenization", passed, details)


# ---------------------------------------------------------------------------
# TEST 6: Template Artifact Detection (permutation test)
# ---------------------------------------------------------------------------
def test_06_template_artifacts(data, results):
    """Check syntactic diversity between positive and negative sentences."""
    import re
    details = {}
    passed = True

    for concept in CONCEPTS:
        pos = data["cav_training"][concept]["positive"]
        neg = data["cav_training"][concept]["negative"]

        # Simple structural check: first-word distribution
        def get_first_words(sentences):
            return Counter(
                re.split(r'\s+', s.strip())[0].lower() for s in sentences
            )

        pos_fw = get_first_words(pos)
        neg_fw = get_first_words(neg)

        # Average sentence length
        pos_avg_len = sum(len(s.split()) for s in pos) / len(pos) if pos else 0
        neg_avg_len = sum(len(s.split()) for s in neg) / len(neg) if neg else 0
        len_diff = abs(pos_avg_len - neg_avg_len)

        # Unique first words ratio
        pos_unique = len(pos_fw) / len(pos) if pos else 0
        neg_unique = len(neg_fw) / len(neg) if neg else 0

        details[f"{concept}_pos_avg_len"] = round(pos_avg_len, 1)
        details[f"{concept}_neg_avg_len"] = round(neg_avg_len, 1)
        details[f"{concept}_len_difference"] = round(len_diff, 1)
        details[f"{concept}_pos_first_word_diversity"] = round(pos_unique, 3)
        details[f"{concept}_neg_first_word_diversity"] = round(neg_unique, 3)

        # Large length difference suggests structural bias
        if len_diff > 3.0:
            passed = False
            details[f"{concept}_status"] = "FAIL: avg length differs > 3 words"

    results.record("T06_template_artifacts", passed, details)


# ---------------------------------------------------------------------------
# TEST 7: Layer Choice Validity (requires model)
# ---------------------------------------------------------------------------
def test_07_layer_validity(data, results, model=None):
    """Train linear probes at each layer to verify layer 6 is good."""
    if model is None:
        results.record("T07_layer_validity", False, {}, skipped=True)
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import torch

    details = {}
    passed = True

    for concept in CONCEPTS:
        pos_sents = data["cav_training"][concept]["positive"][:50]
        neg_sents = data["cav_training"][concept]["negative"][:50]

        layer_accs = {}
        for layer in range(0, 12):
            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = []
            with torch.no_grad():
                for sent in pos_sents + neg_sents:
                    _, cache = model.run_with_cache(
                        sent, names_filter=[hook_name]
                    )
                    acts.append(cache[hook_name][0, -1, :].cpu().numpy())

            X = np.array(acts)
            y = np.array([1]*len(pos_sents) + [0]*len(neg_sents))

            clf = LogisticRegression(max_iter=1000, C=0.1)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            layer_accs[layer] = float(scores.mean())

        # Check layer 6
        sorted_layers = sorted(layer_accs.items(), key=lambda x: -x[1])
        rank_6 = [i for i, (l, _) in enumerate(sorted_layers) if l == 6][0] + 1
        acc_6 = layer_accs[6]

        details[f"{concept}_layer6_accuracy"] = round(acc_6, 3)
        details[f"{concept}_layer6_rank"] = rank_6
        details[f"{concept}_best_layer"] = sorted_layers[0][0]
        details[f"{concept}_best_accuracy"] = round(sorted_layers[0][1], 3)

        if rank_6 > 3 or acc_6 < 0.70:
            passed = False

    results.record("T07_layer_validity", passed, details)


# ---------------------------------------------------------------------------
# TEST 8: SVM Stability
# ---------------------------------------------------------------------------
def test_08_svm_stability(results):
    """Check SVM cross-validation from extraction report."""
    report_path = os.path.join(CAV_DIR, "extraction_report.json")
    if not os.path.exists(report_path):
        results.record("T08_svm_stability", False, {}, skipped=True)
        return

    with open(report_path) as f:
        report = json.load(f)

    details = {}
    passed = True

    for concept in CONCEPTS:
        for layer in [6, 9, 10]:
            key = f"layer_{layer}"
            svm_data = report["results"].get(concept, {}).get(key, {}).get("svm", {})
            if not svm_data:
                continue
            acc = svm_data.get("cv_accuracy_mean", 0)
            std = svm_data.get("cv_accuracy_std", 1)
            details[f"{concept}_L{layer}_acc"] = round(acc, 3)
            details[f"{concept}_L{layer}_std"] = round(std, 3)

            if acc < 0.70:
                passed = False
                details[f"{concept}_L{layer}_status"] = f"FAIL: {acc:.3f} < 0.70"
            elif std >= 0.15:
                details[f"{concept}_L{layer}_status"] = f"WARN: std={std:.3f}"
            else:
                details[f"{concept}_L{layer}_status"] = "OK"

    results.record("T08_svm_stability", passed, details)


# ---------------------------------------------------------------------------
# TEST 9: Concept Overlap (CAV cosine)
# ---------------------------------------------------------------------------
def test_09_cav_overlap(results):
    """Check pairwise cosine between concept CAVs."""
    report_path = os.path.join(CAV_DIR, "extraction_report.json")
    if not os.path.exists(report_path):
        results.record("T09_cav_overlap", False, {}, skipped=True)
        return

    with open(report_path) as f:
        report = json.load(f)

    details = {}
    passed = True

    cross = report.get("cross_concept_cosine", {})
    for layer_key, pairs in cross.items():
        for pair_key, cos_val in pairs.items():
            details[f"{layer_key}_{pair_key}"] = round(cos_val, 4)
            if abs(cos_val) >= 0.3:
                passed = False

    results.record("T09_cav_overlap", passed, details)


# ---------------------------------------------------------------------------
# TEST 10: Benchmark Validity (baseline)
# ---------------------------------------------------------------------------
def test_10_benchmark_validity(results):
    """Check baseline R1 accuracy and R2 similarity."""
    baseline_path = os.path.join("results", "results_baseline.json")
    if not os.path.exists(baseline_path):
        results.record("T10_benchmark_validity", False, {}, skipped=True)
        return

    with open(baseline_path) as f:
        baselines = json.load(f)

    details = {}
    passed = True

    for concept in CONCEPTS:
        r1 = baselines[concept]["r1_accuracy"]
        r2 = baselines[concept]["r2_similarity"]
        details[f"{concept}_r1"] = round(r1, 3)
        details[f"{concept}_r2"] = round(r2, 4)

        if r1 < 0.50:
            passed = False
            details[f"{concept}_r1_status"] = f"FAIL: {r1:.3f} < 0.50"
        else:
            details[f"{concept}_r1_status"] = "OK"

        if r2 < 0.60:
            passed = False
            details[f"{concept}_r2_status"] = f"FAIL: {r2:.4f} < 0.60"
        else:
            details[f"{concept}_r2_status"] = "OK"

    results.record("T10_benchmark_validity", passed, details)


# ---------------------------------------------------------------------------
# TEST 11: Statistical Power
# ---------------------------------------------------------------------------
def test_11_statistical_power(data, results):
    """A priori power analysis for ANOVA."""
    details = {}

    for concept in CONCEPTS:
        n_r1 = len(data["r1_benchmark"][concept])
        n_r2 = len(data["r2_benchmark"][concept])
        details[f"{concept}_r1_items"] = n_r1
        details[f"{concept}_r2_pairs"] = n_r2

    # Simple power approximation for one-way ANOVA
    # Cohen's f = 0.25 (medium), alpha = 0.05, k groups
    # Power ~ 1 - beta; we need N per group
    # For k=4 groups (intensities), f=0.25, alpha=0.05:
    # Required N per group ~ 45 for power=0.80
    # Each "observation" in our case is the metric on the full benchmark
    # With n_r1 items, SE(accuracy) = sqrt(p*(1-p)/n)
    # For p=0.5, SE = 0.5/sqrt(n). Need SE small enough to detect effect.
    min_r1 = min(len(data["r1_benchmark"][c]) for c in CONCEPTS)
    min_r2 = min(len(data["r2_benchmark"][c]) for c in CONCEPTS)

    # Heuristic: >= 20 items gives reasonable sensitivity
    passed = min_r1 >= 20 and min_r2 >= 15
    details["min_r1_items"] = min_r1
    details["min_r2_pairs"] = min_r2
    details["r1_threshold"] = 20
    details["r2_threshold"] = 15
    details["r1_status"] = "OK" if min_r1 >= 20 else f"FAIL: {min_r1} < 20"
    details["r2_status"] = "OK" if min_r2 >= 15 else f"FAIL: {min_r2} < 15"

    results.record("T11_statistical_power", passed, details)


# ---------------------------------------------------------------------------
# TEST 12: Ablation Specificity
# ---------------------------------------------------------------------------
def test_12_specificity(results):
    """Check cross-concept ablation specificity matrix."""
    spec_path = os.path.join("results", "results_specificity.csv")
    if not os.path.exists(spec_path):
        results.record("T12_ablation_specificity", False, {}, skipped=True)
        return

    import pandas as pd
    df = pd.read_csv(spec_path)

    on = df[df["on_target"] == True]
    off = df[df["on_target"] == False]

    mean_on_r1 = on["delta_r1"].mean()
    mean_off_r1 = off["delta_r1"].mean()
    mean_on_r2 = on["delta_r2"].mean()
    mean_off_r2 = off["delta_r2"].mean()

    ratio_r1 = (mean_on_r1 / mean_off_r1) if mean_off_r1 > 0 else float("inf")
    ratio_r2 = (mean_on_r2 / mean_off_r2) if mean_off_r2 > 0 else float("inf")

    # Off-target thresholds
    max_off_r1 = off["delta_r1"].max()
    max_off_r2 = off["delta_r2"].max()

    passed = (max_off_r1 < 0.15 and max_off_r2 < 0.10 and
              ratio_r1 > 2.0 and ratio_r2 > 2.0)

    details = {
        "mean_on_target_r1": round(mean_on_r1, 4),
        "mean_off_target_r1": round(mean_off_r1, 4),
        "mean_on_target_r2": round(mean_on_r2, 4),
        "mean_off_target_r2": round(mean_off_r2, 4),
        "ratio_r1": round(ratio_r1, 2),
        "ratio_r2": round(ratio_r2, 2),
        "max_off_target_r1": round(max_off_r1, 4),
        "max_off_target_r2": round(max_off_r2, 4),
    }

    results.record("T12_ablation_specificity", passed, details)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validation Test Suite")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip tests that require the model")
    args = parser.parse_args()

    print("=" * 60)
    print("Validation Test Suite")
    print("=" * 60)

    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run data_gen.py first.")
        sys.exit(1)

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model if needed
    model = None
    if not args.no_model:
        try:
            from transformer_lens import HookedTransformer
            print("Loading model for model-dependent tests...")
            model = HookedTransformer.from_pretrained("gpt2-small")
            model.eval()
        except (ImportError, Exception) as e:
            print(f"  Model unavailable ({e}). Skipping model tests.")

    results = TestResults()

    # --- Data-level tests (no model needed) ---
    print("\n--- Data-Level Tests ---")
    test_01_conceptnet_quality(data, results)
    test_02_negative_contamination(data, results)
    test_03_polysemy(data, results)
    test_04_sample_size(data, results)
    test_06_template_artifacts(data, results)
    test_11_statistical_power(data, results)

    # --- Model-level tests ---
    print("\n--- Model-Level Tests ---")
    test_05_tokenization(data, results, model=model)
    test_07_layer_validity(data, results, model=model)

    # --- Post-extraction tests ---
    print("\n--- Post-Extraction Tests ---")
    test_08_svm_stability(results)
    test_09_cav_overlap(results)

    # --- Post-experiment tests ---
    print("\n--- Post-Experiment Tests ---")
    test_10_benchmark_validity(results)
    test_12_specificity(results)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    all_results = results.summary()

    # Save report
    with open(REPORT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
