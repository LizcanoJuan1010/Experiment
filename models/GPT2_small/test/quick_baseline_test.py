"""
Quick baseline test — evaluates BEA association accuracy WITHOUT ablation.
Tests only 5 concepts with 5 items each for fast validation.

Usage: python test/quick_baseline_test.py
"""
import json
import os
import sys

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
from metrics import evaluate_bea_association

BENCHMARK_PATH = os.path.join(
    _parent, "Benchmark_construction", "output", "bea_association_benchmark.json",
)

TEST_CONCEPTS = ["camel", "bee", "doctor", "violin", "king"]
ITEMS_PER_CONCEPT = 5  # Only test 5 items per concept for speed


def main():
    from transformer_lens import HookedTransformer

    print("Loading benchmark...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    items_by_concept = benchmark["items"]

    print(f"Loading model: {cfg.MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(cfg.MODEL_NAME)
    model.eval()

    print(f"\n{'='*50}")
    print("QUICK BASELINE TEST (new sentence log-prob scoring)")
    print(f"{'='*50}\n")

    for concept in TEST_CONCEPTS:
        if concept not in items_by_concept:
            print(f"  {concept}: NOT IN BENCHMARK")
            continue

        items = items_by_concept[concept][:ITEMS_PER_CONCEPT]
        acc, per_item = evaluate_bea_association(model, items)

        print(f"\n  {concept.upper()}: accuracy={acc:.3f} ({len(items)} items)")
        for pi in per_item:
            mark = "OK" if pi["is_correct"] else "XX"
            scores_str = "  ".join(
                f"{w}={s:.3f}" for w, s in sorted(
                    pi["scores"].items(), key=lambda x: x[1], reverse=True
                )
            )
            print(f"    [{mark}] correct={pi['correct_answer']:<15s} "
                  f"model={pi['model_answer']:<15s} | {scores_str}")

    print(f"\n{'='*50}")
    print("DONE — compare these accuracies to the ~0.25 baselines from before")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
