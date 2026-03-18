"""
Build Unified CCT Results CSV (Item-Level)
==========================================
Combines CCT results from GPT2-small, Pythia-2.8B, and LLaVA-1.5-7B into a
single CSV with one row per ITEM per condition — includes the question text,
answer options, model scores (logits/losses), and model answer.

Output:
  results/unified_cct_results.csv          — item-level (full detail)
  results/unified_cct_results_agg.csv      — condition-level (aggregated)
"""

import json
import csv
import os
from collections import Counter

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "gpt2_small",
        "family": "text",
        "params_B": 0.124,
        "n_layers": 12,
        "json_path": "models/GPT2_small/results/cct_test_results.json",
        "layer_map": {
            "L6":  (6,  0.50, "single_mid"),
            "L9":  (9,  0.75, "single_late"),
            "L10": (10, 0.83, "single_deep"),
            "all_layers": (-1, 1.0, "all_layers"),
        },
    },
    {
        "name": "pythia_28b",
        "family": "text",
        "params_B": 2.8,
        "n_layers": 32,
        "json_path": "models/Pythia_28B/results/cct_test_results.json",
        "layer_map": {
            "L16": (16, 0.50, "single_mid"),
            "L24": (24, 0.75, "single_late"),
            "L27": (27, 0.84, "single_deep"),
            "all_layers": (-1, 1.0, "all_layers"),
        },
    },
    {
        "name": "llava_15_7b",
        "family": "multimodal",
        "params_B": 7.0,
        "n_layers": 32,
        "json_path": "models/LLaVA_1_5_7B/results/cct_test_results.json",
        "layer_map": {
            "L16": (16, 0.50, "single_mid"),
            "L24": (24, 0.75, "single_late"),
            "L27": (27, 0.84, "single_deep"),
            "all_extraction_layers": (-1, 1.0, "all_layers"),
        },
    },
]

ALL_3_CONCEPTS = {"camel", "bee", "penguin", "violin", "candle"}
LLAVA_UNIQUE = {"golden_retriever", "labrador_retriever", "tabby_cat"}

# ---------------------------------------------------------------------------
# Item-level columns (one row per question per condition)
# ---------------------------------------------------------------------------
ITEM_COLUMNS = [
    # Model info
    "model", "model_family", "model_params_B", "n_layers_total",
    # Experimental condition
    "ablated_concept", "measured_concept", "on_target",
    "cav_method", "layer_condition", "layer_numeric", "layer_depth_frac", "layer_type",
    "alpha", "ablation_mode",
    # Condition-level accuracy
    "condition_accuracy", "baseline_accuracy", "delta_accuracy",
    # Item-level detail
    "item_index",
    "question",
    "correct_answer",
    "model_answer",
    "is_correct",
    "score_correct",       # logit/loss for the correct answer
    "score_model_choice",  # logit/loss for whatever the model chose
    "score_margin",        # score_correct - score_model_choice (negative = wrong)
    "n_options",           # how many answer options
    "options",             # semicolon-separated list of options
    "scores_all",          # semicolon-separated scores (same order as options)
    # Metadata
    "concept_overlap_tier",
    "n_baseline_correct",
]

# Aggregated columns (one row per condition)
AGG_COLUMNS = [
    "model", "model_family", "model_params_B", "n_layers_total",
    "ablated_concept", "measured_concept", "on_target",
    "cav_method", "layer_condition", "layer_numeric", "layer_depth_frac", "layer_type",
    "alpha", "ablation_mode",
    "accuracy", "baseline_accuracy", "delta_accuracy",
    "n_items", "n_correct",
    "concept_overlap_tier", "n_baseline_correct",
]


def get_overlap_tier(concept):
    if concept in ALL_3_CONCEPTS:
        return "all_3"
    elif concept in LLAVA_UNIQUE:
        return "llava_unique"
    return "text_only"


def build_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    item_path = os.path.join(out_dir, "unified_cct_results.csv")
    agg_path = os.path.join(out_dir, "unified_cct_results_agg.csv")

    item_rows = []
    agg_rows = []

    for model_cfg in MODELS:
        json_path = os.path.join(base_dir, model_cfg["json_path"])
        print(f"Loading {model_cfg['name']} from {json_path}...")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Baseline correct counts
        baseline_correct = {}
        for concept, bdata in data.get("baselines", {}).items():
            baseline_correct[concept] = bdata.get("n_correct", bdata.get("n_items", 0))

        results = data["results"]
        print(f"  {len(results)} result entries")

        n_items_written = 0
        for r in results:
            layer_label = r["layer"]
            layer_info = model_cfg["layer_map"].get(layer_label)
            if layer_info is None:
                continue

            layer_numeric, layer_depth_frac, layer_type = layer_info
            ablation_mode = r.get("ablation_mode", "all_tokens")
            ablated = r["ablated_concept"]
            measured = r["measured_concept"]
            overlap_tier = get_overlap_tier(ablated)
            n_bc = baseline_correct.get(ablated, 0)

            # Common fields for this condition
            base = {
                "model": model_cfg["name"],
                "model_family": model_cfg["family"],
                "model_params_B": model_cfg["params_B"],
                "n_layers_total": model_cfg["n_layers"],
                "ablated_concept": ablated,
                "measured_concept": measured,
                "on_target": r["on_target"],
                "cav_method": r["cav_method"],
                "layer_condition": layer_label,
                "layer_numeric": layer_numeric,
                "layer_depth_frac": layer_depth_frac,
                "layer_type": layer_type,
                "alpha": r["alpha"],
                "ablation_mode": ablation_mode,
                "concept_overlap_tier": overlap_tier,
                "n_baseline_correct": n_bc,
            }

            per_item = r.get("per_item", [])

            if per_item:
                # Expand to item-level rows
                for idx, item in enumerate(per_item):
                    scores = item.get("scores", {})
                    correct_ans = item.get("correct_answer", "")
                    model_ans = item.get("model_answer", "")

                    # Sort options alphabetically for consistent ordering
                    options_sorted = sorted(scores.keys())
                    scores_list = [scores[o] for o in options_sorted]

                    score_correct = scores.get(correct_ans, None)
                    score_model = scores.get(model_ans, None)
                    if score_correct is not None and score_model is not None:
                        margin = score_correct - score_model
                    else:
                        margin = None

                    row = dict(base)
                    row.update({
                        "condition_accuracy": r["accuracy"],
                        "baseline_accuracy": r["baseline_accuracy"],
                        "delta_accuracy": r["delta_accuracy"],
                        "item_index": idx,
                        "question": item.get("question", ""),
                        "correct_answer": correct_ans,
                        "model_answer": model_ans,
                        "is_correct": item.get("is_correct", False),
                        "score_correct": round(score_correct, 6) if score_correct is not None else "",
                        "score_model_choice": round(score_model, 6) if score_model is not None else "",
                        "score_margin": round(margin, 6) if margin is not None else "",
                        "n_options": len(options_sorted),
                        "options": ";".join(options_sorted),
                        "scores_all": ";".join(f"{s:.6f}" for s in scores_list),
                    })
                    item_rows.append(row)
                    n_items_written += 1
            else:
                # No per_item data (LLaVA parsed from log) — write one row
                # with item fields empty
                row = dict(base)
                row.update({
                    "condition_accuracy": r["accuracy"],
                    "baseline_accuracy": r["baseline_accuracy"],
                    "delta_accuracy": r["delta_accuracy"],
                    "item_index": "",
                    "question": "",
                    "correct_answer": "",
                    "model_answer": "",
                    "is_correct": "",
                    "score_correct": "",
                    "score_model_choice": "",
                    "score_margin": "",
                    "n_options": "",
                    "options": "",
                    "scores_all": "",
                })
                item_rows.append(row)
                n_items_written += 1

            # Aggregated row (always one per condition)
            agg_row = dict(base)
            n_items = len(per_item) if per_item else ""
            n_correct = sum(1 for it in per_item if it.get("is_correct")) if per_item else ""
            agg_row.update({
                "accuracy": r["accuracy"],
                "baseline_accuracy": r["baseline_accuracy"],
                "delta_accuracy": r["delta_accuracy"],
                "n_items": n_items,
                "n_correct": n_correct,
            })
            agg_rows.append(agg_row)

        print(f"  {n_items_written} item-level rows written")

    # Write item-level CSV
    print(f"\nWriting {len(item_rows)} item-level rows to {item_path}...")
    with open(item_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITEM_COLUMNS)
        writer.writeheader()
        writer.writerows(item_rows)

    # Write aggregated CSV
    print(f"Writing {len(agg_rows)} condition-level rows to {agg_path}...")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_COLUMNS)
        writer.writeheader()
        writer.writerows(agg_rows)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Item-level CSV:      {item_path}")
    print(f"    Rows: {len(item_rows)}, Columns: {len(ITEM_COLUMNS)}")
    print(f"  Aggregated CSV:      {agg_path}")
    print(f"    Rows: {len(agg_rows)}, Columns: {len(AGG_COLUMNS)}")
    print(f"{'='*60}")

    model_counts = Counter(r["model"] for r in item_rows)
    for m, c in model_counts.items():
        print(f"  {m}: {c} item rows")

    tier_counts = Counter(r["concept_overlap_tier"] for r in item_rows)
    print(f"\n  Overlap tiers (item-level):")
    for t, c in tier_counts.items():
        print(f"    {t}: {c} rows")

    # Verify some item data
    items_with_scores = sum(1 for r in item_rows if r["scores_all"])
    items_with_q = sum(1 for r in item_rows if r["question"])
    print(f"\n  Items with scores: {items_with_scores}")
    print(f"  Items with question text: {items_with_q}")


if __name__ == "__main__":
    build_csv()
