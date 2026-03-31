"""
Build Unified CCT Results CSV (Item-Level)
==========================================
Combines CCT results from GPT2-small, Pythia-2.8B, and LLaVA-1.5-7B into a
single CSV with one row per ITEM per condition — includes the question text,
answer options, model scores (logits/losses), and model answer.

Includes computed columns for inferential statistics: softmax probabilities,
Shannon entropy, rank of correct answer, semantic category mappings, and
boolean 0/1 encodings for regression-ready analysis (GLMMs, ANOVAs, etc.).

Output:
  results/unified_cct_results.csv          — item-level (full detail)
  results/unified_cct_results_agg.csv      — condition-level (aggregated)
"""

import json
import csv
import math
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
# Semantic category mapping (all 33 concepts across all models)
# ---------------------------------------------------------------------------
CONCEPT_CATEGORY = {
    # Animals
    "camel": "animal", "bee": "animal", "horse": "animal", "penguin": "animal",
    "bird": "animal", "spider": "animal", "squirrel": "animal",
    "golden_retriever": "animal", "labrador_retriever": "animal", "tabby_cat": "animal",
    # Objects
    "candle": "object", "lock": "object", "toothbrush": "object",
    # Instruments
    "violin": "instrument",
    # Professions
    "writer": "profession", "baker": "profession", "doctor": "profession",
    "barber": "profession", "photographer": "profession", "tailor": "profession",
    "gardener": "profession", "chef": "profession", "musician": "profession",
    "carpenter": "profession", "dentist": "profession", "fisherman": "profession",
    "painter": "profession", "knight": "profession", "king": "profession",
    "pirate": "profession", "soldier": "profession",
    # Characters
    "snowman": "character",
    # Natural phenomena
    "fire": "natural",
}

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
    # --- Inferential statistics columns ---
    "on_target_bool",              # int 0/1 for regression
    "is_correct_bool",             # int 0/1 for regression (empty for LLaVA)
    "accuracy_drop",               # -delta_accuracy (positive = performance dropped)
    "softmax_correct",             # P(correct answer) via softmax over scores
    "softmax_model_choice",        # P(model's chosen answer) via softmax
    "entropy",                     # Shannon entropy of score distribution (nats)
    "rank_correct",                # rank of correct answer (1=best, lower loss)
    "n_concepts_evaluated",        # how many concepts this model evaluated
    "concept_category",            # semantic category of ablated_concept
    "measured_concept_category",   # semantic category of measured_concept
    "same_category",               # 1 if ablated & measured share category, 0 otherwise
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
    # --- Inferential statistics columns ---
    "on_target_bool",
    "accuracy_drop",
    "n_concepts_evaluated",
    "concept_category",
    "measured_concept_category",
    "same_category",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_overlap_tier(concept):
    if concept in ALL_3_CONCEPTS:
        return "all_3"
    elif concept in LLAVA_UNIQUE:
        return "llava_unique"
    return "text_only"


def get_concept_category(concept):
    cat = CONCEPT_CATEGORY.get(concept, "unknown")
    if cat == "unknown":
        print(f"  WARNING: Unknown concept category for '{concept}'")
    return cat


def compute_softmax(scores_dict):
    """Convert loss scores to probabilities. Lower loss = more likely, so negate first."""
    if not scores_dict:
        return {}
    neg_scores = {k: -v for k, v in scores_dict.items()}
    max_val = max(neg_scores.values())
    exps = {k: math.exp(v - max_val) for k, v in neg_scores.items()}
    total = sum(exps.values())
    if total == 0:
        return {k: 0.0 for k in scores_dict}
    return {k: v / total for k, v in exps.items()}


def compute_entropy(probs_dict):
    """Shannon entropy in nats: H = -sum(p * ln(p)) for p > 0."""
    h = 0.0
    for p in probs_dict.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def compute_rank(scores_dict, target_key):
    """Rank of target_key among options (1=best=lowest loss). Returns None if missing."""
    if target_key not in scores_dict:
        return None
    # Lower loss = more likely = rank 1
    sorted_keys = sorted(scores_dict.keys(), key=lambda k: scores_dict[k])
    return sorted_keys.index(target_key) + 1


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_csv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    item_path = os.path.join(out_dir, "unified_cct_results.csv")
    agg_path = os.path.join(out_dir, "unified_cct_results_agg.csv")

    # Pre-load all JSONs and compute per-model concept counts
    loaded_models = []
    n_concepts_per_model = {}

    for model_cfg in MODELS:
        json_path = os.path.join(base_dir, model_cfg["json_path"])
        print(f"Loading {model_cfg['name']} from {json_path}...")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        concepts_evaluated = set(r["ablated_concept"] for r in data["results"])
        n_concepts_per_model[model_cfg["name"]] = len(concepts_evaluated)
        print(f"  {len(data['results'])} result entries, "
              f"{len(concepts_evaluated)} concepts evaluated")

        loaded_models.append((model_cfg, data))

    item_rows = []
    agg_rows = []

    for model_cfg, data in loaded_models:
        # Baseline correct counts
        baseline_correct = {}
        baselines = data.get("baselines", {})
        for concept, bdata in baselines.items():
            baseline_correct[concept] = bdata.get("n_correct", bdata.get("n_items", 0))

        results = data["results"]
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
            cat_ablated = get_concept_category(ablated)
            cat_measured = get_concept_category(measured)

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
                # Inferential stats — condition-level
                "on_target_bool": 1 if r["on_target"] else 0,
                "accuracy_drop": round(-r["delta_accuracy"], 6),
                "n_concepts_evaluated": n_concepts_per_model[model_cfg["name"]],
                "concept_category": cat_ablated,
                "measured_concept_category": cat_measured,
                "same_category": 1 if cat_ablated == cat_measured else 0,
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

                    # Compute inferential stats
                    probs = compute_softmax(scores)
                    softmax_correct = probs.get(correct_ans)
                    softmax_model = probs.get(model_ans)
                    entropy_val = compute_entropy(probs)
                    rank_val = compute_rank(scores, correct_ans)

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
                        # Inferential stats — item-level
                        "is_correct_bool": 1 if item.get("is_correct", False) else 0,
                        "softmax_correct": round(softmax_correct, 6) if softmax_correct is not None else "",
                        "softmax_model_choice": round(softmax_model, 6) if softmax_model is not None else "",
                        "entropy": round(entropy_val, 6),
                        "rank_correct": rank_val if rank_val is not None else "",
                    })
                    item_rows.append(row)
                    n_items_written += 1
            else:
                # No per_item data (LLaVA) — write one row with item fields empty
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
                    # Inferential stats — empty for LLaVA
                    "is_correct_bool": "",
                    "softmax_correct": "",
                    "softmax_model_choice": "",
                    "entropy": "",
                    "rank_correct": "",
                })
                item_rows.append(row)
                n_items_written += 1

            # Aggregated row (always one per condition)
            agg_row = dict(base)
            if per_item:
                n_items = len(per_item)
                n_correct = sum(1 for it in per_item if it.get("is_correct"))
            else:
                # Recover n_items from baselines for LLaVA
                # Strategy is baseline-correct-only: items tested = baseline correct items
                bdata = baselines.get(measured, {})
                n_items = bdata.get("n_correct", "")
                if n_items != "" and r["accuracy"] is not None:
                    n_correct = round(r["accuracy"] * n_items)
                else:
                    n_correct = ""

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

    # Verify item data
    items_with_scores = sum(1 for r in item_rows if r["scores_all"])
    items_with_q = sum(1 for r in item_rows if r["question"])
    items_with_softmax = sum(1 for r in item_rows if r["softmax_correct"] != "")
    items_with_entropy = sum(1 for r in item_rows if r["entropy"] != "")
    agg_with_nitems = sum(1 for r in agg_rows if r["n_items"] != "")

    print(f"\n  Items with scores: {items_with_scores}")
    print(f"  Items with question text: {items_with_q}")
    print(f"  Items with softmax probs: {items_with_softmax}")
    print(f"  Items with entropy: {items_with_entropy}")
    print(f"  Agg rows with n_items: {agg_with_nitems}/{len(agg_rows)}")


if __name__ == "__main__":
    build_csv()
