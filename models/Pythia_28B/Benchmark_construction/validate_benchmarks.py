"""
Benchmark Validator
====================
Runs rigorous checks on the generated R1 and R2 benchmarks to ensure:

1. Ground Truth: Every item is traceable to a verified ConceptNet relation.
2. Isomorfismo Funcional: The benchmark structure mirrors the cognitive
   function it measures (hypernymy for R1, semantic equivalence for R2).
3. Quality: Balance, diversity, mutual exclusivity, and statistical soundness.

Output: Benchmark_construction/output/provenance_report.json

Usage:
    python -m Benchmark_construction.validate_benchmarks
"""

import json
import os
import sys
from collections import Counter

from Benchmark_construction.config import (
    CONCEPTS,
    CONCEPTNET_FILE,
    HYPERNYM_LABELS,
    MIN_CONCEPTNET_WEIGHT,
    OUTPUT_DIR,
    R1_DISTRACTOR_CATEGORIES,
    R1_TEMPLATES,
    R2_TEMPLATES,
    TARGET_R1_PER_CONCEPT,
    TARGET_R2_PER_CONCEPT,
)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check(name, passed, detail=""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    return {"test": name, "status": status, "detail": detail}


def normalize_word(word):
    """Normalize a word to lowercase stripped form for consistent comparison."""
    return word.strip().lower()


# ---------------------------------------------------------------------------
# R1 Validation
# ---------------------------------------------------------------------------
def validate_r1(r1_data, conceptnet_data):
    """Validate R1 benchmark against ground truth and quality criteria."""
    results = []
    items_by_concept = r1_data.get("items", {})

    # --- V1: Every item has provenance ---
    all_items = []
    missing_provenance = 0
    for concept in CONCEPTS:
        concept_items = items_by_concept.get(concept, [])
        all_items.extend(concept_items)
        for item in concept_items:
            if "provenance" not in item:
                missing_provenance += 1

    results.append(check(
        "R1_V01_Provenance",
        missing_provenance == 0,
        f"{missing_provenance} items without provenance (of {len(all_items)})"
    ))

    # --- V2: Every word has verified IsA in ConceptNet ---
    not_verified = []
    for concept in CONCEPTS:
        cn = conceptnet_data.get(concept, {})
        hyponyms_raw = cn.get("hyponyms", [])
        if hyponyms_raw and isinstance(hyponyms_raw[0], dict):
            hyponym_words = {normalize_word(e["word"]) for e in hyponyms_raw}
        else:
            hyponym_words = {normalize_word(w) for w in hyponyms_raw}

        for item in items_by_concept.get(concept, []):
            word = normalize_word(item.get("provenance", {}).get("word", ""))
            if word and word not in hyponym_words:
                not_verified.append((concept, word))

    results.append(check(
        "R1_V02_IsA_Verified",
        len(not_verified) == 0,
        f"{len(not_verified)} words not found in ConceptNet hyponyms: "
        f"{not_verified[:5]}"
    ))

    # --- V3: Mutual exclusivity — no word in 2+ concepts ---
    word_to_concepts = {}
    for concept in CONCEPTS:
        for item in items_by_concept.get(concept, []):
            word = normalize_word(item.get("provenance", {}).get("word", ""))
            if word:
                word_to_concepts.setdefault(word, set()).add(concept)

    overlapping = {w: cs for w, cs in word_to_concepts.items() if len(cs) > 1}
    results.append(check(
        "R1_V03_Mutual_Exclusivity",
        len(overlapping) == 0,
        f"{len(overlapping)} words in multiple concepts: "
        f"{dict(list(overlapping.items())[:5])}"
    ))

    # --- V4: Item count per concept ---
    for concept in CONCEPTS:
        n = len(items_by_concept.get(concept, []))
        results.append(check(
            f"R1_V04_Count_{concept}",
            n >= TARGET_R1_PER_CONCEPT,
            f"{concept}: {n} items (target: {TARGET_R1_PER_CONCEPT})"
        ))

    # --- V5: Correct position balance (chi-squared-like check) ---
    for concept in CONCEPTS:
        positions = Counter()
        concept_items = items_by_concept.get(concept, [])
        for item in concept_items:
            pos = item.get("options", {}).get("correct_position", -1)
            positions[pos] += 1

        n_items = len(concept_items)
        if n_items > 0:
            expected = n_items / 4.0
            max_deviation = max(
                abs(positions.get(p, 0) - expected) / expected
                for p in range(4)
            ) if expected > 0 else 0
            # Allow up to 10% deviation from uniform
            balanced = max_deviation <= 0.10
        else:
            balanced = False
            max_deviation = float("inf")

        results.append(check(
            f"R1_V05_Position_Balance_{concept}",
            balanced,
            f"{concept}: position counts={dict(positions)}, "
            f"max deviation={max_deviation:.1%}"
        ))

    # --- V6: Template count ---
    n_templates = len(R1_TEMPLATES)
    results.append(check(
        "R1_V06_Templates",
        n_templates >= 1,
        f"{n_templates} template(s) used"
    ))

    # --- V7: Typicality distribution (at least 20% atypical) ---
    for concept in CONCEPTS:
        concept_items = items_by_concept.get(concept, [])
        n_atypical = sum(
            1 for item in concept_items
            if item.get("provenance", {}).get("typicality") == "atypical"
        )
        n_items = len(concept_items)
        pct_atypical = n_atypical / n_items if n_items > 0 else 0

        results.append(check(
            f"R1_V07_Typicality_{concept}",
            pct_atypical >= 0.20 or n_items < 10,
            f"{concept}: {n_atypical}/{n_items} atypical ({pct_atypical:.1%})"
        ))

    # --- V8: Distractors are taxonomically disjoint ---
    for concept in CONCEPTS:
        label = HYPERNYM_LABELS[concept]
        bad_distractors = 0
        concept_items = items_by_concept.get(concept, [])
        for item in concept_items:
            dists = item.get("options", {}).get("distractors", [])
            if label in dists:
                bad_distractors += 1

        results.append(check(
            f"R1_V08_Distractors_{concept}",
            bad_distractors == 0,
            f"{concept}: {bad_distractors} items with correct answer in distractors"
        ))

    # --- V9: ConceptNet weight threshold ---
    below_threshold = 0
    for concept in CONCEPTS:
        for item in items_by_concept.get(concept, []):
            wt = item.get("provenance", {}).get("conceptnet_weight", 0)
            if wt < MIN_CONCEPTNET_WEIGHT:
                below_threshold += 1

    results.append(check(
        "R1_V09_Weight_Threshold",
        below_threshold == 0,
        f"{below_threshold} items below weight threshold {MIN_CONCEPTNET_WEIGHT}"
    ))

    return results


# ---------------------------------------------------------------------------
# R2 Validation
# ---------------------------------------------------------------------------
def validate_r2(r2_data, conceptnet_data):
    """Validate R2 benchmark against ground truth and quality criteria."""
    results = []
    items_by_concept = r2_data.get("items", {})

    # --- V1: Every item has provenance ---
    all_items = []
    missing_provenance = 0
    for concept in CONCEPTS:
        concept_items = items_by_concept.get(concept, [])
        all_items.extend(concept_items)
        for item in concept_items:
            if "provenance" not in item:
                missing_provenance += 1

    results.append(check(
        "R2_V01_Provenance",
        missing_provenance == 0,
        f"{missing_provenance} items without provenance (of {len(all_items)})"
    ))

    # --- V2: Both words are in concept's semantic field (all_nodes) ---
    not_in_field = []
    for concept in CONCEPTS:
        cn = conceptnet_data.get(concept, {})
        all_nodes = {normalize_word(w) for w in cn.get("all_nodes", [])}

        for item in items_by_concept.get(concept, []):
            prov = item.get("provenance", {})
            w1 = normalize_word(prov.get("word_1", ""))
            w2 = normalize_word(prov.get("word_2", ""))
            if w1 and w1 not in all_nodes:
                not_in_field.append((concept, w1, "word_1"))
            if w2 and w2 not in all_nodes:
                not_in_field.append((concept, w2, "word_2"))

    results.append(check(
        "R2_V02_Both_In_Semantic_Field",
        len(not_in_field) == 0,
        f"{len(not_in_field)} words not in concept's semantic field: "
        f"{not_in_field[:5]}"
    ))

    # --- V2b: Tier distribution ---
    for concept in CONCEPTS:
        concept_items = items_by_concept.get(concept, [])
        n_t1 = sum(1 for it in concept_items
                    if it.get("provenance", {}).get("verification_tier") == 1)
        n_t2 = sum(1 for it in concept_items
                    if it.get("provenance", {}).get("verification_tier") == 2)
        n_t3 = sum(1 for it in concept_items
                    if it.get("provenance", {}).get("verification_tier") == 3)
        n_items = len(concept_items)
        results.append(check(
            f"R2_V02b_Tier_Distribution_{concept}",
            n_items > 0,
            f"{concept}: {n_t1} Tier 1, {n_t2} Tier 2, {n_t3} Tier 3 "
            f"(total: {n_items})"
        ))

    # --- V3: Synonym relation verified ---
    # (by construction, all pairs come from ConceptNet /r/Synonym,
    #  but we check provenance records it)
    missing_relation = 0
    for concept in CONCEPTS:
        for item in items_by_concept.get(concept, []):
            rel = item.get("provenance", {}).get("relation", "")
            if rel != "Synonym":
                missing_relation += 1

    results.append(check(
        "R2_V03_Synonym_Relation",
        missing_relation == 0,
        f"{missing_relation} items without Synonym relation in provenance"
    ))

    # --- V4: Item count per concept ---
    for concept in CONCEPTS:
        n = len(items_by_concept.get(concept, []))
        results.append(check(
            f"R2_V04_Count_{concept}",
            n >= TARGET_R2_PER_CONCEPT,
            f"{concept}: {n} items (target: {TARGET_R2_PER_CONCEPT})"
        ))

    # --- V5: Sentences are syntactically distinct ---
    identical_pairs = 0
    for concept in CONCEPTS:
        for item in items_by_concept.get(concept, []):
            sents = item.get("sentences", [])
            if len(sents) == 2 and sents[0] == sents[1]:
                identical_pairs += 1

    results.append(check(
        "R2_V05_Syntactic_Diversity",
        identical_pairs == 0,
        f"{identical_pairs} sentence pairs are identical"
    ))

    # --- V6: No duplicate pairs (per concept) ---
    total_duplicates = 0
    for concept in CONCEPTS:
        seen_pairs = set()
        concept_dups = 0
        for item in items_by_concept.get(concept, []):
            prov = item.get("provenance", {})
            pair_key = tuple(sorted([prov.get("word_1", ""),
                                      prov.get("word_2", "")]))
            if pair_key in seen_pairs:
                concept_dups += 1
            seen_pairs.add(pair_key)
        total_duplicates += concept_dups

    results.append(check(
        "R2_V06_No_Duplicates",
        total_duplicates == 0,
        f"{total_duplicates} duplicate pairs within concepts"
    ))

    # --- V7: Weight threshold ---
    below_threshold = 0
    for concept in CONCEPTS:
        for item in items_by_concept.get(concept, []):
            wt = item.get("provenance", {}).get("conceptnet_weight", 0)
            if wt < MIN_CONCEPTNET_WEIGHT:
                below_threshold += 1

    results.append(check(
        "R2_V07_Weight_Threshold",
        below_threshold == 0,
        f"{below_threshold} items below weight threshold {MIN_CONCEPTNET_WEIGHT}"
    ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Benchmark Validator")
    print("=" * 60)

    # Load ConceptNet data for ground truth verification
    conceptnet_data = load_json(CONCEPTNET_FILE)
    if conceptnet_data is None:
        print(f"ERROR: {CONCEPTNET_FILE} not found.")
        sys.exit(1)

    # Load benchmarks
    r1_path = os.path.join(OUTPUT_DIR, "r1_benchmark.json")
    r2_path = os.path.join(OUTPUT_DIR, "r2_benchmark.json")

    r1_data = load_json(r1_path)
    r2_data = load_json(r2_path)

    all_results = []

    # Validate R1
    if r1_data:
        print("\n--- R1 Validation ---")
        r1_results = validate_r1(r1_data, conceptnet_data)
        all_results.extend(r1_results)
        for r in r1_results:
            print(f"  [{r['status']}] {r['test']}: {r['detail']}")
    else:
        print(f"\nWARNING: {r1_path} not found, skipping R1 validation.")

    # Validate R2
    if r2_data:
        print("\n--- R2 Validation ---")
        r2_results = validate_r2(r2_data, conceptnet_data)
        all_results.extend(r2_results)
        for r in r2_results:
            print(f"  [{r['status']}] {r['test']}: {r['detail']}")
    else:
        print(f"\nWARNING: {r2_path} not found, skipping R2 validation.")

    # Summary
    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_fail = sum(1 for r in all_results if r["status"] == "FAIL")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL (of {len(all_results)} checks)")
    print(f"{'=' * 60}")

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "provenance_report.json")

    report = {
        "summary": {
            "total_checks": len(all_results),
            "passed": n_pass,
            "failed": n_fail,
        },
        "checks": all_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved provenance report to {report_path}")

    if n_fail > 0:
        print("\nFailed checks:")
        for r in all_results:
            if r["status"] == "FAIL":
                print(f"  - {r['test']}: {r['detail']}")
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
