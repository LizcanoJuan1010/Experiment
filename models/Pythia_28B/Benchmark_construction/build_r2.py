"""
R2 Benchmark Builder — Ground Truth Paraphrase Pairs from ConceptNet Synonym
=============================================================================
Generates sentence pairs where synonym words (verified via ConceptNet Synonym
relation) are embedded in syntactically distinct but semantically equivalent
templates.

Tiered verification filter (priority order):
  Tier 1 (strongest): Both words are IsA hyponyms of the concept
  Tier 2 (strong):    At least one word is an IsA hyponym, the other is in
                      the concept's semantic field (all_nodes)

  FIX-A4: Tier 3 (both in all_nodes, no IsA) is EXCLUDED.
  all_nodes membership via RelatedTo/AtLocation/UsedFor does not guarantee
  semantic category membership, producing unreliable synonym pairs.

All tiers use exclusively ConceptNet 5.7 as ground truth source.
The Synonym relation itself and each word's concept membership are
independently verified.

Output: Benchmark_construction/output/r2_benchmark.json
Compatible with metrics.py evaluate_r2() — list of (s1, s2) pairs.

Usage:
    python -m Benchmark_construction.build_r2
"""

import json
import os
import sys

import spacy

from Benchmark_construction.config import (
    BLACKLIST,
    CONCEPTS,
    CONCEPTNET_FILE,
    MIN_CONCEPTNET_WEIGHT,
    OUTPUT_DIR,
    R2_TEMPLATES,
    TARGET_R2_PER_CONCEPT,
)


def load_conceptnet_data():
    """Load conceptnet_concepts.json and return parsed dict."""
    if not os.path.exists(CONCEPTNET_FILE):
        print(f"ERROR: {CONCEPTNET_FILE} not found.")
        print("Run: python conceptnet_extractor.py")
        sys.exit(1)
    with open(CONCEPTNET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


nlp = spacy.load("en_core_web_sm")


def normalize_word(word):
    """Normalize a word to lowercase stripped form for consistent comparison."""
    return word.strip().lower()


def is_valid_noun(word):
    """Check if a word is a noun (NOUN or PROPN) using spaCy POS tagging."""
    doc = nlp(word)
    return doc[0].pos_ in ("NOUN", "PROPN")


def get_hyponym_set(concept_data):
    """Get set of hyponym words from concept data (normalized to lowercase)."""
    raw = concept_data.get("hyponyms", [])
    if not raw:
        return set()
    if isinstance(raw[0], dict):
        return {normalize_word(entry["word"]) for entry in raw
                if entry.get("weight", 0) >= MIN_CONCEPTNET_WEIGHT}
    return {normalize_word(w) for w in raw}


def get_all_nodes_set(concept_data):
    """Get set of all concept-related words (normalized to lowercase)."""
    return {normalize_word(w) for w in concept_data.get("all_nodes", [])}


def get_synonym_pairs(concept_data):
    """Extract synonym pairs with weights (words normalized to lowercase)."""
    raw = concept_data.get("synonyms", [])
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [
            (normalize_word(entry["word_1"]),
             normalize_word(entry["word_2"]),
             entry.get("weight", 1.0))
            for entry in raw
        ]
    return [(normalize_word(pair[0]), normalize_word(pair[1]), 1.0)
            for pair in raw if len(pair) == 2]


def filter_quality(word):
    """Apply quality filters to synonym words."""
    parts = word.strip().split()
    if len(parts) > 2:
        return False
    if len(word) < 2 or len(word) > 30:
        return False
    if word.replace(" ", "").isdigit():
        return False
    return True


def deduplicate_pairs(pairs):
    """Remove symmetric duplicates: (a,b) and (b,a) are the same pair."""
    seen = set()
    unique = []
    for entry in pairs:
        w1, w2 = entry["word_1"], entry["word_2"]
        key = tuple(sorted([w1, w2]))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def build_r2_items(concept_name, verified_pairs):
    """Generate R2 sentence pairs for one concept."""
    n_templates = len(R2_TEMPLATES)
    items = []

    for i, pair_info in enumerate(verified_pairs[:TARGET_R2_PER_CONCEPT]):
        template_s1, template_s2 = R2_TEMPLATES[i % n_templates]

        w1 = pair_info["word_1"]
        w2 = pair_info["word_2"]

        s1 = template_s1.format(w1=w1)
        s2 = template_s2.format(w2=w2)

        item = {
            "sentences": [s1, s2],
            "provenance": {
                "word_1": w1,
                "word_2": w2,
                "relation": "Synonym",
                "concept": concept_name,
                "conceptnet_weight": pair_info["weight"],
                "source": "ConceptNet 5.7",
                "verification_tier": pair_info["tier"],
                "w1_isa_verified": pair_info["w1_isa"],
                "w2_isa_verified": pair_info["w2_isa"],
            },
        }
        items.append(item)

    return items


def main():
    print("=" * 60)
    print("R2 Benchmark Builder (Synonym Pairs from ConceptNet)")
    print("=" * 60)

    data = load_conceptnet_data()

    benchmark = {}
    all_items = []

    for concept in CONCEPTS:
        if concept not in data:
            print(f"WARNING: concept '{concept}' not found in {CONCEPTNET_FILE}")
            benchmark[concept] = []
            continue

        concept_data = data[concept]

        # Get verification sets
        hyponym_set = get_hyponym_set(concept_data)
        all_nodes_set = get_all_nodes_set(concept_data)
        print(f"\n  {concept}: {len(hyponym_set)} hyponyms (IsA), "
              f"{len(all_nodes_set)} all_nodes")

        # Get all synonym pairs
        all_synonyms = get_synonym_pairs(concept_data)
        print(f"  {concept}: {len(all_synonyms)} raw synonym pairs")

        # Tiered filtering
        tier1_pairs = []  # Both words are IsA hyponyms
        tier2_pairs = []  # At least one word is IsA hyponym, both in all_nodes
        tier3_pairs = []  # Both words in all_nodes (weakest)

        blacklisted = {normalize_word(b) for b in BLACKLIST.get(concept, set())}
        n_pos_removed = 0
        n_bl_removed = 0
        for w1, w2, wt in all_synonyms:
            if wt < MIN_CONCEPTNET_WEIGHT:
                continue
            if not (filter_quality(w1) and filter_quality(w2)):
                continue
            # Both words must be nouns
            if not (is_valid_noun(w1) and is_valid_noun(w2)):
                n_pos_removed += 1
                continue
            # Skip if either word is blacklisted
            if w1 in blacklisted or w2 in blacklisted:
                n_bl_removed += 1
                continue

            w1_isa = w1 in hyponym_set
            w2_isa = w2 in hyponym_set
            w1_node = w1 in all_nodes_set
            w2_node = w2 in all_nodes_set

            if w1_isa and w2_isa:
                tier1_pairs.append({
                    "word_1": w1, "word_2": w2, "weight": wt,
                    "tier": 1, "w1_isa": True, "w2_isa": True,
                })
            elif (w1_isa or w2_isa) and w1_node and w2_node:
                tier2_pairs.append({
                    "word_1": w1, "word_2": w2, "weight": wt,
                    "tier": 2, "w1_isa": w1_isa, "w2_isa": w2_isa,
                })
            elif w1_node and w2_node:
                tier3_pairs.append({
                    "word_1": w1, "word_2": w2, "weight": wt,
                    "tier": 3, "w1_isa": False, "w2_isa": False,
                })

        print(f"  {concept}: {n_pos_removed} non-noun, {n_bl_removed} blacklisted")
        print(f"  {concept}: Tier 1 (both IsA) = {len(tier1_pairs)}")
        print(f"  {concept}: Tier 2 (one IsA + both nodes) = {len(tier2_pairs)}")
        print(f"  {concept}: Tier 3 (both in all_nodes) = {len(tier3_pairs)}")

        # FIX-A4: Merge Tier 1 and Tier 2 ONLY. Tier 3 is excluded because
        # all_nodes membership (RelatedTo, AtLocation, UsedFor) does not
        # guarantee semantic category membership — it generates pairs where
        # neither word is a verified hyponym, producing unreliable synonyms.
        merged = tier1_pairs + tier2_pairs

        if len(merged) < TARGET_R2_PER_CONCEPT and tier3_pairs:
            print(f"  WARNING: {concept} has only {len(merged)} Tier 1+2 pairs "
                  f"(target: {TARGET_R2_PER_CONCEPT}). "
                  f"{len(tier3_pairs)} Tier 3 pairs available but EXCLUDED "
                  f"(no IsA verification).")

        # Deduplicate
        merged = deduplicate_pairs(merged)

        # Sort: Tier 1 first, then by weight within each tier
        merged.sort(key=lambda x: (-x["tier"], -x["weight"]))
        # Actually sort Tier 1 first (tier=1 < tier=2, so sort ascending for tier)
        merged.sort(key=lambda x: (x["tier"], -x["weight"]))

        print(f"  {concept}: {len(merged)} after dedup (using up to "
              f"{TARGET_R2_PER_CONCEPT})")

        if len(merged) < TARGET_R2_PER_CONCEPT:
            print(f"  WARNING: {concept} has only {len(merged)} verified pairs "
                  f"(target: {TARGET_R2_PER_CONCEPT})")

        # Generate items
        items = build_r2_items(concept, merged)
        benchmark[concept] = items
        all_items.extend(items)

        # Count tiers used
        n_t1 = sum(1 for it in items
                    if it["provenance"]["verification_tier"] == 1)
        n_t2 = sum(1 for it in items
                    if it["provenance"]["verification_tier"] == 2)
        n_t3 = sum(1 for it in items
                    if it["provenance"]["verification_tier"] == 3)
        print(f"  {concept}: {len(items)} items "
              f"({n_t1} Tier 1, {n_t2} Tier 2, {n_t3} Tier 3)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "r2_benchmark.json")

    output = {
        "metadata": {
            "benchmark": "R2_Paraphrase_Similarity",
            "source": "ConceptNet 5.7 (Speer et al., 2017)",
            "relation": "Synonym",
            "verification_tiers": {
                "tier_1": "Both words verified as IsA hyponyms of the concept",
                "tier_2": "At least one word is an IsA hyponym, both in "
                          "the concept's semantic field",
                "tier_3": "EXCLUDED (FIX-A4): no IsA verification, unreliable",
            },
            "ground_truth": "Each pair has: (1) verified Synonym edge in "
                            "ConceptNet, (2) both words verified as belonging "
                            "to the concept's semantic field via ConceptNet "
                            "relation with weight >= {:.1f}".format(
                                MIN_CONCEPTNET_WEIGHT),
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in CONCEPTS},
            "templates_used": len(R2_TEMPLATES),
        },
        "items": benchmark,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved R2 benchmark to {output_path}")
    print(f"Total items: {len(all_items)}")
    print("Done.")


if __name__ == "__main__":
    main()
