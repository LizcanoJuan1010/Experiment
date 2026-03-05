"""
BEA Odd-One-Out Benchmark — Detección de Intruso Semántico
===========================================================
Simulates the semantic intruder detection task used in aphasia assessment
batteries (Caramazza & Shelton, 1998). The patient is given a set of
words from one category plus one intruder from a different category and
must identify which word does not belong.

This test specifically probes category boundary knowledge. In semantic
aphasia, patients struggle to maintain clear category boundaries,
especially when the intruder is from a semantically adjacent category.

Design choices for clinical relevance:
  - "Easy" items: intruder from a distant category (e.g., tool among time words)
  - "Hard" items: intruder from an adjacent/related category (e.g., place among
    tools — since tools are "used in" places)

Output format is compatible with evaluate_r1() in metrics.py.

Output: Benchmark_construction/output/bea_oddoneout_benchmark.json

Usage:
    python -m Benchmark_construction.build_bea_oddoneout

References:
    - Caramazza & Shelton (1998) "Domain-specific knowledge systems in the brain"
    - Jefferies & Lambon Ralph (2006) "Semantic aphasia"
    - Cuetos & González-Nosti (2009) "Batería de Evaluación de la Afasia"
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Concept word pools (subset of CURATED_NODES from data_gen.py)
# ---------------------------------------------------------------------------
# Using prototypical, single-word items for clean evaluation.

CONCEPT_WORDS = {
    "time": [
        "hour", "minute", "second", "day", "week", "month", "year",
        "decade", "century", "moment", "instant", "duration", "period",
        "interval", "clock", "calendar", "schedule", "deadline",
        "noon", "midnight", "dawn", "dusk", "morning", "afternoon",
        "evening", "night", "season", "weekend", "epoch", "era",
    ],
    "place": [
        "city", "town", "village", "country", "island", "mountain",
        "valley", "river", "lake", "ocean", "forest", "desert",
        "park", "garden", "room", "house", "building", "street",
        "road", "bridge", "airport", "station", "hospital", "school",
        "church", "museum", "library", "office", "kitchen", "bedroom",
    ],
    "tools": [
        "hammer", "screwdriver", "wrench", "pliers", "saw", "drill",
        "chisel", "clamp", "vise", "level", "ruler", "scissors",
        "knife", "axe", "shovel", "rake", "hoe", "trowel", "brush",
        "sandpaper", "ladder", "crowbar", "mallet", "anvil", "awl",
        "lathe", "wedge", "bolt", "screw", "nail",
    ],
}

# Adjacent category pairs — intruders from adjacent categories are harder
# because there is natural semantic overlap in how they co-occur in language.
ADJACENT_CATEGORIES = {
    "time": ["place"],       # "dawn at the mountain" — temporal-spatial overlap
    "place": ["tools"],      # "workshop", "garage" — places where tools are used
    "tools": ["time"],       # "clock", "stopwatch" — time-measuring instruments
}

# Distant categories — intruders from distant categories are easy controls
DISTANT_CATEGORIES = {
    "time": ["tools"],
    "place": ["time"],
    "tools": ["place"],
}

# ---------------------------------------------------------------------------
# Odd-one-out prompt templates
# ---------------------------------------------------------------------------
ODDONEOUT_TEMPLATES = [
    "Which word does not belong with the others: {items}.\nThe word that does not belong is",
    "One of these words does not fit the group: {items}.\nThe odd one out is",
    "Three of these words share a category, but one does not: {items}.\nThe intruder is",
    "Identify the word that is not in the same category as the others: {items}.\nThe answer is",
    "All but one of these words belong together: {items}.\nThe one that does not belong is",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_PER_CONCEPT = 30     # Total items per concept (15 easy + 15 hard)
N_SAME_CATEGORY = 3         # Number of same-category words per item


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_oddoneout_items(target_concept, all_concepts):
    """Generate BEA odd-one-out items for one concept.

    Each item has:
      - 3 words from the target concept + 1 intruder from another concept
      - The correct answer is the intruder
      - Distractors are the 3 same-category words

    Generates two difficulty levels:
      - "hard": intruder from an adjacent category (semantic overlap)
      - "easy": intruder from a distant category (clear boundary)
    """
    rng = random.Random(42 + hash(target_concept))
    n_templates = len(ODDONEOUT_TEMPLATES)

    target_words = list(CONCEPT_WORDS[target_concept])
    adjacent_cats = ADJACENT_CATEGORIES[target_concept]
    distant_cats = DISTANT_CATEGORIES[target_concept]

    items = []
    half = TARGET_PER_CONCEPT // 2

    for difficulty, source_cats, n_items in [
        ("hard", adjacent_cats, half),
        ("easy", distant_cats, TARGET_PER_CONCEPT - half),
    ]:
        for i in range(n_items):
            # Pick 3 same-category words (no repetition across items)
            same_words = rng.sample(target_words, N_SAME_CATEGORY)

            # Pick 1 intruder from source category
            source_cat = rng.choice(source_cats)
            intruder_pool = CONCEPT_WORDS[source_cat]
            intruder = rng.choice(intruder_pool)

            # Shuffle the 4 words for presentation
            all_words = list(same_words) + [intruder]
            rng.shuffle(all_words)

            # Format items list as comma-separated string
            items_str = ", ".join(all_words)

            # Select template
            template = ODDONEOUT_TEMPLATES[
                (len(items)) % n_templates
            ]
            question = template.format(items=items_str)

            items.append({
                "question": question,
                "options": {
                    "correct": intruder,
                    "distractors": same_words,
                },
                "provenance": {
                    "target_concept": target_concept,
                    "same_category_words": same_words,
                    "intruder": intruder,
                    "intruder_source": source_cat,
                    "difficulty": difficulty,
                    "test_type": "bea_oddoneout",
                    "clinical_analog": "Detección de intruso semántico (BEA)",
                },
            })

    return items


def main():
    print("=" * 60)
    print("BEA Odd-One-Out Benchmark (Detección de Intruso Semántico)")
    print("=" * 60)

    all_concepts = list(CONCEPT_WORDS.keys())
    benchmark = {}
    all_items = []

    for concept in all_concepts:
        print(f"\n  {concept.upper()}: {len(CONCEPT_WORDS[concept])} words in pool")

        generated = build_oddoneout_items(concept, all_concepts)
        benchmark[concept] = generated
        all_items.extend(generated)

        n_hard = sum(1 for it in generated
                     if it["provenance"]["difficulty"] == "hard")
        n_easy = len(generated) - n_hard
        print(f"  {concept.upper()}: {len(generated)} items "
              f"({n_hard} hard / {n_easy} easy)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "bea_oddoneout_benchmark.json")

    output = {
        "metadata": {
            "benchmark": "BEA_OddOneOut",
            "clinical_analog": "Detección de intruso semántico (BEA) / "
                               "Semantic intruder detection",
            "description": (
                "Odd-one-out test: given 4 words (3 from one category + 1 "
                "intruder), identify which word does not belong. Includes "
                "two difficulty levels: 'hard' (intruder from adjacent "
                "category with semantic overlap) and 'easy' (intruder from "
                "distant category with clear boundary)."
            ),
            "format": "Compatible with evaluate_r1() in metrics.py",
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in benchmark},
            "difficulty_levels": {
                "hard": "Intruder from semantically adjacent category",
                "easy": "Intruder from semantically distant category",
            },
            "references": [
                "Caramazza & Shelton (1998) Domain-specific knowledge",
                "Jefferies & Lambon Ralph (2006) Semantic aphasia",
                "Cuetos & González-Nosti (2009) BEA",
            ],
        },
        "items": benchmark,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"Total items: {len(all_items)}")
    print("Done.")


if __name__ == "__main__":
    main()
