"""
BEA Semantic Association Benchmark — Asociación Semántica
=========================================================
Simulates the BEA semantic association subtest (Cuetos & González-Nosti,
2009) and the Pyramids and Palm Trees test (Howard & Patterson, 1992).

In clinical practice, the patient sees a target word/image and must select
which of several options is most functionally or thematically associated.
For example: "hammer" → "nail" (not "screw", "leaf", or "hole").

Key design: distractors are objects associated with OTHER members of the
same semantic category. This forces the model to distinguish fine-grained
functional relationships within a category — exactly what degrades in
semantic aphasia.

Output format is compatible with evaluate_r1() in metrics.py.

Output: Benchmark_construction/output/bea_association_benchmark.json

Usage:
    python -m Benchmark_construction.build_bea_association

References:
    - Cuetos & González-Nosti (2009) "Batería de Evaluación de la Afasia"
    - Howard & Patterson (1992) "Pyramids and Palm Trees Test"
    - Jefferies & Lambon Ralph (2006) "Semantic aphasia"
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
# Curated functional associations per concept
# ---------------------------------------------------------------------------
# Each entry: (target_word, associated_word, relationship_description)
#
# The associated_word is the object/concept most functionally linked to
# the target. Distractors will be associated_words from OTHER entries
# in the same concept — so they are all plausible but only one is correct.

ASSOCIATION_ITEMS = {
    "time": [
        ("clock", "wall", "typically mounted on"),
        ("calendar", "date", "used to track"),
        ("alarm", "morning", "used to wake up in the"),
        ("stopwatch", "race", "used to time a"),
        ("schedule", "meeting", "used to plan a"),
        ("deadline", "project", "marks the end date of a"),
        ("dawn", "sunrise", "occurs at the time of"),
        ("dusk", "sunset", "occurs at the time of"),
        ("noon", "lunch", "the typical meal at"),
        ("midnight", "darkness", "associated with complete"),
        ("morning", "breakfast", "the typical meal of the"),
        ("evening", "dinner", "the typical meal of the"),
        ("night", "sleep", "the time when people"),
        ("weekend", "rest", "typically associated with"),
        ("countdown", "launch", "performed before a rocket"),
        ("season", "weather", "determines the patterns of"),
        ("hour", "minute", "divided into sixty"),
        ("century", "history", "a major unit used in"),
        ("semester", "exam", "typically ends with an"),
        ("decade", "anniversary", "often marked by a special"),
        ("moment", "memory", "often captured as a"),
        ("era", "civilization", "describes the age of a"),
        ("interval", "pause", "represents a temporary"),
        ("duration", "length", "measures the total"),
        ("period", "cycle", "often repeats in a"),
    ],
    "place": [
        ("airport", "airplane", "where you board an"),
        ("hospital", "doctor", "where you find a"),
        ("school", "teacher", "where you find a"),
        ("library", "book", "where you borrow a"),
        ("kitchen", "cooking", "the room used for"),
        ("church", "worship", "the building used for"),
        ("museum", "painting", "where you view a"),
        ("garden", "flowers", "where you grow"),
        ("bridge", "river", "built to cross a"),
        ("desert", "sand", "mostly covered with"),
        ("forest", "trees", "densely covered with"),
        ("mountain", "snow", "the peak is covered with"),
        ("ocean", "waves", "characterized by large"),
        ("lake", "fishing", "commonly used for"),
        ("city", "traffic", "characterized by heavy"),
        ("village", "farm", "typically surrounded by"),
        ("street", "cars", "regularly traveled by"),
        ("park", "bench", "where you sit on a"),
        ("bedroom", "pillow", "where you rest your head on a"),
        ("office", "computer", "where you work at a"),
        ("station", "train", "where you board a"),
        ("building", "elevator", "tall ones contain an"),
        ("island", "beach", "typically surrounded by"),
        ("valley", "creek", "often has a flowing"),
        ("road", "highway", "a long one is called a"),
    ],
    "tools": [
        ("hammer", "nail", "used to drive a"),
        ("screwdriver", "screw", "used to turn a"),
        ("wrench", "bolt", "used to tighten a"),
        ("pliers", "wire", "used to grip and bend"),
        ("saw", "lumber", "used to cut"),
        ("drill", "hole", "used to bore a"),
        ("chisel", "marble", "used to carve"),
        ("scissors", "fabric", "used to cut"),
        ("knife", "bread", "used to slice"),
        ("axe", "firewood", "used to chop"),
        ("shovel", "dirt", "used to dig and move"),
        ("rake", "leaves", "used to gather"),
        ("brush", "paint", "used to apply"),
        ("sandpaper", "surface", "used to smooth a rough"),
        ("ladder", "roof", "used to climb up to a"),
        ("crowbar", "crate", "used to pry open a"),
        ("mallet", "stake", "used to pound a"),
        ("level", "shelf", "used to make sure a surface is flat for a"),
        ("ruler", "paper", "used to draw straight lines on"),
        ("clamp", "glue", "used to hold pieces together while the dries"),
        ("trowel", "cement", "used to spread"),
        ("hoe", "soil", "used to break up"),
        ("file", "edge", "used to smooth a rough"),
        ("wheelbarrow", "gravel", "used to carry heavy"),
        ("hacksaw", "pipe", "used to cut a metal"),
    ],
}

# ---------------------------------------------------------------------------
# Association prompt templates
# ---------------------------------------------------------------------------
ASSOCIATION_TEMPLATES = [
    "Which of the following is most commonly associated with a {word}? The answer is",
    "The object most frequently used together with a {word} is",
    "A {word} is most closely related in everyday use to",
    "When you think of a {word}, the first thing that comes to mind is",
    "In typical use, a {word} is paired with",
]

# ---------------------------------------------------------------------------
# Target items per concept
# ---------------------------------------------------------------------------
TARGET_PER_CONCEPT = 25


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_association_items(concept, items):
    """Generate BEA semantic association MCQ items for one concept.

    Each item has:
      - A target word and a prompt asking what it's associated with
      - The correct functionally associated word
      - 3 distractors that are associated with OTHER items in the same concept

    This mimics the Pyramids and Palm Trees test / BEA semantic association
    where the patient must identify correct functional relationships.
    """
    rng = random.Random(42)
    n_templates = len(ASSOCIATION_TEMPLATES)

    # Build pool of all associated words for distractors
    all_associations = [(target, assoc) for target, assoc, _ in items]

    benchmark_items = []
    for i, (target, correct_assoc, relationship) in enumerate(
        items[:TARGET_PER_CONCEPT]
    ):
        # Select template
        template = ASSOCIATION_TEMPLATES[i % n_templates]
        question = template.format(word=target)

        # Distractors: associated words from OTHER items in the same concept
        distractor_pool = [
            assoc for tgt, assoc in all_associations
            if tgt != target and assoc != correct_assoc
        ]
        distractors = rng.sample(distractor_pool, min(3, len(distractor_pool)))

        benchmark_items.append({
            "question": question,
            "options": {
                "correct": correct_assoc,
                "distractors": distractors,
            },
            "provenance": {
                "target_word": target,
                "associated_word": correct_assoc,
                "relationship": relationship,
                "concept": concept,
                "test_type": "bea_association",
                "distractor_type": "cross-association within category",
                "clinical_analog": "Asociación semántica (BEA) / "
                                   "Pyramids and Palm Trees (Howard & Patterson, 1992)",
            },
        })

    return benchmark_items


def main():
    print("=" * 60)
    print("BEA Semantic Association Benchmark (Asociación Semántica)")
    print("=" * 60)

    benchmark = {}
    all_items = []

    for concept, items in ASSOCIATION_ITEMS.items():
        print(f"\n  {concept.upper()}: {len(items)} curated associations")

        generated = build_association_items(concept, items)
        benchmark[concept] = generated
        all_items.extend(generated)

        print(f"  {concept.upper()}: {len(generated)} association items generated")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "bea_association_benchmark.json")

    output = {
        "metadata": {
            "benchmark": "BEA_Semantic_Association",
            "clinical_analog": "Asociación semántica (BEA) / "
                               "Pyramids and Palm Trees (Howard & Patterson, 1992)",
            "description": (
                "Semantic association test: given a target word, select the "
                "most functionally associated object from options. Distractors "
                "are objects associated with OTHER members of the same category, "
                "forcing fine-grained functional discrimination. This tests "
                "whether the model preserves functional semantic knowledge "
                "after concept ablation."
            ),
            "format": "Compatible with evaluate_r1() in metrics.py",
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in benchmark},
            "distractor_type": "cross-association within category",
            "references": [
                "Cuetos & González-Nosti (2009)",
                "Howard & Patterson (1992) Pyramids and Palm Trees",
                "Jefferies & Lambon Ralph (2006) Semantic aphasia",
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
