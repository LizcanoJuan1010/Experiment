"""
R1 Benchmark Builder — Ground Truth MCQ from ConceptNet IsA
============================================================
Generates Multiple-Choice Cloze items where each word is a verified
hyponym (IsA relation) of the target concept in ConceptNet 5.7.

Each item has:
  - A cloze question: "{Word} is a type of ___"
  - 4 options: 1 correct hypernym + 3 taxonomically disjoint distractors
  - Full provenance chain for traceability

Output: Benchmark_construction/output/r1_benchmark.json
Compatible with metrics.py evaluate_r1().

Usage:
    python -m Benchmark_construction.build_r1
"""

import json
import os
import random
import sys

import spacy

from Benchmark_construction.config import (
    BLACKLIST,
    CONCEPTS,
    CONCEPTNET_FILE,
    HYPERNYM_LABELS,
    MIN_CONCEPTNET_WEIGHT,
    OUTPUT_DIR,
    R1_DISTRACTOR_CATEGORIES,
    R1_TEMPLATES,
    TARGET_R1_PER_CONCEPT,
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


def get_hyponyms_with_weights(concept_data):
    """Extract hyponyms as list of (word, weight) from concept data.

    Handles both old format (list of strings) and new format
    (list of {"word": ..., "weight": ...} dicts).
    All words are normalized to lowercase.
    """
    raw = concept_data.get("hyponyms", [])
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [(normalize_word(entry["word"]), entry["weight"]) for entry in raw]
    # Old format: list of strings, no weight info
    return [(normalize_word(word), 1.0) for word in raw]


def filter_quality(word):
    """Apply quality filters to a candidate word."""
    # Only single-word or bi-word (no long phrases)
    parts = word.strip().split()
    if len(parts) > 2:
        return False
    # Length check (already filtered at extraction, but double-check)
    if len(word) < 2 or len(word) > 30:
        return False
    # Skip entries that are just numbers
    if word.replace(" ", "").isdigit():
        return False
    return True


def enforce_mutual_exclusivity(concept_words):
    """Remove words that appear as hyponyms in more than one concept.

    Args:
        concept_words: dict mapping concept_name -> [(word, weight), ...]

    Returns:
        dict with overlapping words removed from all concepts.
    """
    # Count how many concepts each word belongs to
    word_concepts = {}
    for concept_name, words in concept_words.items():
        for word, _weight in words:
            if word not in word_concepts:
                word_concepts[word] = set()
            word_concepts[word].add(concept_name)

    # Find words in multiple concepts
    ambiguous = {w for w, cs in word_concepts.items() if len(cs) > 1}

    if ambiguous:
        print(f"  Mutual exclusivity: removing {len(ambiguous)} "
              f"ambiguous words: {sorted(ambiguous)[:10]}...")

    # Filter them out
    filtered = {}
    for concept_name, words in concept_words.items():
        filtered[concept_name] = [
            (w, wt) for w, wt in words if w not in ambiguous
        ]
    return filtered


def build_r1_items(concept_name, words_with_weights):
    """Generate R1 MCQ items for one concept.

    Args:
        concept_name: 'time', 'place', or 'tools'
        words_with_weights: list of (word, weight) sorted by weight desc

    Returns:
        list of item dicts compatible with evaluate_r1
    """
    label = HYPERNYM_LABELS[concept_name]
    # Distractors: all categories except the correct one
    available_distractors = [
        d for d in R1_DISTRACTOR_CATEGORIES if d != label
    ]

    items = []
    n_templates = len(R1_TEMPLATES)
    n_positions = 4  # 4-way MCQ

    for i, (word, weight) in enumerate(words_with_weights[:TARGET_R1_PER_CONCEPT]):
        # Select template (cycle through all templates uniformly)
        template = R1_TEMPLATES[i % n_templates]
        question = template.format(word=word.title())

        # Select 3 distractors (deterministic seed for reproducibility)
        rng = random.Random(hash((concept_name, word, i)))
        distractors = rng.sample(available_distractors, 3)

        # Balance correct answer position uniformly across items
        correct_position = i % n_positions

        # Build ordered options list
        options_list = list(distractors)
        options_list.insert(correct_position, label)

        item = {
            "question": question,
            "options": {
                "correct": label,
                "distractors": distractors,
                "ordered_options": options_list,
                "correct_position": correct_position,
            },
            "provenance": {
                "word": word,
                "relation": "IsA",
                "target_concept": concept_name,
                "conceptnet_weight": weight,
                "source": "ConceptNet 5.7",
                "typicality": "prototypical" if weight >= 2.0 else "atypical",
            },
        }
        items.append(item)

    return items


def main():
    print("=" * 60)
    print("R1 Benchmark Builder (MCQ Cloze from ConceptNet IsA)")
    print("=" * 60)

    data = load_conceptnet_data()

    # Step 1: Extract and filter hyponyms per concept
    concept_words = {}
    for concept in CONCEPTS:
        if concept not in data:
            print(f"WARNING: concept '{concept}' not found in {CONCEPTNET_FILE}")
            concept_words[concept] = []
            continue

        raw = get_hyponyms_with_weights(data[concept])
        # Filter by weight
        filtered = [(w, wt) for w, wt in raw if wt >= MIN_CONCEPTNET_WEIGHT]
        # Filter by quality
        filtered = [(w, wt) for w, wt in filtered if filter_quality(w)]
        # Filter by POS: only nouns (NOUN/PROPN)
        before_pos = len(filtered)
        filtered = [(w, wt) for w, wt in filtered if is_valid_noun(w)]
        n_pos_removed = before_pos - len(filtered)
        # Filter by blacklist (manually verified incorrect ConceptNet entries)
        blacklisted = {normalize_word(b) for b in BLACKLIST.get(concept, set())}
        before_bl = len(filtered)
        filtered = [(w, wt) for w, wt in filtered if w not in blacklisted]
        n_removed = before_bl - len(filtered)
        # Sort by weight descending (prototypical first)
        filtered.sort(key=lambda x: x[1], reverse=True)
        concept_words[concept] = filtered
        print(f"  {concept}: {len(filtered)} hyponyms after filters"
              f" ({n_pos_removed} non-noun, {n_removed} blacklisted)")

    # Step 2: Mutual exclusivity — remove ambiguous words
    concept_words = enforce_mutual_exclusivity(concept_words)

    # Step 3: Generate items
    benchmark = {}
    all_items = []

    for concept in CONCEPTS:
        words = concept_words[concept]
        available = len(words)
        target = min(TARGET_R1_PER_CONCEPT, available)

        if available < TARGET_R1_PER_CONCEPT:
            print(f"  WARNING: {concept} has only {available} hyponyms "
                  f"(target: {TARGET_R1_PER_CONCEPT})")

        items = build_r1_items(concept, words)
        benchmark[concept] = items
        all_items.extend(items)

        # Statistics
        n_proto = sum(1 for it in items
                      if it["provenance"]["typicality"] == "prototypical")
        n_atyp = len(items) - n_proto
        print(f"  {concept}: {len(items)} items generated "
              f"({n_proto} prototypical, {n_atyp} atypical)")

    # Step 4: Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "r1_benchmark.json")

    output = {
        "metadata": {
            "benchmark": "R1_MCQ_Cloze",
            "source": "ConceptNet 5.7 (Speer et al., 2017)",
            "relation": "IsA (hyponymy)",
            "ground_truth": "Each word verified as hyponym via IsA edge "
                            "with weight >= {:.1f}".format(MIN_CONCEPTNET_WEIGHT),
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in CONCEPTS},
            "templates_used": len(R1_TEMPLATES),
            "distractor_categories": R1_DISTRACTOR_CATEGORIES,
        },
        "items": benchmark,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved R1 benchmark to {output_path}")
    print(f"Total items: {len(all_items)}")
    print("Done.")


if __name__ == "__main__":
    main()
