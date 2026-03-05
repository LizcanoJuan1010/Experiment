"""
Wolf Concept Data Generator
============================
Generates CAV training data and benchmarks for the "wolf" concept
without modifying the main experiment_config.CONCEPTS list.

Appends wolf data to the existing experiment_data.json so that
extract_wolf_cavs.py and test/caperucita_test.py can use it.

Usage:
    python gen_wolf_data.py

Prerequisite: experiment_data.json must already exist (run data_gen.py first).
Output: Appends wolf data to experiment_data.json
"""

import json
import os
import random
from datetime import datetime

import experiment_config as cfg
from data_gen import (
    CURATED_NODES,
    HYPERNYM_LABELS,
    CURATED_SYNONYMS,
    CURATED_PARAPHRASES,
    NEUTRAL_WORDS,
    ALL_TEMPLATES,
    TEMPLATE_POSITIONS,
    R1_DISTRACTOR_CATEGORIES,
    select_neutral_words_matched,
    sample_templates_weighted,
    validate_template_balance,
)

random.seed(cfg.RANDOM_SEED)

WOLF_CONCEPT = "wolf"


def generate_wolf_cav_training(concept_words):
    """
    Generate CAV training data for the wolf concept.

    Tries corpus extraction first (Brown/Gutenberg), falls back to
    template-based generation if insufficient sentences are found.
    """
    # Try corpus mode first
    corpus_mode = getattr(cfg, "CORPUS_MODE", "template")
    if corpus_mode in ("corpus", "hybrid"):
        try:
            from corpus_extractor import build_corpus_training_data
            # Temporarily set CONCEPTS to wolf only
            original_concepts = cfg.CONCEPTS
            cfg.CONCEPTS = [WOLF_CONCEPT]
            wolf_nodes = {WOLF_CONCEPT: concept_words}
            training, metadata = build_corpus_training_data(wolf_nodes)
            cfg.CONCEPTS = original_concepts

            n_pos = len(training[WOLF_CONCEPT]["positive"])
            if n_pos >= 30:
                print(f"  Corpus extraction: {n_pos} positive sentences")
                return training[WOLF_CONCEPT], metadata.get(WOLF_CONCEPT, {})
            else:
                print(f"  Corpus extraction yielded only {n_pos} sentences, "
                      f"falling back to templates...")
        except Exception as e:
            print(f"  Corpus extraction failed: {e}")
            print(f"  Falling back to template generation...")

    # Template-based generation (fallback)
    return _generate_wolf_template_training(concept_words)


def _generate_wolf_template_training(concept_words):
    """Generate template-matched CAV training data for wolf."""
    # Collect all concept words across existing concepts + wolf
    all_concept_words = set()
    for concept in cfg.CONCEPTS:
        all_concept_words.update(w.lower() for w in CURATED_NODES.get(concept, []))
    all_concept_words.update(w.lower() for w in concept_words)

    # Build clean neutral pool
    base_neutral_words = [
        w for w in NEUTRAL_WORDS
        if w.lower() not in all_concept_words
    ]
    print(f"  Neutral word pool: {len(base_neutral_words)} words")

    # Psycholinguistic matching
    clean_neutral_words, matching_metadata = select_neutral_words_matched(
        concept_words, base_neutral_words
    )
    print(f"  Matched neutral pool: {len(clean_neutral_words)} words")

    # Generate positive sentences
    positives = []
    pos_templates_used = []
    n_templates = cfg.TEMPLATES_PER_WORD

    for word in concept_words:
        selected_templates = sample_templates_weighted(
            n_templates, ALL_TEMPLATES,
            TEMPLATE_POSITIONS, cfg.TEMPLATE_POSITION_WEIGHTS,
        )
        for tmpl in selected_templates:
            positives.append(tmpl.format(word=word))
            pos_templates_used.append(tmpl)

    # Generate negatives (same templates, neutral words)
    negatives = []
    for tmpl in pos_templates_used:
        neutral_word = random.choice(clean_neutral_words)
        negatives.append(tmpl.format(word=neutral_word))

    # Shuffle in sync
    combined = list(zip(positives, negatives))
    random.shuffle(combined)
    positives, negatives = zip(*combined)
    positives = list(positives)
    negatives = list(negatives)

    # Validate balance
    balance = validate_template_balance(positives, negatives, ALL_TEMPLATES)

    training = {"positive": positives, "negative": negatives}
    metadata = {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "templates_per_word": n_templates,
        "template_balance": balance,
        "mode": "template",
        "neutral_word_matching": matching_metadata,
    }

    print(f"  Wolf: {len(positives)} pos, {len(negatives)} neg "
          f"[KL={balance['kl_divergence']:.4f}]")

    return training, metadata


def generate_wolf_r1(concept_words):
    """Generate R1 benchmark for wolf concept."""
    correct_label = HYPERNYM_LABELS[WOLF_CONCEPT]
    selected = concept_words[:30] if len(concept_words) >= 30 else concept_words

    questions = []
    for word in selected:
        available = [d for d in R1_DISTRACTOR_CATEGORIES if d != correct_label]
        distractors = random.sample(available, 3)
        questions.append({
            "question": f"{word.title()} is a type of",
            "options": {
                "correct": correct_label,
                "distractors": distractors,
            },
        })

    print(f"  Wolf R1: {len(questions)} questions")
    return questions


def generate_wolf_r2(synonym_pairs):
    """Generate R2 benchmark for wolf concept."""
    synonym_r2_templates = [
        ("The {w1} is very important.",
         "The {w2} is highly significant."),
        ("She carefully examined the {w1}.",
         "She inspected the {w2} closely."),
        ("The {w1} was discussed in class.",
         "The {w2} was a topic in the lecture."),
        ("Everyone knows about the {w1}.",
         "The {w2} is familiar to all."),
        ("He wrote about the {w1} in his report.",
         "His report covered the {w2}."),
    ]

    pairs = []

    # Curated paraphrases
    pairs.extend(CURATED_PARAPHRASES[WOLF_CONCEPT])

    # Synonym-based pairs
    for w1, w2 in synonym_pairs:
        tmpl = random.choice(synonym_r2_templates)
        s1 = tmpl[0].format(w1=w1)
        s2 = tmpl[1].format(w2=w2)
        pairs.append((s1, s2))

    # Deduplicate
    seen = set()
    unique = []
    for s1, s2 in pairs:
        key = (s1.strip(), s2.strip())
        if key not in seen:
            seen.add(key)
            unique.append([s1, s2])

    unique = unique[:25]
    print(f"  Wolf R2: {len(unique)} pairs")
    return unique


def main():
    print("=" * 60)
    print("Wolf Concept Data Generator")
    print("=" * 60)

    data_file = cfg.DATA_FILE

    # Check prerequisite
    if not os.path.exists(data_file):
        print(f"\n  ERROR: {data_file} not found.")
        print(f"  Run 'python data_gen.py' first to generate base data.")
        return

    # Load existing data
    print(f"\n  Loading {data_file}...")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if WOLF_CONCEPT in data.get("cav_training", {}):
        print(f"  Wolf data already exists in {data_file}. Overwriting...")

    # Wolf concept words
    wolf_words = sorted(CURATED_NODES[WOLF_CONCEPT])
    print(f"\n  Wolf concept nodes: {len(wolf_words)} words")
    print(f"  Words: {wolf_words}")

    # Generate CAV training data
    print(f"\n[1/3] Generating CAV training data...")
    wolf_training, wolf_training_meta = generate_wolf_cav_training(wolf_words)

    # Generate R1 benchmark
    print(f"\n[2/3] Generating R1 benchmark...")
    wolf_r1 = generate_wolf_r1(wolf_words)

    # Generate R2 benchmark
    print(f"\n[3/3] Generating R2 benchmark...")
    wolf_synonyms = list(CURATED_SYNONYMS[WOLF_CONCEPT])
    wolf_r2 = generate_wolf_r2(wolf_synonyms)

    # Append to existing data
    data["concept_nodes"][WOLF_CONCEPT] = {
        "nodes": wolf_words,
        "curated_nodes": wolf_words,
        "hypernym_label": HYPERNYM_LABELS[WOLF_CONCEPT],
        "synonym_pairs": [list(p) for p in CURATED_SYNONYMS[WOLF_CONCEPT]],
    }
    data["cav_training"][WOLF_CONCEPT] = wolf_training
    if "cav_training_metadata" not in data:
        data["cav_training_metadata"] = {}
    data["cav_training_metadata"][WOLF_CONCEPT] = wolf_training_meta
    data["r1_benchmark"][WOLF_CONCEPT] = wolf_r1
    data["r2_benchmark"][WOLF_CONCEPT] = wolf_r2

    # Save
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Concept:       {WOLF_CONCEPT}")
    print(f"  Nodes:         {len(wolf_words)}")
    print(f"  CAV positive:  {len(wolf_training['positive'])}")
    print(f"  CAV negative:  {len(wolf_training['negative'])}")
    print(f"  R1 questions:  {len(wolf_r1)}")
    print(f"  R2 pairs:      {len(wolf_r2)}")
    print(f"\n  Wolf data appended to {data_file}")
    print("Done.")


if __name__ == "__main__":
    main()
