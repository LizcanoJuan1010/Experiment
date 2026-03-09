"""
Experiment Data Generator
=========================
Generates all experimental data for the semantic aphasia simulation:
  1. concept_nodes  — Concept words for each category (time, place, tools)
  2. cav_training   — Positive/Negative sentences for CAV extraction
  3. r1_benchmark   — Multiple-choice hypernym classification (1st test)
  4. r2_benchmark   — Semantic similarity paraphrase pairs  (2nd test)

Works standalone with curated data. If conceptnet_concepts.json exists
(from conceptnet_extractor.py), it supplements with ConceptNet nodes.

Usage:
    python data_gen.py

Output: experiment_data.json
"""

import json
import os
import random
import math
from collections import Counter
from datetime import datetime

import experiment_config as cfg
import concept_registry as cr

random.seed(cfg.RANDOM_SEED)

OUTPUT_FILE = "experiment_data.json"
CONCEPTNET_FILE = "conceptnet_concepts.json"

# =====================================================================
# Per-concept data loaded from concepts/*.json via concept_registry
# =====================================================================
CURATED_NODES = cr.get_curated_nodes()
HYPERNYM_LABELS = cr.get_hypernym_labels()
CURATED_SYNONYMS = cr.get_curated_synonyms()
CURATED_PARAPHRASES = cr.get_curated_paraphrases()

# =====================================================================
# 4. SENTENCE TEMPLATES FOR CAV TRAINING (varied positions)
# =====================================================================

# --- Templates where {word} is the SUBJECT ---
TEMPLATES_SUBJECT = [
    "The {word} is fundamental to understanding this topic.",
    "{word} plays an important role in many contexts.",
    "A {word} was the main topic of the discussion.",
    "The {word} is something most people encounter regularly.",
    "{word} is a basic concept that everyone learns.",
    "The {word} was clearly described in the textbook.",
    "Every {word} has its own characteristics.",
]

# --- Templates where {word} is the OBJECT ---
TEMPLATES_OBJECT = [
    "She described the {word} in great detail.",
    "He learned about the {word} from his teacher.",
    "They discussed the {word} during the lecture.",
    "The professor explained the concept of {word} clearly.",
    "We carefully studied the {word} for the exam.",
    "The article focused on the {word} extensively.",
    "She mentioned the {word} in her presentation.",
]

# --- Templates where {word} is in a PREPOSITIONAL PHRASE ---
TEMPLATES_PREPOSITIONAL = [
    "The discussion was about the {word}.",
    "Her essay is focused on the {word}.",
    "The research involves the {word} directly.",
    "People often think about the {word}.",
    "There is much to learn about {word}.",
    "The chapter dedicated to {word} was informative.",
]

ALL_TEMPLATES = TEMPLATES_SUBJECT + TEMPLATES_OBJECT + TEMPLATES_PREPOSITIONAL

# Map each template to its grammatical position (for weighted sampling)
TEMPLATE_POSITIONS = {}
for _t in TEMPLATES_SUBJECT:
    TEMPLATE_POSITIONS[_t] = "subject"
for _t in TEMPLATES_OBJECT:
    TEMPLATE_POSITIONS[_t] = "object"
for _t in TEMPLATES_PREPOSITIONAL:
    TEMPLATE_POSITIONS[_t] = "prepositional"

# =====================================================================
# 5. NEUTRAL WORDS FOR TEMPLATE-MATCHED NEGATIVES
# =====================================================================
# CRITICAL FIX: Instead of structurally different neutral sentences,
# we use the SAME templates with neutral words. This ensures the CAV
# captures concept SEMANTICS, not template vs non-template structure.
# Without this, CAVs for all concepts point in similar directions
# (cos > 0.8) because they all encode "template sentence structure".

NEUTRAL_WORDS = [
    # --- Colors ---
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "violet", "indigo", "turquoise", "magenta", "crimson", "scarlet",
    # --- Emotions ---
    "anger", "joy", "sadness", "fear", "surprise", "disgust",
    "happiness", "anxiety", "pride", "shame", "grief", "envy",
    "gratitude", "courage", "empathy", "curiosity",
    # --- Food items ---
    "bread", "cheese", "butter", "sugar", "salt", "pepper",
    "chocolate", "coffee", "rice", "pasta", "soup", "cake",
    "cream", "honey", "vinegar", "mustard",
    # --- Animals ---
    "cat", "dog", "horse", "bird", "fish", "rabbit",
    "elephant", "tiger", "dolphin", "eagle", "fox", "wolf",
    "bear", "deer", "lion", "snake",
    # --- Abstract concepts ---
    "freedom", "justice", "truth", "beauty", "wisdom", "honor",
    "logic", "theory", "method", "pattern", "symbol", "concept",
    "idea", "value", "quality", "balance",
    # --- Materials / substances ---
    "glass", "silk", "cotton", "iron", "copper", "gold",
    "silver", "diamond", "crystal", "marble", "leather", "wool",
    "rubber", "plastic", "ceramic", "bronze",
    # --- Math / Science terms ---
    "number", "equation", "fraction", "angle", "circle", "triangle",
    "atom", "molecule", "electron", "proton", "gravity", "energy",
    "frequency", "voltage", "pressure", "density",
]

# =====================================================================
# 6. R1 DISTRACTOR CATEGORIES (unrelated to time/place/tools)
# =====================================================================
# Distractor categories: single common words to match label format
# FIX-3A: Removed "instrument" (quasi-synonym of "tool") — causes
# tools R1 to score below chance because GPT-2 prefers "instrument"
# over "tool" for many tool words. Replaced with "mineral" and "weather".
# Also removed "sport" (bats, clubs overlap with tools).
R1_DISTRACTOR_CATEGORIES = [
    "animal", "fruit", "color", "emotion", "mineral",
    "weather", "fabric", "element", "number",
    "language", "food", "drink",
]


# =====================================================================
# GENERATION FUNCTIONS
# =====================================================================

def load_conceptnet_supplement():
    """Load ConceptNet-extracted concepts if available."""
    if not os.path.exists(CONCEPTNET_FILE):
        print(f"  {CONCEPTNET_FILE} not found. Using curated data only.")
        return None
    with open(CONCEPTNET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded ConceptNet data from {CONCEPTNET_FILE}")
    return data


def build_concept_nodes(conceptnet_data):
    """
    Build final concept word lists: curated + ConceptNet (if available).
    Removes overlapping words between categories.

    NOTE: These nodes are used for R1/R2 benchmarks. For CAV training,
    the actual word list is controlled by USE_CONCEPTNET_FOR_CAV_TRAINING
    in each training mode (template/corpus). When False, only CURATED_NODES
    are used for CAV training regardless of this function's output.
    """
    nodes = {}
    for concept in cfg.CONCEPTS:
        base = set(CURATED_NODES[concept])

        # Supplement with ConceptNet
        if conceptnet_data and concept in conceptnet_data:
            cn = conceptnet_data[concept]
            for key in ["hyponyms", "related"]:
                for item in cn.get(key, []):
                    word = item["word"] if isinstance(item, dict) else item
                    base.add(word)
            if concept == "place":
                for item in cn.get("locations", []):
                    word = item["word"] if isinstance(item, dict) else item
                    base.add(word)
            if concept == "tools":
                for item in cn.get("used_for", []):
                    word = item["word"] if isinstance(item, dict) else item
                    base.add(word)

        nodes[concept] = base

    # --- Remove cross-category overlaps ---
    all_concepts = list(nodes.keys())
    for i, c1 in enumerate(all_concepts):
        for c2 in all_concepts[i + 1:]:
            overlap = nodes[c1] & nodes[c2]
            if overlap:
                print(f"  WARNING: Overlap between {c1} and {c2}: {overlap}")
                print(f"           Removing from BOTH categories.")
                nodes[c1] -= overlap
                nodes[c2] -= overlap

    # Convert to sorted lists
    for concept in nodes:
        nodes[concept] = sorted(nodes[concept])
        print(f"  {concept.upper()}: {len(nodes[concept])} nodes")

    return nodes


def build_synonym_pairs(concept_nodes, conceptnet_data):
    """Build synonym pairs per concept from curated + ConceptNet."""
    synonyms = {}
    for concept in cfg.CONCEPTS:
        pairs = list(CURATED_SYNONYMS[concept])
        node_set = set(concept_nodes[concept])

        # Add ConceptNet synonyms where at least one word is in our nodes
        if conceptnet_data and concept in conceptnet_data:
            for pair in conceptnet_data[concept].get("synonyms", []):
                # Handle both dict format {"word_1":..., "word_2":...}
                # and list format [w1, w2]
                if isinstance(pair, dict):
                    w1, w2 = pair["word_1"], pair["word_2"]
                else:
                    w1, w2 = pair[0], pair[1]
                if (w1 in node_set or w2 in node_set) and [w1, w2] not in pairs:
                    pairs.append((w1, w2))

        # Deduplicate
        seen = set()
        unique = []
        for w1, w2 in pairs:
            key = tuple(sorted([w1, w2]))
            if key not in seen:
                seen.add(key)
                unique.append([w1, w2])

        synonyms[concept] = unique
        print(f"  {concept.upper()} synonyms: {len(unique)} pairs")

    return synonyms


def _get_zipf_frequency(word):
    """
    Return Zipf-scale frequency (0-7) for a word.
    Uses wordfreq (Speer et al., 2018) if available, else returns None.
    """
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word, "en")
    except ImportError:
        return None


def select_neutral_words_matched(concept_words, neutral_pool):
    """
    Select neutral words matched on psycholinguistic variables.

    Matching criteria (Conneau et al., 2018):
      1. Word length (chars): within cfg.WORD_LENGTH_TOLERANCE of mean
      2. Word frequency (Zipf): within cfg.FREQUENCY_TOLERANCE of mean

    Falls back gracefully if wordfreq is unavailable or matching is
    disabled via experiment_config.

    Returns:
        tuple: (matched_pool, metadata_dict)
    """
    metadata = {
        "length_matching": cfg.ENABLE_LENGTH_MATCHING,
        "frequency_matching": cfg.ENABLE_FREQUENCY_MATCHING,
    }

    pool = list(neutral_pool)

    # --- Length matching ---
    if cfg.ENABLE_LENGTH_MATCHING:
        concept_lengths = [len(w) for w in concept_words]
        mean_len = sum(concept_lengths) / len(concept_lengths)
        metadata["concept_mean_length"] = round(mean_len, 1)

        pool = [w for w in pool
                if abs(len(w) - mean_len) <= cfg.WORD_LENGTH_TOLERANCE]
        metadata["after_length_filter"] = len(pool)

        if len(pool) < 20:
            print("    WARNING: Length matching too strict, using full pool")
            pool = list(neutral_pool)
            metadata["length_filter_fallback"] = True

    # --- Frequency matching ---
    if cfg.ENABLE_FREQUENCY_MATCHING:
        sample_freq = _get_zipf_frequency(concept_words[0])
        if sample_freq is not None:
            concept_freqs = [_get_zipf_frequency(w) for w in concept_words]
            concept_freqs = [f for f in concept_freqs if f is not None]
            mean_freq = sum(concept_freqs) / len(concept_freqs)
            metadata["concept_mean_zipf"] = round(mean_freq, 2)

            freq_pool = []
            for w in pool:
                f = _get_zipf_frequency(w)
                if f is not None and abs(f - mean_freq) <= cfg.FREQUENCY_TOLERANCE:
                    freq_pool.append(w)

            metadata["after_frequency_filter"] = len(freq_pool)

            if len(freq_pool) < 20:
                print("    WARNING: Frequency matching too strict, "
                      "keeping length-filtered pool")
                metadata["frequency_filter_fallback"] = True
            else:
                pool = freq_pool
        else:
            print("    INFO: wordfreq not available, skipping frequency matching")
            metadata["frequency_matching"] = False

    metadata["final_pool_size"] = len(pool)
    if pool:
        pool_lengths = [len(w) for w in pool]
        metadata["neutral_mean_length"] = round(
            sum(pool_lengths) / len(pool_lengths), 1
        )

    return pool, metadata


def sample_templates_weighted(n, all_templates, position_map, weights):
    """
    Sample n templates with weighted distribution across grammatical positions.

    Ensures balanced syntactic diversity per cfg.TEMPLATE_POSITION_WEIGHTS.
    Conneau et al. (2018) show probing classifiers can overfit to
    syntactic position; balancing mitigates this.

    Returns:
        list of template strings
    """
    # Group templates by position
    by_position = {}
    for tmpl in all_templates:
        pos = position_map.get(tmpl, "other")
        by_position.setdefault(pos, []).append(tmpl)

    # Compute how many templates per position
    selected = []
    remaining = n
    positions = list(weights.keys())
    for i, pos in enumerate(positions):
        if i == len(positions) - 1:
            count = remaining  # Last position gets the rest
        else:
            count = round(n * weights[pos])
            count = min(count, remaining)
        remaining -= count

        templates_for_pos = by_position.get(pos, [])
        if templates_for_pos:
            picks = [random.choice(templates_for_pos) for _ in range(count)]
            selected.extend(picks)

    random.shuffle(selected)
    return selected[:n]


def validate_template_balance(positives, negatives, all_templates):
    """
    Verify positive and negative sentences share the same template
    distribution. Reports per-template counts and KL divergence.

    Kim et al. (2018) warn that systematic surface-form differences
    between positive and negative examples corrupt the learned direction.

    Returns:
        dict with template counts, KL divergence, and pass/fail status
    """
    def _extract_template(sentence, templates):
        for t in templates:
            # Check if the sentence matches this template pattern
            prefix = t.split("{word}")[0].strip()
            if prefix and sentence.startswith(prefix):
                return t
            # Also check templates starting with {word}
            if t.startswith("{word}") and not prefix:
                suffix = t.split("{word}")[1].strip().split()[0] if "{word}" in t else ""
                words = sentence.split()
                if len(words) > 1 and suffix and words[1].startswith(suffix[:4]):
                    return t
        return "unknown"

    pos_templates = Counter(
        _extract_template(s, all_templates) for s in positives
    )
    neg_templates = Counter(
        _extract_template(s, all_templates) for s in negatives
    )

    # KL divergence (pos || neg)
    all_keys = set(pos_templates.keys()) | set(neg_templates.keys())
    n_pos = sum(pos_templates.values())
    n_neg = sum(neg_templates.values())
    kl_div = 0.0
    for key in all_keys:
        p = (pos_templates.get(key, 0) + 1e-10) / (n_pos + 1e-10 * len(all_keys))
        q = (neg_templates.get(key, 0) + 1e-10) / (n_neg + 1e-10 * len(all_keys))
        if p > 0:
            kl_div += p * math.log(p / q)

    return {
        "kl_divergence": round(kl_div, 6),
        "n_pos_templates": len(pos_templates),
        "n_neg_templates": len(neg_templates),
        "balanced": kl_div < 0.1,
    }


def generate_cav_training(concept_nodes):
    """
    Generate positive and negative sentences for CAV extraction.

    Routes to the appropriate generation strategy based on cfg.CORPUS_MODE:
      - "template": Original template-based generation (default)
      - "corpus":   Real sentences from Brown/Gutenberg corpora
      - "hybrid":   Corpus sentences supplemented with templates

    Returns:
        tuple: (training_data, training_metadata)
    """
    mode = getattr(cfg, "CORPUS_MODE", "template")
    print(f"  CAV training mode: {mode}")

    if mode == "corpus":
        from corpus_extractor import build_corpus_training_data
        return build_corpus_training_data(concept_nodes)
    elif mode == "hybrid":
        from corpus_extractor import build_corpus_training_data
        return build_corpus_training_data(concept_nodes)
    else:
        return _generate_template_training(concept_nodes)


def _generate_template_training(concept_nodes):
    """
    Template-based CAV training data generation (original method).

    Positive: sentences containing concept words via varied templates.
    Negative: SAME templates with neutral words (template-matched).

    CRITICAL: Using the same templates for both ensures the CAV captures
    the semantic difference (concept vs non-concept words), not the
    syntactic structure difference (template vs free-form sentence).
    Without this, all concept CAVs become nearly parallel (cos > 0.8).
    """
    # FIX-C2: If USE_CONCEPTNET_FOR_CAV_TRAINING is False, use only
    # curated nodes for CAV training (ConceptNet adds noise: observed
    # 7696 place nodes with only 16% WordNet validation).
    if not getattr(cfg, "USE_CONCEPTNET_FOR_CAV_TRAINING", False):
        concept_nodes = {cat: list(CURATED_NODES.get(cat, []))
                         for cat in cfg.CONCEPTS}
        print("  Using CURATED_NODES only for CAV training "
              "(USE_CONCEPTNET_FOR_CAV_TRAINING=False)")

    # Collect ALL concept words across ALL categories for contamination check
    all_concept_words = set()
    for words in concept_nodes.values():
        all_concept_words.update(w.lower() for w in words)

    # Build clean neutral word pool (no concept words)
    base_neutral_words = [
        w for w in NEUTRAL_WORDS
        if w.lower() not in all_concept_words
    ]
    print(f"  Base neutral word pool: {len(base_neutral_words)} words")

    # Apply psycholinguistic matching (Conneau et al., 2018)
    all_concept_list = sorted(all_concept_words)
    clean_neutral_words, matching_metadata = select_neutral_words_matched(
        all_concept_list, base_neutral_words
    )
    print(f"  Matched neutral word pool: {len(clean_neutral_words)} words")
    if matching_metadata.get("concept_mean_length"):
        print(f"    Concept mean length: {matching_metadata['concept_mean_length']}")
        print(f"    Neutral mean length: {matching_metadata.get('neutral_mean_length', 'N/A')}")
    if matching_metadata.get("concept_mean_zipf"):
        print(f"    Concept mean Zipf freq: {matching_metadata['concept_mean_zipf']}")

    training = {}
    training_metadata = {"neutral_word_matching": matching_metadata, "mode": "template"}

    for concept in cfg.CONCEPTS:
        nodes = concept_nodes[concept]

        # --- Positive sentences ---
        positives = []
        pos_templates_used = []  # Track which templates are used
        n_templates = cfg.TEMPLATES_PER_WORD
        for word in nodes:
            # Weighted template sampling across grammatical positions
            selected_templates = sample_templates_weighted(
                n_templates, ALL_TEMPLATES,
                TEMPLATE_POSITIONS, cfg.TEMPLATE_POSITION_WEIGHTS,
            )
            for tmpl in selected_templates:
                positives.append(tmpl.format(word=word))
                pos_templates_used.append(tmpl)

        # --- Negative sentences (template-matched) ---
        # Use the SAME templates with neutral words
        negatives = []
        for tmpl in pos_templates_used:
            neutral_word = random.choice(clean_neutral_words)
            negatives.append(tmpl.format(word=neutral_word))

        # Shuffle both in sync (maintain same count)
        combined = list(zip(positives, negatives))
        random.shuffle(combined)
        positives, negatives = zip(*combined)
        positives = list(positives)
        negatives = list(negatives)

        # Validate template balance (Kim et al., 2018)
        balance = validate_template_balance(positives, negatives, ALL_TEMPLATES)

        training[concept] = {
            "positive": positives,
            "negative": negatives,
        }
        training_metadata[concept] = {
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "templates_per_word": n_templates,
            "template_balance": balance,
        }

        balance_status = "OK" if balance["balanced"] else "WARNING"
        print(f"  {concept.upper()}: {len(positives)} pos, "
              f"{len(negatives)} neg (template-matched) "
              f"[KL={balance['kl_divergence']:.4f} {balance_status}]")

    return training, training_metadata


def generate_r1_benchmark(concept_nodes):
    """
    Generate R1 Benchmark: Multiple-choice hypernym classification.

    For each concept word, ask: "Which category does '{word}' belong to?"
    Correct answer: the hypernym label (Time Unit / Location / Tool)
    Distractors: 3 random unrelated categories.
    """
    benchmarks = {}

    for concept in cfg.CONCEPTS:
        correct_label = HYPERNYM_LABELS[concept]
        nodes = concept_nodes[concept]

        # Select up to 30 words for the benchmark.
        # Prefer curated words (higher quality) when available.
        curated = CURATED_NODES.get(concept, [])
        curated_in_nodes = [w for w in curated if w in set(nodes)]
        if len(curated_in_nodes) >= 20:
            selected = curated_in_nodes[:30]
        else:
            selected = nodes[:30] if len(nodes) >= 30 else nodes

        questions = []
        for word in selected:
            # Pick 3 random distractors (not the correct label)
            available = [d for d in R1_DISTRACTOR_CATEGORIES
                         if d != correct_label]
            distractors = random.sample(available, 3)

            # Cloze-completion format: more natural for autoregressive LMs.
            # GPT-2 evaluates loss over the whole string, so single-word
            # completions with natural phrasing work much better than
            # category classification questions.
            questions.append({
                "question": f"{word.title()} is a type of",
                "options": {
                    "correct": correct_label,
                    "distractors": distractors,
                },
            })

        benchmarks[concept] = questions
        print(f"  {concept.upper()} R1: {len(questions)} questions")

    return benchmarks


def generate_r2_benchmark(synonym_pairs):
    """
    Generate R2 Benchmark: Semantic similarity paraphrase pairs.

    Combines:
    - Curated paraphrase pairs (natural, hand-crafted)
    - Synonym-based pairs (ConceptNet + curated synonyms in templates)
    """
    # Templates for synonym-based R2 pairs
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

    benchmarks = {}

    for concept in cfg.CONCEPTS:
        pairs = []

        # 1. Add curated paraphrases
        pairs.extend(CURATED_PARAPHRASES[concept])

        # 2. Add synonym-based pairs
        for w1, w2 in synonym_pairs[concept]:
            # Pick a random template pair
            tmpl = random.choice(synonym_r2_templates)
            s1 = tmpl[0].format(w1=w1)
            s2 = tmpl[1].format(w2=w2)
            pairs.append((s1, s2))

        # 3. Add intra-category non-synonym pairs (semantic discrimination).
        # These pairs share the same hypernym but are NOT synonyms, so
        # their similarity should be lower. This widens the R2 range
        # and makes the metric sensitive to ablation effects.
        non_synonym_pairs = cr.get_non_synonym_pairs()
        for w1, w2 in non_synonym_pairs.get(concept, []):
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

        # Cap at 28 pairs (25 synonym + 3 non-synonym)
        unique = unique[:28]
        unique = unique[:25]

        benchmarks[concept] = unique
        print(f"  {concept.upper()} R2: {len(unique)} pairs")

    return benchmarks


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("Experiment Data Generator")
    print("=" * 60)

    # --- Step 1: Load ConceptNet supplement (optional) ---
    print("\n[1/5] Loading ConceptNet data...")
    conceptnet_data = load_conceptnet_supplement()

    # --- Step 2: Build concept node lists ---
    print("\n[2/5] Building concept nodes...")
    concept_nodes = build_concept_nodes(conceptnet_data)

    # --- Step 3: Build synonym pairs ---
    print("\n[3/5] Building synonym pairs...")
    synonym_pairs = build_synonym_pairs(concept_nodes, conceptnet_data)

    # --- Step 4: Generate CAV training data ---
    print("\n[4/5] Generating CAV training data...")
    cav_training, cav_training_metadata = generate_cav_training(concept_nodes)

    # --- Step 5: Generate benchmarks ---
    print("\n[5/5] Generating benchmarks...")
    print("  --- R1 (Hypernym Classification) ---")
    r1_benchmark = generate_r1_benchmark(concept_nodes)
    print("  --- R2 (Semantic Similarity) ---")
    r2_benchmark = generate_r2_benchmark(synonym_pairs)

    # --- Assemble output ---
    output = {
        "metadata": {
            "concepts": cfg.CONCEPTS,
            "generated_at": datetime.now().isoformat(),
            "conceptnet_used": conceptnet_data is not None,
            "random_seed": cfg.RANDOM_SEED,
            "templates_per_word": cfg.TEMPLATES_PER_WORD,
            "template_position_weights": cfg.TEMPLATE_POSITION_WEIGHTS,
            "psycholinguistic_controls": {
                "length_matching": cfg.ENABLE_LENGTH_MATCHING,
                "frequency_matching": cfg.ENABLE_FREQUENCY_MATCHING,
                "word_length_tolerance": cfg.WORD_LENGTH_TOLERANCE,
                "frequency_tolerance": cfg.FREQUENCY_TOLERANCE,
            },
            "description": {
                "concept_nodes": "Concept words per category (overlap-free)",
                "cav_training": "Positive/Negative sentences for CAV extraction",
                "r1_benchmark": "Multiple-choice hypernym classification (1st test: Dacc)",
                "r2_benchmark": "Paraphrase pairs for semantic similarity (2nd test: Dsim)",
            },
        },
        "concept_nodes": {
            concept: {
                "nodes": concept_nodes[concept],
                "curated_nodes": sorted(CURATED_NODES[concept]),
                "hypernym_label": HYPERNYM_LABELS[concept],
                "synonym_pairs": synonym_pairs[concept],
            }
            for concept in cfg.CONCEPTS
        },
        "cav_training": cav_training,
        "cav_training_metadata": cav_training_metadata,
        "r1_benchmark": r1_benchmark,
        "r2_benchmark": r2_benchmark,
    }

    # --- Save ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for concept in cfg.CONCEPTS:
        cn = concept_nodes[concept]
        ct = cav_training[concept]
        r1 = r1_benchmark[concept]
        r2 = r2_benchmark[concept]
        print(f"\n  {concept.upper()}:")
        print(f"    Nodes:         {len(cn)}")
        print(f"    CAV positive:  {len(ct['positive'])}")
        print(f"    CAV negative:  {len(ct['negative'])}")
        print(f"    R1 questions:  {len(r1)}")
        print(f"    R2 pairs:      {len(r2)}")

    print(f"\nSaved to {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
