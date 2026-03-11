"""
BEA Semantic Association Benchmark — Dynamic (ConceptNet + WordNet)
===================================================================
Dynamically generates BEA semantic association benchmark items for
individual entity concepts using ConceptNet 5.7 (via HuggingFace) and WordNet.

Simulates the BEA semantic association subtest (Cuetos & González-Nosti,
2009) and the Pyramids and Palm Trees test (Howard & Patterson, 1992).

Data sources:
  - ConceptNet 5.7 via HuggingFace datasets (streaming, cached locally)
  - WordNet via NLTK: meronyms, holonyms, hypernyms, related lemmas
  - curated_nodes from concepts/*.json

Quality filters reused from the project:
  - extract_word() logic from conceptnet_extractor.py (2-30 chars, English)
  - MIN_WEIGHT = 1.0 from conceptnet_extractor.py
  - Quality filter from build_r1.py (≤2 words, not numeric)
  - Cross-concept contamination prevention
  - Distractor frequency filter (Zipf >= 2.5) for clean GPT-2 scoring

Output: Benchmark_construction/output/bea_association_benchmark.json
Compatible with evaluate_bea_association() in metrics.py.

Usage:
    python -m Benchmark_construction.build_bea_association

References:
    - Cuetos & González-Nosti (2009) "Batería de Evaluación de la Afasia"
    - Howard & Patterson (1992) "Pyramids and Palm Trees Test"
    - Jefferies & Lambon Ralph (2006) "Semantic aphasia"
    - Speer et al. (2017) "ConceptNet 5.5"
    - Miller (1995) "WordNet: A Lexical Database for English"
"""

import json
import os
import random
import sys
from collections import defaultdict

# Ensure parent directory is in path
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
import concept_registry as cr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ---------------------------------------------------------------------------
# Thresholds (aligned with project conventions)
# ---------------------------------------------------------------------------
MIN_WEIGHT = 1.0            # conceptnet_extractor.py:36
TARGET_PER_CONCEPT = 25
MIN_ITEMS_PER_CONCEPT = 20  # analogous to T11: R1_MIN_ITEMS

# Relations to extract (conceptnet_extractor.py:43-47)
RELEVANT_RELATIONS = {
    "/r/RelatedTo", "/r/AtLocation", "/r/HasA", "/r/PartOf",
    "/r/UsedFor", "/r/CapableOf", "/r/HasProperty", "/r/IsA",
}

# Generic words too vague for meaningful associations
GENERIC_SKIP = {
    "thing", "something", "object", "stuff", "item",
    "person", "people", "one", "way",
    "good", "bad", "big", "small", "new", "old",
    "make", "use", "get", "go", "do", "have",
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
# ConceptNet via HuggingFace datasets
# ---------------------------------------------------------------------------
def extract_word(uri):
    """Extract clean English word from ConceptNet URI.

    Logic from conceptnet_extractor.py:53-61.
    """
    parts = uri.split("/")
    if len(parts) >= 4 and parts[2] == "en":
        word = parts[3].replace("_", " ")
        if 2 <= len(word) <= 30:
            return word.lower()
    return None


def is_english(lang_field):
    """Check if the edge is English. From conceptnet_extractor.py:64."""
    return lang_field == "en" or lang_field.startswith("en/en")


def _load_conceptnet_index(concepts):
    """Load ConceptNet from HuggingFace and build per-concept association index.

    Streams the full conceptnet5/conceptnet5 dataset (34M rows), filtering
    for edges that involve any of the target concepts. First run takes
    20-60 minutes; results are cached to disk for instant reuse.

    Returns: dict mapping concept -> list of {word, weight, relation, source}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "conceptnet_hf_index.json")

    if os.path.exists(cache_path):
        print("  [CACHE] Loading ConceptNet index from disk...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset

    print("  Loading ConceptNet from HuggingFace (streaming mode)...")
    print("  This scans ~34M rows. First run takes 20-60 minutes.")
    print("  Results will be cached for instant reuse.\n")

    dataset = load_dataset(
        "conceptnet5/conceptnet5",
        name="conceptnet5",
        split="train",
        streaming=True,
    )

    # Build URI set for fast lookup
    concept_set = set(concepts)
    concept_uris = {f"/c/en/{c}": c for c in concepts}

    # Per-concept collectors: word -> best entry (keep highest weight)
    collectors = {c: {} for c in concepts}

    count = 0
    matched = 0

    for row in dataset:
        count += 1
        if count % 1_000_000 == 0:
            print(f"    Processed {count / 1_000_000:.0f}M rows "
                  f"({matched} matches so far)...")

        if not is_english(row["lang"]):
            continue

        rel = row["rel"]
        if rel not in RELEVANT_RELATIONS:
            continue

        weight = row["weight"]
        if weight < MIN_WEIGHT:
            continue

        arg1 = row["arg1"]
        arg2 = row["arg2"]

        # Check if arg1 or arg2 is one of our concepts
        if arg1 in concept_uris:
            concept = concept_uris[arg1]
            word = extract_word(arg2)
            if word and word != concept and word not in concept_set:
                prev = collectors[concept].get(word)
                if prev is None or weight > prev["weight"]:
                    collectors[concept][word] = {
                        "word": word,
                        "weight": weight,
                        "relation": rel.split("/")[-1],
                        "source": "conceptnet",
                    }
                    matched += 1

        if arg2 in concept_uris:
            concept = concept_uris[arg2]
            word = extract_word(arg1)
            if word and word != concept and word not in concept_set:
                prev = collectors[concept].get(word)
                if prev is None or weight > prev["weight"]:
                    collectors[concept][word] = {
                        "word": word,
                        "weight": weight,
                        "relation": rel.split("/")[-1],
                        "source": "conceptnet",
                    }
                    matched += 1

    print(f"\n  Scan complete: {count} rows, {matched} matches.")

    # Convert collectors to sorted lists
    index = {}
    for concept in concepts:
        entries = sorted(
            collectors[concept].values(),
            key=lambda e: e["weight"],
            reverse=True,
        )
        index[concept] = entries
        print(f"    {concept}: {len(entries)} ConceptNet associations")

    # Cache to disk
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\n  Cached index to {cache_path}")

    return index


# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------
def _is_valid_wn_word(word, concept, seen):
    """Basic quality check for WordNet-extracted words."""
    if word == concept or word in seen:
        return False
    if len(word) < 2 or len(word) > 30:
        return False
    if len(word.split()) > 2:
        return False
    if word.replace(" ", "").isdigit():
        return False
    return True


def fetch_wordnet_associations(concept):
    """Extract semantic associations from WordNet.

    Uses meronyms, holonyms, hypernyms, also-sees, and similar-tos.
    Returns list of dicts: [{word, weight, relation, source}, ...]
    """
    from nltk.corpus import wordnet as wn

    entries = []
    seen = set()

    synsets = wn.synsets(concept, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(concept)

    for ss in synsets:
        # Part meronyms (camel → hump)
        for m in ss.part_meronyms():
            for lemma in m.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    entries.append({"word": word, "weight": 1.0,
                                    "relation": "wn_part_meronym",
                                    "source": "wordnet"})

        # Substance meronyms (candle → wax)
        for m in ss.substance_meronyms():
            for lemma in m.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    entries.append({"word": word, "weight": 1.0,
                                    "relation": "wn_substance_meronym",
                                    "source": "wordnet"})

        # Member holonyms (soldier → army)
        for h in ss.member_holonyms():
            for lemma in h.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    entries.append({"word": word, "weight": 1.0,
                                    "relation": "wn_member_holonym",
                                    "source": "wordnet"})

        # Immediate hypernyms (camel → ungulate)
        for hyp in ss.hypernyms():
            for lemma in hyp.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    entries.append({"word": word, "weight": 0.8,
                                    "relation": "wn_hypernym",
                                    "source": "wordnet"})

        # Also-sees and similar-tos
        for related in ss.also_sees() + ss.similar_tos():
            for lemma in related.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    entries.append({"word": word, "weight": 0.9,
                                    "relation": "wn_related",
                                    "source": "wordnet"})

    return entries


# ---------------------------------------------------------------------------
# Curated nodes (from concept JSONs)
# ---------------------------------------------------------------------------
def fetch_curated_associations(concept):
    """Extract associations from curated_nodes in concept JSONs.

    These are high-quality, manually curated semantic associations
    stored in concepts/*.json. Used as a reliable third source
    alongside ConceptNet and WordNet.
    """
    all_curated = cr.get_curated_nodes()
    nodes = all_curated.get(concept, [])

    entries = []
    seen = set()
    for word in nodes:
        word = word.lower().strip()
        if word == concept or word in seen:
            continue
        if len(word) < 2 or len(word) > 30:
            continue
        seen.add(word)
        entries.append({"word": word, "weight": 1.5,
                        "relation": "curated_node",
                        "source": "curated"})
    return entries


# ---------------------------------------------------------------------------
# Merge and filter
# ---------------------------------------------------------------------------
def merge_associations(cn_entries, wn_entries, curated_entries):
    """Merge ConceptNet, WordNet, and curated_nodes. Deduplicate by word.

    Priority: ConceptNet (real weights) > curated (1.5) > WordNet (0.3-1.0).
    """
    by_word = {}
    for entry in cn_entries:
        if entry["word"] not in by_word:
            by_word[entry["word"]] = entry

    for entry in curated_entries:
        if entry["word"] not in by_word:
            by_word[entry["word"]] = entry

    for entry in wn_entries:
        if entry["word"] not in by_word:
            by_word[entry["word"]] = entry

    merged = sorted(by_word.values(), key=lambda e: e["weight"], reverse=True)
    return merged


def filter_associations(concept, entries):
    """Apply quality filters.

    Removes: concept name, numerics, >2 words, too short/long,
    generic stopwords.
    """
    filtered = []
    for entry in entries:
        word = entry["word"]

        if word == concept:
            continue

        # Quality (build_r1.py:79-91)
        parts = word.split()
        if len(parts) > 2:
            continue
        if len(word) < 2 or len(word) > 30:
            continue
        if word.replace(" ", "").isdigit():
            continue

        if word in GENERIC_SKIP:
            continue

        filtered.append(entry)

    return filtered


# ---------------------------------------------------------------------------
# Fallback for insufficient items
# ---------------------------------------------------------------------------
def handle_insufficient_items(concept, associations):
    """If < MIN_ITEMS, apply WordNet fallback strategies.

    Fallback 1: All POS (verbs, adjectives)
    Fallback 2: Siblings via hypernym tree
    Fallback 3: Gloss (definition) words
    """
    if len(associations) >= MIN_ITEMS_PER_CONCEPT:
        return associations

    from nltk.corpus import wordnet as wn

    seen = {e["word"] for e in associations}

    # Fallback 1: all POS
    print(f"    [FALLBACK-1] All POS for '{concept}' "
          f"(have {len(associations)}, need {MIN_ITEMS_PER_CONCEPT})")
    for pos in [wn.VERB, wn.ADJ, wn.ADV]:
        for ss in wn.synsets(concept, pos=pos):
            for lemma in ss.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if _is_valid_wn_word(word, concept, seen):
                    seen.add(word)
                    associations.append({"word": word, "weight": 0.5,
                                         "relation": "wn_fallback_pos",
                                         "source": "wordnet"})

    if len(associations) >= MIN_ITEMS_PER_CONCEPT:
        return associations

    # Fallback 2: siblings via hypernym
    print(f"    [FALLBACK-2] Siblings for '{concept}' "
          f"(have {len(associations)})")
    for ss in wn.synsets(concept, pos=wn.NOUN)[:2]:
        for hypernym in ss.hypernyms()[:2]:
            for hyponym in hypernym.hyponyms()[:10]:
                for lemma in hyponym.lemmas():
                    word = lemma.name().replace("_", " ").lower()
                    if _is_valid_wn_word(word, concept, seen):
                        seen.add(word)
                        associations.append({"word": word, "weight": 0.4,
                                             "relation": "wn_sibling",
                                             "source": "wordnet"})

    if len(associations) >= MIN_ITEMS_PER_CONCEPT:
        return associations

    # Fallback 3: gloss words
    print(f"    [FALLBACK-3] Gloss mining for '{concept}' "
          f"(have {len(associations)})")
    gloss_stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "may", "might", "shall", "should", "must",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "or", "and", "not", "no", "but", "if", "than", "that",
        "which", "who", "whom", "this", "these", "those", "it",
        "its", "any", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "only",
    }
    for ss in wn.synsets(concept, pos=wn.NOUN)[:3]:
        for word in ss.definition().lower().split():
            word = word.strip(".,;:!?()[]\"'")
            if (word not in gloss_stop
                    and _is_valid_wn_word(word, concept, seen)):
                seen.add(word)
                associations.append({"word": word, "weight": 0.3,
                                     "relation": "wn_gloss",
                                     "source": "wordnet"})

    if len(associations) < MIN_ITEMS_PER_CONCEPT:
        print(f"    [WARNING] '{concept}' has only {len(associations)} "
              f"associations after all fallbacks (need {MIN_ITEMS_PER_CONCEPT})")

    return associations


# ---------------------------------------------------------------------------
# Cross-concept analysis
# ---------------------------------------------------------------------------
def check_cross_concept_overlap(all_associations):
    """Identify words appearing in multiple concepts' associations.

    Returns dict: concept -> set of overlapping words.
    """
    word_to_concepts = defaultdict(set)
    for concept, entries in all_associations.items():
        for entry in entries:
            word_to_concepts[entry["word"]].add(concept)

    overlapping = {w for w, cs in word_to_concepts.items() if len(cs) > 1}

    overlap_per_concept = {}
    for concept, entries in all_associations.items():
        overlap_per_concept[concept] = {
            e["word"] for e in entries if e["word"] in overlapping
        }

    if overlapping:
        print(f"  Cross-concept overlap: {len(overlapping)} words in 2+ concepts")

    return overlap_per_concept


def _is_valid_distractor(word):
    """Check if a word is suitable as a distractor for GPT-2 evaluation.

    Filters out multi-word phrases and very rare words that cause
    artifacts in log-probability scoring (rare tokens get artificially
    high or low scores regardless of semantic fit).
    """
    # Single word only — multi-word phrases like "skilled worker",
    # "bird's foot" cause tokenization artifacts
    if " " in word:
        return False
    # ASCII only
    if not word.isascii():
        return False
    # Minimum length
    if len(word) < 3:
        return False
    # Not numeric
    if word.isdigit():
        return False
    # Word frequency: Zipf >= 2.5 keeps common words,
    # filters "garmentmaker", "bookman", "holdfast" etc.
    try:
        from wordfreq import zipf_frequency
        freq = zipf_frequency(word, "en")
        if freq < 2.5:
            return False
    except ImportError:
        pass
    return True


def build_distractor_pool(all_associations):
    """Build per-concept distractor pools from OTHER concepts.

    For concept X, distractors = union of associations from all concepts != X,
    excluding words that are in X's own associations.
    Filters to single, common English words for clean GPT-2 scoring.
    """
    pools = {}
    for target in all_associations:
        target_words = {e["word"] for e in all_associations[target]}
        distractors = []
        for other, entries in all_associations.items():
            if other == target:
                continue
            for entry in entries:
                if (entry["word"] not in target_words
                        and _is_valid_distractor(entry["word"])):
                    distractors.append(entry["word"])
        pools[target] = list(set(distractors))
    return pools


# ---------------------------------------------------------------------------
# MCQ item builder
# ---------------------------------------------------------------------------
def build_association_items(concept, associations, distractor_pool, overlap_set):
    """Generate BEA semantic association MCQ items for one concept.

    Each item:
      - question: template with concept as target word
      - correct: top association from ConceptNet/WordNet
      - distractors: 3 words from other concepts' associations
    """
    rng = random.Random(42 + hash(concept))
    n_templates = len(ASSOCIATION_TEMPLATES)

    items = []
    for i, entry in enumerate(associations[:TARGET_PER_CONCEPT]):
        template = ASSOCIATION_TEMPLATES[i % n_templates]
        question = template.format(word=concept)

        available = [d for d in distractor_pool if d != entry["word"]]
        if len(available) < 3:
            print(f"    [WARN] '{concept}' has only {len(available)} distractors")
            distractors = available[:3]
        else:
            distractors = rng.sample(available, 3)

        items.append({
            "question": question,
            "options": {
                "correct": entry["word"],
                "distractors": distractors,
            },
            "provenance": {
                "target_word": concept,
                "associated_word": entry["word"],
                "relationship": entry["relation"],
                "concept": concept,
                "test_type": "bea_association",
                "source": entry["source"],
                "weight": entry["weight"],
                "distractor_type": "cross-concept association pool",
                "cross_concept_overlap": entry["word"] in overlap_set,
                "clinical_analog": (
                    "Asociación semántica (BEA) / "
                    "Pyramids and Palm Trees (Howard & Patterson, 1992)"
                ),
            },
        })

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BEA Semantic Association Benchmark (Dynamic)")
    print("=" * 60)
    print(f"Concepts: {len(cfg.CONCEPTS)}")
    print(f"Target items per concept: {TARGET_PER_CONCEPT}")
    print(f"Data sources: ConceptNet 5.7 (HuggingFace) + WordNet + curated_nodes")
    print()

    # ---- Phase 1: Load ConceptNet index (one-time scan, cached) ----
    print("--- Phase 1: Loading ConceptNet index ---")
    cn_index = _load_conceptnet_index(cfg.CONCEPTS)

    # ---- Phase 2: Fetch per-concept associations ----
    print("\n--- Phase 2: Fetching associations per concept ---")
    raw_associations = {}

    for concept in cfg.CONCEPTS:
        print(f"\n  [{concept}]")

        # ConceptNet (from pre-built index)
        cn_entries = cn_index.get(concept, [])
        print(f"    ConceptNet: {len(cn_entries)} associations")

        # WordNet
        wn_entries = fetch_wordnet_associations(concept)
        print(f"    WordNet:    {len(wn_entries)} associations")

        # Curated nodes (from concepts/*.json)
        curated_entries = fetch_curated_associations(concept)
        print(f"    Curated:    {len(curated_entries)} associations")

        # Merge (ConceptNet > curated > WordNet)
        merged = merge_associations(cn_entries, wn_entries, curated_entries)
        print(f"    Merged:     {len(merged)} unique")

        # Filter
        filtered = filter_associations(concept, merged)
        print(f"    Filtered:   {len(filtered)} "
              f"(removed {len(merged) - len(filtered)})")

        # Fallback if insufficient
        filtered = handle_insufficient_items(concept, filtered)

        raw_associations[concept] = filtered

    # ---- Phase 3: Cross-concept analysis ----
    print(f"\n{'='*50}")
    print("--- Phase 3: Cross-concept analysis ---")
    overlap_sets = check_cross_concept_overlap(raw_associations)
    distractor_pools = build_distractor_pool(raw_associations)

    for concept in cfg.CONCEPTS:
        pool_size = len(distractor_pools.get(concept, []))
        n_assoc = len(raw_associations[concept])
        n_overlap = len(overlap_sets.get(concept, set()))
        print(f"  {concept}: {n_assoc} assoc, "
              f"{pool_size} distractors, {n_overlap} overlapping")

    # ---- Phase 4: Build items ----
    print(f"\n{'='*50}")
    print("--- Phase 4: Building benchmark items ---")
    benchmark = {}
    all_items = []

    for concept in cfg.CONCEPTS:
        associations = raw_associations[concept]
        pool = distractor_pools.get(concept, [])
        overlaps = overlap_sets.get(concept, set())

        items = build_association_items(concept, associations, pool, overlaps)
        benchmark[concept] = items
        all_items.extend(items)

        n_cn = sum(1 for it in items
                   if it["provenance"]["source"] == "conceptnet")
        n_cur = sum(1 for it in items
                    if it["provenance"]["source"] == "curated")
        n_wn = sum(1 for it in items
                   if it["provenance"]["source"] == "wordnet")
        print(f"  {concept}: {len(items)} items "
              f"({n_cn} CN, {n_cur} curated, {n_wn} WN)")

    # ---- Phase 5: Save ----
    print(f"\n{'='*50}")
    print("--- Phase 5: Saving ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "bea_association_benchmark.json")

    concepts_below = [
        c for c in cfg.CONCEPTS
        if len(benchmark.get(c, [])) < MIN_ITEMS_PER_CONCEPT
    ]

    output = {
        "metadata": {
            "benchmark": "BEA_Semantic_Association",
            "clinical_analog": (
                "Asociación semántica (BEA) / "
                "Pyramids and Palm Trees (Howard & Patterson, 1992)"
            ),
            "description": (
                "Semantic association test: given a target entity concept, "
                "select the most closely associated word from options. "
                "Associations sourced from ConceptNet 5.7 (HuggingFace), "
                "WordNet, and curated nodes. Distractors are words "
                "associated with OTHER entity concepts, filtered for "
                "word frequency (Zipf >= 2.5) for clean LM scoring."
            ),
            "format": "Compatible with evaluate_bea_association() in metrics.py",
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in cfg.CONCEPTS},
            "data_sources": [
                "ConceptNet 5.7 (HuggingFace datasets)",
                "WordNet (NLTK)",
                "curated_nodes",
            ],
            "distractor_type": "cross-concept association pool",
            "concepts_below_minimum": concepts_below,
            "thresholds": {
                "min_conceptnet_weight": MIN_WEIGHT,
                "target_per_concept": TARGET_PER_CONCEPT,
                "min_per_concept": MIN_ITEMS_PER_CONCEPT,
            },
            "references": [
                "Cuetos & González-Nosti (2009)",
                "Howard & Patterson (1992) Pyramids and Palm Trees",
                "Jefferies & Lambon Ralph (2006) Semantic aphasia",
                "Speer et al. (2017) ConceptNet 5.5",
                "Miller (1995) WordNet",
            ],
        },
        "items": benchmark,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"Total items: {len(all_items)}")

    if concepts_below:
        print(f"\n[WARNING] {len(concepts_below)} concepts below minimum "
              f"({MIN_ITEMS_PER_CONCEPT}): {concepts_below}")

    print("Done.")


if __name__ == "__main__":
    main()
