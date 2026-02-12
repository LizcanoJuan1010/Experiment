"""
Corpus-Based CAV Training Data Extractor
=========================================
Extracts real sentences from established linguistic corpora to build
training data for CAV (Concept Activation Vector) extraction.

Methodology:
  1. Extract sentences containing concept words from NLTK Brown/Gutenberg
     corpora and optional PDFs.
  2. Filter for short (8-20 word), grammatically clear sentences using
     spaCy dependency parsing.
  3. Validate concept word category membership via WordNet hypernym
     chains (Miller, 1995).
  4. Generate minimal-pair negatives: same sentence with concept word
     replaced by a frequency-matched neutral word (Kim et al., 2018).

References:
    - Francis & Kučera (1979) "Frequency Analysis of English Usage"
    - Miller (1995) "WordNet: A Lexical Database for English"
    - Kim et al. (2018) "Interpretability Beyond Feature Attribution"
    - Conneau et al. (2018) "What you can cram into a single vector"

Usage:
    python corpus_extractor.py

Input:  experiment_config.py (parameters), CURATED_NODES from data_gen.py
Output: corpus_training_report.json (extraction statistics)
        Training data returned via build_corpus_training_data()
"""

import json
import os
import random
import re
from collections import Counter

import experiment_config as cfg

random.seed(cfg.RANDOM_SEED)


# ---------------------------------------------------------------------------
# NLTK Corpus Extraction
# ---------------------------------------------------------------------------

def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    import nltk
    for resource in ["brown", "gutenberg", "punkt", "punkt_tab",
                     "wordnet", "averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource in
                           ["brown", "gutenberg", "wordnet"]
                           else f"tokenizers/{resource}" if "punkt" in resource
                           else f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def extract_from_nltk_brown(concept_words_by_category):
    """
    Extract sentences from the Brown Corpus (Francis & Kučera, 1979).

    The Brown Corpus contains ~500 texts across 15 genres, totaling
    ~1 million words. It is the most widely used balanced corpus in
    computational linguistics.

    Args:
        concept_words_by_category: dict mapping category -> list of words

    Returns:
        list of dicts: [{"sentence": str, "concept_word": str,
                         "source": "brown", "category": str}]
    """
    from nltk.corpus import brown

    # Build reverse lookup: word -> category
    word_to_category = {}
    for category, words in concept_words_by_category.items():
        for w in words:
            word_to_category[w.lower()] = category

    all_words = set(word_to_category.keys())
    results = []

    for sent_tokens in brown.sents():
        sent_lower = [t.lower() for t in sent_tokens]
        # Check if any concept word appears in the sentence
        matches = all_words & set(sent_lower)
        if not matches:
            continue

        sentence = " ".join(sent_tokens)
        n_words = len(sent_tokens)

        # Length filter
        if n_words < cfg.CORPUS_MIN_SENTENCE_LEN:
            continue
        if n_words > cfg.CORPUS_MAX_SENTENCE_LEN:
            continue

        # Basic quality: no excessive punctuation or special chars
        if _has_quality_issues(sentence):
            continue

        for word in matches:
            results.append({
                "sentence": sentence,
                "concept_word": word,
                "source": "brown",
                "category": word_to_category[word],
            })

    return results


def extract_from_nltk_gutenberg(concept_words_by_category):
    """
    Extract sentences from the Gutenberg Corpus (18 classic English texts).

    Complementary source to Brown — provides literary prose which tends
    to have clearer grammatical structures.

    Args:
        concept_words_by_category: dict mapping category -> list of words

    Returns:
        list of dicts with same schema as extract_from_nltk_brown()
    """
    from nltk.corpus import gutenberg
    from nltk.tokenize import sent_tokenize, word_tokenize

    word_to_category = {}
    for category, words in concept_words_by_category.items():
        for w in words:
            word_to_category[w.lower()] = category

    all_words = set(word_to_category.keys())
    results = []

    for fileid in gutenberg.fileids():
        raw_text = gutenberg.raw(fileid)
        sentences = sent_tokenize(raw_text)

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokens_lower = [t.lower() for t in tokens]
            matches = all_words & set(tokens_lower)
            if not matches:
                continue

            n_words = len(tokens)
            if n_words < cfg.CORPUS_MIN_SENTENCE_LEN:
                continue
            if n_words > cfg.CORPUS_MAX_SENTENCE_LEN:
                continue

            if _has_quality_issues(sentence):
                continue

            # Normalize whitespace
            sentence = " ".join(tokens)

            for word in matches:
                results.append({
                    "sentence": sentence,
                    "concept_word": word,
                    "source": "gutenberg",
                    "category": word_to_category[word],
                })

    return results


def extract_from_pdf(pdf_path, concept_words_by_category):
    """
    Extract sentences from a PDF file using PyMuPDF.

    Allows users to add domain-specific sources (textbooks, papers).

    Args:
        pdf_path: path to PDF file
        concept_words_by_category: dict mapping category -> list of words

    Returns:
        list of dicts with same schema as other extractors
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"  WARNING: PyMuPDF not installed. Skipping PDF: {pdf_path}")
        return []

    from nltk.tokenize import sent_tokenize, word_tokenize

    if not os.path.exists(pdf_path):
        print(f"  WARNING: PDF not found: {pdf_path}")
        return []

    word_to_category = {}
    for category, words in concept_words_by_category.items():
        for w in words:
            word_to_category[w.lower()] = category

    all_words = set(word_to_category.keys())
    results = []

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + " "
    doc.close()

    sentences = sent_tokenize(full_text)
    for sentence in sentences:
        # Clean up PDF artifacts
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        tokens = word_tokenize(sentence)
        tokens_lower = [t.lower() for t in tokens]

        matches = all_words & set(tokens_lower)
        if not matches:
            continue

        n_words = len(tokens)
        if n_words < cfg.CORPUS_MIN_SENTENCE_LEN:
            continue
        if n_words > cfg.CORPUS_MAX_SENTENCE_LEN:
            continue

        if _has_quality_issues(sentence):
            continue

        sentence = " ".join(tokens)
        for word in matches:
            results.append({
                "sentence": sentence,
                "concept_word": word,
                "source": f"pdf:{os.path.basename(pdf_path)}",
                "category": word_to_category[word],
            })

    return results


def _has_quality_issues(sentence):
    """Check for common quality issues in extracted sentences."""
    # Too many non-alpha characters (tables, equations, etc.)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in sentence) / max(len(sentence), 1)
    if alpha_ratio < 0.75:
        return True
    # Contains URLs or email-like patterns
    if re.search(r'http|www\.|@.*\.', sentence):
        return True
    # Contains excessive capitalization (headers, titles)
    words = sentence.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    if caps_ratio > 0.3:
        return True
    return False


# ---------------------------------------------------------------------------
# Linguistic Filtering with spaCy
# ---------------------------------------------------------------------------

def _load_spacy():
    """Load spaCy model, downloading if necessary."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("  Downloading spaCy en_core_web_sm model...")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def filter_sentences(raw_sentences, nlp=None):
    """
    Filter sentences using spaCy dependency parsing.

    Ensures:
    1. The concept word is in a meaningful grammatical position
       (subject, direct object, or prepositional object)
    2. The sentence has a clear SVO structure (has a root verb and subject)

    Args:
        raw_sentences: list of dicts from extraction functions
        nlp: pre-loaded spaCy model (loaded if None)

    Returns:
        list of dicts, each augmented with "grammatical_position" key
    """
    if nlp is None:
        nlp = _load_spacy()

    accepted_deps = set(cfg.CORPUS_GRAMMATICAL_POSITIONS)
    dep_to_position = {
        "nsubj": "subject",
        "nsubjpass": "subject",
        "dobj": "object",
        "pobj": "prepositional",
        "attr": "object",
    }

    filtered = []

    for item in raw_sentences:
        doc = nlp(item["sentence"])
        concept_word = item["concept_word"].lower()

        # Find the concept word token
        concept_token = None
        for token in doc:
            if token.text.lower() == concept_word:
                concept_token = token
                break

        if concept_token is None:
            continue

        # Check grammatical position
        dep = concept_token.dep_
        if dep not in accepted_deps:
            # Also accept if the concept is part of a compound whose
            # head is in an accepted position
            if concept_token.dep_ == "compound" and concept_token.head.dep_ in accepted_deps:
                dep = concept_token.head.dep_
            else:
                continue

        # Check SVO structure if required
        if cfg.CORPUS_REQUIRE_SVO:
            has_root = any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in doc)
            has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc)
            if not (has_root and has_subj):
                continue

        position = dep_to_position.get(dep, "other")
        result = dict(item)
        result["grammatical_position"] = position
        filtered.append(result)

    return filtered


# ---------------------------------------------------------------------------
# WordNet Validation
# ---------------------------------------------------------------------------

def validate_concept_wordnet(word, target_category):
    """
    Validate that a word belongs to a semantic category using WordNet
    hypernym chains (Miller, 1995).

    Traverses the hypernym tree for each synset of the word and checks
    if any target root synset is an ancestor.

    Args:
        word: the concept word to validate
        target_category: one of cfg.CONCEPTS (e.g., "time", "place", "tools")

    Returns:
        dict with keys: valid, synsets_checked, matching_synset,
        hypernym_path, confidence
    """
    from nltk.corpus import wordnet as wn

    target_roots = cfg.WORDNET_HYPERNYM_ROOTS.get(target_category, [])
    target_synsets = []
    for root_name in target_roots:
        try:
            target_synsets.append(wn.synset(root_name))
        except Exception:
            pass

    if not target_synsets:
        return {"valid": False, "confidence": "no_roots", "synsets_checked": 0}

    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return {"valid": False, "confidence": "no_synsets", "synsets_checked": 0}

    for ss in synsets:
        # Get all hypernym paths
        for path in ss.hypernym_paths():
            for ancestor in path:
                if ancestor in target_synsets:
                    return {
                        "valid": True,
                        "synsets_checked": len(synsets),
                        "matching_synset": ss.name(),
                        "hypernym_path": [s.name() for s in path[:5]],
                        "confidence": "high",
                    }

    # Check Lowest Common Hypernym as fallback
    for ss in synsets:
        for target in target_synsets:
            lch = ss.lowest_common_hypernyms(target)
            if lch and lch[0].min_depth() >= 3:
                return {
                    "valid": True,
                    "synsets_checked": len(synsets),
                    "matching_synset": ss.name(),
                    "lch": lch[0].name(),
                    "confidence": "medium",
                }

    return {
        "valid": False,
        "synsets_checked": len(synsets),
        "confidence": "unverified",
    }


def validate_all_concept_words(concept_nodes):
    """
    Validate all concept words against their target category using WordNet.

    Args:
        concept_nodes: dict mapping category -> list of words

    Returns:
        dict with per-category and overall validation statistics
    """
    if not cfg.WORDNET_VALIDATE:
        return {"skipped": True}

    _ensure_nltk_data()
    report = {}

    for category, words in concept_nodes.items():
        results = {}
        valid_count = 0
        for word in words:
            result = validate_concept_wordnet(word, category)
            results[word] = result
            if result["valid"]:
                valid_count += 1

        coverage = valid_count / len(words) if words else 0.0
        report[category] = {
            "total_words": len(words),
            "validated": valid_count,
            "coverage": round(coverage, 3),
            "meets_threshold": coverage >= cfg.WORDNET_MIN_COVERAGE,
            "details": results,
        }
        status = "OK" if coverage >= cfg.WORDNET_MIN_COVERAGE else "WARNING"
        print(f"  {category.upper()}: {valid_count}/{len(words)} "
              f"validated ({coverage:.1%}) [{status}]")

    total_words = sum(r["total_words"] for r in report.values())
    total_valid = sum(r["validated"] for r in report.values())
    report["overall"] = {
        "total_words": total_words,
        "validated": total_valid,
        "coverage": round(total_valid / total_words, 3) if total_words else 0.0,
    }

    return report


# ---------------------------------------------------------------------------
# Minimal Pair Generation
# ---------------------------------------------------------------------------

def _get_zipf_frequency(word):
    """Return Zipf-scale frequency for a word, or None if unavailable."""
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word, "en")
    except ImportError:
        return None


def _select_neutral_replacement(concept_word, neutral_pool, concept_is_plural=False):
    """
    Select a frequency-matched, length-matched, and number-matched neutral word.

    Matching criteria (Conneau et al., 2018):
      - Length: within cfg.WORD_LENGTH_TOLERANCE characters
      - Frequency: within cfg.FREQUENCY_TOLERANCE Zipf-scale units
      - Grammatical number: singular/plural concordance (FIX-C3)

    Returns None if no suitable replacement found (caller should skip sentence).
    """
    target_len = len(concept_word)
    target_freq = _get_zipf_frequency(concept_word)

    candidates = []
    for w in neutral_pool:
        # Length match
        if abs(len(w) - target_len) > cfg.WORD_LENGTH_TOLERANCE:
            continue
        # Rough number agreement: if concept is plural (ends in s),
        # prefer neutral words that also end in s, and vice versa
        if concept_is_plural and not w.endswith("s"):
            continue
        if not concept_is_plural and w.endswith("s") and len(w) > 3:
            continue
        # Frequency match (if available)
        if target_freq is not None and cfg.ENABLE_FREQUENCY_MATCHING:
            w_freq = _get_zipf_frequency(w)
            if w_freq is not None and abs(w_freq - target_freq) > cfg.FREQUENCY_TOLERANCE:
                continue
        candidates.append(w)

    if candidates:
        return random.choice(candidates)

    # Relax: drop number constraint but keep length + frequency
    relaxed = []
    for w in neutral_pool:
        if abs(len(w) - target_len) > cfg.WORD_LENGTH_TOLERANCE + 1:
            continue
        if target_freq is not None and cfg.ENABLE_FREQUENCY_MATCHING:
            w_freq = _get_zipf_frequency(w)
            if w_freq is not None and abs(w_freq - target_freq) > cfg.FREQUENCY_TOLERANCE * 1.5:
                continue
        relaxed.append(w)

    if relaxed:
        return random.choice(relaxed)

    # No good match found — return None so caller can skip this sentence
    return None


def create_minimal_pairs(filtered_sentences, neutral_pool, nlp=None,
                         concept_words_lookup=None):
    """
    Generate minimal-pair negatives for each positive sentence.

    For each sentence containing a concept word, the negative is the
    same sentence with the concept word replaced by a frequency-matched,
    number-matched neutral word. This is the gold standard for CAV
    training data (Kim et al., 2018) — it ensures the CAV captures
    semantic content rather than surface-form differences.

    FIX-C3: Uses spaCy morphological analysis to determine concept word
    plurality and enforce grammatical number agreement in replacements.
    Sentences where no suitable replacement exists are skipped rather
    than using a low-quality fallback.

    Args:
        filtered_sentences: list of dicts from filter_sentences()
        neutral_pool: list of neutral words
        nlp: pre-loaded spaCy model (loaded if None)
        concept_words_lookup: dict mapping category -> list of concept words.
            Used to filter out negatives that still contain same-category
            concept words (self-contamination prevention).

    Returns:
        dict with keys per category: {"positive": [...], "negative": [...],
        "pairs_metadata": [...]}
    """
    if nlp is None:
        nlp = _load_spacy()

    # Build per-category word sets for self-contamination check
    category_word_sets = {}
    if concept_words_lookup:
        for cat, words in concept_words_lookup.items():
            category_word_sets[cat] = {w.lower() for w in words}

    training = {}
    metadata = {}
    skipped_no_replacement = 0
    skipped_self_contaminated = 0

    # Group by category
    by_category = {}
    for item in filtered_sentences:
        cat = item["category"]
        by_category.setdefault(cat, []).append(item)

    for category, items in by_category.items():
        positives = []
        negatives = []
        pairs_meta = []
        own_words = category_word_sets.get(category, set())

        for item in items:
            sentence = item["sentence"]
            concept_word = item["concept_word"]

            # FIX-C3: Use spaCy to determine grammatical number
            doc = nlp(sentence)
            concept_is_plural = False
            for token in doc:
                if token.text.lower() == concept_word.lower():
                    # Check spaCy morphological features for Number=Plur
                    morph = token.morph.get("Number")
                    if morph and "Plur" in morph:
                        concept_is_plural = True
                    break

            replacement = _select_neutral_replacement(
                concept_word, neutral_pool, concept_is_plural=concept_is_plural
            )

            # FIX-C3: Skip sentence if no suitable replacement found
            if replacement is None:
                skipped_no_replacement += 1
                continue

            # Replace concept word with neutral word (case-insensitive)
            pattern = re.compile(re.escape(concept_word), re.IGNORECASE)
            negative_sentence = pattern.sub(replacement, sentence, count=1)

            # Self-contamination check: reject negatives that still contain
            # other concept words from the same category
            if own_words:
                neg_lower = f" {negative_sentence.lower()} "
                still_has_concept = False
                for cw in own_words:
                    if f" {cw} " in neg_lower or f" {cw}." in neg_lower:
                        still_has_concept = True
                        break
                if still_has_concept:
                    skipped_self_contaminated += 1
                    continue

            # Only add if replacement actually changed the sentence
            if negative_sentence != sentence:
                positives.append(sentence)
                negatives.append(negative_sentence)
                pairs_meta.append({
                    "concept_word": concept_word,
                    "replacement": replacement,
                    "concept_is_plural": concept_is_plural,
                    "source": item["source"],
                    "position": item.get("grammatical_position", "unknown"),
                })

        # Shuffle pairs in sync
        if positives:
            combined = list(zip(positives, negatives, pairs_meta))
            random.shuffle(combined)
            positives, negatives, pairs_meta = zip(*combined)
            positives = list(positives)
            negatives = list(negatives)
            pairs_meta = list(pairs_meta)

        training[category] = {
            "positive": positives,
            "negative": negatives,
        }
        metadata[category] = {
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "pairs_metadata": pairs_meta,
        }

    if skipped_no_replacement > 0:
        print(f"  Skipped {skipped_no_replacement} sentences (no suitable "
              f"neutral replacement with number agreement)")
    if skipped_self_contaminated > 0:
        print(f"  Skipped {skipped_self_contaminated} sentences (negative still "
              f"contains same-category concept word)")

    return training, metadata


# ---------------------------------------------------------------------------
# Balance by Grammatical Position
# ---------------------------------------------------------------------------

def _balance_by_position(filtered_sentences, target_weights=None):
    """
    Balance sentences across grammatical positions.

    Target distribution follows cfg.TEMPLATE_POSITION_WEIGHTS:
    subject ~35%, object ~35%, prepositional ~30%.
    """
    if target_weights is None:
        target_weights = cfg.TEMPLATE_POSITION_WEIGHTS

    by_cat_pos = {}
    for item in filtered_sentences:
        cat = item["category"]
        pos = item.get("grammatical_position", "other")
        by_cat_pos.setdefault(cat, {}).setdefault(pos, []).append(item)

    balanced = []
    balance_stats = {}

    max_pos_ratio = getattr(cfg, "CORPUS_MAX_POSITION_RATIO", 0.50)

    for cat, pos_items in by_cat_pos.items():
        total = sum(len(v) for v in pos_items.values())
        selected = []

        for pos, weight in target_weights.items():
            target_n = int(total * weight)
            available = pos_items.get(pos, [])
            if len(available) <= target_n:
                selected.extend(available)
            else:
                selected.extend(random.sample(available, target_n))

        # Enforce max position ratio: downsample any position exceeding cap.
        # To keep pos ≤ R of total: pos ≤ other_total * R / (1 - R)
        n_selected = len(selected)
        if n_selected > 0:
            pos_groups = {}
            for item in selected:
                p = item.get("grammatical_position", "other")
                pos_groups.setdefault(p, []).append(item)
            for pos, items_in_pos in pos_groups.items():
                ratio = len(items_in_pos) / n_selected
                if ratio > max_pos_ratio:
                    other_total = n_selected - len(items_in_pos)
                    max_allowed = int(other_total * max_pos_ratio
                                      / (1 - max_pos_ratio))
                    max_allowed = max(max_allowed, 1)
                    keep = random.sample(items_in_pos, max_allowed)
                    selected = [it for it in selected
                                if it.get("grammatical_position") != pos]
                    selected.extend(keep)
                    n_selected = len(selected)

        balanced.extend(selected)
        pos_counts = Counter(item.get("grammatical_position", "other")
                             for item in selected)
        balance_stats[cat] = {
            "total_before": total,
            "total_after": len(selected),
            "position_distribution": dict(pos_counts),
        }

    return balanced, balance_stats


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def build_corpus_training_data(concept_nodes):
    """
    Build CAV training data from real corpus sentences.

    Pipeline:
    1. Extract from Brown + Gutenberg (+ optional PDFs)
    2. Filter with spaCy dependency parsing
    3. Validate concept words with WordNet
    4. Balance by grammatical position
    5. Generate minimal-pair negatives

    If a concept doesn't reach cfg.CORPUS_MIN_SENTENCES, falls back to
    template generation from data_gen.py.

    Args:
        concept_nodes: dict mapping category -> list of concept words

    Returns:
        tuple: (training_data, metadata) compatible with data_gen.generate_cav_training()
    """
    print("\n" + "=" * 60)
    print("CORPUS-BASED CAV TRAINING DATA EXTRACTION")
    print("=" * 60)

    _ensure_nltk_data()

    # Build concept words lookup.
    # If USE_CONCEPTNET_FOR_CAV_TRAINING is False (default), use ONLY
    # curated nodes to avoid ConceptNet noise (observed: 7696 place nodes
    # with only 16% WordNet validation).
    if not getattr(cfg, "USE_CONCEPTNET_FOR_CAV_TRAINING", False):
        from data_gen import CURATED_NODES
        concept_words = {}
        for cat in cfg.CONCEPTS:
            concept_words[cat] = list(CURATED_NODES.get(cat, []))
        print("  Using CURATED_NODES only (USE_CONCEPTNET_FOR_CAV_TRAINING=False)")
    else:
        concept_words = {}
        for cat in cfg.CONCEPTS:
            concept_words[cat] = concept_nodes.get(cat, [])

    # --- Step 1: Extract from corpora ---
    print("\n--- Step 1: Corpus Extraction ---")
    raw_sentences = []

    if "brown" in cfg.CORPUS_SOURCES:
        print("  Extracting from Brown Corpus...")
        brown_sents = extract_from_nltk_brown(concept_words)
        raw_sentences.extend(brown_sents)
        print(f"    Found {len(brown_sents)} candidate sentences")

    if "gutenberg" in cfg.CORPUS_SOURCES:
        print("  Extracting from Gutenberg Corpus...")
        gutenberg_sents = extract_from_nltk_gutenberg(concept_words)
        raw_sentences.extend(gutenberg_sents)
        print(f"    Found {len(gutenberg_sents)} candidate sentences")

    for pdf_path in cfg.CORPUS_PDF_PATHS:
        print(f"  Extracting from PDF: {pdf_path}...")
        pdf_sents = extract_from_pdf(pdf_path, concept_words)
        raw_sentences.extend(pdf_sents)
        print(f"    Found {len(pdf_sents)} candidate sentences")

    # Deduplicate by sentence text
    seen = set()
    unique = []
    for item in raw_sentences:
        key = (item["sentence"], item["concept_word"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    raw_sentences = unique

    print(f"\n  Total unique candidate sentences: {len(raw_sentences)}")
    for cat in cfg.CONCEPTS:
        n = sum(1 for s in raw_sentences if s["category"] == cat)
        print(f"    {cat.upper()}: {n}")

    # --- Step 2: Filter with spaCy ---
    print("\n--- Step 2: Linguistic Filtering (spaCy) ---")
    nlp = _load_spacy()
    filtered = filter_sentences(raw_sentences, nlp)
    print(f"  After filtering: {len(filtered)} sentences")
    for cat in cfg.CONCEPTS:
        n = sum(1 for s in filtered if s["category"] == cat)
        print(f"    {cat.upper()}: {n}")

    # --- Step 3: WordNet validation ---
    # Validate the words actually used for training (concept_words),
    # not the full ConceptNet-enriched list (concept_nodes).
    print("\n--- Step 3: WordNet Validation ---")
    wordnet_report = validate_all_concept_words(concept_words)

    # --- Step 4: Balance by position ---
    print("\n--- Step 4: Grammatical Position Balancing ---")
    balanced, balance_stats = _balance_by_position(filtered)
    print(f"  After balancing: {len(balanced)} sentences")
    for cat, stats in balance_stats.items():
        dist = stats["position_distribution"]
        print(f"    {cat.upper()}: {stats['total_after']} "
              f"(subj={dist.get('subject', 0)}, "
              f"obj={dist.get('object', 0)}, "
              f"prep={dist.get('prepositional', 0)})")

    # --- Step 5: Generate minimal pairs ---
    print("\n--- Step 5: Minimal Pair Generation ---")

    # Build neutral word pool (reuse from data_gen)
    from data_gen import NEUTRAL_WORDS
    all_concept_words = set()
    for words in concept_nodes.values():
        all_concept_words.update(w.lower() for w in words)
    neutral_pool = [w for w in NEUTRAL_WORDS if w.lower() not in all_concept_words]

    training, pairs_metadata = create_minimal_pairs(
        balanced, neutral_pool, nlp=nlp, concept_words_lookup=concept_words
    )

    # --- Step 6: Check if fallback to templates is needed ---
    needs_template_fallback = False
    for cat in cfg.CONCEPTS:
        n = len(training.get(cat, {}).get("positive", []))
        if n < cfg.CORPUS_MIN_SENTENCES:
            print(f"\n  WARNING: {cat.upper()} has only {n} sentences "
                  f"(min={cfg.CORPUS_MIN_SENTENCES}). "
                  f"Supplementing with template-generated sentences.")
            needs_template_fallback = True

    if needs_template_fallback:
        training, pairs_metadata = _supplement_with_templates(
            training, pairs_metadata, concept_nodes, neutral_pool
        )

    # --- Build metadata ---
    metadata = {
        "mode": "corpus",
        "sources": cfg.CORPUS_SOURCES,
        "pdf_sources": cfg.CORPUS_PDF_PATHS,
        "raw_extracted": len(raw_sentences),
        "after_spacy_filter": len(filtered),
        "after_balancing": len(balanced),
        "wordnet_validation": {
            cat: {
                "coverage": wordnet_report.get(cat, {}).get("coverage", 0),
                "meets_threshold": wordnet_report.get(cat, {}).get(
                    "meets_threshold", False),
            }
            for cat in cfg.CONCEPTS
        } if isinstance(wordnet_report, dict) and "skipped" not in wordnet_report else "skipped",
        "balance_stats": balance_stats,
        "template_fallback_used": needs_template_fallback,
    }

    for cat in cfg.CONCEPTS:
        cat_meta = pairs_metadata.get(cat, {})
        n_pos = len(training.get(cat, {}).get("positive", []))
        n_neg = len(training.get(cat, {}).get("negative", []))

        # Count sources
        source_counts = Counter()
        for pm in cat_meta.get("pairs_metadata", []):
            source_counts[pm.get("source", "unknown")] += 1

        metadata[cat] = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "sources": dict(source_counts),
        }
        print(f"\n  {cat.upper()}: {n_pos} positive, {n_neg} negative")
        for src, count in source_counts.items():
            print(f"    {src}: {count}")

    # Save corpus extraction report
    report_path = cfg.CORPUS_REPORT_FILE
    with open(report_path, "w") as f:
        # Exclude pairs_metadata details (too verbose for report)
        report_meta = {k: v for k, v in metadata.items()}
        json.dump(report_meta, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    return training, metadata


def _supplement_with_templates(training, pairs_metadata, concept_nodes, neutral_pool):
    """
    Supplement corpus-extracted sentences with template-generated ones
    when corpus extraction doesn't yield enough sentences.
    """
    from data_gen import (ALL_TEMPLATES, TEMPLATE_POSITIONS,
                          sample_templates_weighted)

    for cat in cfg.CONCEPTS:
        current_n = len(training.get(cat, {}).get("positive", []))
        needed = cfg.CORPUS_MIN_SENTENCES - current_n

        if needed <= 0:
            continue

        nodes = concept_nodes.get(cat, [])
        if not nodes:
            continue

        positives = list(training.get(cat, {}).get("positive", []))
        negatives = list(training.get(cat, {}).get("negative", []))
        meta_pairs = list(pairs_metadata.get(cat, {}).get("pairs_metadata", []))

        # Generate template sentences to fill the gap
        generated = 0
        word_idx = 0
        max_attempts = needed * 5  # Prevent infinite loop
        attempts = 0
        while generated < needed and attempts < max_attempts:
            word = nodes[word_idx % len(nodes)]
            templates = sample_templates_weighted(
                1, ALL_TEMPLATES, TEMPLATE_POSITIONS,
                cfg.TEMPLATE_POSITION_WEIGHTS,
            )
            for tmpl in templates:
                replacement = _select_neutral_replacement(word, neutral_pool)
                if replacement is None:
                    attempts += 1
                    continue

                pos_sent = tmpl.format(word=word)
                neg_sent = tmpl.format(word=replacement)

                positives.append(pos_sent)
                negatives.append(neg_sent)
                meta_pairs.append({
                    "concept_word": word,
                    "replacement": replacement,
                    "source": "template_fallback",
                    "position": TEMPLATE_POSITIONS.get(tmpl, "unknown"),
                })
                generated += 1
                attempts += 1
                if generated >= needed:
                    break
            word_idx += 1

        training[cat] = {"positive": positives, "negative": negatives}
        pairs_metadata[cat] = {
            "n_positive": len(positives),
            "n_negative": len(negatives),
            "pairs_metadata": meta_pairs,
        }
        print(f"    {cat.upper()}: added {generated} template sentences "
              f"(total now {len(positives)})")

    return training, pairs_metadata


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run corpus extraction standalone for testing."""
    print("=" * 60)
    print("Corpus Extractor — Standalone Test")
    print("=" * 60)

    # Build concept nodes from data_gen curated lists
    from data_gen import CURATED_NODES, build_concept_nodes, load_conceptnet_supplement

    conceptnet_data = load_conceptnet_supplement()
    concept_nodes = build_concept_nodes(conceptnet_data)

    training, metadata = build_corpus_training_data(concept_nodes)

    # Print sample sentences
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES")
    print("=" * 60)
    for cat in cfg.CONCEPTS:
        pos = training.get(cat, {}).get("positive", [])
        neg = training.get(cat, {}).get("negative", [])
        print(f"\n  {cat.upper()} (showing first 5):")
        for i in range(min(5, len(pos))):
            print(f"    + {pos[i]}")
            print(f"    - {neg[i]}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
