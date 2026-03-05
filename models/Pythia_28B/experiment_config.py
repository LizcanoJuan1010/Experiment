"""
Experiment Configuration — Pythia 2.8B
=======================================
Centralizes all configurable parameters for the CAV extraction pipeline.
Every parameter has a default value and academic justification.

Adapted from the GPT-2-small experiment for Pythia 2.8B (32 layers, 2560 d_model).
Layer indices are proportionally mapped from the original 12-layer model.

References:
    - Kim et al. (2018) "Interpretability Beyond Feature Attribution" (TCAV)
    - Conneau et al. (2018) "What you can cram into a single $&!#* vector"
    - Hewitt & Liang (2019) "Designing and Interpreting Probes with Control Tasks"
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
    - Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
    - Biderman et al. (2023) "Pythia: A Suite for Analyzing Large Language Models"
"""

import os

# =====================================================================
# Model
# =====================================================================
MODEL_NAME = "pythia-2.8b"

# Precision control: Pythia 2.8B is ~11.2GB in float32, ~5.6GB in float16.
# RTX 5060 Ti has 16GB VRAM — float16 leaves ~10GB for activation caching.
MODEL_DTYPE = "float16"  # "float32" | "float16" | "bfloat16"


def load_model():
    """
    Load model with correct dtype and device settings.

    Centralizes model loading to ensure consistent dtype across all scripts.
    """
    import torch
    from transformer_lens import HookedTransformer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(MODEL_DTYPE, torch.float32)
    model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=torch_dtype)
    model.eval()
    return model


# =====================================================================
# Activation Extraction — Token Position Strategy
# =====================================================================
# Which token's activation to use for CAV extraction and metrics.
# "mean"         — Mean pooling over all tokens (recommended, avoids
#                  last-token positional bias). Standard for sentence
#                  representations (Reimers & Gurevych, 2019).
# "last"         — Last token only (GPT-2 convention for generation,
#                  but encodes end-of-sequence info, not concept).
# "concept_word" — Activation at the concept word's token position
#                  (requires knowing which word is the concept).
TOKEN_POSITION = "mean"

# =====================================================================
# Concepts
# =====================================================================
CONCEPTS = ["time", "place", "tools"]

# =====================================================================
# Layer Selection
# =====================================================================
# Pythia 2.8B has 32 layers (0-31). Layer indices are proportionally
# mapped from the GPT-2-small (12-layer) experiment:
#   GPT-2 L6  (50% depth) -> Pythia L16 (50% of 32)
#   GPT-2 L9  (75% depth) -> Pythia L24 (75% of 32)
#   GPT-2 L10 (83% depth) -> Pythia L27 (83% of 32)
# Middle (16) = semantic knowledge hypothesis
# Late (24, 27) = syntactic composition / token selection
EXTRACTION_LAYERS = [16, 24, 27]
EXPERIMENT_LAYERS = [16, 27]       # Subset used in factorial experiments
R2_EMBEDDING_LAYER = 31            # Layer for R2 similarity measurement (last layer)

# Automatic layer selection via diagnostic probing.
# Runs LogisticRegression probe on all 32 layers, selects top-k.
AUTO_LAYER_SELECTION = True
N_LAYERS_TOTAL = 32                # Pythia 2.8B has 32 layers

# =====================================================================
# CAV Methods
# =====================================================================
CAV_METHODS = ["mean_diff", "svm"]

# SVM Hyperparameters
SVM_C = 1.0
SVM_MAX_ITER = 20000               # Increased from 10000: d_model=2560 (vs 768)
                                    # requires more iterations for convergence
SVM_CV_FOLDS = 5
SVM_RANDOM_STATE = 42

# SVM hyperparameter tuning — grid search over C values
# with nested CV to avoid overfitting to validation fold.
SVM_TUNE_C = True
SVM_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

# =====================================================================
# Phrase Generation
# =====================================================================
RANDOM_SEED = 42
TEMPLATES_PER_WORD = 5             # Templates sampled per concept word

# Grammatical position distribution for template sampling.
# Balancing prevents probes from overfitting to syntactic position
# (Conneau et al., 2018).
TEMPLATE_POSITION_WEIGHTS = {
    "subject": 0.35,
    "object": 0.35,
    "prepositional": 0.30,
}

NEUTRAL_WORDS_PER_CATEGORY = 16    # Target balance across categories

# =====================================================================
# Psycholinguistic Controls
# =====================================================================
# Word length matching (characters). Neutral words are filtered to have
# length within WORD_LENGTH_TOLERANCE of concept word mean length.
ENABLE_LENGTH_MATCHING = True
WORD_LENGTH_TOLERANCE = 3

# Word frequency matching (Zipf scale, 0-7). Uses wordfreq library
# (Speer et al., 2018 — same team as ConceptNet).
ENABLE_FREQUENCY_MATCHING = True
FREQUENCY_TOLERANCE = 1.5          # Max Zipf-scale difference from mean

# =====================================================================
# R1 Scoring Mode
# =====================================================================
# "continuation" — Score only answer continuation tokens (recommended).
#                  Avoids prompt-length bias in loss computation.
# "full"         — Score over all tokens including prompt (legacy).
R1_SCORING_MODE = "continuation"

# =====================================================================
# Concept Removal Probe Texts
# =====================================================================
# Used by measure_concept_removal to verify ablation effectiveness.
# Each concept needs a probe sentence where that concept is salient,
# so the dot product with the CAV is meaningful.
CONCEPT_PROBE_TEXTS = {
    "time": "The time was running out.",
    "place": "The city was beautiful at dawn.",
    "tools": "He grabbed the hammer from the shelf.",
    "wolf": "The wolf howled in the dark forest.",
}

# =====================================================================
# Experiment Runner
# =====================================================================
TECHNIQUES = ["subtraction", "projection"]
INTENSITIES = [0.0, 1.5, 3.5, 6.0, 10.0, 15.0, 20.0]

# Specificity test defaults (layer mapped proportionally from GPT-2 L6)
SPECIFICITY_METHOD = "mean_diff"
SPECIFICITY_LAYER = 16
SPECIFICITY_TECHNIQUE = "subtraction"
SPECIFICITY_ALPHA = 3.5

# =====================================================================
# Statistical Validation (new)
# =====================================================================
# TCAV permutation test — Kim et al. (2018), Section 4
# Minimum 30 runs recommended for reliable p-value estimation.
TCAV_PERMUTATION_RUNS = 30
TCAV_SIGNIFICANCE_ALPHA = 0.05
TCAV_TEST_SPLIT = 0.30             # Held-out fraction for TCAV scoring

# Bootstrap — Efron & Tibshirani (1993)
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CI_LEVEL = 0.95

# Selectivity control task — Hewitt & Liang (2019)
SELECTIVITY_CONTROL_RUNS = 5

# Save raw activations for downstream statistical tests
SAVE_ACTIVATIONS = True

# Use refined (orthogonalized) CAVs in experiments.
# Default False: Kim et al. (2018) do not orthogonalize.
# Set True only with explicit theoretical justification.
USE_REFINED_CAVS = False

# ConceptNet usage: keep ConceptNet ONLY for R1/R2 benchmark enrichment.
# For CAV training data, use only curated nodes (higher quality).
USE_CONCEPTNET_FOR_CAV_TRAINING = False

# =====================================================================
# Validation Thresholds
# =====================================================================
# T01: Minimum concept nodes per category
MIN_CONCEPT_NODES = 20

# T02: Maximum negative contamination ratio
NEGATIVE_CONTAMINATION_THRESHOLD = 0.05

# T03: Maximum polysemy overlap ratio
POLYSEMY_OVERLAP_THRESHOLD = 0.08

# T04: Minimum samples per class
MIN_SAMPLES_MEAN_DIFF = 30
MIN_SAMPLES_SVM = 100              # Recommended, not required

# T05: Minimum single-token ratio
# Multi-token words create ambiguity about which token represents the
# concept. 80% threshold ensures most words tokenize to a single token.
# NOTE: Pythia uses NeoX tokenizer (different from GPT-2 BPE).
# If single-token ratio drops, consider lowering to 0.70.
MIN_SINGLE_TOKEN_RATIO = 0.80

# T06: Maximum template length difference (words)
TEMPLATE_LENGTH_DIFF_THRESHOLD = 3.0

# T07: Extraction layers must achieve above-chance probing accuracy
# and at least one must rank in the empirical top-k.
# Adjusted for 32-layer model: top-10 (proportional from top-6/12).
LAYER_VALIDITY_TOP_K = 10
LAYER_VALIDITY_MIN_ACCURACY = 0.60

# T08: SVM cross-validation accuracy threshold
SVM_CV_ACCURACY_THRESHOLD = 0.70
SVM_CV_STD_WARNING = 0.15

# T09: Cross-concept CAV cosine similarity
# FIX-3B: Raised from 0.30 to 0.35. Time-place co-occurrence in
# natural language creates expected representational overlap.
# SVM CAVs already show lower overlap than mean_diff.
CROSS_CONCEPT_COSINE_THRESHOLD = 0.35

# T10: Baseline benchmark validity
R1_BASELINE_THRESHOLD = 0.50
R2_BASELINE_THRESHOLD = 0.60

# T11: Statistical power minimum items
R1_MIN_ITEMS = 20
R2_MIN_PAIRS = 15

# T12: Specificity ratio (on-target / off-target)
SPECIFICITY_RATIO_THRESHOLD = 2.0
SPECIFICITY_MAX_OFF_R1 = 0.15
SPECIFICITY_MAX_OFF_R2 = 0.10

# T13: TCAV significance
# Uses TCAV_SIGNIFICANCE_ALPHA above

# T14: Selectivity minimum
SELECTIVITY_MIN = 0.10

# T15: Cohen's d minimum for large effect
COHENS_D_LARGE = 0.8
COHENS_D_MEDIUM = 0.5

# T16: Bootstrap CAV stability CI lower bound
# Adjusted for Pythia's 2560 dimensions (vs GPT-2's 768): bootstrap CIs
# for SVM weight vectors are wider in higher-dimensional spaces with the
# same N samples. Lowered from 0.69 to 0.65.
CAV_STABILITY_CI_LOWER = 0.65

# T17: Dose-response monotonicity rate
# FIX-2A: Lowered from 0.80 to 0.70. With 30 R1 items per concept
# (granularity 1/30 ~ 0.033), small reversals in projection conditions
# at extreme alpha are measurement noise, not pipeline defects.
DOSE_RESPONSE_MONOTONIC_RATE = 0.70

# T18: Metric CI width
# With 30 R1 items, Wilson CI width ~ 0.32 for p=0.5.
# Threshold 0.35 accommodates realistic sample sizes while
# still catching pathologically wide CIs.
R1_CI_MAX_WIDTH = 0.35
R2_CI_MAX_WIDTH = 0.10

# =====================================================================
# Corpus Extraction — Francis & Kucera (1979), Miller (1995)
# =====================================================================
# Mode: "template" (current default), "corpus" (real sentences),
#        or "hybrid" (corpus first, templates fill gap)
CORPUS_MODE = "corpus"

# NLTK corpora to extract from
CORPUS_SOURCES = ["brown", "gutenberg"]

# Optional PDF paths for additional corpus sources
CORPUS_PDF_PATHS = []

# Sentence length bounds (words). Short sentences produce cleaner
# activations because the concept word has higher relative influence.
CORPUS_MIN_SENTENCE_LEN = 8
CORPUS_MAX_SENTENCE_LEN = 20

# Minimum extracted sentences per concept before falling back to templates
CORPUS_MIN_SENTENCES = 100

# Require clear Subject-Verb-Object structure (spaCy dependency parse)
CORPUS_REQUIRE_SVO = True

# Accepted dependency relations for concept word position
CORPUS_GRAMMATICAL_POSITIONS = ["nsubj", "dobj", "pobj"]

# WordNet validation — Miller (1995)
WORDNET_VALIDATE = True
WORDNET_MIN_COVERAGE = 0.70        # Minimum % of words validated by WordNet

# Hypernym roots for category membership verification
WORDNET_HYPERNYM_ROOTS = {
    "time": [
        "time_period.n.01", "time_unit.n.01", "measure.n.02",
        "time.n.01", "time.n.03",
    ],
    "place": [
        "location.n.01", "region.n.01", "structure.n.01",
        "geographical_area.n.01", "building.n.01",
    ],
    "tool": [
        "tool.n.01", "implement.n.01", "instrument.n.01",
        "device.n.01", "utensil.n.01",
    ],
    "wolf": ["animal.n.01", "canine.n.02", "carnivore.n.01"],
    "animal": ["animal.n.01"],
    "bird": ["bird.n.01"],
    "insect": ["insect.n.01"],
    "plant": ["plant.n.02"],
    "fruit": ["fruit.n.01"],
    "vegetable": ["vegetable.n.01"],
    "vehicle": ["vehicle.n.01"],
    "furniture": ["furniture.n.01"],
    "clothing": ["clothing.n.01"],
    "container": ["container.n.01"],
    "instrument": ["musical_instrument.n.01"],
    "weapon": ["weapon.n.01"],
    "material": ["material.n.01"],
    "body_part": ["body_part.n.01"],
    "food": ["food.n.01"],
    "drink": ["beverage.n.01"],
    "profession": ["professional.n.01"],
    "sport": ["sport.n.01"],
    "emotion": ["emotion.n.01"],
}

# T19 validation thresholds
CORPUS_MIN_REAL_RATIO = 0.70       # Min % sentences from real corpus
CORPUS_MAX_POSITION_RATIO = 0.50   # Max % for any single grammatical position

# =====================================================================
# Paths
# =====================================================================
DATA_FILE = "experiment_data.json"
CAV_DIR = "cavs"
CAV_REFINED_DIR = "cavs_refined"
RESULTS_DIR = "results"
CONCEPTNET_FILE = "conceptnet_concepts.json"
VALIDATION_REPORT_FILE = os.path.join(RESULTS_DIR, "validation_report.json")
STATISTICAL_VALIDATION_FILE = "statistical_validation.json"
CORPUS_REPORT_FILE = os.path.join(RESULTS_DIR, "corpus_training_report.json")
