"""
Experiment Configuration — LLaVA-1.5 7B Multimodal Aphasia
==========================================================
Centralized configuration for the multimodal semantic aphasia experiment.
All scripts import constants from here.

References:
    - Kim et al. (2018) "Interpretability Beyond Feature Attribution" (TCAV)
    - Conneau et al. (2018) "What you can cram into a single $&!#* vector"
    - Hewitt & Liang (2019) "Designing and Interpreting Probes with Control Tasks"
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
    - Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
QUANTIZATION = "4bit"           # bitsandbytes NF4
MODEL_DTYPE = "float16"         # compute dtype (activations are fp16)

# ---------------------------------------------------------------------------
# Architecture — Vicuna-7B language model (LLaMA-based)
# ---------------------------------------------------------------------------
N_LAYERS_TOTAL = 32
D_MODEL = 4096
N_HEADS = 32
VOCAB_SIZE = 32000

# Vision encoder (CLIP ViT-L/14-336px) — frozen, not hooked
VISION_HIDDEN_SIZE = 1024
VISION_LAYERS = 24
VISION_PATCH_SIZE = 14
VISION_IMAGE_SIZE = 336
# 336/14 = 24 patches per side → 24*24 = 576 image tokens
N_IMAGE_TOKENS = 576

# ---------------------------------------------------------------------------
# pyvene hook configuration
# ---------------------------------------------------------------------------
# Component name for pyvene — maps to model decoder layers[L]
HOOK_COMPONENT = "block_output"

# Token position strategy for activation pooling
# "text_only_mean" — mean pool over text tokens only (exclude 576 image tokens)
# "all_mean"       — mean pool over all tokens (image + text)
# "image_only_mean"— mean pool over image tokens only
# "last_text_token"— last text token (generation position)
TOKEN_POSITION = "all_mean"        # Image + text tokens: captures multimodal
                                    # integration (was image_only_mean — R2a < 0.60)

# All strategies to compare during layer diagnostic
TOKEN_POSITION_COMPARE = [
    "text_only_mean", "image_only_mean", "all_mean", "last_text_token",
]

# ---------------------------------------------------------------------------
# Image-Only Ablation Mode
# ---------------------------------------------------------------------------
# When True, ablation hooks modify ONLY image token positions.
# This simulates "pure visual agnosia" — the model processes text normally
# but loses the visual concept representation in the 576 image tokens.
#
# When False (default), ablation modifies ALL token positions (image + text).
# This is a stronger intervention that also removes concept information
# that leaked into text tokens via self-attention.
IMAGE_ONLY_ABLATION = False

# ---------------------------------------------------------------------------
# Layer Selection (proportional to model depth)
# ---------------------------------------------------------------------------
# 50% = L16, 75% = L24, 84% = L27 (same ratios as Pythia 2.8B)
EXTRACTION_LAYERS = [16, 24, 27]
EXPERIMENT_LAYERS = [16, 27]
SPECIFICITY_LAYER = 16
SPECIFICITY_METHOD = "svm"
SPECIFICITY_TECHNIQUE = "subtraction"
SPECIFICITY_ALPHA = 3.5

# Automatic layer selection via diagnostic probing.
AUTO_LAYER_SELECTION = True
LAYER_VALIDITY_MIN_ACCURACY = 0.60

# ---------------------------------------------------------------------------
# R2 Embedding Configuration
# ---------------------------------------------------------------------------
R2_EMBEDDING_LAYER = 31          # last layer for embeddings

# R2a: Cross-modal similarity — image token pooling strategy
# Forces image_only_mean regardless of TOKEN_POSITION, because R2a
# specifically measures visual representation vs textual concept.
R2A_IMAGE_POOLING = "image_only_mean"

# R2b: Output similarity — layer for last-token embedding
R2B_EMBEDDING_LAYER = 31

# ---------------------------------------------------------------------------
# Concepts (overridden by PILOT_MODE below)
# ---------------------------------------------------------------------------
CONCEPTS = ["time", "place", "tools"]

# ---------------------------------------------------------------------------
# Image Dataset Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DATA_FILE = os.path.join(DATA_DIR, "experiment_data.json")

MIN_IMAGES_PER_CONCEPT = 50
MAX_IMAGES_PER_CONCEPT = 200

# ---------------------------------------------------------------------------
# Train / Test Split — prevents data leakage between CAV training and eval
# ---------------------------------------------------------------------------
# 70% of images go to CAV training, 30% to R1/R2 benchmarks.
# Images in the test set are NEVER seen during CAV extraction.
TRAIN_SPLIT_RATIO = 0.70

# ---------------------------------------------------------------------------
# Image Sources — Priority Order
# ---------------------------------------------------------------------------
# Source 1: ImageNet-1K subsets via HuggingFace datasets
# These are single-object, centered images — ideal for concept activation.
# NOTE: HuggingFace uses ENGLISH class names (e.g. "analog clock"),
# NOT WordNet synset IDs (e.g. "n02708093"). Names are matched via
# case-insensitive substring search against dataset.features["label"].names.
IMAGENET_CLASS_NAMES = {
    "time": [
        "analog clock",     # WNID n02708093, idx 409
        "digital clock",    # WNID n03196217, idx 530
        "wall clock",       # WNID n04548280, idx 892
        "digital watch",    # WNID n03196217, idx 531
    ],
    "place": [
        "altar",            # WNID n02699494, idx 406
        "barn",             # WNID n02793495, idx 425
        "church",           # WNID n03028079, idx 497
        "planetarium",      # WNID n03956157, idx 727
        "palace",           # WNID n03877845, idx 698
        "monastery",        # WNID n03781244, idx 663
    ],
    "tools": [
        "hammer",           # WNID n03481172, idx 587
        "screwdriver",      # WNID n04154565, idx 784
        "chain saw",        # WNID n03000684, idx 491
        "lawn mower",       # WNID n03649909, idx 621
        "power drill",      # WNID n03995372, idx 740
        "hatchet",          # WNID n03494278, idx 596
    ],
}

# Source 2: Open Images V7 labels for supplementary download (with bbox crop)
OPEN_IMAGES_LABELS = {
    "time": ["Clock", "Watch", "Alarm clock"],
    "place": ["Building", "Skyscraper", "Tower", "House", "Castle",
              "Church", "Bridge", "Lighthouse"],
    "tools": ["Hammer", "Screwdriver", "Wrench", "Drill (Tool)",
              "Pliers", "Saw", "Axe", "Shovel"],
}

# Neutral images — ImageNet classes unrelated to any experimental concept.
# Used as negative examples for CAV extraction (concept vs neutral).
NEUTRAL_IMAGENET_CLASS_NAMES = [
    "coffee mug",      # idx 504
    "laptop",          # idx 620
    "umbrella",        # idx 879
    "banana",          # idx 954
    "sports car",      # idx 817
    "school bus",      # idx 779
    "tennis ball",     # idx 852
    "pizza",           # idx 963
]

# ---------------------------------------------------------------------------
# VQA Prompt Templates
# ---------------------------------------------------------------------------
# Generic prompt for CAV training (same for positive AND negative)
CAV_TRAINING_PROMPT = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"

# R1: Visual MCQ
R1_VQA_PROMPT = "USER: <image>\nWhat type of object or concept is primarily shown in this image? Choose one: {options}\nASSISTANT:"

# R2a: Image description for cross-modal similarity
R2_IMAGE_PROMPT = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"
# R2a: Text-only prompt (NO image — measures pure text concept embedding)
R2_TEXT_ONLY_PROMPT = "USER: {description}\nASSISTANT:"

# R2b: Prompt for output comparison (baseline vs intervened)
R2B_OUTPUT_PROMPT = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"

# ---------------------------------------------------------------------------
# R1 Distractors — INTRA-DOMAIN for harder discrimination
# ---------------------------------------------------------------------------
# Each concept's distractors are semantically CLOSE categories,
# forcing the model to discriminate between related concepts
# instead of trivially choosing between "time" and "mineral".
R1_DISTRACTORS_PER_CONCEPT = {
    "time": ["place", "tool", "furniture", "appliance"],
    "place": ["time", "tool", "vehicle", "landscape"],
    "tools": ["time", "place", "utensil", "appliance"],
}

# Correct R1 labels per concept
R1_CORRECT_LABELS = {
    "time": "time",
    "place": "place",
    "tools": "tool",
}

# ---------------------------------------------------------------------------
# CAV Extraction
# ---------------------------------------------------------------------------
CAV_DIR = "cavs"
CAV_METHODS = ["mean_diff", "svm"]
SVM_C = 1.0
SVM_MAX_ITER = 30000            # Higher for d_model=4096
SVM_TUNE_C = True
SVM_C_VALUES = [0.01, 0.1, 1.0, 10.0]
SVM_CV_FOLDS = 5
RANDOM_SEED = 42
SVM_RANDOM_STATE = 42

# Save raw activations for downstream statistical tests
# Enables cav_statistical_tests.py to run without reloading the model.
SAVE_ACTIVATIONS = True

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
TECHNIQUES = ["subtraction", "projection"]
INTENSITIES = [0.0, 1.5, 3.5, 6.0]

# ---------------------------------------------------------------------------
# Statistical Validation — Kim et al. (2018), Hewitt & Liang (2019)
# ---------------------------------------------------------------------------
# TCAV permutation test — Kim et al. (2018), Section 4
TCAV_PERMUTATION_RUNS = 30
TCAV_SIGNIFICANCE_ALPHA = 0.05
TCAV_TEST_SPLIT = 0.30             # Held-out fraction for TCAV scoring

# Bootstrap — Efron & Tibshirani (1993)
# Lower than GPT-2 (1000) to fit compute budget with 4096-dim activations
BOOTSTRAP_N_RESAMPLES = 500
BOOTSTRAP_CI_LEVEL = 0.95

# Selectivity control task — Hewitt & Liang (2019)
SELECTIVITY_CONTROL_RUNS = 5

# ---------------------------------------------------------------------------
# Validation Thresholds
# ---------------------------------------------------------------------------
# T01-T09: Data integrity
MIN_CONCEPT_NODES = 20

# T10-T14: CAV quality
CAV_OVERLAP_THRESHOLD = 0.35
CAV_STABILITY_CI_LOWER = 0.65
SVM_CV_ACCURACY_THRESHOLD = 0.70
SVM_CV_STD_WARNING = 0.15

# T15-T18: Model & baseline
R1_BASELINE_THRESHOLD = 0.50
R2A_BASELINE_THRESHOLD = 0.60     # R2a: cross-modal similarity baseline
R2B_BASELINE_THRESHOLD = 0.90     # R2b: output similarity (should be ~1.0 at baseline)
SPECIFICITY_RATIO_THRESHOLD = 2.0
SPECIFICITY_MAX_OFF_R1 = 0.15

# T19-T22: Experiment results
LAYER_VALIDITY_TOP_K = 10

# T23-T28: Statistical rigor (NEW — ported from GPT-2)
SELECTIVITY_MIN = 0.10
COHENS_D_LARGE = 0.8
COHENS_D_MEDIUM = 0.5
DOSE_RESPONSE_MONOTONIC_RATE = 0.70
R1_CI_MAX_WIDTH = 0.35
R2A_CI_MAX_WIDTH = 0.10           # R2a confidence interval
R2B_CI_MAX_WIDTH = 0.10           # R2b confidence interval
R1_MIN_ITEMS = 15
R2_MIN_PAIRS = 10

# Multiple comparisons correction
# With 3 concepts x 3 modes x 2 alphas x 2 centering = 36 conditions,
# correction is mandatory to control family-wise error rate.
MULTIPLE_COMPARISON_METHOD = "fdr_bh"   # Benjamini-Hochberg FDR

# ---------------------------------------------------------------------------
# Pilot Mode — Visually Distinct Concepts
# ---------------------------------------------------------------------------
# When True, replaces broad concepts with visually distinct categories.
#
# V2: Changed from golden_retriever/labrador/tabby to dog/car/flower.
# Reason: golden↔labrador CAV overlap was 0.865 (max threshold 0.35),
# baselines were invalid (golden=6.7%, tabby=0%, labrador=100%).
# Dog/car/flower are maximally distinct in visual space.
PILOT_MODE = True

PILOT_CONCEPTS = ["dog", "car", "flower"]

PILOT_IMAGENET_CLASS_NAMES = {
    "dog": ["golden retriever", "Labrador retriever",
            "German shepherd dog"],
    "car": ["sports car", "minivan", "convertible"],
    "flower": ["daisy", "sunflower", "pot, flowerpot"],
}

PILOT_NEUTRAL_IMAGENET_CLASS_NAMES = [
    "coffee mug",      # idx 504
    "laptop",          # idx 620
    "umbrella",        # idx 879
    "banana",          # idx 954
]

PILOT_R1_DISTRACTORS_PER_CONCEPT = {
    "dog": ["car", "flower", "furniture", "food"],
    "car": ["dog", "flower", "tool", "building"],
    "flower": ["dog", "car", "food", "furniture"],
}

PILOT_R1_CORRECT_LABELS = {
    "dog": "dog",
    "car": "car",
    "flower": "flower",
}

# ---------------------------------------------------------------------------
# Apply Pilot Mode overrides
# ---------------------------------------------------------------------------
if PILOT_MODE:
    CONCEPTS = PILOT_CONCEPTS
    IMAGENET_CLASS_NAMES = PILOT_IMAGENET_CLASS_NAMES
    NEUTRAL_IMAGENET_CLASS_NAMES = PILOT_NEUTRAL_IMAGENET_CLASS_NAMES
    R1_DISTRACTORS_PER_CONCEPT = PILOT_R1_DISTRACTORS_PER_CONCEPT
    R1_CORRECT_LABELS = PILOT_R1_CORRECT_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATISTICAL_VALIDATION_FILE = "statistical_validation.json"
CORPUS_REPORT_FILE = os.path.join(RESULTS_DIR, "corpus_training_report.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "validation_report.json")
