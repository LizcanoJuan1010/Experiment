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
EXPERIMENT_LAYERS = [16, 24, 27]
SPECIFICITY_LAYER = 16
SPECIFICITY_METHOD = "svm"
SPECIFICITY_TECHNIQUE = "projection"
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
CONCEPTS = [
    "camel", "bee", "horse", "penguin", "violin", "bird", "spider", "candle",
    "tabby_cat", "golden_retriever", "labrador_retriever",
]

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
    # Animals
    "camel":       ["Arabian camel", "dromedary"],
    "squirrel":    ["fox squirrel", "grey squirrel"],
    "bee":         ["bee"],
    "horse":       ["sorrel", "Arabian", "appaloosa"],
    "penguin":     ["African penguin", "king penguin"],
    "bird":        ["robin", "jay", "American crow", "indigo bunting"],
    "spider":      ["garden spider", "tarantula", "black and gold garden spider"],
    # Objects
    "lock":        ["padlock"],
    "toothbrush":  ["toothbrush"],
    "violin":      ["violin", "fiddle"],
    "snowman":     ["snowman"],
    "fire":        [],        # Not in ImageNet — use Open Images
    "candle":      ["candle"],
    # Domestic animals
    "tabby_cat":         ["tabby", "tabby cat"],
    "golden_retriever":  ["golden retriever"],
    "labrador_retriever": ["Labrador retriever", "yellow Labrador retriever",
                           "chocolate Labrador retriever"],
}

# Source 2: Open Images V7 labels for supplementary download (with bbox crop)
OPEN_IMAGES_LABELS = {
    # Animals
    "camel":       ["Camel", "Bactrian camel"],
    "squirrel":    ["Squirrel", "Red squirrel"],
    "bee":         ["Bee"],
    "horse":       ["Horse"],
    "penguin":     ["Penguin"],
    "bird":        ["Bird"],
    "spider":      ["Spider"],
    # Objects
    "lock":        ["Padlock", "Lock"],
    "toothbrush":  ["Toothbrush", "Brush"],
    "violin":      ["Violin", "Musical instrument", "String instrument"],
    "snowman":     ["Snowman", "Snow"],
    "fire":        ["Fire"],
    "candle":      ["Candle"],
    # Domestic animals
    "tabby_cat":         ["Cat", "Kitten"],
    "golden_retriever":  ["Dog", "Golden retriever"],
    "labrador_retriever": ["Dog", "Labrador retriever"],
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

# SFA: Semantic Feature Analysis — binary yes/no probe
# Answer appended as " yes" or " no" (with leading space), matching R1 MCQ pattern.
SFA_PROMPT_TEMPLATE = "USER: <image>\n{question} Answer only yes or no.\nASSISTANT:"

# SFA feature bank — 10 items per concept (2 per dimension: 1 "yes" + 1 "no" foil)
# Dimensions: category, function, perceptual, structural, associative
SFA_FEATURE_BANK = {
    # ── Animals ────────────────────────────────────────────────────────────────
    "camel": [
        ("Is this an animal?",                              "yes", "category"),
        ("Is this a vehicle?",                              "no",  "category"),
        ("Is this used for transportation in deserts?",     "yes", "function"),
        ("Is this used for swimming?",                      "no",  "function"),
        ("Does this have a hump on its back?",              "yes", "perceptual"),
        ("Does this have wings?",                           "no",  "perceptual"),
        ("Does this have four legs?",                       "yes", "structural"),
        ("Does this have fins?",                            "no",  "structural"),
        ("Is this typically found in deserts?",             "yes", "associative"),
        ("Is this typically found in oceans?",              "no",  "associative"),
    ],
    "squirrel": [
        ("Is this an animal?",                              "yes", "category"),
        ("Is this a plant?",                                "no",  "category"),
        ("Does this collect and store nuts or seeds?",      "yes", "function"),
        ("Does this produce milk for humans?",              "no",  "function"),
        ("Does this have a bushy tail?",                    "yes", "perceptual"),
        ("Does this have scales?",                          "no",  "perceptual"),
        ("Does this have four legs?",                       "yes", "structural"),
        ("Does this have a shell?",                         "no",  "structural"),
        ("Is this typically found in trees or parks?",      "yes", "associative"),
        ("Is this typically found in oceans?",              "no",  "associative"),
    ],
    "bee": [
        ("Is this an insect?",                              "yes", "category"),
        ("Is this a mammal?",                               "no",  "category"),
        ("Does this produce honey?",                        "yes", "function"),
        ("Does this produce milk?",                         "no",  "function"),
        ("Does this have yellow and black stripes?",        "yes", "perceptual"),
        ("Does this have fur?",                             "no",  "perceptual"),
        ("Does this have wings?",                           "yes", "structural"),
        ("Does this have four legs?",                       "no",  "structural"),
        ("Is this typically found near flowers?",           "yes", "associative"),
        ("Is this typically found underwater?",             "no",  "associative"),
    ],
    "horse": [
        ("Is this an animal?",                              "yes", "category"),
        ("Is this a vehicle?",                              "no",  "category"),
        ("Is this used for riding?",                        "yes", "function"),
        ("Is this used for flight?",                        "no",  "function"),
        ("Does this have a mane?",                          "yes", "perceptual"),
        ("Does this have scales?",                          "no",  "perceptual"),
        ("Does this have four legs?",                       "yes", "structural"),
        ("Does this have wings?",                           "no",  "structural"),
        ("Is this typically found on farms or stables?",    "yes", "associative"),
        ("Is this typically found in oceans?",              "no",  "associative"),
    ],
    "penguin": [
        ("Is this a bird?",                                 "yes", "category"),
        ("Is this a fish?",                                 "no",  "category"),
        ("Is this able to swim?",                           "yes", "function"),
        ("Is this able to fly?",                            "no",  "function"),
        ("Does this have black and white feathers?",        "yes", "perceptual"),
        ("Does this have scales?",                          "no",  "perceptual"),
        ("Does this have flippers?",                        "yes", "structural"),
        ("Does this have four legs?",                       "no",  "structural"),
        ("Is this typically found in cold climates?",       "yes", "associative"),
        ("Is this typically found in deserts?",             "no",  "associative"),
    ],
    "bird": [
        ("Is this an animal?",                              "yes", "category"),
        ("Is this an insect?",                              "no",  "category"),
        ("Can this typically fly?",                         "yes", "function"),
        ("Can this swim underwater?",                       "no",  "function"),
        ("Does this have feathers?",                        "yes", "perceptual"),
        ("Does this have fur?",                             "no",  "perceptual"),
        ("Does this have a beak?",                          "yes", "structural"),
        ("Does this have a mane?",                          "no",  "structural"),
        ("Is this typically found in trees or the sky?",    "yes", "associative"),
        ("Is this typically found underground?",            "no",  "associative"),
    ],
    "spider": [
        ("Is this an arachnid?",                            "yes", "category"),
        ("Is this an insect?",                              "no",  "category"),
        ("Does this catch prey using webs?",                "yes", "function"),
        ("Does this produce honey?",                        "no",  "function"),
        ("Is this typically small and dark-colored?",       "yes", "perceptual"),
        ("Does this have feathers?",                        "no",  "perceptual"),
        ("Does this have eight legs?",                      "yes", "structural"),
        ("Does this have a backbone?",                      "no",  "structural"),
        ("Is this typically found in corners or webs?",     "yes", "associative"),
        ("Is this typically found in open water?",          "no",  "associative"),
    ],
    # ── Objects ────────────────────────────────────────────────────────────────
    "lock": [
        ("Is this a device or tool?",                       "yes", "category"),
        ("Is this an animal?",                              "no",  "category"),
        ("Is this used for security or preventing access?", "yes", "function"),
        ("Is this used for cooking?",                       "no",  "function"),
        ("Is this typically metallic?",                     "yes", "perceptual"),
        ("Is this typically soft and flexible?",            "no",  "perceptual"),
        ("Does this have a keyhole?",                       "yes", "structural"),
        ("Does this have legs?",                            "no",  "structural"),
        ("Is this typically found on doors or safes?",      "yes", "associative"),
        ("Is this typically found in kitchens?",            "no",  "associative"),
    ],
    "toothbrush": [
        ("Is this a hygiene tool?",                         "yes", "category"),
        ("Is this a food item?",                            "no",  "category"),
        ("Is this used for cleaning teeth?",                "yes", "function"),
        ("Is this used for cooking?",                       "no",  "function"),
        ("Does this have bristles?",                        "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Does this have a handle?",                        "yes", "structural"),
        ("Does this have wheels?",                          "no",  "structural"),
        ("Is this typically found in bathrooms?",           "yes", "associative"),
        ("Is this typically found in garages?",             "no",  "associative"),
    ],
    "violin": [
        ("Is this a musical instrument?",                   "yes", "category"),
        ("Is this a tool?",                                 "no",  "category"),
        ("Is this used to make music?",                     "yes", "function"),
        ("Is this used for cooking?",                       "no",  "function"),
        ("Is this typically made of wood?",                 "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Does this have strings?",                         "yes", "structural"),
        ("Does this have keys?",                            "no",  "structural"),
        ("Is this typically found in orchestras?",          "yes", "associative"),
        ("Is this typically found in kitchens?",            "no",  "associative"),
    ],
    "snowman": [
        ("Is this a figure made from snow?",                "yes", "category"),
        ("Is this a living creature?",                      "no",  "category"),
        ("Is this primarily decorative?",                   "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Is this typically white?",                        "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Is this typically spherical in shape?",           "yes", "structural"),
        ("Does this have wheels?",                          "no",  "structural"),
        ("Is this typically found in winter?",              "yes", "associative"),
        ("Is this typically found in summer?",              "no",  "associative"),
    ],
    "fire": [
        ("Is this a form of combustion?",                   "yes", "category"),
        ("Is this a solid object?",                         "no",  "category"),
        ("Is this used for warmth or cooking?",             "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Is this typically orange or red?",                "yes", "perceptual"),
        ("Is this typically blue or green?",                "no",  "perceptual"),
        ("Does this produce heat and light?",               "yes", "structural"),
        ("Does this have a solid shape?",                   "no",  "structural"),
        ("Is this typically associated with fireplaces?",   "yes", "associative"),
        ("Is this typically associated with water?",        "no",  "associative"),
    ],
    "candle": [
        ("Is this a light source?",                         "yes", "category"),
        ("Is this a food item?",                            "no",  "category"),
        ("Is this used to provide light?",                  "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Is this typically made of wax?",                  "yes", "perceptual"),
        ("Is this typically made of metal?",                "no",  "perceptual"),
        ("Does this have a wick?",                          "yes", "structural"),
        ("Does this have wheels?",                          "no",  "structural"),
        ("Is this typically found at dinner tables?",       "yes", "associative"),
        ("Is this typically found in garages?",             "no",  "associative"),
    ],
    # ── Professions ────────────────────────────────────────────────────────────
    "fisherman": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Is this person catching fish?",                      "yes", "function"),
        ("Is this person treating patients?",                  "no",  "function"),
        ("Is this person typically near water?",               "yes", "perceptual"),
        ("Is this person typically in a kitchen?",             "no",  "perceptual"),
        ("Does this person use a fishing rod or net?",         "yes", "structural"),
        ("Does this person use a scalpel?",                    "no",  "structural"),
        ("Is this profession associated with rivers or sea?",  "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "painter": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an object?",                                 "no",  "category"),
        ("Does this person create visual art?",                "yes", "function"),
        ("Does this person repair machines?",                  "no",  "function"),
        ("Is this person typically holding a brush?",          "yes", "perceptual"),
        ("Is this person typically holding scissors?",         "no",  "perceptual"),
        ("Does this person work with paints and canvases?",    "yes", "structural"),
        ("Does this person work with food?",                   "no",  "structural"),
        ("Is this profession associated with art studios?",    "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "baker": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person make bread and pastries?",          "yes", "function"),
        ("Does this person repair machines?",                  "no",  "function"),
        ("Is this person typically in a bakery?",              "yes", "perceptual"),
        ("Is this person typically outdoors?",                 "no",  "perceptual"),
        ("Does this person work with ovens and dough?",        "yes", "structural"),
        ("Does this person work with animals?",                "no",  "structural"),
        ("Is this profession associated with bakeries?",       "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "doctor": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this a tool?",                                    "no",  "category"),
        ("Does this person treat patients?",                   "yes", "function"),
        ("Does this person bake bread?",                       "no",  "function"),
        ("Is this person typically wearing a white coat?",     "yes", "perceptual"),
        ("Is this person typically wearing armor?",            "no",  "perceptual"),
        ("Does this person use medical instruments?",          "yes", "structural"),
        ("Does this person use cooking tools?",                "no",  "structural"),
        ("Is this profession associated with hospitals?",      "yes", "associative"),
        ("Is this profession associated with bakeries?",       "no",  "associative"),
    ],
    "barber": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person cut hair?",                         "yes", "function"),
        ("Does this person treat illnesses?",                  "no",  "function"),
        ("Is this person typically holding scissors?",         "yes", "perceptual"),
        ("Is this person typically holding a paintbrush?",     "no",  "perceptual"),
        ("Does this person work with scissors and combs?",     "yes", "structural"),
        ("Does this person use medical instruments?",          "no",  "structural"),
        ("Is this profession associated with barbershops?",    "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "photographer": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person take photographs?",                 "yes", "function"),
        ("Does this person bake food?",                        "no",  "function"),
        ("Is this person typically holding a camera?",         "yes", "perceptual"),
        ("Is this person typically holding scissors?",         "no",  "perceptual"),
        ("Does this person use cameras and lenses?",           "yes", "structural"),
        ("Does this person use cooking equipment?",            "no",  "structural"),
        ("Is this profession associated with studios?",        "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "tailor": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person make or alter clothing?",           "yes", "function"),
        ("Does this person cut hair?",                         "no",  "function"),
        ("Is this person typically working with fabric?",      "yes", "perceptual"),
        ("Is this person typically in a kitchen?",             "no",  "perceptual"),
        ("Does this person use needles and thread?",           "yes", "structural"),
        ("Does this person use medical instruments?",          "no",  "structural"),
        ("Is this profession associated with clothing?",       "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "gardener": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this a plant?",                                   "no",  "category"),
        ("Does this person tend gardens and plants?",          "yes", "function"),
        ("Does this person treat patients?",                   "no",  "function"),
        ("Is this person typically outdoors?",                 "yes", "perceptual"),
        ("Is this person typically in a kitchen?",             "no",  "perceptual"),
        ("Does this person use gardening tools?",              "yes", "structural"),
        ("Does this person use medical instruments?",          "no",  "structural"),
        ("Is this profession associated with gardens?",        "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "king": [
        ("Is this a royal figure?",                            "yes", "category"),
        ("Is this a common worker?",                           "no",  "category"),
        ("Does this person rule or govern?",                   "yes", "function"),
        ("Does this person serve food?",                       "no",  "function"),
        ("Is this person typically wearing a crown?",          "yes", "perceptual"),
        ("Is this person typically wearing work clothes?",     "no",  "perceptual"),
        ("Does this person sit on a throne?",                  "yes", "structural"),
        ("Does this person stand at a workbench?",             "no",  "structural"),
        ("Is this role associated with palaces?",              "yes", "associative"),
        ("Is this role associated with hospitals?",            "no",  "associative"),
    ],
    "pirate": [
        ("Is this a sea adventurer or outlaw?",                "yes", "category"),
        ("Is this a medical professional?",                    "no",  "category"),
        ("Does this person sail and plunder?",                 "yes", "function"),
        ("Does this person treat patients?",                   "no",  "function"),
        ("Does this person typically wear an eye patch?",      "yes", "perceptual"),
        ("Does this person typically wear a white coat?",      "no",  "perceptual"),
        ("Does this person use a ship and weapons?",           "yes", "structural"),
        ("Does this person use medical tools?",                "no",  "structural"),
        ("Is this figure associated with ships and the sea?",  "yes", "associative"),
        ("Is this figure associated with hospitals?",          "no",  "associative"),
    ],
    "soldier": [
        ("Is this a military person?",                         "yes", "category"),
        ("Is this a medical professional?",                    "no",  "category"),
        ("Does this person serve in the military?",            "yes", "function"),
        ("Does this person bake food?",                        "no",  "function"),
        ("Is this person typically wearing a uniform?",        "yes", "perceptual"),
        ("Is this person typically wearing a white coat?",     "no",  "perceptual"),
        ("Does this person carry weapons?",                    "yes", "structural"),
        ("Does this person use cooking equipment?",            "no",  "structural"),
        ("Is this profession associated with military?",       "yes", "associative"),
        ("Is this profession associated with bakeries?",       "no",  "associative"),
    ],
    "chef": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person prepare and cook food?",            "yes", "function"),
        ("Does this person treat patients?",                   "no",  "function"),
        ("Is this person typically wearing a chef's hat?",     "yes", "perceptual"),
        ("Is this person typically wearing a military uniform?", "no", "perceptual"),
        ("Does this person work with cooking equipment?",      "yes", "structural"),
        ("Does this person use medical instruments?",          "no",  "structural"),
        ("Is this profession associated with restaurants?",    "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "musician": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person perform music?",                    "yes", "function"),
        ("Does this person treat patients?",                   "no",  "function"),
        ("Is this person typically holding an instrument?",    "yes", "perceptual"),
        ("Is this person typically holding a camera?",         "no",  "perceptual"),
        ("Does this person use musical instruments?",          "yes", "structural"),
        ("Does this person use cooking equipment?",            "no",  "structural"),
        ("Is this profession associated with concerts?",       "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "carpenter": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person build or work with wood?",          "yes", "function"),
        ("Does this person treat patients?",                   "no",  "function"),
        ("Is this person typically in a workshop?",            "yes", "perceptual"),
        ("Is this person typically in a kitchen?",             "no",  "perceptual"),
        ("Does this person use woodworking tools?",            "yes", "structural"),
        ("Does this person use medical instruments?",          "no",  "structural"),
        ("Is this profession associated with woodworking?",    "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "writer": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person write books or articles?",          "yes", "function"),
        ("Does this person build furniture?",                  "no",  "function"),
        ("Is this person typically at a desk?",                "yes", "perceptual"),
        ("Is this person typically in a kitchen?",             "no",  "perceptual"),
        ("Does this person use a pen or keyboard?",            "yes", "structural"),
        ("Does this person use woodworking tools?",            "no",  "structural"),
        ("Is this profession associated with writing?",        "yes", "associative"),
        ("Is this profession associated with hospitals?",      "no",  "associative"),
    ],
    "dentist": [
        ("Is this a person?",                                  "yes", "category"),
        ("Is this an animal?",                                 "no",  "category"),
        ("Does this person treat teeth and gums?",             "yes", "function"),
        ("Does this person cut hair?",                         "no",  "function"),
        ("Is this person typically working in a patient's mouth?", "yes", "perceptual"),
        ("Is this person typically painting?",                 "no",  "perceptual"),
        ("Does this person use dental instruments?",           "yes", "structural"),
        ("Does this person use woodworking tools?",            "no",  "structural"),
        ("Is this profession associated with dental offices?", "yes", "associative"),
        ("Is this profession associated with bakeries?",       "no",  "associative"),
    ],
    "tabby_cat": [
        ("Is this a cat?",                                  "yes", "category"),
        ("Is this a dog?",                                  "no",  "category"),
        ("Is this kept as a pet?",                          "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Does this have tabby stripes or patches?",        "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Does this have fur?",                             "yes", "structural"),
        ("Does this have wings?",                           "no",  "structural"),
        ("Is this typically found in homes?",               "yes", "associative"),
        ("Is this typically found in the ocean?",           "no",  "associative"),
    ],
    "golden_retriever": [
        ("Is this a dog?",                                  "yes", "category"),
        ("Is this a cat?",                                  "no",  "category"),
        ("Is this kept as a pet?",                          "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Is this typically golden or yellow in color?",    "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Does this have fur?",                             "yes", "structural"),
        ("Does this have scales?",                          "no",  "structural"),
        ("Is this typically found in homes?",               "yes", "associative"),
        ("Is this typically found in the ocean?",           "no",  "associative"),
    ],
    "labrador_retriever": [
        ("Is this a dog?",                                  "yes", "category"),
        ("Is this a cat?",                                  "no",  "category"),
        ("Is this kept as a pet?",                          "yes", "function"),
        ("Is this used for transportation?",                "no",  "function"),
        ("Is this typically black, yellow, or chocolate?",  "yes", "perceptual"),
        ("Is this typically metallic?",                     "no",  "perceptual"),
        ("Does this have fur?",                             "yes", "structural"),
        ("Does this have feathers?",                        "no",  "structural"),
        ("Is this typically found in homes?",               "yes", "associative"),
        ("Is this typically found in the ocean?",           "no",  "associative"),
    ],
}

# ---------------------------------------------------------------------------
# R1 Distractors — INTRA-DOMAIN for harder discrimination
# ---------------------------------------------------------------------------
# Each concept's distractors are semantically CLOSE categories,
# forcing the model to discriminate between related concepts
# instead of trivially choosing between "time" and "mineral".
R1_DISTRACTORS_PER_CONCEPT = {
    # Animals
    "camel":              ["horse", "bee", "penguin", "bird"],
    "bee":                ["spider", "bird", "camel", "horse"],
    "horse":              ["camel", "penguin", "bird", "bee"],
    "penguin":            ["bird", "bee", "horse", "camel"],
    "bird":               ["penguin", "bee", "spider", "camel"],
    "spider":             ["bee", "bird", "tabby_cat", "golden_retriever"],
    "tabby_cat":          ["golden_retriever", "labrador_retriever", "spider", "bird"],
    "golden_retriever":   ["labrador_retriever", "tabby_cat", "horse", "bee"],
    "labrador_retriever": ["golden_retriever", "tabby_cat", "horse", "bird"],
    # Objects
    "violin":             ["candle", "bird", "spider", "bee"],
    "candle":             ["violin", "spider", "bee", "bird"],
}

# Correct R1 labels per concept
R1_CORRECT_LABELS = {
    "camel":              "camel",
    "bee":                "bee",
    "horse":              "horse",
    "penguin":            "penguin",
    "violin":             "violin",
    "bird":               "bird",
    "spider":             "spider",
    "candle":             "candle",
    "tabby_cat":          "tabby cat",
    "golden_retriever":   "golden retriever",
    "labrador_retriever": "labrador retriever",
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
TECHNIQUES = ["projection"]
INTENSITIES = [3.0, 6.0, 10.0, 15.0, 20.0]

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
PILOT_MODE = False

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

# SFA feature bank for pilot concepts.
# Each entry: (question, expected_answer, semantic_dimension)
# Dimensions: category, function, perceptual, structural, associative
# One "yes" (true feature) + one "no" (foil) per dimension per concept.
PILOT_SFA_FEATURE_BANK = {
    "dog": [
        # category
        ("Is this an animal?",                    "yes", "category"),
        ("Is this a vehicle?",                    "no",  "category"),
        # function
        ("Is this used for companionship?",       "yes", "function"),
        ("Is this used for transportation?",      "no",  "function"),
        # perceptual
        ("Is this typically furry?",              "yes", "perceptual"),
        ("Is this typically metallic?",           "no",  "perceptual"),
        # structural
        ("Does this have four legs?",             "yes", "structural"),
        ("Does this have wheels?",                "no",  "structural"),
        # associative
        ("Is this typically found in homes?",     "yes", "associative"),
        ("Is this typically found on roads?",     "no",  "associative"),
    ],
    "car": [
        # category
        ("Is this a vehicle?",                    "yes", "category"),
        ("Is this an animal?",                    "no",  "category"),
        # function
        ("Is this used for transportation?",      "yes", "function"),
        ("Is this used for companionship?",       "no",  "function"),
        # perceptual
        ("Is this typically metallic?",           "yes", "perceptual"),
        ("Is this typically furry?",              "no",  "perceptual"),
        # structural
        ("Does this have wheels?",                "yes", "structural"),
        ("Does this have four legs?",             "no",  "structural"),
        # associative
        ("Is this typically found on roads?",     "yes", "associative"),
        ("Is this typically found in homes?",     "no",  "associative"),
    ],
    "flower": [
        # category
        ("Is this a plant?",                      "yes", "category"),
        ("Is this an animal?",                    "no",  "category"),
        # function
        ("Is this used for decoration?",          "yes", "function"),
        ("Is this used for transportation?",      "no",  "function"),
        # perceptual
        ("Is this typically colorful?",           "yes", "perceptual"),
        ("Is this typically metallic?",           "no",  "perceptual"),
        # structural
        ("Does this have petals?",                "yes", "structural"),
        ("Does this have wheels?",                "no",  "structural"),
        # associative
        ("Is this typically found in gardens?",   "yes", "associative"),
        ("Is this typically found on roads?",     "no",  "associative"),
    ],
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
    SFA_FEATURE_BANK = PILOT_SFA_FEATURE_BANK

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATISTICAL_VALIDATION_FILE = "statistical_validation.json"
CORPUS_REPORT_FILE = os.path.join(RESULTS_DIR, "corpus_training_report.json")
REPORT_FILE = os.path.join(RESULTS_DIR, "validation_report.json")
