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
from datetime import datetime

random.seed(42)

OUTPUT_FILE = "experiment_data.json"
CONCEPTNET_FILE = "conceptnet_concepts.json"

# =====================================================================
# 1. CURATED CONCEPT WORD LISTS (baseline, always available)
# =====================================================================
# NOTE: Words are assigned to ONE category only to avoid overlap.
# Ambiguous words (watch, spring, etc.) are excluded or assigned to
# the single most appropriate category.

CURATED_NODES = {
    "time": [
        "hour", "minute", "second", "day", "week", "month", "year",
        "decade", "century", "millennium", "epoch", "era", "moment",
        "instant", "duration", "period", "interval", "clock",
        "calendar", "schedule", "deadline", "noon", "midnight",
        "dawn", "dusk", "morning", "afternoon", "evening", "night",
        "semester", "quarter", "season", "weekend", "weekday",
        "stopwatch", "alarm", "countdown", "timeline",
    ],
    "place": [
        "city", "town", "village", "country", "continent", "island",
        "mountain", "valley", "river", "lake", "ocean", "forest",
        "desert", "park", "garden", "room", "house", "building",
        "street", "road", "bridge", "airport", "station", "hospital",
        "school", "church", "museum", "library", "office",
        "kitchen", "bedroom", "bathroom", "basement", "attic",
        "neighborhood", "district", "province", "territory",
    ],
    "tools": [
        "hammer", "screwdriver", "wrench", "pliers", "saw", "drill",
        "chisel", "clamp", "vise", "level", "ruler", "scissors",
        "knife", "axe", "shovel", "rake", "hoe", "trowel", "brush",
        "sandpaper", "ladder", "wheelbarrow", "crowbar", "mallet",
        "tape measure", "caliper", "file", "bolt cutter", "wire cutter",
        "soldering iron", "hacksaw", "plane", "pickaxe", "spanner",
    ],
}

# Hypernym category labels used in R1 benchmark
# IMPORTANT: Must be single common words for GPT-2 loss-based evaluation.
# Multi-word labels like "Time Unit" have inherently higher loss than
# single-word distractors like "Fruit", causing baseline failure.
HYPERNYM_LABELS = {
    "time": "time",
    "place": "place",
    "tools": "tool",
}

# =====================================================================
# 2. CURATED SYNONYM PAIRS (for R2 benchmark)
# =====================================================================
CURATED_SYNONYMS = {
    "time": [
        ("moment", "instant"),
        ("duration", "period"),
        ("clock", "timepiece"),
        ("era", "epoch"),
        ("schedule", "timetable"),
        ("dawn", "sunrise"),
        ("dusk", "sunset"),
        ("noon", "midday"),
        ("midnight", "twelve o'clock"),
        ("deadline", "due date"),
        ("decade", "ten years"),
        ("century", "hundred years"),
        ("morning", "forenoon"),
        ("afternoon", "post meridiem"),
        ("interval", "gap"),
        ("countdown", "timer"),
    ],
    "place": [
        ("city", "metropolis"),
        ("ocean", "sea"),
        ("house", "home"),
        ("road", "street"),
        ("forest", "woods"),
        ("country", "nation"),
        ("building", "structure"),
        ("lake", "pond"),
        ("mountain", "peak"),
        ("garden", "yard"),
        ("village", "hamlet"),
        ("island", "isle"),
        ("desert", "wasteland"),
        ("room", "chamber"),
        ("neighborhood", "district"),
        ("airport", "airfield"),
    ],
    "tools": [
        ("hammer", "mallet"),
        ("knife", "blade"),
        ("wrench", "spanner"),
        ("shovel", "spade"),
        ("pliers", "tongs"),
        ("axe", "hatchet"),
        ("scissors", "shears"),
        ("saw", "hacksaw"),
        ("drill", "bore"),
        ("brush", "broom"),
        ("chisel", "gouge"),
        ("ladder", "stepladder"),
        ("crowbar", "pry bar"),
        ("clamp", "grip"),
        ("ruler", "straightedge"),
        ("file", "rasp"),
    ],
}

# =====================================================================
# 3. CURATED PARAPHRASE PAIRS (for R2, richer than synonym substitution)
# =====================================================================
CURATED_PARAPHRASES = {
    "time": [
        ("The meeting starts at 5 PM.",
         "The gathering begins at seventeen hundred hours."),
        ("It takes sixty seconds.",
         "The duration is one minute."),
        ("The train arrived on time.",
         "The train was punctual."),
        ("It is a decade long.",
         "It lasts for ten years."),
        ("Wait a moment.",
         "Hold on a second."),
        ("The alarm went off at seven in the morning.",
         "The alarm rang at 7 AM."),
        ("The deadline is next week.",
         "The due date is in seven days."),
        ("He checked his watch frequently.",
         "He kept looking at the clock."),
        ("Noon is the middle of the day.",
         "Midday divides the day in half."),
        ("The century is coming to an end.",
         "The hundred year period is nearly over."),
    ],
    "place": [
        ("Paris is the capital of France.",
         "The French capital is Paris."),
        ("He is at home.",
         "He is in his house."),
        ("The store is nearby.",
         "The shop is close."),
        ("Go north for two miles.",
         "Travel towards the north for two miles."),
        ("The room is empty.",
         "There is nobody in the chamber."),
        ("The city center is crowded.",
         "Downtown is packed with people."),
        ("She lives in a small village.",
         "Her home is in a tiny hamlet."),
        ("The mountain peak is covered in snow.",
         "The summit is blanketed with snow."),
        ("The library is next to the park.",
         "The park is adjacent to the library."),
        ("The desert stretches for miles.",
         "The wasteland extends a great distance."),
    ],
    "tools": [
        ("He used a hammer to drive the nail.",
         "He pounded the nail with a mallet."),
        ("The screwdriver tightened the screw.",
         "The screw was fastened with a screwdriver."),
        ("She cut the paper with scissors.",
         "She used shears to cut the paper."),
        ("The wrench loosened the bolt.",
         "The spanner unfastened the bolt."),
        ("He sawed the wood in half.",
         "He cut the timber using a saw."),
        ("The shovel dug a hole in the ground.",
         "A spade was used to dig into the earth."),
        ("The drill made a hole in the wall.",
         "The bore created an opening in the wall."),
        ("He sharpened the blade with a file.",
         "The knife was honed using a rasp."),
        ("The level ensures a flat surface.",
         "The surface is checked for flatness with a level."),
        ("She used an axe to chop the firewood.",
         "The firewood was chopped with a hatchet."),
    ],
}

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
R1_DISTRACTOR_CATEGORIES = [
    "animal", "fruit", "color", "emotion", "instrument",
    "sport", "fabric", "element", "number",
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
    """
    nodes = {}
    for concept in ["time", "place", "tools"]:
        base = set(CURATED_NODES[concept])

        # Supplement with ConceptNet
        if conceptnet_data and concept in conceptnet_data:
            cn = conceptnet_data[concept]
            base.update(cn.get("hyponyms", []))
            base.update(cn.get("related", []))
            if concept == "place":
                base.update(cn.get("locations", []))
            if concept == "tools":
                base.update(cn.get("used_for", []))

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
    for concept in ["time", "place", "tools"]:
        pairs = list(CURATED_SYNONYMS[concept])
        node_set = set(concept_nodes[concept])

        # Add ConceptNet synonyms where at least one word is in our nodes
        if conceptnet_data and concept in conceptnet_data:
            for pair in conceptnet_data[concept].get("synonyms", []):
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


def generate_cav_training(concept_nodes):
    """
    Generate positive and negative sentences for CAV extraction.

    Positive: sentences containing concept words via varied templates.
    Negative: SAME templates with neutral words (template-matched).

    CRITICAL: Using the same templates for both ensures the CAV captures
    the semantic difference (concept vs non-concept words), not the
    syntactic structure difference (template vs free-form sentence).
    Without this, all concept CAVs become nearly parallel (cos > 0.8).
    """
    # Collect ALL concept words across ALL categories for contamination check
    all_concept_words = set()
    for words in concept_nodes.values():
        all_concept_words.update(w.lower() for w in words)

    # Build clean neutral word pool (no concept words)
    clean_neutral_words = [
        w for w in NEUTRAL_WORDS
        if w.lower() not in all_concept_words
    ]
    print(f"  Neutral word pool: {len(clean_neutral_words)} words")

    training = {}

    for concept in ["time", "place", "tools"]:
        nodes = concept_nodes[concept]

        # --- Positive sentences ---
        positives = []
        pos_templates_used = []  # Track which templates are used
        for word in nodes:
            # Pick 5 random templates for each word
            selected_templates = random.sample(
                ALL_TEMPLATES, min(5, len(ALL_TEMPLATES))
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

        training[concept] = {
            "positive": positives,
            "negative": negatives,
        }

        print(f"  {concept.upper()}: {len(positives)} pos, "
              f"{len(negatives)} neg (template-matched)")

    return training


def generate_r1_benchmark(concept_nodes):
    """
    Generate R1 Benchmark: Multiple-choice hypernym classification.

    For each concept word, ask: "Which category does '{word}' belong to?"
    Correct answer: the hypernym label (Time Unit / Location / Tool)
    Distractors: 3 random unrelated categories.
    """
    benchmarks = {}

    for concept in ["time", "place", "tools"]:
        correct_label = HYPERNYM_LABELS[concept]
        nodes = concept_nodes[concept]

        # Select up to 30 words for the benchmark
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

    for concept in ["time", "place", "tools"]:
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

        # Deduplicate
        seen = set()
        unique = []
        for s1, s2 in pairs:
            key = (s1.strip(), s2.strip())
            if key not in seen:
                seen.add(key)
                unique.append([s1, s2])

        # Cap at 25 pairs
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
    cav_training = generate_cav_training(concept_nodes)

    # --- Step 5: Generate benchmarks ---
    print("\n[5/5] Generating benchmarks...")
    print("  --- R1 (Hypernym Classification) ---")
    r1_benchmark = generate_r1_benchmark(concept_nodes)
    print("  --- R2 (Semantic Similarity) ---")
    r2_benchmark = generate_r2_benchmark(synonym_pairs)

    # --- Assemble output ---
    output = {
        "metadata": {
            "concepts": ["time", "place", "tools"],
            "generated_at": datetime.now().isoformat(),
            "conceptnet_used": conceptnet_data is not None,
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
                "hypernym_label": HYPERNYM_LABELS[concept],
                "synonym_pairs": synonym_pairs[concept],
            }
            for concept in ["time", "place", "tools"]
        },
        "cav_training": cav_training,
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
    for concept in ["time", "place", "tools"]:
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
