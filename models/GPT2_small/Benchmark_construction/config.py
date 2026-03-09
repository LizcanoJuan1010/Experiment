import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONCEPTNET_FILE = os.path.join(BASE_DIR, "..", "conceptnet_concepts.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Concepts and their hypernym labels (as they appear in ConceptNet IsA)
# ---------------------------------------------------------------------------
CONCEPTS = [
    "time", "place", "tool",
    "animal", "bird", "insect", "plant", "fruit", "vegetable",
    "vehicle", "furniture", "clothing", "container", "instrument",
    "weapon", "material", "body_part", "food", "drink",
    "profession", "sport", "emotion"
]

# The correct answer for R1: "X is a type of ___"
# Must match the superordinate category name used in ConceptNet.
HYPERNYM_LABELS = {
    "time": "time",
    "place": "place",
    "tool": "tool",
    "animal": "animal",
    "bird": "bird",
    "insect": "insect",
    "plant": "plant",
    "fruit": "fruit",
    "vegetable": "vegetable",
    "vehicle": "vehicle",
    "furniture": "furniture",
    "clothing": "clothing",
    "container": "container",
    "instrument": "musical instrument",  # Specific to avoid confusion with tools
    "weapon": "weapon",
    "material": "material",
    "body_part": "body part",
    "food": "food",
    "drink": "drink",
    "profession": "profession",
    "sport": "sport",
    "emotion": "emotion",
}

# ---------------------------------------------------------------------------
# R1: MCQ Cloze — Template expressing IsA relation
# ---------------------------------------------------------------------------
# Five templates that probe taxonomic knowledge from different angles.
# Each uses {word} as placeholder for the hyponym (filled with .title()).
R1_TEMPLATES = [
    "In a taxonomic hierarchy, the entity '{word}' is classified as a type of",
    "From a semantic perspective, the concept of '{word}' belongs to the category of",
    "The fundamental category that defines '{word}' is",
    "Task: Classify the following term into a broad category.\nTerm: {word}\nCategory:",
    "Logically, '{word}' is considered an instance of",
]

# Distractor categories — taxonomically disjoint from the 22 target concepts.
# These categories are used when we need options that are clearly NOT the target.
R1_DISTRACTOR_CATEGORIES = [
    "color",
    "shape",
    "number",
    "language",
    "element",
    "science",
    "art",
    "music",
    "weather",
    "feeling",
    "idea",
    "system",
]

# ---------------------------------------------------------------------------
# R2: Paraphrase Similarity — Sentence pair templates
# ---------------------------------------------------------------------------
# Each tuple is (sentence_with_w1, sentence_with_w2).
# The two sentences must be semantically equivalent but syntactically distinct,
# so that cosine similarity measures semantic (not surface) representation.
# All templates naturally require noun arguments.
R2_TEMPLATES = [
    ("The {w1} played a crucial role in the outcome.",
     "The {w2} was a key factor in the result."),
    ("We carefully observed the {w1}.",
     "They inspected the {w2} closely."),
    ("Using the {w1} solved the problem.",
     "The application of the {w2} fixed the issue."),
    ("The {w1} was discussed during the meeting.",
     "The {w2} came up in the conference."),
    ("He described the {w1} in great detail.",
     "She provided a thorough account of the {w2}."),
]

# ---------------------------------------------------------------------------
# Quality blacklist — ConceptNet entries that are factually incorrect or
# noise for the target concept. Each entry was manually verified against
# the ConceptNet IsA relation and excluded because it represents a wrong
# taxonomic classification (e.g., "ohm IsA measure" → not a time concept).
# ---------------------------------------------------------------------------
BLACKLIST = {
    "time": {
        # Units of length/resistance/amount from overly broad seed
        "foot",             # unit of length, not time
        "yard",             # unit of length, not time
        "ohm",              # unit of electrical resistance
        "mole",             # unit of amount (chemistry)
        "complex measure",  # mathematical term
        "positive measure", # mathematical term
        "compile time",     # CS jargon, not a natural time unit
        "fourth dimension", # physics/philosophy, not a time unit
        # Weak synonymy / wrong concept (R2 cleanup)
        "order",            # "appointment/order" — weak synonymy
        "cock",             # ambiguous (clock piece / vulgar)
        "hammer",           # tool, not time (paired with cock)
        "convenience",      # not a time concept
        "appliance",        # not a time concept
        "crow's age",       # idiom, culturally biased
        "coon's age",       # idiom, culturally offensive
        "dimension",        # abstract general, not time
        "aspect",           # abstract general, not time
        "dinner",           # event/meal, not time itself
        "supper",           # event/meal, not time itself
        "employment",       # not time
        "use",              # not time
        "harvest",          # agricultural event, not time
        "issue",            # not time
        "idle",             # adjective/state, not time
        "light",            # not time
        "impulse",          # physics
        "force",            # physics
        # Generic classification terms — not time-specific
        "category",         # abstract classification
        "class",            # abstract classification
        "division",         # abstract classification
        # Opposite / non-temporal
        "extratemporal",    # means "outside time"
        # Generic measurement terms — not time-specific
        "instrument",       # generic device
        "measure",          # too generic (cooking, length, etc.)
        "measurement",      # too generic
        "meter",            # unit of length, not time
        # Wrong category
        "sleep",            # activity, not a time concept
        "number",           # math concept
        "digit",            # math concept
        "space",            # polysemous — place concept
        "hour",             # minute≠hour — not synonyms (different units)
        # Concept name / abstract non-temporal (R2 Tier 3 cleanup)
        "time",             # concept name itself
        "purpose",          # not time-related
        "function",         # not time-related
        "shadow",           # not time-related
        "tail",             # not time-related
        "sith",             # archaic, not useful
        "progression",      # generic, not time-specific
        "sequence",         # generic, not time-specific
        # R2 Tier 3 pass 2
        "speed",            # velocity, not time
        "rush",             # velocity, not time
        "speedup",          # velocity, not time
        "acceleration",     # velocity, not time
        "story",            # narrative, not time
        "stratum",          # geological layers
    },
    "place": {
        # Furniture/appliances — not places
        "bed",
        "chair",
        "desk",
        "fridge",
        "grill",
        # Noise / garbage strings from ConceptNet
        "england still",
        "find downtown",
        "find upstairs",
        "another way to say",
        "another way to say germany",
        "purging place",
        # Misspellings
        "ethopia",          # misspelling of "ethiopia"
        "camerun",          # misspelling of "cameroon"
        # Too abstract / deictic
        "here",
        # Food (wrong concept)
        "chili",            # food, not Chile the country
        # Abstract/legal/philosophical — not physical places
        "accident",
        "property",         # legal concept, not place
        "state",            # too polysemous (condition vs. nation-state)
        # Emotions — not places
        "agony",
        "distress",
        # Substances/media — not geographic places
        "air",
        "atmosphere",
        # Polysemy errors — wrong sense
        "argument",         # fight, not place
        "row",              # fight sense, not spatial row
        "arm",              # body part / weapon, not place
        "branch",           # polysemous (tree/org branch)
        "weapon",           # not a place
        # Vulgar / body parts — not places
        "arse",
        "ass",
        "asshole",
        "butt",
        # Vehicles — not places
        "aeroplane",        # vehicle
        "airplane",         # vehicle
        "plane",            # vehicle
        "automobile",       # vehicle
        "car",              # vehicle
        # Groups / organizations — not physical places
        "army",             # group of people
        "host",             # abstract / group
        "humane society",   # organization, not physical place
        # Objects / containers — not places
        "backpack",         # object
        "bookbag",          # object
        "knapsack",         # object
        "bag",              # object
        "handbag",          # object
        "poke",             # dialectal for bag
        "pocket",           # object / body part
        "band",             # object / music group
        "ring",             # object / jewelry
        "base",             # too polysemous
        "cap",              # object (headwear)
        "pipe",             # object, not place
        # Other wrong category
        "dog",              # animal
        "sand",             # substance, not place
        "bomb",             # weapon
        "facility",         # too generic
        "gap",              # abstract
        "cell phone",       # object
        # Wrong synonym relationships (R2 Tier cleanup)
        "michigan",         # chicago≠michigan — not synonyms
        "taiwan",           # china≠taiwan — not synonyms, politically sensitive
        "box",              # object, not a place (box≠shelter)
        "diner",            # diner≠pub — not synonyms
        # R2 Tier 2 pass 2
        "family",           # not a place (house≠family)
        "firm",             # organization, not place (house≠firm)
        "shop",             # house≠shop — wrong synonym sense
        "store",            # house≠store — wrong synonym sense
    },
    "tool": {
        # Clearly not tools
        "open mind",
        "rubber chicken",
        # Vehicles — not tools
        "car",
        "airplane",
        "plane",
        "bicycle",
        "bike",
        "motorcycle",
        "baby buggy",       # baby gear
        "perambulator",     # baby gear
        # Musical instruments — separate category
        "saxophone",
        "triangle",
        "bagpipe",
        "bagpipes",
        "alto",             # voice type
        "contralto",        # voice type
        "bass",             # music
        "basso",            # music
        # Overly broad / wrong category
        "weapon",
        "atomic bomb",      # weapon
        "bomb",             # weapon
        # Financial documents / money — not tools
        "account book",
        "ledger",
        "billfold",
        "wallet",
        "coin",
        # Appliances/systems — not hand tools
        "air conditioning",
        "air conditioner",
        # Tickets — not tools
        "air ticket",
        "plane ticket",
        # Places — not tools
        "aisle",
        "isle",
        "alley",
        "apartment",
        "apartment building",
        "bathroom",
        "toilet",
        "bay window",
        "arena",
        "stadium",
        "bar",              # place (pub)
        "barroom",          # place
        "basement",         # place
        "cellar",           # place
        "beauty salon",     # place
        "salon",            # place
        # Biology / people — not tools
        "animal",
        "creature",
        "baby",
        "child",
        "bird",
        # Furniture — not tools
        "beanbag",
        "beanbag chair",
        "bench",
        "bath",
        "bathtub",
        # Food / herbs — not tools
        "cake",
        "basil",
        # Sports equipment — not tools
        "ball",
        "baseball",
        "musket ball",
        "basket",
        "basketball",
        "bat",
        "club",
        # Containers — not tools
        "bag",
        "bin",
        "dustbin",
        "garbage can",
        "trash can",
        "container",
        # Media / clothing / linen — not tools
        "album",
        "disk",
        "jacket",
        "belt",
        "bed sheet",
        "sheet",
        "poster",
        # Abstract / generic / wrong
        "dog",
        "pot",
        "love",
        "award",
        "crown",
        "baggage",
        "luggage",
        "bank",
        "line",
        "pipe",
        "base",
        "foundation",
        "floor",
        "beam",
        "radio beam",
        "judiciary",
        "terrace",
        "book",
        "band",
        "ring",
        "time",             # different concept
        "bit",              # too polysemous
        "portion",          # abstract
        # R2 Tier 2 cleanup — wrong synonym pairs with tool hyponyms
        "tongue",           # body part, not a tool
        "degree",           # academic/measurement
        "stab",             # verb/action, not a tool
        "bomber",           # aircraft/person, not a tool
        "stage",            # place/platform, not a tool
        "stick",            # too polysemous
        # R2 Tier 3 cleanup — vehicles
        "boat",
        "ship",
        # R2 Tier 3 cleanup — places/shops
        "bookshop",
        "bookstore",
        "boulevard",
        "avenue",
        "box office",
        "ticket office",
        "counter",
        # R2 Tier 3 cleanup — organizations/people
        "committee",
        "council",
        # R2 Tier 3 cleanup — clothing/footwear
        "boot",
        "buskin",
        # R2 Tier 3 cleanup — bodies of water/nature
        "brook",
        "creek",
        "stream",
        # R2 Tier 3 cleanup — weapons
        "sword",
        "lance",
        # R2 Tier 3 cleanup — furniture/food/containers
        "table",
        "dining table",
        "bottle",
        "flask",
        "cup",
        "case",
        "box",
        # R2 Tier 3 cleanup — music
        "brass",
        "brass instrument",
        "plate",
        # R2 Tier 3 cleanup — abstract/misc
        "border",
        "boundary",
        "edge",
        "frame",
        "props",
        "bones",
        "tool",             # concept name itself
        "control panel",
        "board",            # too polysemous (furniture/org/game)
        # R2 Tier 3 pass 2 — places
        "car park",
        "parking lot",
        "cemetery",
        "graveyard",
        "church",
        "chapel",
        "churchyard",
        "cinema",
        "movie theater",
        # R2 Tier 3 pass 2 — furniture/furnishing
        "buffet",
        "sideboard",
        "chair",
        "stool",
        "electric chair",
        "carpet",
        "rug",
        "carpeting",
        "chest",
        "coffin",
        "trunk",
        # R2 Tier 3 pass 2 — people/roles
        "captain",
        "pilot",
        "president",
        # R2 Tier 3 pass 2 — vehicles
        "helicopter",
        "bullet train",
        # R2 Tier 3 pass 2 — non-tool objects / wrong category
        "tail",
        "bug",
        "head",
        "bullet",
        "canvas",
        "sail",
        "card",
        "menu",
        "cavity",
        "pit",
        "candle",
        "chant",
        "tone",
        "cigarette",
        "tab",
        "pickaxe",          # IS a tool, but bill≠pickaxe
        # R2 Tier 3 pass 3 — textiles / materials
        "clay",
        "mud",
        "cloth",
        "material",
        "fabric",
        "clothes",
        "clothing",
        # R2 Tier 3 pass 3 — food / drink
        "coffee",
        "chocolate",
        # R2 Tier 3 pass 3 — accessories / bedding
        "coin purse",
        "purse",
        "collar",
        "choker",
        "comforter",
        "duvet",
        # R2 Tier 3 pass 3 — places
        "commons",
        "park",
        "convenience store",
        "corner shop",
        "courtroom",
        "court",
        "courtyard",
        # R2 Tier 3 pass 3 — organizations / abstract
        "company",
        "business",
        "party",
        "contract",
        "sign",
        "credit",
        "faith",
        # R2 Tier 3 pass 3 — furniture
        "couch",
        "sofa",
        "cradle",
        "rocker",
        # R2 Tier 3 pass 3 — misc objects
        "cube",
        "dice",
        # R2 Tier 3 pass 4 — places
        "deli",
        "delicatessen",
        "disco",
        "nightclub",
        "dressing room",
        "locker room",
        "cupboard",
        "pantry",
        # R2 Tier 3 pass 4 — furniture / bedding
        "wardrobe",
        "cushion",
        "pillow",
        # R2 Tier 3 pass 4 — medical / nature / money
        "disease",
        "illness",
        "earth",
        "soil",
        "world",
        "dollar bill",
        "dollar",
        # R2 Tier 3 pass 4 — weapons / abstract
        "cutlass",
        "culture",
        "polish",
        "disguise",
        "doctor",
        "repair",
        "hanger",
        "drum",
        "barrel",
        "diner",
        "pub",
        # R2 Tier 3 pass 5 — places
        "eatery",
        "restaurant",
        "entrance hall",
        "english channel",
        "channel",
        # R2 Tier 3 pass 5 — food / nature
        "eggs",
        "egg",
        "feather",
        "feathers",
        # R2 Tier 3 pass 5 — abstract / structural / misc
        "execution",
        "performance",
        "pawn",
        "wall",
        "fence",
        # R2 Tier 3 pass 6 — food / nature / media / textile
        "food",
        "bread",
        "meat",
        "flower",
        "cream",
        "film",
        "movie",
        "flannel",
        "washcloth",
        "field",
        "address",
        "document",
        # R2 Tier 3 pass 7 — sports / nature / places / abstract
        "football",
        "soccerball",
        "forest",
        "wood",
        "form",
        "formula",
        "rule",
        "foyer",
        "lobby",
        "freeway",
        "motorway",
        # R2 Tier 3 pass 8 — not tools (garden/farm)
        "garden",
        "yard",
        "farm",
    },
}

# ---------------------------------------------------------------------------
# Targets and thresholds
# ---------------------------------------------------------------------------
TARGET_R1_PER_CONCEPT = 150
TARGET_R2_PER_CONCEPT = 50
MIN_CONCEPTNET_WEIGHT = 1.0  # Minimum ConceptNet edge weight for inclusion
