"""
ConceptNet Concept Extractor
=============================
Extracts concept nodes for TIME, PLACE, and TOOLS from ConceptNet 5.7
via the HuggingFace datasets library (streaming mode for memory efficiency).

Usage:
    pip install datasets
    python conceptnet_extractor.py

Output: conceptnet_concepts.json
"""

import json
import os
import sys
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed.")
    print("Run: pip install datasets")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_WEIGHT = 1.0
OUTPUT_FILE = "conceptnet_concepts.json"

# Seed URIs for each concept category
CONCEPT_SEEDS = {
    "time": {
        "isa_targets": {
            "/c/en/time", "/c/en/time_unit", "/c/en/time_period",
            "/c/en/unit_of_time", "/c/en/period_of_time",
            "/c/en/temporal_quantity", "/c/en/time_interval",
            "/c/en/unit_of_measurement", "/c/en/measure",
        },
        "related_targets": {
            "/c/en/time", "/c/en/temporal", "/c/en/chronology",
            "/c/en/clock", "/c/en/calendar",
        },
    },
    "place": {
        "isa_targets": {
            "/c/en/place", "/c/en/location", "/c/en/area",
            "/c/en/region", "/c/en/geographic_location",
            "/c/en/geographical_area", "/c/en/body_of_water",
            "/c/en/landform", "/c/en/building", "/c/en/room",
            "/c/en/country", "/c/en/city",
        },
        "related_targets": {
            "/c/en/place", "/c/en/location", "/c/en/geography",
            "/c/en/spatial",
        },
    },
    "tools": {
        "isa_targets": {
            "/c/en/tool", "/c/en/hand_tool", "/c/en/power_tool",
            "/c/en/implement", "/c/en/cutting_tool",
            "/c/en/measuring_instrument", "/c/en/garden_tool",
            "/c/en/woodworking_tool",
        },
        "related_targets": {
            "/c/en/tool", "/c/en/instrument", "/c/en/implement",
            "/c/en/utensil",
        },
    },
}

# Relations we care about
RELEVANT_RELATIONS = {
    "/r/IsA", "/r/RelatedTo", "/r/Synonym", "/r/Antonym",
    "/r/AtLocation", "/r/UsedFor", "/r/HasProperty",
    "/r/HasA", "/r/PartOf", "/r/CapableOf",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_word(node_uri: str) -> str | None:
    """Extract clean English word from a ConceptNet URI like /c/en/clock."""
    parts = node_uri.split("/")
    if len(parts) >= 4 and parts[2] == "en":
        word = parts[3].replace("_", " ")
        # Filter: skip very short or very long entries
        if 2 <= len(word) <= 30:
            return word.lower()
    return None


def is_english(lang_field: str) -> bool:
    """Check if the edge is English (handles 'en', 'en/en', etc.)."""
    return lang_field == "en" or lang_field.startswith("en/en")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ConceptNet Concept Extractor")
    print("=" * 60)
    print(f"Loading ConceptNet from HuggingFace (streaming mode)...")
    print(f"This scans ~34M rows. It may take 20-60 minutes.\n")

    dataset = load_dataset(
        "conceptnet5/conceptnet5",
        name="conceptnet5",
        split="train",
        streaming=True,
    )

    # -- Collectors per concept --
    # Each category maps word -> max_weight (keeps highest weight per word)
    concepts = {}
    for name in CONCEPT_SEEDS:
        concepts[name] = {
            "hyponyms": {},     # X IsA <seed_target> → {word: weight}
            "related": {},      # RelatedTo connections
            "properties": {},   # HasProperty
            "used_for": {},     # UsedFor (mainly tools)
            "locations": {},    # AtLocation (mainly place)
        }

    # Global synonym / antonym collectors (filter later)
    all_en_synonyms = []   # (word1, word2, weight)
    all_en_antonyms = []   # (word1, word2, weight)

    # Surface text sentences from ConceptNet (bonus natural-language data)
    concept_sentences = defaultdict(list)  # concept_name -> [sentence]

    count = 0
    matched = 0

    for row in dataset:
        count += 1
        if count % 1_000_000 == 0:
            print(f"  Processed {count / 1_000_000:.0f}M rows "
                  f"({matched} matches so far)...")

        # -- Filter: English only --
        if not is_english(row["lang"]):
            continue

        rel = row["rel"]
        if rel not in RELEVANT_RELATIONS:
            continue

        arg1 = row["arg1"]
        arg2 = row["arg2"]
        weight = row["weight"]

        if weight < MIN_WEIGHT:
            continue

        # -- Collect ALL English synonyms/antonyms for later filtering --
        if rel == "/r/Synonym":
            w1 = extract_word(arg1)
            w2 = extract_word(arg2)
            if w1 and w2 and w1 != w2:
                all_en_synonyms.append((w1, w2, weight))
            continue

        if rel == "/r/Antonym":
            w1 = extract_word(arg1)
            w2 = extract_word(arg2)
            if w1 and w2 and w1 != w2:
                all_en_antonyms.append((w1, w2, weight))
            continue

        # -- Check each concept category --
        for concept_name, seeds in CONCEPT_SEEDS.items():

            # IsA: arg1 IsA arg2. If arg2 is a seed target, arg1 is a hyponym.
            if rel == "/r/IsA":
                if arg2 in seeds["isa_targets"]:
                    word = extract_word(arg1)
                    if word:
                        prev = concepts[concept_name]["hyponyms"].get(word, 0)
                        concepts[concept_name]["hyponyms"][word] = max(prev, weight)
                        matched += 1
                        # Capture natural sentence if available
                        sent = row.get("sentence", "")
                        if sent and "[[" in sent:
                            clean = sent.replace("[[", "").replace("]]", "")
                            concept_sentences[concept_name].append(clean)

            # RelatedTo: bidirectional
            elif rel == "/r/RelatedTo":
                if arg1 in seeds["related_targets"]:
                    word = extract_word(arg2)
                    if word:
                        prev = concepts[concept_name]["related"].get(word, 0)
                        concepts[concept_name]["related"][word] = max(prev, weight)
                        matched += 1
                elif arg2 in seeds["related_targets"]:
                    word = extract_word(arg1)
                    if word:
                        prev = concepts[concept_name]["related"].get(word, 0)
                        concepts[concept_name]["related"][word] = max(prev, weight)
                        matched += 1

            # AtLocation (mainly for place)
            elif rel == "/r/AtLocation" and concept_name == "place":
                word = extract_word(arg2)
                if word:
                    prev = concepts[concept_name]["locations"].get(word, 0)
                    concepts[concept_name]["locations"][word] = max(prev, weight)
                    matched += 1

            # UsedFor (mainly for tools)
            elif rel == "/r/UsedFor" and concept_name == "tools":
                word = extract_word(arg1)
                if word:
                    prev = concepts[concept_name]["used_for"].get(word, 0)
                    concepts[concept_name]["used_for"][word] = max(prev, weight)
                    matched += 1

            # HasProperty
            elif rel == "/r/HasProperty":
                for target in seeds["related_targets"]:
                    if arg1 == target:
                        prop = extract_word(arg2)
                        if prop:
                            prev = concepts[concept_name]["properties"].get(prop, 0)
                            concepts[concept_name]["properties"][prop] = max(prev, weight)
                            matched += 1
                        break

    print(f"\nScan complete: {count} rows processed, {matched} matches.")
    print(f"Synonyms collected: {len(all_en_synonyms)}")
    print(f"Antonyms collected: {len(all_en_antonyms)}")

    # ------------------------------------------------------------------
    # Post-processing: build per-concept word lists (with weights)
    # ------------------------------------------------------------------
    output = {}

    def weighted_list(word_weight_dict):
        """Convert {word: weight} dict to sorted list of {word, weight}."""
        return sorted(
            [{"word": w, "weight": wt} for w, wt in word_weight_dict.items()],
            key=lambda x: x["weight"],
            reverse=True,
        )

    for concept_name in CONCEPT_SEEDS:
        c = concepts[concept_name]

        # Merge all discovered nodes (set of words for filtering)
        all_nodes = set()
        all_nodes.update(c["hyponyms"].keys())
        all_nodes.update(c["related"].keys())
        all_nodes.update(c["locations"].keys())
        all_nodes.update(c["used_for"].keys())

        # Filter synonyms/antonyms: keep pairs where at least one word
        # is in this concept's node list (preserve weights)
        concept_synonyms = []
        for w1, w2, wt in all_en_synonyms:
            if w1 in all_nodes or w2 in all_nodes:
                concept_synonyms.append({"word_1": w1, "word_2": w2, "weight": wt})

        concept_antonyms = []
        for w1, w2, wt in all_en_antonyms:
            if w1 in all_nodes or w2 in all_nodes:
                concept_antonyms.append({"word_1": w1, "word_2": w2, "weight": wt})

        # Sort synonyms/antonyms by weight descending
        concept_synonyms.sort(key=lambda x: x["weight"], reverse=True)
        concept_antonyms.sort(key=lambda x: x["weight"], reverse=True)

        output[concept_name] = {
            "hyponyms": weighted_list(c["hyponyms"]),
            "related": weighted_list(c["related"]),
            "locations": weighted_list(c["locations"]),
            "used_for": weighted_list(c["used_for"]),
            "properties": weighted_list(c["properties"]),
            "all_nodes": sorted(all_nodes),
            "synonyms": concept_synonyms,
            "antonyms": concept_antonyms,
            "natural_sentences": concept_sentences.get(concept_name, [])[:50],
        }

        print(f"\n--- {concept_name.upper()} ---")
        print(f"  Hyponyms: {len(c['hyponyms'])}")
        print(f"  Related:  {len(c['related'])}")
        print(f"  Locations:{len(c['locations'])}")
        print(f"  UsedFor:  {len(c['used_for'])}")
        print(f"  Total nodes: {len(all_nodes)}")
        print(f"  Synonyms: {len(concept_synonyms)}")
        print(f"  Antonyms: {len(concept_antonyms)}")
        print(f"  Sentences: {len(concept_sentences.get(concept_name, []))}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
