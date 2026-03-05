"""
BEA Naming Benchmark — Denominación por Confrontación
======================================================
Simulates the BEA confrontation naming subtest (Cuetos & González-Nosti,
2009). In clinical practice, the patient sees an image and must produce
the correct name. Here we substitute the visual stimulus with a functional
description and ask the model to select the correct word among
intra-category distractors.

Key difference from R1: distractors are from the SAME semantic category,
not from taxonomically distant categories. This makes the task harder and
more clinically relevant — semantic aphasia degrades intra-category
distinctions, not inter-category ones.

Output format is compatible with evaluate_r1() in metrics.py.

Output: Benchmark_construction/output/bea_naming_benchmark.json

Usage:
    python -m Benchmark_construction.build_bea_naming

References:
    - Cuetos & González-Nosti (2009) "Batería de Evaluación de la Afasia"
    - Howard & Patterson (1992) "Pyramids and Palm Trees Test"
    - Warrington & McCarthy (1987) "Categories of knowledge"
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Curated functional descriptions per concept word
# ---------------------------------------------------------------------------
# Each entry: (word, description)
# Descriptions are functional/perceptual, NOT taxonomic.
# They describe what the object IS, DOES, or LOOKS LIKE — mimicking
# what a patient would see in a confrontation naming test.

NAMING_ITEMS = {
    "time": [
        ("hour", "The unit of time equal to sixty minutes or three thousand six hundred seconds"),
        ("minute", "The unit of time equal to sixty seconds, one sixtieth of an hour"),
        ("second", "The base unit of time, one sixtieth of a minute"),
        ("day", "The period of twenty-four hours, measured from midnight to midnight"),
        ("week", "A period consisting of seven consecutive days"),
        ("month", "One of the twelve divisions of the calendar year"),
        ("year", "The period of approximately three hundred sixty-five days, one orbit around the Sun"),
        ("decade", "A period spanning exactly ten consecutive years"),
        ("century", "A period spanning exactly one hundred years"),
        ("millennium", "A period spanning exactly one thousand years"),
        ("epoch", "A long and distinct period of history marked by notable characteristics"),
        ("era", "A major division of historical time, distinguished by significant developments"),
        ("moment", "A very brief and fleeting period of time"),
        ("instant", "An extremely short span of time, happening immediately"),
        ("duration", "The total length of time that something continues or lasts"),
        ("period", "A defined length or portion of time with particular characteristics"),
        ("interval", "The time that elapses between two events"),
        ("clock", "The device with hands or digits used to measure and display the current time"),
        ("calendar", "The printed chart organizing days, weeks, and months within a year"),
        ("schedule", "A timetable listing planned activities and the times they should occur"),
        ("deadline", "The latest date or time by which a task must be completed"),
        ("noon", "Twelve o'clock in the middle of the day"),
        ("midnight", "Twelve o'clock in the middle of the night"),
        ("dawn", "The time when light first appears in the sky before the sun rises"),
        ("dusk", "The time after sunset when the sky gradually becomes darker"),
        ("morning", "The part of the day from sunrise until noon"),
        ("afternoon", "The part of the day from noon until evening"),
        ("evening", "The later part of the day from late afternoon until night begins"),
        ("night", "The period of darkness between sunset and sunrise"),
        ("season", "One of the four divisions of the year: spring, summer, autumn, winter"),
    ],
    "place": [
        ("city", "A large densely populated urban area with its own government"),
        ("town", "A populated settlement larger than a village but smaller than a city"),
        ("village", "A small rural settlement with only a few hundred inhabitants"),
        ("country", "A nation with its own government, defined borders, and sovereignty"),
        ("continent", "One of the seven large continuous land masses on Earth"),
        ("island", "A piece of land completely surrounded by water on all sides"),
        ("mountain", "A large natural elevation of the earth rising steeply above its surroundings"),
        ("valley", "A low-lying area of land between hills or mountains"),
        ("river", "A large natural stream of fresh water flowing toward the sea or a lake"),
        ("lake", "A large body of fresh water surrounded by land on all sides"),
        ("ocean", "The vast body of salt water covering most of the earth's surface"),
        ("forest", "A large area of land densely covered with trees and undergrowth"),
        ("desert", "An arid barren region with very little rainfall and sparse vegetation"),
        ("park", "A public area of green land set aside for recreation and enjoyment"),
        ("garden", "A cultivated piece of ground used for growing flowers and plants"),
        ("room", "An enclosed area within a building, separated by walls"),
        ("house", "A building designed for people to live in, typically for one family"),
        ("building", "A permanent structure with walls and a roof, constructed for a purpose"),
        ("street", "A public paved road in a city or town, usually lined with buildings"),
        ("road", "A wide path with a hard surface built for vehicles to travel along"),
        ("bridge", "A structure built to span a river or obstacle to allow passage across"),
        ("airport", "The complex of runways and terminals where aircraft take off and land"),
        ("station", "The place where trains or buses regularly stop for passengers"),
        ("hospital", "The large building where sick and injured people receive medical treatment"),
        ("school", "The institution where children and young people receive education"),
        ("museum", "The building where objects of historical or artistic interest are displayed"),
        ("library", "The building containing organized collections of books for public use"),
        ("office", "A room or building where people work at desks for business purposes"),
        ("kitchen", "The room in a dwelling where food is prepared and cooked"),
        ("bedroom", "The room furnished with a bed, used primarily for sleeping"),
    ],
    "tools": [
        ("hammer", "The hand tool with a heavy metal head on a handle, used to drive nails"),
        ("screwdriver", "The tool with a flat or cross-shaped tip, designed to turn screws"),
        ("wrench", "The adjustable metal tool used to grip and tighten nuts and bolts"),
        ("pliers", "The pivoted gripping tool with two handles and jaws, used to hold or bend wire"),
        ("saw", "The cutting tool with a toothed metal blade, used to cut through wood"),
        ("drill", "The rotating tool used to bore circular holes into wood or metal"),
        ("chisel", "The tool with a sharp flat blade at one end, used to carve wood or stone"),
        ("clamp", "The fastening device with adjustable jaws, used to hold objects tightly together"),
        ("vise", "The heavy clamping device mounted on a workbench with two parallel jaws"),
        ("level", "The instrument with a bubble in liquid, used to check if a surface is horizontal"),
        ("ruler", "The straight flat tool marked with units of length along its edge"),
        ("scissors", "The cutting instrument with two sharp blades joined at a central pivot"),
        ("knife", "The sharp-edged cutting blade with a handle, used for slicing"),
        ("axe", "The heavy chopping tool with a broad metal blade fixed to a long handle"),
        ("shovel", "The broad-bladed tool with a long handle, used for digging earth"),
        ("rake", "The long-handled garden tool with a row of prongs, used to gather leaves"),
        ("hoe", "The flat-bladed garden tool on a long handle, used to break up soil"),
        ("trowel", "The small hand tool with a flat pointed blade, used to spread mortar"),
        ("brush", "The tool with bristles on a handle, used for painting or cleaning surfaces"),
        ("sandpaper", "The abrasive coated paper used to smooth and polish rough surfaces"),
        ("ladder", "The portable climbing structure with rungs between two side rails"),
        ("wheelbarrow", "The single-wheeled cart with two handles, used to carry heavy loads"),
        ("crowbar", "The heavy flat iron lever with a curved end, used to pry things open"),
        ("mallet", "The hammer-like striking tool with a large head made of wood or rubber"),
        ("caliper", "The precision measuring instrument with two adjustable legs for measuring thickness"),
        ("file", "The metal tool with a rough abrasive surface, used to smooth hard materials"),
        ("hacksaw", "The fine-toothed saw in a metal frame, used for cutting metal pipes"),
        ("anvil", "The heavy iron block with a flat top, used as a surface for shaping metal"),
        ("awl", "The small pointed tool used to pierce holes in leather or wood"),
        ("lathe", "The machine that rotates a workpiece to perform cutting and shaping operations"),
    ],
}

# ---------------------------------------------------------------------------
# Naming prompt templates
# ---------------------------------------------------------------------------
# Varied phrasing to avoid the model overfitting to a single surface form.
NAMING_TEMPLATES = [
    "{description}. This is called a",
    "{description}. The name for this is a",
    "{description}. The word that describes this is",
    "What is the following? {description}. It is a",
    "{description}. This object is known as a",
]

# ---------------------------------------------------------------------------
# Target items per concept
# ---------------------------------------------------------------------------
TARGET_PER_CONCEPT = 30


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_naming_items(concept, items, all_words_in_concept):
    """Generate BEA naming MCQ items for one concept.

    Each item has:
      - A functional description as the prompt
      - The correct word
      - 3 distractors from the SAME concept (intra-category)

    This mimics the BEA confrontation naming subtest where the patient
    must discriminate between semantically similar alternatives.
    """
    rng = random.Random(42)
    n_templates = len(NAMING_TEMPLATES)

    benchmark_items = []
    for i, (word, description) in enumerate(items[:TARGET_PER_CONCEPT]):
        # Select template (cycle through)
        template = NAMING_TEMPLATES[i % n_templates]
        question = template.format(description=description)

        # Intra-category distractors: other words from the SAME concept
        candidates = [w for w, _ in items if w != word]
        distractors = rng.sample(candidates, min(3, len(candidates)))

        benchmark_items.append({
            "question": question,
            "options": {
                "correct": word,
                "distractors": distractors,
            },
            "provenance": {
                "word": word,
                "concept": concept,
                "description": description,
                "test_type": "bea_naming",
                "distractor_type": "intra-category",
                "clinical_analog": "Denominación por confrontación (BEA)",
            },
        })

    return benchmark_items


def main():
    print("=" * 60)
    print("BEA Naming Benchmark (Denominación por Confrontación)")
    print("=" * 60)

    benchmark = {}
    all_items = []

    for concept, items in NAMING_ITEMS.items():
        print(f"\n  {concept.upper()}: {len(items)} curated descriptions")

        all_words = [w for w, _ in items]
        generated = build_naming_items(concept, items, all_words)
        benchmark[concept] = generated
        all_items.extend(generated)

        print(f"  {concept.upper()}: {len(generated)} naming items generated")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "bea_naming_benchmark.json")

    output = {
        "metadata": {
            "benchmark": "BEA_Naming",
            "clinical_analog": "Denominación por confrontación visual (BEA)",
            "description": (
                "Confrontation naming test: given a functional description, "
                "select the correct word among intra-category distractors. "
                "Unlike R1 (inter-category distractors like 'color' vs 'tool'), "
                "distractors here are ALL from the same semantic category, "
                "making the task harder and more clinically relevant."
            ),
            "format": "Compatible with evaluate_r1() in metrics.py",
            "total_items": len(all_items),
            "per_concept": {c: len(benchmark[c]) for c in benchmark},
            "distractor_type": "intra-category (same semantic field)",
            "references": [
                "Cuetos & González-Nosti (2009)",
                "Howard & Patterson (1992) Pyramids and Palm Trees",
                "Warrington & McCarthy (1987) Categories of knowledge",
            ],
        },
        "items": benchmark,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"Total items: {len(all_items)}")
    print("Done.")


if __name__ == "__main__":
    main()
