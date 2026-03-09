"""
Caperucita Test -- Wolf Concept Ablation in Little Red Riding Hood
==================================================================
Tests how GPT-2 small narrates the Little Red Riding Hood story
when the "wolf" concept is ablated via CAV projection.

This simulates semantic aphasia for a specific concept: the model
should struggle to produce wolf-related language while narrating
a story where the wolf is a central character.

Experimental design:
  - Two prompt styles: instruction (clinical paradigm) + completion
    (narrative context) to compare access vs storage deficits
  - Baseline (no ablation) vs ablated at multiple alphas
  - Both single-layer and multi-layer intervention modes
  - Control prompts (non-wolf stories) to verify specificity
  - Analysis of wolf words, circumlocution, ICU, and dose-response

Configuration matches the CCT test exactly:
  - Layers: [6, 9, 10] (individual) + all layers (0-11)
  - Alphas: [3.0, 6.0, 10.0, 15.0, 20.0]
  - Methods: mean_diff, svm
  - Technique: projection

Usage (inside Docker):
    docker run --gpus all -it tesis-gpt2 python -m test.caperucita_test

Output: results/caperucita_test_results.json
"""

import torch
import json
import os
import sys
import re
from functools import partial
from itertools import product
from datetime import datetime

# Ensure parent directory is in path
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
from experiment_runner import ablation_hook, load_cav

# ---------------------------------------------------------------------------
# Configuration — all from experiment_config.py (same as CCT test)
# ---------------------------------------------------------------------------
ABLATED_CONCEPT = "wolf"
CAV_METHODS = cfg.CAV_METHODS                # ["mean_diff", "svm"]
SINGLE_LAYERS = cfg.EXPERIMENT_LAYERS        # [6, 9, 10]
ALL_LAYERS = cfg.EXPERIMENT_LAYERS_ALL       # list(range(12))
ALPHAS = cfg.INTENSITIES                     # [3.0, 6.0, 10.0, 15.0, 20.0]
TECHNIQUE = "projection"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.0
RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, "caperucita_test_results.json")

# ---------------------------------------------------------------------------
# Story Prompts — Two styles: instruction + completion
# ---------------------------------------------------------------------------
# Instruction = clinical paradigm (patient is asked to narrate)
# Completion = narrative context (model continues the story)
# Both test the same ablation; the comparison reveals access vs storage

STORY_PROMPTS = {
    # ── INSTRUCTION (clinical paradigm) ──
    "instruction_full_story": {
        "prompt": (
            "Tell me the story of Little Red Riding Hood from "
            "beginning to end."
        ),
        "expected_concept": "wolf",
        "story_phase": "full_narrative",
        "prompt_style": "instruction",
    },
    "instruction_forest": {
        "prompt": (
            "Tell me what happened when Little Red Riding Hood "
            "walked into the forest."
        ),
        "expected_concept": "wolf",
        "story_phase": "encounter",
        "prompt_style": "instruction",
    },
    "instruction_grandmother": {
        "prompt": (
            "Tell me what happened when Little Red Riding Hood "
            "arrived at her grandmother's house."
        ),
        "expected_concept": "wolf",
        "story_phase": "deception",
        "prompt_style": "instruction",
    },
    # ── COMPLETION (narrative context) ──
    "completion_full_story": {
        "prompt": (
            "Once upon a time, there was a little girl called "
            "Little Red Riding Hood. She walked through the forest and"
        ),
        "expected_concept": "wolf",
        "story_phase": "full_narrative",
        "prompt_style": "completion",
    },
    "completion_forest": {
        "prompt": (
            "Little Red Riding Hood was walking through the dark "
            "forest when she met a"
        ),
        "expected_concept": "wolf",
        "story_phase": "encounter",
        "prompt_style": "completion",
    },
    "completion_grandmother": {
        "prompt": (
            "Little Red Riding Hood arrived at her grandmother's "
            "house. She knocked on the door and a voice said"
        ),
        "expected_concept": "wolf",
        "story_phase": "deception",
        "prompt_style": "completion",
    },
    # ── CONTROLS (no wolf) ──
    "control_cinderella": {
        "prompt": (
            "Tell me the story of Cinderella from beginning to end."
        ),
        "expected_concept": None,
        "story_phase": "control",
        "prompt_style": "instruction",
    },
    "control_generic": {
        "prompt": (
            "The old man sat by the fireplace and told his "
            "grandchildren about the time he"
        ),
        "expected_concept": None,
        "story_phase": "control",
        "prompt_style": "completion",
    },
}

# ---------------------------------------------------------------------------
# Wolf-related words organized by semantic domain
# ---------------------------------------------------------------------------
WOLF_WORDS_BY_DOMAIN = {
    "identity": {
        "wolf", "wolves", "wolfish", "lupine",
    },
    "vocalization": {
        "howl", "howled", "howling", "howls",
        "growl", "growled", "growling", "growls",
        "snarl", "snarled", "snarling",
    },
    "anatomy": {
        "fangs", "teeth", "claws", "paws", "snout", "fur", "jaws", "tail",
    },
    "predation": {
        "predator", "beast", "creature", "monster",
        "prey", "hunt", "hunted", "hunting", "stalk", "stalked",
    },
    "consumption": {
        "devour", "devoured", "eaten", "ate",
        "swallow", "swallowed", "bite", "bitten", "bit",
    },
    "habitat": {
        "den", "lair", "pack",
    },
}

# Flat set for fast lookup
WOLF_RELATED_WORDS = set()
for _domain_words in WOLF_WORDS_BY_DOMAIN.values():
    WOLF_RELATED_WORDS |= _domain_words

# ---------------------------------------------------------------------------
# Content Units (ICU) — clinical narrative assessment metric
# ---------------------------------------------------------------------------
CONTENT_UNITS = {
    "wolf_dependent": {
        "wolf_appears": ["wolf", "wolves", "beast", "creature", "predator"],
        "wolf_meets_girl": ["met", "encounter", "approach", "found"],
        "wolf_eats_grandmother": ["ate", "devour", "swallow", "eaten"],
        "wolf_disguise": ["disguise", "dress", "pretend", "bed", "nightgown"],
        "big_dialogue": ["big", "eyes", "ears", "teeth", "better"],
        "wolf_defeated": ["cut", "rescue", "save", "hunter", "woodcutter", "stones"],
    },
    "wolf_independent": {
        "girl_character": ["girl", "little", "red", "riding", "hood"],
        "grandmother": ["grandmother", "grandma", "granny"],
        "forest_setting": ["forest", "woods", "path", "trees"],
        "basket_food": ["basket", "cake", "food", "bread", "wine", "flowers"],
        "mother_instruction": ["mother", "stray", "path"],
        "happy_ending": ["happily", "ever", "after", "safe", "happy"],
    },
}


# ---------------------------------------------------------------------------
# Hook builders (same pattern as cct_test.py)
# ---------------------------------------------------------------------------
def build_single_layer_hooks(model, cav_method, layer, alpha):
    """Build hook list for single-layer ablation."""
    cav = load_cav(ABLATED_CONCEPT, cav_method, layer)
    cav = cav.to(model.cfg.device)
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique=TECHNIQUE)
    return [(hook_name, hook_fn)]


def build_multi_layer_hooks(model, cav_method, layers, alpha):
    """Build hook list for multi-layer ablation (all layers simultaneously)."""
    fwd_hooks = []
    for layer in layers:
        try:
            cav = load_cav(ABLATED_CONCEPT, cav_method, layer)
        except FileNotFoundError:
            print(f"    WARNING: CAV not found for {ABLATED_CONCEPT}/"
                  f"{cav_method}/L{layer}, skipping layer")
            continue
        cav = cav.to(model.cfg.device)
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique=TECHNIQUE)
        fwd_hooks.append((hook_name, hook_fn))
    return fwd_hooks


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------
def generate_text(model, prompt):
    """Generate text completion. Returns only the continuation."""
    input_ids = model.to_tokens(prompt)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
    full_text = model.to_string(output_ids[0])
    return full_text[len(prompt):].strip()


def generate_with_hooks(model, prompt, fwd_hooks):
    """Generate text with ablation hooks active."""
    if not fwd_hooks:
        return generate_text(model, prompt)
    with model.hooks(fwd_hooks=fwd_hooks):
        return generate_text(model, prompt)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyze_wolf_content(text):
    """Analyze wolf-related content in generated text."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    wolf_words_found = words & WOLF_RELATED_WORDS
    total_words = len(text.split())

    # Per-domain breakdown
    domain_counts = {}
    for domain, domain_words in WOLF_WORDS_BY_DOMAIN.items():
        found = words & domain_words
        domain_counts[domain] = {
            "count": len(found),
            "words": sorted(found),
        }

    return {
        "wolf_word_count": len(wolf_words_found),
        "wolf_words_found": sorted(wolf_words_found),
        "total_words": total_words,
        "wolf_density": round(len(wolf_words_found) / max(total_words, 1), 4),
        "contains_wolf": "wolf" in text.lower(),
        "contains_wolves": "wolves" in text.lower(),
        "domain_breakdown": domain_counts,
    }


def compute_circumlocution_score(baseline_text, ablated_text):
    """
    Measure circumlocution via Jaccard distance.
    Higher score = more different from baseline (talks around concepts).
    """
    baseline_words = set(baseline_text.lower().split())
    ablated_words = set(ablated_text.lower().split())

    if not baseline_words or not ablated_words:
        return 0.0

    intersection = baseline_words & ablated_words
    union = baseline_words | ablated_words
    jaccard_distance = 1.0 - len(intersection) / len(union) if union else 0.0
    return round(jaccard_distance, 4)


def compute_icu_score(text):
    """
    Information Content Unit analysis for Little Red Riding Hood.

    Returns counts of key narrative elements, split into
    wolf-dependent and wolf-independent units.
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))

    results = {}
    for domain_name, domain_units in CONTENT_UNITS.items():
        domain_results = {}
        for unit_name, keywords in domain_units.items():
            found = [w for w in keywords if w in words]
            domain_results[unit_name] = {
                "present": len(found) > 0,
                "keywords_found": found,
            }
        results[domain_name] = domain_results

    wolf_dep_count = sum(
        1 for u in results["wolf_dependent"].values() if u["present"]
    )
    wolf_indep_count = sum(
        1 for u in results["wolf_independent"].values() if u["present"]
    )
    wolf_dep_max = len(CONTENT_UNITS["wolf_dependent"])
    wolf_indep_max = len(CONTENT_UNITS["wolf_independent"])
    total = wolf_dep_count + wolf_indep_count
    max_total = wolf_dep_max + wolf_indep_max

    return {
        "total_icu": total,
        "max_icu": max_total,
        "icu_ratio": round(total / max(max_total, 1), 4),
        "wolf_dependent_icu": wolf_dep_count,
        "wolf_dependent_max": wolf_dep_max,
        "wolf_independent_icu": wolf_indep_count,
        "wolf_independent_max": wolf_indep_max,
        "unit_details": results,
    }


def compute_text_length_ratio(baseline_text, ablated_text):
    """Ratio of ablated text length to baseline text length."""
    bl = len(baseline_text.split())
    al = len(ablated_text.split())
    return round(al / max(bl, 1), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from transformer_lens import HookedTransformer

    sep = "=" * 70

    print(f"\n{sep}")
    print("  CAPERUCITA TEST -- Wolf Concept Ablation")
    print(f"  Little Red Riding Hood Narrative Analysis")
    print(f"{sep}")
    print(f"  Model:      {cfg.MODEL_NAME}")
    print(f"  Concept:    {ABLATED_CONCEPT}")
    print(f"  Methods:    {CAV_METHODS}")
    print(f"  Layers:     {SINGLE_LAYERS} + all_layers")
    print(f"  Alphas:     {ALPHAS}")
    print(f"  Technique:  {TECHNIQUE}")
    print(f"  Max tokens: {MAX_NEW_TOKENS}")
    print(f"  Prompts:    {len(STORY_PROMPTS)} "
          f"({sum(1 for p in STORY_PROMPTS.values() if p['expected_concept']=='wolf')} wolf, "
          f"{sum(1 for p in STORY_PROMPTS.values() if p['expected_concept'] is None)} control)")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    print(f"\nLoading {cfg.MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(cfg.MODEL_NAME)
    model.eval()

    # Build layer conditions (same pattern as CCT)
    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_layers", ALL_LAYERS))

    total_conditions = (
        len(CAV_METHODS) * len(layer_conditions)
        * len(ALPHAS) * len(STORY_PROMPTS)
    )
    print(f"  Total ablation conditions: {total_conditions}")

    # =================================================================
    # PHASE 1: BASELINES (no ablation)
    # =================================================================
    print(f"\n{sep}")
    print("  PHASE 1: BASELINES (no ablation)")
    print(f"{sep}")

    baselines = {}
    for prompt_id, prompt_info in STORY_PROMPTS.items():
        text = generate_text(model, prompt_info["prompt"])
        wolf_analysis = analyze_wolf_content(text)
        icu = compute_icu_score(text)
        baselines[prompt_id] = {
            "prompt": prompt_info["prompt"],
            "text": text,
            "wolf_analysis": wolf_analysis,
            "icu_score": icu,
            "story_phase": prompt_info["story_phase"],
            "prompt_style": prompt_info["prompt_style"],
        }
        wolf_tag = f"wolf words: {wolf_analysis['wolf_word_count']}"
        icu_tag = f"ICU: {icu['total_icu']}/{icu['max_icu']}"
        print(f"\n  {prompt_id} [{prompt_info['prompt_style']}]:")
        print(f"    {wolf_tag}, {icu_tag}")
        print(f"    -> {text[:150]}...")

    # =================================================================
    # PHASE 2: ABLATION SWEEP
    # =================================================================
    print(f"\n{sep}")
    print("  PHASE 2: ABLATION SWEEP")
    print(f"{sep}")

    all_results = []
    condition_num = 0

    for method, (layer_label, layers), alpha in product(
        CAV_METHODS, layer_conditions, ALPHAS
    ):
        # Build hooks once per (method, layers, alpha)
        if len(layers) == 1:
            fwd_hooks = build_single_layer_hooks(model, method, layers[0], alpha)
        else:
            fwd_hooks = build_multi_layer_hooks(model, method, layers, alpha)

        if not fwd_hooks:
            continue

        for prompt_id, prompt_info in STORY_PROMPTS.items():
            condition_num += 1
            if condition_num % 40 == 1:
                print(f"  [{condition_num}/{total_conditions}] "
                      f"{method}/{layer_label}/α={alpha}")

            ablated_text = generate_with_hooks(
                model, prompt_info["prompt"], fwd_hooks
            )
            baseline_text = baselines[prompt_id]["text"]

            wolf_analysis = analyze_wolf_content(ablated_text)
            circumlocution = compute_circumlocution_score(
                baseline_text, ablated_text
            )
            icu = compute_icu_score(ablated_text)
            length_ratio = compute_text_length_ratio(baseline_text, ablated_text)

            baseline_wolf = baselines[prompt_id]["wolf_analysis"]
            baseline_icu = baselines[prompt_id]["icu_score"]

            all_results.append({
                "prompt_id": prompt_id,
                "method": method,
                "layers": layers,
                "layer_label": layer_label,
                "alpha": alpha,
                "story_phase": prompt_info["story_phase"],
                "prompt_style": prompt_info["prompt_style"],
                "expected_concept": prompt_info["expected_concept"],
                "prompt": prompt_info["prompt"],
                "baseline_text": baseline_text,
                "ablated_text": ablated_text,
                "changed": baseline_text != ablated_text,
                "wolf_analysis": wolf_analysis,
                "baseline_wolf_analysis": baseline_wolf,
                "wolf_words_removed": (
                    baseline_wolf["wolf_word_count"]
                    - wolf_analysis["wolf_word_count"]
                ),
                "circumlocution_score": circumlocution,
                "icu_score": icu,
                "baseline_icu_score": baseline_icu,
                "icu_delta": {
                    "total": baseline_icu["total_icu"] - icu["total_icu"],
                    "wolf_dependent": (
                        baseline_icu["wolf_dependent_icu"]
                        - icu["wolf_dependent_icu"]
                    ),
                    "wolf_independent": (
                        baseline_icu["wolf_independent_icu"]
                        - icu["wolf_independent_icu"]
                    ),
                },
                "text_length_ratio": length_ratio,
            })

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    output = {
        "metadata": {
            "test": "Caperucita (Little Red Riding Hood Narrative Ablation)",
            "model": cfg.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "ablated_concept": ABLATED_CONCEPT,
            "layers_single": SINGLE_LAYERS,
            "layers_all": ALL_LAYERS,
            "alphas": ALPHAS,
            "cav_methods": CAV_METHODS,
            "technique": TECHNIQUE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "total_conditions": len(all_results),
            "config_source": "experiment_config.py (same as CCT test)",
        },
        "story_prompts": {
            pid: {k: v for k, v in info.items()}
            for pid, info in STORY_PROMPTS.items()
        },
        "content_units_definition": CONTENT_UNITS,
        "baselines": baselines,
        "results": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {OUTPUT_FILE}")

    # =================================================================
    # ANALYSIS SUMMARY
    # =================================================================
    print(f"\n{sep}")
    print("  ANALYSIS: Wolf Concept Ablation in Narrative")
    print(f"{sep}")

    # --- Baseline wolf content ---
    print(f"\n  BASELINE Wolf Content:")
    for pid, bl in baselines.items():
        a = bl["wolf_analysis"]
        icu = bl["icu_score"]
        style = bl["prompt_style"]
        print(f"    {pid:30s} [{style:11s}]: "
              f"{a['wolf_word_count']:2d} wolf words, "
              f"ICU={icu['total_icu']}/{icu['max_icu']} "
              f"{a['wolf_words_found']}")

    # --- Separate story vs control ---
    story_results = [
        r for r in all_results if r["expected_concept"] == "wolf"
    ]
    control_results = [
        r for r in all_results if r["expected_concept"] is None
    ]

    # --- Prompt style comparison ---
    print(f"\n  PROMPT STYLE COMPARISON (baselines):")
    for style in ["instruction", "completion"]:
        style_baselines = [
            bl for bl in baselines.values()
            if bl["prompt_style"] == style and bl["story_phase"] != "control"
        ]
        if style_baselines:
            mean_wolf = sum(
                bl["wolf_analysis"]["wolf_word_count"] for bl in style_baselines
            ) / len(style_baselines)
            mean_icu = sum(
                bl["icu_score"]["total_icu"] for bl in style_baselines
            ) / len(style_baselines)
            print(f"    {style:12s}: mean wolf words={mean_wolf:.1f}, "
                  f"mean ICU={mean_icu:.1f}")

    # --- Dose-response ---
    print(f"\n  DOSE-RESPONSE: Mean wolf words by alpha (story prompts):")
    for alpha in ALPHAS:
        alpha_results = [r for r in story_results if r["alpha"] == alpha]
        if alpha_results:
            mean_wolf = sum(
                r["wolf_analysis"]["wolf_word_count"] for r in alpha_results
            ) / len(alpha_results)
            mean_baseline = sum(
                r["baseline_wolf_analysis"]["wolf_word_count"]
                for r in alpha_results
            ) / len(alpha_results)
            mean_circ = sum(
                r["circumlocution_score"] for r in alpha_results
            ) / len(alpha_results)
            pct_changed = sum(
                1 for r in alpha_results if r["changed"]
            ) / len(alpha_results) * 100
            print(f"    α={alpha:5.1f}: "
                  f"wolf words={mean_wolf:.1f} (baseline={mean_baseline:.1f}), "
                  f"circumlocution={mean_circ:.2f}, "
                  f"changed={pct_changed:.0f}%")

    # --- Dose-response by prompt style ---
    print(f"\n  DOSE-RESPONSE by prompt style:")
    for style in ["instruction", "completion"]:
        style_results = [
            r for r in story_results if r["prompt_style"] == style
        ]
        if style_results:
            print(f"    [{style}]:")
            for alpha in ALPHAS:
                ar = [r for r in style_results if r["alpha"] == alpha]
                if ar:
                    mw = sum(
                        r["wolf_analysis"]["wolf_word_count"] for r in ar
                    ) / len(ar)
                    print(f"      α={alpha:5.1f}: wolf words={mw:.1f}")

    # --- Method comparison ---
    print(f"\n  METHOD COMPARISON (story prompts):")
    for method in CAV_METHODS:
        method_results = [r for r in story_results if r["method"] == method]
        if method_results:
            mean_removed = sum(
                r["wolf_words_removed"] for r in method_results
            ) / len(method_results)
            mean_circ = sum(
                r["circumlocution_score"] for r in method_results
            ) / len(method_results)
            print(f"    {method:10s}: mean wolf words removed={mean_removed:.2f}, "
                  f"circumlocution={mean_circ:.2f}")

    # --- Top ablation conditions ---
    if story_results:
        print(f"\n  TOP 10 ABLATION CONDITIONS (most wolf words removed):")
        sorted_results = sorted(
            story_results, key=lambda r: -r["wolf_words_removed"]
        )
        for i, r in enumerate(sorted_results[:10]):
            print(f"    {i+1}. {r['method']}/{r['layer_label']}/"
                  f"α={r['alpha']} [{r['prompt_id']}]: "
                  f"removed {r['wolf_words_removed']} wolf words, "
                  f"circumlocution={r['circumlocution_score']:.2f}")

    # --- Specificity check ---
    if control_results:
        changed_controls = sum(1 for r in control_results if r["changed"])
        total_controls = len(control_results)
        print(f"\n  SPECIFICITY CHECK (control prompts):")
        print(f"    Changed: {changed_controls}/{total_controls} "
              f"({100*changed_controls/max(total_controls,1):.0f}%)")

    # --- ICU analysis ---
    print(f"\n  ICU ANALYSIS (highest alpha, all_layers):")
    max_alpha = max(ALPHAS)
    for style in ["instruction", "completion"]:
        icu_results = [
            r for r in story_results
            if r["alpha"] == max_alpha
            and r["layer_label"] == "all_layers"
            and r["prompt_style"] == style
        ]
        if icu_results:
            mean_dep_delta = sum(
                r["icu_delta"]["wolf_dependent"] for r in icu_results
            ) / len(icu_results)
            mean_indep_delta = sum(
                r["icu_delta"]["wolf_independent"] for r in icu_results
            ) / len(icu_results)
            print(f"    [{style}] α={max_alpha}: "
                  f"wolf-dependent ICU Δ={mean_dep_delta:+.1f}, "
                  f"wolf-independent ICU Δ={mean_indep_delta:+.1f}")

    # --- Example narratives ---
    print(f"\n{sep}")
    print("  EXAMPLE NARRATIVES: Best ablation per story prompt")
    print(f"{sep}")

    for pid in STORY_PROMPTS:
        if STORY_PROMPTS[pid]["expected_concept"] != "wolf":
            continue
        pid_results = [r for r in all_results if r["prompt_id"] == pid]
        if not pid_results:
            continue

        best = max(pid_results, key=lambda r: r["wolf_words_removed"])

        print(f"\n  [{pid}] Phase: {STORY_PROMPTS[pid]['story_phase']} "
              f"| Style: {STORY_PROMPTS[pid]['prompt_style']}")
        print(f"  Prompt: \"{best['prompt'][:80]}...\"")
        print(f"  Best: {best['method']}/{best['layer_label']}/α={best['alpha']}")
        print(f"  BASELINE:  {best['baseline_text'][:200]}")
        print(f"  ABLATED:   {best['ablated_text'][:200]}")
        print(f"  Wolf words baseline: "
              f"{best['baseline_wolf_analysis']['wolf_words_found']}")
        print(f"  Wolf words ablated:  "
              f"{best['wolf_analysis']['wolf_words_found']}")
        print(f"  Circumlocution: {best['circumlocution_score']:.2f}")

    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
