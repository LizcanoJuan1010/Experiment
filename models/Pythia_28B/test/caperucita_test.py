"""
Caperucita Test -- Wolf Concept Ablation in Little Red Riding Hood
==================================================================
Tests how Pythia 2.8B narrates the Little Red Riding Hood story
when the "wolf" concept is ablated via CAV intervention.

This simulates semantic aphasia for a specific concept: the model
should struggle to produce wolf-related language while narrating
a story where the wolf is a central character.

Experimental design:
  - Multiple story prompts that naturally require wolf-related concepts
  - Baseline (no ablation) vs ablated at multiple alphas (0-20)
  - Both single-layer and multi-layer intervention modes
  - Both subtraction and projection techniques
  - Analysis of circumlocution, wolf word density, and dose-response
  - Control prompts (non-wolf stories) to verify specificity

Usage:
    python caperucita_test.py

Output:
    results/caperucita_test_results.json
"""

import torch
import json
import os
import sys
import re
from functools import partial
from itertools import product
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import experiment_config as cfg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = cfg.MODEL_NAME
CAV_DIR = os.path.join(PROJECT_DIR, cfg.CAV_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, cfg.RESULTS_DIR)

ABLATED_CONCEPT = "wolf"

# Alpha sweep including extreme values for narrative breakdown analysis
ALPHAS = [0.0, 1.5, 3.5, 6.0, 10.0, 15.0, 20.0]

CAV_METHODS = ["mean_diff", "svm"]
LAYERS = cfg.EXTRACTION_LAYERS    # [16, 24, 27]
TECHNIQUES = ["subtraction", "projection"]

MAX_NEW_TOKENS = 150   # Longer generation for narrative continuations
TEMPERATURE = 0.0       # Greedy = deterministic comparisons

# ---------------------------------------------------------------------------
# Story Prompts -- Little Red Riding Hood
# ---------------------------------------------------------------------------
# Multiple prompts covering different story moments where the wolf concept
# is central. Organized by narrative phase.

STORY_PROMPTS = {
    # Phase 1: Forest encounter
    "forest_encounter": {
        "prompt": (
            "Once upon a time, Little Red Riding Hood walked through "
            "the dark forest. Suddenly, she met a"
        ),
        "expected_concept": "wolf",
        "story_phase": "encounter",
        "notes": "Model should naturally produce 'wolf' or 'big bad wolf'",
    },
    # Phase 2: The wolf's deception plan
    "wolf_plan": {
        "prompt": (
            "The big bad wolf had a cunning plan. He would go to "
            "grandmother's house and"
        ),
        "expected_concept": "wolf",
        "story_phase": "deception",
        "notes": "Wolf's actions are central -- ablation should disrupt planning",
    },
    # Phase 3: The disguise
    "grandmother_bed": {
        "prompt": (
            "Little Red Riding Hood arrived at grandmother's house. "
            "She noticed something strange about the figure in the bed. "
            "'What big eyes you have!' she said. The"
        ),
        "expected_concept": "wolf",
        "story_phase": "disguise",
        "notes": "Classic scene -- 'the wolf said' should be disrupted",
    },
    # Phase 4: The reveal
    "big_teeth": {
        "prompt": (
            "'What big teeth you have, grandmother!' said Little Red "
            "Riding Hood. 'All the better to"
        ),
        "expected_concept": "wolf",
        "story_phase": "reveal",
        "notes": "Wolf's signature line -- 'eat you with' said the wolf",
    },
    # Phase 5: The rescue
    "woodcutter": {
        "prompt": (
            "The woodcutter heard screams from the cottage. He rushed "
            "inside and saw the"
        ),
        "expected_concept": "wolf",
        "story_phase": "rescue",
        "notes": "Should produce 'wolf' attacking or having eaten grandmother",
    },
    # Phase 6: Full narration
    "full_story": {
        "prompt": (
            "Tell the story of Little Red Riding Hood. Once upon a "
            "time, there was a little girl who"
        ),
        "expected_concept": "wolf",
        "story_phase": "full_narrative",
        "notes": "Open-ended -- test if wolf appears organically in the story",
    },
    # Control 1: Non-wolf fairy tale
    "control_cinderella": {
        "prompt": (
            "Once upon a time, there was a girl named Cinderella who "
            "lived with her stepmother and"
        ),
        "expected_concept": None,
        "story_phase": "control",
        "notes": "Control: Cinderella has no wolf -- ablation should not affect",
    },
    # Control 2: Generic narrative
    "control_generic": {
        "prompt": (
            "The old man sat by the fireplace and told his grandchildren "
            "about the time he"
        ),
        "expected_concept": None,
        "story_phase": "control",
        "notes": "Control: Generic narrative -- ablation should not affect",
    },
}

# Wolf-related words for automated content analysis
WOLF_RELATED_WORDS = {
    "wolf", "wolves", "wolfish", "lupine",
    "howl", "howled", "howling", "howls",
    "growl", "growled", "growling", "growls",
    "snarl", "snarled", "snarling",
    "fangs", "teeth", "claws", "paws", "snout", "fur", "jaws",
    "predator", "beast", "creature", "monster",
    "devour", "devoured", "eaten", "ate", "swallow", "swallowed",
    "hunt", "hunted", "hunting", "prey", "stalk", "stalked",
    "den", "lair", "pack",
    "bite", "bitten", "bit",
}


# ---------------------------------------------------------------------------
# Hooks (same pattern as aphasia_test.py)
# ---------------------------------------------------------------------------
def subtraction_hook(resid_post, hook, cav, alpha):
    """Direct subtraction: resid -= alpha * cav."""
    resid_post -= alpha * cav
    return resid_post


def projection_hook(resid_post, hook, cav, alpha):
    """Project out the CAV component: resid -= alpha * (resid . cav) * cav."""
    dot = torch.einsum("bsd,d->bs", resid_post, cav)
    proj = torch.einsum("bs,d->bsd", dot, cav)
    resid_post -= alpha * proj
    return resid_post


HOOK_FNS = {
    "subtraction": subtraction_hook,
    "projection": projection_hook,
}


# ---------------------------------------------------------------------------
# Generation
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


def generate_with_ablation(model, prompt, cavs_by_layer, layers, technique, alpha):
    """Generate with ablation hooks on specified layers."""
    if alpha == 0.0:
        return generate_text(model, prompt)

    hook_fn_cls = HOOK_FNS[technique]
    hooks = []
    for layer in layers:
        cav = cavs_by_layer[layer]
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(hook_fn_cls, cav=cav, alpha=alpha)
        hooks.append((hook_name, hook_fn))

    with model.hooks(fwd_hooks=hooks):
        return generate_text(model, prompt)


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------
def analyze_wolf_content(text):
    """Analyze how much wolf-related content appears in generated text."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    wolf_words_found = words & WOLF_RELATED_WORDS
    total_words = len(text.split())
    return {
        "wolf_word_count": len(wolf_words_found),
        "wolf_words_found": sorted(wolf_words_found),
        "total_words": total_words,
        "wolf_density": round(len(wolf_words_found) / max(total_words, 1), 4),
        "contains_wolf": "wolf" in text.lower(),
        "contains_wolves": "wolves" in text.lower(),
    }


def compute_circumlocution_score(baseline_text, ablated_text):
    """
    Measure how much the model circumlocutes (talks around the concept).

    Uses Jaccard distance: 1 - |intersection| / |union|.
    Higher score = more different from baseline.
    """
    baseline_words = set(baseline_text.lower().split())
    ablated_words = set(ablated_text.lower().split())

    if not baseline_words or not ablated_words:
        return 0.0

    intersection = baseline_words & ablated_words
    union = baseline_words | ablated_words
    jaccard_distance = 1.0 - len(intersection) / len(union) if union else 0.0
    return round(jaccard_distance, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 70

    print(f"\n{sep}")
    print("  CAPERUCITA TEST -- Wolf Concept Ablation")
    print(f"  Little Red Riding Hood Narrative Analysis")
    print(f"  Concept: {ABLATED_CONCEPT}")
    print(f"  Methods: {CAV_METHODS}")
    print(f"  Layers:  {LAYERS}")
    print(f"  Techniques: {TECHNIQUES}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Max tokens: {MAX_NEW_TOKENS}")
    print(f"{sep}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = cfg.load_model()
    device = model.cfg.device

    # Load wolf CAVs: cavs[method][layer] = tensor
    print(f"\nLoading wolf CAVs...")
    cavs = {}
    for method in CAV_METHODS:
        cavs[method] = {}
        for layer in LAYERS:
            path = os.path.join(
                CAV_DIR, f"{ABLATED_CONCEPT}_{method}_layer{layer}.pt"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Wolf CAV not found: {path}\n"
                    f"Run 'python extract_wolf_cavs.py' first."
                )
            cavs[method][layer] = torch.load(
                path, weights_only=True
            ).to(device)
            print(f"  Loaded: {path}")

    # ===================================================================
    # PHASE 1: BASELINES (no ablation)
    # ===================================================================
    print(f"\n{sep}")
    print("  PHASE 1: BASELINES (no ablation)")
    print(f"{sep}")

    baselines = {}
    for prompt_id, prompt_info in STORY_PROMPTS.items():
        text = generate_text(model, prompt_info["prompt"])
        analysis = analyze_wolf_content(text)
        baselines[prompt_id] = {
            "prompt": prompt_info["prompt"],
            "text": text,
            "analysis": analysis,
            "story_phase": prompt_info["story_phase"],
        }
        wolf_tag = f" [wolf words: {analysis['wolf_word_count']}]"
        print(f"  {prompt_id}: done{wolf_tag}")
        print(f"    -> {text[:120]}...")

    # ===================================================================
    # PHASE 2: ABLATION SWEEP
    # ===================================================================
    print(f"\n{sep}")
    print("  PHASE 2: ABLATION SWEEP")
    print(f"{sep}")

    # Build layer configs: single layers + all combined
    layer_configs = []
    for l in LAYERS:
        layer_configs.append(([l], f"layer_{l}"))
    layer_configs.append(
        (LAYERS, f"layers_{'_'.join(str(l) for l in LAYERS)}")
    )

    total = (len(CAV_METHODS) * len(layer_configs) * len(TECHNIQUES)
             * len(ALPHAS) * len(STORY_PROMPTS))
    print(f"  Total conditions: {total}")

    all_results = []
    condition_num = 0

    for method, (layers, layer_label), technique, alpha in product(
        CAV_METHODS, layer_configs, TECHNIQUES, ALPHAS
    ):
        condition_label = f"{method}/{layer_label}/{technique}/alpha={alpha}"

        for prompt_id, prompt_info in STORY_PROMPTS.items():
            condition_num += 1
            if condition_num % 25 == 1:
                print(f"  [{condition_num}/{total}] {condition_label}")

            ablated_text = generate_with_ablation(
                model, prompt_info["prompt"],
                cavs[method], layers, technique, alpha,
            )

            baseline_text = baselines[prompt_id]["text"]
            analysis = analyze_wolf_content(ablated_text)
            circumlocution = compute_circumlocution_score(
                baseline_text, ablated_text
            )

            all_results.append({
                "prompt_id": prompt_id,
                "method": method,
                "layers": layers,
                "layer_label": layer_label,
                "technique": technique,
                "alpha": alpha,
                "story_phase": prompt_info["story_phase"],
                "expected_concept": prompt_info["expected_concept"],
                "prompt": prompt_info["prompt"],
                "baseline": baseline_text,
                "ablated": ablated_text,
                "changed": baseline_text != ablated_text,
                "wolf_analysis": analysis,
                "baseline_wolf_analysis": baselines[prompt_id]["analysis"],
                "circumlocution_score": circumlocution,
                "wolf_words_removed": (
                    baselines[prompt_id]["analysis"]["wolf_word_count"]
                    - analysis["wolf_word_count"]
                ),
            })

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    output = {
        "config": {
            "model": MODEL_NAME,
            "ablated_concept": ABLATED_CONCEPT,
            "methods": CAV_METHODS,
            "layers": LAYERS,
            "techniques": TECHNIQUES,
            "alphas": ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "total_conditions": len(all_results),
            "generated_at": datetime.now().isoformat(),
        },
        "story_prompts": {
            pid: {k: v for k, v in info.items()}
            for pid, info in STORY_PROMPTS.items()
        },
        "baselines": baselines,
        "results": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "caperucita_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ===================================================================
    # ANALYSIS SUMMARY
    # ===================================================================
    print(f"\n{sep}")
    print("  ANALYSIS: Wolf Concept Ablation in Narrative")
    print(f"{sep}")

    # --- Baseline wolf content ---
    print(f"\n  BASELINE Wolf Content:")
    for pid, bl in baselines.items():
        a = bl["analysis"]
        phase = bl["story_phase"]
        print(f"    {pid:25s} [{phase:15s}]: "
              f"{a['wolf_word_count']:2d} wolf words "
              f"{a['wolf_words_found']}")

    # --- Wolf story results only ---
    story_results = [
        r for r in all_results if r["expected_concept"] == "wolf"
    ]
    control_results = [
        r for r in all_results if r["expected_concept"] is None
    ]

    # --- Alpha dose-response for wolf content ---
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
            print(f"    alpha={alpha:5.1f}: "
                  f"wolf words={mean_wolf:.1f} (baseline={mean_baseline:.1f}), "
                  f"circumlocution={mean_circ:.2f}, "
                  f"changed={pct_changed:.0f}%")

    # --- Best ablation conditions ---
    if story_results:
        print(f"\n  TOP 10 ABLATION CONDITIONS (most wolf words removed):")
        non_zero = [r for r in story_results if r["alpha"] > 0]
        sorted_results = sorted(
            non_zero, key=lambda r: -r["wolf_words_removed"]
        )
        for i, r in enumerate(sorted_results[:10]):
            print(f"    {i+1}. {r['method']}/{r['layer_label']}/"
                  f"{r['technique']}/alpha={r['alpha']} "
                  f"[{r['prompt_id']}]: "
                  f"removed {r['wolf_words_removed']} wolf words, "
                  f"circumlocution={r['circumlocution_score']:.2f}")

    # --- Method comparison ---
    print(f"\n  METHOD COMPARISON (non-zero alpha, story prompts):")
    for method in CAV_METHODS:
        method_results = [
            r for r in story_results
            if r["method"] == method and r["alpha"] > 0
        ]
        if method_results:
            mean_removed = sum(
                r["wolf_words_removed"] for r in method_results
            ) / len(method_results)
            mean_circ = sum(
                r["circumlocution_score"] for r in method_results
            ) / len(method_results)
            print(f"    {method:10s}: mean wolf words removed={mean_removed:.2f}, "
                  f"circumlocution={mean_circ:.2f}")

    # --- Technique comparison ---
    print(f"\n  TECHNIQUE COMPARISON (non-zero alpha, story prompts):")
    for technique in TECHNIQUES:
        tech_results = [
            r for r in story_results
            if r["technique"] == technique and r["alpha"] > 0
        ]
        if tech_results:
            mean_removed = sum(
                r["wolf_words_removed"] for r in tech_results
            ) / len(tech_results)
            mean_circ = sum(
                r["circumlocution_score"] for r in tech_results
            ) / len(tech_results)
            print(f"    {technique:12s}: mean wolf words removed={mean_removed:.2f}, "
                  f"circumlocution={mean_circ:.2f}")

    # --- Control prompts: specificity check ---
    if control_results:
        changed_controls = sum(1 for r in control_results if r["changed"])
        total_controls = len(control_results)
        non_zero_controls = [r for r in control_results if r["alpha"] > 0]
        changed_nz = sum(1 for r in non_zero_controls if r["changed"])
        total_nz = len(non_zero_controls)
        print(f"\n  SPECIFICITY CHECK (control prompts):")
        print(f"    All conditions:      {changed_controls}/{total_controls} "
              f"changed ({100*changed_controls/max(total_controls,1):.0f}%)")
        print(f"    Non-zero alpha only: {changed_nz}/{total_nz} "
              f"changed ({100*changed_nz/max(total_nz,1):.0f}%)")

    # --- Example narrative comparisons ---
    print(f"\n{sep}")
    print("  EXAMPLE NARRATIVES: Best ablation per story prompt")
    print(f"{sep}")

    for pid in STORY_PROMPTS:
        if STORY_PROMPTS[pid]["expected_concept"] != "wolf":
            continue
        pid_results = [
            r for r in all_results
            if r["prompt_id"] == pid and r["alpha"] > 0
        ]
        if not pid_results:
            continue

        # Find the condition with most wolf words removed
        best = max(pid_results, key=lambda r: r["wolf_words_removed"])

        print(f"\n  [{pid}] Phase: {STORY_PROMPTS[pid]['story_phase']}")
        print(f"  Prompt: \"{best['prompt'][:80]}...\"")
        print(f"  Best condition: {best['method']}/{best['layer_label']}/"
              f"{best['technique']}/alpha={best['alpha']}")
        print(f"  BASELINE:  {best['baseline'][:200]}")
        print(f"  ABLATED:   {best['ablated'][:200]}")
        print(f"  Wolf words baseline: "
              f"{best['baseline_wolf_analysis']['wolf_words_found']}")
        print(f"  Wolf words ablated:  "
              f"{best['wolf_analysis']['wolf_words_found']}")
        print(f"  Circumlocution: {best['circumlocution_score']:.2f}")

    # --- Per-alpha narrative degradation example ---
    print(f"\n{sep}")
    print("  DOSE-RESPONSE NARRATIVE: forest_encounter prompt")
    print(f"{sep}")

    # Pick the best method/layer/technique for this analysis
    encounter_results = [
        r for r in all_results if r["prompt_id"] == "forest_encounter"
    ]
    if encounter_results:
        # Use mean_diff / all layers / subtraction as reference
        for alpha in ALPHAS:
            ref = [
                r for r in encounter_results
                if r["alpha"] == alpha
                and r["method"] == "mean_diff"
                and r["layer_label"].startswith("layers_")
                and r["technique"] == "subtraction"
            ]
            if ref:
                r = ref[0]
                wolf_count = r["wolf_analysis"]["wolf_word_count"] if alpha > 0 else r["baseline_wolf_analysis"]["wolf_word_count"]
                text = r["ablated"] if alpha > 0 else r["baseline"]
                print(f"  alpha={alpha:5.1f} [wolf words: {wolf_count}]: "
                      f"{text[:150]}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
