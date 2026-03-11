"""
BEA Synonym vs Antonym Discrimination Test (Pythia 2.8B)
============================================
Tests whether Pythia 2.8B can discriminate synonym pairs from antonym
pairs using embedding similarity, and whether concept ablation via CAV
projection disrupts this discrimination.

Simulates a clinical semantic discrimination task analogous to BEA
subtests for synonym/antonym judgment.

Metric: R2-Rank AUC — proportion of (synonym, antonym) pairs where
        cosine_similarity(synonym_pair) > cosine_similarity(antonym_pair).
        AUC > 0.5 means the model correctly assigns higher similarity
        to synonyms than antonyms.

Uses the same layers, methods, and alphas as CCT and caperucita tests.

Usage:
    python -m test.bea_synonym_test

Prerequisites:
    - CAVs extracted at layers [16, 24, 27] for all concepts
    - concepts/*.json enriched with curated_synonyms + curated_antonyms
"""

import torch
import json
import os
import sys
import random
from functools import partial
from datetime import datetime

# Ensure parent directory is in path
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import experiment_config as cfg
import concept_registry as cr
from experiment_runner import ablation_hook, load_cav
from metrics import evaluate_r2_ranking

# ---------------------------------------------------------------------------
# Configuration — all from experiment_config.py
# ---------------------------------------------------------------------------
CONCEPTS = cfg.CONCEPTS
CAV_METHODS = cfg.CAV_METHODS
SINGLE_LAYERS = cfg.EXTRACTION_LAYERS
ALL_LAYERS = cfg.EXPERIMENT_LAYERS_ALL
ALPHAS = cfg.INTENSITIES

RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, "bea_synonym_test_results.json")

# Minimum synonym/antonym pairs required to include a concept
MIN_SYNONYM_PAIRS = 5
MIN_ANTONYM_PAIRS = 5

# Off-target sampling for specificity
N_OFF_TARGET = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_single_layer_hooks(model, concept, cav_method, layer, alpha):
    """Build hook list for single-layer ablation."""
    cav = load_cav(concept, cav_method, layer)
    cav = cav.to(model.cfg.device)
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique="projection")
    return [(hook_name, hook_fn)]


def build_multi_layer_hooks(model, concept, cav_method, layers, alpha):
    """Build hook list for multi-layer ablation."""
    fwd_hooks = []
    for layer in layers:
        try:
            cav = load_cav(concept, cav_method, layer)
        except FileNotFoundError:
            print(f"    WARNING: CAV not found for {concept}/{cav_method}/L{layer}, skipping")
            continue
        cav = cav.to(model.cfg.device)
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = partial(ablation_hook, cav=cav, alpha=alpha, technique="projection")
        fwd_hooks.append((hook_name, hook_fn))
    return fwd_hooks


def wrap_words_as_sentences(pairs):
    """Wrap word pairs in simple template sentences for embedding extraction."""
    return [
        (f"The concept of {w1} is important.",
         f"The concept of {w2} is important.")
        for w1, w2 in pairs
    ]


def evaluate_with_hooks(model, syn_sentences, ant_sentences, fwd_hooks):
    """Run R2-Rank evaluation with optional ablation hooks."""
    if not fwd_hooks:
        return evaluate_r2_ranking(model, syn_sentences, ant_sentences)
    with model.hooks(fwd_hooks=fwd_hooks):
        return evaluate_r2_ranking(model, syn_sentences, ant_sentences)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BEA Synonym vs Antonym Discrimination Test (Pythia 2.8B)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load synonym and antonym data from concept registry
    all_synonyms = cr.get_curated_synonyms()
    all_antonyms = cr.get_curated_antonyms()

    # Include all concepts that have sufficient data
    all_concept_names = list(set(CONCEPTS) | set(all_synonyms.keys()))

    available = []
    for c in sorted(all_concept_names):
        syn = all_synonyms.get(c, [])
        ant = all_antonyms.get(c, [])
        if len(syn) >= MIN_SYNONYM_PAIRS and len(ant) >= MIN_ANTONYM_PAIRS:
            available.append(c)

    excluded = [c for c in sorted(all_concept_names)
                if c not in available
                and (all_synonyms.get(c, []) or all_antonyms.get(c, []))]
    if excluded:
        print(f"\n  Excluded {len(excluded)} concepts with insufficient data: {excluded}")

    assert available, "No concepts have sufficient synonym/antonym data!"
    print(f"\n  Concepts with data: {len(available)}")
    for c in available:
        print(f"    {c}: {len(all_synonyms.get(c, []))} syn, "
              f"{len(all_antonyms.get(c, []))} ant")

    # Prepare sentence-wrapped pairs
    syn_sentences = {}
    ant_sentences = {}
    for c in available:
        syn_sentences[c] = wrap_words_as_sentences(all_synonyms[c])
        ant_sentences[c] = wrap_words_as_sentences(all_antonyms[c])

    # Load model
    print(f"\nLoading model: {cfg.MODEL_NAME}...")
    model = cfg.load_model()

    # =================================================================
    # PHASE 1: BASELINE (no ablation)
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 1: BASELINE (healthy brain)")
    print(f"{'='*50}")

    baselines = {}
    for concept in available:
        result = evaluate_r2_ranking(
            model, syn_sentences[concept], ant_sentences[concept]
        )
        baselines[concept] = result
        print(f"  {concept.upper()}: AUC={result['r2_rank_auc']:.3f}  "
              f"syn={result['syn_mean']:.3f}  ant={result['nonsyn_mean']:.3f}  "
              f"gap={result['gap']:.3f}")

    # Filter: only keep concepts where baseline AUC > 0.5
    baseline_good = [c for c in available if baselines[c]["r2_rank_auc"] > 0.5]
    baseline_bad = [c for c in available if c not in baseline_good]
    if baseline_bad:
        print(f"\n  WARNING: {len(baseline_bad)} concepts with baseline AUC <= 0.5 "
              f"(model cannot discriminate): {baseline_bad}")
    available = baseline_good
    assert available, "No concepts have baseline AUC > 0.5!"

    # =================================================================
    # PHASE 2: FACTORIAL ABLATION
    # =================================================================
    layer_conditions = [(f"L{l}", [l]) for l in SINGLE_LAYERS]
    layer_conditions.append(("all_layers", ALL_LAYERS))

    # Only ablate concepts that have CAVs (in cfg.CONCEPTS)
    ablatable = [c for c in available if c in CONCEPTS]
    non_ablatable = [c for c in available if c not in CONCEPTS]
    if non_ablatable:
        print(f"\n  Non-ablatable (no CAVs): {non_ablatable}")
        print(f"  These will only be used as off-target measurement targets.")

    total_conditions = (
        len(ablatable) * len(CAV_METHODS) * len(layer_conditions) * len(ALPHAS)
    )
    print(f"\n{'='*50}")
    print("PHASE 2: FACTORIAL ABLATION")
    print(f"{'='*50}")
    print(f"  Ablatable concepts: {len(ablatable)}")
    print(f"  Total conditions: {total_conditions}")

    results = []
    condition_num = 0

    for ablated_concept in ablatable:
        for cav_method in CAV_METHODS:
            for layer_label, layers in layer_conditions:
                for alpha in ALPHAS:
                    condition_num += 1
                    print(f"\n  [{condition_num}/{total_conditions}] "
                          f"Ablate {ablated_concept}/{cav_method}/{layer_label}/α={alpha}")

                    if len(layers) == 1:
                        fwd_hooks = build_single_layer_hooks(
                            model, ablated_concept, cav_method, layers[0], alpha
                        )
                    else:
                        fwd_hooks = build_multi_layer_hooks(
                            model, ablated_concept, cav_method, layers, alpha
                        )

                    if not fwd_hooks:
                        print("    No hooks built, skipping")
                        continue

                    # ON-TARGET
                    on_result = evaluate_with_hooks(
                        model, syn_sentences[ablated_concept],
                        ant_sentences[ablated_concept], fwd_hooks
                    )
                    baseline_auc = baselines[ablated_concept]["r2_rank_auc"]
                    delta_auc = baseline_auc - on_result["r2_rank_auc"]

                    results.append({
                        "ablated_concept": ablated_concept,
                        "measured_concept": ablated_concept,
                        "on_target": True,
                        "cav_method": cav_method,
                        "layer": layer_label,
                        "alpha": alpha,
                        "auc": on_result["r2_rank_auc"],
                        "syn_mean": on_result["syn_mean"],
                        "ant_mean": on_result["nonsyn_mean"],
                        "gap": on_result["gap"],
                        "baseline_auc": baseline_auc,
                        "delta_auc": delta_auc,
                    })

                    print(f"    ON-TARGET  {ablated_concept}: "
                          f"AUC={on_result['r2_rank_auc']:.3f} "
                          f"(Δ={delta_auc:+.3f})")

                    # OFF-TARGET
                    off_candidates = [c for c in available if c != ablated_concept]
                    n_off = min(N_OFF_TARGET, len(off_candidates))
                    off_sample = random.sample(off_candidates, n_off) if n_off > 0 else []

                    for off_concept in off_sample:
                        off_result = evaluate_with_hooks(
                            model, syn_sentences[off_concept],
                            ant_sentences[off_concept], fwd_hooks
                        )
                        off_baseline = baselines[off_concept]["r2_rank_auc"]
                        off_delta = off_baseline - off_result["r2_rank_auc"]

                        results.append({
                            "ablated_concept": ablated_concept,
                            "measured_concept": off_concept,
                            "on_target": False,
                            "cav_method": cav_method,
                            "layer": layer_label,
                            "alpha": alpha,
                            "auc": off_result["r2_rank_auc"],
                            "syn_mean": off_result["syn_mean"],
                            "ant_mean": off_result["nonsyn_mean"],
                            "gap": off_result["gap"],
                            "baseline_auc": off_baseline,
                            "delta_auc": off_delta,
                        })

                        print(f"    OFF-TARGET {off_concept}: "
                              f"AUC={off_result['r2_rank_auc']:.3f} "
                              f"(Δ={off_delta:+.3f})")

    # =================================================================
    # PHASE 3: SPECIFICITY ANALYSIS
    # =================================================================
    print(f"\n{'='*50}")
    print("PHASE 3: SPECIFICITY ANALYSIS")
    print(f"{'='*50}")

    on_deltas = [r["delta_auc"] for r in results if r["on_target"]]
    off_deltas = [r["delta_auc"] for r in results if not r["on_target"]]

    import numpy as np
    mean_on = np.mean(on_deltas) if on_deltas else 0.0
    mean_off = np.mean(off_deltas) if off_deltas else 0.0
    ratio = mean_on / mean_off if mean_off > 0 else float("inf")

    print(f"  Mean on-target  Δ AUC: {mean_on:.4f}")
    print(f"  Mean off-target Δ AUC: {mean_off:.4f}")
    print(f"  Specificity ratio:     {ratio:.2f}")
    print(f"  {'PASS' if ratio > 2.0 else 'FAIL'}: "
          f"ratio {'>' if ratio > 2.0 else '<='} 2.0")

    specificity = {
        "mean_on_target_delta": float(mean_on),
        "mean_off_target_delta": float(mean_off),
        "specificity_ratio": float(ratio),
    }

    # =================================================================
    # Save results
    # =================================================================
    output = {
        "metadata": {
            "test": "BEA Synonym vs Antonym Discrimination",
            "model": cfg.MODEL_NAME,
            "layers_single": SINGLE_LAYERS,
            "layers_all": list(ALL_LAYERS),
            "alphas": ALPHAS,
            "cav_methods": CAV_METHODS,
            "concepts_evaluated": available,
            "concepts_ablated": ablatable,
            "min_synonym_pairs": MIN_SYNONYM_PAIRS,
            "min_antonym_pairs": MIN_ANTONYM_PAIRS,
            "n_off_target_sample": N_OFF_TARGET,
            "timestamp": datetime.now().isoformat(),
        },
        "baselines": {c: baselines[c] for c in available},
        "results": results,
        "specificity": specificity,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
