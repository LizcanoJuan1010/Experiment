"""
Comprehensive Sweep Experiment — Golden Gate Claude Style
=========================================================
Systematic sweep over vector type, alpha, layer configuration, and concept
to produce dose-response curves and identify sweet spots for concept ablation.

Design:
  2 vector types (SVM, mean-diff)
  x 9 alphas (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)
  x N layer configs (32 singles + 8 pairs + 5 triples + 3 multi)
  x 3 concepts (time, place, tools)
  (Total conditions depend on N_LAYERS from config)

Cross-concept specificity: each condition measures R1/R2 for ALL 3 concepts.

References:
  - Anthropic (2024) — Golden Gate Claude: steering factors -20 to +20
  - Turner et al. (2023) — CAA: sweep all layers, coefficients 3-15
  - Zou et al. (2023) — RepE: mean-centering, features in superposition
  - Kim et al. (2018) — TCAV: SVM-based CAV extraction

Usage:
    python experiment_sweep.py

Input:  experiment_data.json, cavs_steering/*.pt, cavs_steering_meandiff/*.pt
Output: results_sweep/results_sweep.csv
        results_sweep/baselines.json
        results_sweep/sweep_report.json
        results_sweep/figures/F01-F08
"""

import torch
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from functools import partial

import experiment_config as cfg
from metrics import evaluate_r1, evaluate_r2


# =========================================================================
# Constants
# =========================================================================
CONCEPTS = ["time", "place", "tools"]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
VECTOR_TYPES = ["svm", "meandiff"]

VECTOR_DIRS = {
    "svm": "cavs_steering",
    "meandiff": "cavs_steering_meandiff",
}

N_LAYERS = cfg.N_LAYERS_TOTAL  # Pythia 2.8B: 32 layers

LAYER_CONFIGS = OrderedDict()
# Singles (32)
for i in range(N_LAYERS):
    LAYER_CONFIGS[f"single_L{i}"] = [i]
# Pairs (8) — evenly spaced across depth
for i in range(0, N_LAYERS - 1, 4):
    LAYER_CONFIGS[f"pair_{i}_{i+1}"] = [i, i + 1]
# Triples (5) — evenly spaced across depth
for i in range(0, N_LAYERS - 2, 6):
    LAYER_CONFIGS[f"triple_{i}_{i+2}"] = list(range(i, i + 3))
# Multi (3)
LAYER_CONFIGS["half_early"] = list(range(0, N_LAYERS // 2))
LAYER_CONFIGS["half_late"] = list(range(N_LAYERS // 2, N_LAYERS))
LAYER_CONFIGS["all"] = list(range(N_LAYERS))

RESULTS_DIR = "results_sweep"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CSV_PATH = os.path.join(RESULTS_DIR, "results_sweep.csv")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baselines.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "sweep_report.json")
CHECKPOINT_INTERVAL = 100

DATA_FILE = cfg.DATA_FILE
MODEL_NAME = cfg.MODEL_NAME


# =========================================================================
# Load Steering Vectors
# =========================================================================
def load_all_vectors(device):
    """Load all pre-extracted steering vectors into memory."""
    vectors = {}
    for vtype, vdir in VECTOR_DIRS.items():
        vectors[vtype] = {}
        for concept in CONCEPTS:
            vectors[vtype][concept] = {}
            for layer in range(N_LAYERS):
                path = os.path.join(vdir, f"{concept}_steer_layer{layer}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing vector: {path}")
                vectors[vtype][concept][layer] = torch.load(
                    path, map_location=device, weights_only=True
                )
    return vectors


# =========================================================================
# Ablation Hook
# =========================================================================
def ablation_hook(resid_post, hook, steering_vec, alpha, direction):
    """
    Feature steering: add or subtract scaled vector from residual stream.

    direction=-1: ablation (subtract concept)
    direction=+1: amplification (add concept, control condition)
    """
    return resid_post + direction * alpha * steering_vec


# =========================================================================
# Run One Condition (with cross-concept measurement)
# =========================================================================
def run_condition(model, vectors, vector_type, concept, layers, alpha,
                  direction, all_r1_data, all_r2_data, device):
    """
    Run one experimental condition and measure R1/R2 for ALL concepts.

    Returns dict with R1/R2 for target + other concepts.
    """
    fwd_hooks = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        sv = vectors[vector_type][concept][layer].to(device)
        hook_fn = partial(
            ablation_hook,
            steering_vec=sv,
            alpha=alpha,
            direction=direction,
        )
        fwd_hooks.append((hook_name, hook_fn))

    results = {}
    with model.hooks(fwd_hooks=fwd_hooks):
        for c in CONCEPTS:
            r1_acc, _ = evaluate_r1(model, all_r1_data[c])
            r2_sim, _ = evaluate_r2(model, all_r2_data[c])
            results[c] = {"r1": r1_acc, "r2": r2_sim}

    return results


# =========================================================================
# Build Condition List
# =========================================================================
def build_conditions():
    """Build the full list of experimental conditions."""
    conditions = []

    # Ablation conditions: all combinations
    for vtype in VECTOR_TYPES:
        for alpha in ALPHAS:
            for lconfig_name, layers in LAYER_CONFIGS.items():
                for concept in CONCEPTS:
                    conditions.append({
                        "vector_type": vtype,
                        "alpha": alpha,
                        "layer_config": lconfig_name,
                        "layers": layers,
                        "concept": concept,
                        "direction": -1,
                        "direction_label": "ablation",
                    })

    # Amplification controls: only multi_all config
    for vtype in VECTOR_TYPES:
        for alpha in ALPHAS:
            for concept in CONCEPTS:
                conditions.append({
                    "vector_type": vtype,
                    "alpha": alpha,
                    "layer_config": "all",
                    "layers": list(range(N_LAYERS)),
                    "concept": concept,
                    "direction": +1,
                    "direction_label": "amplification",
                })

    return conditions


# =========================================================================
# Sweet Spot Identification
# =========================================================================
def find_sweet_spots(df, baselines):
    """Identify optimal ablation conditions per concept."""
    df_abl = df[df["direction_label"] == "ablation"].copy()
    sweet_spots = {}

    for concept in CONCEPTS:
        concept_df = df_abl[df_abl["concept"] == concept].copy()
        others = [c for c in CONCEPTS if c != concept]

        # Compute cross-concept damage
        cross_cols = [f"delta_r1_{c}" for c in others]
        concept_df["cross_damage"] = concept_df[cross_cols].abs().mean(axis=1)

        # Filter: minimum target effect
        candidates = concept_df[concept_df["delta_r1_target"] > 0.05]

        # Filter: maximum cross-concept damage
        strict = candidates[candidates["cross_damage"] < 0.10]
        if len(strict) > 0:
            best = strict.loc[strict["delta_r1_target"].idxmax()]
            threshold_used = 0.10
        else:
            relaxed = candidates[candidates["cross_damage"] < 0.15]
            if len(relaxed) > 0:
                best = relaxed.loc[relaxed["delta_r1_target"].idxmax()]
                threshold_used = 0.15
            else:
                if len(candidates) > 0:
                    best = candidates.loc[candidates["delta_r1_target"].idxmax()]
                    threshold_used = None
                else:
                    sweet_spots[concept] = {
                        "found": False,
                        "note": "No conditions with delta_r1_target > 0.05",
                    }
                    continue

        sweet_spots[concept] = {
            "found": True,
            "vector_type": best["vector_type"],
            "alpha": float(best["alpha"]),
            "layer_config": best["layer_config"],
            "delta_r1_target": float(best["delta_r1_target"]),
            "cross_damage": float(best["cross_damage"]),
            "cross_threshold": threshold_used,
            "selectivity": (
                float(best["delta_r1_target"]) /
                (float(best["delta_r1_target"]) + float(best["cross_damage"]))
                if float(best["cross_damage"]) > 0 else 1.0
            ),
        }

    return sweet_spots


# =========================================================================
# Figure Generation
# =========================================================================
def generate_figures(df, baselines):
    """Generate 8 analysis figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_abl = df[df["direction_label"] == "ablation"].copy()
    df_amp = df[df["direction_label"] == "amplification"].copy()

    # --- F01: Heatmap ΔR1 by (layer_config x alpha) per concept/vtype ---
    print("  Generating F01: Heatmaps...")
    for vtype in VECTOR_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(
            f"F01: Delta R1 Heatmap — {vtype.upper()} vectors",
            fontsize=14, fontweight="bold",
        )
        for idx, concept in enumerate(CONCEPTS):
            mask = (
                (df_abl["vector_type"] == vtype) &
                (df_abl["concept"] == concept)
            )
            subset = df_abl[mask].copy()
            pivot = subset.pivot_table(
                values="delta_r1_target",
                index="layer_config",
                columns="alpha",
                aggfunc="first",
            )
            # Order rows by LAYER_CONFIGS order
            ordered_rows = [r for r in LAYER_CONFIGS.keys() if r in pivot.index]
            pivot = pivot.reindex(ordered_rows)

            ax = axes[idx]
            im = ax.imshow(
                pivot.values, aspect="auto", cmap="RdYlGn",
                vmin=-0.2, vmax=0.5,
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{a}" for a in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Layer Config")
            ax.set_title(f"{concept.upper()}")
            fig.colorbar(im, ax=ax, shrink=0.6, label="ΔR1")

        plt.tight_layout()
        fig.savefig(
            os.path.join(FIGURES_DIR, f"F01_heatmap_{vtype}.png"), dpi=150
        )
        plt.close(fig)

    # --- F02: Dose-response curves for best layer configs ---
    print("  Generating F02: Dose-response (best configs)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "F02: Dose-Response — Best Layer Configs per Concept",
        fontsize=14, fontweight="bold",
    )
    for idx, concept in enumerate(CONCEPTS):
        ax = axes[idx]
        mask = df_abl["concept"] == concept
        subset = df_abl[mask]
        # Find top-3 layer configs by max ΔR1
        top_configs = (
            subset.groupby(["vector_type", "layer_config"])["delta_r1_target"]
            .max().nlargest(5).index
        )
        for vtype, lconfig in top_configs:
            curve = subset[
                (subset["vector_type"] == vtype) &
                (subset["layer_config"] == lconfig)
            ].sort_values("alpha")
            label = f"{vtype}/{lconfig}"
            ax.plot(curve["alpha"], curve["delta_r1_target"],
                    "o-", label=label, markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("ΔR1")
        ax.set_title(f"{concept.upper()}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "F02_dose_response_best.png"), dpi=150)
    plt.close(fig)

    # --- F03: Dose-response by grouping (singles vs pairs vs triples vs multi) ---
    print("  Generating F03: Dose-response by layer group...")
    def classify_config(name):
        if name.startswith("single"):
            return "single"
        elif name.startswith("pair"):
            return "pair"
        elif name.startswith("triple"):
            return "triple"
        else:
            return "multi"

    df_abl["config_group"] = df_abl["layer_config"].apply(classify_config)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "F03: Dose-Response — Singles vs Pairs vs Triples vs Multi",
        fontsize=14, fontweight="bold",
    )
    for idx, concept in enumerate(CONCEPTS):
        ax = axes[idx]
        mask = (df_abl["concept"] == concept) & (df_abl["vector_type"] == "svm")
        subset = df_abl[mask]
        for group in ["single", "pair", "triple", "multi"]:
            group_data = subset[subset["config_group"] == group]
            curve = group_data.groupby("alpha")["delta_r1_target"].mean()
            ax.plot(curve.index, curve.values, "o-", label=group, markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Mean ΔR1")
        ax.set_title(f"{concept.upper()} (SVM)")
        ax.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "F03_dose_response_groups.png"), dpi=150
    )
    plt.close(fig)

    # --- F04: SVM vs Mean-diff comparison ---
    print("  Generating F04: SVM vs Mean-diff...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "F04: Top-5 Conditions — SVM vs Mean-Diff",
        fontsize=14, fontweight="bold",
    )
    for vtype_idx, vtype in enumerate(VECTOR_TYPES):
        top5 = (
            df_abl[df_abl["vector_type"] == vtype]
            .nlargest(5, "delta_r1_target")
        )
        labels = [
            f"{r['concept']}/{r['layer_config']}/α={r['alpha']}"
            for _, r in top5.iterrows()
        ]
        x = np.arange(5) + vtype_idx * 0.35
        ax.bar(x, top5["delta_r1_target"], width=0.3, label=vtype.upper())
        for i, lbl in enumerate(labels):
            ax.text(x[i], top5.iloc[i]["delta_r1_target"] + 0.01, lbl,
                    rotation=45, fontsize=6, ha="left")
    ax.set_ylabel("ΔR1")
    ax.set_xlabel("Rank")
    ax.set_xticks(np.arange(5) + 0.175)
    ax.set_xticklabels([f"#{i+1}" for i in range(5)])
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "F04_svm_vs_meandiff.png"), dpi=150)
    plt.close(fig)

    # --- F05: Specificity scatter ---
    print("  Generating F05: Specificity scatter...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "F05: Specificity — ΔR1 Target vs Cross-Concept Damage",
        fontsize=14, fontweight="bold",
    )
    for idx, concept in enumerate(CONCEPTS):
        ax = axes[idx]
        mask = df_abl["concept"] == concept
        subset = df_abl[mask].copy()
        others = [c for c in CONCEPTS if c != concept]
        cross_cols = [f"delta_r1_{c}" for c in others]
        subset["cross_damage"] = subset[cross_cols].abs().mean(axis=1)

        for vtype in VECTOR_TYPES:
            vt_data = subset[subset["vector_type"] == vtype]
            ax.scatter(
                vt_data["delta_r1_target"], vt_data["cross_damage"],
                alpha=0.4, s=15, label=vtype.upper(),
            )
        ax.axhline(y=0.10, color="red", linestyle="--", alpha=0.5,
                    label="Cross threshold")
        ax.set_xlabel("ΔR1 Target (higher = better ablation)")
        ax.set_ylabel("Cross Damage (lower = more specific)")
        ax.set_title(f"Ablating {concept.upper()}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "F05_specificity.png"), dpi=150)
    plt.close(fig)

    # --- F06: Single-layer heatmap (12 layers x 9 alphas) ---
    print("  Generating F06: Single-layer heatmap...")
    singles = df_abl[df_abl["layer_config"].str.startswith("single")]
    for vtype in VECTOR_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"F06: Single-Layer ΔR1 — {vtype.upper()}",
            fontsize=14, fontweight="bold",
        )
        for idx, concept in enumerate(CONCEPTS):
            mask = (
                (singles["vector_type"] == vtype) &
                (singles["concept"] == concept)
            )
            subset = singles[mask].copy()
            subset["layer_num"] = subset["layer_config"].str.extract(
                r"single_L(\d+)"
            ).astype(int)
            pivot = subset.pivot_table(
                values="delta_r1_target",
                index="layer_num",
                columns="alpha",
                aggfunc="first",
            )
            ax = axes[idx]
            im = ax.imshow(
                pivot.values, aspect="auto", cmap="RdYlGn",
                vmin=-0.1, vmax=0.3,
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{a}" for a in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"L{l}" for l in pivot.index], fontsize=8)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Layer")
            ax.set_title(f"{concept.upper()}")
            fig.colorbar(im, ax=ax, shrink=0.6, label="ΔR1")
        plt.tight_layout()
        fig.savefig(
            os.path.join(FIGURES_DIR, f"F06_single_layer_{vtype}.png"), dpi=150
        )
        plt.close(fig)

    # --- F07: Amplification vs ablation curves ---
    print("  Generating F07: Amplification vs ablation...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "F07: Ablation vs Amplification (multi_all config)",
        fontsize=14, fontweight="bold",
    )
    for idx, concept in enumerate(CONCEPTS):
        ax = axes[idx]
        for vtype in VECTOR_TYPES:
            # Ablation curve (multi_all)
            abl_mask = (
                (df_abl["concept"] == concept) &
                (df_abl["vector_type"] == vtype) &
                (df_abl["layer_config"] == "all")
            )
            abl_curve = df_abl[abl_mask].sort_values("alpha")

            # Amplification curve
            amp_mask = (
                (df_amp["concept"] == concept) &
                (df_amp["vector_type"] == vtype)
            )
            amp_curve = df_amp[amp_mask].sort_values("alpha")

            ax.plot(abl_curve["alpha"], abl_curve["delta_r1_target"],
                    "o-", label=f"{vtype}/ablation", markersize=4)
            ax.plot(amp_curve["alpha"], amp_curve["delta_r1_target"],
                    "s--", label=f"{vtype}/amplification", markersize=4)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("ΔR1")
        ax.set_title(f"{concept.upper()}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "F07_ablation_vs_amplification.png"), dpi=150
    )
    plt.close(fig)

    # --- F08: Sweet spot summary ---
    print("  Generating F08: Sweet spots...")
    sweet_spots = find_sweet_spots(df, baselines)
    found = {c: ss for c, ss in sweet_spots.items() if ss.get("found", False)}

    if found:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            "F08: Sweet Spots — Target ΔR1 vs Cross-Concept Damage",
            fontsize=14, fontweight="bold",
        )
        concepts_found = list(found.keys())
        x = np.arange(len(concepts_found))
        target_vals = [found[c]["delta_r1_target"] for c in concepts_found]
        cross_vals = [found[c]["cross_damage"] for c in concepts_found]

        ax.bar(x - 0.15, target_vals, 0.3, label="ΔR1 Target", color="green")
        ax.bar(x + 0.15, cross_vals, 0.3, label="Cross Damage", color="red",
               alpha=0.7)

        for i, c in enumerate(concepts_found):
            ss = found[c]
            ax.text(
                i, max(target_vals[i], cross_vals[i]) + 0.02,
                f"{ss['vector_type']}/{ss['layer_config']}\nα={ss['alpha']}",
                ha="center", fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in concepts_found])
        ax.set_ylabel("ΔR1")
        ax.legend()
        ax.axhline(y=0.10, color="red", linestyle=":", alpha=0.5,
                    label="Cross threshold")
        plt.tight_layout()
        fig.savefig(
            os.path.join(FIGURES_DIR, "F08_sweet_spots.png"), dpi=150
        )
        plt.close(fig)

    return sweet_spots


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("COMPREHENSIVE SWEEP EXPERIMENT — Golden Gate Claude Style")
    print("=" * 70)

    conditions = build_conditions()
    n_ablation = sum(1 for c in conditions if c["direction"] == -1)
    n_amplification = sum(1 for c in conditions if c["direction"] == +1)
    print(f"  Vector types:   {VECTOR_TYPES}")
    print(f"  Alphas:         {ALPHAS}")
    n_singles = sum(1 for k in LAYER_CONFIGS if k.startswith("single"))
    n_pairs = sum(1 for k in LAYER_CONFIGS if k.startswith("pair"))
    n_triples = sum(1 for k in LAYER_CONFIGS if k.startswith("triple"))
    n_multi = len(LAYER_CONFIGS) - n_singles - n_pairs - n_triples
    print(f"  Layer configs:  {len(LAYER_CONFIGS)} "
          f"({n_singles} singles + {n_pairs} pairs + {n_triples} triples + {n_multi} multi)")
    print(f"  Concepts:       {CONCEPTS}")
    print(f"  Ablation:       {n_ablation}")
    print(f"  Amplification:  {n_amplification}")
    print(f"  TOTAL:          {len(conditions)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    model = cfg.load_model()
    device = model.cfg.device

    # Load all vectors into memory
    print("Loading steering vectors...")
    vectors = load_all_vectors(device)
    n_vecs = sum(
        len(vectors[vt][c]) for vt in vectors for c in vectors[vt]
    )
    print(f"  Loaded {n_vecs} vectors into memory")

    # Prepare benchmark data
    all_r1_data = {c: data["r1_benchmark"][c] for c in CONCEPTS}
    all_r2_data = {c: data["r2_benchmark"][c] for c in CONCEPTS}

    # ===== PHASE 1: BASELINES =====
    print(f"\n{'='*60}")
    print("PHASE 1: BASELINES (no intervention)")
    print(f"{'='*60}")

    baselines = {}
    for concept in CONCEPTS:
        r1_acc, _ = evaluate_r1(model, all_r1_data[concept])
        r2_sim, _ = evaluate_r2(model, all_r2_data[concept])
        baselines[concept] = {"r1": r1_acc, "r2": r2_sim}
        print(f"  {concept.upper()}: R1={r1_acc:.3f}, R2={r2_sim:.4f}")

    with open(BASELINE_PATH, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Saved: {BASELINE_PATH}")

    # ===== PHASE 2: SWEEP =====
    print(f"\n{'='*60}")
    print("PHASE 2: SWEEP ({} conditions)".format(len(conditions)))
    print(f"{'='*60}")

    # Check for checkpoint
    results_rows = []
    start_idx = 0
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        results_rows = existing.to_dict("records")
        start_idx = len(results_rows)
        print(f"  Resuming from checkpoint: {start_idx} conditions done")

    t_start = time.time()

    for i in range(start_idx, len(conditions)):
        cond = conditions[i]
        elapsed = time.time() - t_start
        done = i - start_idx
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        remaining = (len(conditions) - i) / rate if rate > 0 else 0

        label = (
            f"[{i+1}/{len(conditions)}] "
            f"{cond['vector_type']}/{cond['concept']}/"
            f"{cond['layer_config']}/α={cond['alpha']}/"
            f"{cond['direction_label']}"
        )
        if done > 0 and done % 10 == 0:
            eta_min = remaining / 60
            print(f"  {label}  (ETA: {eta_min:.0f} min)")
        elif i == start_idx:
            print(f"  {label}")

        # Run condition
        results = run_condition(
            model, vectors, cond["vector_type"], cond["concept"],
            cond["layers"], cond["alpha"], cond["direction"],
            all_r1_data, all_r2_data, device,
        )

        # Build result row with cross-concept metrics
        target = cond["concept"]
        others = [c for c in CONCEPTS if c != target]

        row = {
            "vector_type": cond["vector_type"],
            "alpha": cond["alpha"],
            "layer_config": cond["layer_config"],
            "n_layers": len(cond["layers"]),
            "concept": target,
            "direction_label": cond["direction_label"],
            # Target concept
            "r1_target": results[target]["r1"],
            "r2_target": results[target]["r2"],
            "r1_baseline_target": baselines[target]["r1"],
            "r2_baseline_target": baselines[target]["r2"],
            "delta_r1_target": baselines[target]["r1"] - results[target]["r1"],
            "delta_r2_target": baselines[target]["r2"] - results[target]["r2"],
        }

        # Cross-concept metrics
        for j, other in enumerate(others):
            suffix = f"_{other}"
            row[f"r1{suffix}"] = results[other]["r1"]
            row[f"r2{suffix}"] = results[other]["r2"]
            row[f"delta_r1{suffix}"] = (
                baselines[other]["r1"] - results[other]["r1"]
            )
            row[f"delta_r2{suffix}"] = (
                baselines[other]["r2"] - results[other]["r2"]
            )

        results_rows.append(row)

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            df_checkpoint = pd.DataFrame(results_rows)
            df_checkpoint.to_csv(CSV_PATH, index=False)
            elapsed_total = time.time() - t_start
            print(f"  --- CHECKPOINT at {i+1}: saved {len(results_rows)} rows "
                  f"({elapsed_total/60:.1f} min elapsed) ---")

    # Save final CSV
    df = pd.DataFrame(results_rows)
    df.to_csv(CSV_PATH, index=False)
    elapsed_total = time.time() - t_start
    print(f"\n  Saved: {CSV_PATH} ({len(df)} rows, "
          f"{elapsed_total/60:.1f} min total)")

    # ===== PHASE 3: ANALYSIS =====
    print(f"\n{'='*60}")
    print("PHASE 3: ANALYSIS & FIGURES")
    print(f"{'='*60}")

    # Sweet spots
    sweet_spots = find_sweet_spots(df, baselines)

    # Generate figures
    sweet_spots = generate_figures(df, baselines)

    # ===== PHASE 4: REPORT =====
    print(f"\n{'='*60}")
    print("PHASE 4: REPORT")
    print(f"{'='*60}")

    df_abl = df[df["direction_label"] == "ablation"]
    df_amp = df[df["direction_label"] == "amplification"]

    # Best per concept
    best_per_concept = {}
    for concept in CONCEPTS:
        concept_df = df_abl[df_abl["concept"] == concept]
        if len(concept_df) > 0:
            best_idx = concept_df["delta_r1_target"].idxmax()
            best = concept_df.loc[best_idx]
            best_per_concept[concept] = {
                "vector_type": best["vector_type"],
                "alpha": float(best["alpha"]),
                "layer_config": best["layer_config"],
                "delta_r1": float(best["delta_r1_target"]),
                "delta_r2": float(best["delta_r2_target"]),
            }

    # Effect of each factor
    factor_effects = {
        "vector_type": {},
        "alpha": {},
        "config_group": {},
    }

    for vtype in VECTOR_TYPES:
        vt_df = df_abl[df_abl["vector_type"] == vtype]
        factor_effects["vector_type"][vtype] = float(
            vt_df["delta_r1_target"].mean()
        )

    for alpha in ALPHAS:
        a_df = df_abl[df_abl["alpha"] == alpha]
        factor_effects["alpha"][str(alpha)] = float(
            a_df["delta_r1_target"].mean()
        )

    def classify_config(name):
        if name.startswith("single"):
            return "single"
        elif name.startswith("pair"):
            return "pair"
        elif name.startswith("triple"):
            return "triple"
        else:
            return "multi"

    df_abl_copy = df_abl.copy()
    df_abl_copy["config_group"] = df_abl_copy["layer_config"].apply(
        classify_config
    )
    for group in ["single", "pair", "triple", "multi"]:
        g_df = df_abl_copy[df_abl_copy["config_group"] == group]
        factor_effects["config_group"][group] = float(
            g_df["delta_r1_target"].mean()
        )

    # Amplification validation
    amp_validation = {}
    for _, row in df_amp.iterrows():
        key = f"{row['concept']}_{row['vector_type']}_a{row['alpha']}"
        amp_validation[key] = {
            "delta_r1": float(row["delta_r1_target"]),
            "valid": float(row["delta_r1_target"]) < 0,
        }

    report = {
        "total_conditions": len(df),
        "n_ablation": len(df_abl),
        "n_amplification": len(df_amp),
        "runtime_minutes": elapsed_total / 60,
        "baselines": baselines,
        "best_per_concept": best_per_concept,
        "factor_effects": factor_effects,
        "sweet_spots": sweet_spots,
        "amplification_validation": amp_validation,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {REPORT_PATH}")

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SWEEP COMPLETE — SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Total: {len(df)} conditions in {elapsed_total/60:.1f} min")

    print(f"\n  Factor effects (mean ΔR1):")
    print(f"    Vector type:")
    for vt, val in factor_effects["vector_type"].items():
        print(f"      {vt:12s}: {val:+.4f}")
    print(f"    Config group:")
    for grp, val in factor_effects["config_group"].items():
        print(f"      {grp:12s}: {val:+.4f}")
    print(f"    Alpha:")
    for a, val in factor_effects["alpha"].items():
        print(f"      α={a:6s}:     {val:+.4f}")

    print(f"\n  Best ablation per concept:")
    for concept, best in best_per_concept.items():
        print(f"    {concept.upper():8s}: ΔR1={best['delta_r1']:+.4f} "
              f"({best['vector_type']}/{best['layer_config']}/α={best['alpha']})")

    print(f"\n  Sweet spots:")
    for concept, ss in sweet_spots.items():
        if ss.get("found"):
            print(f"    {concept.upper():8s}: ΔR1={ss['delta_r1_target']:+.3f}, "
                  f"cross={ss['cross_damage']:.3f}, "
                  f"selectivity={ss['selectivity']:.2f} "
                  f"({ss['vector_type']}/{ss['layer_config']}/α={ss['alpha']})")
        else:
            print(f"    {concept.upper():8s}: {ss.get('note', 'Not found')}")

    n_valid = sum(
        1 for v in amp_validation.values() if v["valid"]
    )
    print(f"\n  Amplification controls: {n_valid}/{len(amp_validation)} valid")

    print(f"\n  Output files:")
    print(f"    {CSV_PATH} ({len(df)} rows)")
    print(f"    {BASELINE_PATH}")
    print(f"    {REPORT_PATH}")
    print(f"    {FIGURES_DIR}/ (8 figures)")


if __name__ == "__main__":
    main()
