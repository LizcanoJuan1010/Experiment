"""
Generate Thesis Figures — Script 1 (No Model Required)
=======================================================
Generates F1-F14 from existing CSV/JSON/PT data.
Uses matplotlib + seaborn + SciencePlots for publication quality.

Usage:
    python generate_figures.py

Output:
    results/figures/fig01_dose_response_r1.{png,pdf}
    ...
    results/figures/fig14_cohens_d_ci.{png,pdf}
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import experiment_config as cfg

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex'])
except ImportError:
    print("WARNING: SciencePlots not installed. Using default style.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIGURES_DIR = "results/figures"
RESULTS_DIR = "results"
CAVS_DIR = "cavs"
CAVS_REFINED_DIR = "cavs_refined"

CONCEPTS = cfg.CONCEPTS
LAYERS = [6, 9, 10]
METHODS = ["mean_diff", "svm"]

# Colorblind-friendly palette (Wong 2011)
CONCEPT_COLORS = {
    'time':  '#0072B2',
    'place': '#D55E00',
    'tools': '#009E73',
}
TECHNIQUE_COLORS = {
    'projection':  '#E69F00',
}
METHOD_STYLES = {
    'mean_diff': {'linestyle': '-',  'marker': 'o'},
    'svm':       {'linestyle': '--', 'marker': 's'},
}

STATUS_COLORS = {
    'PASS': '#2ecc71',
    'FAIL': '#e74c3c',
    'SKIP': '#95a5a6',
}

# Override SciencePlots defaults for thesis
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'figure.figsize': (7, 5),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_figure(fig, name, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved: {name}.png + .pdf")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# F1: Dose-Response R1
# ---------------------------------------------------------------------------
def fig01_dose_response_r1(df, baselines):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
    fig.suptitle("F1: Dose-Response — R1 Accuracy", fontsize=14, fontweight='bold')

    for i, concept in enumerate(CONCEPTS):
        for j, layer in enumerate([6, 10]):
            ax = axes[i, j]
            subset = df[(df['concept'] == concept) & (df['layer'] == layer)]

            for method in METHODS:
                for technique in ['projection']:
                    data = subset[(subset['cav_method'] == method) &
                                  (subset['technique'] == technique)]
                    if data.empty:
                        continue

                    style = METHOD_STYLES[method]
                    color = TECHNIQUE_COLORS[technique]
                    label = f"{technique} ({method})"

                    ax.plot(data['alpha'], data['r1_accuracy'],
                            color=color, linestyle=style['linestyle'],
                            marker=style['marker'], markersize=4,
                            label=label)

            # Baseline
            bl = baselines[concept]['r1_accuracy']
            ax.axhline(y=bl, color='gray', linestyle=':', alpha=0.7,
                       label=f'Baseline ({bl:.3f})')

            ax.set_title(f"{concept.capitalize()} — Layer {layer}", fontsize=10)
            ax.set_ylabel("R1 Accuracy")
            ax.set_ylim(-0.05, 1.05)

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='lower left')

    for ax in axes[-1]:
        ax.set_xlabel("Alpha (α)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig01_dose_response_r1")


# ---------------------------------------------------------------------------
# F2: Dose-Response R2
# ---------------------------------------------------------------------------
def fig02_dose_response_r2(df, baselines):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
    fig.suptitle("F2: Dose-Response — ΔR2 Semantic Similarity", fontsize=14,
                 fontweight='bold')

    for i, concept in enumerate(CONCEPTS):
        for j, layer in enumerate([6, 10]):
            ax = axes[i, j]
            subset = df[(df['concept'] == concept) & (df['layer'] == layer)]

            for method in METHODS:
                for technique in ['projection']:
                    data = subset[(subset['cav_method'] == method) &
                                  (subset['technique'] == technique)]
                    if data.empty:
                        continue

                    style = METHOD_STYLES[method]
                    color = TECHNIQUE_COLORS[technique]
                    label = f"{technique} ({method})"

                    ax.plot(data['alpha'], data['delta_r2'],
                            color=color, linestyle=style['linestyle'],
                            marker=style['marker'], markersize=4,
                            label=label)

            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
            ax.set_title(f"{concept.capitalize()} — Layer {layer}", fontsize=10)
            ax.set_ylabel("ΔR2 (Similarity Drop)")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best')

    for ax in axes[-1]:
        ax.set_xlabel("Alpha (α)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig02_dose_response_r2")


# ---------------------------------------------------------------------------
# F3: Removal Ratio Heatmap
# ---------------------------------------------------------------------------
def fig03_removal_ratio(df):
    if 'removal_ratio' not in df.columns:
        print("  SKIP F3: removal_ratio column not found in factorial CSV.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("F3: Concept Removal Effectiveness (removal_ratio)",
                 fontsize=14, fontweight='bold')

    for idx, layer in enumerate([6, 10]):
        ax = axes[idx]
        subset = df[(df['layer'] == layer) & (df['technique'] == 'projection')]

        if subset.empty:
            ax.set_visible(False)
            continue

        # Build pivot table
        subset = subset.copy()
        subset['condition'] = subset['concept'] + ' / ' + subset['cav_method']
        pivot = subset.pivot_table(
            index='condition', columns='alpha', values='removal_ratio'
        )

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0.5, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'Removal Ratio'})

        ax.set_title(f"Layer {layer} — Projection", fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("Alpha (α)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig03_removal_ratio")


# ---------------------------------------------------------------------------
# F4: Specificity Heatmap 3×3
# ---------------------------------------------------------------------------
def fig04_specificity_heatmap(df_spec):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("F4: Cross-Concept Specificity (Ablation Effect on Other Concepts)",
                 fontsize=13, fontweight='bold')

    for ax_idx, (metric, title) in enumerate([
        ('delta_r1', 'ΔR1 (Accuracy Drop)'),
        ('delta_r2', 'ΔR2 (Similarity Drop)')
    ]):
        ax = axes[ax_idx]
        pivot = df_spec.pivot_table(
            index='ablated_concept', columns='measured_concept', values=metric
        )
        # Reorder
        pivot = pivot.reindex(index=CONCEPTS, columns=CONCEPTS)

        vmax = max(abs(pivot.values.min()), abs(pivot.values.max())) or 0.1
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdBu_r',
                    center=0, vmin=-vmax, vmax=vmax,
                    linewidths=1, ax=ax, square=True,
                    cbar_kws={'label': title})

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Ablated Concept")
        ax.set_xlabel("Measured Concept")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig04_specificity_heatmap")


# ---------------------------------------------------------------------------
# F6: Cross-Concept Cosine Heatmap
# ---------------------------------------------------------------------------
def fig06_cav_cosine_heatmap(report):
    cross = report.get('cross_concept_cosine', {})
    if not cross:
        print("  SKIP F6: No cross_concept_cosine data.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("F6: Cross-Concept Cosine Similarity of CAVs",
                 fontsize=14, fontweight='bold')

    for j, layer in enumerate(LAYERS):
        layer_key = f"layer_{layer}"
        layer_data = cross.get(layer_key, {})

        for i, method in enumerate(METHODS):
            ax = axes[i, j]
            mat = np.eye(3)
            pairs = [('time', 'place'), ('time', 'tools'), ('place', 'tools')]

            for (c1, c2) in pairs:
                key = f"{method}_{c1}_{c2}"
                val = layer_data.get(key, 0)
                i1, i2 = CONCEPTS.index(c1), CONCEPTS.index(c2)
                mat[i1, i2] = val
                mat[i2, i1] = val

            sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r',
                        vmin=-1, vmax=1, center=0,
                        xticklabels=[c.capitalize() for c in CONCEPTS],
                        yticklabels=[c.capitalize() for c in CONCEPTS],
                        linewidths=1, ax=ax, square=True)

            ax.set_title(f"Layer {layer} — {method}", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig06_cav_cosine_heatmap")


# ---------------------------------------------------------------------------
# F7: SVM CV Accuracy + C Óptimo
# ---------------------------------------------------------------------------
def fig07_svm_accuracy(report):
    results = report.get('results', {})
    records = []

    for concept in CONCEPTS:
        concept_data = results.get(concept, {})
        for layer in LAYERS:
            layer_key = f"layer_{layer}"
            layer_data = concept_data.get(layer_key, {})
            svm_data = layer_data.get('svm', {})

            if 'cv_accuracy_mean' in svm_data:
                records.append({
                    'concept': concept.capitalize(),
                    'layer': f"L{layer}",
                    'cv_accuracy': svm_data['cv_accuracy_mean'],
                    'cv_std': svm_data.get('cv_accuracy_std', 0),
                    'C_used': svm_data.get('C_used', None),
                    'best_C': svm_data.get('best_C', None),
                })

    if not records:
        print("  SKIP F7: No SVM accuracy data.")
        return

    df_svm = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("F7: SVM Cross-Validation Accuracy by Concept and Layer",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(df_svm))
    bars = ax.bar(x, df_svm['cv_accuracy'],
                  yerr=df_svm['cv_std'], capsize=4,
                  color=[CONCEPT_COLORS.get(c.lower(), '#999')
                         for c in df_svm['concept']],
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['concept']}\n{r['layer']}" for _, r in df_svm.iterrows()],
        fontsize=9
    )

    # Annotate best C
    for idx, (_, row) in enumerate(df_svm.iterrows()):
        if row['best_C'] is not None:
            ax.annotate(f"C={row['best_C']}",
                       xy=(idx, row['cv_accuracy'] + row['cv_std'] + 0.01),
                       ha='center', fontsize=7, color='gray')

    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.6, label='Chance (0.5)')
    ax.set_ylabel("CV Accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)

    # Concept legend
    handles = [mpatches.Patch(color=c, label=n.capitalize())
               for n, c in CONCEPT_COLORS.items()]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color='red', linestyle=':',
              label='Chance (0.5)')], fontsize=8, loc='lower right')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig07_svm_accuracy")


# ---------------------------------------------------------------------------
# F8: Cross-Method Cosine
# ---------------------------------------------------------------------------
def fig08_cross_method_cosine(report):
    cross = report.get('cross_method_cosine', {})
    if not cross:
        print("  SKIP F8: No cross_method_cosine data.")
        return

    mat = np.zeros((3, 3))
    labels_x = [f"L{l}" for l in LAYERS]
    labels_y = [c.capitalize() for c in CONCEPTS]

    for i, concept in enumerate(CONCEPTS):
        for j, layer in enumerate(LAYERS):
            key = f"{concept}_layer{layer}"
            mat[i, j] = cross.get(key, 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("F8: Mean-Diff vs SVM Cosine Similarity",
                 fontsize=13, fontweight='bold')

    sns.heatmap(mat, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1,
                xticklabels=labels_x, yticklabels=labels_y,
                linewidths=1, ax=ax, square=True,
                cbar_kws={'label': 'Cosine Similarity'})

    ax.set_xlabel("Layer")
    ax.set_ylabel("Concept")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig08_cross_method_cosine")


# ---------------------------------------------------------------------------
# F9: Before/After Orthogonalization
# ---------------------------------------------------------------------------
def fig09_orthogonalization():
    if not HAS_TORCH:
        print("  SKIP F9: torch not available (needed to load .pt files).")
        return

    # Check if refined CAVs exist
    refined_exists = os.path.isdir(CAVS_REFINED_DIR) and len(
        [f for f in os.listdir(CAVS_REFINED_DIR) if f.endswith('.pt')]
    ) > 0

    if not refined_exists:
        print("  SKIP F9: No refined CAVs in cavs_refined/.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("F9: CAV Orthogonalization — Before vs After",
                 fontsize=13, fontweight='bold')

    for ax_idx, (cav_dir, title) in enumerate([
        (CAVS_DIR, "Before (Original)"),
        (CAVS_REFINED_DIR, "After (Orthogonalized)")
    ]):
        ax = axes[ax_idx]
        # Compute cross-concept cosines for SVM CAVs at each layer
        # Average across layers for a single 3×3 matrix
        mat = np.eye(3)
        count = 0

        for layer in LAYERS:
            cavs = {}
            for concept in CONCEPTS:
                path = os.path.join(cav_dir, f"{concept}_svm_layer{layer}.pt")
                if os.path.exists(path):
                    cavs[concept] = torch.load(path, weights_only=True)

            if len(cavs) == 3:
                for ci, c1 in enumerate(CONCEPTS):
                    for cj, c2 in enumerate(CONCEPTS):
                        if ci < cj:
                            cos = torch.nn.functional.cosine_similarity(
                                cavs[c1].unsqueeze(0), cavs[c2].unsqueeze(0)
                            ).item()
                            mat[ci, cj] += cos
                            mat[cj, ci] += cos
                count += 1

        if count > 0:
            # Average the off-diagonal entries
            for ci in range(3):
                for cj in range(3):
                    if ci != cj:
                        mat[ci, cj] /= count

        sns.heatmap(mat, annot=True, fmt='.3f', cmap='coolwarm',
                    vmin=-1, vmax=1, center=0,
                    xticklabels=[c.capitalize() for c in CONCEPTS],
                    yticklabels=[c.capitalize() for c in CONCEPTS],
                    linewidths=1, ax=ax, square=True)
        ax.set_title(title, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig09_orthogonalization")


# ---------------------------------------------------------------------------
# F10: Validation Dashboard (20 tests)
# ---------------------------------------------------------------------------
def fig10_validation_dashboard(val_report):
    # Extract test IDs, names, status, and key values
    rows_data = []
    for key, val in sorted(val_report.items()):
        if not key.startswith('T'):
            continue
        test_id = key.split('_')[0]
        test_name = key.replace(test_id + '_', '').replace('_', ' ').title()
        status = val.get('status', 'SKIP')

        # Extract a key value from details
        details = val.get('details', {})
        key_val = ""
        if isinstance(details, dict):
            # Pick a representative value
            for dk, dv in details.items():
                if isinstance(dv, (int, float)) and 'ratio' in dk.lower():
                    key_val = f"{dk}: {dv:.4f}"
                    break
                elif isinstance(dv, (int, float)) and 'count' in dk.lower():
                    key_val = f"{dk}: {dv}"
                    break
            if not key_val and details:
                first_key = list(details.keys())[0]
                first_val = details[first_key]
                if isinstance(first_val, (int, float)):
                    key_val = f"{first_key}: {first_val}"
                elif isinstance(first_val, str) and len(first_val) < 40:
                    key_val = f"{first_key}: {first_val}"

        rows_data.append([test_id, test_name, status, key_val])

    if not rows_data:
        print("  SKIP F10: No validation data.")
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(rows_data) * 0.35)))
    ax.axis('off')
    fig.suptitle("F10: Validation Test Dashboard",
                 fontsize=14, fontweight='bold')

    table = ax.table(
        cellText=rows_data,
        colLabels=['Test ID', 'Test Name', 'Status', 'Key Value'],
        cellLoc='left',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color cells based on status
    for i, row in enumerate(rows_data):
        status = row[2]
        color = STATUS_COLORS.get(status, '#ffffff')
        for j in range(4):
            cell = table[i + 1, j]  # +1 for header row
            if j == 2:  # Status column
                cell.set_facecolor(color)
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig10_validation_dashboard")


# ---------------------------------------------------------------------------
# F11: TCAV Score vs Null Distribution
# ---------------------------------------------------------------------------
def fig11_tcav_null_dist(stat_val):
    results = stat_val.get('results', {})
    if not results:
        print("  SKIP F11: No statistical validation results.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("F11: TCAV Score vs Null Distribution (Permutation Test)",
                 fontsize=14, fontweight='bold')

    has_data = False

    for i, concept in enumerate(CONCEPTS):
        concept_data = results.get(concept, {})
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            layer_key = f"layer_{layer}"
            layer_data = concept_data.get(layer_key, {})

            # Try SVM first, then mean_diff
            tcav_data = None
            method_label = ""
            for method in ['svm', 'mean_diff']:
                md = layer_data.get(method, {})
                if 'tcav_permutation' in md:
                    tcav_data = md['tcav_permutation']
                    method_label = method
                    break

            if tcav_data is None:
                ax.set_visible(False)
                continue

            has_data = True
            real_score = tcav_data.get('real_tcav_score', 0)
            random_mean = tcav_data.get('random_tcav_scores_mean', 0)
            random_std = tcav_data.get('random_tcav_scores_std', 0)
            p_val = tcav_data.get('p_value', 1.0)
            n_perm = tcav_data.get('n_permutations', 30)

            # Simulate null distribution from mean/std (Normal approximation)
            null_scores = np.random.normal(random_mean, max(random_std, 0.01),
                                           size=n_perm * 10)

            sns.histplot(null_scores, kde=True, color='#95a5a6', alpha=0.6,
                        ax=ax, stat='density', label='Null Distribution')

            ax.axvline(x=real_score, color='#e74c3c', linewidth=2,
                      linestyle='-', label=f'Real TCAV = {real_score:.3f}')

            # Significance annotation
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
                  "*" if p_val < 0.05 else "n.s."
            ax.text(0.95, 0.95, f"p = {p_val:.3f} {sig}",
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

            ax.set_title(f"{concept.capitalize()} — L{layer} ({method_label})",
                        fontsize=9)
            ax.set_xlabel("TCAV Score")
            ax.set_ylabel("Density")

            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='upper left')

    if not has_data:
        plt.close(fig)
        print("  SKIP F11: No TCAV permutation data found.")
        return

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig11_tcav_null_dist")


# ---------------------------------------------------------------------------
# F12: Selectivity Bar Chart
# ---------------------------------------------------------------------------
def fig12_selectivity(stat_val):
    results = stat_val.get('results', {})
    records = []

    for concept in CONCEPTS:
        concept_data = results.get(concept, {})
        for layer in LAYERS:
            layer_key = f"layer_{layer}"
            layer_data = concept_data.get(layer_key, {})

            for method in METHODS:
                md = layer_data.get(method, {})
                sel = md.get('selectivity', {})
                if sel:
                    records.append({
                        'concept': concept.capitalize(),
                        'layer': f"L{layer}",
                        'method': method,
                        'label': f"{concept[:3].upper()}\nL{layer}\n{method[:3]}",
                        'real_accuracy': sel.get('real_accuracy', 0),
                        'control_accuracy': sel.get('mean_control_accuracy', 0),
                        'selectivity': sel.get('selectivity', 0),
                    })

    if not records:
        print("  SKIP F12: No selectivity data.")
        return

    df_sel = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.6), 5))
    fig.suptitle("F12: Selectivity — Real vs Shuffled Control Accuracy",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(df_sel))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_sel['real_accuracy'], width,
                   color=[CONCEPT_COLORS.get(c.lower(), '#999')
                          for c in df_sel['concept']],
                   edgecolor='black', linewidth=0.5, label='Real')

    bars2 = ax.bar(x + width/2, df_sel['control_accuracy'], width,
                   color=[CONCEPT_COLORS.get(c.lower(), '#999')
                          for c in df_sel['concept']],
                   edgecolor='black', linewidth=0.5, alpha=0.4,
                   hatch='///', label='Shuffled Control')

    # Annotate selectivity
    for idx in range(len(df_sel)):
        row = df_sel.iloc[idx]
        ax.annotate(f"Δ={row['selectivity']:.3f}",
                   xy=(x[idx], max(row['real_accuracy'],
                                    row['control_accuracy']) + 0.02),
                   ha='center', fontsize=7, color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels(df_sel['label'], fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Chance')
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig12_selectivity")


# ---------------------------------------------------------------------------
# F13: Bootstrap CI Forest Plot
# ---------------------------------------------------------------------------
def fig13_bootstrap_ci(stat_val):
    results = stat_val.get('results', {})
    records = []

    for concept in CONCEPTS:
        concept_data = results.get(concept, {})
        for layer in LAYERS:
            layer_key = f"layer_{layer}"
            layer_data = concept_data.get(layer_key, {})

            for method in METHODS:
                md = layer_data.get(method, {})
                boot = md.get('bootstrap_ci', {})
                if boot and 'ci_lower' in boot:
                    records.append({
                        'label': f"{concept.capitalize()} / L{layer} / {method}",
                        'concept': concept,
                        'median': boot.get('median_cosine', 0),
                        'ci_lower': boot['ci_lower'],
                        'ci_upper': boot['ci_upper'],
                        'ci_width': boot['ci_upper'] - boot['ci_lower'],
                    })

    if not records:
        print("  SKIP F13: No bootstrap CI data.")
        return

    # Sort by CI width (narrowest on top)
    df_boot = pd.DataFrame(records).sort_values('ci_width')

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_boot) * 0.35)))
    fig.suptitle("F13: Bootstrap 95% CI — CAV Directional Stability",
                 fontsize=13, fontweight='bold')

    y = np.arange(len(df_boot))
    colors = [CONCEPT_COLORS.get(r['concept'], '#999') for _, r in df_boot.iterrows()]

    for i, (_, row) in enumerate(df_boot.iterrows()):
        ax.errorbar(row['median'], i,
                   xerr=[[row['median'] - row['ci_lower']],
                          [row['ci_upper'] - row['median']]],
                   fmt='o', color=colors[i], markersize=6, capsize=4,
                   linewidth=1.5)

    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.6,
              label='Perfect Stability (cos=1.0)')
    ax.set_yticks(y)
    ax.set_yticklabels(df_boot['label'], fontsize=8)
    ax.set_xlabel("Median Cosine Similarity (BCa 95% CI)")
    ax.set_xlim(-0.1, 1.1)
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig13_bootstrap_ci")


# ---------------------------------------------------------------------------
# F14: Cohen's d with CI (Forest Plot)
# ---------------------------------------------------------------------------
def fig14_cohens_d_ci(stat_val, report):
    records = []

    # Try statistical_validation first
    sv_results = stat_val.get('results', {}) if stat_val else {}

    for concept in CONCEPTS:
        for layer in LAYERS:
            for method in METHODS:
                d_data = None

                # Source 1: statistical_validation.json
                if sv_results:
                    layer_key = f"layer_{layer}"
                    md = sv_results.get(concept, {}).get(layer_key, {}).get(method, {})
                    d_data = md.get('cohens_d', {})

                # Source 2: extraction_report.json (fallback)
                if not d_data or 'cohens_d' not in d_data:
                    rpt = report.get('results', {}).get(concept, {})
                    layer_key = f"layer_{layer}"
                    method_data = rpt.get(layer_key, {}).get(method, {})
                    if 'cohens_d' in method_data:
                        d_data = {'cohens_d': method_data['cohens_d']}

                if d_data and 'cohens_d' in d_data:
                    records.append({
                        'label': f"{concept.capitalize()} / L{layer} / {method}",
                        'concept': concept,
                        'd': d_data['cohens_d'],
                        'g': d_data.get('hedges_g', d_data['cohens_d']),
                        'ci_lower': d_data.get('ci_95_lower', None),
                        'ci_upper': d_data.get('ci_95_upper', None),
                    })

    if not records:
        print("  SKIP F14: No Cohen's d data.")
        return

    df_d = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_d) * 0.4)))
    fig.suptitle("F14: Effect Size — Cohen's d with 95% CI",
                 fontsize=13, fontweight='bold')

    # Effect size bands
    ax.axvspan(0, 0.2, alpha=0.05, color='gray')
    ax.axvspan(0.2, 0.5, alpha=0.08, color='#f1c40f', label='Small (0.2–0.5)')
    ax.axvspan(0.5, 0.8, alpha=0.08, color='#e67e22', label='Medium (0.5–0.8)')
    ax.axvspan(0.8, max(df_d['d'].max() + 1, 2), alpha=0.08, color='#e74c3c',
               label='Large (>0.8)')

    y = np.arange(len(df_d))
    colors = [CONCEPT_COLORS.get(r['concept'], '#999') for _, r in df_d.iterrows()]

    for i, (_, row) in enumerate(df_d.iterrows()):
        if row['ci_lower'] is not None and row['ci_upper'] is not None:
            ax.errorbar(row['d'], i,
                       xerr=[[row['d'] - row['ci_lower']],
                              [row['ci_upper'] - row['d']]],
                       fmt='o', color=colors[i], markersize=6, capsize=4,
                       linewidth=1.5)
        else:
            ax.plot(row['d'], i, 'o', color=colors[i], markersize=6)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df_d['label'], fontsize=8)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig14_cohens_d_ci")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Generate Thesis Figures — Script 1 (No Model)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df_factorial = load_csv(os.path.join(RESULTS_DIR, "results_factorial.csv"))
    df_specificity = load_csv(os.path.join(RESULTS_DIR, "results_specificity.csv"))
    baselines = load_json(os.path.join(RESULTS_DIR, "results_baseline.json"))
    report = load_json(os.path.join(CAVS_DIR, "extraction_report.json"))

    # Optional files
    val_report = None
    val_path = "validation_report.json"
    if os.path.exists(val_path):
        val_report = load_json(val_path)

    stat_val = None
    stat_path = os.path.join(CAVS_DIR, "statistical_validation.json")
    if os.path.exists(stat_path):
        stat_val = load_json(stat_path)

    print(f"  Factorial: {len(df_factorial)} rows")
    print(f"  Specificity: {len(df_specificity)} rows")
    print(f"  Validation report: {'loaded' if val_report else 'not found'}")
    print(f"  Statistical validation: {'loaded' if stat_val else 'not found'}")

    # --- F1-F5: Experimental Results ---
    print("\n--- Experimental Results ---")
    fig01_dose_response_r1(df_factorial, baselines)
    fig02_dose_response_r2(df_factorial, baselines)
    fig03_removal_ratio(df_factorial)
    fig04_specificity_heatmap(df_specificity)
    # --- F6-F9: CAV Quality ---
    print("\n--- CAV Quality ---")
    fig06_cav_cosine_heatmap(report)
    fig07_svm_accuracy(report)
    fig08_cross_method_cosine(report)
    fig09_orthogonalization()

    # --- F10-F14: Statistical Validation ---
    print("\n--- Statistical Validation ---")
    if val_report:
        fig10_validation_dashboard(val_report)
    else:
        print("  SKIP F10: validation_report.json not found.")

    if stat_val:
        fig11_tcav_null_dist(stat_val)
        fig12_selectivity(stat_val)
        fig13_bootstrap_ci(stat_val)
        fig14_cohens_d_ci(stat_val, report)
    else:
        print("  SKIP F11-F14: statistical_validation.json not found.")

    print("\n" + "=" * 60)
    print("DONE — All figures saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
