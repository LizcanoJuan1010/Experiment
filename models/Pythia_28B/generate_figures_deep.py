"""
Generate Thesis Figures — Script 2 (Requires Model)
=====================================================
Generates F15-F20 deep activation visualizations.
Requires Pythia 2.8B via TransformerLens (GPU required).

Usage (inside Docker):
    python generate_figures_deep.py

Output:
    results/figures/fig15_pca_activations.{png,pdf}
    ...
    results/figures/fig20_layer_diagnostic.{png,pdf}
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from functools import partial
from sklearn.decomposition import PCA

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex'])
except ImportError:
    print("WARNING: SciencePlots not installed. Using default style.")

from transformer_lens import HookedTransformer

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_config as cfg
from cav_extraction import collect_activations, cohens_d_activations
from experiment_runner import ablation_hook

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIGURES_DIR = "results/figures"
CAVS_DIR = "cavs"
DATA_FILE = cfg.DATA_FILE
MODEL_NAME = cfg.MODEL_NAME
CONCEPTS = cfg.CONCEPTS
LAYERS = cfg.EXTRACTION_LAYERS  # [16, 24, 27] for Pythia 2.8B

# Colorblind-friendly palette (Wong 2011)
CONCEPT_COLORS = {
    'time':  '#0072B2',
    'place': '#D55E00',
    'tools': '#009E73',
}

# Override rcParams for thesis quality
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
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png + .pdf")


def load_cav(concept, method, layer):
    path = os.path.join(CAVS_DIR, f"{concept}_{method}_layer{layer}.pt")
    return torch.load(path, weights_only=True)


def load_activations(concept, layer):
    """Load saved activations if available."""
    path = os.path.join(CAVS_DIR, f"{concept}_acts_layer{layer}.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            return data.get('pos', None), data.get('neg', None)
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            return data[0], data[1]
    return None, None


def get_sentences(data, concept, split="positive", max_n=50):
    """Get sentences from experiment_data.json."""
    return data["cav_training"][concept][split][:max_n]


def draw_confidence_ellipse(ax, x, y, color, n_std=2.0, alpha=0.2):
    """Draw a 95% confidence ellipse."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color,
                      linewidth=1.5, linestyle='--')
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# F15: PCA Activation Space (Kim 2018)
# ---------------------------------------------------------------------------
def fig15_pca_activations(model, data):
    print("\n  F15: PCA Activation Space...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("F15: PCA of Activation Space — Positive vs Negative",
                 fontsize=14, fontweight='bold')

    for j, layer in enumerate(LAYERS):
        ax = axes[j]
        all_acts_list = []
        all_labels = []
        all_concepts = []
        concept_cavs = {}

        for concept in CONCEPTS:
            pos_sents = get_sentences(data, concept, "positive", max_n=50)
            neg_sents = get_sentences(data, concept, "negative", max_n=50)

            # Try loading saved activations first
            pos_acts, neg_acts = load_activations(concept, layer)

            if pos_acts is None:
                print(f"    Collecting activations: {concept} layer {layer}...")
                pos_acts = collect_activations(model, pos_sents, layer)
                neg_acts = collect_activations(model, neg_sents, layer)
            else:
                # Limit to same number as sentences
                pos_acts = pos_acts[:len(pos_sents)]
                neg_acts = neg_acts[:len(neg_sents)]

            all_acts_list.append(pos_acts.numpy())
            all_acts_list.append(neg_acts.numpy())
            all_labels.extend(['pos'] * len(pos_acts) + ['neg'] * len(neg_acts))
            all_concepts.extend([concept] * (len(pos_acts) + len(neg_acts)))

            # Load CAV for arrow
            try:
                cav = load_cav(concept, 'svm', layer)
                concept_cavs[concept] = cav.numpy()
            except FileNotFoundError:
                pass

        all_acts = np.vstack(all_acts_list)
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)

        # Plot each concept
        idx = 0
        for concept in CONCEPTS:
            color = CONCEPT_COLORS[concept]
            n_pos = len(get_sentences(data, concept, "positive", max_n=50))
            n_neg = len(get_sentences(data, concept, "negative", max_n=50))

            pos_proj = projected[idx:idx+n_pos]
            neg_proj = projected[idx+n_pos:idx+n_pos+n_neg]

            ax.scatter(pos_proj[:, 0], pos_proj[:, 1], c=color,
                      marker='o', s=25, alpha=0.6, label=f'{concept} (+)')
            ax.scatter(neg_proj[:, 0], neg_proj[:, 1], c=color,
                      marker='x', s=25, alpha=0.6, label=f'{concept} (−)')

            # Confidence ellipses
            draw_confidence_ellipse(ax, pos_proj[:, 0], pos_proj[:, 1],
                                   color, alpha=0.15)
            draw_confidence_ellipse(ax, neg_proj[:, 0], neg_proj[:, 1],
                                   color, alpha=0.08)

            # CAV arrow
            if concept in concept_cavs:
                cav_2d = pca.transform(concept_cavs[concept].reshape(1, -1))[0]
                origin = np.mean(np.vstack([pos_proj, neg_proj]), axis=0)
                scale = np.std(projected[:, 0]) * 0.5
                ax.annotate('', xy=(origin[0] + cav_2d[0]*scale,
                                    origin[1] + cav_2d[1]*scale),
                           xytext=(origin[0], origin[1]),
                           arrowprops=dict(arrowstyle='->', color='red',
                                          lw=2.0, mutation_scale=15))

            idx += n_pos + n_neg

        var_exp = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.set_title(f"Layer {layer}", fontsize=11)

        if j == 0:
            ax.legend(fontsize=6, loc='best', ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig15_pca_activations")


# ---------------------------------------------------------------------------
# F16: CAV Projection Histogram with CI
# ---------------------------------------------------------------------------
def fig16_cav_projection_hist(model, data):
    print("\n  F16: CAV Projection Histograms...")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("F16: CAV Projection Distributions — Positive vs Negative",
                 fontsize=14, fontweight='bold')

    for i, concept in enumerate(CONCEPTS):
        pos_sents = get_sentences(data, concept, "positive", max_n=50)
        neg_sents = get_sentences(data, concept, "negative", max_n=50)

        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]

            # Get activations
            pos_acts, neg_acts = load_activations(concept, layer)
            if pos_acts is None:
                print(f"    Collecting: {concept} layer {layer}...")
                pos_acts = collect_activations(model, pos_sents, layer)
                neg_acts = collect_activations(model, neg_sents, layer)
            else:
                pos_acts = pos_acts[:len(pos_sents)]
                neg_acts = neg_acts[:len(neg_sents)]

            # Load CAV
            try:
                cav = load_cav(concept, 'svm', layer)
            except FileNotFoundError:
                cav = load_cav(concept, 'mean_diff', layer)

            # Compute projections
            cav_np = cav.numpy()
            pos_proj = pos_acts.numpy() @ cav_np
            neg_proj = neg_acts.numpy() @ cav_np

            # Plot histograms
            color = CONCEPT_COLORS[concept]
            sns.histplot(pos_proj, kde=True, color=color, alpha=0.5,
                        label='Positive', ax=ax, stat='density')
            sns.histplot(neg_proj, kde=True, color=color, alpha=0.25,
                        label='Negative', ax=ax, stat='density',
                        linestyle='--')

            # Compute effect size
            d_result = cohens_d_activations(pos_acts, neg_acts, cav)

            # Annotation box
            textstr = (f"d = {d_result['cohens_d']:.2f}\n"
                       f"g = {d_result['hedges_g']:.2f}\n"
                       f"CI: [{d_result['ci_95'][0]:.2f}, {d_result['ci_95'][1]:.2f}]")
            props = dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.8)
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=props)

            ax.set_title(f"{concept.capitalize()} — Layer {layer}", fontsize=9)
            ax.set_xlabel("CAV Projection")
            ax.set_ylabel("Density")

            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig16_cav_projection_hist")


# ---------------------------------------------------------------------------
# F17: Token-Level Heatmap (RepE-inspired)
# ---------------------------------------------------------------------------
def fig17_token_heatmap(model, data):
    print("\n  F17: Token-Level CAV Projection Heatmap...")

    n_layers_total = model.cfg.n_layers  # 32 for Pythia 2.8B

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("F17: Token-Level CAV Projection Across All Layers",
                 fontsize=14, fontweight='bold')

    for col, concept in enumerate(CONCEPTS):
        # Pick 1 positive and 1 negative sentence
        pos_sent = get_sentences(data, concept, "positive", max_n=1)[0]
        neg_sent = get_sentences(data, concept, "negative", max_n=1)[0]

        # Load CAV (SVM preferred, at middle experiment layer)
        try:
            cav = load_cav(concept, 'svm', LAYERS[0])
        except FileNotFoundError:
            cav = load_cav(concept, 'mean_diff', LAYERS[0])
        cav_np = cav.numpy()

        for row, (sent, label) in enumerate([(pos_sent, "Positive"),
                                              (neg_sent, "Negative")]):
            ax = axes[row, col]

            # Get activations at ALL layers
            hook_names = [f"blocks.{l}.hook_resid_post"
                         for l in range(n_layers_total)]

            with torch.no_grad():
                _, cache = model.run_with_cache(sent, names_filter=hook_names)

            tokens = model.to_str_tokens(sent)
            n_tokens = len(tokens)

            # Build [n_layers, n_tokens] matrix
            proj_matrix = np.zeros((n_layers_total, n_tokens))
            for l in range(n_layers_total):
                hook = f"blocks.{l}.hook_resid_post"
                act = cache[hook][0].cpu().numpy()  # [seq_len, d_model]
                proj_matrix[l, :] = act @ cav_np

            # Plot heatmap
            vmax = max(abs(proj_matrix.min()), abs(proj_matrix.max()))
            im = ax.imshow(proj_matrix, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, interpolation='nearest')

            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels([t.replace(' ', '·') for t in tokens],
                              rotation=45, ha='right', fontsize=6)
            ax.set_yticks(range(n_layers_total))
            ax.set_yticklabels([f"L{l}" for l in range(n_layers_total)],
                              fontsize=7)

            # Highlight experiment layers
            for exp_l in LAYERS:
                ax.axhline(y=exp_l - 0.5, color='gold', linewidth=0.8,
                          linestyle='--', alpha=0.7)
                ax.axhline(y=exp_l + 0.5, color='gold', linewidth=0.8,
                          linestyle='--', alpha=0.7)

            ax.set_title(f"{concept.capitalize()} — {label}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Layer")

            plt.colorbar(im, ax=ax, shrink=0.8, label='CAV Projection')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig17_token_heatmap")


# ---------------------------------------------------------------------------
# F18: Layer-wise Evolution (Logit Lens adapted)
# ---------------------------------------------------------------------------
def fig18_layerwise_evolution(model, data):
    print("\n  F18: Layer-wise CAV Projection Evolution...")

    n_layers_total = model.cfg.n_layers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("F18: Layer-wise CAV Projection — Positive vs Negative",
                 fontsize=14, fontweight='bold')

    for j, concept in enumerate(CONCEPTS):
        ax = axes[j]
        color = CONCEPT_COLORS[concept]

        pos_sents = get_sentences(data, concept, "positive", max_n=20)
        neg_sents = get_sentences(data, concept, "negative", max_n=20)

        # Load CAV
        try:
            cav = load_cav(concept, 'svm', LAYERS[0])
        except FileNotFoundError:
            cav = load_cav(concept, 'mean_diff', LAYERS[0])
        cav_np = cav.numpy()

        pos_projs = np.zeros(n_layers_total)
        neg_projs = np.zeros(n_layers_total)
        pos_stds = np.zeros(n_layers_total)
        neg_stds = np.zeros(n_layers_total)

        hook_names = [f"blocks.{l}.hook_resid_post"
                     for l in range(n_layers_total)]

        # Collect mean projections per layer
        for sents, projs_arr, stds_arr in [
            (pos_sents, pos_projs, pos_stds),
            (neg_sents, neg_projs, neg_stds)
        ]:
            layer_values = [[] for _ in range(n_layers_total)]
            for sent in sents:
                with torch.no_grad():
                    _, cache = model.run_with_cache(sent,
                                                     names_filter=hook_names)
                for l in range(n_layers_total):
                    hook = f"blocks.{l}.hook_resid_post"
                    act = cache[hook][0].mean(dim=0).cpu().numpy()
                    proj = float(act @ cav_np)
                    layer_values[l].append(proj)

            for l in range(n_layers_total):
                projs_arr[l] = np.mean(layer_values[l])
                stds_arr[l] = np.std(layer_values[l])

        layers_x = np.arange(n_layers_total)

        # Plot with fill_between
        ax.plot(layers_x, pos_projs, '-o', color=color, markersize=4,
                label='Positive', linewidth=1.5)
        ax.fill_between(layers_x, pos_projs - pos_stds, pos_projs + pos_stds,
                        color=color, alpha=0.15)

        ax.plot(layers_x, neg_projs, '--s', color=color, markersize=4,
                label='Negative', alpha=0.7, linewidth=1.5)
        ax.fill_between(layers_x, neg_projs - neg_stds, neg_projs + neg_stds,
                        color=color, alpha=0.08)

        # Mark experiment layers
        for exp_l in LAYERS:
            ax.axvline(x=exp_l, color='gold', linestyle=':', alpha=0.6)
            if exp_l == LAYERS[0]:
                ax.plot(exp_l, pos_projs[exp_l], '*', color='gold',
                       markersize=12, zorder=5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean CAV Projection")
        ax.set_title(f"{concept.capitalize()}", fontsize=11)
        ax.set_xticks(layers_x)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig18_layerwise_evolution")


# ---------------------------------------------------------------------------
# F19: Before/After Ablation Token Heatmap
# ---------------------------------------------------------------------------
def fig19_ablation_comparison(model, data):
    print("\n  F19: Ablation Before/After Token Heatmap...")

    n_layers_total = model.cfg.n_layers
    ablation_layer = LAYERS[0]  # Layer 6
    alpha_val = 3.5
    technique = "projection"

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"F19: Token-Level Activation — Before/After Ablation "
                 f"(L{ablation_layer}, α={alpha_val}, {technique})",
                 fontsize=14, fontweight='bold')

    col_titles = ["Pre-Ablation", "Post-Ablation", "Difference"]

    for row, concept in enumerate(CONCEPTS):
        pos_sent = get_sentences(data, concept, "positive", max_n=1)[0]

        try:
            cav = load_cav(concept, 'svm', ablation_layer)
        except FileNotFoundError:
            cav = load_cav(concept, 'mean_diff', ablation_layer)
        cav_np = cav.numpy()
        cav_device = cav.to(model.cfg.device)

        hook_names = [f"blocks.{l}.hook_resid_post"
                     for l in range(n_layers_total)]

        tokens = model.to_str_tokens(pos_sent)
        n_tokens = len(tokens)

        # Pre-ablation
        with torch.no_grad():
            _, cache_pre = model.run_with_cache(pos_sent,
                                                 names_filter=hook_names)

        pre_matrix = np.zeros((n_layers_total, n_tokens))
        for l in range(n_layers_total):
            act = cache_pre[f"blocks.{l}.hook_resid_post"][0].cpu().numpy()
            pre_matrix[l, :] = act @ cav_np

        # Post-ablation
        hook_fn = partial(ablation_hook, cav=cav_device, alpha=alpha_val,
                         technique=technique)
        abl_hook_name = f"blocks.{ablation_layer}.hook_resid_post"

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(abl_hook_name, hook_fn)]):
                _, cache_post = model.run_with_cache(pos_sent,
                                                      names_filter=hook_names)

        post_matrix = np.zeros((n_layers_total, n_tokens))
        for l in range(n_layers_total):
            act = cache_post[f"blocks.{l}.hook_resid_post"][0].cpu().numpy()
            post_matrix[l, :] = act @ cav_np

        diff_matrix = pre_matrix - post_matrix

        # Shared vmax for pre/post
        vmax_pp = max(abs(pre_matrix).max(), abs(post_matrix).max())
        vmax_diff = abs(diff_matrix).max()

        matrices = [pre_matrix, post_matrix, diff_matrix]
        cmaps = ['RdBu_r', 'RdBu_r', 'PiYG']
        vmaxes = [vmax_pp, vmax_pp, vmax_diff]

        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(matrices[col], aspect='auto', cmap=cmaps[col],
                          vmin=-vmaxes[col], vmax=vmaxes[col],
                          interpolation='nearest')

            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels([t.replace(' ', '·') for t in tokens],
                              rotation=45, ha='right', fontsize=5)
            ax.set_yticks(range(n_layers_total))
            ax.set_yticklabels([f"L{l}" for l in range(n_layers_total)],
                              fontsize=6)

            # Mark ablation layer
            ax.axhline(y=ablation_layer - 0.5, color='gold', linewidth=1.5)
            ax.axhline(y=ablation_layer + 0.5, color='gold', linewidth=1.5)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{concept.capitalize()}\nLayer", fontsize=9)

            plt.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig19_ablation_comparison")


# ---------------------------------------------------------------------------
# F20: Layer Diagnostic — Probing Accuracy All 12 Layers
# ---------------------------------------------------------------------------
def fig20_layer_diagnostic(report):
    """Plot layer diagnostic from extraction_report.json (no model needed)."""
    print("\n  F20: Layer Diagnostic Probing Accuracy...")

    diag = report.get('layer_diagnostic', {})
    layer_accs = diag.get('layer_accuracies', {})

    if not layer_accs:
        print("  SKIP F20: No layer_diagnostic data in extraction_report.json.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("F20: Layer Diagnostic — Probing Accuracy (All 12 Layers)",
                 fontsize=13, fontweight='bold')

    layers_x = sorted([int(k) for k in layer_accs.keys()])
    accuracies = [layer_accs[str(l)] for l in layers_x]

    ax.plot(layers_x, accuracies, '-o', color='#2c3e50', markersize=6,
            linewidth=2, label='Probing Accuracy')

    # Highlight experiment layers with stars
    for exp_l in LAYERS:
        if exp_l in layers_x:
            idx = layers_x.index(exp_l)
            ax.plot(exp_l, accuracies[idx], '*', color='gold',
                   markersize=16, zorder=5, markeredgecolor='black',
                   markeredgewidth=0.5)

    # Chance line
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5,
              label='Chance (0.5)')

    # Annotate selected layers
    ax.annotate("Selected layers\nfor experiment",
               xy=(LAYERS[0], layer_accs.get(str(LAYERS[0]), 0.5)),
               xytext=(LAYERS[0] + 1.5, max(accuracies) + 0.02),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=8, color='gray')

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_xticks(layers_x)
    ax.set_ylim(0.45, max(accuracies) + 0.05)
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig20_layer_diagnostic")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Generate Thesis Figures — Script 2 (Deep Activations)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load experiment data
    print(f"\nLoading data from {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Load extraction report (for F20)
    report_path = os.path.join(CAVS_DIR, "extraction_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    # F20 doesn't need the model — generate it first
    fig20_layer_diagnostic(report)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    model = cfg.load_model()
    print(f"  Model loaded on: {model.cfg.device}")

    # Generate deep figures
    fig15_pca_activations(model, data)
    fig16_cav_projection_hist(model, data)
    fig17_token_heatmap(model, data)
    fig18_layerwise_evolution(model, data)
    fig19_ablation_comparison(model, data)

    print("\n" + "=" * 60)
    print("DONE — Deep figures saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
