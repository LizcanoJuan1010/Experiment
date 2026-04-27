"""Activations projected onto the CAV axis (+ orthogonal PCA) per model.

For each concept × layer, project (pos, neg) activations to 2D where:
  x-axis = CAV direction (class-discriminative)
  y-axis = max-variance direction orthogonal to CAV

Overlay the CAV direction as an arrow and the SVM boundary (orthogonal to
the CAV at the origin of the projection), and add a 95% covariance ellipse
per group to visualize class spread and overlap.

Generic PCA fails for small models (GPT-2, Pythia) because PC1 and PC2 are
dominated by sentence-length or positional variance unrelated to the
concept. Using the CAV as the x-axis guarantees that whatever class
separability exists is visible on the first axis.

An inset in each subplot reports the Mahalanobis distance between the pos
and neg centroids in the PCA 2D space (stratified bootstrap 95% CI),
converting a descriptive scatter into a contrast plot: D=0 means no
separation; the CI shows how robust the separation is to resampling.
The 2D space is used because it is what the reader actually sees and because
pinv of a 4096×4096 covariance per bootstrap iteration is prohibitive.

Runs once per model (GPT-2 Small, Pythia-2.8B, LLaVA-1.5-7B) with the
concept/layer set appropriate to each, saving one PNG per model.
"""
from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA

from viz_style import (MODELS_DIR, apply_style, savefig, depth_frac,
                       MODEL_META)
from ci_utils import confidence_ellipse

apply_style()

N_BOOT_MAHA = 600

# Per-model: which concepts to plot and at which layers (must have _acts_layer*.pt)
MODEL_CONFIG = {
    "gpt2_small": {
        "concepts": ["camel", "bee", "penguin", "violin", "candle"],
        "layers":   [6, 9, 10],
    },
    "pythia_28b": {
        "concepts": ["camel", "bee", "penguin", "violin", "candle"],
        "layers":   [16, 24, 27],
    },
    "llava_15_7b": {
        "concepts": ["golden_retriever", "labrador_retriever", "tabby_cat"],
        "layers":   [16, 24, 27],
    },
}


def load_acts(model_key, concept, layer):
    p = MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs" / f"{concept}_acts_layer{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["pos"].numpy(), d["neg"].numpy()


def load_cav(model_key, concept, layer):
    p = MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs" / f"{concept}_svm_layer{layer}.pt"
    t = torch.load(p, map_location="cpu", weights_only=False)
    if hasattr(t, "numpy"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def project_with_cav(pos, neg, cav):
    """Project onto (CAV, orthogonal-to-CAV) axes.

    x-axis = CAV direction (class-discriminative)
    y-axis = direction with max remaining variance orthogonal to CAV

    This is NOT generic PCA: PCA finds directions of max variance, which for
    GPT-2 and Pythia are dominated by sentence-level variability unrelated
    to the concept. Projecting onto the CAV guarantees that any class
    separation is visible on the x-axis, which is what the figure intends
    to show. In the returned basis the CAV is exactly the unit vector (1, 0).
    """
    X = np.vstack([pos, neg])
    mu = X.mean(axis=0)
    Xc = X - mu
    cav_u = cav / (np.linalg.norm(cav) + 1e-12)
    x_coord = Xc @ cav_u
    # Remove CAV component, then pick the direction of max remaining variance
    X_perp = Xc - np.outer(x_coord, cav_u)
    pca_perp = PCA(n_components=1, random_state=0).fit(X_perp)
    y_dir = pca_perp.components_[0]
    y_coord = Xc @ y_dir
    X2 = np.stack([x_coord, y_coord], axis=1)
    n_pos = len(pos)
    cav_2 = np.array([1.0, 0.0])
    return X2[:n_pos], X2[n_pos:], cav_2


def mahalanobis_between_2d(p2, n2):
    """Mahalanobis distance between centroids in a 2D space (fast, closed form)."""
    mu_p = p2.mean(axis=0)
    mu_n = n2.mean(axis=0)
    diff = mu_p - mu_n
    pooled = np.cov(np.vstack([p2 - mu_p, n2 - mu_n]).T)
    try:
        inv = np.linalg.inv(pooled + 1e-9 * np.eye(2))
    except np.linalg.LinAlgError:
        return np.nan
    d2 = float(diff @ inv @ diff)
    return float(np.sqrt(max(d2, 0.0)))


def mahalanobis_ci_2d(p2, n2, n_boot=N_BOOT_MAHA,
                     rng=None) -> tuple[float, float, float]:
    """Stratified bootstrap CI for 2D Mahalanobis distance between centroids."""
    rng = rng or np.random.default_rng(20260417)
    d_hat = mahalanobis_between_2d(p2, n2)
    if np.isnan(d_hat) or p2.shape[0] < 3 or n2.shape[0] < 3:
        return d_hat, np.nan, np.nan
    boots = np.empty(n_boot)
    for k in range(n_boot):
        ip = rng.integers(0, p2.shape[0], p2.shape[0])
        in_ = rng.integers(0, n2.shape[0], n2.shape[0])
        boots[k] = mahalanobis_between_2d(p2[ip], n2[in_])
    boots = boots[~np.isnan(boots)]
    if boots.size < 20:
        return d_hat, np.nan, np.nan
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return d_hat, float(lo), float(hi)


def make_figure(model_key: str, concepts: list[str], layers: list[int]):
    """Render and save one activation PCA figure for a single model."""
    # Drop concepts that don't have all requested layers on disk
    available = []
    for c in concepts:
        if all((MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs" /
                f"{c}_acts_layer{L}.pt").exists() for L in layers):
            available.append(c)
        else:
            print(f"  [{model_key}] skipping {c}: acts missing for some layer")
    if not available:
        print(f"  [{model_key}] no concepts with complete activations — skipped")
        return
    concepts = available

    fig, axes = plt.subplots(len(concepts), len(layers),
                             figsize=(4.3 * len(layers), 3.6 * len(concepts)),
                             squeeze=False)

    for ri, concept in enumerate(concepts):
        for ci, L in enumerate(layers):
            ax = axes[ri, ci]
            try:
                pos, neg = load_acts(model_key, concept, L)
                cav = load_cav(model_key, concept, L)
            except FileNotFoundError:
                ax.axis("off"); continue

            p2, n2, cav_dir = project_with_cav(pos, neg, cav)

            # Different markers + equal zorder + lower alpha so overlapping
            # groups remain visible through color blending
            n_each = max(len(p2), len(n2))
            a = 0.65 if n_each <= 40 else 0.4 if n_each <= 80 else 0.3
            ax.scatter(n2[:, 0], n2[:, 1], s=32, c="#9D6FB3", alpha=a,
                       marker="v", edgecolors="white", linewidths=0.4,
                       label="negativo", zorder=2)
            ax.scatter(p2[:, 0], p2[:, 1], s=28, c="#DD8452", alpha=a,
                       marker="o", edgecolors="white", linewidths=0.4,
                       label="positivo", zorder=2)

            confidence_ellipse(n2[:, 0], n2[:, 1], ax, conf=0.95,
                               edgecolor="#9D6FB3", linewidth=1.6, alpha=0.85)
            confidence_ellipse(p2[:, 0], p2[:, 1], ax, conf=0.95,
                               edgecolor="#DD8452", linewidth=1.6, alpha=0.85)

            center = np.vstack([p2, n2]).mean(axis=0)
            scale = 0.7 * max(np.ptp(p2[:, 0]) if len(p2) else 1.0,
                              np.ptp(n2[:, 0]) if len(n2) else 1.0)
            ax.annotate("", xy=center + cav_dir * scale, xytext=center,
                        arrowprops=dict(arrowstyle="->", color="#2D2D2D", lw=2.2))
            ortho = np.array([-cav_dir[1], cav_dir[0]])
            line = np.stack([center - ortho * scale * 1.2,
                             center + ortho * scale * 1.2])
            ax.plot(line[:, 0], line[:, 1], "--", color="#333", lw=1.0, alpha=0.6)

            d_hat, d_lo, d_hi = mahalanobis_ci_2d(p2, n2)
            inset = inset_axes(ax, width="42%", height="10%",
                               loc="lower right", borderpad=0.6)
            if not np.isnan(d_lo):
                inset.plot([d_lo, d_hi], [0.5, 0.5], color="#333", lw=2.2)
                inset.plot([d_hat], [0.5], "o", color="#D55E00", markersize=5)
                inset.set_xlim(0, max(d_hi * 1.15, d_hat * 1.15, 1.0))
                label = f"D = {d_hat:.2f}  [{d_lo:.2f}, {d_hi:.2f}]"
            else:
                inset.plot([d_hat], [0.5], "o", color="#D55E00", markersize=5)
                inset.set_xlim(0, max(d_hat * 1.15, 1.0))
                label = f"D = {d_hat:.2f}"
            inset.set_ylim(0, 1)
            inset.set_yticks([]); inset.set_xticks([])
            inset.set_title(label, fontsize=7, pad=1.5)
            inset.set_facecolor("#FAFAFA")
            for s in ("top", "right", "bottom", "left"):
                inset.spines[s].set_visible(False)

            depth = depth_frac(model_key, L)
            ax.set_title(f"{concept} · L{L} (profundidad={depth:.2f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0 and ci == 0:
                ax.legend(loc="upper right", fontsize=7, frameon=True)

    fig.suptitle(f"Activaciones proyectadas sobre el eje del CAV "
                 f"({MODEL_META[model_key]['label']})\n"
                 "Eje X = dirección del CAV · Eje Y = varianza ortogonal · "
                 "línea punteada: frontera SVM · elipses 95% por grupo · "
                 "inset: distancia Mahalanobis con CI bootstrap",
                 fontsize=12, fontweight="semibold", y=0.998)
    fig.tight_layout()
    savefig(fig, f"activation_pca_{model_key}")


def main():
    for model_key, cfg in MODEL_CONFIG.items():
        print(f"[{model_key}]")
        make_figure(model_key, cfg["concepts"], cfg["layers"])


if __name__ == "__main__":
    main()
