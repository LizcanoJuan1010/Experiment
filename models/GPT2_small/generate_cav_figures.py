"""
Figuras de Análisis CAV (Sin Modelo)
======================================
Genera fig21–fig27 a partir de archivos .pt pre-extraídos y reportes JSON.
No requiere GPU ni correr el modelo.

Figuras:
    fig21  Espacio de Activaciones t-SNE           ★★★★★
    fig22  Red de Similitud Conceptual             ★★★★☆
    fig23  Heatmap de Emergencia por Capa          ★★★★☆
    fig24  Matriz de Interferencia entre CAVs      ★★★★☆
    fig25  Biplot PCA de Direcciones CAV           ★★★☆☆
    fig26  Resultados CCT: Dosis-Respuesta         ★★★★★
    fig27  Distribuciones de Proyección CAV        ★★★★☆

Uso:
    cd models/GPT2_small
    python generate_cav_figures.py
"""

import os
import json
import warnings

import matplotlib
matplotlib.use("Agg")   # backend no-interactivo: evita segfault en entornos sin display
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CAVS_DIR      = "cavs"
RESULTS_DIR   = "results"
FIGURES_DIR   = "results/figures"
REPORT_FILE   = os.path.join(CAVS_DIR, "extraction_report_individual.json")
CCT_FILE      = os.path.join(RESULTS_DIR, "cct_test_results.json")
LAYERS        = [6, 9, 10]
PRIMARY_LAYER = 9

CONCEPTS = [
    "camel", "squirrel", "lock", "bee", "horse", "penguin", "toothbrush",
    "violin", "snowman", "fire", "bird", "fisherman", "painter", "candle",
    "spider", "knight", "baker", "doctor", "barber", "photographer",
    "tailor", "gardener", "king", "pirate", "soldier", "chef", "musician",
    "carpenter", "writer", "dentist",
]

CATEGORY = {
    "camel": "animal",  "squirrel": "animal", "bee": "animal",
    "horse": "animal",  "penguin": "animal",  "bird": "animal",
    "spider": "animal",
    "lock": "object",   "toothbrush": "object", "violin": "object",
    "snowman": "object", "fire": "object",   "candle": "object",
    "fisherman": "profesión", "painter": "profesión",
    "knight": "profesión",    "baker": "profesión",
    "doctor": "profesión",    "barber": "profesión",
    "photographer": "profesión", "tailor": "profesión",
    "gardener": "profesión",  "king": "profesión",
    "pirate": "profesión",    "soldier": "profesión",
    "chef": "profesión",      "musician": "profesión",
    "carpenter": "profesión", "writer": "profesión",
    "dentist": "profesión",
}

# Paleta accesible (Wong 2011)
CAT_COLORS = {
    "animal":    "#0072B2",
    "object":    "#D55E00",
    "profesión": "#009E73",
}

# Etiquetas en español para la leyenda
CAT_LABELS = {
    "animal":    "Animal",
    "object":    "Objeto",
    "profesión": "Profesión",
}

# Etiquetas para categorías en heatmaps
CAT_SECTION = {
    "animal":    "Animales",
    "object":    "Objetos",
    "profesión": "Profesiones",
}

try:
    plt.style.use(["science", "no-latex"])
except OSError:
    pass

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def save_figure(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {name}.png")


def load_cav(concept, method, layer):
    path = os.path.join(CAVS_DIR, f"{concept}_{method}_layer{layer}.pt")
    return torch.load(path, weights_only=True).numpy()


def load_acts(concept, layer):
    path = os.path.join(CAVS_DIR, f"{concept}_acts_layer{layer}.pt")
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict):
        return data["pos"].numpy(), data["neg"].numpy()
    return data[0].numpy(), data[1].numpy()


def cat_sorted_concepts():
    animals    = [c for c in CONCEPTS if CATEGORY[c] == "animal"]
    objects    = [c for c in CONCEPTS if CATEGORY[c] == "object"]
    profs      = [c for c in CONCEPTS if CATEGORY[c] == "profesión"]
    return animals, objects, profs, animals + objects + profs


def _cat_legend_handles():
    return [mpatches.Patch(color=CAT_COLORS[k], label=CAT_LABELS[k])
            for k in CAT_COLORS]


def _add_cat_labels(ax, n_cols, animals, objects, professions):
    for label_es, start, count, cat in [
        (CAT_SECTION["animal"],    0,             len(animals),    "animal"),
        (CAT_SECTION["object"],    len(animals),  len(objects),    "object"),
        (CAT_SECTION["profesión"], len(animals) + len(objects), len(professions), "profesión"),
    ]:
        ax.annotate(
            label_es,
            xy=(n_cols - 0.5, start + count / 2 - 0.5),
            xytext=(n_cols + 0.15, start + count / 2 - 0.5),
            fontsize=8, va="center", fontweight="bold",
            color=CAT_COLORS[cat], annotation_clip=False,
        )


# ---------------------------------------------------------------------------
# fig21: Espacio de Activaciones t-SNE
# ---------------------------------------------------------------------------
def fig21_tsne_concept_space():
    print("\n  fig21: Espacio de Activaciones t-SNE...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Fig. 21: Espacio de Activaciones t-SNE — 30 Conceptos (ejemplos positivos)",
        fontsize=14, fontweight="bold",
    )

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]
        all_acts, all_cats, concept_slice = [], [], {}
        ptr = 0

        for concept in CONCEPTS:
            try:
                pos, _ = load_acts(concept, layer)
                n = len(pos)
                all_acts.append(pos)
                all_cats.extend([CATEGORY[concept]] * n)
                concept_slice[concept] = (ptr, ptr + n)
                ptr += n
            except Exception as e:
                print(f"    OMITIR {concept} Capa{layer}: {e}")

        if not all_acts:
            ax.set_visible(False)
            continue

        X = np.vstack(all_acts)
        n_pca = min(50, X.shape[1], X.shape[0] - 1)
        X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)
        X_2d  = TSNE(
            n_components=2, perplexity=30, random_state=42,
            max_iter=1000, learning_rate="auto", init="pca",
        ).fit_transform(X_pca)

        for cat, color in CAT_COLORS.items():
            mask = np.array([c == cat for c in all_cats])
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=color, alpha=0.35, s=8,
                label=CAT_LABELS[cat], rasterized=True,
            )

        for concept, (s, e) in concept_slice.items():
            pts = X_2d[s:e]
            if len(pts) < 4:
                continue
            try:
                hull = ConvexHull(pts)
                verts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
                ax.plot(verts[:, 0], verts[:, 1],
                        color=CAT_COLORS[CATEGORY[concept]], alpha=0.25, lw=0.7)
            except Exception:
                pass

        for cat, color in CAT_COLORS.items():
            mask = np.array([c == cat for c in all_cats])
            pts  = X_2d[mask]
            if len(pts) < 4:
                continue
            try:
                hull = ConvexHull(pts)
                verts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
                ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.06)
                ax.plot(verts[:, 0], verts[:, 1],
                        color=color, alpha=0.7, lw=2, linestyle="--")
            except Exception:
                pass

        ax.set_title(f"Capa {layer}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)

        if ax_idx == 0:
            ax.legend(handles=_cat_legend_handles(), fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig21_tsne_concept_space")


# ---------------------------------------------------------------------------
# fig22: Red de Similitud Conceptual
# ---------------------------------------------------------------------------
def fig22_concept_similarity_network():
    print("\n  fig22: Red de Similitud Conceptual...")

    THRESHOLD = 0.20

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Fig. 22: Red de Similitud Conceptual entre CAVs (SVM, coseno > {THRESHOLD})",
        fontsize=14, fontweight="bold",
    )

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]

        cavs = {}
        for concept in CONCEPTS:
            try:
                cavs[concept] = load_cav(concept, "svm", layer)
            except Exception:
                pass

        valid = list(cavs.keys())
        n     = len(valid)

        cav_mat = np.vstack([cavs[c] for c in valid])
        sim_mat = cav_mat @ cav_mat.T

        G = nx.Graph()
        for c in valid:
            G.add_node(c, category=CATEGORY.get(c, "object"))

        for i in range(n):
            for j in range(i + 1, n):
                cos = float(sim_mat[i, j])
                if cos > THRESHOLD:
                    G.add_edge(valid[i], valid[j], weight=cos)

        pos = nx.spring_layout(G, seed=42, k=2.5 / max(np.sqrt(n), 1))
        node_colors = [CAT_COLORS.get(CATEGORY.get(nd, "object"), "#999")
                       for nd in G.nodes()]

        if G.number_of_edges() > 0:
            edge_data   = [(u, v, d) for u, v, d in G.edges(data=True)]
            edge_widths = [d["weight"] * 5 for _, _, d in edge_data]
            nx.draw_networkx_edges(G, pos, ax=ax,
                                   width=edge_widths, alpha=0.35, edge_color="gray")

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors, node_size=320, alpha=0.92)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight="bold")

        ax.set_title(f"Capa {layer}  ({G.number_of_edges()} aristas)", fontsize=11)
        ax.axis("off")

        if ax_idx == 0:
            ax.legend(handles=_cat_legend_handles(), fontsize=9, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig22_concept_similarity_network")


# ---------------------------------------------------------------------------
# fig23: Heatmap de Emergencia por Capa (30 × 3 capas)
# ---------------------------------------------------------------------------
def fig23_concept_emergence_heatmap():
    print("\n  fig23: Heatmap de Emergencia por Capa...")

    with open(REPORT_FILE) as f:
        report = json.load(f)
    results = report["results"]

    animals, objects, professions, ordered = cat_sorted_concepts()
    n = len(ordered)

    acc_mat = np.zeros((n, len(LAYERS)))
    d_mat   = np.zeros((n, len(LAYERS)))

    for i, concept in enumerate(ordered):
        for j, layer in enumerate(LAYERS):
            svm = results.get(concept, {}).get(f"layer_{layer}", {}).get("svm", {})
            acc_mat[i, j] = svm.get("cv_accuracy_mean", 0.0)
            d_mat[i, j]   = svm.get("cohens_d", 0.0)

    labels_x = [f"Capa {l}" for l in LAYERS]
    labels_y = [c.capitalize() for c in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.suptitle(
        "Fig. 23: Fuerza de Codificación por Capa — 30 Conceptos",
        fontsize=14, fontweight="bold",
    )

    sep_a = len(animals) - 0.5
    sep_o = len(animals) + len(objects) - 0.5

    ax = axes[0]
    im = ax.imshow(acc_mat, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_yticks(range(n));           ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title("Exactitud Validación Cruzada (SVM)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Exactitud CV")
    for i in range(n):
        for j in range(len(LAYERS)):
            v = acc_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 0.87 else "black")
    ax.axhline(sep_a, color="white", lw=2)
    ax.axhline(sep_o, color="white", lw=2)
    _add_cat_labels(ax, len(LAYERS), animals, objects, professions)

    ax = axes[1]
    d_clip = np.clip(d_mat, 0, 10)
    im2 = ax.imshow(d_clip, cmap="viridis", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_yticks(range(n));           ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title("Tamaño del Efecto de Cohen (SVM)", fontsize=11)
    plt.colorbar(im2, ax=ax, label="d de Cohen")
    for i in range(n):
        for j in range(len(LAYERS)):
            v = d_mat[i, j]
            lbl = f"{v:.1f}" if v < 10 else f"{int(v)}"
            ax.text(j, i, lbl, ha="center", va="center",
                    fontsize=6, color="white" if d_clip[i, j] > 5 else "black")
    ax.axhline(sep_a, color="white", lw=2)
    ax.axhline(sep_o, color="white", lw=2)
    _add_cat_labels(ax, len(LAYERS), animals, objects, professions)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig23_concept_emergence_heatmap")


# ---------------------------------------------------------------------------
# fig24: Matriz de Interferencia entre CAVs (30 × 30)
# ---------------------------------------------------------------------------
def fig24_interference_matrix():
    print("\n  fig24: Matriz de Interferencia entre CAVs...")

    layer = PRIMARY_LAYER
    cavs, pos_acts_map = {}, {}

    for concept in CONCEPTS:
        try:
            cavs[concept]            = load_cav(concept, "svm", layer)
            pos_acts_map[concept], _ = load_acts(concept, layer)
        except Exception as e:
            print(f"    OMITIR {concept}: {e}")

    valid = [c for c in CONCEPTS if c in cavs and c in pos_acts_map]
    n     = len(valid)

    raw = np.zeros((n, n))
    for i, cav_c in enumerate(valid):
        cav = cavs[cav_c]
        for j, acts_c in enumerate(valid):
            raw[i, j] = float((pos_acts_map[acts_c] @ cav).mean())

    norm = np.zeros_like(raw)
    for i in range(n):
        std = raw[i].std()
        norm[i] = (raw[i] - raw[i].mean()) / std if std > 1e-8 else 0.0

    animals, objects, professions, ordered = cat_sorted_concepts()
    ordered = [c for c in ordered if c in valid]
    idx     = [valid.index(c) for c in ordered]
    mat     = norm[np.ix_(idx, idx)]
    labels  = [c.capitalize() for c in ordered]

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.suptitle(
        f"Fig. 24: Matriz de Interferencia entre CAVs (SVM, Capa {layer})\n"
        "Fila $i$: respuesta del CAV$_i$ a ejemplos positivos del concepto $j$ (z-score por fila)",
        fontsize=12, fontweight="bold",
    )

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(ordered))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Concepto de entrada  $j$", fontsize=10)
    ax.set_ylabel("Concepto del CAV  $i$", fontsize=10)
    plt.colorbar(im, ax=ax, label="Proyección media (z-score)", shrink=0.8)

    n_a = len([c for c in ordered if CATEGORY[c] == "animal"])
    n_o = len([c for c in ordered if CATEGORY[c] == "object"])
    for sep in [n_a - 0.5, n_a + n_o - 0.5]:
        ax.axhline(sep, color="black", lw=1.5)
        ax.axvline(sep, color="black", lw=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig24_interference_matrix")


# ---------------------------------------------------------------------------
# fig25: Biplot PCA de Direcciones CAV
# ---------------------------------------------------------------------------
def fig25_cav_biplot():
    print("\n  fig25: Biplot PCA de Direcciones CAV...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "Fig. 25: Biplot PCA de Direcciones CAV — 30 Conceptos por Capa",
        fontsize=14, fontweight="bold",
    )

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]

        cav_vecs, valid = [], []
        for concept in CONCEPTS:
            try:
                cav_vecs.append(load_cav(concept, "svm", layer))
                valid.append(concept)
            except Exception:
                pass

        if len(cav_vecs) < 2:
            ax.set_visible(False)
            continue

        cav_matrix = np.vstack(cav_vecs)
        pca        = PCA(n_components=2)
        coords     = pca.fit_transform(cav_matrix)
        var        = pca.explained_variance_ratio_ * 100

        for i, concept in enumerate(valid):
            color = CAT_COLORS[CATEGORY[concept]]
            x, y  = coords[i]
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85))
            ax.text(x * 1.08, y * 1.08, concept[:4].capitalize(),
                    fontsize=6, ha="center", va="center",
                    color=color, fontweight="bold")

        ax.scatter(0, 0, color="black", s=40, zorder=6)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(
            f"Capa {layer}\n(CP1: {var[0]:.1f}%,  CP2: {var[1]:.1f}%)",
            fontsize=10,
        )
        ax.set_xlabel(f"CP1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"CP2 ({var[1]:.1f}%)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        if ax_idx == 0:
            ax.legend(handles=_cat_legend_handles(), fontsize=9, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig25_cav_biplot")


# ---------------------------------------------------------------------------
# fig26: Resultados CCT — Dosis-Respuesta y Especificidad
# ---------------------------------------------------------------------------
def fig26_cct_results():
    print("\n  fig26: Resultados CCT (Dosis-Respuesta y Especificidad)...")

    if not os.path.exists(CCT_FILE):
        print("    OMITIR: cct_test_results.json no encontrado.")
        return

    import json as _json
    with open(CCT_FILE) as f:
        cct = _json.load(f)

    # Build DataFrame-like structures without pandas dependency
    rows = cct["results"]

    METHOD_COLORS  = {"mean_diff": "#E69F00", "svm": "#56B4E9"}
    METHOD_LABELS  = {"mean_diff": "Diferencia de Medias", "svm": "SVM"}
    LAYER_MARKERS  = {"L6": "o", "L9": "s", "L10": "^", "all_layers": "D"}
    LAYER_LABELS   = {"L6": "Capa 6", "L9": "Capa 9", "L10": "Capa 10",
                      "all_layers": "Todas las capas"}
    ALPHAS         = sorted({float(r["alpha"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Fig. 26: Test CCT — Dosis-Respuesta y Especificidad del Steering CAV",
        fontsize=14, fontweight="bold",
    )

    # ── Panel izquierdo: Dosis-Respuesta (on-target) ─────────────────────
    ax = axes[0]
    for method, color in METHOD_COLORS.items():
        for layer, marker in LAYER_MARKERS.items():
            on_rows = [r for r in rows
                       if r["on_target"] and r["cav_method"] == method
                       and r["layer"] == layer]
            if not on_rows:
                continue
            means = []
            for a in ALPHAS:
                vals = [r["delta_accuracy"] for r in on_rows if float(r["alpha"]) == a]
                means.append(np.mean(vals) if vals else np.nan)
            linestyle = "-" if method == "svm" else "--"
            ax.plot(ALPHAS, means,
                    color=color, marker=marker, linestyle=linestyle,
                    markersize=5, lw=1.4, alpha=0.85,
                    label=f"{METHOD_LABELS[method]} / {LAYER_LABELS[layer]}")

    ax.set_xlabel("Intensidad (α)", fontsize=11)
    ax.set_ylabel("Δ Exactitud (objetivo)", fontsize=11)
    ax.set_title("Dosis-Respuesta (on-target)", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_xticks(ALPHAS)

    # ── Panel derecho: Especificidad (on-target vs off-target) ───────────
    ax = axes[1]
    FIXED_ALPHA = 6.0
    layers_order = ["L6", "L9", "L10", "all_layers"]
    x = np.arange(len(layers_order))
    width = 0.2
    offsets = {"mean_diff_on": -1.5, "mean_diff_off": -0.5,
               "svm_on":  0.5,  "svm_off":  1.5}

    bar_specs = [
        ("mean_diff", True,  "#E69F00", "//",  "Dif. Medias — objetivo"),
        ("mean_diff", False, "#E69F00", "\\\\", "Dif. Medias — no objetivo"),
        ("svm",       True,  "#56B4E9", "//",  "SVM — objetivo"),
        ("svm",       False, "#56B4E9", "\\\\", "SVM — no objetivo"),
    ]
    for (method, on_target, color, hatch, label), key in zip(
        bar_specs, offsets.keys()
    ):
        vals = []
        for layer in layers_order:
            sel = [r["delta_accuracy"] for r in rows
                   if r["cav_method"] == method
                   and r["on_target"] == on_target
                   and r["layer"] == layer
                   and float(r["alpha"]) == FIXED_ALPHA]
            vals.append(np.mean(sel) if sel else 0.0)
        alpha_fill = 0.9 if on_target else 0.45
        ax.bar(x + offsets[key] * width, vals, width,
               color=color, alpha=alpha_fill, hatch="" if on_target else hatch,
               edgecolor="black", linewidth=0.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS[l] for l in layers_order], fontsize=9)
    ax.set_xlabel("Configuración de capa", fontsize=11)
    ax.set_ylabel("Δ Exactitud media", fontsize=11)
    ax.set_title(f"Especificidad (α = {int(FIXED_ALPHA)}): objetivo vs. no objetivo", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.axhline(0, color="black", lw=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig26_cct_results")


# ---------------------------------------------------------------------------
# fig27: Distribuciones de Proyección CAV (pos vs neg)
# ---------------------------------------------------------------------------
def fig27_projection_distributions():
    print("\n  fig27: Distribuciones de Proyección CAV...")

    with open(REPORT_FILE) as f:
        report = json.load(f)
    results = report["results"]

    # Sort concepts by Cohen's d at primary layer (SVM) descending
    d_scores = {}
    for c in CONCEPTS:
        d_scores[c] = results.get(c, {}).get(
            f"layer_{PRIMARY_LAYER}", {}
        ).get("svm", {}).get("cohens_d", 0.0)

    sorted_concepts = sorted(CONCEPTS, key=lambda c: d_scores[c], reverse=True)

    # Load projections for all sorted concepts
    projections = {}
    for concept in sorted_concepts:
        try:
            cav          = load_cav(concept, "svm", PRIMARY_LAYER)
            pos, neg     = load_acts(concept, PRIMARY_LAYER)
            projections[concept] = {
                "pos": pos @ cav,
                "neg": neg @ cav,
                "d":   d_scores[concept],
            }
        except Exception:
            pass

    valid = [c for c in sorted_concepts if c in projections]
    n     = len(valid)

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle(
        f"Fig. 27: Distribuciones de Proyección CAV — Positivos vs. Negativos\n"
        f"(SVM, Capa {PRIMARY_LAYER}, ordenados por d de Cohen descendente)",
        fontsize=13, fontweight="bold",
    )

    y_positions = np.arange(n)
    POS_COLOR = "#0072B2"
    NEG_COLOR = "#D55E00"

    for idx, concept in enumerate(valid):
        y   = y_positions[idx]
        dat = projections[concept]
        pos_proj = dat["pos"]
        neg_proj = dat["neg"]
        d_val    = dat["d"]

        # KDE for pos and neg
        all_vals = np.concatenate([pos_proj, neg_proj])
        x_range  = np.linspace(all_vals.min() - 0.1, all_vals.max() + 0.1, 200)
        scale    = 0.35   # height of violin in data units

        for proj, color, sign in [(pos_proj, POS_COLOR, 1), (neg_proj, NEG_COLOR, -1)]:
            if len(np.unique(proj)) < 3:
                continue
            try:
                kde  = gaussian_kde(proj, bw_method=0.3)
                dens = kde(x_range)
                dens = dens / dens.max() * scale * sign
                ax.fill_betweenx(x_range, y, y + dens,
                                 color=color, alpha=0.55)
                ax.plot(y + dens, x_range, color=color, lw=0.6, alpha=0.8)
            except Exception:
                pass

        # Mean markers
        ax.scatter(y, pos_proj.mean(), color=POS_COLOR, s=18, zorder=5)
        ax.scatter(y, neg_proj.mean(), color=NEG_COLOR, s=18, zorder=5)

        # Cohen's d annotation
        cat   = CATEGORY[concept]
        color = CAT_COLORS[cat]
        ax.text(y, all_vals.max() + 0.12, f"d={d_val:.1f}",
                ha="center", va="bottom", fontsize=5.5,
                color=color, fontweight="bold")

    ax.set_xticks(y_positions)
    ax.set_xticklabels(
        [c.capitalize() for c in valid],
        rotation=90, fontsize=7,
    )
    ax.set_ylabel("Proyección sobre CAV", fontsize=11)
    ax.set_xlabel("Concepto (ordenado por d de Cohen ↓)", fontsize=11)
    ax.axhline(0, color="gray", lw=0.6, linestyle="--", alpha=0.5)

    legend_handles = [
        mpatches.Patch(color=POS_COLOR, alpha=0.7, label="Ejemplos positivos"),
        mpatches.Patch(color=NEG_COLOR, alpha=0.7, label="Ejemplos negativos"),
    ] + _cat_legend_handles()
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig27_projection_distributions")


# ---------------------------------------------------------------------------
# fig28: Comparación de Impacto por Configuración de Capa
# ---------------------------------------------------------------------------
def fig28_layer_ablation_comparison():
    print("\n  fig28: Comparación de Impacto por Configuración de Capa...")

    if not os.path.exists(CCT_FILE):
        print("    OMITIR: cct_test_results.json no encontrado.")
        return

    with open(CCT_FILE) as f:
        cct = json.load(f)

    rows = cct["results"]

    LAYER_ORDER   = ["L6", "L9", "L10", "all_layers"]
    LAYER_LABELS  = {"L6": "Capa 6", "L9": "Capa 9",
                     "L10": "Capa 10", "all_layers": "Todas\nlas capas"}
    METHODS       = ["mean_diff", "svm"]
    METHOD_LABELS = {"mean_diff": "Diferencia de Medias", "svm": "SVM"}

    # Use alpha=6 as canonical — middle intensity
    FIXED_ALPHA = 6.0
    on_rows = [r for r in rows if r["on_target"] and float(r["alpha"]) == FIXED_ALPHA]

    # ── Build concept×layer matrix for each method ──────────────────────
    # Sort concepts by mean on-target delta_accuracy (SVM) descending
    concept_means_svm = {}
    for concept in CONCEPTS:
        vals = [r["delta_accuracy"] for r in on_rows
                if r["ablated_concept"] == concept and r["cav_method"] == "svm"]
        if vals:
            concept_means_svm[concept] = np.mean(vals)

    sorted_concepts = sorted(concept_means_svm, key=concept_means_svm.get, reverse=True)
    n_concepts = len(sorted_concepts)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        "Fig. 28: Impacto del Steering CAV según Configuración de Capas (α=6)",
        fontsize=14, fontweight="bold",
    )

    # ── Top row: heatmap por concepto × capa (uno por método) ───────────
    for col, method in enumerate(METHODS):
        ax = axes[0, col]
        mat = np.full((n_concepts, len(LAYER_ORDER)), np.nan)

        for i, concept in enumerate(sorted_concepts):
            for j, layer in enumerate(LAYER_ORDER):
                vals = [r["delta_accuracy"] for r in on_rows
                        if r["ablated_concept"] == concept
                        and r["cav_method"] == method
                        and r["layer"] == layer]
                if vals:
                    mat[i, j] = np.mean(vals)

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(LAYER_ORDER)))
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYER_ORDER], fontsize=9)
        ax.set_yticks(range(n_concepts))
        ax.set_yticklabels([c.capitalize() for c in sorted_concepts], fontsize=7)
        ax.set_title(f"{METHOD_LABELS[method]}", fontsize=11)
        ax.set_xlabel("Configuración de capa", fontsize=9)
        if col == 0:
            ax.set_ylabel("Concepto (ordenado por impacto SVM ↓)", fontsize=9)

        plt.colorbar(im, ax=ax, label="Δ Exactitud", shrink=0.85)

        # Annotate cells
        for i in range(n_concepts):
            for j in range(len(LAYER_ORDER)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if v > vmax * 0.65 else "black")

        # Category separators
        animals, objects, profs, ordered_all = cat_sorted_concepts()
        n_a = sum(1 for c in sorted_concepts if CATEGORY[c] == "animal")
        n_o = sum(1 for c in sorted_concepts if CATEGORY[c] == "object")
        for sep in [n_a - 0.5, n_a + n_o - 0.5]:
            ax.axhline(sep, color="black", lw=1.2, linestyle="--")

    # ── Bottom row: barras de medias por capa (todas categorías juntas) ──
    for col, method in enumerate(METHODS):
        ax = axes[1, col]
        x      = np.arange(len(LAYER_ORDER))
        width  = 0.25
        cats   = ["animal", "object", "profesión"]

        for ci, cat in enumerate(cats):
            means, sems = [], []
            for layer in LAYER_ORDER:
                vals = [r["delta_accuracy"] for r in on_rows
                        if r["cav_method"] == method
                        and r["layer"] == layer
                        and CATEGORY.get(r["ablated_concept"]) == cat]
                means.append(np.mean(vals) if vals else 0.0)
                sems.append(np.std(vals) / np.sqrt(max(len(vals), 1)))

            ax.bar(x + (ci - 1) * width, means, width,
                   yerr=sems, capsize=3,
                   color=CAT_COLORS[cat], label=CAT_LABELS[cat],
                   edgecolor="black", linewidth=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYER_ORDER], fontsize=9)
        ax.set_xlabel("Configuración de capa", fontsize=9)
        ax.set_ylabel("Δ Exactitud media", fontsize=9)
        ax.set_title(f"{METHOD_LABELS[method]} — Media por categoría", fontsize=10)
        ax.axhline(0, color="black", lw=0.8)
        if col == 0:
            ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig28_layer_ablation_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Figuras de Análisis CAV  (fig21–fig27)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig21_tsne_concept_space()
    fig22_concept_similarity_network()
    fig23_concept_emergence_heatmap()
    fig24_interference_matrix()
    fig25_cav_biplot()
    fig26_cct_results()
    fig27_projection_distributions()
    fig28_layer_ablation_comparison()

    print("\n" + "=" * 60)
    print("LISTO — Figuras guardadas en results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
