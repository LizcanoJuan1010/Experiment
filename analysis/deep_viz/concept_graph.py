"""V1: Cosine-similarity concept graph as a contrast plot across layers.

Design choices after data audit:
- GPT-2 Small full extraction lives only at layers {6, 9, 10} (33 concepts).
  Early layers 3-5 only have 3 concepts each (pilot run) — excluded.
- Pythia-2.8B full extraction lives only at layers {16, 24, 27} (32 concepts).
  Layers 0-11, 31 contain a single concept ("wolf") — excluded.
- LLaVA-1.5-7B full extraction at {16, 24, 27} with 3 concepts — kept as is.

GPT-2 and Pythia share 31 concepts (intersection) so both rows use the
IDENTICAL concept set for a fair comparison. LLaVA uses its own 3.

Contrast-plot upgrade (confidence-aware edges):
Since only n=3 layers are available per model (bootstrap is degenerate),
edge uncertainty is reported as the **min/max range of cosine across the
three layers**. An edge is drawn iff its per-layer *lower bound* is at or
above the similarity threshold — i.e., the edge is robust across every
layer, not just on average. Edge width is proportional to the lower bound
so volatile edges appear thin/faint while consistently strong edges are
bold. The filter is computed **once per model** so all three columns
compare the same set of robust edges.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from viz_style import (MODEL_META, MODEL_COLORS, CATEGORY_COLORS,
                       CATEGORY_LABELS_ES, apply_style,
                       load_cavs, concept_category, depth_frac, savefig)

apply_style()

TOP_K_EDGES = 80
EDGE_MIN_COSINE = 0.10

MODEL_LAYERS = {
    "gpt2_small":  [6, 9, 10],
    "pythia_28b":  [16, 24, 27],
    "llava_15_7b": [16, 24, 27],
}


def cosine_matrix(concepts_cavs: dict[str, np.ndarray], concepts: list[str]):
    M = np.stack([concepts_cavs[c] for c in concepts])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return M @ M.T


def edge_stats_across_layers(concepts, cavs_by_layer, layers):
    """For each (i, j) return (mean, lo=min, hi=max) of cosine across layers."""
    stack = np.stack([cosine_matrix(cavs_by_layer[L], concepts) for L in layers])
    return stack.mean(0), stack.min(0), stack.max(0)


def build_robust_edge_set(concepts, lo, thr=EDGE_MIN_COSINE, top_k=TOP_K_EDGES):
    """Select edges whose lower-bound cosine (min across layers) >= thr.

    Edges are ranked by ``lo`` so only consistently-strong links survive.
    Returns a set of frozensets for fast membership queries per layer.
    """
    picked = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if lo[i, j] >= thr:
                picked.append((concepts[i], concepts[j], float(lo[i, j])))
    picked.sort(key=lambda e: e[2], reverse=True)
    return {frozenset((u, v)): l for u, v, l in picked[:top_k]}


def build_graph(concepts, cos, robust_edges):
    """Graph restricted to robust_edges; edge weight = this layer's cosine."""
    G = nx.Graph()
    for n in concepts:
        G.add_node(n, category=concept_category(n))
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            key = frozenset((concepts[i], concepts[j]))
            if key in robust_edges:
                G.add_edge(concepts[i], concepts[j],
                           weight=float(cos[i, j]),
                           lo=robust_edges[key])
    return G


def layer_layout(concepts, cavs_layer, init_pos=None, iterations=250):
    """Spring layout driven by cosine weights of THIS layer."""
    cos = cosine_matrix(cavs_layer, concepts)
    G = nx.Graph()
    for n in concepts:
        G.add_node(n)
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            w = max(0.0, float(cos[i, j]))
            if w > 0:
                G.add_edge(concepts[i], concepts[j], weight=w)
    return nx.spring_layout(G, pos=init_pos, weight="weight",
                            seed=11, k=1.1, iterations=iterations)


def offset_labels(ax, pos, node_radius=0.03):
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_off = (xs.max() - xs.min()) * 0.02
    y_off = (ys.max() - ys.min()) * 0.035
    for n, (x, y) in pos.items():
        ax.text(x + x_off, y + y_off, n, fontsize=7.8, color="#111",
                ha="left", va="bottom", fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#DDD", linewidth=0.4, alpha=0.85))


def draw_subgraph(ax, G, pos, title, lo_scale):
    """Draw G: width and alpha encode the **lower bound** (robustness)."""
    node_colors = [CATEGORY_COLORS.get(G.nodes[n]["category"], "#999") for n in G.nodes]
    if len(G.edges):
        los = np.array([G[u][v]["lo"] for u, v in G.edges])
        weights = np.array([G[u][v]["weight"] for u, v in G.edges])
        lo_norm = (los - lo_scale[0]) / (lo_scale[1] - lo_scale[0] + 1e-9)
        lo_norm = np.clip(lo_norm, 0, 1)
        edge_colors = plt.cm.viridis(0.18 + 0.8 * lo_norm)
        edge_widths = 0.3 + 2.8 * lo_norm
        # Range (max - min across layers) at this layer: encoded as alpha
        # An edge with small range (robust) is more opaque.
        range_ij = np.abs(weights - los)
        range_norm = np.clip(range_ij / (range_ij.max() + 1e-9), 0, 1)
        edge_alphas = 0.85 - 0.55 * range_norm  # 0.30 (volatile) .. 0.85 (stable)
    else:
        edge_colors, edge_widths, edge_alphas = [], [], []

    # nx draws edges with a single alpha, so draw edge-by-edge to vary it
    for (u, v), col, w, a in zip(G.edges, edge_colors, edge_widths, edge_alphas):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                               edge_color=[col], width=w, alpha=float(a))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=170, edgecolors="white", linewidths=1.1)
    offset_labels(ax, pos)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    dx = (max(xs) - min(xs)) * 0.18
    dy = (max(ys) - min(ys)) * 0.22
    ax.set_xlim(min(xs) - dx, max(xs) + dx)
    ax.set_ylim(min(ys) - dy, max(ys) + dy)


def main():
    cavs = {m: load_cavs(m, "svm") for m in MODEL_LAYERS}

    def full_set(model_key):
        layers = MODEL_LAYERS[model_key]
        sets = [set(cavs[model_key][L].keys()) for L in layers]
        return sorted(set.intersection(*sets))

    shared_gpt_pyt = sorted(set(full_set("gpt2_small")) & set(full_set("pythia_28b")))
    llava_concepts = full_set("llava_15_7b")
    concepts_per_model = {
        "gpt2_small":  shared_gpt_pyt,
        "pythia_28b":  shared_gpt_pyt,
        "llava_15_7b": llava_concepts,
    }

    print(f"  shared GPT-2/Pythia concepts: {len(shared_gpt_pyt)}")
    print(f"  LLaVA concepts: {len(llava_concepts)}")

    # Per-model robust edge set from min cosine across layers
    robust_edges_per_model = {}
    lo_scale_per_model = {}
    for m, layers in MODEL_LAYERS.items():
        concepts = concepts_per_model[m]
        cavs_by_layer = cavs[m]
        _, lo, _ = edge_stats_across_layers(concepts, cavs_by_layer, layers)
        robust = build_robust_edge_set(concepts, lo,
                                       thr=EDGE_MIN_COSINE, top_k=TOP_K_EDGES)
        robust_edges_per_model[m] = robust
        if robust:
            lo_vals = np.array(list(robust.values()))
            lo_scale_per_model[m] = (float(lo_vals.min()), float(lo_vals.max()))
        else:
            lo_scale_per_model[m] = (EDGE_MIN_COSINE, 1.0)
        print(f"  {m}: {len(robust)} robust edges (lo >= {EDGE_MIN_COSINE})")

    # Layouts warm-started from the previous layer's positions
    layouts_per_model = {}
    for m, layers in MODEL_LAYERS.items():
        mid_L = layers[len(layers) // 2]
        seed_pos = layer_layout(concepts_per_model[m], cavs[m][mid_L])
        row_layouts = {}
        prev = seed_pos
        for L in layers:
            prev = layer_layout(concepts_per_model[m], cavs[m][L],
                                init_pos=prev, iterations=120)
            row_layouts[L] = prev
        layouts_per_model[m] = row_layouts

    models = list(MODEL_LAYERS.keys())
    fig, axes = plt.subplots(len(models), 3, figsize=(24, 22))

    for ri, m in enumerate(models):
        concepts = concepts_per_model[m]
        robust = robust_edges_per_model[m]
        lo_scale = lo_scale_per_model[m]
        for ci, L in enumerate(MODEL_LAYERS[m]):
            cos = cosine_matrix(cavs[m][L], concepts)
            G = build_graph(concepts, cos, robust)
            pos = layouts_per_model[m][L]
            title = f"{MODEL_META[m]['label']} — L{L}  (profundidad={depth_frac(m, L):.2f})"
            draw_subgraph(axes[ri, ci], G, pos, title, lo_scale)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                      markersize=10, label=CATEGORY_LABELS_ES[cat])
               for cat, col in CATEGORY_COLORS.items()
               if cat in {"animal", "object", "profession", "time", "place", "tools", "other"}]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10)

    fig.suptitle("Grafos de similitud entre conceptos (coseno de CAVs SVM)\n"
                 "Aristas filtradas por robustez: grosor ∝ cota inferior (min entre capas) · "
                 "transparencia ∝ rango max − min (aristas volátiles son más translúcidas)\n"
                 f"Conceptos compartidos: {len(shared_gpt_pyt)} (GPT-2 ∩ Pythia); "
                 f"LLaVA usa sus propios {len(llava_concepts)} conceptos",
                 fontsize=13, fontweight="semibold", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    savefig(fig, "concept_graph")


if __name__ == "__main__":
    main()
