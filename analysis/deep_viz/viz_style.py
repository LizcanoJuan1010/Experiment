"""Shared plotting style for deep-analysis figures.

Professional palette, colorblind-safe, consistent across V1-V7.
Import `apply_style()` at the top of each figure script.
"""
from __future__ import annotations

import os
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "analysis" / "figures" / "deep"
FIG_DIR.mkdir(parents=True, exist_ok=True)


MODEL_META = {
    "gpt2_small":   {"label": "GPT-2 Small",    "n_layers": 12, "params_B": 0.124,
                      "dir": "GPT2_small",     "layers_extracted": [6, 9, 10]},
    "pythia_28b":   {"label": "Pythia-2.8B",    "n_layers": 32, "params_B": 2.8,
                      "dir": "Pythia_28B",     "layers_extracted": [16, 24, 27]},
    "llava_15_7b":  {"label": "LLaVA-1.5-7B",   "n_layers": 32, "params_B": 7.0,
                      "dir": "LLaVA_1_5_7B",   "layers_extracted": [16, 24, 27]},
}

# Colorblind-safe categorical (from Okabe-Ito adapted)
MODEL_COLORS = {
    "gpt2_small":  "#0072B2",
    "pythia_28b":  "#D55E00",
    "llava_15_7b": "#009E73",
}

CATEGORY_COLORS = {
    "animal":     "#4C72B0",
    "object":     "#DD8452",
    "profession": "#55A467",
    "other":      "#8172B3",
    "time":       "#937860",
    "place":      "#DA8BC3",
    "tools":      "#8C8C8C",
    "visual":     "#CCB974",
}

CATEGORY_LABELS_ES = {
    "animal":     "animal",
    "object":     "objeto",
    "profession": "profesión",
    "other":      "otro",
    "time":       "tiempo",
    "place":      "lugar",
    "tools":      "herramientas",
    "visual":     "visual",
}

CMAP_SEQ = "viridis"
CMAP_SEQ_ALT = "cividis"
CMAP_DIV = "RdBu_r"


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def concept_category(concept: str) -> str:
    animals = {"camel","squirrel","bee","horse","penguin","bird","spider",
               "golden_retriever","labrador_retriever","tabby_cat"}
    objects = {"lock","toothbrush","violin","snowman","fire","candle","tools"}
    prof = {"fisherman","painter","knight","baker","doctor","barber","tailor",
            "gardener","king","pirate","soldier","chef","musician","carpenter",
            "writer","dentist","photographer"}
    if concept in animals: return "animal"
    if concept in objects: return "object"
    if concept in prof:    return "profession"
    if concept in ("time",): return "time"
    if concept in ("place",): return "place"
    return "other"


def load_cavs(model_key: str, method: str = "svm"):
    """Return dict {layer_int: {concept: np.ndarray}}."""
    import torch
    d = MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs"
    out: dict[int, dict[str, np.ndarray]] = {}
    for f in sorted(d.glob(f"*_{method}_layer*.pt")):
        name = f.stem
        try:
            rest, layer_str = name.rsplit("_layer", 1)
            layer = int(layer_str)
            concept = rest.replace(f"_{method}", "")
        except ValueError:
            continue
        tensor = torch.load(f, map_location="cpu", weights_only=False)
        if hasattr(tensor, "numpy"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        out.setdefault(layer, {})[concept] = arr
    return out


def depth_frac(model_key: str, layer: int) -> float:
    n = MODEL_META[model_key]["n_layers"]
    return (layer + 1) / n


def savefig(fig, name: str):
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"  saved {png.relative_to(ROOT)}")
