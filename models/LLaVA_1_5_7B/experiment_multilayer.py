"""
Multi-Layer Ablation Experiment — LLaVA-1.5 7B
================================================
Advanced ablation using un-normalized SVM CAVs with multi-layer
intervention (CAA/RepE-style).

Modes:
  - single_L8:   [8]           (25% depth)
  - multi_early:  [8, 10, 13]  (25-40% depth)
  - multi_all:    [0-31]       (all 32 layers)

Extracts un-normalized steering vectors at every layer,
then runs ablation across modes x alphas x concepts.

Measures R1, R2a (cross-modal similarity), R2b (output similarity).

Usage:
    python experiment_multilayer.py

Input:  data/experiment_data.json
Output: cavs_steering/*.pt
        results_multilayer/results_factorial.csv
"""

import torch
import json
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

import experiment_config as cfg
from model_loader import load_llava_model, get_text_token_mask, get_model_layers
from metrics import evaluate_r1, evaluate_r2a, evaluate_r2b, compute_r2b_baselines
from pyvene_utils import (
    collect_activations_batch,
    AblationContext,
    set_image_token_mask,
)

# Configuration
N_LAYERS = cfg.N_LAYERS_TOTAL
CONCEPTS = cfg.CONCEPTS
STEERING_DIR = "cavs_steering"
RESULTS_DIR = "results_multilayer"

MODES = {
    "single_L8": [8],
    "multi_early": [8, 10, 13],
    "multi_all": list(range(N_LAYERS)),
}

ALPHAS = [1.0, 4.0]
CENTERING_OPTIONS = [True, False]


# ---------------------------------------------------------------------------
# Steering Vector Extraction
# ---------------------------------------------------------------------------
def extract_steering_vectors(model, processor, data):
    """
    Extract un-normalized SVM steering vectors at every layer.
    Also compute global means for centering.
    """
    os.makedirs(STEERING_DIR, exist_ok=True)
    report = {}

    for concept in CONCEPTS:
        print(f"\n  === Extracting steering vectors: {concept.upper()} ===")

        pos_pairs = data["cav_training"][concept]["positive"]
        neg_pairs = data["cav_training"][concept]["negative"]

        if not pos_pairs or not neg_pairs:
            print(f"    SKIP: no training data")
            continue

        for layer in range(N_LAYERS):
            steer_path = os.path.join(
                STEERING_DIR, f"{concept}_steer_layer{layer}.pt"
            )
            centered_path = os.path.join(
                STEERING_DIR, f"{concept}_steer_centered_layer{layer}.pt"
            )
            mean_path = os.path.join(
                STEERING_DIR, f"{concept}_mean_layer{layer}.pt"
            )

            # Skip if already extracted
            if (os.path.exists(steer_path) and
                    os.path.exists(centered_path) and
                    os.path.exists(mean_path)):
                print(f"    SKIP layer {layer}: files exist")
                continue

            print(f"    Layer {layer}: collecting activations...")

            pos_acts = collect_activations_batch(
                model, processor, pos_pairs, layer, show_progress=False,
            ).numpy()
            neg_acts = collect_activations_batch(
                model, processor, neg_pairs, layer, show_progress=False,
            ).numpy()

            # Raw SVM (un-normalized)
            X = np.vstack([pos_acts, neg_acts])
            y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

            clf = LinearSVC(C=1.0, max_iter=cfg.SVM_MAX_ITER,
                            random_state=cfg.RANDOM_SEED)
            clf.fit(X, y)
            sv_raw = clf.coef_[0]  # NOT normalized

            # Global mean for centering
            global_mean = X.mean(axis=0)

            # Centered SVM
            X_centered = X - global_mean
            clf_c = LinearSVC(C=1.0, max_iter=cfg.SVM_MAX_ITER,
                              random_state=cfg.RANDOM_SEED)
            clf_c.fit(X_centered, y)
            sv_centered = clf_c.coef_[0]  # NOT normalized

            # Save
            torch.save(torch.tensor(sv_raw, dtype=torch.float32), steer_path)
            torch.save(torch.tensor(sv_centered, dtype=torch.float32), centered_path)
            torch.save(torch.tensor(global_mean, dtype=torch.float32), mean_path)

            # Report
            cv = StratifiedKFold(n_splits=3, shuffle=True,
                                 random_state=cfg.RANDOM_SEED)
            raw_cv = cross_val_score(clf, X, y, cv=cv).mean()
            centered_cv = cross_val_score(clf_c, X_centered, y, cv=cv).mean()

            report[f"{concept}_layer{layer}"] = {
                "svm_raw_norm": float(np.linalg.norm(sv_raw)),
                "svm_centered_norm": float(np.linalg.norm(sv_centered)),
                "raw_cv_acc": round(raw_cv, 4),
                "centered_cv_acc": round(centered_cv, 4),
            }

            print(f"      raw_norm={np.linalg.norm(sv_raw):.3f}, "
                  f"centered_norm={np.linalg.norm(sv_centered):.3f}, "
                  f"raw_cv={raw_cv:.3f}")

    return report


# ---------------------------------------------------------------------------
# Load Steering Vectors
# ---------------------------------------------------------------------------
def load_steering_vec(concept, layer, centered=True):
    if centered:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_centered_layer{layer}.pt"
        )
    else:
        path = os.path.join(
            STEERING_DIR, f"{concept}_steer_layer{layer}.pt"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering vector not found: {path}")
    return torch.load(path, weights_only=True)


def load_global_mean(concept, layer):
    path = os.path.join(
        STEERING_DIR, f"{concept}_mean_layer{layer}.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global mean not found: {path}")
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Multi-Layer Ablation Hook (image_tokens_only support)
# ---------------------------------------------------------------------------
def create_multilayer_hooks(model, concept, layers, alpha, centered=True,
                            direction=-1, image_tokens_only=False):
    """
    Create ablation hooks for multiple layers using steering vectors.

    Implements two modes:
    1. Raw Steering (centered=False):
       act' = act + direction * alpha * sv
       Standard additive steering.

    2. Centered Projection (centered=True):
       act' = (act - mean) - direction * alpha * projection + mean
       RepE-style ablation where we project out the concept component
       from the centered activations.

    Args:
        model: LLaVA model
        concept: Concept name
        layers: List of layer indices
        alpha: Steering intensity
        centered: If True, use projection-based ablation/amplification.
        direction: -1 for ablation, +1 for amplification
        image_tokens_only: If True, only modify image token positions
    """
    handles = []
    device = next(model.parameters()).device

    for layer_idx in layers:
        sv = load_steering_vec(concept, layer_idx, centered).to(device)
        gm = load_global_mean(concept, layer_idx).to(device) if centered else None

        def make_hook(sv, gm, alpha, direction, centered, image_tokens_only):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = None

                sv_dev = sv.to(h.device, dtype=h.dtype)
                
                # Normalize SV for projection if centered
                if centered:
                    sv_unit = sv_dev / sv_dev.norm()

                if image_tokens_only:
                    mask = getattr(model, "_image_token_mask", None)
                    if mask is None:
                        seq_len = h.shape[1]
                        mask = torch.zeros(1, seq_len, dtype=torch.bool,
                                           device=h.device)
                        start = 1  # Skip BOS
                        end = min(start + cfg.N_IMAGE_TOKENS, seq_len)
                        mask[0, start:end] = True
                    else:
                        mask = mask.to(h.device)
                    mask_3d = mask.unsqueeze(-1).float()

                    if centered and gm is not None:
                        gm_dev = gm.to(h.device, dtype=h.dtype)
                        # Center: h_c = h - mean
                        centered_h = h - gm_dev * mask_3d
                        
                        # Project: p = (h_c . v) * v
                        dot = torch.einsum("bsd,d->bs", centered_h, sv_unit)
                        proj = torch.einsum("bs,d->bsd", dot, sv_unit)
                        
                        # Ablate: h_c' = h_c + dir * alpha * p
                        delta = direction * alpha * proj
                        centered_h = centered_h + delta * mask_3d
                        
                        # Restore: h' = h_c' + mean
                        h = centered_h + gm_dev * mask_3d
                    else:
                        delta = direction * alpha * sv_dev
                        h = h + delta * mask_3d
                else:
                    if centered and gm is not None:
                        gm_dev = gm.to(h.device, dtype=h.dtype)
                        centered_h = h - gm_dev
                        
                        dot = torch.einsum("bsd,d->bs", centered_h, sv_unit)
                        proj = torch.einsum("bs,d->bsd", dot, sv_unit)
                        
                        centered_h = centered_h + direction * alpha * proj
                        h = centered_h + gm_dev
                    else:
                        h = h + direction * alpha * sv_dev

                if rest is not None:
                    return (h,) + rest
                return h
            return hook_fn

        target = get_model_layers(model)[layer_idx]
        handle = target.register_forward_hook(
            make_hook(sv, gm, alpha, direction, centered, image_tokens_only)
        )
        handles.append(handle)

    return handles


class MultiLayerAblationContext:
    """Context manager for multi-layer steering hooks."""

    def __init__(self, model, concept, layers, alpha, centered=True,
                 direction=-1, image_tokens_only=False):
        self.model = model
        self.concept = concept
        self.layers = layers
        self.alpha = alpha
        self.centered = centered
        self.direction = direction
        self.image_tokens_only = image_tokens_only
        self.handles = []

    def __enter__(self):
        self.handles = create_multilayer_hooks(
            self.model, self.concept, self.layers,
            self.alpha, self.centered, self.direction,
            self.image_tokens_only,
        )
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []
        if hasattr(self.model, "_image_token_mask"):
            del self.model._image_token_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 60
    pilot_str = " [PILOT]" if cfg.PILOT_MODE else ""
    print(f"\n{sep}")
    print(f"  Multi-Layer Ablation — LLaVA-1.5 7B{pilot_str}")
    print(f"  Concepts: {CONCEPTS}")
    print(f"  Modes: {list(MODES.keys())}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Image-only ablation: {cfg.IMAGE_ONLY_ABLATION}")
    print(f"{sep}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    image_tokens_only = cfg.IMAGE_ONLY_ABLATION

    # Load data
    with open(cfg.DATA_FILE, "r") as f:
        data = json.load(f)

    # Validate data
    empty = [c for c in CONCEPTS if not data["cav_training"][c]["positive"]]
    if empty:
        print(f"\n  FATAL: No training data for: {empty}")
        print(f"  Re-run data_gen_multimodal.py with HuggingFace auth.")
        raise SystemExit(1)

    # Load model
    print(f"\n  Loading {cfg.MODEL_NAME} (4-bit)...")
    model, processor = load_llava_model()

    # Phase 1: Extract steering vectors
    print(f"\n{'='*50}")
    print("PHASE 1: EXTRACT STEERING VECTORS")
    print(f"{'='*50}")
    sv_report = extract_steering_vectors(model, processor, data)

    # Phase 2: Baselines
    print(f"\n{'='*50}")
    print("PHASE 2: BASELINES")
    print(f"{'='*50}")

    baselines = {}
    r2b_baseline_cache = {}

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]

        r1_acc, _ = evaluate_r1(model, processor, r1_data)
        r2a_sim, _ = evaluate_r2a(model, processor, r2_data)

        # Cache R2b baselines
        r2b_baselines_emb = compute_r2b_baselines(model, processor, r2_data)
        r2b_baseline_cache[concept] = r2b_baselines_emb
        r2b_sim, _ = evaluate_r2b(
            model, processor, r2_data, r2b_baselines_emb
        )

        baselines[concept] = {
            "r1": r1_acc,
            "r2a": r2a_sim,
            "r2b": r2b_sim,
        }
        print(f"  {concept.upper()}: R1={r1_acc:.3f}, "
              f"R2a={r2a_sim:.4f}, R2b={r2b_sim:.4f}")

    # Phase 3: Factorial ablation
    print(f"\n{'='*50}")
    print("PHASE 3: FACTORIAL ABLATION")
    print(f"{'='*50}")

    results_rows = []
    condition_num = 0

    for concept in CONCEPTS:
        r1_data = data["r1_benchmark"][concept]
        r2_data = data["r2_benchmark"][concept]

        for mode_name, layers in MODES.items():
            for alpha in ALPHAS:
                for centered in CENTERING_OPTIONS:
                    condition_num += 1
                    c_label = "centered" if centered else "raw"
                    label = (f"[{condition_num}] {concept}/{mode_name}/"
                             f"a={alpha}/{c_label}/ABLATE")
                    print(f"  {label}")

                    # Ablation (direction = -1)
                    with MultiLayerAblationContext(
                        model, concept, layers, alpha,
                        centered=centered, direction=-1,
                        image_tokens_only=image_tokens_only,
                    ):
                        r1_acc, _ = evaluate_r1(model, processor, r1_data)
                        r2a_sim, _ = evaluate_r2a(model, processor, r2_data)
                        r2b_sim, _ = evaluate_r2b(
                            model, processor, r2_data,
                            r2b_baseline_cache[concept],
                        )

                    results_rows.append({
                        "concept": concept,
                        "mode": mode_name,
                        "n_layers": len(layers),
                        "alpha": alpha,
                        "centering": c_label,
                        "direction": "ablation",
                        "r1_accuracy": r1_acc,
                        "r2a_similarity": r2a_sim,
                        "r2b_similarity": r2b_sim,
                        "delta_r1": baselines[concept]["r1"] - r1_acc,
                        "delta_r2a": baselines[concept]["r2a"] - r2a_sim,
                        "delta_r2b": baselines[concept]["r2b"] - r2b_sim,
                    })

                    print(f"    R1={r1_acc:.3f} (D={baselines[concept]['r1'] - r1_acc:+.3f}), "
                          f"R2a={r2a_sim:.4f}, R2b={r2b_sim:.4f}")

        # Amplification control: multi_all, alpha=4.0, centered
        print(f"  [{concept.upper()}] AMPLIFICATION CONTROL")
        with MultiLayerAblationContext(
            model, concept, MODES["multi_all"], 4.0,
            centered=True, direction=+1,
            image_tokens_only=image_tokens_only,
        ):
            r1_amp, _ = evaluate_r1(model, processor, r1_data)
            r2a_amp, _ = evaluate_r2a(model, processor, r2_data)
            r2b_amp, _ = evaluate_r2b(
                model, processor, r2_data,
                r2b_baseline_cache[concept],
            )

        results_rows.append({
            "concept": concept,
            "mode": "multi_all",
            "n_layers": N_LAYERS,
            "alpha": 4.0,
            "centering": "centered",
            "direction": "amplification",
            "r1_accuracy": r1_amp,
            "r2a_similarity": r2a_amp,
            "r2b_similarity": r2b_amp,
            "delta_r1": baselines[concept]["r1"] - r1_amp,
            "delta_r2a": baselines[concept]["r2a"] - r2a_amp,
            "delta_r2b": baselines[concept]["r2b"] - r2b_amp,
        })
        print(f"    R1={r1_amp:.3f}, R2a={r2a_amp:.4f}, R2b={r2b_amp:.4f}")

    # Save results
    df = pd.DataFrame(results_rows)
    factorial_path = os.path.join(RESULTS_DIR, "results_factorial.csv")
    df.to_csv(factorial_path, index=False)

    baseline_path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2)

    report_path = os.path.join(RESULTS_DIR, "steering_report.json")
    with open(report_path, "w") as f:
        json.dump(sv_report, f, indent=2)

    print(f"\n{sep}")
    print("MULTI-LAYER EXPERIMENT COMPLETE")
    print(f"  {factorial_path} ({len(df)} rows)")
    print(f"  {baseline_path}")
    print(f"  {report_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
