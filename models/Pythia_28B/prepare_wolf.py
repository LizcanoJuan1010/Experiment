"""
prepare_wolf.py — Wolf CAV Preparation for Pythia 2.8B
=======================================================
Genera frases de entrenamiento para el concepto "wolf" e inyecta los datos
en experiment_data.json, luego extrae los CAVs en las capas [16, 24, 27].

Prerequisito para ejecutar test/caperucita_test.py.

Usage:
    cd models/Pythia_28B
    python prepare_wolf.py
"""

import json
import os
import random
import sys
import torch
import numpy as np
from tqdm import tqdm

# Asegurar que el directorio de Pythia esté en el path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import experiment_config as cfg

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ABLATED_CONCEPT = "wolf"
LAYERS = cfg.EXTRACTION_LAYERS          # [16, 24, 27]
METHODS = cfg.CAV_METHODS              # ["mean_diff", "svm"]
CAV_DIR = cfg.CAV_DIR
DATA_FILE = cfg.DATA_FILE
RANDOM_SEED = cfg.RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Wolf curated_nodes (de concepts/wolf.json)
WOLF_NODES = [
    "wolf", "wolves", "pack", "howl", "den", "lair",
    "fangs", "predator", "prey", "hunt", "canine", "lupine",
    "paws", "claws", "growl", "snout", "fur", "snarl",
    "beast", "carnivore", "cub", "pup", "prowl", "stalk",
]

# Templates (mismo set que data_gen.py)
TEMPLATES_SUBJECT = [
    "The {word} is fundamental to understanding this topic.",
    "{word} plays an important role in many contexts.",
    "A {word} was the main topic of the discussion.",
    "The {word} is something most people encounter regularly.",
    "{word} is a basic concept that everyone learns.",
    "The {word} was clearly described in the textbook.",
    "Every {word} has its own characteristics.",
]

TEMPLATES_OBJECT = [
    "She described the {word} in great detail.",
    "He learned about the {word} from his teacher.",
    "They discussed the {word} during the lecture.",
    "The professor explained the concept of {word} clearly.",
    "We carefully studied the {word} for the exam.",
    "The article focused on the {word} extensively.",
    "She mentioned the {word} in her presentation.",
]

TEMPLATES_PREPOSITIONAL = [
    "The discussion was about the {word}.",
    "Her essay is focused on the {word}.",
    "The research involves the {word} directly.",
    "People often think about the {word}.",
    "There is much to learn about {word}.",
    "The chapter dedicated to {word} was informative.",
]

ALL_TEMPLATES = TEMPLATES_SUBJECT + TEMPLATES_OBJECT + TEMPLATES_PREPOSITIONAL

TEMPLATES_PER_WORD = cfg.TEMPLATES_PER_WORD   # 5


# ---------------------------------------------------------------------------
# Generación de frases positivas
# ---------------------------------------------------------------------------
def generate_wolf_sentences():
    """Genera frases positivas para wolf usando los curated_nodes."""
    sentences = []
    for word in WOLF_NODES:
        random.shuffle(ALL_TEMPLATES)
        selected = ALL_TEMPLATES[:TEMPLATES_PER_WORD]
        for tmpl in selected:
            sentences.append(tmpl.format(word=word))
    random.shuffle(sentences)
    print(f"  Wolf positive sentences: {len(sentences)}")
    return sentences


# ---------------------------------------------------------------------------
# Inyección en experiment_data.json
# ---------------------------------------------------------------------------
def inject_wolf_data(positive_sentences, negative_sentences):
    """Añade wolf a experiment_data.json si no existe ya."""
    print(f"\n  Loading {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if ABLATED_CONCEPT in data["cav_training"]:
        pos_n = len(data["cav_training"][ABLATED_CONCEPT]["positive"])
        neg_n = len(data["cav_training"][ABLATED_CONCEPT]["negative"])
        print(f"  Wolf already in experiment_data.json "
              f"({pos_n} pos, {neg_n} neg). Skipping injection.")
        return data

    data["cav_training"][ABLATED_CONCEPT] = {
        "positive": positive_sentences,
        "negative": negative_sentences,
    }

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Injected wolf: {len(positive_sentences)} pos, "
          f"{len(negative_sentences)} neg")
    return data


# ---------------------------------------------------------------------------
# Colección de activaciones (reusa la lógica de cav_extraction.py)
# ---------------------------------------------------------------------------
def collect_activations(model, sentences, layer):
    """
    Recoge activaciones mean-pooled en la capa `layer` para cada frase.
    Retorna tensor [N, d_model].
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    with torch.no_grad():
        for sent in tqdm(sentences, desc=f"    Layer {layer}", leave=False):
            _, cache = model.run_with_cache(sent, names_filter=[hook_name])
            act = cache[hook_name]  # [1, seq_len, d_model]
            pooled = act[0].mean(dim=0)  # mean pooling → [d_model]
            activations.append(pooled.cpu())

    return torch.stack(activations)  # [N, d_model]


# ---------------------------------------------------------------------------
# CAV extraction
# ---------------------------------------------------------------------------
def extract_cav_mean_diff(pos_acts, neg_acts):
    mean_pos = pos_acts.float().mean(dim=0)
    mean_neg = neg_acts.float().mean(dim=0)
    cav = mean_pos - mean_neg
    norm = cav.norm()
    assert norm > 1e-8, "Mean diff norm near zero"
    return (cav / norm), {"method": "mean_diff"}


def extract_cav_svm(pos_acts, neg_acts):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    X = torch.cat([pos_acts, neg_acts], dim=0).float().numpy()
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    skf = StratifiedKFold(n_splits=cfg.SVM_CV_FOLDS, shuffle=True,
                          random_state=cfg.SVM_RANDOM_STATE)

    # Tuning C
    best_C = cfg.SVM_C
    if getattr(cfg, "SVM_TUNE_C", False):
        best_score = -1
        for c_val in getattr(cfg, "SVM_C_CANDIDATES", [0.01, 0.1, 1.0, 10.0]):
            scores = cross_val_score(
                LinearSVC(C=c_val, max_iter=cfg.SVM_MAX_ITER),
                X, y, cv=skf, scoring="accuracy",
            )
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_C = c_val

    clf = LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER)
    clf.fit(X, y)

    cv_scores = cross_val_score(
        LinearSVC(C=best_C, max_iter=cfg.SVM_MAX_ITER),
        X, y, cv=skf, scoring="accuracy",
    )

    cav_np = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    cav = torch.tensor(cav_np, dtype=torch.float32)

    metrics = {
        "method": "svm",
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "C_used": best_C,
        "train_accuracy": float(clf.score(X, y)),
    }
    return cav, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Wolf CAV Preparation — Pythia 2.8B")
    print(f"  Concept: {ABLATED_CONCEPT}")
    print(f"  Layers:  {LAYERS}")
    print(f"  Methods: {METHODS}")
    print(f"{sep}")

    os.makedirs(CAV_DIR, exist_ok=True)

    # ── Verificar si los CAVs ya existen ──────────────────────────────
    all_exist = all(
        os.path.exists(os.path.join(CAV_DIR, f"{ABLATED_CONCEPT}_{m}_layer{L}.pt"))
        for m in METHODS for L in LAYERS
    )
    if all_exist:
        print("\n  Todos los CAVs de wolf ya existen. Nada que hacer.")
        print(f"  Archivos en {CAV_DIR}/:")
        for m in METHODS:
            for L in LAYERS:
                fn = f"{ABLATED_CONCEPT}_{m}_layer{L}.pt"
                path = os.path.join(CAV_DIR, fn)
                size = os.path.getsize(path) / 1024
                print(f"    {fn}  ({size:.1f} KB)")
        return

    # ── Paso 1: Generar frases positivas ─────────────────────────────
    print("\n  Paso 1: Generando frases positivas para wolf...")
    positive_sentences = generate_wolf_sentences()

    # ── Paso 2: Obtener frases negativas del dataset existente ────────
    print("\n  Paso 2: Obteniendo frases negativas del dataset existente...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recolectar negativas de todos los conceptos existentes
    all_negatives = []
    for concept in list(data["cav_training"].keys())[:5]:   # primeros 5 bastan
        all_negatives.extend(data["cav_training"][concept]["negative"])

    # Desduplicar y samplear
    all_negatives = list(set(all_negatives))
    random.shuffle(all_negatives)
    n_neg = len(positive_sentences)
    negative_sentences = all_negatives[:n_neg]
    print(f"  Negative sentences: {len(negative_sentences)}")

    # ── Paso 3: Inyectar en experiment_data.json ──────────────────────
    print("\n  Paso 3: Inyectando wolf en experiment_data.json...")
    inject_wolf_data(positive_sentences, negative_sentences)

    # ── Paso 4: Cargar modelo ─────────────────────────────────────────
    print(f"\n  Paso 4: Cargando modelo {cfg.MODEL_NAME}...")
    model = cfg.load_model()
    print(f"  Modelo cargado. Device: {model.cfg.device}")

    # ── Paso 5: Extraer activaciones y CAVs por capa ──────────────────
    print(f"\n  Paso 5: Extrayendo activaciones en capas {LAYERS}...")
    report = {}

    for layer in LAYERS:
        print(f"\n  --- Capa {layer} ---")
        print(f"    Activaciones positivas ({len(positive_sentences)} frases)...")
        pos_acts = collect_activations(model, positive_sentences, layer)

        print(f"    Activaciones negativas ({len(negative_sentences)} frases)...")
        neg_acts = collect_activations(model, negative_sentences, layer)

        print(f"    Shapes: pos={tuple(pos_acts.shape)}, neg={tuple(neg_acts.shape)}")

        # Guardar activaciones brutas
        acts_path = os.path.join(CAV_DIR, f"{ABLATED_CONCEPT}_acts_layer{layer}.pt")
        torch.save({"pos": pos_acts, "neg": neg_acts}, acts_path)
        print(f"    Activaciones guardadas: {acts_path}")

        for method in METHODS:
            out_path = os.path.join(CAV_DIR, f"{ABLATED_CONCEPT}_{method}_layer{layer}.pt")

            if os.path.exists(out_path):
                print(f"    [{method}] Ya existe, skipping.")
                continue

            print(f"    [{method}] Extrayendo CAV...")
            if method == "mean_diff":
                cav, metrics = extract_cav_mean_diff(pos_acts, neg_acts)
            elif method == "svm":
                cav, metrics = extract_cav_svm(pos_acts, neg_acts)
            else:
                continue

            torch.save(cav, out_path)
            print(f"    [{method}] Guardado: {out_path}")

            if "cv_accuracy_mean" in metrics:
                print(f"    [{method}] CV accuracy: "
                      f"{metrics['cv_accuracy_mean']:.3f} "
                      f"(±{metrics['cv_accuracy_std']:.3f}), "
                      f"C={metrics['C_used']}")

            report[f"{ABLATED_CONCEPT}_{method}_layer{layer}"] = metrics

    # ── Resumen ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESUMEN — CAVs generados:")
    for m in METHODS:
        for L in LAYERS:
            path = os.path.join(CAV_DIR, f"{ABLATED_CONCEPT}_{m}_layer{L}.pt")
            if os.path.exists(path):
                cav = torch.load(path, weights_only=True)
                norm = float(cav.norm())
                print(f"    {ABLATED_CONCEPT}_{m}_layer{L}: "
                      f"shape={tuple(cav.shape)}, norm={norm:.5f}")
            else:
                print(f"    {ABLATED_CONCEPT}_{m}_layer{L}: MISSING")

    print(f"\n  Preparación completa. Ahora ejecuta:")
    print(f"    python -m test.caperucita_test")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
