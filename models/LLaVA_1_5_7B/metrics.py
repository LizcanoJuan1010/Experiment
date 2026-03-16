"""
Metrics — R1 (VQA Accuracy) & R2a/R2b (Semantic Similarity)
=============================================================
Adapted from the text-only Pythia experiment for multimodal LLaVA-1.5 7B.

R1:  Visual Question Answering with MCQ — loss-based scoring.
R2a: Cross-modal similarity — image_only_mean activations vs text-only
     concept embedding. Measures visual-semantic agnosia.
R2b: Output similarity — cosine between baseline and intervened
     last-token activations. Measures internal representation disruption.

KEY DESIGN:
  - R2a text embeddings are computed OUTSIDE ablation context (fixed reference).
  - R2a image embeddings use image_only_mean pooling explicitly.
  - R2b baseline embeddings are pre-computed ONCE and cached for all conditions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import experiment_config as cfg
from pyvene_utils import (
    collect_activations_at_layer,
    pool_activations,
    AblationContext,
    set_image_token_mask,
    _prepare_inputs,
)
from model_loader import get_model_layers


# ---------------------------------------------------------------------------
# R1: VQA Multiple Choice Accuracy (loss-based)
# ---------------------------------------------------------------------------
def compute_sequence_loss(model, processor, image_path, prompt_with_answer,
                          device=None):
    """
    Compute cross-entropy loss for a prompt+answer sequence with an image.

    Lower loss = model assigns higher probability to the answer text.
    """
    if device is None:
        device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=prompt_with_answer,
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return outputs.loss.item()


def evaluate_r1(model, processor, r1_data, cav=None, alpha=0.0,
                layers=None, technique="subtraction",
                image_tokens_only=False):
    """
    R1: VQA MCQ accuracy via loss-based scoring.

    For each item:
      1. Load image
      2. For each candidate answer, compute loss of prompt + answer
      3. Model picks answer with lowest loss
      4. Check if correct

    Args:
        model: LLaVA model
        processor: AutoProcessor
        r1_data: List of dicts with image_path, correct_answer, options
        cav: CAV tensor (None for baseline)
        alpha: Ablation intensity (0.0 for baseline)
        layers: Layer indices for ablation
        technique: "subtraction" or "projection"
        image_tokens_only: If True, ablate only image tokens

    Returns:
        accuracy: float
        per_item: list of dicts
    """
    device = next(model.parameters()).device
    correct = 0
    total = 0
    per_item = []

    # Prepare ablation context
    use_ablation = cav is not None and alpha > 0.0 and layers is not None
    ctx = AblationContext(model, cav, alpha, layers, technique,
                          image_tokens_only) if use_ablation else None

    if ctx:
        ctx.__enter__()

    try:
        for item in r1_data:
            image_path = item["image_path"]
            correct_answer = item["correct_answer"]
            options = item["options"]

            # Set image token mask if needed (for image_tokens_only mode)
            if use_ablation and image_tokens_only:
                img = Image.open(image_path).convert("RGB")
                # Use first option's prompt to get input_ids shape
                sample_prompt = cfg.R1_VQA_PROMPT.format(
                    options=", ".join(options)
                ) + f" {options[0]}"
                sample_inputs = processor(
                    text=sample_prompt, images=img, return_tensors="pt"
                )
                set_image_token_mask(
                    model, sample_inputs["input_ids"], processor
                )

            # Score each option
            scores = {}
            for option in options:
                prompt = cfg.R1_VQA_PROMPT.format(
                    options=", ".join(options)
                ) + f" {option}"
                loss = compute_sequence_loss(
                    model, processor, image_path, prompt, device
                )
                scores[option] = loss

            # Pick option with lowest loss
            best_option = min(scores, key=scores.get)
            is_correct = best_option == correct_answer

            if is_correct:
                correct += 1
            total += 1

            per_item.append({
                "image_path": image_path,
                "correct_answer": correct_answer,
                "model_answer": best_option,
                "is_correct": is_correct,
                "scores": scores,
            })
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, per_item


# ---------------------------------------------------------------------------
# SFA: Semantic Feature Analysis — binary yes/no probe
# ---------------------------------------------------------------------------
def evaluate_sfa(model, processor, sfa_items,
                 cav=None, alpha=0.0,
                 layers=None, technique="subtraction",
                 image_tokens_only=False):
    """
    SFA: Semantic Feature Analysis accuracy via binary yes/no loss comparison.

    For each item the model is shown an image and asked a feature probe question.
    The answer with lower cross-entropy loss is selected as the model's response.

    Scoring (identical pattern to evaluate_r1):
        prompt = SFA_PROMPT_TEMPLATE.format(question=question)
        loss_yes = compute_sequence_loss(model, processor, image_path, prompt + " yes")
        loss_no  = compute_sequence_loss(model, processor, image_path, prompt + " no")
        model_answer = "yes" if loss_yes < loss_no else "no"

    Args:
        model: LLaVA model
        processor: AutoProcessor
        sfa_items: List of dicts, each with keys:
            image_path      (str)  — path to image
            question        (str)  — feature probe, e.g. "Is this an animal?"
            expected_answer (str)  — "yes" or "no"
            dimension       (str)  — semantic category of the probe:
                                     "category" / "function" / "perceptual" /
                                     "structural" / "associative"
        cav: CAV tensor [d_model], or None for baseline
        alpha: Ablation intensity (0.0 = no ablation)
        layers: List of layer indices for ablation
        technique: "subtraction" or "projection"
        image_tokens_only: If True, ablate only the 576 CLIP image-token positions

    Returns:
        accuracy:              float  — overall fraction correct
        accuracy_by_dimension: dict[str, float]  — per-dimension fraction correct
        per_item:              list[dict]  — per-item details
    """
    DIMENSIONS = ["category", "function", "perceptual", "structural", "associative"]
    device = next(model.parameters()).device

    use_ablation = cav is not None and alpha > 0.0 and layers is not None
    ctx = AblationContext(model, cav, alpha, layers, technique,
                          image_tokens_only) if use_ablation else None

    correct = 0
    total = 0
    correct_by_dim = {d: 0 for d in DIMENSIONS}
    total_by_dim   = {d: 0 for d in DIMENSIONS}
    per_item = []

    if ctx:
        ctx.__enter__()

    try:
        for item in sfa_items:
            image_path      = item["image_path"]
            question        = item["question"]
            expected        = item["expected_answer"]   # "yes" or "no"
            dimension       = item["dimension"]

            # Set image token mask once per item; image tokens are at fixed
            # positions 1–576 regardless of the single appended answer token.
            if use_ablation and image_tokens_only:
                img = Image.open(image_path).convert("RGB")
                sample_prompt = cfg.SFA_PROMPT_TEMPLATE.format(question=question)
                sample_inputs = processor(
                    text=sample_prompt, images=img, return_tensors="pt"
                )
                set_image_token_mask(model, sample_inputs["input_ids"], processor)

            prompt = cfg.SFA_PROMPT_TEMPLATE.format(question=question)

            loss_yes = compute_sequence_loss(
                model, processor, image_path, prompt + " yes", device
            )
            loss_no = compute_sequence_loss(
                model, processor, image_path, prompt + " no", device
            )

            model_answer = "yes" if loss_yes < loss_no else "no"
            is_correct   = (model_answer == expected)

            if is_correct:
                correct += 1
                if dimension in correct_by_dim:
                    correct_by_dim[dimension] += 1
            total += 1
            if dimension in total_by_dim:
                total_by_dim[dimension] += 1

            per_item.append({
                "image_path":      image_path,
                "question":        question,
                "dimension":       dimension,
                "expected_answer": expected,
                "model_answer":    model_answer,
                "is_correct":      is_correct,
                "loss_yes":        loss_yes,
                "loss_no":         loss_no,
            })
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    accuracy = correct / total if total > 0 else 0.0
    accuracy_by_dimension = {
        d: (correct_by_dim[d] / total_by_dim[d] if total_by_dim[d] > 0 else 0.0)
        for d in DIMENSIONS
    }
    return accuracy, accuracy_by_dimension, per_item


# ---------------------------------------------------------------------------
# Shared Embedding Helpers
# ---------------------------------------------------------------------------
def get_image_embedding(model, processor, image_path, text, layer=None,
                        token_position=None):
    """
    Get pooled embedding from a specific layer for an image+text input.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_path: Path to image
        text: Text prompt (should contain <image> token)
        layer: Layer index (default: cfg.R2_EMBEDDING_LAYER)
        token_position: Override pooling strategy (default: cfg.TOKEN_POSITION)

    Returns:
        Tensor of shape [d_model]
    """
    if layer is None:
        layer = cfg.R2_EMBEDDING_LAYER

    device = next(model.parameters()).device

    act = collect_activations_at_layer(
        model, processor, image_path, text, layer, device
    )

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"]

    pooled = pool_activations(act, input_ids, processor, token_position)
    return pooled


def get_text_only_embedding(model, processor, text, layer=None):
    """
    Get pooled embedding from a specific layer for TEXT-ONLY input (no image).

    The text embedding serves as the REFERENCE for R2a: it represents
    the pure textual concept against which image representations are measured.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        text: Text description (NO <image> token)
        layer: Layer index (default: cfg.R2_EMBEDDING_LAYER)

    Returns:
        Tensor of shape [d_model]
    """
    if layer is None:
        layer = cfg.R2_EMBEDDING_LAYER

    device = next(model.parameters()).device

    # Text-only input: no image, no <image> token
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["act"] = output[0].detach().cpu()
        else:
            activations["act"] = output.detach().cpu()

    target_layer = get_model_layers(model)[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    act = activations["act"][0]  # [seq_len, d_model]
    # Mean pool over all tokens (all are text tokens, no image tokens)
    return act.mean(dim=0)


def get_last_token_embedding(model, processor, image_path, text, layer=None):
    """
    Get the last-token activation from a specific layer for image+text input.

    The last token is the generation position — it summarizes the model's
    internal state before producing output. Used by R2b to compare baseline
    vs intervened representations.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_path: Path to image
        text: Text prompt (should contain <image> token)
        layer: Layer index (default: cfg.R2B_EMBEDDING_LAYER)

    Returns:
        Tensor of shape [d_model]
    """
    if layer is None:
        layer = cfg.R2B_EMBEDDING_LAYER

    device = next(model.parameters()).device

    act = collect_activations_at_layer(
        model, processor, image_path, text, layer, device
    )

    # Last position = generation point
    return act[-1]  # [d_model]


# ---------------------------------------------------------------------------
# R2a: Cross-Modal Similarity (Visual-Semantic Agnosia)
# ---------------------------------------------------------------------------
def evaluate_r2a(model, processor, r2_data, layer=None,
                 cav=None, alpha=0.0, layers_ablation=None,
                 technique="subtraction", image_tokens_only=False):
    """
    R2a: Image token activations vs text-only concept embedding.

    Measures whether the model's visual representation (image tokens)
    aligns with the textual concept description. After CAV ablation,
    this similarity should DROP — the model "sees" but can't connect
    the visual representation to the concept (visual-semantic agnosia).

    For each (image, text_description) pair:
      1. Get text-only embedding FIRST, WITHOUT ablation (fixed reference)
      2. Get image embedding (image_only_mean pooling) WITH ablation
      3. Cosine similarity between them

    CRITICAL: Text embeddings are computed OUTSIDE the ablation context.
    They serve as the ground-truth concept representation.

    Args:
        model, processor, r2_data: Standard args
        layer: Layer for embedding extraction (default: cfg.R2_EMBEDDING_LAYER)
        cav, alpha, layers_ablation, technique: Ablation parameters
        image_tokens_only: If True, ablate only image tokens

    Returns:
        mean_similarity: float
        per_pair: list of dicts
    """
    if layer is None:
        layer = cfg.R2_EMBEDDING_LAYER

    use_ablation = cav is not None and alpha > 0.0 and layers_ablation is not None

    # Phase 1: Collect ALL text embeddings WITHOUT ablation (fixed reference)
    text_embeddings = []
    for pair in r2_data:
        text_desc = pair["text_description"]
        text_prompt = cfg.R2_TEXT_ONLY_PROMPT.format(description=text_desc)
        emb_text = get_text_only_embedding(model, processor, text_prompt, layer)
        text_embeddings.append(emb_text)

    # Phase 2: Collect image embeddings WITH ablation (if active)
    ctx = AblationContext(model, cav, alpha, layers_ablation, technique,
                          image_tokens_only) if use_ablation else None
    if ctx:
        ctx.__enter__()

    per_pair = []
    try:
        for i, pair in enumerate(r2_data):
            image_path = pair["image_path"]
            text_desc = pair["text_description"]

            # Set image token mask for image_tokens_only mode
            if use_ablation and image_tokens_only:
                img = Image.open(image_path).convert("RGB")
                inputs = processor(
                    text=cfg.R2_IMAGE_PROMPT, images=img, return_tensors="pt"
                )
                set_image_token_mask(
                    model, inputs["input_ids"], processor
                )

            # Image embedding WITH ablation, using image_only_mean pooling
            emb_img = get_image_embedding(
                model, processor, image_path, cfg.R2_IMAGE_PROMPT, layer,
                token_position=cfg.R2A_IMAGE_POOLING,
            )

            # Text embedding (pre-computed WITHOUT ablation)
            emb_text = text_embeddings[i]

            # Cosine similarity
            sim = F.cosine_similarity(
                emb_img.unsqueeze(0).float(),
                emb_text.unsqueeze(0).float(),
            ).item()

            per_pair.append({
                "image_path": image_path,
                "text_description": text_desc,
                "similarity": sim,
            })
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    similarities = [p["similarity"] for p in per_pair]
    mean_sim = float(np.mean(similarities)) if similarities else 0.0

    return mean_sim, per_pair


# ---------------------------------------------------------------------------
# R2b: Output Similarity (Internal Representation Disruption)
# ---------------------------------------------------------------------------
def compute_r2b_baselines(model, processor, r2_data, layer=None):
    """
    Pre-compute baseline last-token embeddings for R2b.

    These are computed ONCE without ablation and cached. All subsequent
    R2b evaluations compare intervened embeddings against these baselines.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        r2_data: List of dicts with image_path, text_description
        layer: Layer index (default: cfg.R2B_EMBEDDING_LAYER)

    Returns:
        baseline_embeddings: List of Tensors [d_model], one per item
    """
    if layer is None:
        layer = cfg.R2B_EMBEDDING_LAYER

    baselines = []
    for pair in r2_data:
        image_path = pair["image_path"]
        emb = get_last_token_embedding(
            model, processor, image_path, cfg.R2B_OUTPUT_PROMPT, layer
        )
        baselines.append(emb)

    return baselines


def evaluate_r2b(model, processor, r2_data, baseline_embeddings,
                 layer=None, cav=None, alpha=0.0, layers_ablation=None,
                 technique="subtraction", image_tokens_only=False):
    """
    R2b: Cosine similarity between baseline and intervened last-token embeddings.

    Measures how much the ablation disrupts the model's internal state
    at the generation position. R2b ≈ 1.0 means no effect; R2b → 0.0
    means the ablation has fundamentally altered the model's output
    representation.

    Args:
        model, processor: Standard args
        r2_data: List of dicts with image_path, text_description
        baseline_embeddings: Pre-computed baselines from compute_r2b_baselines()
        layer: Layer index (default: cfg.R2B_EMBEDDING_LAYER)
        cav, alpha, layers_ablation, technique: Ablation parameters
        image_tokens_only: If True, ablate only image tokens

    Returns:
        mean_similarity: float
        per_pair: list of dicts
    """
    if layer is None:
        layer = cfg.R2B_EMBEDDING_LAYER

    use_ablation = cav is not None and alpha > 0.0 and layers_ablation is not None

    ctx = AblationContext(model, cav, alpha, layers_ablation, technique,
                          image_tokens_only) if use_ablation else None
    if ctx:
        ctx.__enter__()

    per_pair = []
    try:
        for i, pair in enumerate(r2_data):
            image_path = pair["image_path"]

            # Set image token mask for image_tokens_only mode
            if use_ablation and image_tokens_only:
                img = Image.open(image_path).convert("RGB")
                inputs = processor(
                    text=cfg.R2B_OUTPUT_PROMPT, images=img, return_tensors="pt"
                )
                set_image_token_mask(
                    model, inputs["input_ids"], processor
                )

            # Intervened last-token embedding
            emb_int = get_last_token_embedding(
                model, processor, image_path, cfg.R2B_OUTPUT_PROMPT, layer
            )

            # Baseline embedding (pre-computed)
            emb_base = baseline_embeddings[i]

            # Cosine similarity
            sim = F.cosine_similarity(
                emb_int.unsqueeze(0).float(),
                emb_base.unsqueeze(0).float(),
            ).item()

            per_pair.append({
                "image_path": image_path,
                "text_description": pair.get("text_description", ""),
                "similarity": sim,
            })
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    similarities = [p["similarity"] for p in per_pair]
    mean_sim = float(np.mean(similarities)) if similarities else 0.0

    return mean_sim, per_pair
