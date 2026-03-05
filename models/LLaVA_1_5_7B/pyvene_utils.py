"""
pyvene Utilities — Activation Collection & Intervention Hooks
==============================================================
Replaces TransformerLens API with pyvene for LLaVA-1.5 7B.

TransformerLens → pyvene mapping:
    model.run_with_cache()     → collect_activations_at_layer()
    model.hooks(fwd_hooks=...) → create_intervention()
    cache["blocks.L.hook_resid_post"] → collected activations tensor
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import pyvene as pv

import experiment_config as cfg
from model_loader import get_text_token_mask, get_model_layers


# ---------------------------------------------------------------------------
# Custom Intervention Classes
# ---------------------------------------------------------------------------
class CAVSubtractionIntervention(pv.TrainableIntervention):
    """Subtract alpha * CAV from residual stream: act' = act - alpha * cav."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cav = None
        self.alpha = 1.0

    def set_params(self, cav, alpha):
        self.cav = cav
        self.alpha = alpha

    def forward(self, base, source=None, subspaces=None):
        if self.cav is None:
            return base
        return base - self.alpha * self.cav.to(base.device)


class CAVProjectionIntervention(pv.TrainableIntervention):
    """Project out CAV direction: act' = act - alpha * (act . cav) * cav."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cav = None
        self.alpha = 1.0

    def set_params(self, cav, alpha):
        self.cav = cav
        self.alpha = alpha

    def forward(self, base, source=None, subspaces=None):
        if self.cav is None:
            return base
        cav = self.cav.to(base.device)
        # base: [batch, seq_len, d_model], cav: [d_model]
        dot = torch.einsum("bsd,d->bs", base, cav)
        proj = torch.einsum("bs,d->bsd", dot, cav)
        return base - self.alpha * proj


# ---------------------------------------------------------------------------
# Activation Collection
# ---------------------------------------------------------------------------
def _prepare_inputs(processor, image_path, prompt, device):
    """Prepare multimodal inputs for LLaVA."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def collect_activations_at_layer(model, processor, image_path, prompt, layer,
                                 device=None):
    """
    Collect residual stream activations at a specific layer.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_path: Path to image file
        prompt: Text prompt
        layer: Layer index (0-31)
        device: torch device

    Returns:
        Tensor of shape [seq_len, d_model]
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = _prepare_inputs(processor, image_path, prompt, device)

    # Use PyTorch hooks directly (more reliable than pyvene for collection)
    activations = {}

    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states, ...) for LLaMA layers
        if isinstance(output, tuple):
            activations["act"] = output[0].detach().cpu()
        else:
            activations["act"] = output.detach().cpu()

    # Hook into the specific language model layer
    target_layer = get_model_layers(model)[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    return activations["act"][0]  # [seq_len, d_model]


def pool_activations(activations, input_ids, processor, token_position=None):
    """
    Pool activations according to token position strategy.

    Args:
        activations: [seq_len, d_model]
        input_ids: [1, seq_len] or [seq_len]
        processor: AutoProcessor
        token_position: Strategy from cfg.TOKEN_POSITION

    Returns:
        Tensor of shape [d_model]
    """
    if token_position is None:
        token_position = cfg.TOKEN_POSITION

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if token_position == "text_only_mean":
        mask = get_text_token_mask(input_ids, processor=processor)
        mask = mask[0]  # [seq_len]
        text_acts = activations[mask]
        if text_acts.shape[0] == 0:
            return activations.mean(dim=0)
        return text_acts.mean(dim=0)

    elif token_position == "all_mean":
        return activations.mean(dim=0)

    elif token_position == "image_only_mean":
        mask = get_text_token_mask(input_ids, processor=processor)
        mask = ~mask[0]  # Invert: True for image tokens
        img_acts = activations[mask]
        if img_acts.shape[0] == 0:
            return activations.mean(dim=0)
        return img_acts.mean(dim=0)

    elif token_position == "last_text_token":
        mask = get_text_token_mask(input_ids, processor=processor)
        mask = mask[0]
        text_positions = torch.where(mask)[0]
        if len(text_positions) == 0:
            return activations[-1]
        return activations[text_positions[-1]]

    else:
        return activations.mean(dim=0)


def collect_activations_batch(model, processor, image_text_pairs, layer,
                              token_position=None, show_progress=True):
    """
    Collect pooled activations for a batch of (image_path, prompt) pairs.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_text_pairs: List of (image_path, prompt) tuples
        layer: Layer index
        token_position: Pooling strategy
        show_progress: Show tqdm progress bar

    Returns:
        Tensor of shape [N, d_model]
    """
    device = next(model.parameters()).device
    all_acts = []

    iterator = tqdm(image_text_pairs, desc=f"Layer {layer}") if show_progress else image_text_pairs

    for image_path, prompt in iterator:
        act = collect_activations_at_layer(
            model, processor, image_path, prompt, layer, device
        )
        inputs = _prepare_inputs(processor, image_path, prompt, device)
        pooled = pool_activations(
            act, inputs["input_ids"].cpu(), processor, token_position
        )
        all_acts.append(pooled)

    return torch.stack(all_acts)  # [N, d_model]


# ---------------------------------------------------------------------------
# Intervention (Ablation)
# ---------------------------------------------------------------------------
def set_image_token_mask(model, input_ids, processor):
    """
    Compute and store the image token mask on the model for use by hooks.

    Call this BEFORE each forward pass when using image_tokens_only ablation,
    so the hook knows which positions are image tokens.

    The mask is stored as model._image_token_mask: [1, seq_len] bool tensor,
    True for image token positions.

    Args:
        model: LLaVA model
        input_ids: [1, seq_len] or [seq_len] tensor
        processor: AutoProcessor
    """
    text_mask = get_text_token_mask(input_ids, processor=processor)  # True = text
    # Invert: True = image tokens
    model._image_token_mask = ~text_mask  # [1, seq_len]


def create_ablation_hooks(model, cav, alpha, layers, technique="subtraction",
                          image_tokens_only=False):
    """
    Register forward hooks for ablation on specified layers.

    Instead of pyvene IntervenableModel (which may have compatibility issues
    with 4-bit models), we use PyTorch register_forward_hook directly.

    Args:
        model: LLaVA model
        cav: CAV tensor [d_model]
        alpha: Ablation intensity
        layers: List of layer indices
        technique: "subtraction" or "projection"
        image_tokens_only: If True, ablate only image token positions.
            Requires set_image_token_mask() to be called before each forward pass.
            Simulates "pure visual agnosia" — text processing remains intact.

    Returns:
        List of hook handles (call handle.remove() when done)
    """
    handles = []
    cav_device = None  # Will be set on first use

    for layer_idx in layers:
        def make_hook(technique, cav, alpha, image_tokens_only):
            def hook_fn(module, input, output):
                nonlocal cav_device
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = None

                if cav_device is None or cav_device != hidden_states.device:
                    cav_device = hidden_states.device

                cav_on_device = cav.to(hidden_states.device, dtype=hidden_states.dtype)

                if image_tokens_only:
                    # Only modify image token positions
                    mask = getattr(model, "_image_token_mask", None)
                    if mask is None:
                        # Fallback: assume first N_IMAGE_TOKENS after BOS
                        seq_len = hidden_states.shape[1]
                        mask = torch.zeros(1, seq_len, dtype=torch.bool,
                                           device=hidden_states.device)
                        start = 1  # Skip BOS
                        end = min(start + cfg.N_IMAGE_TOKENS, seq_len)
                        mask[0, start:end] = True
                    else:
                        mask = mask.to(hidden_states.device)

                    # mask: [1, seq_len], expand to [1, seq_len, 1] for broadcasting
                    mask_3d = mask.unsqueeze(-1)  # [1, seq_len, 1]

                    if technique == "subtraction":
                        delta = alpha * cav_on_device
                        hidden_states = hidden_states - delta * mask_3d.float()
                    elif technique == "projection":
                        dot = torch.einsum("bsd,d->bs", hidden_states, cav_on_device)
                        proj = torch.einsum("bs,d->bsd", dot, cav_on_device)
                        hidden_states = hidden_states - alpha * proj * mask_3d.float()
                else:
                    # Modify ALL token positions
                    if technique == "subtraction":
                        hidden_states = hidden_states - alpha * cav_on_device
                    elif technique == "projection":
                        dot = torch.einsum("bsd,d->bs", hidden_states, cav_on_device)
                        proj = torch.einsum("bs,d->bsd", dot, cav_on_device)
                        hidden_states = hidden_states - alpha * proj

                if rest is not None:
                    return (hidden_states,) + rest
                return hidden_states

            return hook_fn

        target_layer = get_model_layers(model)[layer_idx]
        handle = target_layer.register_forward_hook(
            make_hook(technique, cav, alpha, image_tokens_only)
        )
        handles.append(handle)

    return handles


class AblationContext:
    """Context manager for ablation hooks. Removes hooks on exit."""

    def __init__(self, model, cav, alpha, layers, technique="subtraction",
                 image_tokens_only=False):
        self.model = model
        self.cav = cav
        self.alpha = alpha
        self.layers = layers
        self.technique = technique
        self.image_tokens_only = image_tokens_only
        self.handles = []

    def __enter__(self):
        self.handles = create_ablation_hooks(
            self.model, self.cav, self.alpha, self.layers, self.technique,
            self.image_tokens_only,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        # Clean up stored mask
        if hasattr(self.model, "_image_token_mask"):
            del self.model._image_token_mask
        return False


# ---------------------------------------------------------------------------
# Generation with Intervention
# ---------------------------------------------------------------------------
def generate_with_ablation(model, processor, image_path, prompt, cav, alpha,
                           layers, technique="subtraction", max_new_tokens=50):
    """
    Generate text with ablation hooks active.

    Args:
        model: LLaVA model
        processor: AutoProcessor
        image_path: Path to image
        prompt: Text prompt
        cav: CAV tensor [d_model]
        alpha: Ablation intensity
        layers: List of layer indices
        technique: "subtraction" or "projection"
        max_new_tokens: Max tokens to generate

    Returns:
        Generated text string (continuation only)
    """
    device = next(model.parameters()).device
    inputs = _prepare_inputs(processor, image_path, prompt, device)

    with AblationContext(model, cav, alpha, layers, technique):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
            )

    # Decode only the generated tokens (skip input)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def forward_with_ablation(model, processor, image_path, prompt, cav, alpha,
                          layers, technique="subtraction"):
    """
    Forward pass with ablation hooks. Returns logits.

    Args:
        model, processor, image_path, prompt, cav, alpha, layers, technique

    Returns:
        logits: Tensor [1, seq_len, vocab_size]
    """
    device = next(model.parameters()).device
    inputs = _prepare_inputs(processor, image_path, prompt, device)

    with AblationContext(model, cav, alpha, layers, technique):
        with torch.no_grad():
            outputs = model(**inputs)

    return outputs.logits
