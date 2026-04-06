"""
Model Loader — LLaVA-1.5 7B
==============================
Centralizes model loading for LLaVA-1.5 7B.

Loads in fp16 on CPU first, then moves to GPU to avoid segfault
with device_map='auto' + accelerate on RTX 5060 Ti (Blackwell).

VRAM budget on RTX 5060 Ti (16GB):
    Model weights (fp16):       ~14.1 GB
    KV cache + activations:     ~2-3 GB (tight — use torch.no_grad)
"""

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

import experiment_config as cfg


def load_llava_model():
    """
    Load LLaVA-1.5 7B in fp16.

    Loads on CPU first to avoid device_map='auto' segfault on Blackwell GPUs,
    then moves to CUDA.

    Returns:
        model: LlavaForConditionalGeneration (fp16 on CUDA)
        processor: AutoProcessor (handles text tokenization + image preprocessing)
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = model.to("cuda")
    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.MODEL_NAME)

    return model, processor


def get_text_token_mask(input_ids, image_token_id=None, processor=None):
    """
    Return a boolean mask identifying text token positions (not image tokens).

    LLaVA injects 576 image tokens into the sequence. This mask allows
    collecting activations from text tokens only.

    Args:
        input_ids: Token IDs tensor [batch, seq_len]
        image_token_id: The special token ID for image patches.
                       If None, auto-detected from processor.
        processor: AutoProcessor, used to detect image_token_id.

    Returns:
        mask: Boolean tensor [batch, seq_len], True for text tokens
    """
    if image_token_id is None:
        if processor is not None and hasattr(processor, "tokenizer"):
            # LLaVA uses a special <image> token that gets expanded to 576 tokens
            # The image token ID is typically 32000 in llava-hf models
            image_token_id = getattr(
                processor.tokenizer, "image_token_id",
                getattr(processor, "image_token_id", 32000)
            )
        else:
            image_token_id = 32000  # Default for llava-hf

    # Text tokens are those that are NOT the image placeholder token
    mask = input_ids != image_token_id
    return mask


def get_model_layers(model):
    """
    Get the transformer decoder layers from LLaVA's language model.

    Handles different transformers versions / model structures:
      Path 1: model.model.layers                        (LM merged into top-level)
      Path 2: model.language_model.model.layers          (old transformers < 4.46)
      Path 3: model.model.language_model.layers          (LlamaModel directly)
      Path 4: model.model.language_model.model.layers    (LlamaForCausalLM wrapper)

    The actual structure for llava-hf with current transformers is:
      LlavaForConditionalGeneration
        └── .model (LlavaModel)
              └── .language_model (LlamaModel)  ← has .layers directly
                    └── .layers (ModuleList of LlamaDecoderLayer)

    Returns:
        torch.nn.ModuleList of decoder layers
    """
    # Path 1: LM layers merged directly into model.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # Path 2: top-level language_model (old transformers < 4.46)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers

    # Path 3: LlavaModel.language_model = LlamaModel (has .layers directly)
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers

    # Path 4: LlavaModel.language_model = LlamaForCausalLM (has .model.layers)
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "model")):
        return model.model.language_model.model.layers

    raise AttributeError(
        f"Cannot find decoder layers in {type(model).__name__}. "
        f"Explored paths: model.model.layers, model.language_model.model.layers, "
        f"model.model.language_model.layers, model.model.language_model.model.layers"
    )


def get_device(model):
    """Get the device of the model's first parameter."""
    return next(model.parameters()).device
