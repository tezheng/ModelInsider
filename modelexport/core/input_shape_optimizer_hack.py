"""
FIXME: TEMPORARY HACKY INPUT SHAPE OPTIMIZATION MODULE

WARNING: This module contains hardcoded model type lists and hacky optimizations.
This is a temporary solution that violates the universal design principles.

TODO: Replace with a proper solution that:
1. Uses model config attributes without hardcoding model names
2. Implements a plugin system for model-specific optimizations
3. Moves model type lists to configuration files
4. Uses proper feature detection instead of model type matching

This module will be refactored in a future iteration.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# FIXME: These hardcoded model type lists violate universal design principles
# TODO: Move to configuration file or detect from model features
TEXT_MODEL_TYPES = {
    'bert', 'gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'qwen', 'opt', 
    'bloom', 'falcon', 't5', 'bart', 'roberta', 'distilbert', 
    'electra', 'albert'
}

VISION_MODEL_TYPES = {
    'vit', 'resnet', 'convnext', 'swin', 'deit', 'beit', 'levit', 
    'mobilenet', 'efficientnet', 'regnet', 'sam'
}

MULTIMODAL_MODEL_TYPES = {
    'clip', 'blip', 'albef', 'flava', 'layoutlm', 'lxmert', 'vilbert'
}


def optimize_input_shapes_for_model_hack(
    shapes: dict[str, Any],
    model_name_or_path: str,
    export_config: Any,
    task: str | None = None
) -> dict[str, Any]:
    """
    HACKY: Intelligently optimize input shapes based on model architecture and domain.
    
    FIXME: This function contains hardcoded logic and should be refactored.
    
    This function applies universal improvements to Optimum's default shapes by:
    1. Using model config attributes when available (e.g., image_size, max_position_embeddings)
    2. Applying domain-specific standards (ImageNet 224x224, BERT 128 sequence)
    3. Falling back to reasonable defaults
    4. Remaining architecture-agnostic (no hardcoded model names) [VIOLATED - FIXME]
    
    Args:
        shapes: Base shapes from Optimum's DEFAULT_DUMMY_SHAPES
        model_name_or_path: HuggingFace model name/path
        export_config: Optimum export config instance
        task: Task name (optional)
        
    Returns:
        Optimized shapes dictionary
        
    TODO: Refactor to remove hardcoded model type lists
    """
    optimized_shapes = shapes.copy()
    
    try:
        # Lazy import to avoid circular dependencies
        from transformers import AutoConfig
        
        # Load model config to get architecture info
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = config.model_type.lower()
        
        logger.info(f"[HACK] Optimizing input shapes for model_type='{model_type}', task='{task}'")
        
        # Apply universal optimizations based on model domain and config
        optimized_shapes = _apply_universal_shape_optimizations_hack(
            shapes=optimized_shapes,
            config=config,
            model_type=model_type,
            task=task
        )
        
        logger.info(f"[HACK] Shape optimizations applied: {optimized_shapes}")
        return optimized_shapes
        
    except Exception as e:
        logger.warning(f"[HACK] Could not optimize input shapes: {e}")
        logger.info("[HACK] Using default shapes without optimization")
        return shapes


def _apply_universal_shape_optimizations_hack(
    shapes: dict[str, Any],
    config: Any,
    model_type: str,
    task: str | None = None
) -> dict[str, Any]:
    """
    HACKY: Apply universal shape optimizations based on model domain and config.
    
    FIXME: Uses hardcoded model type lists - violates universal design.
    
    This function implements domain-specific logic without hardcoding model names:
    - Text models: Better sequence lengths, batch size 1
    - Vision models: ImageNet standards (224x224) unless config specifies otherwise
    - Multimodal models: Appropriate dimensions for both text and vision
    
    Args:
        shapes: Current shapes dictionary
        config: HuggingFace model config
        model_type: Model type from config (e.g., 'bert', 'vit', 'resnet')
        task: Task name (optional)
        
    Returns:
        Updated shapes dictionary
    """
    # Universal improvements for all models
    shapes["batch_size"] = 1  # Prefer batch size 1 for ONNX export
    
    # Domain-specific optimizations
    # FIXME: These use hardcoded model type lists
    if _is_text_model_hack(model_type, config):
        shapes = _optimize_text_model_shapes_hack(shapes, config, model_type, task)
    elif _is_vision_model_hack(model_type, config):
        shapes = _optimize_vision_model_shapes_hack(shapes, config, model_type, task)
    elif _is_multimodal_model_hack(model_type, config):
        shapes = _optimize_multimodal_model_shapes_hack(shapes, config, model_type, task)
    
    return shapes


def _is_text_model_hack(model_type: str, config: Any) -> bool:
    """
    HACKY: Check if model is primarily for text processing.
    FIXME: Uses hardcoded model type list - should detect from features.
    """
    return model_type in TEXT_MODEL_TYPES


def _is_vision_model_hack(model_type: str, config: Any) -> bool:
    """
    HACKY: Check if model is primarily for vision processing.
    FIXME: Uses hardcoded model type list - should detect from features.
    """
    return model_type in VISION_MODEL_TYPES


def _is_multimodal_model_hack(model_type: str, config: Any) -> bool:
    """
    HACKY: Check if model handles multiple modalities.
    FIXME: Uses hardcoded model type list - should detect from features.
    """
    return model_type in MULTIMODAL_MODEL_TYPES


def _optimize_text_model_shapes_hack(
    shapes: dict[str, Any],
    config: Any,
    model_type: str,
    task: str | None = None
) -> dict[str, Any]:
    """
    HACKY: Optimize shapes for text models (BERT, GPT, etc.).
    
    TODO: Move magic numbers to configuration.
    """
    # Use model config for sequence length if available
    if hasattr(config, 'max_position_embeddings'):
        # Use a reasonable sequence length (not the full max)
        max_seq = config.max_position_embeddings
        shapes["sequence_length"] = min(128, max_seq)  # FIXME: Magic number 128
    elif hasattr(config, 'n_positions'):
        # GPT-style models
        max_seq = config.n_positions
        shapes["sequence_length"] = min(128, max_seq)  # FIXME: Magic number 128
    else:
        # Fallback to standard BERT-like sequence length
        shapes["sequence_length"] = 128  # FIXME: Magic number 128
    
    logger.debug(f"[HACK] Text model optimization: sequence_length={shapes['sequence_length']}")
    return shapes


def _optimize_vision_model_shapes_hack(
    shapes: dict[str, Any],
    config: Any,
    model_type: str,
    task: str | None = None
) -> dict[str, Any]:
    """
    HACKY: Optimize shapes for vision models (ResNet, ViT, etc.).
    
    TODO: Move ImageNet standard dimensions to configuration.
    """
    # Use model config for image size if available
    if hasattr(config, 'image_size'):
        image_size = config.image_size
        shapes["height"] = image_size
        shapes["width"] = image_size
    elif hasattr(config, 'input_size'):
        # Some models use input_size instead
        if isinstance(config.input_size, list | tuple) and len(config.input_size) >= 2:
            shapes["height"] = config.input_size[-2]  # Second to last dimension
            shapes["width"] = config.input_size[-1]   # Last dimension
        else:
            shapes["height"] = shapes["width"] = 224  # FIXME: Magic number 224 (ImageNet)
    else:
        # Fallback to ImageNet standard dimensions
        shapes["height"] = shapes["width"] = 224  # FIXME: Magic number 224 (ImageNet)
    
    # Standard RGB channels for vision models
    shapes["num_channels"] = 3  # FIXME: Magic number 3 (RGB)
    
    logger.debug(f"[HACK] Vision model optimization: {shapes['height']}x{shapes['width']}, channels={shapes['num_channels']}")
    return shapes


def _optimize_multimodal_model_shapes_hack(
    shapes: dict[str, Any],
    config: Any,
    model_type: str,
    task: str | None = None
) -> dict[str, Any]:
    """
    HACKY: Optimize shapes for multimodal models (CLIP, BLIP, etc.).
    
    TODO: Refactor to use proper feature detection.
    """
    # Apply both text and vision optimizations
    shapes = _optimize_text_model_shapes_hack(shapes, config, model_type, task)
    shapes = _optimize_vision_model_shapes_hack(shapes, config, model_type, task)
    
    # Model-specific adjustments based on config attributes (not model names)
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings'):
        # Use text config if available
        max_text = config.text_config.max_position_embeddings
        shapes["sequence_length"] = min(77, max_text)  # FIXME: Magic number 77 (CLIP max)
    
    logger.debug(f"[HACK] Multimodal model optimization: text_seq={shapes.get('sequence_length')}, image={shapes.get('height')}x{shapes.get('width')}")
    return shapes