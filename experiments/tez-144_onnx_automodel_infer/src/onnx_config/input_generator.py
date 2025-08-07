"""
Input specification and dummy input generation for different model types.

This module handles the generation of input specifications and dummy inputs
for various model architectures and modalities.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from .patterns import (
    MODEL_TYPE_TO_INPUTS,
    DEFAULT_SHAPES,
    DYNAMIC_AXES_PATTERNS,
    get_model_family
)


class InputSpecGenerator:
    """Generate input specifications for ONNX export."""
    
    @classmethod
    def get_input_names(cls, model_type: str, task: str, config: Dict[str, Any]) -> List[str]:
        """
        Get input names for a model.
        
        Args:
            model_type: Model type string
            task: Task name
            config: Model configuration
            
        Returns:
            List of input names
        """
        model_type_lower = model_type.lower()
        
        # Check direct mapping
        if model_type_lower in MODEL_TYPE_TO_INPUTS:
            return MODEL_TYPE_TO_INPUTS[model_type_lower]
        
        # Check for partial matches
        for known_type, inputs in MODEL_TYPE_TO_INPUTS.items():
            if known_type in model_type_lower:
                return inputs
        
        # Determine by model family
        model_family = get_model_family(model_type)
        
        if model_family == "vision":
            return ["pixel_values"]
        elif model_family == "audio":
            # Check for specific audio model types
            if "whisper" in model_type_lower or "seamless" in model_type_lower:
                return ["input_features"]
            else:
                return ["input_values", "attention_mask"]
        elif model_family == "multimodal":
            return ["input_ids", "attention_mask", "pixel_values"]
        elif model_family == "document":
            return cls._get_document_inputs(model_type_lower, config)
        elif model_family == "encoder-decoder":
            return ["input_ids", "attention_mask", "decoder_input_ids"]
        else:
            # Default text inputs
            return cls._get_text_inputs(model_type_lower, config)
    
    @classmethod
    def _get_text_inputs(cls, model_type: str, config: Dict[str, Any]) -> List[str]:
        """Get text model inputs based on configuration."""
        inputs = ["input_ids", "attention_mask"]
        
        # Check if model uses token type ids
        uses_token_type = any(x in model_type for x in [
            "bert", "albert", "electra", "layoutlm", "tapas", "deberta"
        ])
        
        # But not these variants
        not_token_type = any(x in model_type for x in [
            "roberta", "distilbert", "camembert", "xlm-roberta"
        ])
        
        if uses_token_type and not not_token_type:
            # Double check with config
            if config.get("type_vocab_size", 0) > 0:
                inputs.append("token_type_ids")
        
        # Check for position ids (some models need explicit position ids)
        if config.get("position_embedding_type") == "absolute":
            # Only some models need explicit position_ids
            if any(x in model_type for x in ["gpt", "opt", "bloom"]):
                inputs.append("position_ids")
        
        return inputs
    
    @classmethod
    def _get_document_inputs(cls, model_type: str, config: Dict[str, Any]) -> List[str]:
        """Get document model inputs."""
        inputs = ["input_ids", "attention_mask"]
        
        # LayoutLM family needs bbox
        if "layoutlm" in model_type:
            inputs.append("bbox")
            
            # Some versions need token_type_ids
            if "layoutlmv3" not in model_type:
                inputs.append("token_type_ids")
            
            # Visual versions need images
            if "layoutlmv2" in model_type or "layoutlmv3" in model_type:
                if "layoutlmv3" in model_type:
                    inputs.append("pixel_values")
                else:
                    inputs.append("image")
        
        # MarkupLM needs special xpath inputs
        elif "markuplm" in model_type:
            inputs.extend(["token_type_ids", "xpath_tags_seq", "xpath_subs_seq"])
        
        return inputs
    
    @classmethod
    def get_dynamic_axes(
        cls,
        input_names: List[str],
        output_names: List[str],
        task: str,
        config: Dict[str, Any]
    ) -> Dict[str, Dict[int, str]]:
        """
        Get dynamic axes for inputs and outputs.
        
        Args:
            input_names: List of input names
            output_names: List of output names
            task: Task name
            config: Model configuration
            
        Returns:
            Dictionary mapping tensor names to dynamic axes
        """
        dynamic_axes = {}
        
        # Add input dynamic axes
        for input_name in input_names:
            if input_name in DYNAMIC_AXES_PATTERNS:
                dynamic_axes[input_name] = DYNAMIC_AXES_PATTERNS[input_name].copy()
            else:
                # Default dynamic axes for unknown inputs
                if "pixel" in input_name or "image" in input_name:
                    dynamic_axes[input_name] = {0: "batch_size"}
                else:
                    dynamic_axes[input_name] = {0: "batch_size", 1: "sequence_length"}
        
        # Add output dynamic axes
        for output_name in output_names:
            if output_name in DYNAMIC_AXES_PATTERNS:
                dynamic_axes[output_name] = DYNAMIC_AXES_PATTERNS[output_name].copy()
            else:
                # Determine by task
                if "classification" in task:
                    if output_name == "logits":
                        if "token" in task:
                            dynamic_axes[output_name] = {0: "batch_size", 1: "sequence_length"}
                        else:
                            dynamic_axes[output_name] = {0: "batch_size"}
                elif "generation" in task:
                    if output_name == "logits":
                        dynamic_axes[output_name] = {0: "batch_size", 1: "sequence_length"}
                    elif "past_key_values" in output_name:
                        # Past key values have complex structure
                        dynamic_axes[output_name] = {0: "batch_size", 2: "past_sequence_length"}
                else:
                    # Default
                    dynamic_axes[output_name] = {0: "batch_size"}
        
        return dynamic_axes
    
    @classmethod
    def generate_dummy_inputs(
        cls,
        model_type: str,
        task: str,
        config: Dict[str, Any],
        batch_size: int = 1,
        seq_length: int = 128,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate dummy inputs for model tracing.
        
        Args:
            model_type: Model type string
            task: Task name
            config: Model configuration
            batch_size: Batch size
            seq_length: Sequence length for text inputs
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of input tensors
        """
        model_type_lower = model_type.lower()
        model_family = get_model_family(model_type)
        
        # Get input names
        input_names = cls.get_input_names(model_type, task, config)
        dummy_inputs = {}
        
        for input_name in input_names:
            if input_name in DEFAULT_SHAPES:
                # Use default shape as template
                default_shape = DEFAULT_SHAPES[input_name]
                
                # Adjust shape based on batch size and seq length
                if input_name in ["input_ids", "attention_mask", "token_type_ids", "position_ids"]:
                    shape = (batch_size, seq_length)
                elif input_name == "bbox":
                    shape = (batch_size, seq_length, 4)
                elif input_name in ["xpath_tags_seq", "xpath_subs_seq"]:
                    shape = (batch_size, seq_length, 50)  # max_depth
                elif input_name in ["pixel_values", "image"]:
                    # Get image size from config
                    image_size = cls._get_image_size(config)
                    num_channels = config.get("num_channels", 3)
                    shape = (batch_size, num_channels, image_size, image_size)
                elif input_name == "pixel_mask":
                    image_size = cls._get_image_size(config)
                    shape = (batch_size, image_size, image_size)
                elif input_name == "input_values":
                    # Audio input
                    audio_length = kwargs.get("audio_length", 16000)
                    shape = (batch_size, audio_length)
                elif input_name == "input_features":
                    # Processed audio features (e.g., Whisper)
                    feature_dim = config.get("num_mel_bins", 80)
                    max_source_positions = config.get("max_source_positions", 1500)
                    shape = (batch_size, feature_dim, max_source_positions)
                else:
                    shape = default_shape
                
                # Generate tensor based on input type
                dummy_inputs[input_name] = cls._generate_tensor_for_input(
                    input_name, shape, config
                )
            else:
                # Handle unknown inputs
                if "decoder" in input_name:
                    shape = (batch_size, seq_length)
                    dummy_inputs[input_name] = cls._generate_tensor_for_input(
                        input_name, shape, config
                    )
                else:
                    # Default shape
                    shape = (batch_size, seq_length)
                    dummy_inputs[input_name] = cls._generate_tensor_for_input(
                        input_name, shape, config
                    )
        
        return dummy_inputs
    
    @classmethod
    def _get_image_size(cls, config: Dict[str, Any]) -> int:
        """Get image size from config."""
        # Try different config keys
        for key in ["image_size", "vision_config.image_size", "img_size", "input_size"]:
            if key in config:
                size = config[key]
                if isinstance(size, int):
                    return size
                elif isinstance(size, (list, tuple)):
                    return size[0]
        
        # Check vision config
        if "vision_config" in config:
            vision_config = config["vision_config"]
            if isinstance(vision_config, dict):
                return cls._get_image_size(vision_config)
        
        # Default
        return 224
    
    @classmethod
    def _generate_tensor_for_input(
        cls,
        input_name: str,
        shape: Tuple[int, ...],
        config: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate appropriate tensor for input type."""
        
        if "input_ids" in input_name or "decoder_input_ids" in input_name:
            # Token IDs - random integers in vocab range
            vocab_size = config.get("vocab_size", 30000)
            return torch.randint(0, vocab_size, shape, dtype=torch.long)
        
        elif "attention_mask" in input_name:
            # Attention mask - mostly 1s
            return torch.ones(shape, dtype=torch.long)
        
        elif "token_type_ids" in input_name:
            # Token type IDs - 0s and 1s
            return torch.zeros(shape, dtype=torch.long)
        
        elif "position_ids" in input_name:
            # Position IDs - sequential
            if len(shape) == 2:
                batch_size, seq_len = shape
                return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            else:
                return torch.arange(shape[-1], dtype=torch.long)
        
        elif "bbox" in input_name:
            # Bounding boxes - random coordinates
            return torch.randint(0, 1000, shape, dtype=torch.long)
        
        elif "pixel_values" in input_name or "image" in input_name:
            # Image data - random floats
            return torch.randn(shape, dtype=torch.float32)
        
        elif "pixel_mask" in input_name:
            # Pixel mask - mostly 1s
            return torch.ones(shape, dtype=torch.long)
        
        elif "input_values" in input_name or "input_features" in input_name:
            # Audio data - random floats
            return torch.randn(shape, dtype=torch.float32)
        
        elif "xpath" in input_name:
            # XPath data - random integers
            return torch.randint(0, 100, shape, dtype=torch.long)
        
        elif "input_points" in input_name:
            # SAM model points
            return torch.randn(shape, dtype=torch.float32)
        
        elif "input_labels" in input_name:
            # SAM model labels
            return torch.randint(0, 2, shape, dtype=torch.long)
        
        else:
            # Default - depends on dtype inference
            if "mask" in input_name or "ids" in input_name:
                return torch.ones(shape, dtype=torch.long)
            else:
                return torch.randn(shape, dtype=torch.float32)
    
    @classmethod
    def get_preprocessor_type(cls, model_type: str, task: str) -> str:
        """
        Determine the type of preprocessor needed.
        
        Args:
            model_type: Model type string
            task: Task name
            
        Returns:
            Preprocessor type: 'tokenizer', 'image_processor', 'feature_extractor', 'processor'
        """
        model_type_lower = model_type.lower()
        model_family = get_model_family(model_type)
        
        if model_family == "vision":
            return "image_processor"
        elif model_family == "audio":
            return "feature_extractor"
        elif model_family == "multimodal":
            return "processor"  # Multi-modal processor
        elif model_family == "document":
            return "processor"  # Document processors handle text + layout
        else:
            return "tokenizer"