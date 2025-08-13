"""
Universal OnnxConfig implementation that works with any HuggingFace model.

This is the main class that provides automatic OnnxConfig generation for any model,
eliminating the need for model-specific configuration classes.
"""

from typing import Any

import torch

from .input_generator import InputSpecGenerator
from .shape_inference import ShapeInferencer
from .task_detector import TaskDetector


class UniversalOnnxConfig:
    """
    Universal OnnxConfig that automatically generates configuration for any HuggingFace model.
    
    This class eliminates the need for model-specific OnnxConfig implementations by:
    1. Automatically detecting the task from model architecture
    2. Generating appropriate input specifications
    3. Inferring output shapes and names
    4. Providing dummy inputs for tracing
    """
    
    # Default ONNX opset version
    DEFAULT_ONNX_OPSET = 14
    
    # Tolerance for validation
    ATOL_FOR_VALIDATION = 1e-5
    
    def __init__(
        self,
        config: dict[str, Any] | Any,
        task: str | None = None,
        use_past: bool | None = None,
        use_past_in_inputs: bool | None = None,
        preprocessors: dict[str, Any] | None = None
    ):
        """
        Initialize UniversalOnnxConfig.
        
        Args:
            config: Model configuration (dict or PretrainedConfig object)
            task: Optional task override (auto-detected if not provided)
            use_past: Whether to use past key values for generation
            use_past_in_inputs: Whether to include past key values in inputs
            preprocessors: Optional preprocessor information
        """
        # Convert config to dict if needed
        if hasattr(config, "to_dict"):
            self.config_dict = config.to_dict()
            self.config_obj = config
        else:
            self.config_dict = config
            self.config_obj = None
        
        # Get model type
        self.model_type = self.config_dict.get("model_type", "unknown")
        
        # Detect or use provided task
        if task:
            self.task = task
        else:
            self.task = TaskDetector.detect_from_config(self.config_dict)
        
        # Get task family
        self.task_family = TaskDetector.get_task_family(self.task)
        
        # Determine if we should use past key values
        if use_past is not None:
            self.use_past = use_past
        else:
            self.use_past = TaskDetector.requires_past_key_values(self.task, self.model_type)
        
        # Whether to include past in inputs (for generation with cache)
        if use_past_in_inputs is not None:
            self.use_past_in_inputs = use_past_in_inputs
        else:
            # Default: don't include past in initial export
            self.use_past_in_inputs = False
        
        # Store preprocessor info
        self.preprocessors = preprocessors or {}
        
        # Cache for generated values
        self._input_names_cache = None
        self._output_names_cache = None
        self._dynamic_axes_cache = None
    
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """
        Get input specifications with dynamic axes.
        
        Returns:
            Dictionary mapping input names to their dynamic axes
        """
        input_names = self.get_input_names()
        dynamic_axes = {}
        
        for input_name in input_names:
            axes = self._get_dynamic_axes_for_input(input_name)
            if axes:
                dynamic_axes[input_name] = axes
        
        return dynamic_axes
    
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """
        Get output specifications with dynamic axes.
        
        Returns:
            Dictionary mapping output names to their dynamic axes
        """
        output_names = self.get_output_names()
        dynamic_axes = {}
        
        for output_name in output_names:
            axes = self._get_dynamic_axes_for_output(output_name)
            if axes:
                dynamic_axes[output_name] = axes
        
        return dynamic_axes
    
    def get_input_names(self) -> list[str]:
        """
        Get list of input names for the model.
        
        Returns:
            List of input tensor names
        """
        if self._input_names_cache is not None:
            return self._input_names_cache
        
        # Get base input names
        input_names = InputSpecGenerator.get_input_names(
            self.model_type,
            self.task,
            self.config_dict
        )
        
        # Add past key values if needed
        if self.use_past and self.use_past_in_inputs:
            input_names.extend(self._get_past_key_value_names(is_input=True))
        
        self._input_names_cache = input_names
        return input_names
    
    def get_output_names(self) -> list[str]:
        """
        Get list of output names for the model.
        
        Returns:
            List of output tensor names
        """
        if self._output_names_cache is not None:
            return self._output_names_cache
        
        output_names = ShapeInferencer.get_output_names(
            self.task,
            self.model_type,
            self.config_dict
        )
        
        # Add past key values if needed
        if self.use_past and "past_key_values" not in output_names:
            # For generation models, add past_key_values
            if self.task_family == "generation":
                output_names.append("past_key_values")
        
        self._output_names_cache = output_names
        return output_names
    
    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """
        Get combined dynamic axes for inputs and outputs.
        
        Returns:
            Dictionary mapping all tensor names to their dynamic axes
        """
        if self._dynamic_axes_cache is not None:
            return self._dynamic_axes_cache
        
        dynamic_axes = {}
        dynamic_axes.update(self.inputs)
        dynamic_axes.update(self.outputs)
        
        self._dynamic_axes_cache = dynamic_axes
        return dynamic_axes
    
    def generate_dummy_inputs(
        self,
        preprocessor: Any | None = None,
        batch_size: int = 1,
        seq_length: int = 128,
        framework: str = "pt",
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Generate dummy inputs for model tracing.
        
        Args:
            preprocessor: Optional tokenizer/processor to use
            batch_size: Batch size for inputs
            seq_length: Sequence length for text inputs
            framework: Framework ("pt" for PyTorch, "np" for NumPy)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of input tensors
        """
        # Try to use preprocessor if available
        if preprocessor is not None:
            dummy_inputs = self._generate_dummy_inputs_with_preprocessor(
                preprocessor,
                batch_size,
                seq_length,
                framework,
                **kwargs
            )
            if dummy_inputs:
                return dummy_inputs
        
        # Fall back to manual generation
        dummy_inputs = InputSpecGenerator.generate_dummy_inputs(
            self.model_type,
            self.task,
            self.config_dict,
            batch_size,
            seq_length,
            **kwargs
        )
        
        # Add past key values if needed
        if self.use_past and self.use_past_in_inputs:
            past_kv = self._generate_dummy_past_key_values(batch_size, seq_length)
            dummy_inputs.update(past_kv)
        
        # Convert to numpy if requested
        if framework == "np":
            dummy_inputs = {
                name: tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in dummy_inputs.items()
            }
        
        return dummy_inputs
    
    def _generate_dummy_inputs_with_preprocessor(
        self,
        preprocessor: Any,
        batch_size: int,
        seq_length: int,
        framework: str,
        **kwargs
    ) -> dict[str, torch.Tensor] | None:
        """Generate dummy inputs using preprocessor."""
        try:
            # Determine preprocessor type
            preprocessor_type = InputSpecGenerator.get_preprocessor_type(
                self.model_type,
                self.task
            )
            
            if preprocessor_type == "tokenizer":
                # Generate text inputs
                dummy_text = "This is a dummy input " * (seq_length // 5)
                inputs = preprocessor(
                    [dummy_text] * batch_size,
                    padding="max_length",
                    truncation=True,
                    max_length=seq_length,
                    return_tensors=framework
                )
                return dict(inputs)
            
            elif preprocessor_type == "image_processor":
                # Generate image inputs
                import numpy as np
                image_size = self._get_image_size()
                num_channels = self.config_dict.get("num_channels", 3)
                
                # Create dummy images
                dummy_images = np.random.randn(
                    batch_size, image_size, image_size, num_channels
                ).astype(np.float32)
                
                inputs = preprocessor(
                    images=dummy_images,
                    return_tensors=framework
                )
                return dict(inputs)
            
            elif preprocessor_type == "feature_extractor":
                # Generate audio inputs
                import numpy as np
                sample_rate = self.config_dict.get("sample_rate", 16000)
                audio_length = kwargs.get("audio_length", sample_rate)
                
                # Create dummy audio
                dummy_audio = np.random.randn(batch_size, audio_length).astype(np.float32)
                
                inputs = preprocessor(
                    dummy_audio,
                    sampling_rate=sample_rate,
                    return_tensors=framework
                )
                return dict(inputs)
            
            elif preprocessor_type == "processor":
                # Multi-modal processor
                # This is more complex and depends on the specific processor
                return None
            
        except Exception:
            # Fall back to manual generation
            return None
    
    def _get_dynamic_axes_for_input(self, input_name: str) -> dict[int, str]:
        """Get dynamic axes for a specific input."""
        # Common patterns
        if "input_ids" in input_name or "attention_mask" in input_name or "token_type_ids" in input_name or "position_ids" in input_name:
            return {0: "batch_size", 1: "sequence_length"}
        elif "decoder_input_ids" in input_name:
            return {0: "batch_size", 1: "decoder_sequence_length"}
        elif "pixel_values" in input_name or "image" in input_name:
            return {0: "batch_size"}
        elif "input_values" in input_name:
            return {0: "batch_size", 1: "sequence_length"}
        elif "input_features" in input_name:
            return {0: "batch_size", 2: "sequence_length"}
        elif "bbox" in input_name:
            return {0: "batch_size", 1: "sequence_length"}
        elif "past_key_values" in input_name:
            # Past key values have complex structure
            return {0: "batch_size", 2: "past_sequence_length"}
        else:
            # Default for unknown inputs
            return {0: "batch_size"}
    
    def _get_dynamic_axes_for_output(self, output_name: str) -> dict[int, str]:
        """Get dynamic axes for a specific output."""
        if output_name == "logits":
            if "classification" in self.task:
                if "token" in self.task:
                    return {0: "batch_size", 1: "sequence_length"}
                else:
                    return {0: "batch_size"}
            else:
                return {0: "batch_size", 1: "sequence_length"}
        elif output_name == "last_hidden_state":
            return {0: "batch_size", 1: "sequence_length"}
        elif output_name == "pooler_output":
            return {0: "batch_size"}
        elif output_name in ["start_logits", "end_logits"] or output_name == "encoder_last_hidden_state":
            return {0: "batch_size", 1: "sequence_length"}
        elif "past_key_values" in output_name:
            return {0: "batch_size", 2: "sequence_length"}
        elif output_name in ["text_embeds", "image_embeds", "sentence_embedding"] or output_name == "pred_boxes" or output_name == "pred_masks":
            return {0: "batch_size"}
        else:
            # Default
            return {0: "batch_size"}
    
    def _get_past_key_value_names(self, is_input: bool = True) -> list[str]:
        """Get past key value tensor names."""
        num_layers = self.config_dict.get(
            "num_hidden_layers",
            self.config_dict.get("n_layer", 12)
        )
        
        names = []
        for i in range(num_layers):
            names.extend([
                f"past_key_values.{i}.key",
                f"past_key_values.{i}.value"
            ])
        
        return names
    
    def _generate_dummy_past_key_values(
        self,
        batch_size: int,
        past_seq_length: int = 10
    ) -> dict[str, torch.Tensor]:
        """Generate dummy past key values."""
        num_layers = self.config_dict.get(
            "num_hidden_layers",
            self.config_dict.get("n_layer", 12)
        )
        hidden_size = self.config_dict.get(
            "hidden_size",
            self.config_dict.get("d_model", 768)
        )
        num_heads = self.config_dict.get(
            "num_attention_heads",
            self.config_dict.get("n_head", 12)
        )
        head_dim = hidden_size // num_heads
        
        past_kv = {}
        for i in range(num_layers):
            # Key and value shape: (batch_size, num_heads, past_seq_length, head_dim)
            past_kv[f"past_key_values.{i}.key"] = torch.randn(
                batch_size, num_heads, past_seq_length, head_dim
            )
            past_kv[f"past_key_values.{i}.value"] = torch.randn(
                batch_size, num_heads, past_seq_length, head_dim
            )
        
        return past_kv
    
    def _get_image_size(self) -> int:
        """Get image size from config."""
        # Try different config keys
        for key in ["image_size", "vision_config.image_size", "img_size", "input_size"]:
            if key in self.config_dict:
                size = self.config_dict[key]
                if isinstance(size, int):
                    return size
                elif isinstance(size, (list, tuple)):
                    return size[0]
        
        # Check vision config
        if "vision_config" in self.config_dict:
            vision_config = self.config_dict["vision_config"]
            if isinstance(vision_config, dict):
                for key in ["image_size", "img_size", "input_size"]:
                    if key in vision_config:
                        size = vision_config[key]
                        if isinstance(size, int):
                            return size
                        elif isinstance(size, (list, tuple)):
                            return size[0]
        
        # Default
        return 224
    
    def flatten_output_collection_property(self, name: str, field: str) -> dict[str, Any]:
        """
        Flatten nested output specifications.
        
        This method is for compatibility with HuggingFace Optimum.
        
        Args:
            name: Property name (e.g., "outputs")
            field: Field to extract
            
        Returns:
            Flattened dictionary
        """
        if name == "inputs":
            return self.inputs
        elif name == "outputs":
            return self.outputs
        else:
            return {}
    
    @property
    def default_onnx_opset(self) -> int:
        """Get default ONNX opset version."""
        return self.DEFAULT_ONNX_OPSET
    
    @property
    def atol_for_validation(self) -> float:
        """Get tolerance for validation."""
        return self.ATOL_FOR_VALIDATION
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UniversalOnnxConfig(\n"
            f"  model_type={self.model_type},\n"
            f"  task={self.task},\n"
            f"  task_family={self.task_family},\n"
            f"  use_past={self.use_past},\n"
            f"  inputs={list(self.get_input_names())},\n"
            f"  outputs={list(self.get_output_names())}\n"
            f")"
        )