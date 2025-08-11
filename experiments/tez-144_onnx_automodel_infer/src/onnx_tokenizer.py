"""ONNX Tokenizer Wrapper with Auto-Detection of Static Input Dimensions"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from transformers import BatchEncoding, PreTrainedTokenizerBase


def parse_onnx_input_shapes(onnx_model: str | Path | onnx.ModelProto) -> dict[str, list[int]]:
    """
    Parse input shapes from an ONNX model.
    
    Args:
        onnx_model: Path to ONNX file, or loaded ONNX ModelProto
        
    Returns:
        Dictionary mapping input names to their shapes
    """
    # Load model if path is provided
    if isinstance(onnx_model, str | Path):
        model = onnx.load(str(onnx_model))
    else:
        model = onnx_model
    
    input_shapes = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                # Dynamic dimension - use -1 to indicate
                shape.append(-1)
            else:
                shape.append(-1)
        input_shapes[name] = shape
    
    return input_shapes


class ONNXTokenizer:
    """
    Tokenizer wrapper that enforces fixed batch size and sequence length for ONNX models.

    This wrapper can automatically detect input shapes from ONNX models or accept
    manual shape specification.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        onnx_model: str | Path | onnx.ModelProto | Any | None = None,
        fixed_batch_size: int | None = None,
        fixed_sequence_length: int | None = None,
        padding_value: int = 0,
        attention_mask_value: int = 0,
    ):
        """
        Initialize the fixed shape tokenizer wrapper.

        Args:
            tokenizer: Base HuggingFace tokenizer
            onnx_model: ONNX model path, ModelProto, or ORTModel to auto-detect shapes
            fixed_batch_size: Override batch size (if None, auto-detect from model)
            fixed_sequence_length: Override sequence length (if None, auto-detect from model)
            padding_value: Value to use for padding tokens (usually pad_token_id)
            attention_mask_value: Value for padded attention mask positions
        """
        self.tokenizer = tokenizer
        self.padding_value = padding_value or tokenizer.pad_token_id or 0
        self.attention_mask_value = attention_mask_value
        
        # Auto-detect shapes from ONNX model if provided
        if onnx_model is not None and fixed_batch_size is None and fixed_sequence_length is None:
            self._auto_detect_shapes(onnx_model)
        else:
            # Use provided values or defaults
            self.fixed_batch_size = fixed_batch_size or 1
            self.fixed_sequence_length = fixed_sequence_length or 128
            
        # Validate that we have valid shapes
        if self.fixed_batch_size <= 0 or self.fixed_sequence_length <= 0:
            raise ValueError(
                f"Invalid shapes: batch_size={self.fixed_batch_size}, "
                f"sequence_length={self.fixed_sequence_length}"
            )
    
    def _auto_detect_shapes(self, onnx_model: Any):
        """Auto-detect input shapes from ONNX model."""
        try:
            # Handle ORTModel objects (from Optimum)
            if hasattr(onnx_model, "path"):
                # Use the newer 'path' property (Path object)
                model_path = onnx_model.path
                if model_path:
                    input_shapes = parse_onnx_input_shapes(model_path)
                else:
                    # Fall back to session inputs
                    input_shapes = self._parse_from_ort_session(onnx_model)
            elif hasattr(onnx_model, "model_path"):
                # Fallback to deprecated model_path if path doesn't exist
                model_path = onnx_model.model_path
                if model_path is None and hasattr(onnx_model, "model"):
                    # Try to get from the underlying session
                    model_path = onnx_model.model._model_path if hasattr(onnx_model.model, "_model_path") else None
                
                if model_path:
                    input_shapes = parse_onnx_input_shapes(model_path)
                else:
                    # Fall back to session inputs
                    input_shapes = self._parse_from_ort_session(onnx_model)
            elif hasattr(onnx_model, "get_inputs"):
                # ONNX Runtime InferenceSession
                input_shapes = {}
                for input_info in onnx_model.get_inputs():
                    input_shapes[input_info.name] = input_info.shape
            else:
                # Assume it's a path or ONNX ModelProto
                input_shapes = parse_onnx_input_shapes(onnx_model)
            
            # Extract batch size and sequence length from input_ids shape
            if "input_ids" in input_shapes:
                shape = input_shapes["input_ids"]
                if len(shape) >= 2:
                    self.fixed_batch_size = shape[0] if shape[0] > 0 else 1
                    self.fixed_sequence_length = shape[1] if shape[1] > 0 else 128
                else:
                    raise ValueError(f"Unexpected input_ids shape: {shape}")
            else:
                # Try to find any input with 2D shape
                for name, shape in input_shapes.items():
                    if len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
                        self.fixed_batch_size = shape[0]
                        self.fixed_sequence_length = shape[1]
                        print(f"Auto-detected shapes from '{name}': batch_size={shape[0]}, seq_length={shape[1]}")
                        break
                else:
                    raise ValueError(f"Could not auto-detect shapes from inputs: {list(input_shapes.keys())}")
                    
            print(f"Auto-detected ONNX input shapes: batch_size={self.fixed_batch_size}, "
                  f"sequence_length={self.fixed_sequence_length}")
                  
        except Exception as e:
            print(f"Warning: Could not auto-detect shapes from ONNX model: {e}")
            print("Using defaults: batch_size=1, sequence_length=128")
            self.fixed_batch_size = 1
            self.fixed_sequence_length = 128
    
    def _parse_from_ort_session(self, ort_model: Any) -> dict[str, list[int]]:
        """Parse shapes from ORTModel's session."""
        input_shapes = {}
        if hasattr(ort_model, "model") and hasattr(ort_model.model, "get_inputs"):
            # Access the underlying ONNX Runtime session
            for input_info in ort_model.model.get_inputs():
                input_shapes[input_info.name] = input_info.shape
        elif hasattr(ort_model, "session") and hasattr(ort_model.session, "get_inputs"):
            for input_info in ort_model.session.get_inputs():
                input_shapes[input_info.name] = input_info.shape
        return input_shapes

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        return_tensors: str | None = "np",
        **kwargs: Any,
    ) -> BatchEncoding:
        """
        Tokenize and pad/truncate to fixed dimensions.

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            return_tensors: Return type ("np" for ONNX, "pt" for PyTorch)
            **kwargs: Additional tokenizer arguments

        Returns:
            BatchEncoding with fixed shape tensors
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        # Handle batch size adjustment
        if len(text) < self.fixed_batch_size:
            # Pad with empty strings to reach fixed batch size
            text = text + [""] * (self.fixed_batch_size - len(text))
        elif len(text) > self.fixed_batch_size:
            # Truncate to fixed batch size
            print(
                f"Warning: Truncating {len(text)} inputs to fixed batch size {self.fixed_batch_size}"
            )
            text = text[: self.fixed_batch_size]

        # Force fixed shape parameters
        kwargs.update(
            {
                "padding": "max_length",
                "max_length": self.fixed_sequence_length,
                "truncation": True,
                "add_special_tokens": add_special_tokens,
                "return_tensors": return_tensors,
            }
        )

        # Tokenize with fixed parameters
        encoded = self.tokenizer(text, **kwargs)

        # Validate shapes
        self._validate_shapes(encoded)

        return encoded

    def _validate_shapes(self, encoded: BatchEncoding):
        """Validate that all tensors have expected fixed shapes."""
        for key, value in encoded.items():
            if hasattr(value, "shape"):
                expected_shape = (self.fixed_batch_size, self.fixed_sequence_length)
                actual_shape = value.shape[:2] if len(value.shape) >= 2 else value.shape

                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Tensor '{key}' has shape {value.shape}, "
                        f"expected ({self.fixed_batch_size}, {self.fixed_sequence_length}, ...)"
                    )

    def batch_decode(self, *args, **kwargs):
        """Pass through to underlying tokenizer."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Pass through to underlying tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    def __getattr__(self, name):
        """Forward other attributes to the underlying tokenizer."""
        return getattr(self.tokenizer, name)


def create_auto_shape_tokenizer(tokenizer: PreTrainedTokenizerBase, onnx_model: Any) -> ONNXTokenizer:
    """
    Create an ONNXTokenizer that automatically detects shapes from an ONNX model.
    
    Args:
        tokenizer: Base HuggingFace tokenizer
        onnx_model: ONNX model (path, ORTModel, or InferenceSession)
        
    Returns:
        ONNXTokenizer with auto-detected shapes
        
    Examples:
        >>> from transformers import AutoTokenizer
        >>> from optimum.onnxruntime import ORTModelForFeatureExtraction
        >>> 
        >>> # Load model and tokenizer
        >>> model = ORTModelForFeatureExtraction.from_pretrained("path/to/model")
        >>> base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> 
        >>> # Create tokenizer with auto-detected shapes
        >>> onnx_tokenizer = create_auto_shape_tokenizer(base_tokenizer, model)
        >>> print(f"Detected: batch_size={onnx_tokenizer.fixed_batch_size}, "
        ...       f"seq_length={onnx_tokenizer.fixed_sequence_length}")
    """
    return ONNXTokenizer(tokenizer=tokenizer, onnx_model=onnx_model)
