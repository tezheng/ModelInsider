"""
ONNX Processor Types and Data Classes

This module provides type definitions, enums, data classes, and exception classes
for the ONNX Auto Processor system. It contains all the shared types used across
the various processor implementations.

Key Components:
- ModalityType: Enum for different input modalities (text, image, audio, video)
- TensorType: Enum for ONNX tensor data types with utility methods
- Processor data classes: Structured representation of processor configurations
- Exception classes: Specific exceptions for ONNX processing errors
- Input/Output specifications: Type-safe representations of model I/O

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

# Type aliases for better code readability
type PathLike = str | Path
type TensorDict = dict[str, NDArray[Any]]
type ShapeDict = dict[str, list[int]]
type ConfigDict = dict[str, Any]
type ProcessorResult = dict[str, Any]

# Constants for type processing defaults
class TypeDefaults:
    """Constants for type processing and validation."""
    
    # Shape analysis constants
    MIN_TENSOR_RANK = 2
    AUDIO_WAVEFORM_THRESHOLD = 1000  # Threshold to distinguish audio from text by sequence length
    VALID_IMAGE_CHANNELS: ClassVar[list[int]] = [1, 3, 4]  # Common image channel counts
    VIDEO_TENSOR_RANK = 5  # Expected video tensor rank (NCTHW)
    IMAGE_TENSOR_RANK = 4  # Expected image tensor rank (NCHW)
    AUDIO_FEATURE_RANK = 3  # Audio feature tensor rank
    TEXT_TENSOR_RANK = 2   # Text tensor rank
    
    # Default modality settings
    DEFAULT_BATCH_SIZE = 1


class ModalityType(Enum):
    """
    Enumeration of supported input modalities for ONNX models.
    
    This enum represents the different types of data that can be processed
    by ONNX models, with each modality having specific processing requirements
    and tensor patterns.
    
    Attributes:
        TEXT: Natural language text processing (tokenization)
        IMAGE: Computer vision image processing
        AUDIO: Audio signal processing and feature extraction
        VIDEO: Video frame sequence processing
        MULTIMODAL: Multiple modalities combined (e.g., CLIP, Whisper)
        UNKNOWN: Unrecognized modality requiring fallback handling
    """
    
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_tensor_name(cls, tensor_name: str) -> ModalityType:
        """
        Detect modality type from tensor name patterns.
        
        Args:
            tensor_name: Name of the input tensor
            
        Returns:
            Detected modality type or UNKNOWN if no pattern matches
            
        Examples:
            >>> ModalityType.from_tensor_name("input_ids")
            ModalityType.TEXT
            >>> ModalityType.from_tensor_name("pixel_values")
            ModalityType.IMAGE
        """
        tensor_name_lower = tensor_name.lower()
        
        # Text patterns
        if any(pattern in tensor_name_lower for pattern in [
            'input_ids', 'token', 'text', 'attention_mask', 'token_type_ids'
        ]):
            return cls.TEXT
            
        # Image patterns
        elif any(pattern in tensor_name_lower for pattern in [
            'pixel', 'image', 'vision', 'visual'
        ]):
            return cls.IMAGE
            
        # Audio patterns
        elif any(pattern in tensor_name_lower for pattern in [
            'audio', 'wave', 'mel', 'spectrogram', 'input_values', 'input_features'
        ]):
            return cls.AUDIO
            
        # Video patterns
        elif any(pattern in tensor_name_lower for pattern in [
            'video', 'frames', 'clip', 'temporal'
        ]):
            return cls.VIDEO
            
        return cls.UNKNOWN
    
    @classmethod
    def from_tensor_shape(cls, shape: list[int]) -> ModalityType:
        """
        Detect modality type from tensor shape patterns.
        
        Args:
            shape: List of tensor dimensions
            
        Returns:
            Most likely modality type based on shape pattern
            
        Shape Patterns:
            - [batch, seq_len]: TEXT (2D)
            - [batch, channels, height, width]: IMAGE (4D NCHW)
            - [batch, seq_len, features]: AUDIO (3D) or TEXT
            - [batch, channels, frames, height, width]: VIDEO (5D NCTHW)
        """
        if not shape or len(shape) < TypeDefaults.MIN_TENSOR_RANK:
            return cls.UNKNOWN
            
        if len(shape) == TypeDefaults.TEXT_TENSOR_RANK:
            # 2D tensor - could be text or audio waveform
            # Audio waveforms typically have much larger second dimension
            if len(shape) >= TypeDefaults.MIN_TENSOR_RANK and shape[1] > TypeDefaults.AUDIO_WAVEFORM_THRESHOLD:  # Audio samples are typically much larger
                return cls.AUDIO
            else:
                return cls.TEXT
        elif len(shape) == TypeDefaults.AUDIO_FEATURE_RANK:
            # 3D tensor - could be audio features or text with features
            return cls.AUDIO
        elif len(shape) == TypeDefaults.IMAGE_TENSOR_RANK:
            # 4D tensor - likely image (NCHW format)
            if shape[1] in TypeDefaults.VALID_IMAGE_CHANNELS:  # Common channel counts
                return cls.IMAGE
            return cls.UNKNOWN
        elif len(shape) == TypeDefaults.VIDEO_TENSOR_RANK:
            # 5D tensor - likely video (NCTHW format)
            return cls.VIDEO
            
        return cls.UNKNOWN
    
    def is_unimodal(self) -> bool:
        """Check if this modality represents a single modality type."""
        return self in {self.TEXT, self.IMAGE, self.AUDIO, self.VIDEO}
    
    def requires_sequence_padding(self) -> bool:
        """Check if this modality requires sequence padding for fixed shapes."""
        return self in {self.TEXT, self.AUDIO}
    
    def default_batch_size(self) -> int:
        """Get the default batch size for this modality."""
        return TypeDefaults.DEFAULT_BATCH_SIZE  # All modalities default to batch_size=1 for ONNX optimization


class TensorType(IntEnum):
    """
    ONNX tensor data types mapped to their numeric values.
    
    This enum provides a mapping from ONNX's internal tensor type representations
    to human-readable names with utility methods for type conversion and validation.
    
    Values correspond to ONNX TensorProto.DataType enumeration.
    """
    
    UNDEFINED = 0
    FLOAT32 = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    
    def to_numpy_dtype(self) -> np.dtype:
        """
        Convert ONNX tensor type to NumPy dtype.
        
        Returns:
            Corresponding NumPy dtype
            
        Raises:
            ValueError: If tensor type is not supported for NumPy conversion
        """
        type_mapping = {
            self.FLOAT32: np.float32,
            self.UINT8: np.uint8,
            self.INT8: np.int8,
            self.UINT16: np.uint16,
            self.INT16: np.int16,
            self.INT32: np.int32,
            self.INT64: np.int64,
            self.BOOL: np.bool_,
            self.FLOAT16: np.float16,
            self.DOUBLE: np.float64,
            self.UINT32: np.uint32,
            self.UINT64: np.uint64,
        }
        
        if self not in type_mapping:
            raise ValueError(f"Cannot convert ONNX type {self.name} to NumPy dtype")
        
        return np.dtype(type_mapping[self])
    
    def to_python_type(self) -> type:
        """Convert ONNX tensor type to Python type."""
        type_mapping = {
            self.FLOAT32: float,
            self.FLOAT16: float,
            self.DOUBLE: float,
            self.INT8: int,
            self.INT16: int,
            self.INT32: int,
            self.INT64: int,
            self.UINT8: int,
            self.UINT16: int,
            self.UINT32: int,
            self.UINT64: int,
            self.BOOL: bool,
            self.STRING: str,
        }
        
        return type_mapping.get(self, object)
    
    def is_integer(self) -> bool:
        """Check if this tensor type represents an integer type."""
        return self in {
            self.INT8, self.INT16, self.INT32, self.INT64,
            self.UINT8, self.UINT16, self.UINT32, self.UINT64
        }
    
    def is_floating_point(self) -> bool:
        """Check if this tensor type represents a floating point type."""
        return self in {self.FLOAT16, self.FLOAT32, self.DOUBLE, self.BFLOAT16}
    
    def size_in_bytes(self) -> int:
        """Get the size of this tensor type in bytes."""
        size_mapping = {
            self.FLOAT32: 4,
            self.UINT8: 1,
            self.INT8: 1,
            self.UINT16: 2,
            self.INT16: 2,
            self.INT32: 4,
            self.INT64: 8,
            self.BOOL: 1,
            self.FLOAT16: 2,
            self.DOUBLE: 8,
            self.UINT32: 4,
            self.UINT64: 8,
            self.COMPLEX64: 8,
            self.COMPLEX128: 16,
            self.BFLOAT16: 2,
        }
        
        return size_mapping.get(self, 0)


@dataclass(frozen=True)
class TensorSpec:
    """
    Specification for a single tensor in an ONNX model.
    
    This immutable data class provides a complete specification of a tensor
    including its name, shape, data type, and associated metadata.
    
    Attributes:
        name: Tensor name in the ONNX model
        shape: List of dimension sizes (-1 for dynamic dimensions)
        dtype: ONNX tensor data type
        modality: Associated modality type
        is_input: Whether this is an input or output tensor
        description: Optional human-readable description
    """
    
    name: str
    shape: list[int]
    dtype: TensorType
    modality: ModalityType = ModalityType.UNKNOWN
    is_input: bool = True
    description: str | None = None
    
    def __post_init__(self) -> None:
        """Validate tensor specification after initialization."""
        if not self.name:
            raise ValueError("Tensor name cannot be empty")
        
        if not self.shape:
            raise ValueError("Tensor shape cannot be empty")
        
        if any(dim < -1 for dim in self.shape):
            raise ValueError("Invalid dimension size: dimensions must be >= -1")
    
    @property
    def is_dynamic(self) -> bool:
        """Check if tensor has any dynamic dimensions."""
        return -1 in self.shape
    
    @property
    def fixed_shape(self) -> list[int]:
        """Get shape with dynamic dimensions replaced by reasonable defaults."""
        return [dim if dim != -1 else 1 for dim in self.shape]
    
    @property
    def rank(self) -> int:
        """Get the number of dimensions (rank) of this tensor."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Calculate total number of elements (using fixed shape for dynamic dims)."""
        result = 1
        for dim in self.fixed_shape:
            result *= dim
        return result
    
    def memory_size(self) -> int:
        """Calculate memory size in bytes for this tensor."""
        return self.size * self.dtype.size_in_bytes()
    
    def is_compatible_shape(self, other_shape: list[int]) -> bool:
        """
        Check if another shape is compatible with this tensor's shape.
        
        Compatible means same rank and each dimension either matches or
        this tensor has a dynamic dimension (-1).
        """
        if len(other_shape) != len(self.shape):
            return False
        
        return all(
            self_dim == -1 or self_dim == other_dim
            for self_dim, other_dim in zip(self.shape, other_shape, strict=False)
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tensor spec to dictionary for serialization."""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": int(self.dtype),
            "dtype_name": self.dtype.name,
            "modality": self.modality.value,
            "is_input": self.is_input,
            "description": self.description,
            "is_dynamic": self.is_dynamic,
            "memory_size": self.memory_size()
        }


@dataclass
class ModalityConfig:
    """
    Configuration for a specific modality within an ONNX model.
    
    This class encapsulates all the configuration needed to process
    a specific modality type, including tensor specifications and
    processing parameters.
    
    Attributes:
        modality_type: Type of modality (text, image, etc.)
        tensors: List of tensor specifications for this modality
        batch_size: Fixed batch size for ONNX optimization
        config: Additional modality-specific configuration
        processor_class: Name of the HuggingFace processor class
    """
    
    modality_type: ModalityType
    tensors: list[TensorSpec] = field(default_factory=list)
    batch_size: int = 1
    config: ConfigDict = field(default_factory=dict)
    processor_class: str | None = None
    
    def __post_init__(self) -> None:
        """Validate modality configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not self.tensors:
            raise ValueError("At least one tensor must be specified")
        
        # Validate that all tensors belong to this modality
        for tensor in self.tensors:
            if tensor.modality != ModalityType.UNKNOWN and tensor.modality != self.modality_type:
                raise ValueError(
                    f"Tensor {tensor.name} modality {tensor.modality} "
                    f"doesn't match config modality {self.modality_type}"
                )
    
    @property
    def input_tensors(self) -> list[TensorSpec]:
        """Get only input tensors for this modality."""
        return [t for t in self.tensors if t.is_input]
    
    @property
    def output_tensors(self) -> list[TensorSpec]:
        """Get only output tensors for this modality."""
        return [t for t in self.tensors if not t.is_input]
    
    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names for this modality."""
        return [t.name for t in self.tensors]
    
    @property
    def has_dynamic_shapes(self) -> bool:
        """Check if any tensor in this modality has dynamic shapes."""
        return any(t.is_dynamic for t in self.tensors)
    
    def get_tensor_by_name(self, name: str) -> TensorSpec | None:
        """Get tensor specification by name."""
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        return None
    
    def total_memory_size(self) -> int:
        """Calculate total memory size for all tensors in this modality."""
        return sum(t.memory_size() for t in self.tensors)
    
    def update_batch_size(self, new_batch_size: int) -> None:
        """
        Update batch size for all tensors in this modality.
        
        This updates the first dimension of all tensors to the new batch size,
        assuming the first dimension is the batch dimension.
        """
        if new_batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        self.batch_size = new_batch_size
        
        # Update tensor shapes
        updated_tensors = []
        for tensor in self.tensors:
            if tensor.shape and tensor.shape[0] != -1:  # Skip dynamic batch dims
                new_shape = [new_batch_size, *tensor.shape[1:]]
                updated_tensor = TensorSpec(
                    name=tensor.name,
                    shape=new_shape,
                    dtype=tensor.dtype,
                    modality=tensor.modality,
                    is_input=tensor.is_input,
                    description=tensor.description
                )
                updated_tensors.append(updated_tensor)
            else:
                updated_tensors.append(tensor)
        
        self.tensors = updated_tensors


@dataclass
class ProcessorMetadata:
    """
    Complete metadata for an ONNX model processor.
    
    This class provides a comprehensive description of an ONNX model's
    processing requirements, including all modalities, configurations,
    and model information.
    
    Attributes:
        model_name: Original model name or identifier
        model_type: HuggingFace model type (e.g., "bert", "clip")
        task: Primary task this model performs
        modalities: Configuration for each modality
        is_multimodal: Whether model processes multiple modalities
        onnx_opset_version: ONNX opset version used
        metadata_source: Source of the metadata (onnx, json, auto-detected)
    """
    
    model_name: str
    model_type: str
    task: str | None = None
    modalities: dict[str, ModalityConfig] = field(default_factory=dict)
    is_multimodal: bool = False
    onnx_opset_version: int = 17
    metadata_source: str = "unknown"
    
    def __post_init__(self) -> None:
        """Validate processor metadata after initialization."""
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        
        # Update is_multimodal based on modalities
        unimodal_count = sum(
            1 for config in self.modalities.values() 
            if config.modality_type.is_unimodal()
        )
        self.is_multimodal = unimodal_count > 1
        
        # Validate modality consistency
        if self.is_multimodal and len(self.modalities) < 2:
            raise ValueError("Multimodal model must have at least 2 modalities")
    
    @property
    def modality_types(self) -> set[ModalityType]:
        """Get set of all modality types in this model."""
        return {config.modality_type for config in self.modalities.values()}
    
    @property
    def all_input_tensors(self) -> list[TensorSpec]:
        """Get all input tensors across all modalities."""
        tensors = []
        for config in self.modalities.values():
            tensors.extend(config.input_tensors)
        return tensors
    
    @property
    def all_output_tensors(self) -> list[TensorSpec]:
        """Get all output tensors across all modalities."""
        tensors = []
        for config in self.modalities.values():
            tensors.extend(config.output_tensors)
        return tensors
    
    @property
    def total_input_count(self) -> int:
        """Get total number of input tensors."""
        return len(self.all_input_tensors)
    
    @property
    def total_output_count(self) -> int:
        """Get total number of output tensors."""
        return len(self.all_output_tensors)
    
    def get_modality_config(self, modality_type: ModalityType) -> ModalityConfig | None:
        """Get configuration for a specific modality type."""
        for config in self.modalities.values():
            if config.modality_type == modality_type:
                return config
        return None
    
    def has_modality(self, modality_type: ModalityType) -> bool:
        """Check if model supports a specific modality."""
        return modality_type in self.modality_types
    
    def estimate_memory_usage(self) -> int:
        """Estimate total memory usage in bytes for all tensors."""
        return sum(config.total_memory_size() for config in self.modalities.values())


# Protocol definitions for type checking
class ProcessorProtocol(Protocol):
    """Protocol defining the interface for ONNX processors."""
    
    def __call__(self, inputs: Any, **kwargs: Any) -> ProcessorResult:
        """Process inputs and return ONNX-compatible tensors."""
        ...
    
    def preprocess(self, inputs: Any, **kwargs: Any) -> TensorDict:
        """Preprocess inputs into ONNX tensor format."""
        ...


class TokenizerProtocol(ProcessorProtocol, Protocol):
    """Protocol for ONNX tokenizer processors."""
    
    batch_size: int
    sequence_length: int
    
    def encode(self, text: str | list[str], **kwargs: Any) -> ProcessorResult:
        """Encode text into token tensors."""
        ...
    
    def decode(self, token_ids: NDArray[Any], **kwargs: Any) -> str | list[str]:
        """Decode token tensors back to text."""
        ...


class ImageProcessorProtocol(ProcessorProtocol, Protocol):
    """Protocol for ONNX image processor."""
    
    batch_size: int
    height: int
    width: int
    num_channels: int


class AudioProcessorProtocol(ProcessorProtocol, Protocol):
    """Protocol for ONNX audio processor."""
    
    batch_size: int
    sequence_length: int
    sampling_rate: int


class VideoProcessorProtocol(ProcessorProtocol, Protocol):
    """Protocol for ONNX video processor."""
    
    batch_size: int
    num_frames: int
    height: int
    width: int
    num_channels: int


# Exception classes
class ONNXProcessorError(Exception):
    """Base exception for ONNX processor errors."""
    
    def __init__(self, message: str, processor_type: str | None = None):
        self.processor_type = processor_type
        super().__init__(message)


class ONNXModelLoadError(ONNXProcessorError):
    """Exception raised when ONNX model cannot be loaded or parsed."""
    
    def __init__(self, model_path: PathLike, original_error: Exception | None = None):
        self.model_path = str(model_path)
        self.original_error = original_error
        
        message = f"Failed to load ONNX model from {self.model_path}"
        if original_error:
            message += f": {original_error}"
        
        super().__init__(message, "model_loader")


class ONNXMetadataError(ONNXProcessorError):
    """Exception raised when metadata extraction or validation fails."""
    
    def __init__(self, message: str, metadata_source: str | None = None):
        self.metadata_source = metadata_source
        
        if metadata_source:
            message = f"Metadata error from {metadata_source}: {message}"
        
        super().__init__(message, "metadata_extractor")


class ONNXShapeError(ONNXProcessorError):
    """Exception raised for tensor shape mismatches or validation failures."""
    
    def __init__(self, expected_shape: list[int], actual_shape: list[int], 
                 tensor_name: str | None = None):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.tensor_name = tensor_name
        
        message = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        if tensor_name:
            message = f"Tensor '{tensor_name}' {message}"
        
        super().__init__(message, "shape_validator")


class ONNXUnsupportedModalityError(ONNXProcessorError):
    """Exception raised when encountering unsupported modality types."""
    
    def __init__(self, modality: ModalityType, supported_modalities: list[ModalityType] | None = None):
        self.modality = modality
        self.supported_modalities = supported_modalities or []
        
        message = f"Unsupported modality: {modality.value}"
        if self.supported_modalities:
            supported_names = [m.value for m in self.supported_modalities]
            message += f". Supported modalities: {', '.join(supported_names)}"
        
        super().__init__(message, "modality_detector")


class ONNXProcessorNotFoundError(ONNXProcessorError):
    """Exception raised when required processor cannot be found or loaded."""
    
    def __init__(self, processor_type: str, model_path: PathLike | None = None):
        self.processor_type = processor_type
        self.model_path = str(model_path) if model_path else None
        
        message = f"Processor of type '{processor_type}' not found"
        if model_path:
            message += f" for model at {model_path}"
        
        super().__init__(message, processor_type)


class ONNXConfigurationError(ONNXProcessorError):
    """Exception raised for invalid processor configurations."""
    
    def __init__(self, config_name: str, config_value: Any, reason: str):
        self.config_name = config_name
        self.config_value = config_value
        self.reason = reason
        
        message = f"Invalid configuration '{config_name}' = {config_value}: {reason}"
        super().__init__(message, "configurator")


# Utility functions
def validate_tensor_spec(spec: TensorSpec) -> list[str]:
    """
    Validate a tensor specification and return list of validation errors.
    
    Args:
        spec: Tensor specification to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not spec.name:
        errors.append("Tensor name cannot be empty")
    
    if not spec.shape:
        errors.append("Tensor shape cannot be empty")
    
    if any(dim < -1 for dim in spec.shape):
        errors.append("Invalid dimension size: dimensions must be >= -1")
    
    if spec.dtype == TensorType.UNDEFINED:
        errors.append("Tensor data type cannot be UNDEFINED")
    
    # Shape-specific validations based on modality
    if spec.modality == ModalityType.TEXT:
        if len(spec.shape) < 2:
            errors.append("Text tensors should have at least 2 dimensions [batch, sequence]")
    elif spec.modality == ModalityType.IMAGE:
        if len(spec.shape) not in [3, 4]:
            errors.append("Image tensors should have 3 or 4 dimensions")
        if len(spec.shape) == 4 and spec.shape[1] not in [1, 3, 4]:
            errors.append("Image tensors should have 1, 3, or 4 channels")
    elif spec.modality == ModalityType.VIDEO and len(spec.shape) != 5:
            errors.append("Video tensors should have 5 dimensions [batch, channels, frames, height, width]")
    
    return errors


def create_tensor_spec_from_dict(tensor_dict: dict[str, Any]) -> TensorSpec:
    """
    Create a TensorSpec from a dictionary representation.
    
    Args:
        tensor_dict: Dictionary containing tensor specification
        
    Returns:
        TensorSpec instance
        
    Raises:
        ONNXConfigurationError: If dictionary is invalid
    """
    try:
        return TensorSpec(
            name=tensor_dict["name"],
            shape=tensor_dict["shape"],
            dtype=TensorType(tensor_dict.get("dtype", TensorType.FLOAT32)),
            modality=ModalityType(tensor_dict.get("modality", ModalityType.UNKNOWN)),
            is_input=tensor_dict.get("is_input", True),
            description=tensor_dict.get("description")
        )
    except (KeyError, ValueError, TypeError) as e:
        raise ONNXConfigurationError(
            "tensor_dict", tensor_dict, f"Invalid tensor dictionary: {e}"
        ) from e


def merge_processor_configs(base_config: ConfigDict, override_config: ConfigDict) -> ConfigDict:
    """
    Merge two processor configuration dictionaries with override precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration with overrides applied
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_processor_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged