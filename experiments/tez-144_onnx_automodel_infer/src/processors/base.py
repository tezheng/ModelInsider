"""
Base ONNX Processor Implementation

This module provides the abstract base class for all ONNX processors.
It defines the common interface and functionality shared by all
ONNX processor implementations.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Import type definitions
try:
    # When imported as part of package structure
    from ..onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXConfigurationError,
        ONNXProcessorError,
        ONNXShapeError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        validate_tensor_spec,
    )
except ImportError:
    # When imported directly in tests
    from onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXConfigurationError,
        ONNXProcessorError,
        ONNXShapeError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        validate_tensor_spec,
    )

# Configure logging
logger = logging.getLogger(__name__)


# Constants for default values
class ProcessorDefaults:
    """Constants for default processor configuration values."""

    # Text processing defaults
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_SEQUENCE_LENGTH = 128
    DEFAULT_VOCAB_SIZE = 30522  # BERT vocab size
    BERT_MAX_LENGTH = 512

    # Image processing defaults
    DEFAULT_IMAGE_HEIGHT = 224
    DEFAULT_IMAGE_WIDTH = 224
    DEFAULT_NUM_CHANNELS = 3
    IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
    FALLBACK_NORMALIZE_VALUE = 0.5

    # Audio processing defaults
    DEFAULT_AUDIO_SEQUENCE_LENGTH = 16000  # 1 second at 16kHz
    DEFAULT_SAMPLING_RATE = 16000
    DEFAULT_FEATURE_SIZE = 1
    DEFAULT_N_FFT = 400
    DEFAULT_FRAME_SAMPLING_RATE = 4
    AUDIO_PADDING_VALUE = 0.0

    # Video processing defaults
    DEFAULT_NUM_FRAMES = 16

    # ONNX processing defaults
    DEFAULT_ONNX_OPSET_VERSION = 17
    SPEEDUP_FACTOR = 40  # Documented performance improvement
    MEMORY_LEAK_THRESHOLD_MB = 100  # Memory leak detection threshold
    PERFORMANCE_ITERATIONS = 1000  # Performance test iterations

    # Tensor shape validation
    MIN_SHAPE_RANK = 2
    VALID_CHANNEL_COUNTS: list[int] = [1, 3, 4]  # Valid image channel counts
    VIDEO_SHAPE_RANK = 5  # Expected video tensor rank (NCTHW)
    IMAGE_SHAPE_RANK = 4  # Expected image tensor rank (NCHW)
    TEXT_AUDIO_SHAPE_THRESHOLD = 1000  # Threshold to distinguish audio from text


class BaseONNXProcessor(ABC):
    """
    Abstract base class for all ONNX processors.

    This class defines the common interface and functionality shared by all
    ONNX processor implementations. It provides fixed-shape optimization,
    validation, and error handling capabilities.

    Attributes:
        base_processor: Original HuggingFace processor instance
        modality_config: Configuration for this processor's modality
        batch_size: Fixed batch size for ONNX optimization
        _validation_enabled: Whether to validate inputs and outputs
    """

    def __init__(
        self,
        base_processor: Any,
        modality_config: ModalityConfig,
        validation_enabled: bool = True,
    ):
        """
        Initialize base ONNX processor.

        Args:
            base_processor: HuggingFace processor to wrap
            modality_config: Configuration for this modality
            validation_enabled: Whether to enable input/output validation

        Raises:
            ONNXConfigurationError: If configuration is invalid
        """
        self.base_processor = base_processor
        self.modality_config = modality_config
        self.batch_size = modality_config.batch_size
        self._validation_enabled = validation_enabled

        # Validate configuration
        self._validate_configuration()

        logger.info(
            f"Initialized {self.__class__.__name__} for {modality_config.modality_type.value} "
            f"modality with batch_size={self.batch_size}"
        )

    @abstractmethod
    def __call__(self, inputs: Any, **kwargs: Any) -> ProcessorResult:
        """
        Process inputs and return ONNX-compatible tensors.

        Args:
            inputs: Input data (text, images, audio, etc.)
            **kwargs: Additional processing arguments

        Returns:
            Dictionary of tensor names to arrays ready for ONNX inference
        """
        pass

    @abstractmethod
    def preprocess(self, inputs: Any, **kwargs: Any) -> TensorDict:
        """
        Preprocess inputs into ONNX tensor format with fixed shapes.

        Args:
            inputs: Raw input data
            **kwargs: Processing parameters

        Returns:
            Dictionary of tensor names to NumPy arrays
        """
        pass

    def _validate_configuration(self) -> None:
        """Validate processor configuration."""
        if not self.modality_config.tensors:
            raise ONNXConfigurationError(
                "tensors", [], "At least one tensor specification required"
            )

        if self.batch_size <= 0:
            raise ONNXConfigurationError(
                "batch_size", self.batch_size, "Batch size must be positive"
            )

        # Validate tensor specifications
        for tensor in self.modality_config.tensors:
            errors = validate_tensor_spec(tensor)
            if errors:
                raise ONNXConfigurationError(
                    f"tensor_{tensor.name}",
                    tensor.to_dict(),
                    f"Invalid tensor specification: {'; '.join(errors)}",
                )

    def _validate_inputs(self, inputs: Any) -> None:
        """Validate input data before processing."""
        if not self._validation_enabled:
            return

        if inputs is None:
            raise ONNXProcessorError("Inputs cannot be None", processor_type=self.__class__.__name__)

        # Additional validation for specific input types
        if isinstance(inputs, list | tuple) and len(inputs) == 0:
            raise ONNXProcessorError("Input list cannot be empty", processor_type=self.__class__.__name__)

        if isinstance(inputs, str) and len(inputs.strip()) == 0:
            raise ONNXProcessorError("Text input cannot be empty or whitespace-only", processor_type=self.__class__.__name__)

    def _validate_outputs(self, outputs: TensorDict) -> None:
        """Validate output tensors match expected specifications."""
        if not self._validation_enabled:
            return

        if not outputs:
            raise ONNXProcessorError("Output tensor dictionary cannot be empty", processor_type=self.__class__.__name__)

        expected_tensors = {t.name: t for t in self.modality_config.input_tensors}

        for tensor_name, tensor_array in outputs.items():
            # Validate tensor array is not None and has valid shape
            if tensor_array is None:
                raise ONNXProcessorError(f"Output tensor '{tensor_name}' cannot be None", processor_type=self.__class__.__name__)

            if not hasattr(tensor_array, "shape") or not hasattr(tensor_array, "dtype"):
                raise ONNXProcessorError(
                    f"Output tensor '{tensor_name}' must be a numpy array or similar tensor",
                    processor_type=self.__class__.__name__
                )

            if tensor_array.size == 0:
                raise ONNXProcessorError(f"Output tensor '{tensor_name}' cannot be empty", processor_type=self.__class__.__name__)

            if tensor_name not in expected_tensors:
                logger.warning(f"Unexpected output tensor: {tensor_name}")
                continue

            expected_spec = expected_tensors[tensor_name]
            if not expected_spec.is_compatible_shape(list(tensor_array.shape)):
                raise ONNXShapeError(
                    expected_spec.shape, list(tensor_array.shape), tensor_name
                )

    def _ensure_fixed_batch_size(self, tensor_dict: TensorDict) -> TensorDict:
        """
        Ensure all tensors have the correct fixed batch size.

        Args:
            tensor_dict: Dictionary of tensors to adjust

        Returns:
            Dictionary with adjusted batch sizes
        """
        adjusted = {}

        for name, tensor in tensor_dict.items():
            if len(tensor.shape) == 0:
                # Scalar tensor - no batch dimension
                adjusted[name] = tensor
                continue

            current_batch = tensor.shape[0]

            if current_batch == self.batch_size:
                # Already correct batch size
                adjusted[name] = tensor
            elif current_batch < self.batch_size:
                # Pad batch dimension
                pad_width = [(0, self.batch_size - current_batch)] + [(0, 0)] * (
                    len(tensor.shape) - 1
                )
                adjusted[name] = np.pad(
                    tensor, pad_width, mode="constant", constant_values=0
                )
            else:
                # Truncate batch dimension
                adjusted[name] = tensor[: self.batch_size]

        return adjusted

    def get_input_spec(self, tensor_name: str) -> TensorSpec | None:
        """Get tensor specification for a given input tensor name."""
        for tensor in self.modality_config.input_tensors:
            if tensor.name == tensor_name:
                return tensor
        return None

    def get_output_spec(self, tensor_name: str) -> TensorSpec | None:
        """Get tensor specification for a given output tensor name."""
        for tensor in self.modality_config.output_tensors:
            if tensor.name == tensor_name:
                return tensor
        return None

    @property
    def modality_type(self) -> ModalityType:
        """Get the modality type for this processor."""
        return self.modality_config.modality_type

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names handled by this processor."""
        return self.modality_config.tensor_names
