"""
ONNX Audio Processor Implementation

This module provides the ONNX-optimized audio processor for speech and audio models.
It wraps HuggingFace audio processors (Wav2Vec2, Whisper, etc.) and provides
fixed-shape audio preprocessing optimized for ONNX inference.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    # When imported as part of package structure
    from ..types import (
        ModalityConfig,
        ModalityType,
        ONNXProcessorError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from .base import BaseONNXProcessor, ProcessorDefaults
except ImportError:
    # When imported directly in tests
    from types import (
        ModalityConfig,
        ModalityType,
        ONNXProcessorError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from processors.base import BaseONNXProcessor, ProcessorDefaults

# Configure logging
logger = logging.getLogger(__name__)


class ONNXAudioProcessor(BaseONNXProcessor):
    """
    ONNX-optimized audio processor for speech and audio models.

    This class wraps HuggingFace audio processors (Wav2Vec2, Whisper, etc.) and
    provides fixed-shape audio preprocessing optimized for ONNX inference.

    Key Features:
    - Fixed sequence length for optimal ONNX performance
    - Automatic resampling and feature extraction
    - Support for raw waveforms and spectrograms
    - Padding and truncation handling
    - Multiple audio format support

    Attributes:
        sequence_length: Fixed sequence length for audio features
        sampling_rate: Target sampling rate for audio processing
        feature_size: Number of features per time step
        n_fft: FFT window size for spectrogram models
    """

    def __init__(
        self,
        feature_extractor: Any,
        batch_size: int = ProcessorDefaults.DEFAULT_BATCH_SIZE,
        sequence_length: int = ProcessorDefaults.DEFAULT_AUDIO_SEQUENCE_LENGTH,
        sampling_rate: int = ProcessorDefaults.DEFAULT_SAMPLING_RATE,
        **kwargs: Any,
    ):
        """
        Initialize ONNX audio processor.

        Args:
            feature_extractor: HuggingFace feature extractor instance
            batch_size: Fixed batch size for ONNX optimization
            sequence_length: Fixed sequence length for audio features
            sampling_rate: Target sampling rate
            **kwargs: Additional configuration options
        """
        # Create modality configuration
        modality_config = self._create_modality_config(
            feature_extractor, batch_size, sequence_length, sampling_rate, **kwargs
        )

        super().__init__(
            feature_extractor, modality_config, kwargs.get("validation_enabled", True)
        )

        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.feature_size = getattr(
            feature_extractor, "feature_size", ProcessorDefaults.DEFAULT_FEATURE_SIZE
        )
        self.n_fft = getattr(
            feature_extractor, "n_fft", ProcessorDefaults.DEFAULT_N_FFT
        )

        # Configure for fixed shapes
        self._configure_for_fixed_shapes()

    @staticmethod
    def _create_modality_config(
        feature_extractor: Any,
        batch_size: int,
        sequence_length: int,
        sampling_rate: int,
        **kwargs: Any,
    ) -> ModalityConfig:
        """Create modality configuration for audio processing."""
        # Determine tensor configuration based on feature extractor type
        feature_size = getattr(
            feature_extractor, "feature_size", ProcessorDefaults.DEFAULT_FEATURE_SIZE
        )

        if feature_size == ProcessorDefaults.DEFAULT_FEATURE_SIZE:
            # Raw waveform input (e.g., Wav2Vec2)
            tensors = [
                TensorSpec(
                    name="input_values",
                    shape=[batch_size, sequence_length],
                    dtype=TensorType.FLOAT32,
                    modality=ModalityType.AUDIO,
                    is_input=True,
                    description="Raw audio waveform values",
                )
            ]
        else:
            # Feature-based input (e.g., Whisper spectrograms)
            tensors = [
                TensorSpec(
                    name="input_features",
                    shape=[batch_size, feature_size, sequence_length],
                    dtype=TensorType.FLOAT32,
                    modality=ModalityType.AUDIO,
                    is_input=True,
                    description="Extracted audio features (e.g., mel spectrogram)",
                )
            ]

        # Add attention mask for some models
        if (
            hasattr(feature_extractor, "return_attention_mask")
            and feature_extractor.return_attention_mask
        ):
            tensors.append(
                TensorSpec(
                    name="attention_mask",
                    shape=[batch_size, sequence_length],
                    dtype=TensorType.INT64,
                    modality=ModalityType.AUDIO,
                    is_input=True,
                    description="Attention mask for audio features",
                )
            )

        config = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "sampling_rate": sampling_rate,
            "feature_size": feature_size,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
            **kwargs,
        }

        return ModalityConfig(
            modality_type=ModalityType.AUDIO,
            tensors=tensors,
            batch_size=batch_size,
            config=config,
            processor_class=feature_extractor.__class__.__name__,
        )

    def _configure_for_fixed_shapes(self) -> None:
        """Configure feature extractor for fixed-shape processing."""
        # Set sampling rate
        if hasattr(self.base_processor, "sampling_rate"):
            self.base_processor.sampling_rate = self.sampling_rate

        # Configure padding behavior
        if hasattr(self.base_processor, "padding_value"):
            self.base_processor.padding_value = getattr(
                self.base_processor,
                "padding_value",
                ProcessorDefaults.AUDIO_PADDING_VALUE,
            )

    def __call__(
        self,
        audio: NDArray[np.floating] | list[NDArray[np.floating]],
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """
        Process audio with fixed shapes for ONNX inference.

        Args:
            audio: Input audio array(s) or list of arrays
            sampling_rate: Sampling rate of input audio (will be resampled if needed)
            **kwargs: Additional processing arguments

        Returns:
            Dictionary with processed audio tensors ready for ONNX inference

        Examples:
            >>> processor = ONNXAudioProcessor(hf_extractor, batch_size=1, sequence_length=16000)
            >>> result = processor(audio_array, sampling_rate=16000)
            >>> result['input_values'].shape
            (1, 16000)
        """
        self._validate_inputs(audio)

        # Ensure audio is a list
        if not isinstance(audio, list):
            audio = [audio]

        # Handle batch size mismatch
        if len(audio) > self.batch_size:
            logger.warning(
                f"Input batch size {len(audio)} exceeds configured batch size {self.batch_size}. "
                f"Truncating to {self.batch_size} samples."
            )
            audio = audio[: self.batch_size]
        elif len(audio) < self.batch_size:
            # Pad with zero arrays of appropriate length
            if audio:
                sample_length = (
                    len(audio[0])
                    if hasattr(audio[0], "__len__")
                    else self.sequence_length
                )
                zero_audio = np.zeros(sample_length, dtype=np.float32)
                audio = audio + [zero_audio] * (self.batch_size - len(audio))
            else:
                raise ValueError("Cannot process empty audio list")

        # Process audio with fixed parameters
        processor_kwargs = {
            "sampling_rate": sampling_rate or self.sampling_rate,
            "padding": "max_length",
            "max_length": self.sequence_length,
            "truncation": True,
            "return_tensors": "np",
            **kwargs,
        }

        try:
            result = self.base_processor(audio, **processor_kwargs)
        except Exception as e:
            raise ONNXProcessorError(
                f"Audio processing failed: {e}", processor_type=self.__class__.__name__
            ) from e

        # Convert to tensor dictionary
        tensor_dict = self._convert_to_tensor_dict(result)

        # Ensure fixed batch size
        tensor_dict = self._ensure_fixed_batch_size(tensor_dict)

        # Validate outputs
        self._validate_outputs(tensor_dict)

        return tensor_dict

    def preprocess(
        self, audio: NDArray[np.floating] | list[NDArray[np.floating]], **kwargs: Any
    ) -> TensorDict:
        """Preprocess audio into ONNX tensor format."""
        return self.__call__(audio, **kwargs)

    def _convert_to_tensor_dict(self, processor_output: Any) -> TensorDict:
        """Convert audio processor output to tensor dictionary."""
        tensor_dict = {}

        if hasattr(processor_output, "data"):
            # BatchFeature or similar object
            for key, value in processor_output.data.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value, dtype=np.float32)
        elif isinstance(processor_output, dict):
            # Dictionary output
            for key, value in processor_output.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value, dtype=np.float32)
        else:
            raise ONNXProcessorError(
                f"Unexpected processor output type: {type(processor_output)}"
            )

        return tensor_dict
