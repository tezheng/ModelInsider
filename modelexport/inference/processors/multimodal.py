"""
ONNX Multimodal Processor Implementation

This module provides the ONNX-optimized multimodal processor for models with multiple input types.
It wraps HuggingFace multimodal processors (CLIP, LayoutLM, Whisper, etc.) and provides
fixed-shape preprocessing optimized for ONNX inference across multiple modalities simultaneously.

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
    from .audio import ONNXAudioProcessor
    from .base import BaseONNXProcessor, ProcessorDefaults
    from .image import ONNXImageProcessor
    from .text import ONNXTokenizer
    from .video import ONNXVideoProcessor
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
    from processors.audio import ONNXAudioProcessor
    from processors.base import BaseONNXProcessor, ProcessorDefaults
    from processors.image import ONNXImageProcessor
    from processors.text import ONNXTokenizer
    from processors.video import ONNXVideoProcessor

# Configure logging
logger = logging.getLogger(__name__)


class ONNXProcessor(BaseONNXProcessor):
    """
    ONNX-optimized multimodal processor for models with multiple input types.

    This class wraps HuggingFace multimodal processors (CLIP, LayoutLM, Whisper, etc.)
    and provides fixed-shape preprocessing optimized for ONNX inference across
    multiple modalities simultaneously.

    Key Features:
    - Handles multiple modalities (text + image, audio + text, etc.)
    - Fixed shapes for all modality types
    - Automatic modality detection and routing
    - Unified interface for multimodal processing
    - Support for all HuggingFace multimodal processors

    Attributes:
        modalities: Dictionary mapping modality names to their configurations
        sub_processors: Dictionary of specialized processors for each modality
        is_multimodal: Always True for this class
    """

    def __init__(
        self, processor: Any, modality_configs: dict[str, ModalityConfig], **kwargs: Any
    ):
        """
        Initialize ONNX multimodal processor.

        Args:
            processor: HuggingFace multimodal processor instance
            modality_configs: Dictionary of modality configurations
            **kwargs: Additional configuration options
        """
        # Store modality configs first
        self.modalities = modality_configs

        # Combine all modality configs into a single metadata structure
        combined_config = self._create_combined_modality_config(
            modality_configs, processor
        )

        super().__init__(
            processor, combined_config, kwargs.get("validation_enabled", True)
        )

        self.sub_processors = {}
        self.is_multimodal = True

        # Create specialized sub-processors for each modality
        self._create_sub_processors(**kwargs)

    def _create_combined_modality_config(
        self, modality_configs: dict[str, ModalityConfig], processor: Any
    ) -> ModalityConfig:
        """Create a dummy multimodal configuration for base class compatibility."""
        # Create a dummy tensor for the multimodal config (just for validation)
        # The actual tensors will be handled by sub-processors
        dummy_tensor = TensorSpec(
            name="multimodal_dummy",
            shape=[
                ProcessorDefaults.DEFAULT_BATCH_SIZE,
                ProcessorDefaults.DEFAULT_BATCH_SIZE,
            ],
            dtype=TensorType.FLOAT32,
            modality=ModalityType.MULTIMODAL,
            is_input=True,
            description="Dummy tensor for multimodal processor",
        )

        max_batch_size = (
            max(config.batch_size for config in modality_configs.values())
            if modality_configs
            else ProcessorDefaults.DEFAULT_BATCH_SIZE
        )
        combined_config = {
            name: config.config for name, config in modality_configs.items()
        }

        return ModalityConfig(
            modality_type=ModalityType.MULTIMODAL,
            tensors=[dummy_tensor],
            batch_size=max_batch_size,
            config=combined_config,
            processor_class=processor.__class__.__name__,
        )

    def _create_sub_processors(self, **kwargs: Any) -> None:
        """Create specialized processors for each modality."""
        for modality_name, config in self.modalities.items():
            if config.modality_type == ModalityType.TEXT:
                # Extract tokenizer from multimodal processor
                tokenizer = getattr(self.base_processor, "tokenizer", None)
                if tokenizer:
                    self.sub_processors[modality_name] = ONNXTokenizer(
                        tokenizer=tokenizer,
                        batch_size=config.batch_size,
                        sequence_length=config.config.get(
                            "sequence_length", ProcessorDefaults.DEFAULT_SEQUENCE_LENGTH
                        ),
                        **kwargs,
                    )
            elif config.modality_type == ModalityType.IMAGE:
                # Extract image processor from multimodal processor
                image_processor = getattr(self.base_processor, "image_processor", None)
                if image_processor:
                    self.sub_processors[modality_name] = ONNXImageProcessor(
                        image_processor=image_processor,
                        batch_size=config.batch_size,
                        height=config.config.get(
                            "height", ProcessorDefaults.DEFAULT_IMAGE_HEIGHT
                        ),
                        width=config.config.get(
                            "width", ProcessorDefaults.DEFAULT_IMAGE_WIDTH
                        ),
                        num_channels=config.config.get(
                            "num_channels", ProcessorDefaults.DEFAULT_NUM_CHANNELS
                        ),
                        **kwargs,
                    )
            elif config.modality_type == ModalityType.AUDIO:
                # Extract feature extractor from multimodal processor
                feature_extractor = getattr(
                    self.base_processor, "feature_extractor", None
                )
                if feature_extractor:
                    self.sub_processors[modality_name] = ONNXAudioProcessor(
                        feature_extractor=feature_extractor,
                        batch_size=config.batch_size,
                        sequence_length=config.config.get(
                            "sequence_length",
                            ProcessorDefaults.DEFAULT_AUDIO_SEQUENCE_LENGTH,
                        ),
                        sampling_rate=config.config.get(
                            "sampling_rate", ProcessorDefaults.DEFAULT_SAMPLING_RATE
                        ),
                        **kwargs,
                    )
            elif config.modality_type == ModalityType.VIDEO:
                # Extract video processor from multimodal processor
                video_processor = getattr(
                    self.base_processor, "video_processor", None
                ) or getattr(self.base_processor, "image_processor", None)
                if video_processor:
                    self.sub_processors[modality_name] = ONNXVideoProcessor(
                        video_processor=video_processor,
                        batch_size=config.batch_size,
                        num_frames=config.config.get(
                            "num_frames", ProcessorDefaults.DEFAULT_NUM_FRAMES
                        ),
                        height=config.config.get(
                            "height", ProcessorDefaults.DEFAULT_IMAGE_HEIGHT
                        ),
                        width=config.config.get(
                            "width", ProcessorDefaults.DEFAULT_IMAGE_WIDTH
                        ),
                        num_channels=config.config.get(
                            "num_channels", ProcessorDefaults.DEFAULT_NUM_CHANNELS
                        ),
                        **kwargs,
                    )

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any | list[Any] | None = None,
        audio: NDArray[np.floating] | list[NDArray[np.floating]] | None = None,
        videos: Any | list[Any] | None = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """
        Process multimodal inputs with fixed shapes for ONNX inference.

        Args:
            text: Optional text input(s)
            images: Optional image input(s)
            audio: Optional audio input(s)
            videos: Optional video input(s)
            **kwargs: Additional processing arguments

        Returns:
            Dictionary with all processed tensors ready for ONNX inference

        Examples:
            >>> processor = ONNXProcessor(clip_processor, modality_configs)
            >>> result = processor(text="Hello world", images=pil_image)
            >>> result.keys()
            dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
        """
        combined_result = {}

        # Process each modality if input is provided
        if text is not None and "text" in self.sub_processors:
            text_result = self.sub_processors["text"](text, **kwargs)
            combined_result.update(text_result)

        if images is not None and "image" in self.sub_processors:
            image_result = self.sub_processors["image"](images, **kwargs)
            combined_result.update(image_result)

        if audio is not None and "audio" in self.sub_processors:
            audio_result = self.sub_processors["audio"](audio, **kwargs)
            combined_result.update(audio_result)

        if videos is not None and "video" in self.sub_processors:
            video_result = self.sub_processors["video"](videos, **kwargs)
            combined_result.update(video_result)

        # If no sub-processors, fall back to base processor
        if not combined_result and not self.sub_processors:
            try:
                # Use base processor directly
                processor_kwargs = {"return_tensors": "np", **kwargs}

                # Build arguments based on what's provided
                args = []
                if text is not None:
                    args.append(text)
                if images is not None:
                    args.append(images)
                if audio is not None:
                    args.append(audio)
                if videos is not None:
                    args.append(videos)

                if len(args) == 1:
                    result = self.base_processor(args[0], **processor_kwargs)
                else:
                    result = self.base_processor(*args, **processor_kwargs)

                combined_result = self._convert_to_tensor_dict(result)

            except Exception as e:
                raise ONNXProcessorError(
                    f"Multimodal processing failed: {e}",
                    processor_type=self.__class__.__name__,
                ) from e

        # Ensure consistent batch sizes across all tensors
        combined_result = self._ensure_consistent_batch_sizes(combined_result)

        # Validate outputs
        self._validate_outputs(combined_result)

        return combined_result

    def preprocess(self, inputs: dict[str, Any] | Any, **kwargs: Any) -> TensorDict:
        """
        Preprocess multimodal inputs into ONNX tensor format.

        Args:
            inputs: Dictionary of inputs by modality or single input
            **kwargs: Processing parameters

        Returns:
            Dictionary of tensor names to NumPy arrays
        """
        if isinstance(inputs, dict):
            return self.__call__(**inputs, **kwargs)
        else:
            # Try to auto-detect modality
            return self.__call__(inputs, **kwargs)

    def _convert_to_tensor_dict(self, processor_output: Any) -> TensorDict:
        """Convert multimodal processor output to tensor dictionary."""
        tensor_dict = {}

        if hasattr(processor_output, "data"):
            # BatchFeature or similar object
            for key, value in processor_output.data.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value)
        elif isinstance(processor_output, dict):
            # Dictionary output
            for key, value in processor_output.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value)
        else:
            raise ONNXProcessorError(
                f"Unexpected processor output type: {type(processor_output)}"
            )

        return tensor_dict

    def _ensure_consistent_batch_sizes(self, tensor_dict: TensorDict) -> TensorDict:
        """Ensure all tensors have consistent batch sizes."""
        if not tensor_dict:
            return tensor_dict

        # Find the target batch size (maximum across all tensors)
        target_batch_size = self.batch_size
        for tensor in tensor_dict.values():
            if len(tensor.shape) > 0:
                target_batch_size = max(target_batch_size, tensor.shape[0])

        # Adjust each tensor to have the target batch size
        adjusted = {}
        for name, tensor in tensor_dict.items():
            if len(tensor.shape) == 0:
                # Scalar tensor
                adjusted[name] = tensor
            elif tensor.shape[0] == target_batch_size:
                # Already correct batch size
                adjusted[name] = tensor
            elif tensor.shape[0] < target_batch_size:
                # Pad batch dimension
                pad_width = [(0, target_batch_size - tensor.shape[0])] + [(0, 0)] * (
                    len(tensor.shape) - 1
                )
                adjusted[name] = np.pad(
                    tensor, pad_width, mode="constant", constant_values=0
                )
            else:
                # Truncate batch dimension
                adjusted[name] = tensor[:target_batch_size]

        return adjusted
