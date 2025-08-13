"""
ONNX Video Processor Implementation

This module provides the ONNX-optimized video processor for video understanding models.
It wraps HuggingFace video processors (VideoMAE, TimeSformer, etc.) and provides
fixed-shape video preprocessing optimized for ONNX inference.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    # When imported as part of package structure
    from ..onnx_processor_types import (
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
    from onnx_processor_types import (
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


class ONNXVideoProcessor(BaseONNXProcessor):
    """
    ONNX-optimized video processor for video understanding models.

    This class wraps HuggingFace video processors (VideoMAE, TimeSformer, etc.) and
    provides fixed-shape video preprocessing optimized for ONNX inference.

    Key Features:
    - Fixed frame count and dimensions for optimal ONNX performance
    - Automatic frame sampling and resizing
    - Support for various video formats
    - Temporal and spatial preprocessing
    - Batch processing capabilities

    Attributes:
        num_frames: Fixed number of frames per video
        height: Fixed frame height
        width: Fixed frame width
        num_channels: Number of channels per frame
        frame_sampling_rate: Rate of frame sampling from video
    """

    def __init__(
        self,
        video_processor: Any,
        batch_size: int = ProcessorDefaults.DEFAULT_BATCH_SIZE,
        num_frames: int = ProcessorDefaults.DEFAULT_NUM_FRAMES,
        height: int = ProcessorDefaults.DEFAULT_IMAGE_HEIGHT,
        width: int = ProcessorDefaults.DEFAULT_IMAGE_WIDTH,
        num_channels: int = ProcessorDefaults.DEFAULT_NUM_CHANNELS,
        **kwargs: Any,
    ):
        """
        Initialize ONNX video processor.

        Args:
            video_processor: HuggingFace video processor instance
            batch_size: Fixed batch size for ONNX optimization
            num_frames: Fixed number of frames per video
            height: Fixed frame height
            width: Fixed frame width
            num_channels: Number of channels per frame
            **kwargs: Additional configuration options
        """
        # Create modality configuration
        modality_config = self._create_modality_config(
            video_processor,
            batch_size,
            num_frames,
            height,
            width,
            num_channels,
            **kwargs,
        )

        super().__init__(
            video_processor, modality_config, kwargs.get("validation_enabled", True)
        )

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.frame_sampling_rate = kwargs.get(
            "frame_sampling_rate", ProcessorDefaults.DEFAULT_FRAME_SAMPLING_RATE
        )

        # Extract normalization parameters if available
        self.image_mean = getattr(
            video_processor, "image_mean", ProcessorDefaults.IMAGENET_MEAN
        )
        self.image_std = getattr(
            video_processor, "image_std", ProcessorDefaults.IMAGENET_STD
        )

    @staticmethod
    def _create_modality_config(
        video_processor: Any,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        num_channels: int,
        **kwargs: Any,
    ) -> ModalityConfig:
        """Create modality configuration for video processing."""
        tensors = [
            TensorSpec(
                name="pixel_values",
                shape=[batch_size, num_channels, num_frames, height, width],
                dtype=TensorType.FLOAT32,
                modality=ModalityType.VIDEO,
                is_input=True,
                description="Preprocessed video frames in NCTHW format",
            )
        ]

        config = {
            "batch_size": batch_size,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_channels": num_channels,
            "frame_sampling_rate": kwargs.get(
                "frame_sampling_rate", ProcessorDefaults.DEFAULT_FRAME_SAMPLING_RATE
            ),
            "image_mean": getattr(
                video_processor, "image_mean", ProcessorDefaults.IMAGENET_MEAN
            ),
            "image_std": getattr(
                video_processor, "image_std", ProcessorDefaults.IMAGENET_STD
            ),
            "do_resize": getattr(video_processor, "do_resize", True),
            "do_normalize": getattr(video_processor, "do_normalize", True),
            "return_tensors": "np",
            **kwargs,
        }

        return ModalityConfig(
            modality_type=ModalityType.VIDEO,
            tensors=tensors,
            batch_size=batch_size,
            config=config,
            processor_class=video_processor.__class__.__name__,
        )

    def __call__(self, videos: Any | list[Any], **kwargs: Any) -> ProcessorResult:
        """
        Process videos with fixed shapes for ONNX inference.

        Args:
            videos: Input video(s) - list of frames, video arrays, or list of videos
            **kwargs: Additional processing arguments

        Returns:
            Dictionary with processed video tensors ready for ONNX inference

        Examples:
            >>> processor = ONNXVideoProcessor(hf_processor, batch_size=1, num_frames=16)
            >>> result = processor(video_frames)  # List of PIL Images
            >>> result['pixel_values'].shape
            (1, 3, 16, 224, 224)
        """
        self._validate_inputs(videos)

        # Ensure videos is a list
        if not isinstance(videos, list):
            videos = [videos]

        # Handle batch size mismatch
        if len(videos) > self.batch_size:
            logger.warning(
                f"Input batch size {len(videos)} exceeds configured batch size {self.batch_size}. "
                f"Truncating to {self.batch_size} videos."
            )
            videos = videos[: self.batch_size]
        elif len(videos) < self.batch_size:
            # Pad with the last video or create blank videos
            if videos:
                last_video = videos[-1]
                videos = videos + [last_video] * (self.batch_size - len(videos))
            else:
                raise ValueError("Cannot process empty video list")

        # Process videos with fixed parameters
        processor_kwargs = {
            "size": {"height": self.height, "width": self.width},
            "num_frames": self.num_frames,
            "return_tensors": "np",
            **kwargs,
        }

        try:
            result = self.base_processor(videos, **processor_kwargs)
        except Exception as e:
            raise ONNXProcessorError(
                f"Video processing failed: {e}", processor_type=self.__class__.__name__
            ) from e

        # Convert to tensor dictionary
        tensor_dict = self._convert_to_tensor_dict(result)

        # Ensure fixed batch size and shape
        tensor_dict = self._ensure_fixed_video_shapes(tensor_dict)

        # Validate outputs
        self._validate_outputs(tensor_dict)

        return tensor_dict

    def preprocess(self, videos: Any | list[Any], **kwargs: Any) -> TensorDict:
        """Preprocess videos into ONNX tensor format."""
        return self.__call__(videos, **kwargs)

    def _convert_to_tensor_dict(self, processor_output: Any) -> TensorDict:
        """Convert video processor output to tensor dictionary."""
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

    def _ensure_fixed_video_shapes(self, tensor_dict: TensorDict) -> TensorDict:
        """Ensure all video tensors have the correct fixed shapes."""
        adjusted = {}

        for name, tensor in tensor_dict.items():
            if name == "pixel_values":
                # Ensure NCTHW format with correct dimensions
                target_shape = (
                    self.batch_size,
                    self.num_channels,
                    self.num_frames,
                    self.height,
                    self.width,
                )

                if tensor.shape != target_shape:
                    logger.warning(
                        f"Video tensor shape {tensor.shape} doesn't match expected {target_shape}. "
                        f"Attempting to reshape."
                    )

                    if tensor.size == np.prod(target_shape):
                        tensor = tensor.reshape(target_shape)
                    else:
                        # Fallback: truncate or pad as needed
                        # This is a simplified approach - real implementation would be more sophisticated
                        if tensor.ndim == 5:
                            tensor = tensor[
                                : self.batch_size,
                                : self.num_channels,
                                : self.num_frames,
                                : self.height,
                                : self.width,
                            ]
                        else:
                            # Handle other dimensionalities
                            tensor = tensor.reshape(target_shape[: tensor.ndim])

                adjusted[name] = tensor
            else:
                # Handle other tensor types
                adjusted[name] = self._ensure_fixed_batch_size({name: tensor})[name]

        return adjusted
