"""
ONNX Image Processor Implementation

This module provides the ONNX-optimized image processor for computer vision models.
It wraps HuggingFace image processors (ViT, ResNet, DETR, etc.) and provides
fixed-shape image preprocessing optimized for ONNX inference.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

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


class ONNXImageProcessor(BaseONNXProcessor):
    """
    ONNX-optimized image processor for computer vision models.

    This class wraps HuggingFace image processors (ViT, ResNet, DETR, etc.) and
    provides fixed-shape image preprocessing optimized for ONNX inference.

    Key Features:
    - Fixed image dimensions for optimal ONNX performance
    - Automatic resizing, normalization, and format conversion
    - Support for all HuggingFace image processor features
    - Batch processing capabilities
    - Multiple image format support (PIL, numpy, tensors)

    Attributes:
        height: Fixed image height
        width: Fixed image width
        num_channels: Number of image channels (1, 3, or 4)
        image_mean: Normalization means for each channel
        image_std: Normalization standard deviations for each channel
    """

    def __init__(
        self,
        image_processor: Any,
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
        num_channels: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize ONNX image processor.

        Args:
            image_processor: HuggingFace image processor instance
            batch_size: Fixed batch size for ONNX optimization
            height: Fixed image height
            width: Fixed image width
            num_channels: Number of image channels
            **kwargs: Additional configuration options
        """
        # Create modality configuration
        modality_config = self._create_modality_config(
            image_processor, batch_size, height, width, num_channels, **kwargs
        )

        super().__init__(
            image_processor, modality_config, kwargs.get("validation_enabled", True)
        )

        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Extract normalization parameters
        self.image_mean = getattr(
            image_processor, "image_mean", ProcessorDefaults.IMAGENET_MEAN
        )
        self.image_std = getattr(
            image_processor, "image_std", ProcessorDefaults.IMAGENET_STD
        )

        # Ensure we have the right number of normalization values
        if len(self.image_mean) != num_channels:
            self.image_mean = self.image_mean[:num_channels] or [0.5] * num_channels
        if len(self.image_std) != num_channels:
            self.image_std = self.image_std[:num_channels] or [0.5] * num_channels

    @staticmethod
    def _create_modality_config(
        image_processor: Any,
        batch_size: int,
        height: int,
        width: int,
        num_channels: int,
        **kwargs: Any,
    ) -> ModalityConfig:
        """Create modality configuration for image processing."""
        tensors = [
            TensorSpec(
                name="pixel_values",
                shape=[batch_size, num_channels, height, width],
                dtype=TensorType.FLOAT32,
                modality=ModalityType.IMAGE,
                is_input=True,
                description="Preprocessed image pixel values in NCHW format",
            )
        ]

        config = {
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "num_channels": num_channels,
            "image_mean": getattr(
                image_processor, "image_mean", ProcessorDefaults.IMAGENET_MEAN
            ),
            "image_std": getattr(
                image_processor, "image_std", ProcessorDefaults.IMAGENET_STD
            ),
            "do_resize": getattr(image_processor, "do_resize", True),
            "do_normalize": getattr(image_processor, "do_normalize", True),
            "return_tensors": "np",
            **kwargs,
        }

        return ModalityConfig(
            modality_type=ModalityType.IMAGE,
            tensors=tensors,
            batch_size=batch_size,
            config=config,
            processor_class=image_processor.__class__.__name__,
        )

    def __call__(self, images: Any | list[Any], **kwargs: Any) -> ProcessorResult:
        """
        Process images with fixed shapes for ONNX inference.

        Args:
            images: Input image(s) - PIL Image, numpy array, or list of images
            **kwargs: Additional processing arguments

        Returns:
            Dictionary with processed image tensors ready for ONNX inference

        Examples:
            >>> processor = ONNXImageProcessor(hf_processor, batch_size=1, height=224, width=224)
            >>> result = processor(pil_image)
            >>> result['pixel_values'].shape
            (1, 3, 224, 224)
        """
        self._validate_inputs(images)

        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Handle batch size mismatch
        if len(images) > self.batch_size:
            logger.warning(
                f"Input batch size {len(images)} exceeds configured batch size {self.batch_size}. "
                f"Truncating to {self.batch_size} images."
            )
            images = images[: self.batch_size]
        elif len(images) < self.batch_size:
            # Pad with the last image or create blank images
            if images:
                last_image = images[-1]
                images = images + [last_image] * (self.batch_size - len(images))
            else:
                raise ValueError("Cannot process empty image list")

        # Process images with fixed parameters
        processor_kwargs = {
            "size": {"height": self.height, "width": self.width},
            "return_tensors": "np",
            **kwargs,
        }

        try:
            result = self.base_processor(images, **processor_kwargs)
        except Exception as e:
            raise ONNXProcessorError(
                f"Image processing failed: {e}", processor_type=self.__class__.__name__
            ) from e

        # Convert to tensor dictionary
        tensor_dict = self._convert_to_tensor_dict(result)

        # Ensure fixed batch size and shape
        tensor_dict = self._ensure_fixed_shapes(tensor_dict)

        # Validate outputs
        self._validate_outputs(tensor_dict)

        return tensor_dict

    def preprocess(self, images: Any | list[Any], **kwargs: Any) -> TensorDict:
        """Preprocess images into ONNX tensor format."""
        return self.__call__(images, **kwargs)

    def _convert_to_tensor_dict(self, processor_output: Any) -> TensorDict:
        """Convert image processor output to tensor dictionary."""
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

    def _ensure_fixed_shapes(self, tensor_dict: TensorDict) -> TensorDict:
        """Ensure all image tensors have the correct fixed shapes."""
        adjusted = {}

        for name, tensor in tensor_dict.items():
            if name == "pixel_values":
                # Ensure NCHW format with correct dimensions
                if tensor.ndim == 3:
                    # Add batch dimension
                    tensor = tensor[np.newaxis, ...]

                if tensor.shape != (
                    self.batch_size,
                    self.num_channels,
                    self.height,
                    self.width,
                ):
                    # Reshape to correct dimensions
                    target_shape = (
                        self.batch_size,
                        self.num_channels,
                        self.height,
                        self.width,
                    )
                    if tensor.size == np.prod(target_shape):
                        tensor = tensor.reshape(target_shape)
                    else:
                        logger.warning(
                            f"Image tensor shape {tensor.shape} doesn't match expected "
                            f"{target_shape}. Attempting to resize."
                        )
                        # This is a fallback - in practice, the image processor should handle this
                        tensor = tensor[
                            : self.batch_size,
                            : self.num_channels,
                            : self.height,
                            : self.width,
                        ]

                adjusted[name] = tensor
            else:
                # Handle other tensor types
                adjusted[name] = self._ensure_fixed_batch_size({name: tensor})[name]

        return adjusted
