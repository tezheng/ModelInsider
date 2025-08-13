"""
ONNX Processors Module

This module exports all ONNX processor implementations for backward compatibility
and easy importing. It provides the structured processor architecture with
separate files for each processor type.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

# Import base processor and defaults
try:
    # When imported as part of package structure
    # Import specialized processors
    from .audio import ONNXAudioProcessor
    from .base import BaseONNXProcessor, ProcessorDefaults
    from .image import ONNXImageProcessor
    from .multimodal import ONNXProcessor
    from .text import ONNXTokenizer
    from .video import ONNXVideoProcessor
except ImportError:
    # When imported directly in tests
    # Import specialized processors
    from processors.audio import ONNXAudioProcessor
    from processors.base import BaseONNXProcessor, ProcessorDefaults
    from processors.image import ONNXImageProcessor
    from processors.multimodal import ONNXProcessor
    from processors.text import ONNXTokenizer
    from processors.video import ONNXVideoProcessor

# Export all processors for backward compatibility
__all__ = [
    # Base classes
    "BaseONNXProcessor",
    "ProcessorDefaults",
    # Specialized processors
    "ONNXTokenizer",  # Text processing
    "ONNXImageProcessor",  # Image processing
    "ONNXAudioProcessor",  # Audio processing
    "ONNXVideoProcessor",  # Video processing
    "ONNXProcessor",  # Multimodal processing
]
