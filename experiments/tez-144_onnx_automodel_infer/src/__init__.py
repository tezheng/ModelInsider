"""
ONNX AutoModel Inference Utilities for Optimum.

This package provides AutoModel-like interface for ONNX models,
making it easy to load and use exported models from ModelExport.

Updated with ONNXAutoProcessor: Universal hierarchy-preserving ONNX model inference
with automatic processor detection and fixed-shape optimization for maximum performance.
"""

# Legacy AutoModel interface
from .auto_model_loader import AutoModelForONNX
from .inference_utils import (
    create_inference_pipeline,
    detect_model_task,
    get_ort_model_class,
    load_preprocessor,
)
from .onnx_auto_processor import (
    # Main factory class
    ONNXAutoProcessor,
)

# New ONNX Auto Processor interface
from .onnx_processor_types import (
    AudioProcessorProtocol,
    ConfigDict,
    ImageProcessorProtocol,
    ModalityConfig,
    # Enums
    ModalityType,
    ONNXConfigurationError,
    ONNXMetadataError,
    ONNXModelLoadError,
    # Exceptions
    ONNXProcessorError,
    ONNXProcessorNotFoundError,
    ONNXShapeError,
    ONNXUnsupportedModalityError,
    # Type aliases
    PathLike,
    ProcessorMetadata,
    # Protocols
    ProcessorProtocol,
    ProcessorResult,
    ShapeDict,
    TensorDict,
    # Data classes
    TensorSpec,
    TensorType,
    TokenizerProtocol,
    VideoProcessorProtocol,
    create_tensor_spec_from_dict,
    merge_processor_configs,
    # Utility functions
    validate_tensor_spec,
)
from .processors import (
    # Base class
    BaseONNXProcessor,
    ONNXAudioProcessor,
    ONNXImageProcessor,
    ONNXProcessor,
    # Specialized processors
    ONNXVideoProcessor,
)

__all__ = [
    # Legacy API
    "AutoModelForONNX",
    "detect_model_task",
    "get_ort_model_class",
    "load_preprocessor",
    "create_inference_pipeline",
    
    # Core ONNX Auto Processor classes
    "ONNXAutoProcessor",
    "BaseONNXProcessor",
    
    # Specialized processors
    "ONNXImageProcessor", 
    "ONNXAudioProcessor",
    "ONNXVideoProcessor",
    "ONNXProcessor",
    
    # Type definitions
    "ModalityType",
    "TensorType",
    "TensorSpec",
    "ModalityConfig",
    "ProcessorMetadata",
    
    # Type aliases
    "PathLike",
    "TensorDict",
    "ShapeDict", 
    "ConfigDict",
    "ProcessorResult",
    
    # Protocols
    "ProcessorProtocol",
    "TokenizerProtocol",
    "ImageProcessorProtocol",
    "AudioProcessorProtocol",
    "VideoProcessorProtocol",
    
    # Exceptions
    "ONNXProcessorError",
    "ONNXModelLoadError",
    "ONNXMetadataError",
    "ONNXShapeError",
    "ONNXUnsupportedModalityError",
    "ONNXProcessorNotFoundError",
    "ONNXConfigurationError",
    
    # Utilities
    "validate_tensor_spec",
    "create_tensor_spec_from_dict",
    "merge_processor_configs",
]

__version__ = "0.2.0"  # Updated version with ONNX Auto Processor

# Convenience function for common usage pattern
def create_processor_from_onnx(onnx_model_path: str, **kwargs) -> ONNXAutoProcessor:
    """
    Convenience function to create a processor from an ONNX model.
    
    Args:
        onnx_model_path: Path to ONNX model file
        **kwargs: Additional configuration options
        
    Returns:
        Configured ONNX auto processor
        
    Examples:
        >>> processor = create_processor_from_onnx("bert.onnx")
        >>> result = processor("Hello world!")
    """
    return ONNXAutoProcessor.from_model(onnx_model_path, **kwargs)

# Add convenience function to __all__
__all__.extend([
    "create_processor_from_onnx",
])