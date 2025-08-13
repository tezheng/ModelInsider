"""
Universal OnnxConfig generation for dynamic ONNX export.

This module provides automatic OnnxConfig generation for any HuggingFace model,
eliminating the need for model-specific configuration classes.
"""

from .input_generator import InputSpecGenerator
from .patterns import (
    ARCHITECTURE_TO_TASK,
    DEFAULT_SHAPES,
    DYNAMIC_AXES_PATTERNS,
    MODEL_TYPE_TO_INPUTS,
    SPECIAL_MODEL_FAMILIES,
    TASK_TO_OUTPUTS,
    get_model_family,
)
from .shape_inference import ShapeInferencer
from .task_detector import TaskDetector
from .universal_config import UniversalOnnxConfig

__all__ = [
    "UniversalOnnxConfig",
    "TaskDetector", 
    "InputSpecGenerator",
    "ShapeInferencer",
    # Pattern exports
    "ARCHITECTURE_TO_TASK",
    "MODEL_TYPE_TO_INPUTS",
    "TASK_TO_OUTPUTS",
    "DEFAULT_SHAPES",
    "DYNAMIC_AXES_PATTERNS",
    "SPECIAL_MODEL_FAMILIES",
    "get_model_family",
]