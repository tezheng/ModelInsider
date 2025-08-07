"""
Universal OnnxConfig generation for dynamic ONNX export.

This module provides automatic OnnxConfig generation for any HuggingFace model,
eliminating the need for model-specific configuration classes.
"""

from .task_detector import TaskDetector
from .input_generator import InputSpecGenerator
from .shape_inference import ShapeInferencer
from .universal_config import UniversalOnnxConfig
from .patterns import (
    ARCHITECTURE_TO_TASK,
    MODEL_TYPE_TO_INPUTS,
    TASK_TO_OUTPUTS,
    DEFAULT_SHAPES,
    DYNAMIC_AXES_PATTERNS,
    SPECIAL_MODEL_FAMILIES,
    get_model_family
)

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