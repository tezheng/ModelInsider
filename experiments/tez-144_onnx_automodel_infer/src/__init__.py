"""
ONNX AutoModel Inference Utilities for Optimum.

This package provides AutoModel-like interface for ONNX models,
making it easy to load and use exported models from ModelExport.
"""

from .auto_model_loader import AutoModelForONNX
from .inference_utils import (
    detect_model_task,
    get_ort_model_class,
    load_preprocessor,
    create_inference_pipeline,
)

__all__ = [
    "AutoModelForONNX",
    "detect_model_task",
    "get_ort_model_class",
    "load_preprocessor",
    "create_inference_pipeline",
]

__version__ = "0.1.0"