"""
GraphML-specific test fixtures.

Provides ONNX models and related fixtures for GraphML testing.
"""

# Import ONNX fixtures from tests.fixtures
from ..fixtures.onnx_fixtures import (
    bert_tiny_metadata_path,
    bert_tiny_onnx_path,
    large_onnx_model,
    malformed_onnx_file,
    medium_onnx_model,
    simple_onnx_model,
)

# Re-export fixtures
__all__ = [
    "bert_tiny_metadata_path",
    "bert_tiny_onnx_path",
    "large_onnx_model",
    "malformed_onnx_file",
    "medium_onnx_model",
    "simple_onnx_model",
]
