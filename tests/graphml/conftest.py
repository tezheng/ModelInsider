"""
GraphML-specific test fixtures.

Provides ONNX models and related fixtures for GraphML testing.
"""

import pytest
from pathlib import Path
import sys

# Add fixtures directory to path
sys.path.append(str(Path(__file__).parent.parent / "fixtures"))

from onnx_fixtures import (
    simple_onnx_model,
    medium_onnx_model, 
    large_onnx_model,
    malformed_onnx_file,
    bert_tiny_onnx_path,
    bert_tiny_metadata_path
)

# Re-export fixtures
__all__ = [
    'simple_onnx_model',
    'medium_onnx_model',
    'large_onnx_model', 
    'malformed_onnx_file',
    'bert_tiny_onnx_path',
    'bert_tiny_metadata_path'
]