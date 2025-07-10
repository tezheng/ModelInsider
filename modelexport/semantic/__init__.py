"""
Semantic ONNX Export Module

This module provides semantic mapping between ONNX nodes and HuggingFace modules,
enabling users to trace any ONNX operation back to its originating HF module.
"""

from .semantic_mapper import (
    SemanticMapper,
    SemanticQueryInterface, 
    ScopePathParser,
    HFModuleMapper
)

__all__ = [
    'SemanticMapper',
    'SemanticQueryInterface',
    'ScopePathParser', 
    'HFModuleMapper',
]