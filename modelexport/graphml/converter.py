"""
Base ONNX to GraphML Converter

This module provides the base converter class for transforming ONNX models
into GraphML format. It focuses on the computational graph structure,
excluding parameter tensors by default.
"""

from pathlib import Path
from typing import Optional, Set

import onnx

from .onnx_parser import ONNXGraphParser
from .graphml_writer import GraphMLWriter


class ONNXToGraphMLConverter:
    """
    Base converter for ONNX to GraphML transformation.
    
    This converter extracts the computational graph structure from ONNX models
    and generates GraphML output suitable for visualization tools.
    
    Args:
        exclude_initializers: Whether to exclude weight/parameter tensors (default: True)
        exclude_attributes: Set of node attributes to exclude from output
    """
    
    def __init__(
        self,
        exclude_initializers: bool = True,
        exclude_attributes: Optional[Set[str]] = None
    ):
        self.exclude_initializers = exclude_initializers
        self.exclude_attributes = exclude_attributes or set()
        self.parser = ONNXGraphParser(
            exclude_initializers=exclude_initializers,
            exclude_attributes=exclude_attributes
        )
        self.writer = GraphMLWriter()
    
    def convert(self, onnx_model_path: str) -> str:
        """
        Convert ONNX model to GraphML string.
        
        Args:
            onnx_model_path: Path to ONNX model file
            
        Returns:
            GraphML XML as string
            
        Raises:
            FileNotFoundError: If ONNX model file doesn't exist
            onnx.onnx_cpp2py_export.checker.ValidationError: If ONNX model is invalid
        """
        # Validate input path
        model_path = Path(onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Load and parse ONNX model
        onnx_model = onnx.load(str(model_path))
        graph_data = self.parser.parse(onnx_model)
        
        # Add source file metadata
        graph_data.metadata["source_file"] = model_path.name
        
        # Generate GraphML
        graphml_element = self.writer.write(graph_data)
        return self.writer.to_string(graphml_element)
    
    def save(self, onnx_model_path: str, output_path: str) -> None:
        """
        Convert ONNX model and save to GraphML file.
        
        Args:
            onnx_model_path: Path to ONNX model file
            output_path: Path for output GraphML file
            
        Raises:
            FileNotFoundError: If ONNX model file doesn't exist
            onnx.onnx_cpp2py_export.checker.ValidationError: If ONNX model is invalid
        """
        # Convert to GraphML
        graphml_content = self.convert(onnx_model_path)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        output_file.write_text(graphml_content, encoding='utf-8')
    
    def get_statistics(self) -> dict:
        """
        Get statistics from the last conversion.
        
        Returns:
            Dictionary with conversion statistics
        """
        return {
            "nodes": self.parser.last_node_count,
            "edges": self.parser.last_edge_count,
            "excluded_initializers": self.parser.last_initializer_count
        }