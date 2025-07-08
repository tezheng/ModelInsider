#!/usr/bin/env python3
"""
Semantic ONNX Exporter

Provides integrated ONNX export with semantic mapping capabilities.
"""

import torch
import torch.nn as nn
import onnx
from typing import Tuple, Union, Dict, Any, Optional
from pathlib import Path
from transformers import PreTrainedModel

from .semantic_mapper import SemanticMapper, SemanticQueryInterface


class SemanticONNXExporter:
    """
    ONNX exporter that preserves and provides access to semantic mappings
    between ONNX nodes and HuggingFace modules.
    """
    
    def __init__(self, opset_version: int = 17, verbose: bool = False):
        """
        Initialize semantic ONNX exporter.
        
        Args:
            opset_version: ONNX opset version to use
            verbose: Enable verbose output during export
        """
        self.opset_version = opset_version
        self.verbose = verbose
    
    def export_with_semantics(
        self,
        hf_model: PreTrainedModel,
        sample_input: torch.Tensor,
        output_path: Union[str, Path],
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        **onnx_export_kwargs
    ) -> Tuple[onnx.ModelProto, SemanticMapper]:
        """
        Export HuggingFace model to ONNX with semantic mapping.
        
        Args:
            hf_model: HuggingFace model to export
            sample_input: Sample input tensor for tracing
            output_path: Path to save ONNX model
            input_names: Names for ONNX input tensors
            output_names: Names for ONNX output tensors
            **onnx_export_kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Tuple of (ONNX model, SemanticMapper instance)
        """
        # Set default parameters
        export_params = {
            'input_names': input_names or ['input'],
            'output_names': output_names or ['output'],
            'opset_version': self.opset_version,
            'verbose': self.verbose,
            **onnx_export_kwargs
        }
        
        # Export to ONNX
        if self.verbose:
            print(f"🚀 Exporting HuggingFace model to ONNX: {output_path}")
        
        torch.onnx.export(
            hf_model,
            sample_input,
            output_path,
            **export_params
        )
        
        # Load exported ONNX model
        onnx_model = onnx.load(str(output_path))
        
        # Create semantic mapper
        semantic_mapper = SemanticMapper(hf_model, onnx_model)
        
        if self.verbose:
            stats = semantic_mapper.get_mapping_statistics()
            print(f"✅ Export complete with semantic mapping")
            print(f"📊 Mapping coverage: {stats['mapping_coverage']:.1%}")
            print(f"📊 Total nodes: {stats['total_onnx_nodes']}")
            print(f"📊 Mapped nodes: {stats['mapped_nodes']}")
        
        return onnx_model, semantic_mapper
    
    def export_with_analysis(
        self,
        hf_model: PreTrainedModel,
        sample_input: torch.Tensor,
        output_path: Union[str, Path],
        analysis_output_path: Optional[Union[str, Path]] = None,
        **export_kwargs
    ) -> Tuple[onnx.ModelProto, SemanticMapper, Dict[str, Any]]:
        """
        Export with complete semantic analysis and save analysis results.
        
        Args:
            hf_model: HuggingFace model to export
            sample_input: Sample input tensor for tracing
            output_path: Path to save ONNX model
            analysis_output_path: Path to save semantic analysis results
            **export_kwargs: Additional arguments for export_with_semantics
            
        Returns:
            Tuple of (ONNX model, SemanticMapper, analysis results)
        """
        # Export with semantics
        onnx_model, semantic_mapper = self.export_with_semantics(
            hf_model, sample_input, output_path, **export_kwargs
        )
        
        # Create query interface
        query = SemanticQueryInterface(semantic_mapper)
        
        # Perform comprehensive analysis
        analysis_results = self._perform_comprehensive_analysis(semantic_mapper, query)
        
        # Save analysis results if path provided
        if analysis_output_path:
            self._save_analysis_results(analysis_results, analysis_output_path)
        
        return onnx_model, semantic_mapper, analysis_results
    
    def _perform_comprehensive_analysis(
        self, 
        semantic_mapper: SemanticMapper,
        query: SemanticQueryInterface
    ) -> Dict[str, Any]:
        """Perform comprehensive semantic analysis."""
        
        # Basic statistics
        stats = semantic_mapper.get_mapping_statistics()
        
        # Attention analysis
        attention_nodes = query.get_attention_components()
        attention_by_layer = {}
        for node_name, info in attention_nodes.items():
            layer_id = info['scope_info']['layer_id'] if info['scope_info'] else None
            if layer_id is not None:
                if layer_id not in attention_by_layer:
                    attention_by_layer[layer_id] = []
                attention_by_layer[layer_id].append(node_name)
        
        # Module type analysis
        module_type_nodes = {}
        for module_type in stats['module_type_distribution'].keys():
            nodes = query.find_nodes_by_module_type(module_type)
            module_type_nodes[module_type] = [node.name for node, _ in nodes]
        
        # Complete mapping
        complete_mapping = semantic_mapper.build_complete_mapping()
        
        return {
            'export_timestamp': torch.utils.data._utils.collate.default_collate([]),  # Current time placeholder
            'model_info': {
                'model_class': semantic_mapper.hf_model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in semantic_mapper.hf_model.parameters()),
                'total_modules': len(list(semantic_mapper.hf_model.named_modules()))
            },
            'mapping_statistics': stats,
            'attention_analysis': {
                'total_attention_nodes': len(attention_nodes),
                'attention_by_layer': attention_by_layer,
                'attention_node_details': attention_nodes
            },
            'module_type_analysis': {
                'nodes_by_type': module_type_nodes,
                'type_distribution': stats['module_type_distribution']
            },
            'complete_node_mapping': complete_mapping
        }
    
    def _save_analysis_results(self, analysis_results: Dict[str, Any], output_path: Union[str, Path]):
        """Save analysis results to JSON file."""
        import json
        
        # Convert non-serializable objects to strings
        serializable_results = self._make_serializable(analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"📄 Semantic analysis saved to: {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.nn.Module, onnx.NodeProto)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


# Convenience function for simple usage
def export_hf_model_with_semantics(
    hf_model: PreTrainedModel,
    sample_input: torch.Tensor, 
    output_path: Union[str, Path],
    **kwargs
) -> Tuple[onnx.ModelProto, SemanticMapper]:
    """
    Convenience function to export HuggingFace model with semantic mapping.
    
    Args:
        hf_model: HuggingFace model to export
        sample_input: Sample input tensor for tracing
        output_path: Path to save ONNX model
        **kwargs: Additional arguments for SemanticONNXExporter
        
    Returns:
        Tuple of (ONNX model, SemanticMapper instance)
    """
    exporter = SemanticONNXExporter(**kwargs)
    return exporter.export_with_semantics(hf_model, sample_input, output_path)