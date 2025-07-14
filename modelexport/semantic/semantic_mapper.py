#!/usr/bin/env python3
"""
Semantic ONNX-to-HuggingFace Module Mapper

This module provides direct mapping from ONNX nodes to their originating 
HuggingFace modules using PyTorch's built-in scoping information.
"""

from typing import Any

import onnx
import torch.nn as nn
from transformers import PreTrainedModel


class ScopePathParser:
    """Parse ONNX node names to extract semantic scope information."""
    
    @staticmethod
    def parse_onnx_node_name(node_name: str) -> dict[str, Any] | None:
        """
        Parse ONNX node name into semantic components.
        
        Args:
            node_name: ONNX node name like "/bert/encoder/layer.0/attention/self/query/MatMul"
            
        Returns:
            Dict with parsed components or None if parsing fails
        """
        if not node_name or '/' not in node_name:
            return None
        
        parts = node_name.strip('/').split('/')
        if len(parts) < 2:
            return None
            
        return {
            'full_path': node_name,
            'module_path': '/'.join(parts[:-1]),
            'operation': parts[-1],
            'hierarchy_levels': parts[:-1],
            'depth': len(parts) - 1,
            'is_attention': 'attention' in node_name.lower(),
            'layer_id': ScopePathParser._extract_layer_id(parts[:-1])
        }
    
    @staticmethod
    def _extract_layer_id(path_parts: list[str]) -> int | None:
        """Extract transformer layer ID from scope path parts."""
        for part in path_parts:
            if 'layer.' in part:
                try:
                    return int(part.split('.')[-1])
                except ValueError:
                    pass
        return None
    
    @staticmethod
    def scope_path_to_hf_name(scope_path: str) -> str:
        """
        Convert scope path to HuggingFace module name.
        
        Args:
            scope_path: Scope path like "bert/encoder/layer.0/attention/self/query"
            
        Returns:
            HF module name like "encoder.layer.0.attention.self.query"
        """
        # Remove root model name (first component) and convert to dot notation
        parts = scope_path.split('/')
        if len(parts) > 1:
            return '.'.join(parts[1:])  # Skip root (e.g., 'bert')
        return scope_path


class HFModuleMapper:
    """Map scope paths to HuggingFace modules and provide module information."""
    
    def __init__(self, hf_model: PreTrainedModel):
        """
        Initialize mapper with HuggingFace model.
        
        Args:
            hf_model: HuggingFace PreTrainedModel instance
        """
        self.hf_model = hf_model
        self.name_to_module = dict(hf_model.named_modules())
        self.module_to_name = {module: name for name, module in self.name_to_module.items()}
    
    def get_module_by_scope_path(self, scope_path: str) -> nn.Module | None:
        """
        Get HuggingFace module by scope path.
        
        Args:
            scope_path: Scope path from ONNX node name
            
        Returns:
            HuggingFace module or None if not found
        """
        hf_name = ScopePathParser.scope_path_to_hf_name(scope_path)
        return self.name_to_module.get(hf_name)
    
    def get_module_name(self, module: nn.Module) -> str | None:
        """Get HuggingFace module name for a module instance."""
        return self.module_to_name.get(module)
    
    def get_module_info(self, module: nn.Module) -> dict[str, Any]:
        """
        Get detailed information about a HuggingFace module.
        
        Args:
            module: HuggingFace module instance
            
        Returns:
            Dict with module information
        """
        return {
            'class_name': module.__class__.__name__,
            'module_name': self.get_module_name(module),
            'parameters': dict(module.named_parameters()),
            'parameter_count': sum(p.numel() for p in module.parameters()),
            'submodules': list(module.named_children()),
            'module_type': self._classify_module_type(module),
            'is_leaf': len(list(module.children())) == 0
        }
    
    def _classify_module_type(self, module: nn.Module) -> str:
        """Classify HuggingFace module type for semantic understanding."""
        class_name = module.__class__.__name__
        
        if 'Attention' in class_name:
            return 'attention'
        elif 'Linear' in class_name:
            return 'linear_projection'
        elif 'LayerNorm' in class_name or 'Norm' in class_name:
            return 'normalization'
        elif 'Embedding' in class_name:
            return 'embedding'
        elif 'Dropout' in class_name:
            return 'regularization'
        elif 'Activation' in class_name or 'GELU' in class_name or 'ReLU' in class_name:
            return 'activation'
        elif 'Pooler' in class_name:
            return 'pooling'
        elif 'Encoder' in class_name:
            return 'encoder'
        elif 'Decoder' in class_name:
            return 'decoder'
        elif 'Layer' in class_name:
            return 'transformer_layer'
        else:
            return 'other'


class SemanticMapper:
    """Main semantic mapping interface for ONNX nodes to HuggingFace modules."""
    
    def __init__(self, hf_model: PreTrainedModel, onnx_model: onnx.ModelProto):
        """
        Initialize semantic mapper.
        
        Args:
            hf_model: HuggingFace model instance
            onnx_model: ONNX model proto
        """
        self.hf_model = hf_model
        self.onnx_model = onnx_model
        self.scope_parser = ScopePathParser()
        self.module_mapper = HFModuleMapper(hf_model)
        self._node_to_module_cache = {}
    
    def get_hf_module_for_onnx_node(self, onnx_node: onnx.NodeProto) -> nn.Module | None:
        """
        Get the HuggingFace module that produced this ONNX node.
        
        Args:
            onnx_node: ONNX node proto
            
        Returns:
            HuggingFace module or None if mapping not found
        """
        if onnx_node.name in self._node_to_module_cache:
            return self._node_to_module_cache[onnx_node.name]
        
        scope_info = self.scope_parser.parse_onnx_node_name(onnx_node.name)
        if not scope_info:
            return None
        
        hf_module = self.module_mapper.get_module_by_scope_path(scope_info['module_path'])
        self._node_to_module_cache[onnx_node.name] = hf_module
        
        return hf_module
    
    def get_semantic_info_for_node(self, onnx_node: onnx.NodeProto) -> dict[str, Any]:
        """
        Get complete semantic information for an ONNX node.
        
        Args:
            onnx_node: ONNX node proto
            
        Returns:
            Dict with complete semantic mapping information
        """
        scope_info = self.scope_parser.parse_onnx_node_name(onnx_node.name)
        hf_module = self.get_hf_module_for_onnx_node(onnx_node)
        
        result = {
            'onnx_node_name': onnx_node.name,
            'onnx_op_type': onnx_node.op_type,
            'scope_info': scope_info,
            'hf_module': hf_module,
            'module_info': None
        }
        
        if hf_module:
            result['module_info'] = self.module_mapper.get_module_info(hf_module)
        
        return result
    
    def build_complete_mapping(self) -> dict[str, dict[str, Any]]:
        """Build complete mapping from all ONNX nodes to HF modules."""
        mapping = {}
        
        for node in self.onnx_model.graph.node:
            semantic_info = self.get_semantic_info_for_node(node)
            mapping[node.name] = semantic_info
        
        return mapping
    
    def get_mapping_statistics(self) -> dict[str, Any]:
        """Get statistics about the semantic mapping coverage."""
        total_nodes = len(self.onnx_model.graph.node)
        mapped_nodes = 0
        module_types = {}
        layer_distribution = {}
        
        for node in self.onnx_model.graph.node:
            semantic_info = self.get_semantic_info_for_node(node)
            
            if semantic_info['hf_module'] is not None:
                mapped_nodes += 1
                
                # Count module types
                if semantic_info['module_info']:
                    module_type = semantic_info['module_info']['module_type']
                    module_types[module_type] = module_types.get(module_type, 0) + 1
                
                # Count layer distribution
                if semantic_info['scope_info'] and semantic_info['scope_info']['layer_id'] is not None:
                    layer_id = semantic_info['scope_info']['layer_id']
                    layer_distribution[layer_id] = layer_distribution.get(layer_id, 0) + 1
        
        return {
            'total_onnx_nodes': total_nodes,
            'mapped_nodes': mapped_nodes,
            'mapping_coverage': mapped_nodes / total_nodes if total_nodes > 0 else 0,
            'module_type_distribution': module_types,
            'layer_distribution': layer_distribution
        }


class SemanticQueryInterface:
    """Advanced query interface for semantic mappings."""
    
    def __init__(self, semantic_mapper: SemanticMapper):
        """
        Initialize query interface.
        
        Args:
            semantic_mapper: SemanticMapper instance
        """
        self.mapper = semantic_mapper
    
    def find_nodes_by_module_type(self, module_type: str) -> list[tuple[onnx.NodeProto, nn.Module]]:
        """Find all ONNX nodes from modules of specific type."""
        matching_nodes = []
        
        for node in self.mapper.onnx_model.graph.node:
            hf_module = self.mapper.get_hf_module_for_onnx_node(node)
            if hf_module:
                module_info = self.mapper.module_mapper.get_module_info(hf_module)
                if module_info['module_type'] == module_type:
                    matching_nodes.append((node, hf_module))
        
        return matching_nodes
    
    def find_nodes_by_layer(self, layer_id: int) -> list[tuple[onnx.NodeProto, nn.Module | None]]:
        """Find all ONNX nodes from specific transformer layer."""
        matching_nodes = []
        
        for node in self.mapper.onnx_model.graph.node:
            scope_info = self.mapper.scope_parser.parse_onnx_node_name(node.name)
            if scope_info and scope_info['layer_id'] == layer_id:
                hf_module = self.mapper.get_hf_module_for_onnx_node(node)
                matching_nodes.append((node, hf_module))
        
        return matching_nodes
    
    def get_attention_components(self, layer_id: int | None = None) -> dict[str, dict[str, Any]]:
        """Get all attention-related ONNX nodes with their HF modules."""
        attention_nodes = {}
        
        for node in self.mapper.onnx_model.graph.node:
            scope_info = self.mapper.scope_parser.parse_onnx_node_name(node.name)
            if scope_info and scope_info['is_attention']:
                if layer_id is None or scope_info['layer_id'] == layer_id:
                    semantic_info = self.mapper.get_semantic_info_for_node(node)
                    attention_nodes[node.name] = semantic_info
        
        return attention_nodes
    
    def find_nodes_by_hf_module_name(self, module_name: str) -> list[onnx.NodeProto]:
        """Find all ONNX nodes that originate from a specific HF module name."""
        target_module = self.mapper.module_mapper.name_to_module.get(module_name)
        if not target_module:
            return []
        
        matching_nodes = []
        for node in self.mapper.onnx_model.graph.node:
            hf_module = self.mapper.get_hf_module_for_onnx_node(node)
            if hf_module is target_module:
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_module_hierarchy_for_node(self, onnx_node: onnx.NodeProto) -> list[str]:
        """Get the complete module hierarchy path for an ONNX node."""
        scope_info = self.mapper.scope_parser.parse_onnx_node_name(onnx_node.name)
        if not scope_info:
            return []
        
        return scope_info['hierarchy_levels']
    
    def find_similar_nodes(self, onnx_node: onnx.NodeProto) -> list[onnx.NodeProto]:
        """Find ONNX nodes that come from the same HF module."""
        hf_module = self.mapper.get_hf_module_for_onnx_node(onnx_node)
        if not hf_module:
            return []
        
        similar_nodes = []
        for node in self.mapper.onnx_model.graph.node:
            if node.name != onnx_node.name:
                node_module = self.mapper.get_hf_module_for_onnx_node(node)
                if node_module is hf_module:
                    similar_nodes.append(node)
        
        return similar_nodes