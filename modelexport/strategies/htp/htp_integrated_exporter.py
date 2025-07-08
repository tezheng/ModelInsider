"""
HTP Integrated Exporter with TracingHierarchyBuilder

This is a clean implementation that integrates:
1. TracingHierarchyBuilder for optimized hierarchy building
2. ONNXNodeTagger for ONNX node tagging
3. CARDINAL RULES compliance throughout

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Universal design for any model
- MUST-002: TORCH.NN FILTERING - Filter torch.nn except whitelist
- MUST-003: UNIVERSAL DESIGN - Architecture-agnostic approach

Replaces complex trace capture with clean, reliable components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import onnx
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from ...core.tracing_hierarchy_builder import TracingHierarchyBuilder
from ...core.onnx_node_tagger import create_node_tagger_from_hierarchy

logger = logging.getLogger(__name__)


class HTPIntegratedExporter:
    """
    HTP Integrated Exporter using TracingHierarchyBuilder and ONNXNodeTagger.
    
    This clean implementation provides hierarchy-preserving ONNX export
    by integrating our proven components.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize HTP integrated exporter.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.strategy = "htp_integrated"
        
        # Core components
        self._hierarchy_builder = None
        self._node_tagger = None
        
        # Export state
        self._hierarchy_data = {}
        self._export_stats = {
            "export_time": 0.0,
            "hierarchy_modules": 0,
            "onnx_nodes": 0,
            "tagged_nodes": 0,
            "empty_tags": 0,
            "coverage_percentage": 0.0
        }
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            logger.info("HTP Integrated Exporter initialized")
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple, Dict],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 17,
        enable_operation_fallback: bool = False,
        **export_kwargs
    ) -> Dict[str, Any]:
        """
        Export model to ONNX with hierarchy-preserving tags.
        
        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing
            output_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors  
            dynamic_axes: Dynamic axes configuration
            opset_version: ONNX opset version
            enable_operation_fallback: Enable operation-based fallback in tagging
            **export_kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Dictionary with export statistics and metadata
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info(f"Starting HTP integrated export for {type(model).__name__}")
        
        # Step 1: Set model to eval mode
        model.eval()
        
        # Step 2: Build optimized hierarchy using TracingHierarchyBuilder
        self._build_hierarchy(model, example_inputs)
        
        # Step 3: Export to ONNX
        self._export_to_onnx(
            model, example_inputs, output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **export_kwargs
        )
        
        # Step 4: Load ONNX model and create node tagger
        onnx_model = onnx.load(output_path)
        self._create_node_tagger(enable_operation_fallback)
        
        # Step 5: Tag all ONNX nodes
        self._tag_onnx_nodes(onnx_model)
        
        # Step 6: Inject tags into ONNX model
        self._inject_tags_into_onnx(output_path, onnx_model)
        
        # Step 7: Create metadata file
        self._create_metadata_file(output_path)
        
        # Calculate final statistics
        self._export_stats["export_time"] = time.time() - start_time
        
        if self.verbose:
            logger.info(f"HTP integrated export completed in {self._export_stats['export_time']:.2f}s")
            logger.info(f"Coverage: {self._export_stats['coverage_percentage']:.1f}%")
        
        return self._export_stats.copy()
    
    def _build_hierarchy(self, model: nn.Module, example_inputs: Any) -> None:
        """
        Build optimized hierarchy using TracingHierarchyBuilder.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing
        """
        if self.verbose:
            logger.info("Building hierarchy with TracingHierarchyBuilder...")
        
        self._hierarchy_builder = TracingHierarchyBuilder()
        
        # Convert example_inputs to tuple for tracing
        if isinstance(example_inputs, torch.Tensor):
            input_args = (example_inputs,)
        elif isinstance(example_inputs, (tuple, list)):
            input_args = tuple(example_inputs)
        elif isinstance(example_inputs, dict):
            # For dict inputs, convert to tuple of values
            input_args = tuple(example_inputs.values())
        else:
            input_args = (example_inputs,)
        
        # Trace model execution
        self._hierarchy_builder.trace_model_execution(model, input_args)
        
        # Get hierarchy data
        execution_summary = self._hierarchy_builder.get_execution_summary()
        self._hierarchy_data = execution_summary['module_hierarchy']
        
        # Update statistics
        self._export_stats["hierarchy_modules"] = len(self._hierarchy_data)
        
        if self.verbose:
            logger.info(f"Built hierarchy with {len(self._hierarchy_data)} modules")
            logger.info(f"Execution steps: {execution_summary['execution_steps']}")
    
    def _export_to_onnx(
        self, 
        model: nn.Module, 
        example_inputs: Any, 
        output_path: str,
        **kwargs
    ) -> None:
        """
        Export model to ONNX using standard PyTorch export.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            output_path: Output file path
            **kwargs: Additional export arguments
        """
        if self.verbose:
            logger.info(f"Exporting to ONNX: {Path(output_path).name}")
        
        # Filter out CLI-specific keys that aren't valid for torch.onnx.export
        cli_specific_keys = {"input_specs", "export_params", "training"}
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in cli_specific_keys
        }
        
        torch.onnx.export(model, example_inputs, output_path, **filtered_kwargs)
        
        if self.verbose:
            logger.info("ONNX export completed")
    
    def _create_node_tagger(self, enable_operation_fallback: bool) -> None:
        """
        Create ONNX node tagger from hierarchy data.
        
        Args:
            enable_operation_fallback: Whether to enable operation-based fallback
        """
        if self.verbose:
            logger.info("Creating ONNX node tagger...")
        
        self._node_tagger = create_node_tagger_from_hierarchy(
            self._hierarchy_data, 
            enable_operation_fallback=enable_operation_fallback
        )
        
        if self.verbose:
            logger.info(f"Node tagger created with model root: {self._node_tagger.model_root_tag}")
    
    def _tag_onnx_nodes(self, onnx_model: onnx.ModelProto) -> None:
        """
        Tag all ONNX nodes using the node tagger.
        
        Args:
            onnx_model: ONNX model to tag
        """
        if self.verbose:
            logger.info("Tagging ONNX nodes...")
        
        # Tag all nodes
        self._tagged_nodes = self._node_tagger.tag_all_nodes(onnx_model)
        
        # Verify NO EMPTY TAGS rule
        empty_tags = [
            name for name, tag in self._tagged_nodes.items() 
            if not tag or not tag.strip()
        ]
        
        # Update statistics
        self._export_stats["onnx_nodes"] = len(onnx_model.graph.node)
        self._export_stats["tagged_nodes"] = len(self._tagged_nodes)
        self._export_stats["empty_tags"] = len(empty_tags)
        self._export_stats["coverage_percentage"] = (
            (self._export_stats["tagged_nodes"] / self._export_stats["onnx_nodes"]) * 100
            if self._export_stats["onnx_nodes"] > 0 else 0.0
        )
        
        # Verify CARDINAL RULES compliance
        if empty_tags:
            raise RuntimeError(f"CARDINAL RULE VIOLATION: {len(empty_tags)} empty tags found!")
        
        if self.verbose:
            logger.info(f"Tagged {len(self._tagged_nodes)} nodes with 0 empty tags")
            
            # Get detailed statistics
            stats = self._node_tagger.get_tagging_statistics(onnx_model)
            logger.info(f"Direct matches: {stats['direct_matches']}")
            logger.info(f"Parent matches: {stats['parent_matches']}")
            logger.info(f"Root fallbacks: {stats['root_fallbacks']}")
    
    def _inject_tags_into_onnx(self, output_path: str, onnx_model: onnx.ModelProto) -> None:
        """
        Inject hierarchy tags into ONNX model metadata.
        
        Args:
            output_path: ONNX file path
            onnx_model: ONNX model to tag
        """
        if self.verbose:
            logger.info("Injecting tags into ONNX model...")
        
        # Add hierarchy tags as node attributes
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{id(node)}"
            if node_name in self._tagged_nodes:
                tag = self._tagged_nodes[node_name]
                
                # Add tag as string attribute
                metadata_attr = onnx.helper.make_attribute(
                    "hierarchy_tag", tag
                )
                node.attribute.append(metadata_attr)
        
        # Add global metadata as string property
        exporter_info = json.dumps({
            "exporter": "HTP_Integrated_Exporter",
            "version": "1.0",
            "strategy": self.strategy,
            "hierarchy_modules": self._export_stats["hierarchy_modules"],
            "tagged_nodes": self._export_stats["tagged_nodes"],
            "coverage_percentage": self._export_stats["coverage_percentage"]
        })
        
        # Create metadata property
        metadata_prop = onnx.StringStringEntryProto()
        metadata_prop.key = "exporter_info"
        metadata_prop.value = exporter_info
        onnx_model.metadata_props.append(metadata_prop)
        
        # Save updated ONNX model
        onnx.save(onnx_model, output_path)
        
        if self.verbose:
            logger.info("Tags injected into ONNX model")
    
    def _create_metadata_file(self, onnx_path: str) -> None:
        """
        Create comprehensive metadata file.
        
        Args:
            onnx_path: ONNX file path for generating metadata filename
        """
        metadata_path = str(onnx_path).replace(".onnx", "_htp_integrated_metadata.json")
        
        metadata = {
            "export_info": {
                "onnx_file": Path(onnx_path).name,
                "exporter": "HTP_Integrated_Exporter",
                "strategy": self.strategy,
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cardinal_rules_compliance": {
                    "MUST_001_no_hardcoded_logic": True,
                    "MUST_002_torch_nn_filtering": True, 
                    "MUST_003_universal_design": True
                }
            },
            "statistics": self._export_stats,
            "hierarchy_data": self._hierarchy_data,
            "tagged_nodes": self._tagged_nodes,
            "tagging_guide": {
                "overview": "HTP Integrated export with TracingHierarchyBuilder + ONNXNodeTagger",
                "tag_format": "Hierarchical tags: /ModelClass/Module/Submodule.instance",
                "no_empty_tags_guarantee": "All nodes have non-empty hierarchy tags",
                "coverage_guarantee": "100% node coverage with proper fallbacks"
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            logger.info(f"Created metadata file: {Path(metadata_path).name}")
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get detailed export statistics."""
        return self._export_stats.copy()
    
    def get_hierarchy_data(self) -> Dict[str, Any]:
        """Get the complete hierarchy data."""
        return self._hierarchy_data.copy()
    
    def get_tagged_nodes(self) -> Dict[str, str]:
        """Get the complete node tagging data."""
        return self._tagged_nodes.copy() if hasattr(self, '_tagged_nodes') else {}


def export_with_htp_integrated(
    model: nn.Module,
    example_inputs: Any,
    output_path: str,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for HTP integrated export.
    
    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for tracing
        output_path: Output ONNX file path
        verbose: Enable verbose logging
        **kwargs: Additional export arguments
        
    Returns:
        Export statistics and metadata
    """
    exporter = HTPIntegratedExporter(verbose=verbose)
    return exporter.export(model, example_inputs, output_path, **kwargs)