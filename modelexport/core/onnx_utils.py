"""
ONNX Model Manipulation Utilities

This module provides utilities for working with ONNX models, including:
- Model loading and validation
- Node manipulation and metadata injection
- Hierarchy information management
- Model analysis and statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx


class ONNXUtils:
    """Utilities for ONNX model manipulation and analysis."""
    
    @staticmethod
    def load_and_validate(onnx_path: str) -> onnx.ModelProto:
        """
        Load and validate ONNX model from file.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            Loaded ONNX model
            
        Raises:
            FileNotFoundError: If file doesn't exist
            onnx.ValidationError: If model is invalid
        """
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return model
    
    @staticmethod
    def inject_hierarchy_metadata(
        onnx_model: onnx.ModelProto,
        node_tags: dict[str, dict[str, Any]],
        method: str = "unknown"
    ) -> int:
        """
        Inject hierarchy metadata into ONNX node doc_strings.
        
        Args:
            onnx_model: ONNX model to modify
            node_tags: Dictionary mapping node names to their tag information
            method: Tagging method name for metadata
            
        Returns:
            Number of nodes that received hierarchy metadata
        """
        injected_count = 0
        
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            
            if node_name in node_tags:
                tag_info = node_tags[node_name]
                
                # Create hierarchy info for doc_string
                hierarchy_info = {
                    "hierarchy_tags": tag_info.get("tags", []),
                    "hierarchy_path": tag_info.get("primary_path", ""),
                    "hierarchy_method": method,
                    "hierarchy_count": len(tag_info.get("tags", []))
                }
                
                # Additional metadata if available
                if "confidence" in tag_info:
                    hierarchy_info["confidence"] = tag_info["confidence"]
                if "source" in tag_info:
                    hierarchy_info["source"] = tag_info["source"]
                
                # Inject as JSON in doc_string
                node.doc_string = json.dumps(hierarchy_info)
                injected_count += 1
        
        return injected_count
    
    @staticmethod
    def extract_hierarchy_metadata(onnx_model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
        """
        Extract hierarchy metadata from ONNX node doc_strings.
        
        Args:
            onnx_model: ONNX model to analyze
            
        Returns:
            Dictionary mapping node names to their hierarchy information
        """
        node_hierarchy = {}
        
        for node in onnx_model.graph.node:
            node_name = node.name or f"{node.op_type}_{hash(str(node))}"
            
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        node_hierarchy[node_name] = {
                            "tags": hierarchy_info.get("hierarchy_tags", []),
                            "primary_path": hierarchy_info.get("hierarchy_path", ""),
                            "method": hierarchy_info.get("hierarchy_method", "unknown"),
                            "op_type": node.op_type
                        }
                        
                        # Extract additional metadata
                        if "confidence" in hierarchy_info:
                            node_hierarchy[node_name]["confidence"] = hierarchy_info["confidence"]
                        if "source" in hierarchy_info:
                            node_hierarchy[node_name]["source"] = hierarchy_info["source"]
                            
                except (json.JSONDecodeError, TypeError):
                    # Skip nodes with invalid JSON
                    continue
        
        return node_hierarchy
    
    @staticmethod
    def analyze_model_structure(onnx_model: onnx.ModelProto) -> dict[str, Any]:
        """
        Analyze ONNX model structure and provide statistics.
        
        Args:
            onnx_model: ONNX model to analyze
            
        Returns:
            Model structure analysis
        """
        # Count node types
        node_types = {}
        total_nodes = len(onnx_model.graph.node)
        
        for node in onnx_model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
        
        # Count inputs/outputs
        inputs = len(onnx_model.graph.input)
        outputs = len(onnx_model.graph.output)
        
        # Count initializers (parameters)
        initializers = len(onnx_model.graph.initializer)
        
        # Extract hierarchy statistics
        hierarchy_stats = ONNXUtils._analyze_hierarchy_coverage(onnx_model)
        
        return {
            "total_nodes": total_nodes,
            "node_types": node_types,
            "inputs": inputs,
            "outputs": outputs,
            "initializers": initializers,
            "hierarchy_coverage": hierarchy_stats,
            "opset_version": onnx_model.opset_import[0].version if onnx_model.opset_import else None
        }
    
    @staticmethod
    def _analyze_hierarchy_coverage(onnx_model: onnx.ModelProto) -> dict[str, Any]:
        """Analyze hierarchy coverage in the model."""
        total_nodes = len(onnx_model.graph.node)
        tagged_nodes = 0
        unique_tags = set()
        
        for node in onnx_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        tagged_nodes += 1
                        tags = hierarchy_info.get("hierarchy_tags", [])
                        unique_tags.update(tags)
                except (json.JSONDecodeError, TypeError):
                    continue
        
        coverage_ratio = tagged_nodes / total_nodes if total_nodes > 0 else 0.0
        
        return {
            "total_nodes": total_nodes,
            "tagged_nodes": tagged_nodes,
            "untagged_nodes": total_nodes - tagged_nodes,
            "coverage_ratio": coverage_ratio,
            "coverage_percentage": f"{coverage_ratio * 100:.1f}%",
            "unique_hierarchy_paths": len(unique_tags)
        }
    
    @staticmethod
    def create_sidecar_file(
        onnx_path: str,
        node_tags: dict[str, dict[str, Any]],
        metadata: dict[str, Any]
    ) -> str:
        """
        Create a sidecar JSON file with complete hierarchy information.
        
        Args:
            onnx_path: Path to ONNX model
            node_tags: Node tag mapping
            metadata: Additional export metadata
            
        Returns:
            Path to created sidecar file
        """
        sidecar_path = onnx_path.replace('.onnx', '_hierarchy.json')
        
        sidecar_data = {
            "version": "1.0",
            "model_path": onnx_path,
            "export_method": metadata.get("strategy", "unknown"),
            "node_tags": node_tags,
            "statistics": {
                "total_nodes": len(node_tags),
                "tagged_nodes": len([n for n in node_tags.values() if n.get('tags')]),
                "unique_tags": len(set(tag for node in node_tags.values() 
                                      for tag in node.get('tags', [])))
            },
            "metadata": metadata
        }
        
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        return sidecar_path
    
    @staticmethod
    def validate_hierarchy_consistency(onnx_path: str) -> dict[str, Any]:
        """
        Validate consistency between ONNX model hierarchy and sidecar file.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Validation report
        """
        try:
            # Load ONNX model hierarchy
            onnx_model = ONNXUtils.load_and_validate(onnx_path)
            onnx_hierarchy = ONNXUtils.extract_hierarchy_metadata(onnx_model)
            
            # Load sidecar hierarchy
            sidecar_path = onnx_path.replace('.onnx', '_hierarchy.json')
            if not Path(sidecar_path).exists():
                return {
                    "consistent": False,
                    "error": "Sidecar file not found"
                }
            
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            
            sidecar_hierarchy = sidecar_data.get("node_tags", {})
            
            # Compare hierarchies
            mismatches = []
            onnx_only = set(onnx_hierarchy.keys()) - set(sidecar_hierarchy.keys())
            sidecar_only = set(sidecar_hierarchy.keys()) - set(onnx_hierarchy.keys())
            
            for node_name in set(onnx_hierarchy.keys()) & set(sidecar_hierarchy.keys()):
                onnx_tags = set(onnx_hierarchy[node_name].get("tags", []))
                sidecar_tags = set(sidecar_hierarchy[node_name].get("tags", []))
                
                if onnx_tags != sidecar_tags:
                    mismatches.append({
                        "node": node_name,
                        "onnx_tags": list(onnx_tags),
                        "sidecar_tags": list(sidecar_tags)
                    })
            
            is_consistent = (len(mismatches) == 0 and 
                           len(onnx_only) == 0 and 
                           len(sidecar_only) == 0)
            
            return {
                "consistent": is_consistent,
                "total_onnx_nodes": len(onnx_hierarchy),
                "total_sidecar_nodes": len(sidecar_hierarchy),
                "tag_mismatches": mismatches,
                "onnx_only_nodes": list(onnx_only),
                "sidecar_only_nodes": list(sidecar_only)
            }
            
        except Exception as e:
            return {
                "consistent": False,
                "error": str(e)
            }
    
    @staticmethod
    def compare_models(onnx_path1: str, onnx_path2: str) -> dict[str, Any]:
        """
        Compare hierarchy information between two ONNX models.
        
        Args:
            onnx_path1: Path to first ONNX model
            onnx_path2: Path to second ONNX model
            
        Returns:
            Comparison report
        """
        try:
            model1 = ONNXUtils.load_and_validate(onnx_path1)
            model2 = ONNXUtils.load_and_validate(onnx_path2)
            
            hierarchy1 = ONNXUtils.extract_hierarchy_metadata(model1)
            hierarchy2 = ONNXUtils.extract_hierarchy_metadata(model2)
            
            # Extract unique tags from each model
            tags1 = set(tag for node in hierarchy1.values() for tag in node.get('tags', []))
            tags2 = set(tag for node in hierarchy2.values() for tag in node.get('tags', []))
            
            return {
                "model1_path": onnx_path1,
                "model2_path": onnx_path2,
                "model1_nodes": len(hierarchy1),
                "model2_nodes": len(hierarchy2),
                "model1_unique_tags": len(tags1),
                "model2_unique_tags": len(tags2),
                "common_tags": list(tags1 & tags2),
                "model1_only_tags": list(tags1 - tags2),
                "model2_only_tags": list(tags2 - tags1),
                "tag_overlap_ratio": len(tags1 & tags2) / max(len(tags1 | tags2), 1)
            }
            
        except Exception as e:
            return {
                "error": str(e)
            }