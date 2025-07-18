"""
CLI utilities leveraging advanced JSON Schema features for metadata analysis.

This module provides practical utilities that can be integrated into
the modelexport CLI commands for powerful metadata querying.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .advanced_metadata import MetadataPointer, MetadataQuery


class MetadataCLI:
    """CLI utilities for metadata analysis using JSON Pointer and queries."""
    
    @staticmethod
    def query_metadata(metadata_path: Path, query: str) -> Any:
        """
        Query metadata using JSON Pointer or simple patterns.
        
        Args:
            metadata_path: Path to metadata JSON file
            query: Query string (JSON Pointer or pattern)
            
        Examples:
            # JSON Pointer queries
            query_metadata("metadata.json", "/modules/encoder.layer.0")
            query_metadata("metadata.json", "/tagging/coverage/coverage_percentage")
            
            # Pattern queries
            query_metadata("metadata.json", "find:modules:BertLayer")
            query_metadata("metadata.json", "find:tags:*/attention/*")
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # JSON Pointer query
        if query.startswith("/"):
            try:
                result = MetadataPointer.get(metadata, query)
                return result
            except KeyError as e:
                return {"error": str(e)}
        
        # Pattern-based queries
        elif query.startswith("find:"):
            parts = query.split(":")
            if len(parts) < 3:
                return {"error": "Invalid find query format. Use find:type:pattern"}
            
            query_type = parts[1]
            pattern = ":".join(parts[2:])  # Handle patterns with colons
            
            mq = MetadataQuery(metadata)
            
            if query_type == "modules":
                # Find modules by class name
                return mq.find_modules_by_class(pattern)
            elif query_type == "tags":
                # Find nodes by tag pattern
                return mq.find_nodes_by_tag_pattern(pattern)
            elif query_type == "stats":
                # Get layer statistics
                return mq.calculate_layer_statistics()
            else:
                return {"error": f"Unknown query type: {query_type}"}
        
        else:
            return {"error": "Query must start with '/' (JSON Pointer) or 'find:' (pattern)"}
    
    @staticmethod
    def compare_metadata(metadata1_path: Path, metadata2_path: Path) -> dict[str, Any]:
        """
        Compare two metadata files and highlight differences.
        
        Useful for comparing exports of the same model with different settings.
        """
        with open(metadata1_path) as f:
            meta1 = json.load(f)
        with open(metadata2_path) as f:
            meta2 = json.load(f)
        
        differences = {
            "coverage_delta": None,
            "export_time_delta": None,
            "module_differences": [],
            "tagging_differences": [],
        }
        
        # Compare coverage
        try:
            cov1 = MetadataPointer.get(meta1, "/tagging/coverage/coverage_percentage")
            cov2 = MetadataPointer.get(meta2, "/tagging/coverage/coverage_percentage")
            differences["coverage_delta"] = cov2 - cov1
        except KeyError:
            pass
        
        # Compare export times
        try:
            time1 = MetadataPointer.get(meta1, "/statistics/export_time")
            time2 = MetadataPointer.get(meta2, "/statistics/export_time")
            differences["export_time_delta"] = time2 - time1
        except KeyError:
            pass
        
        # Compare module counts
        modules1 = set(meta1.get("modules", {}).keys())
        modules2 = set(meta2.get("modules", {}).keys())
        
        if modules1 != modules2:
            differences["module_differences"] = {
                "only_in_first": list(modules1 - modules2),
                "only_in_second": list(modules2 - modules1),
            }
        
        return differences
    
    @staticmethod
    def validate_metadata_consistency(metadata_path: Path) -> dict[str, Any]:
        """
        Validate internal consistency of metadata using JSON Pointer cross-references.
        
        Checks:
        - All traced modules have corresponding tags
        - Coverage percentage matches actual counts
        - No orphaned tags
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        issues = []
        
        # Check 1: Module-tag consistency
        modules = metadata.get("modules", {})
        tagged_nodes = metadata.get("tagging", {}).get("tagged_nodes", {})
        
        module_tags = {info.get("traced_tag") for info in modules.values() if info.get("traced_tag")}
        used_tags = set(tagged_nodes.values())
        
        # Find modules without any tagged nodes
        for module_name, module_info in modules.items():
            module_tag = module_info.get("traced_tag")
            if module_tag and module_tag not in used_tags:
                # Check if any child modules have tags
                has_child_tags = any(
                    tag.startswith(module_tag + "/") for tag in used_tags
                )
                if not has_child_tags:
                    issues.append({
                        "type": "missing_tags",
                        "module": module_name,
                        "tag": module_tag,
                        "severity": "warning"
                    })
        
        # Check 2: Coverage calculation
        try:
            total_nodes = MetadataPointer.get(metadata, "/tagging/coverage/total_onnx_nodes")
            tagged_count = MetadataPointer.get(metadata, "/tagging/coverage/tagged_nodes")
            reported_coverage = MetadataPointer.get(metadata, "/tagging/coverage/coverage_percentage")
            
            calculated_coverage = (tagged_count / total_nodes * 100) if total_nodes > 0 else 0
            
            if abs(calculated_coverage - reported_coverage) > 0.1:
                issues.append({
                    "type": "coverage_mismatch",
                    "reported": reported_coverage,
                    "calculated": calculated_coverage,
                    "severity": "error"
                })
        except KeyError:
            pass
        
        # Check 3: Orphaned tags
        for node, tag in tagged_nodes.items():
            # Check if tag corresponds to any module
            tag_exists = any(
                tag == module_info.get("traced_tag") or 
                tag.startswith(module_info.get("traced_tag", "") + "/")
                for module_info in modules.values()
            )
            
            if not tag_exists and tag:
                issues.append({
                    "type": "orphaned_tag",
                    "node": node,
                    "tag": tag,
                    "severity": "warning"
                })
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "summary": {
                "errors": sum(1 for i in issues if i["severity"] == "error"),
                "warnings": sum(1 for i in issues if i["severity"] == "warning"),
            }
        }


# CLI command integration examples
def example_cli_integration():
    """
    Example of how to integrate into modelexport CLI.
    
    These would be added as subcommands to the analyze command:
    
    modelexport analyze model.onnx --query "/modules/encoder.layer.0"
    modelexport analyze model.onnx --query "find:modules:BertLayer"
    modelexport analyze model.onnx --validate-consistency
    modelexport compare model1.onnx model2.onnx --metadata
    """
    pass