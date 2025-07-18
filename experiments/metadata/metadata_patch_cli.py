"""
CLI commands for patching HTP metadata using JSON Patch.

This module provides practical patch operations that could be added
to the modelexport CLI for updating metadata without re-exporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .advanced_metadata import MetadataPatch, MetadataPointer


class MetadataPatchCLI:
    """CLI utilities for patching metadata files."""
    
    @staticmethod
    def update_coverage(
        metadata_path: Path,
        coverage_percentage: float,
        tagged_nodes: int,
        empty_tags: int = 0
    ) -> Path:
        """
        Update coverage statistics in metadata.
        
        Useful after re-tagging or fixing tagging issues.
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        patch = (
            MetadataPatch()
            .replace("/tagging/coverage/coverage_percentage", coverage_percentage)
            .replace("/tagging/coverage/tagged_nodes", tagged_nodes)
            .replace("/tagging/coverage/empty_tags", empty_tags)
            .add("/tagging/coverage/last_updated", {
                "timestamp": json.dumps(
                    __import__("datetime").datetime.now().isoformat()
                ).strip('"'),
                "reason": "coverage_update"
            })
        )
        
        updated = patch.apply(metadata)
        
        # Save with _updated suffix
        output_path = metadata_path.with_stem(metadata_path.stem + "_updated")
        with open(output_path, "w") as f:
            json.dump(updated, f, indent=2)
        
        return output_path
    
    @staticmethod
    def add_custom_analysis(
        metadata_path: Path,
        analysis_name: str,
        analysis_data: dict[str, Any]
    ) -> Path:
        """
        Add custom analysis results to metadata.
        
        Examples:
            add_custom_analysis(
                "model_metadata.json",
                "complexity_analysis",
                {
                    "total_complexity": 0.85,
                    "layer_complexity": {...},
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            )
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Create custom_analyses section if it doesn't exist
        if "custom_analyses" not in metadata:
            metadata["custom_analyses"] = {}
        
        patch = MetadataPatch().add(
            f"/custom_analyses/{analysis_name}",
            analysis_data
        )
        
        updated = patch.apply(metadata)
        
        output_path = metadata_path.with_stem(metadata_path.stem + "_analyzed")
        with open(output_path, "w") as f:
            json.dump(updated, f, indent=2)
        
        return output_path
    
    @staticmethod
    def mark_problematic_modules(
        metadata_path: Path,
        module_issues: dict[str, str]
    ) -> Path:
        """
        Mark modules with known issues.
        
        Args:
            metadata_path: Path to metadata file
            module_issues: Dict of module_name -> issue_description
            
        Example:
            mark_problematic_modules(
                "metadata.json",
                {
                    "encoder.layer.0.attention": "Low coverage: 45%",
                    "decoder.layer.1": "Missing tags"
                }
            )
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        patch = MetadataPatch()
        
        for module_name, issue in module_issues.items():
            if MetadataPointer.exists(metadata, f"/modules/{module_name}"):
                patch.add(
                    f"/modules/{module_name}/validation_issue",
                    issue
                )
        
        updated = patch.apply(metadata)
        
        output_path = metadata_path.with_stem(metadata_path.stem + "_validated")
        with open(output_path, "w") as f:
            json.dump(updated, f, indent=2)
        
        return output_path
    
    @staticmethod
    def create_patch_from_diff(
        original_path: Path,
        modified_path: Path
    ) -> list[dict[str, Any]]:
        """
        Create JSON Patch operations from two metadata files.
        
        Useful for understanding what changed between exports.
        """
        with open(original_path) as f:
            original = json.load(f)
        with open(modified_path) as f:
            modified = json.load(f)
        
        # Simple diff algorithm (in practice, use a library)
        patches = []
        
        def diff_objects(path: str, obj1: Any, obj2: Any) -> None:
            if type(obj1) != type(obj2):
                patches.append({
                    "op": "replace",
                    "path": path,
                    "value": obj2
                })
                return
            
            if isinstance(obj1, dict):
                all_keys = set(obj1.keys()) | set(obj2.keys())
                for key in all_keys:
                    # Escape special characters in JSON Pointer
                    escaped_key = key.replace("~", "~0").replace("/", "~1")
                    key_path = f"{path}/{escaped_key}"
                    
                    if key not in obj1:
                        patches.append({
                            "op": "add",
                            "path": key_path,
                            "value": obj2[key]
                        })
                    elif key not in obj2:
                        patches.append({
                            "op": "remove",
                            "path": key_path
                        })
                    else:
                        diff_objects(key_path, obj1[key], obj2[key])
            
            elif isinstance(obj1, list):
                # Simple list diff (not optimal)
                if obj1 != obj2:
                    patches.append({
                        "op": "replace",
                        "path": path,
                        "value": obj2
                    })
            
            else:
                # Primitive values
                if obj1 != obj2:
                    patches.append({
                        "op": "replace",
                        "path": path,
                        "value": obj2
                    })
        
        diff_objects("", original, modified)
        return patches
    
    @staticmethod
    def apply_patch_file(
        metadata_path: Path,
        patch_file: Path
    ) -> Path:
        """
        Apply a JSON Patch file to metadata.
        
        Patch file format:
        [
            {"op": "replace", "path": "/statistics/export_time", "value": 5.2},
            {"op": "add", "path": "/custom_field", "value": "custom_value"}
        ]
        """
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        with open(patch_file) as f:
            patch_ops = json.load(f)
        
        patch = MetadataPatch()
        patch.operations = patch_ops
        
        updated = patch.apply(metadata)
        
        output_path = metadata_path.with_stem(metadata_path.stem + "_patched")
        with open(output_path, "w") as f:
            json.dump(updated, f, indent=2)
        
        return output_path


# Example CLI integration
"""
# Update coverage after fixing tags
modelexport patch model_metadata.json --update-coverage 98.5 134 0

# Add custom analysis
modelexport patch model_metadata.json --add-analysis complexity analysis_results.json

# Mark problematic modules
modelexport patch model_metadata.json --mark-issues issues.json

# Create patch from diff
modelexport patch diff original.json modified.json -o changes.patch

# Apply patch file
modelexport patch apply model_metadata.json changes.patch
"""