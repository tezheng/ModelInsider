"""
Advanced metadata features using JSON Schema 2020-12 capabilities.

This module demonstrates how to leverage advanced JSON Schema features
for HTP metadata, including JSON Pointer, dynamic references, and patches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar, cast

# Type variable for generic metadata operations
T = TypeVar('T')


class MetadataPointer:
    """
    JSON Pointer utilities for HTP metadata navigation.
    
    Provides efficient access to nested metadata using RFC 6901 JSON Pointer syntax.
    """
    
    @staticmethod
    def get(metadata: dict[str, Any], pointer: str) -> Any:
        """
        Get value at JSON pointer location.
        
        Args:
            metadata: The metadata dictionary
            pointer: JSON Pointer string (e.g., "/modules/encoder.layer.0")
            
        Returns:
            Value at pointer location
            
        Raises:
            KeyError: If pointer path doesn't exist
        """
        if not pointer:
            return metadata
            
        if not pointer.startswith("/"):
            raise ValueError(f"JSON Pointer must start with '/': {pointer}")
        
        # Split pointer into segments
        segments = pointer[1:].split("/")
        current = metadata
        
        for segment in segments:
            # Handle array indices
            if isinstance(current, list):
                try:
                    index = int(segment)
                    current = current[index]
                except (ValueError, IndexError) as e:
                    raise KeyError(f"Invalid array index '{segment}' in pointer '{pointer}'") from e
            # Handle object keys
            elif isinstance(current, dict):
                # Unescape JSON Pointer tokens
                key = segment.replace("~1", "/").replace("~0", "~")
                if key not in current:
                    raise KeyError(f"Key '{key}' not found in pointer '{pointer}'")
                current = current[key]
            else:
                raise KeyError(f"Cannot navigate into {type(current).__name__} with segment '{segment}'")
        
        return current
    
    @staticmethod
    def set(metadata: dict[str, Any], pointer: str, value: Any) -> None:
        """Set value at JSON pointer location."""
        if not pointer or pointer == "/":
            raise ValueError("Cannot set root with JSON Pointer")
        
        # Navigate to parent
        segments = pointer[1:].split("/")
        parent_pointer = "/" + "/".join(segments[:-1]) if len(segments) > 1 else ""
        parent = MetadataPointer.get(metadata, parent_pointer) if parent_pointer else metadata
        
        # Set the value
        last_segment = segments[-1]
        if isinstance(parent, list):
            parent[int(last_segment)] = value
        else:
            key = last_segment.replace("~1", "/").replace("~0", "~")
            parent[key] = value
    
    @staticmethod
    def exists(metadata: dict[str, Any], pointer: str) -> bool:
        """Check if pointer path exists."""
        try:
            MetadataPointer.get(metadata, pointer)
            return True
        except KeyError:
            return False


class MetadataQuery:
    """
    Advanced query capabilities for HTP metadata.
    
    Supports JSON Pointer and pattern-based queries.
    """
    
    def __init__(self, metadata: dict[str, Any]):
        """Initialize with metadata."""
        self.metadata = metadata
    
    def pointer(self, path: str) -> Any:
        """Query using JSON Pointer."""
        return MetadataPointer.get(self.metadata, path)
    
    def find_modules_by_class(self, class_name: str) -> dict[str, dict[str, Any]]:
        """Find all modules matching a class name."""
        modules = self.metadata.get("modules", {})
        return {
            name: info
            for name, info in modules.items()
            if info.get("class_name") == class_name
        }
    
    def find_nodes_by_tag_pattern(self, pattern: str) -> dict[str, str]:
        """Find all nodes whose tags match a pattern."""
        tagged_nodes = self.metadata.get("tagging", {}).get("tagged_nodes", {})
        if "*" in pattern:
            # Simple wildcard matching
            prefix = pattern.split("*")[0]
            return {
                node: tag
                for node, tag in tagged_nodes.items()
                if tag.startswith(prefix)
            }
        return {
            node: tag
            for node, tag in tagged_nodes.items()
            if pattern in tag
        }
    
    def get_module_nodes(self, module_path: str) -> list[str]:
        """Get all ONNX nodes belonging to a specific module."""
        module_info = self.pointer(f"/modules/{module_path}")
        module_tag = module_info.get("traced_tag", "")
        
        tagged_nodes = self.metadata.get("tagging", {}).get("tagged_nodes", {})
        return [
            node
            for node, tag in tagged_nodes.items()
            if tag == module_tag
        ]
    
    def calculate_layer_statistics(self) -> dict[str, dict[str, Any]]:
        """Calculate statistics for each layer."""
        stats = {}
        modules = self.metadata.get("modules", {})
        tagged_nodes = self.metadata.get("tagging", {}).get("tagged_nodes", {})
        
        for name, info in modules.items():
            if "layer" in name:  # Focus on layer modules
                tag = info.get("traced_tag", "")
                # Count nodes for this layer and its children
                node_count = sum(
                    1 for node_tag in tagged_nodes.values()
                    if node_tag.startswith(tag)
                )
                stats[name] = {
                    "class": info.get("class_name", ""),
                    "node_count": node_count,
                    "tag": tag
                }
        
        return stats


class MetadataPatch:
    """
    JSON Patch operations for metadata updates.
    
    Implements a subset of RFC 6902 JSON Patch.
    """
    
    def __init__(self):
        """Initialize patch operations."""
        self.operations: list[dict[str, Any]] = []
    
    def add(self, path: str, value: Any) -> MetadataPatch:
        """Add operation."""
        self.operations.append({
            "op": "add",
            "path": path,
            "value": value
        })
        return self
    
    def remove(self, path: str) -> MetadataPatch:
        """Remove operation."""
        self.operations.append({
            "op": "remove",
            "path": path
        })
        return self
    
    def replace(self, path: str, value: Any) -> MetadataPatch:
        """Replace operation."""
        self.operations.append({
            "op": "replace",
            "path": path,
            "value": value
        })
        return self
    
    def test(self, path: str, value: Any) -> MetadataPatch:
        """Test operation (for validation)."""
        self.operations.append({
            "op": "test",
            "path": path,
            "value": value
        })
        return self
    
    def apply(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Apply patch operations to metadata.
        
        Returns:
            New metadata dict with patches applied (original unchanged)
        """
        import copy
        result = copy.deepcopy(metadata)
        
        for op in self.operations:
            operation = op["op"]
            path = op["path"]
            
            if operation == "add":
                MetadataPointer.set(result, path, op["value"])
            elif operation == "remove":
                # Navigate to parent and remove key
                segments = path[1:].split("/")
                parent_path = "/" + "/".join(segments[:-1]) if len(segments) > 1 else ""
                parent = MetadataPointer.get(result, parent_path) if parent_path else result
                
                last_segment = segments[-1]
                if isinstance(parent, list):
                    parent.pop(int(last_segment))
                else:
                    key = last_segment.replace("~1", "/").replace("~0", "~")
                    del parent[key]
            elif operation == "replace":
                MetadataPointer.set(result, path, op["value"])
            elif operation == "test":
                current = MetadataPointer.get(result, path)
                if current != op["value"]:
                    raise ValueError(f"Test failed at {path}: expected {op['value']}, got {current}")
        
        return result


class ExtensibleMetadataSchema:
    """
    Demonstrates extensible schema patterns using dynamic references.
    
    This would be used with Pydantic models that support JSON Schema 2020-12.
    """
    
    @staticmethod
    def create_base_schema() -> dict[str, Any]:
        """Create base schema with dynamic anchor points."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://modelexport/schemas/htp-base",
            "$dynamicAnchor": "module-extension",
            
            "type": "object",
            "properties": {
                "export_context": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string"},
                        "strategy": {"type": "string"}
                    }
                },
                "modules": {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#/$defs/baseModule"
                    }
                }
            },
            
            "$defs": {
                "baseModule": {
                    "$dynamicAnchor": "module-extension",
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "class_name": {"type": "string"},
                        "module_type": {"type": "string"},
                        "traced_tag": {"type": "string"},
                        "execution_order": {"type": "integer"}
                    },
                    "required": ["name", "class_name"]
                }
            }
        }
    
    @staticmethod
    def create_transformer_extension() -> dict[str, Any]:
        """Create transformer-specific schema extension."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://modelexport/schemas/htp-transformer",
            "$ref": "https://modelexport/schemas/htp-base",
            
            "$defs": {
                "transformerModule": {
                    "$dynamicAnchor": "module-extension",
                    "allOf": [
                        {"$ref": "https://modelexport/schemas/htp-base#/$defs/baseModule"},
                        {
                            "if": {
                                "properties": {
                                    "class_name": {"pattern": ".*Attention.*"}
                                }
                            },
                            "then": {
                                "properties": {
                                    "attention_heads": {"type": "integer"},
                                    "hidden_size": {"type": "integer"}
                                }
                            }
                        }
                    ]
                }
            }
        }


# Example usage functions
def demonstrate_pointer_usage(metadata_path: Path) -> None:
    """Demonstrate JSON Pointer usage."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Access nested module information
    encoder_0 = MetadataPointer.get(metadata, "/modules/encoder.layer.0")
    print(f"Encoder Layer 0: {encoder_0.get('class_name')}")
    
    # Update coverage statistics
    MetadataPointer.set(
        metadata,
        "/tagging/coverage/coverage_percentage",
        99.5
    )
    
    # Check if specific path exists
    has_attention = MetadataPointer.exists(
        metadata,
        "/modules/encoder.layer.0.attention"
    )
    print(f"Has attention module: {has_attention}")


def demonstrate_query_usage(metadata_path: Path) -> None:
    """Demonstrate query capabilities."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    query = MetadataQuery(metadata)
    
    # Find all BERT layers
    bert_layers = query.find_modules_by_class("BertLayer")
    print(f"Found {len(bert_layers)} BERT layers")
    
    # Find nodes for a specific module
    attention_nodes = query.get_module_nodes("encoder.layer.0.attention.self")
    print(f"Attention module has {len(attention_nodes)} nodes")
    
    # Calculate layer statistics
    stats = query.calculate_layer_statistics()
    for layer, info in stats.items():
        print(f"{layer}: {info['node_count']} nodes")


def demonstrate_patch_usage(metadata_path: Path) -> None:
    """Demonstrate patch operations."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Create a patch to update metadata
    patch = (
        MetadataPatch()
        .replace("/statistics/export_time", 5.2)
        .add("/custom_analysis", {
            "timestamp": "2024-01-01T12:00:00Z",
            "tool_version": "1.0"
        })
        .test("/export_context/version", "1.0")
    )
    
    # Apply patch
    updated = patch.apply(metadata)
    print(f"Export time updated to: {updated['statistics']['export_time']}")
    print(f"Added custom analysis: {updated.get('custom_analysis')}")