"""
HTP Metadata Reader

This module reads HTP (Hierarchical Trace-and-Project) metadata files
and extracts hierarchy information for GraphML integration.
"""

import json
from pathlib import Path
from typing import Any


class MetadataReader:
    """
    Reader for HTP metadata JSON files.
    
    This class extracts hierarchy information from HTP export metadata
    to enable hierarchical GraphML generation.
    """
    
    def __init__(self, metadata_path: str):
        """
        Initialize metadata reader with HTP metadata file path.
        
        Args:
            metadata_path: Path to HTP metadata JSON file
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is invalid JSON
        """
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"HTP metadata file not found: {metadata_path}")
        
        # Load metadata
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)
        
        # Extract key sections
        self.hierarchy_data = self._extract_hierarchy_data()
        self.tagged_nodes = self._extract_tagged_nodes()
        self.module_info = self._extract_module_info()
    
    def _extract_hierarchy_data(self) -> dict[str, Any]:
        """Extract hierarchy data from metadata."""
        # Try different possible locations for hierarchy data
        if "hierarchy_data" in self.metadata:
            return self.metadata["hierarchy_data"]
        elif "export_data" in self.metadata and "hierarchy_data" in self.metadata["export_data"]:
            return self.metadata["export_data"]["hierarchy_data"]
        elif "hierarchy" in self.metadata:
            return self.metadata["hierarchy"]
        else:
            # Return empty dict if no hierarchy data found
            return {}
    
    def _extract_tagged_nodes(self) -> dict[str, dict[str, Any]]:
        """Extract node tagging information."""
        # Try different possible locations
        result = None
        if "tagged_nodes" in self.metadata:
            result = self.metadata["tagged_nodes"]
        elif "export_data" in self.metadata and "tagged_nodes" in self.metadata["export_data"]:
            result = self.metadata["export_data"]["tagged_nodes"]
        elif "node_tags" in self.metadata:
            result = self.metadata["node_tags"]
        elif "nodes" in self.metadata:
            result = self.metadata["nodes"]
        else:
            return {}
        
        # Ensure result is a dict to prevent AttributeError
        if not isinstance(result, dict):
            raise AttributeError(f"tagged_nodes must be a dictionary, got {type(result)}")
        
        return result
    
    def _extract_module_info(self) -> dict[str, dict[str, Any]]:
        """Extract module information."""
        # Try to find module list
        if "modules" in self.metadata:
            modules_data = self.metadata["modules"]
        elif "export_data" in self.metadata and "modules" in self.metadata["export_data"]:
            modules_data = self.metadata["export_data"]["modules"]
        else:
            # Build from hierarchy data if available
            modules = {}
            for module_path, info in self.hierarchy_data.items():
                modules[module_path] = {
                    "class_name": info.get("class_name", ""),
                    "module_type": info.get("module_type", ""),
                    "execution_order": info.get("execution_order", 0),
                    "traced_tag": info.get("traced_tag", module_path)
                }
            return modules
        
        # Handle nested module structure
        if isinstance(modules_data, dict) and "class_name" in modules_data:
            # This is a hierarchical nested structure
            modules = {}
            self._flatten_module_hierarchy(modules_data, "", modules)
            return modules
        else:
            # Already flat structure
            return modules_data
    
    def _flatten_module_hierarchy(self, module_dict: dict[str, Any], parent_path: str, result: dict[str, dict[str, Any]]) -> None:
        """Recursively flatten nested module hierarchy."""
        # Extract current module info
        traced_tag = module_dict.get("traced_tag", parent_path)
        result[traced_tag] = {
            "class_name": module_dict.get("class_name", ""),
            "module_type": module_dict.get("module_type", module_dict.get("class_name", "")),
            "execution_order": module_dict.get("execution_order", 0),
            "traced_tag": traced_tag,
            "scope": module_dict.get("scope", "")
        }
        
        # Process children
        if "children" in module_dict:
            for child_name, child_data in module_dict["children"].items():
                self._flatten_module_hierarchy(child_data, traced_tag, result)
    
    def get_node_hierarchy_tag(self, node_name: str) -> str | None:
        """
        Get hierarchy tag for a specific node.
        
        Args:
            node_name: ONNX node name
            
        Returns:
            Hierarchy tag or None if not found
        """
        node_info = self.tagged_nodes.get(node_name)
        
        # Handle different formats
        if isinstance(node_info, str):
            # Direct string mapping: {"node_name": "module_path"}
            return node_info
        elif isinstance(node_info, dict):
            # Dictionary format: {"node_name": {"hierarchy_tag": "module_path"}}
            return node_info.get("hierarchy_tag") or node_info.get("tag")
        else:
            return None
    
    def get_module_hierarchy(self) -> dict[str, list[str]]:
        """
        Get module hierarchy as parent-child relationships.
        
        Returns:
            Dict mapping module paths to their direct children
        """
        hierarchy = {}
        
        # Build parent-child relationships
        for module_path in self.module_info.keys():
            # Normalize path by removing leading slash
            normalized_path = module_path.lstrip("/")
            
            if not normalized_path:  # Skip root
                continue
            
            # Find parent module
            parts = normalized_path.split('/')
            if len(parts) > 1:
                parent_path = '/'.join(parts[:-1])
            else:
                parent_path = ""
            
            # Add to parent's children
            if parent_path not in hierarchy:
                hierarchy[parent_path] = []
            hierarchy[parent_path].append(normalized_path)
        
        return hierarchy
    
    def get_module_info(self, module_path: str) -> dict[str, Any]:
        """
        Get information about a specific module.
        
        Args:
            module_path: Module hierarchy path
            
        Returns:
            Module information dict
        """
        return self.module_info.get(module_path, {})
    
    def get_all_modules(self) -> list[str]:
        """
        Get list of all module paths in hierarchy order.
        
        Returns:
            List of module paths sorted by hierarchy depth
        """
        modules = list(self.module_info.keys())
        # Normalize paths by removing leading slashes
        normalized_modules = []
        for module in modules:
            normalized = module.lstrip("/")
            # Keep the original mapping for lookup
            if normalized not in self.module_info and module in self.module_info:
                self.module_info[normalized] = self.module_info[module]
            normalized_modules.append(normalized)
        
        # Sort by depth (number of slashes) then alphabetically
        normalized_modules.sort(key=lambda x: (x.count('/'), x))
        return normalized_modules