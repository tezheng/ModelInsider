"""
Metadata Reader for HTP metadata files.

This module reads HTP metadata JSON files containing module hierarchy and
node tagging information for GraphML conversion.
"""

import json
from pathlib import Path
from typing import Any, Dict


class MetadataReader:
    """
    Reader for HTP metadata JSON files.
    
    Reads and validates HTP metadata containing:
    - Module hierarchy information
    - Node tagging data
    - Export context and configuration
    """
    
    def __init__(self, metadata_path: str):
        """
        Initialize metadata reader with file path.
        
        Args:
            metadata_path: Path to HTP metadata JSON file
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If metadata file is invalid
        """
        self.metadata_path = Path(metadata_path)
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"HTP metadata file not found: {metadata_path}")
        
        # Load and validate metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load and validate metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                
            # Basic validation
            if not isinstance(data, dict):
                raise ValueError("Metadata must be a JSON object")
                
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metadata file: {e}")
    
    def get_modules(self) -> Dict[str, Any]:
        """Get module hierarchy data."""
        return self.metadata.get("modules", {})
    
    def get_tagged_nodes(self) -> Dict[str, Any]:
        """Get tagged nodes mapping."""
        return self.metadata.get("tagged_nodes", {})
    
    def get_export_context(self) -> Dict[str, Any]:
        """Get export context information."""
        return self.metadata.get("export_context", {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.metadata.get("model", {})