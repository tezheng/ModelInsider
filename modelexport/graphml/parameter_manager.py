"""
Parameter Manager for GraphML v1.1

Handles extraction, storage, and management of ONNX model parameters
to enable complete model reconstruction from GraphML format.

Supports multiple storage strategies:
- sidecar: Separate .onnxdata file 
- embedded: Base64 encoded in GraphML
- reference: Reference to original ONNX file

Linear Task: TEZ-124
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import onnx
from onnx import TensorProto


class ParameterManager:
    """Manages parameter storage and retrieval for GraphML v1.1 format."""
    
    SUPPORTED_STRATEGIES: ClassVar[list[str]] = ["sidecar", "embedded", "reference"]
    
    def __init__(self, strategy: str = "sidecar"):
        """Initialize parameter manager with storage strategy."""
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}. "
                           f"Must be one of {self.SUPPORTED_STRATEGIES}")
        
        self.strategy = strategy
        
    def extract_parameters(
        self, 
        onnx_model: onnx.ModelProto, 
        output_base: str
    ) -> dict[str, Any]:
        """
        Extract parameters from ONNX model according to storage strategy.
        
        Args:
            onnx_model: ONNX model proto
            output_base: Base path for output files
            
        Returns:
            Dictionary with parameter information and file paths
        """
        
        # Extract initializer tensors (model parameters)
        parameters = {}
        total_size = 0
        
        for initializer in onnx_model.graph.initializer:
            param_data = self._extract_tensor_data(initializer)
            parameters[initializer.name] = {
                "data": param_data["data"],
                "shape": param_data["shape"],
                "dtype": param_data["dtype"],
                "size_bytes": param_data["size_bytes"]
            }
            total_size += param_data["size_bytes"]
        
        # Store parameters according to strategy
        if self.strategy == "sidecar":
            return self._store_sidecar(parameters, output_base, total_size)
        elif self.strategy == "embedded":
            return self._store_embedded(parameters, total_size)
        elif self.strategy == "reference":
            return self._store_reference(output_base, total_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _extract_tensor_data(self, tensor: TensorProto) -> dict[str, Any]:
        """Extract data from ONNX tensor."""
        
        # Convert to numpy array
        np_array = onnx.numpy_helper.to_array(tensor)
        
        # Serialize data
        data_bytes = np_array.tobytes()
        
        return {
            "data": base64.b64encode(data_bytes).decode('utf-8'),
            "shape": list(np_array.shape),
            "dtype": str(np_array.dtype),
            "size_bytes": len(data_bytes)
        }
    
    def _store_sidecar(
        self, 
        parameters: dict[str, Any], 
        output_base: str, 
        total_size: int
    ) -> dict[str, Any]:
        """Store parameters in separate .onnxdata file."""
        
        # Create parameter file path
        param_file = f"{output_base}.onnxdata"
        
        # Prepare parameter data
        param_data = {
            "format_version": "1.1",
            "total_size_bytes": total_size,
            "parameter_count": len(parameters),
            "parameters": parameters
        }
        
        # Save to file with compression
        import gzip
        with gzip.open(param_file, 'wt', encoding='utf-8') as f:
            json.dump(param_data, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_file_checksum(param_file)
        
        return {
            "parameter_strategy": self.strategy,
            "parameter_file": Path(param_file).name,
            "checksum": checksum,
            "total_size_bytes": total_size,
            "parameter_count": len(parameters),
            "files": {
                "parameters": param_file
            }
        }
    
    def _store_embedded(self, parameters: dict[str, Any], total_size: int) -> dict[str, Any]:
        """Store parameters embedded in GraphML metadata."""
        
        # For embedded storage, we return the parameters directly
        # They will be embedded in the GraphML g3 key
        
        embedded_data = {
            "format_version": "1.1",
            "total_size_bytes": total_size,
            "parameter_count": len(parameters),
            "parameters": parameters
        }
        
        # Create checksum of parameter data
        param_json = json.dumps(embedded_data, sort_keys=True)
        checksum = hashlib.sha256(param_json.encode()).hexdigest()
        
        return {
            "parameter_strategy": self.strategy,
            "parameter_file": "",  # Empty for embedded
            "checksum": f"sha256:{checksum}",
            "total_size_bytes": total_size,
            "parameter_count": len(parameters),
            "embedded_data": embedded_data
        }
    
    def _store_reference(self, output_base: str, total_size: int) -> dict[str, Any]:
        """Store reference to original ONNX file for parameters."""
        
        # Assume original ONNX file exists
        onnx_file = f"{output_base}.onnx"
        
        if not Path(onnx_file).exists():
            raise FileNotFoundError(f"Original ONNX file not found: {onnx_file}")
        
        # Calculate checksum of ONNX file
        checksum = self._calculate_file_checksum(onnx_file)
        
        return {
            "parameter_strategy": self.strategy,
            "parameter_file": Path(onnx_file).name,
            "checksum": checksum,
            "total_size_bytes": total_size,
            "parameter_count": "unknown",  # Would need to parse ONNX
            "files": {
                "reference": onnx_file
            }
        }
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return f"sha256:{sha256_hash.hexdigest()}"
    
    def load_parameters(
        self, 
        parameter_info: dict[str, Any], 
        base_path: str = ""
    ) -> dict[str, Any]:
        """
        Load parameters based on storage strategy and info.
        
        Args:
            parameter_info: Parameter information from GraphML
            base_path: Base path for resolving relative file paths
            
        Returns:
            Dictionary of parameter name -> numpy array
        """
        
        # If no parameters to load, return empty dict
        param_count = parameter_info.get("parameter_count", 0)
        # Handle both string and int values
        if str(param_count) == "0":
            return {}
        
        strategy = parameter_info.get("parameter_strategy", "sidecar")
        
        if strategy == "sidecar":
            return self._load_sidecar(parameter_info, base_path)
        elif strategy == "embedded":
            return self._load_embedded(parameter_info)
        elif strategy == "reference":
            return self._load_reference(parameter_info, base_path)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _load_sidecar(self, parameter_info: dict[str, Any], base_path: str) -> dict[str, Any]:
        """Load parameters from sidecar .onnxdata file."""
        
        param_file = parameter_info.get("parameter_file", "")
        if not param_file:
            raise ValueError("No parameter file specified for sidecar strategy")
        
        # Resolve file path
        param_path = Path(base_path) / param_file if base_path else Path(param_file)
        
        if not param_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_path}")
        
        # Verify checksum
        expected_checksum = parameter_info.get("checksum", "")
        if expected_checksum:
            actual_checksum = self._calculate_file_checksum(str(param_path))
            if actual_checksum != expected_checksum:
                raise ValueError(f"Parameter file checksum mismatch: "
                               f"expected {expected_checksum}, got {actual_checksum}")
        
        # Load parameter data
        import gzip
        with gzip.open(param_path, 'rt', encoding='utf-8') as f:
            param_data = json.load(f)
        
        # Convert back to numpy arrays
        parameters = {}
        for name, param_info in param_data["parameters"].items():
            data_bytes = base64.b64decode(param_info["data"])
            np_array = np.frombuffer(data_bytes, dtype=param_info["dtype"])
            np_array = np_array.reshape(param_info["shape"])
            parameters[name] = np_array
        
        return parameters
    
    def _load_embedded(self, parameter_info: dict[str, Any]) -> dict[str, Any]:
        """Load parameters from embedded data."""
        
        embedded_data = parameter_info.get("embedded_data")
        if not embedded_data:
            raise ValueError("No embedded parameter data found")
        
        # Verify checksum
        expected_checksum = parameter_info.get("checksum", "")
        if expected_checksum:
            param_json = json.dumps(embedded_data, sort_keys=True)
            actual_checksum = f"sha256:{hashlib.sha256(param_json.encode()).hexdigest()}"
            if actual_checksum != expected_checksum:
                raise ValueError("Embedded parameter checksum mismatch")
        
        # Convert back to numpy arrays
        parameters = {}
        for name, param_info in embedded_data["parameters"].items():
            data_bytes = base64.b64decode(param_info["data"])
            np_array = np.frombuffer(data_bytes, dtype=param_info["dtype"])
            np_array = np_array.reshape(param_info["shape"])
            parameters[name] = np_array
        
        return parameters
    
    def _load_reference(self, parameter_info: dict[str, Any], base_path: str) -> dict[str, Any]:
        """Load parameters from referenced ONNX file."""
        
        onnx_file = parameter_info.get("parameter_file", "")
        if not onnx_file:
            raise ValueError("No ONNX reference file specified")
        
        # Resolve file path
        onnx_path = Path(base_path) / onnx_file if base_path else Path(onnx_file)
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"Referenced ONNX file not found: {onnx_path}")
        
        # Verify checksum
        expected_checksum = parameter_info.get("checksum", "")
        if expected_checksum:
            actual_checksum = self._calculate_file_checksum(str(onnx_path))
            if actual_checksum != expected_checksum:
                raise ValueError("Referenced ONNX file checksum mismatch")
        
        # Load ONNX model and extract parameters
        onnx_model = onnx.load(str(onnx_path))
        
        parameters = {}
        for initializer in onnx_model.graph.initializer:
            np_array = onnx.numpy_helper.to_array(initializer)
            parameters[initializer.name] = np_array
        
        return parameters
    
    def get_storage_info(self, parameter_info: dict[str, Any]) -> dict[str, Any]:
        """Get human-readable storage information."""
        
        strategy = parameter_info.get("parameter_strategy", "unknown")
        size_bytes = parameter_info.get("total_size_bytes", 0)
        size_mb = size_bytes / (1024 * 1024)
        param_count = parameter_info.get("parameter_count", 0)
        
        return {
            "strategy": strategy,
            "size_mb": round(size_mb, 2),
            "size_bytes": size_bytes,
            "parameter_count": param_count,
            "parameter_file": parameter_info.get("parameter_file", ""),
            "checksum": parameter_info.get("checksum", "")
        }