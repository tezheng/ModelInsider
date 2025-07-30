"""
Enhanced GraphML Converter v1.1 - Complete Model Interchange Format

This module implements the v1.1 GraphML specification that transforms GraphML from a 
"visualization format" into a "complete model interchange format" capable of perfect 
ONNX reconstruction.

Key Features:
- Complete ONNX node attributes capture
- Tensor type and shape information
- Model metadata preservation  
- Parameter storage management
- Graph structure specifications
- Bidirectional conversion support

Linear Task: TEZ-124
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import hashlib
import base64

import onnx
import numpy as np
from onnx import AttributeProto, TensorProto

from .hierarchical_converter import EnhancedHierarchicalConverter
from .utils import GraphData, NodeData
from .parameter_manager import ParameterManager


class EnhancedGraphMLConverter(EnhancedHierarchicalConverter):
    """
    GraphML v1.1 converter with complete ONNX reconstruction capability.
    
    Transforms GraphML from visualization → complete model interchange format.
    """
    
    def __init__(
        self,
        htp_metadata_path: str,
        parameter_strategy: str = "sidecar",
        exclude_initializers: bool = True,
        exclude_attributes: set[str] | None = None,
        use_hybrid_hierarchy: bool = True
    ):
        """Initialize v1.1 converter with parameter management."""
        super().__init__(
            htp_metadata_path=htp_metadata_path,
            exclude_initializers=exclude_initializers,
            exclude_attributes=exclude_attributes,
            use_hybrid_hierarchy=use_hybrid_hierarchy
        )
        
        self.parameter_strategy = parameter_strategy
        self.parameter_manager = ParameterManager(strategy=parameter_strategy)
        self.format_version = "1.1"
        
    def convert(self, onnx_model_path: str, output_base: str = None) -> Dict[str, str]:
        """
        Convert ONNX model to GraphML v1.1 format.
        
        Args:
            onnx_model_path: Path to ONNX model
            output_base: Base path for output files (without extension)
            
        Returns:
            Dictionary with paths to generated files
        """
        # Load ONNX model
        model_path = Path(onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        onnx_model = onnx.load(str(model_path))
        
        # Set output paths
        if output_base is None:
            output_base = model_path.stem
        
        graphml_path = f"{output_base}.graphml"
        
        # Extract and store parameters
        parameter_info = self.parameter_manager.extract_parameters(
            onnx_model, output_base
        )
        
        # Parse ONNX structure with enhanced information
        from .onnx_parser import ONNXGraphParser
        parser = ONNXGraphParser(self.exclude_initializers, self.exclude_attributes)
        graph_data = parser.parse(onnx_model)
        
        # Enhance graph data with v1.1 information
        enhanced_graph_data = self._enhance_graph_data(graph_data, onnx_model)
        
        # Create hierarchical GraphML v1.1
        graphml = self._create_enhanced_hierarchical_graphml(
            enhanced_graph_data, onnx_model, parameter_info
        )
        
        # Save GraphML
        ET.indent(graphml, space="  ")
        graphml_content = ET.tostring(graphml, encoding='unicode', xml_declaration=True)
        
        with open(graphml_path, 'w', encoding='utf-8') as f:
            f.write(graphml_content)
        
        # Return file paths
        result = {
            "graphml": graphml_path,
            "format_version": self.format_version
        }
        
        if parameter_info:
            result.update(parameter_info.get("files", {}))
            
        return result
    
    def _enhance_graph_data(self, graph_data: GraphData, onnx_model: onnx.ModelProto) -> GraphData:
        """Enhance graph data with v1.1 information."""
        
        # Build node attribute mapping
        node_attributes = {}
        node_inputs_outputs = {}
        
        for node in onnx_model.graph.node:
            # Extract ONNX attributes
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = self._extract_attribute_value(attr)
            
            node_attributes[node.name] = attrs
            node_inputs_outputs[node.name] = {
                "inputs": list(node.input),
                "outputs": list(node.output),
                "domain": getattr(node, 'domain', '')
            }
        
        # Enhance nodes with v1.1 data
        for node in graph_data.nodes:
            node_name = node.name
            if node_name in node_attributes:
                # Store actual ONNX attributes (not empty {})
                node.onnx_attributes = node_attributes[node_name]
                node.input_names = node_inputs_outputs[node_name]["inputs"]
                node.output_names = node_inputs_outputs[node_name]["outputs"] 
                node.domain = node_inputs_outputs[node_name]["domain"]
        
        # Add tensor type information from value_info and inputs/outputs
        tensor_info = {}
        
        # Process graph inputs
        for input_info in onnx_model.graph.input:
            tensor_info[input_info.name] = self._extract_tensor_info(input_info)
        
        # Process graph outputs  
        for output_info in onnx_model.graph.output:
            tensor_info[output_info.name] = self._extract_tensor_info(output_info)
            
        # Process value_info
        for value_info in onnx_model.graph.value_info:
            tensor_info[value_info.name] = self._extract_tensor_info(value_info)
        
        # Enhance edges with tensor information
        for edge in graph_data.edges:
            tensor_name = edge.tensor_name
            if tensor_name in tensor_info:
                edge.tensor_type = tensor_info[tensor_name].get("type", "")
                edge.tensor_shape = tensor_info[tensor_name].get("shape", [])
                edge.tensor_data_ref = None  # Will be set if parameter
        
        return graph_data
    
    def _extract_attribute_value(self, attr: AttributeProto) -> Any:
        """Extract value from ONNX attribute."""
        if attr.type == AttributeProto.FLOAT:
            return attr.f
        elif attr.type == AttributeProto.INT:
            return attr.i
        elif attr.type == AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == AttributeProto.TENSOR:
            # Store actual tensor data as base64 for reconstruction
            return self._encode_tensor_attribute(attr.t)
        elif attr.type == AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return str(attr)
    
    def _encode_tensor_attribute(self, tensor: TensorProto) -> Dict[str, Any]:
        """Encode tensor attribute as base64 data for storage in GraphML."""
        
        # Convert tensor to numpy array then to base64
        np_array = onnx.numpy_helper.to_array(tensor)
        data_bytes = np_array.tobytes()
        
        return {
            "tensor_type": "onnx_tensor",
            "name": tensor.name,
            "data_type": tensor.data_type,
            "dims": list(tensor.dims),
            "data_b64": base64.b64encode(data_bytes).decode('utf-8'),
            "numpy_dtype": str(np_array.dtype)
        }
    
    def _extract_tensor_info(self, tensor_info) -> Dict[str, Any]:
        """Extract tensor type and shape information."""
        info = {"type": "", "shape": []}
        
        if tensor_info.type.tensor_type:
            # Get data type
            elem_type = tensor_info.type.tensor_type.elem_type
            info["type"] = self._onnx_type_to_string(elem_type)
            
            # Get shape
            shape = []
            for dim in tensor_info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)  # Unknown dimension
            info["shape"] = shape
            
        return info
    
    def _onnx_type_to_string(self, elem_type: int) -> str:
        """Convert ONNX element type to string."""
        type_map = {
            TensorProto.FLOAT: "float32",
            TensorProto.UINT8: "uint8", 
            TensorProto.INT8: "int8",
            TensorProto.UINT16: "uint16",
            TensorProto.INT16: "int16",
            TensorProto.INT32: "int32",
            TensorProto.INT64: "int64",
            TensorProto.BOOL: "bool",
            TensorProto.FLOAT16: "float16",
            TensorProto.DOUBLE: "float64",
            TensorProto.UINT32: "uint32",
            TensorProto.UINT64: "uint64",
            TensorProto.STRING: "string"
        }
        return type_map.get(elem_type, f"unknown_{elem_type}")
    
    def _create_enhanced_hierarchical_graphml(
        self, 
        graph_data: GraphData, 
        onnx_model: onnx.ModelProto,
        parameter_info: Dict[str, Any]
    ) -> ET.Element:
        """Create GraphML v1.1 with enhanced schema."""
        
        # Create root
        graphml = self._create_graphml_root()
        
        # Define v1.1 keys
        self._define_v2_keys(graphml)
        
        # Get model info
        model_info = self.htp_data.get("model", {})
        model_class = model_info.get("class_name", "Model")
        
        # Create main graph with v1.1 metadata
        main_graph = ET.Element("graph", attrib={
            "id": model_class,
            "edgedefault": "directed"
        })
        
        # Add enhanced metadata
        self._add_v2_graph_metadata(main_graph, onnx_model, parameter_info)
        
        # Initialize tracking for node placement to prevent duplication
        self.placed_nodes = set()
        
        # Add hierarchical structure (compound nodes)
        self._add_hierarchical_structure(main_graph, graph_data)
        
        # Add remaining ONNX operation nodes with v1.1 attributes
        self._add_remaining_v2_onnx_nodes(main_graph, graph_data)
        
        # Add edges with tensor information
        self._add_v2_edges(main_graph, graph_data)
        
        graphml.append(main_graph)
        return graphml
    
    def _define_v2_keys(self, graphml: ET.Element):
        """Define GraphML v1.1 key schema."""
        
        # Keep existing keys from v1.0
        keys = [
            # Graph keys (compound nodes)
            ("d0", "graph", "class_name", "string"),
            ("d1", "graph", "module_type", "string"), 
            ("d2", "graph", "execution_order", "int"),
            ("d3", "graph", "traced_tag", "string"),
            
            # Node keys (enhanced for v1.1)
            ("n0", "node", "op_type", "string"),
            ("n1", "node", "hierarchy_tag", "string"),
            ("n2", "node", "onnx_attributes", "string"),  # CHANGED: now contains actual ONNX attrs
            ("n3", "node", "name", "string"),
            ("n4", "node", "input_names", "string"),      # NEW
            ("n5", "node", "output_names", "string"),     # NEW  
            ("n6", "node", "domain", "string"),           # NEW
            
            # Edge keys (enhanced for v1.1)
            ("e0", "edge", "tensor_name", "string"),
            ("t0", "edge", "tensor_type", "string"),      # NEW
            ("t1", "edge", "tensor_shape", "string"),     # NEW
            ("t2", "edge", "tensor_data_ref", "string"),  # NEW
            
            # Model metadata keys (existing + new)
            ("m0", "graph", "source_onnx_text", "string"),
            ("m1", "graph", "source_htp", "string"),
            ("m2", "graph", "format_version", "string"),
            ("m3", "graph", "export_timestamp", "string"),
            ("m4", "graph", "opset_imports", "string"),   # NEW
            ("m5", "graph", "producer_name", "string"),   # NEW
            ("m6", "graph", "producer_version", "string"), # NEW
            ("m7", "graph", "model_version", "string"),   # NEW
            ("m8", "graph", "doc_string", "string"),      # NEW
            
            # Parameter storage keys (new)
            ("p0", "graph", "parameter_strategy", "string"), # NEW
            ("p1", "graph", "parameter_file", "string"),     # NEW
            ("p2", "graph", "parameter_checksum", "string"), # NEW
            
            # Graph structure keys (new)
            ("g0", "graph", "graph_inputs", "string"),       # NEW
            ("g1", "graph", "graph_outputs", "string"),      # NEW
            ("g2", "graph", "value_info", "string"),         # NEW
            ("g3", "graph", "initializers_ref", "string"),   # NEW
        ]
        
        for key_id, for_type, attr_name, attr_type in keys:
            key_elem = ET.Element("key", attrib={
                "id": key_id,
                "for": for_type,
                "attr.name": attr_name,
                "attr.type": attr_type
            })
            graphml.append(key_elem)
    
    def _add_v2_graph_metadata(
        self, 
        graph: ET.Element, 
        onnx_model: onnx.ModelProto,
        parameter_info: Dict[str, Any]
    ):
        """Add v1.1 graph metadata with complete ONNX information."""
        
        # Existing metadata
        model_info = self.htp_data.get("model", {})
        graph.append(self._create_data_element("d0", model_info.get("class_name", "Model")))
        graph.append(self._create_data_element("d1", "huggingface"))
        graph.append(self._create_data_element("d2", "0"))
        graph.append(self._create_data_element("d3", f"/{model_info.get('class_name', 'Model')}"))
        
        # Format and timestamp
        graph.append(self._create_data_element("m2", self.format_version))
        from datetime import datetime
        graph.append(self._create_data_element("m3", datetime.now().isoformat()))
        
        # NEW: ONNX model metadata
        opset_imports = []
        for imp in onnx_model.opset_import:
            opset_imports.append({
                "domain": imp.domain or "",
                "version": imp.version
            })
        graph.append(self._create_data_element("m4", json.dumps(opset_imports)))
        
        graph.append(self._create_data_element("m5", onnx_model.producer_name or ""))
        graph.append(self._create_data_element("m6", onnx_model.producer_version or ""))
        graph.append(self._create_data_element("m7", str(onnx_model.model_version)))
        graph.append(self._create_data_element("m8", onnx_model.doc_string or ""))
        
        # NEW: Parameter storage information
        graph.append(self._create_data_element("p0", self.parameter_strategy))
        graph.append(self._create_data_element("p1", parameter_info.get("parameter_file", "")))
        graph.append(self._create_data_element("p2", parameter_info.get("checksum", "")))
        
        # NEW: Graph structure information
        graph_inputs = []
        for input_info in onnx_model.graph.input:
            tensor_info = self._extract_tensor_info(input_info)
            graph_inputs.append({
                "name": input_info.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        graph.append(self._create_data_element("g0", json.dumps(graph_inputs)))
        
        graph_outputs = []
        for output_info in onnx_model.graph.output:
            tensor_info = self._extract_tensor_info(output_info)
            graph_outputs.append({
                "name": output_info.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        graph.append(self._create_data_element("g1", json.dumps(graph_outputs)))
        
        # Value info (intermediate tensors)
        value_info = []
        for vi in onnx_model.graph.value_info:
            tensor_info = self._extract_tensor_info(vi)
            value_info.append({
                "name": vi.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        graph.append(self._create_data_element("g2", json.dumps(value_info)))
        
        # Initializers reference
        graph.append(self._create_data_element("g3", parameter_info.get("parameter_file", "")))
    
    def _add_remaining_v2_onnx_nodes(self, graph: ET.Element, graph_data: GraphData):
        """Add ONNX operation nodes that haven't been placed in subgraphs yet."""
        
        for node in graph_data.nodes:
            if hasattr(node, 'op_type') and node.op_type:
                # Only add nodes that haven't been placed yet to prevent duplication
                if node.name not in self.placed_nodes:
                    node_elem = ET.Element("node", attrib={"id": node.name})
                    
                    # Basic node information
                    node_elem.append(self._create_data_element("n0", node.op_type))
                    node_elem.append(self._create_data_element("n1", node.hierarchy_tag or ""))
                    
                    # ENHANCED: Actual ONNX attributes (not empty {})
                    onnx_attrs = getattr(node, 'onnx_attributes', {})
                    node_elem.append(self._create_data_element("n2", json.dumps(onnx_attrs)))
                    
                    node_elem.append(self._create_data_element("n3", node.name))
                    
                    # NEW: Input/output names and domain
                    input_names = getattr(node, 'input_names', [])
                    output_names = getattr(node, 'output_names', [])
                    domain = getattr(node, 'domain', '')
                    
                    node_elem.append(self._create_data_element("n4", json.dumps(input_names)))
                    node_elem.append(self._create_data_element("n5", json.dumps(output_names)))
                    node_elem.append(self._create_data_element("n6", domain))
                    
                    graph.append(node_elem)
                    # Track that this node has been placed
                    self.placed_nodes.add(node.name)
    
    def _add_v2_edges(self, graph: ET.Element, graph_data: GraphData):
        """Add edges with v1.1 tensor information."""
        
        for edge in graph_data.edges:
            edge_elem = ET.Element("edge", attrib={
                "source": edge.source_id,
                "target": edge.target_id
            })
            
            # Basic tensor name
            edge_elem.append(self._create_data_element("e0", edge.tensor_name))
            
            # NEW: Tensor type and shape information
            tensor_type = getattr(edge, 'tensor_type', '')
            tensor_shape = getattr(edge, 'tensor_shape', [])
            tensor_data_ref = getattr(edge, 'tensor_data_ref', None)
            
            edge_elem.append(self._create_data_element("t0", tensor_type))
            edge_elem.append(self._create_data_element("t1", json.dumps(tensor_shape)))
            edge_elem.append(self._create_data_element("t2", tensor_data_ref or ""))
            
            graph.append(edge_elem)
    
    def _create_data_element(self, key: str, value: str) -> ET.Element:
        """Create a data element with key and value."""
        data_elem = ET.Element("data", attrib={"key": key})
        data_elem.text = str(value) if value is not None else ""
        return data_elem
    
    def _add_hierarchical_structure(self, graph: ET.Element, graph_data: GraphData):
        """Add hierarchical compound node structure (reuse from parent class)."""
        # Get modules data from HTP metadata (nested structure)
        modules_data = self.htp_data.get("modules", {})
        
        # Create module hierarchy using parent class approach
        if hasattr(self, 'use_hybrid_hierarchy') and self.use_hybrid_hierarchy:
            # Enhance with structural discovery like parent class
            enhanced_modules_data = self._enhance_with_structural_hierarchy(modules_data)
            self._create_module_hierarchy(graph, enhanced_modules_data, graph_data)
        else:
            self._create_module_hierarchy(graph, modules_data, graph_data)
    
    def _add_compound_nodes_recursive(
        self, 
        parent_elem: ET.Element,
        hierarchy_data: Dict[str, Any],
        graph_data: GraphData,
        parent_scope: str
    ):
        """Recursively add compound nodes for module hierarchy."""
        
        if not isinstance(hierarchy_data, dict):
            return
        
        for module_name, module_data in hierarchy_data.items():
            if not isinstance(module_data, dict):
                continue
                
            # Skip the root model entry
            if module_name in ["class_name", "traced_tag", "scope", "execution_order", "children"]:
                continue
            
            # Create node ID from scope
            current_scope = f"{parent_scope}.{module_name}" if parent_scope else module_name
            node_id = current_scope
            
            # Create compound node
            node_elem = ET.Element("node", attrib={"id": node_id})
            
            # Add node data
            class_name = module_data.get("class_name", module_name)
            traced_tag = module_data.get("traced_tag", f"/{class_name}")
            execution_order = module_data.get("execution_order", -1)
            
            node_elem.append(self._create_data_element("n0", class_name))
            node_elem.append(self._create_data_element("n1", traced_tag))
            node_elem.append(self._create_data_element("n2", json.dumps({
                "module_type": "huggingface" if "Bert" in class_name else "pytorch",
                "execution_order": execution_order
            })))
            node_elem.append(self._create_data_element("n3", node_id))
            
            # Create nested graph for this compound node
            nested_graph = ET.Element("graph", attrib={
                "id": f"{node_id}::",
                "edgedefault": "directed"
            })
            
            # Add nested graph metadata
            nested_graph.append(self._create_data_element("d0", class_name))
            nested_graph.append(self._create_data_element("d1", "huggingface" if "Bert" in class_name else "pytorch"))
            nested_graph.append(self._create_data_element("d2", str(execution_order)))
            nested_graph.append(self._create_data_element("d3", traced_tag))
            
            # Add ONNX nodes that belong to this module
            self._add_module_onnx_nodes(nested_graph, traced_tag, graph_data)
            
            # Recursively add children
            children = module_data.get("children", {})
            if children:
                self._add_compound_nodes_recursive(
                    nested_graph, children, graph_data, current_scope
                )
            
            node_elem.append(nested_graph)
            parent_elem.append(node_elem)
    
    def _add_module_onnx_nodes(
        self, 
        graph_elem: ET.Element, 
        module_tag: str, 
        graph_data: GraphData
    ):
        """Add ONNX nodes that belong to specific module."""
        for node in graph_data.nodes:
            if hasattr(node, 'hierarchy_tag') and node.hierarchy_tag == module_tag:
                if hasattr(node, 'op_type') and node.op_type:
                    # This is an ONNX operation node, add it with v1.1 attributes
                    node_elem = ET.Element("node", attrib={"id": node.name})
                    
                    # Basic attributes
                    node_elem.append(self._create_data_element("n0", node.op_type))
                    node_elem.append(self._create_data_element("n1", node.hierarchy_tag))
                    
                    # Enhanced v1.1 attributes
                    onnx_attrs = getattr(node, 'onnx_attributes', {})
                    node_elem.append(self._create_data_element("n2", json.dumps(onnx_attrs)))
                    node_elem.append(self._create_data_element("n3", node.name))
                    
                    # New v1.1 fields
                    input_names = getattr(node, 'input_names', [])
                    output_names = getattr(node, 'output_names', [])
                    domain = getattr(node, 'domain', '')
                    
                    node_elem.append(self._create_data_element("n4", json.dumps(input_names)))
                    node_elem.append(self._create_data_element("n5", json.dumps(output_names)))
                    node_elem.append(self._create_data_element("n6", domain))
                    
                    graph_elem.append(node_elem)
                    # Track that this node has been placed to prevent duplication
                    self.placed_nodes.add(node.name)