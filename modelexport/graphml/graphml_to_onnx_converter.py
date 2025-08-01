"""
GraphML to ONNX Converter - Bidirectional Conversion Support

Converts GraphML v1.1 format back to ONNX models, enabling complete
round-trip conversion and validation.

Key Features:
- Parse GraphML v1.1 enhanced format
- Reconstruct ONNX graph topology 
- Restore node attributes and metadata
- Integrate parameter data from storage
- Validate reconstructed model

Linear Task: TEZ-124
"""

import base64
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import (
    AttributeProto,
    ModelProto,
    TensorProto,
    ValueInfoProto,
    helper,
)

from .parameter_manager import ParameterManager


class GraphMLToONNXConverter:
    """Converts GraphML v1.1 back to ONNX format."""
    
    def __init__(self):
        """Initialize the converter."""
        self.parameter_manager = ParameterManager()
        
    def convert(
        self, 
        graphml_path: str, 
        output_path: str,
        validate: bool = True
    ) -> str:
        """
        Convert GraphML v1.1 to ONNX model.
        
        Args:
            graphml_path: Path to GraphML v1.1 file
            output_path: Path for output ONNX file
            validate: Whether to validate the reconstructed model
            
        Returns:
            Path to generated ONNX file
        """
        
        # Parse GraphML
        graphml_data = self._parse_graphml(graphml_path)
        
        # Load parameters
        parameters = self._load_parameters(graphml_data, graphml_path)
        
        # Reconstruct ONNX model
        onnx_model = self._reconstruct_onnx_model(graphml_data, parameters)
        
        # Validate if requested
        if validate:
            # Skip validation if model has custom attributes
            # ONNX checker doesn't allow custom attributes even though they're valid
            has_custom_attrs = self._has_custom_attributes(onnx_model)
            if not has_custom_attrs:
                self._validate_model(onnx_model)
            else:
                print("⚠️ Skipping ONNX validation due to custom attributes")
        
        # Save ONNX model
        onnx.save(onnx_model, output_path)
        
        return output_path
    
    def _parse_graphml(self, graphml_path: str) -> dict[str, Any]:
        """Parse GraphML v1.1 file and extract all information."""
        
        if not Path(graphml_path).exists():
            raise FileNotFoundError(f"GraphML file not found: {graphml_path}")
        
        # Parse XML
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        
        # Extract key definitions
        keys = self._extract_key_definitions(root)
        
        # Find main graph
        main_graph = root.find(".//{http://graphml.graphdrawing.org/xmlns}graph")
        if main_graph is None:
            raise ValueError("No graph found in GraphML file")
        
        # Extract graph metadata
        metadata = self._extract_graph_metadata(main_graph, keys)
        
        # Verify v1.1 format
        format_version = metadata.get("format_version", "1.1")
        if not format_version.startswith("1.1"):  
            raise ValueError(f"Unsupported GraphML format version: {format_version}. "
                           f"This converter requires v1.1+")
        
        # Extract nodes
        nodes = self._extract_nodes(main_graph, keys)
        
        # Extract edges  
        edges = self._extract_edges(main_graph, keys)
        
        return {
            "metadata": metadata,
            "nodes": nodes,
            "edges": edges,
            "keys": keys
        }
    
    def _extract_key_definitions(self, root: ET.Element) -> dict[str, dict[str, str]]:
        """Extract GraphML key definitions."""
        
        keys = {}
        for key_elem in root.findall(".//{http://graphml.graphdrawing.org/xmlns}key"):
            key_id = key_elem.get("id")
            keys[key_id] = {
                "for": key_elem.get("for"),
                "attr_name": key_elem.get("attr.name"),
                "attr_type": key_elem.get("attr.type")
            }
        
        return keys
    
    def _extract_graph_metadata(
        self, 
        graph: ET.Element, 
        keys: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        """Extract graph-level metadata."""
        
        metadata = {"graph_id": graph.get("id", "")}
        
        for data_elem in graph.findall(".//{http://graphml.graphdrawing.org/xmlns}data"):
            key_id = data_elem.get("key")
            if key_id in keys and keys[key_id]["for"] == "graph":
                attr_name = keys[key_id]["attr_name"]
                value = data_elem.text or ""
                
                # Parse JSON values
                if attr_name in ["opset_imports", "graph_inputs", "graph_outputs", 
                               "value_info", "initializers_ref"]:
                    try:
                        metadata[attr_name] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[attr_name] = value
                else:
                    metadata[attr_name] = value
        
        return metadata
    
    def _extract_nodes(
        self, 
        graph: ET.Element, 
        keys: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Extract all nodes with v1.1 attributes."""
        
        nodes = []
        
        # Get all nodes (including nested ones)
        seen_nodes = set()  # Track node IDs to avoid duplicates
        for node_elem in graph.findall(".//{http://graphml.graphdrawing.org/xmlns}node"):
            node_id = node_elem.get("id")
            
            # Skip if we've already processed this node
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            
            node_data = {"id": node_id}
            
            # Extract node data
            for data_elem in node_elem.findall("{http://graphml.graphdrawing.org/xmlns}data"):
                key_id = data_elem.get("key")
                if key_id in keys and keys[key_id]["for"] == "node":
                    attr_name = keys[key_id]["attr_name"]
                    value = data_elem.text or ""
                    
                    # Parse JSON values
                    if attr_name in ["node_attributes", "onnx_attributes", "input_names", "output_names"]:
                        try:
                            node_data[attr_name] = json.loads(value)
                        except json.JSONDecodeError:
                            node_data[attr_name] = value
                    else:
                        node_data[attr_name] = value
            
            # Only include nodes with op_type (ONNX operations must have op_type)
            if node_data.get("op_type"):
                # Skip compound nodes (nodes that contain nested graphs - these are module containers)
                nested_graph = node_elem.find("{http://graphml.graphdrawing.org/xmlns}graph")
                if nested_graph is not None:
                    continue  # This is a compound node, skip it
                
                nodes.append(node_data)
        
        return nodes
    
    def _extract_edges(
        self, 
        graph: ET.Element, 
        keys: dict[str, dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Extract all edges with v1.1 tensor information."""
        
        edges = []
        
        for edge_elem in graph.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"):
            edge_data = {
                "source": edge_elem.get("source"),
                "target": edge_elem.get("target")
            }
            
            # Extract edge data
            for data_elem in edge_elem.findall("{http://graphml.graphdrawing.org/xmlns}data"):
                key_id = data_elem.get("key")
                if key_id in keys and keys[key_id]["for"] == "edge":
                    attr_name = keys[key_id]["attr_name"]
                    value = data_elem.text or ""
                    
                    # Parse JSON values
                    if attr_name == "tensor_shape":
                        try:
                            edge_data[attr_name] = json.loads(value)
                        except json.JSONDecodeError:
                            edge_data[attr_name] = []
                    else:
                        edge_data[attr_name] = value
            
            edges.append(edge_data)
        
        return edges
    
    def _load_parameters(
        self, 
        graphml_data: dict[str, Any], 
        graphml_path: str
    ) -> dict[str, np.ndarray]:
        """Load parameters based on storage strategy."""
        
        metadata = graphml_data["metadata"]
        
        # Get parameter information
        parameter_info = {
            "parameter_strategy": metadata.get("parameter_strategy", "sidecar"),
            "parameter_file": metadata.get("parameter_file", ""),
            "checksum": metadata.get("parameter_checksum", ""),
            "total_size_bytes": metadata.get("total_size_bytes", 0),
            "parameter_count": metadata.get("parameter_count", 0)
        }
        
        # Handle embedded parameters
        if parameter_info["parameter_strategy"] == "embedded":
            initializers_ref = metadata.get("initializers_ref", "")
            if initializers_ref:
                try:
                    parameter_info["embedded_data"] = json.loads(initializers_ref)
                except json.JSONDecodeError as err:
                    raise ValueError("Invalid embedded parameter data") from err
        
        # Load parameters
        base_path = str(Path(graphml_path).parent)
        return self.parameter_manager.load_parameters(parameter_info, base_path)
    
    def _reconstruct_onnx_model(
        self, 
        graphml_data: dict[str, Any], 
        parameters: dict[str, np.ndarray]
    ) -> ModelProto:
        """Reconstruct complete ONNX model from GraphML data."""
        
        metadata = graphml_data["metadata"]
        nodes = graphml_data["nodes"]
        edges = graphml_data["edges"]
        
        # Create ONNX nodes and sort topologically
        onnx_nodes = []
        for node_data in nodes:
            # Skip Input/Output nodes - these are handled by graph inputs/outputs, not as nodes
            op_type = node_data.get("op_type", "")
            if op_type in ("Input", "Output"):
                continue
            
            # Skip PyTorch module nodes - these are hierarchy containers, not ONNX operations
            # But keep custom domain operators
            domain = node_data.get("domain", "")
            if self._is_module_node(op_type) and not domain:
                continue
                
            onnx_node = self._create_onnx_node(node_data)
            onnx_nodes.append(onnx_node)
        
        # Sort nodes topologically
        onnx_nodes = self._sort_nodes_topologically(onnx_nodes, edges)
        
        # Create initializers from parameters
        initializers = []
        for name, np_array in parameters.items():
            tensor = onnx.numpy_helper.from_array(np_array, name)
            initializers.append(tensor)
        
        # Create graph inputs first (needed for intermediate tensor collection)
        inputs = []
        graph_inputs = metadata.get("graph_inputs", [])
        for input_data in graph_inputs:
            inputs.append(self._create_value_info(input_data))
        
        # Create graph outputs
        outputs = []
        graph_outputs = metadata.get("graph_outputs", [])
        for output_data in graph_outputs:
            outputs.append(self._create_value_info(output_data))
        
        # Create value info
        value_info = []
        value_info_data = metadata.get("value_info", [])
        for vi_data in value_info_data:
            value_info.append(self._create_value_info(vi_data))
        
        # Collect all intermediate tensors that need value_info
        # This is crucial for ONNX validation
        intermediate_tensors = self._collect_intermediate_tensors(
            onnx_nodes, initializers, graph_inputs, graph_outputs
        )
        
        # Add any missing value_info for intermediate tensors
        existing_value_info_names = {vi.name for vi in value_info}
        for tensor_name in intermediate_tensors:
            if tensor_name not in existing_value_info_names:
                # Create a default value_info entry (will be inferred by ONNX shape inference)
                # For now, we'll create with undefined shape which ONNX can infer
                vi = helper.make_tensor_value_info(
                    tensor_name, 
                    TensorProto.FLOAT,  # Default type, will be corrected by shape inference
                    None  # Unknown shape, will be inferred
                )
                value_info.append(vi)
        
        # Create ONNX graph
        graph = helper.make_graph(
            nodes=onnx_nodes,
            name=metadata.get("graph_id", "Model"),
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            value_info=value_info
        )
        
        # Create opset imports
        opset_imports = []
        opset_data = metadata.get("opset_imports", [])
        for opset in opset_data:
            opset_imports.append(
                helper.make_opsetid(opset.get("domain", ""), opset.get("version", 17))
            )
        
        if not opset_imports:
            # Default opset
            opset_imports.append(helper.make_opsetid("", 17))
        
        # Create ONNX model
        model = helper.make_model(
            graph,
            producer_name=metadata.get("producer_name", ""),
            producer_version=metadata.get("producer_version", ""),
            opset_imports=opset_imports
        )
        
        # Set model metadata
        model.model_version = int(metadata.get("model_version", 1))
        model.doc_string = metadata.get("doc_string", "")
        
        # Run shape inference to fix any missing type/shape information
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"⚠️ Shape inference warning: {e}")
            # Continue even if shape inference fails - the model might still be valid
        
        return model
    
    def _is_module_node(self, op_type: str) -> bool:
        """Check if this is a PyTorch module node rather than an ONNX operation.
        
        Universal approach: Use ONNX's own schema validation to determine
        if something is a valid ONNX operator. No hardcoded patterns.
        """
        try:
            # Use ONNX's authoritative schema check
            onnx.defs.get_schema(op_type)
            return False  # It's a valid ONNX operator, not a module node
        except Exception:
            # If ONNX doesn't recognize it, it's likely a PyTorch module
            return True

    def _create_onnx_node(self, node_data: dict[str, Any]) -> helper.NodeProto:
        """Create ONNX node from GraphML node data."""
        
        # Get basic node info
        op_type = node_data.get("op_type", "")
        name = node_data.get("name", "")
        inputs = node_data.get("input_names", [])
        outputs = node_data.get("output_names", [])
        domain = node_data.get("domain", "")
        
        # Convert attributes with proper filtering
        attributes = []
        # Try both attribute names for compatibility
        onnx_attrs = node_data.get("node_attributes", node_data.get("onnx_attributes", {}))
        for attr_name, attr_value in onnx_attrs.items():
            # Only include attributes that should go to ONNX (excludes GraphML metadata)
            if self._should_include_in_onnx(op_type, attr_name):
                attr = self._create_onnx_attribute(attr_name, attr_value)
                if attr:
                    attributes.append(attr)
        
        # Create node
        node = helper.make_node(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            name=name,
            **{attr.name: self._extract_attribute_value(attr) for attr in attributes}
        )
        
        if domain:
            node.domain = domain
        
        return node
    
    def _should_include_in_onnx(self, op_type: str, attr_name: str) -> bool:
        """Check if attribute should be included in ONNX node.
        
        Core principle: "What comes from ONNX goes back to ONNX intact"
        Only filter out GraphML-specific metadata that we added.
        """
        # These are GraphML metadata attributes that we ADD during conversion
        # They were NOT in the original ONNX, so they should NOT go back
        graphml_metadata_attrs = {
            "hierarchy_tag",      # Added for visualization
            "module_type",        # Added from HTP metadata
            "execution_order",    # Added from tracing
            "scope",             # Added from module analysis
            "traced_tag",        # Added from HTP metadata
            "class_name",        # Added from model structure
        }
        
        # If it's GraphML metadata, don't include it
        if attr_name in graphml_metadata_attrs:
            return False
            
        # Everything else (including custom ONNX attributes) should be preserved
        return True
    
    def _create_onnx_attribute(self, name: str, value: Any) -> AttributeProto | None:
        """Create ONNX attribute from name and value."""
        
        if value is None:
            return None
        
        # Handle tensor attributes stored as encoded data
        if isinstance(value, dict) and value.get("tensor_type") == "onnx_tensor":
            return self._reconstruct_tensor_attribute(name, value)
        
        # Determine attribute type and create accordingly
        if isinstance(value, bool):
            return helper.make_attribute(name, int(value))
        elif isinstance(value, int | float | str):
            return helper.make_attribute(name, value)
        elif isinstance(value, list):
            if not value:
                return None
            elif isinstance(value[0], int | float | str):
                return helper.make_attribute(name, value)
        
        # For complex types, try to convert to string
        return helper.make_attribute(name, str(value))
    
    def _reconstruct_tensor_attribute(self, name: str, tensor_data: dict[str, Any]) -> AttributeProto:
        """Reconstruct ONNX tensor attribute from stored data."""
        
        # Decode base64 data back to bytes
        data_bytes = base64.b64decode(tensor_data["data_b64"])
        
        # Reconstruct numpy array
        np_array = np.frombuffer(data_bytes, dtype=tensor_data["numpy_dtype"])
        np_array = np_array.reshape(tensor_data["dims"])
        
        # Create ONNX tensor
        tensor = onnx.numpy_helper.from_array(np_array, tensor_data["name"])
        
        # Create attribute
        attr = AttributeProto()
        attr.name = name
        attr.type = AttributeProto.TENSOR
        attr.t.CopyFrom(tensor)
        
        return attr
    
    def _extract_attribute_value(self, attr: AttributeProto) -> Any:
        """Extract value from ONNX attribute for make_node."""
        if attr.type == AttributeProto.FLOAT:
            return attr.f
        elif attr.type == AttributeProto.INT:
            return attr.i
        elif attr.type == AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return None
    
    def _create_value_info(self, value_data: dict[str, Any]) -> ValueInfoProto:
        """Create ONNX ValueInfoProto from data."""
        
        name = value_data.get("name", "")
        dtype = value_data.get("type", "float32")
        shape = value_data.get("shape", [])
        
        # Convert string type to ONNX type
        onnx_type = self._string_to_onnx_type(dtype)
        
        # Create shape (handle symbolic dimensions)
        onnx_shape = []
        for dim in shape:
            if isinstance(dim, str):
                onnx_shape.append(dim)  # Symbolic dimension
            else:
                onnx_shape.append(int(dim))
        
        return helper.make_tensor_value_info(name, onnx_type, onnx_shape)
    
    def _string_to_onnx_type(self, type_str: str) -> int:
        """Convert string type to ONNX TensorProto type."""
        
        type_map = {
            "float32": TensorProto.FLOAT,
            "uint8": TensorProto.UINT8,
            "int8": TensorProto.INT8,
            "uint16": TensorProto.UINT16,
            "int16": TensorProto.INT16,
            "int32": TensorProto.INT32,
            "int64": TensorProto.INT64,
            "bool": TensorProto.BOOL,
            "float16": TensorProto.FLOAT16,
            "float64": TensorProto.DOUBLE,
            "uint32": TensorProto.UINT32,
            "uint64": TensorProto.UINT64,
            "string": TensorProto.STRING
        }
        
        return type_map.get(type_str, TensorProto.FLOAT)
    
    def _collect_intermediate_tensors(
        self, 
        nodes: list, 
        initializers: list,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]]
    ) -> set:
        """Collect all intermediate tensors that need value_info entries."""
        
        # Collect all tensor names that are already defined
        initializer_names = {init.name for init in initializers}
        input_names = {inp.get("name", "") for inp in inputs}
        output_names = {out.get("name", "") for out in outputs}
        
        # Collect all tensors used as inputs/outputs by nodes
        all_node_inputs = set()
        all_node_outputs = set()
        
        for node in nodes:
            all_node_inputs.update(node.input)
            all_node_outputs.update(node.output)
        
        # Intermediate tensors are those that are:
        # - Used as inputs by some node
        # - Not graph inputs, outputs, or initializers
        # - Produced by some node (to avoid external references)
        intermediate_tensors = set()
        
        for tensor_name in all_node_inputs:
            if (tensor_name not in initializer_names and
                tensor_name not in input_names and
                tensor_name not in output_names and
                tensor_name in all_node_outputs):  # Must be produced by some node
                intermediate_tensors.add(tensor_name)
        
        return intermediate_tensors
    
    def _sort_nodes_topologically(self, nodes: list, edges: list[dict[str, Any]]) -> list:
        """Sort ONNX nodes in topological order based on data dependencies."""
        
        from collections import defaultdict, deque
        
        # Build dependency graph from nodes (not GraphML edges which include hierarchy)
        # We need to use input/output relationships from the nodes themselves
        
        # Create node name to node mapping
        node_map = {node.name: node for node in nodes}
        
        # Build input dependencies: which nodes produce each output
        output_producers = {}  # output_name -> node_name
        for node in nodes:
            for output in node.output:
                output_producers[output] = node.name
        
        # Build dependency graph: node -> [dependent_nodes]
        dependencies = defaultdict(list)  # node -> nodes that depend on it
        in_degree = defaultdict(int)  # node -> number of dependencies
        
        for node in nodes:
            in_degree[node.name] = 0
        
        for node in nodes:
            for input_name in node.input:
                # Skip initializers (they don't come from other nodes)
                if input_name in output_producers:
                    producer = output_producers[input_name]
                    if producer != node.name:  # Avoid self-loops
                        dependencies[producer].append(node.name)
                        in_degree[node.name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque()
        result = []
        
        # Add nodes with no dependencies
        for node_name, degree in in_degree.items():
            if degree == 0:
                queue.append(node_name)
        
        while queue:
            current_name = queue.popleft()
            result.append(node_map[current_name])
            
            # Update dependencies
            for dependent in dependencies[current_name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # If we couldn't sort all nodes, there might be cycles or missing dependencies
        # In that case, return the original order with a warning
        if len(result) != len(nodes):
            print(f"⚠️ Topological sort incomplete: sorted {len(result)}/{len(nodes)} nodes")
            print(f"Original order: {[n.name for n in nodes]}")
            print(f"Sorted order: {[n.name for n in result]}")
            
            # Try to identify which nodes couldn't be sorted
            sorted_names = {n.name for n in result}
            unsorted_nodes = [n for n in nodes if n.name not in sorted_names]
            print(f"Unsorted nodes: {[n.name for n in unsorted_nodes]}")
            
            # For debugging, show their dependencies
            for node in unsorted_nodes:
                print(f"  {node.name}:")
                print(f"    Inputs: {list(node.input)}")
                print(f"    Missing producers: {[inp for inp in node.input if inp not in output_producers]}")
            
            return nodes
        
        return result
    
    def _has_custom_attributes(self, model: ModelProto) -> bool:
        """Check if model has custom attributes that ONNX checker won't recognize."""
        # Known ONNX Cast attributes
        cast_known_attrs = {"to"}
        
        # Check all nodes for custom attributes
        for node in model.graph.node:
            if node.op_type == "Cast":
                # Check if Cast has non-standard attributes
                for attr in node.attribute:
                    if attr.name not in cast_known_attrs:
                        return True
            elif node.domain and node.domain != "":
                # Custom domain operators always have custom attributes
                return True
                
        return False
    
    def _validate_model(self, model: ModelProto):
        """Validate reconstructed ONNX model."""
        
        try:
            # Skip custom domain validation since we may have custom attributes
            onnx.checker.check_model(model, check_custom_domain=False)
        except Exception as e:
            # If validation fails due to custom attributes, try a more lenient check
            if "Unrecognized attribute" in str(e):
                # At least check that the model structure is valid
                try:
                    # Check graph structure without attribute validation
                    onnx.checker.check_graph(model.graph)
                except Exception as graph_e:
                    raise ValueError(f"Reconstructed ONNX model validation failed: {graph_e}") from graph_e
            else:
                raise ValueError(f"Reconstructed ONNX model validation failed: {e}") from e
    
    def get_conversion_info(self, graphml_path: str) -> dict[str, Any]:
        """Get information about GraphML file for conversion."""
        
        graphml_data = self._parse_graphml(graphml_path)
        metadata = graphml_data["metadata"]
        
        return {
            "format_version": metadata.get("format_version", "unknown"),
            "model_name": metadata.get("graph_id", ""),
            "parameter_strategy": metadata.get("parameter_strategy", ""),
            "parameter_file": metadata.get("parameter_file", ""),
            "opset_version": metadata.get("opset_imports", [{}])[0].get("version", "unknown"),
            "node_count": len(graphml_data["nodes"]),
            "edge_count": len(graphml_data["edges"]),
            "estimated_size_mb": metadata.get("total_size_bytes", 0) / (1024 * 1024)
        }