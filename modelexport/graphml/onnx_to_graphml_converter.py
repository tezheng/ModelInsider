"""
Unified ONNX to GraphML Converter v1.1

This module provides a unified converter that transforms ONNX models into GraphML format
with support for both flat and hierarchical structures. It combines the functionality
of the previous separate converters into a single, configurable converter.

Key Features:
- Configurable hierarchical vs flat output (hierarchical by default)
- Complete ONNX node attributes capture
- Tensor type and shape information
- Model metadata preservation
- Parameter storage management
- Bidirectional conversion support

Linear Task: TEZ-124
"""

import base64
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto
from transformers import AutoModel

from ..core.structural_hierarchy_builder import StructuralHierarchyBuilder
from .graphml_writer import GraphMLWriter
from .metadata_reader import MetadataReader
from .onnx_parser import ONNXGraphParser
from .parameter_manager import ParameterManager
from .utils import GraphData, NodeData
from .utils import GraphMLConstants as GC


class ONNXToGraphMLConverter:
    """
    Unified converter for ONNX to GraphML transformation with bidirectional support.
    
    This converter can produce both flat and hierarchical GraphML outputs,
    with full support for ONNX reconstruction (GraphML v1.1 format).
    
    Args:
        hierarchical: Whether to create hierarchical GraphML with compound nodes (default: True)
        htp_metadata_path: Path to HTP metadata JSON file (required for hierarchical mode)
        parameter_strategy: How to store parameters ('sidecar', 'embedded', 'reference')
        exclude_initializers: Whether to exclude weight/parameter tensors (default: True)
        exclude_attributes: Set of node attributes to exclude from output
        use_hybrid_hierarchy: Enable structural discovery for complete module hierarchy
    """
    
    def __init__(
        self,
        hierarchical: bool = True,
        htp_metadata_path: str | None = None,
        parameter_strategy: str = "sidecar",
        exclude_initializers: bool = True,
        exclude_attributes: set[str] | None = None,
        use_hybrid_hierarchy: bool = True
    ):
        """Initialize unified converter with configuration."""
        self.hierarchical = hierarchical
        self.htp_metadata_path = htp_metadata_path
        self.parameter_strategy = parameter_strategy
        self.exclude_initializers = exclude_initializers
        self.exclude_attributes = exclude_attributes or set()
        self.use_hybrid_hierarchy = use_hybrid_hierarchy
        
        # Validate hierarchical mode requirements
        if self.hierarchical and not self.htp_metadata_path:
            raise ValueError("HTP metadata path is required for hierarchical mode")
        
        # Initialize components
        self.parser = ONNXGraphParser(
            exclude_initializers=exclude_initializers,
            exclude_attributes=exclude_attributes
        )
        self.writer = GraphMLWriter() if not hierarchical else None
        self.parameter_manager = ParameterManager(strategy=parameter_strategy)
        self.format_version = "1.1"
        
        # Load HTP metadata if provided
        if self.htp_metadata_path:
            self.metadata_reader = MetadataReader(htp_metadata_path)
            with open(htp_metadata_path) as f:
                self.htp_data = json.load(f)
            
            # Initialize structural builder for hybrid hierarchy
            if self.use_hybrid_hierarchy:
                # Universal approach: Use default MUST-002 compliant exceptions (empty list)
                # This ensures no hardcoded patterns while maintaining universal design
                self.structural_builder = StructuralHierarchyBuilder(
                    exceptions=None  # Uses MUST-002 compliant default: [] (no torch.nn modules)
                )
        else:
            self.htp_data = None
            self.metadata_reader = None
            self.structural_builder = None
    
    def convert(self, onnx_model_path: str, output_base: str | None = None) -> str | dict[str, str]:
        """
        Convert ONNX model to GraphML format.
        
        Args:
            onnx_model_path: Path to ONNX model file
            output_base: Base path for output files (without extension)
            
        Returns:
            For flat mode: GraphML XML as string
            For hierarchical mode: Dictionary with paths to generated files
        """
        # Validate input path
        model_path = Path(onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Load ONNX model
        onnx_model = onnx.load(str(model_path))
        
        if self.hierarchical:
            return self._convert_hierarchical(onnx_model, model_path, output_base)
        else:
            return self._convert_flat(onnx_model, model_path)
    
    def _convert_flat(self, onnx_model: onnx.ModelProto, model_path: Path) -> str:
        """Convert to flat GraphML (visualization only)."""
        # Parse ONNX model
        graph_data = self.parser.parse(onnx_model)
        
        # Add source file metadata
        graph_data.metadata["source_file"] = model_path.name
        
        # Generate GraphML
        graphml_element = self.writer.write(graph_data)
        return self.writer.to_string(graphml_element)
    
    def _convert_hierarchical(
        self, 
        onnx_model: onnx.ModelProto, 
        model_path: Path,
        output_base: str | None
    ) -> dict[str, str]:
        """Convert to hierarchical GraphML with bidirectional support."""
        # Set output paths
        if output_base is None:
            output_base = model_path.stem
        
        graphml_path = f"{output_base}.graphml"
        
        # Extract and store parameters
        parameter_info = self.parameter_manager.extract_parameters(
            onnx_model, output_base
        )
        
        # Parse ONNX structure with enhanced information
        graph_data = self.parser.parse(onnx_model)
        
        # Add metadata
        graph_data.metadata["source_file"] = model_path.name
        graph_data.metadata["htp_file"] = Path(self.htp_metadata_path).name
        
        # Enhance graph data with v1.1 information
        enhanced_graph_data = self._enhance_graph_data(graph_data, onnx_model)
        
        # Create hierarchical GraphML
        graphml = self._create_hierarchical_graphml(
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
        """Enhance graph data with v1.1 information for bidirectional conversion."""
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
                # Store actual ONNX attributes
                node.onnx_attributes = node_attributes[node_name]
                node.input_names = node_inputs_outputs[node_name]["inputs"]
                node.output_names = node_inputs_outputs[node_name]["outputs"]
                node.domain = node_inputs_outputs[node_name]["domain"]
        
        # Add tensor type information
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
    
    def _create_hierarchical_graphml(
        self,
        graph_data: GraphData,
        onnx_model: onnx.ModelProto,
        parameter_info: dict[str, Any]
    ) -> ET.Element:
        """Create hierarchical GraphML with compound nodes and v1.1 metadata."""
        # Create root
        graphml = self._create_graphml_root()
        
        # Define keys
        self._define_v11_keys(graphml)
        
        # Get model info
        model_info = self.htp_data.get("model", {})
        model_class = model_info.get("class_name", "Model")
        
        # Create main graph
        main_graph = ET.Element("graph", attrib={
            "id": model_class,
            "edgedefault": "directed"
        })
        
        # Add graph metadata
        self._add_graph_metadata(main_graph, model_class, onnx_model, parameter_info)
        
        # Initialize tracking for node placement
        self.placed_nodes = set()
        
        # Build module hierarchy
        modules_data = self.htp_data.get("modules", {})
        
        if self.use_hybrid_hierarchy:
            # Enhance with structural discovery
            enhanced_modules_data = self._enhance_with_structural_hierarchy(modules_data)
            self._create_module_hierarchy(main_graph, enhanced_modules_data, graph_data)
        else:
            self._create_module_hierarchy(main_graph, modules_data, graph_data)
        
        # Add remaining ONNX nodes
        self._add_remaining_onnx_nodes(main_graph, f"/{model_class}", graph_data)
        
        # Add input/output nodes
        self._add_io_nodes(main_graph, graph_data)
        
        # Add edges
        self._add_edges(main_graph, graph_data)
        
        graphml.append(main_graph)
        return graphml
    
    def _enhance_with_structural_hierarchy(self, traced_modules_data: dict[str, Any]) -> dict[str, Any]:
        """
        Enhance traced module hierarchy with structural discovery.
        
        This method loads the original model and discovers ALL modules using named_modules(),
        then merges with the traced hierarchy to fill gaps.
        """
        try:
            # Load the original model for complete structural discovery
            model_name = self.htp_data.get("model", {}).get("name_or_path", "")
            if not model_name:
                print("⚠️ No model name found, falling back to traced hierarchy only")
                return traced_modules_data
            
            print(f"🔍 Loading model for structural discovery: {model_name}")
            model = AutoModel.from_pretrained(model_name)
            
            # Build complete structural hierarchy
            structural_hierarchy = self.structural_builder.build_complete_hierarchy(model)
            
            print(f"📊 Structural discovery found: {len(list(self._flatten_hierarchy(structural_hierarchy)))} modules")
            print(f"📊 Traced hierarchy has: {len(list(self._flatten_hierarchy(traced_modules_data)))} modules")
            
            # Merge the hierarchies
            enhanced_hierarchy = self._merge_hierarchies(traced_modules_data, structural_hierarchy)
            
            print(f"📊 Enhanced hierarchy has: {len(list(self._flatten_hierarchy(enhanced_hierarchy)))} modules")
            
            return enhanced_hierarchy
            
        except Exception as e:
            print(f"⚠️ Error in structural discovery: {e}")
            print("⚠️ Falling back to traced hierarchy only")
            return traced_modules_data
    
    def _merge_hierarchies(self, traced: dict[str, Any], structural: dict[str, Any]) -> dict[str, Any]:
        """Merge traced and structural hierarchies with conflict resolution."""
        # Start with traced hierarchy as base
        merged = traced.copy()
        
        # Add missing modules from structural hierarchy
        if "children" in structural:
            if "children" not in merged:
                merged["children"] = {}
            
            for struct_key, struct_data in structural["children"].items():
                if struct_key not in merged["children"]:
                    # Check if this structural module would create duplicates
                    if self._would_create_duplicates(struct_data, merged):
                        print(f"⚠️ Skipping structural module '{struct_key}' to prevent duplication")
                        continue
                    # Add missing structural module
                    merged["children"][struct_key] = struct_data
                else:
                    # Recursively merge children
                    merged["children"][struct_key] = self._merge_hierarchies(
                        merged["children"][struct_key], 
                        struct_data
                    )
        
        return merged
    
    def _would_create_duplicates(self, struct_module: dict[str, Any], traced_hierarchy: dict[str, Any]) -> bool:
        """Check if adding a structural module would create duplicate compound nodes."""
        if "children" not in struct_module:
            return False
        
        # Get all node IDs that exist in traced hierarchy
        traced_node_ids = set()
        self._collect_node_ids(traced_hierarchy, traced_node_ids)
        
        # Check if structural module's children would conflict
        for _child_key, child_data in struct_module["children"].items():
            child_scope = child_data.get("scope", "").lstrip("/").replace("/", ".")
            if child_scope in traced_node_ids:
                return True
        
        return False
    
    def _collect_node_ids(self, hierarchy: dict[str, Any], node_ids: set):
        """Recursively collect all node IDs from hierarchy."""
        scope = hierarchy.get("scope", "")
        if scope:
            node_id = scope.lstrip("/").replace("/", ".")
            node_ids.add(node_id)
        
        if "children" in hierarchy:
            for child in hierarchy["children"].values():
                self._collect_node_ids(child, node_ids)
    
    def _flatten_hierarchy(self, hierarchy: dict[str, Any]):
        """Flatten hierarchy to count total modules."""
        yield hierarchy
        if "children" in hierarchy:
            for child in hierarchy["children"].values():
                yield from self._flatten_hierarchy(child)
    
    def _create_module_hierarchy(self, parent_elem: ET.Element, modules_data: dict, graph_data: GraphData):
        """Recursively create module hierarchy matching baseline structure."""
        # Handle root module's children
        if "children" in modules_data:
            for _child_name, child_data in modules_data["children"].items():
                self._create_compound_node(parent_elem, child_data, graph_data)
    
    def _create_compound_node(self, parent_elem: ET.Element, module_data: dict, graph_data: GraphData):
        """Create a compound node for a module."""
        # Use scope as node ID
        scope = module_data.get("scope", "")
        if not scope:
            return
        
        class_name = module_data.get("class_name", "")
        traced_tag = module_data.get("traced_tag", "")
        execution_order = module_data.get("execution_order", 0)
        
        # Universal module type determination - no hardcoded patterns
        # All PyTorch modules inherit from nn.Module, so we use a universal approach
        module_type = "torch_module"  # Universal type that works for any PyTorch module
        
        # Extract basename from scope for node ID
        node_id = scope.lstrip("/").replace("/", ".")
        
        # Create compound node
        compound_node = ET.Element("node", attrib={"id": node_id})
        
        # Add node attributes
        self._add_data(compound_node, GC.NODE_OP_TYPE, class_name)
        self._add_data(compound_node, GC.NODE_HIERARCHY_TAG, traced_tag)
        
        # Create JSON attributes
        node_attrs = {
            "module_type": module_type,
            "execution_order": execution_order
        }
        self._add_data(compound_node, GC.NODE_ATTRIBUTES_JSON, json.dumps(node_attrs))
        self._add_data(compound_node, GC.NODE_NAME, node_id)
        
        # Create nested graph
        nested_graph = ET.Element("graph", attrib={
            "id": f"{node_id}::",
            "edgedefault": "directed"
        })
        
        # Add graph attributes
        self._add_data(nested_graph, GC.GRAPH_CLASS_NAME, class_name)
        self._add_data(nested_graph, GC.GRAPH_MODULE_TYPE, module_type)
        self._add_data(nested_graph, GC.GRAPH_EXECUTION_ORDER, str(execution_order))
        self._add_data(nested_graph, GC.GRAPH_TRACED_TAG, traced_tag)
        
        # Add ONNX nodes that belong to this module
        self._add_module_onnx_nodes(nested_graph, traced_tag, graph_data)
        
        # Recursively add children
        if "children" in module_data:
            for _child_name, child_data in module_data["children"].items():
                self._create_compound_node(nested_graph, child_data, graph_data)
        
        compound_node.append(nested_graph)
        parent_elem.append(compound_node)
    
    def _add_module_onnx_nodes(self, graph_elem: ET.Element, module_tag: str, graph_data: GraphData):
        """Add ONNX operation nodes that belong to a specific module."""
        for node in graph_data.nodes:
            # Skip nodes that have already been placed
            if node.id in self.placed_nodes:
                continue
                
            if node.hierarchy_tag and node.hierarchy_tag.startswith(module_tag):
                # Check if this node belongs directly to this module (not to a submodule)
                relative_path = node.hierarchy_tag[len(module_tag):].lstrip('/')
                if '/' not in relative_path:  # Direct child - no deeper nesting
                    self._add_onnx_node(graph_elem, node)
                    self.placed_nodes.add(node.id)
    
    def _add_onnx_node(self, graph_elem: ET.Element, node: NodeData):
        """Add an ONNX operation node with v1.1 attributes."""
        node_elem = ET.Element("node", attrib={"id": node.id})
        
        # Add attributes
        self._add_data(node_elem, GC.NODE_OP_TYPE, node.op_type or "")
        self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, node.hierarchy_tag or "")
        
        # Add actual ONNX attributes (v1.1 feature)
        onnx_attrs = getattr(node, 'onnx_attributes', {})
        self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, json.dumps(onnx_attrs))
        self._add_data(node_elem, GC.NODE_NAME, node.name)
        
        # Add v1.1 fields
        input_names = getattr(node, 'input_names', [])
        output_names = getattr(node, 'output_names', [])
        domain = getattr(node, 'domain', '')
        
        self._add_data(node_elem, "n4", json.dumps(input_names))
        self._add_data(node_elem, "n5", json.dumps(output_names))
        self._add_data(node_elem, "n6", domain)
        
        graph_elem.append(node_elem)
    
    def _add_remaining_onnx_nodes(self, graph_elem: ET.Element, root_tag: str, graph_data: GraphData):
        """Add ONNX operation nodes that haven't been placed in subgraphs yet."""
        nodes_by_parent = {}
        for node in graph_data.nodes:
            if node.id not in self.placed_nodes:
                parent_module = self._determine_parent_module(node, root_tag)
                if parent_module not in nodes_by_parent:
                    nodes_by_parent[parent_module] = []
                nodes_by_parent[parent_module].append(node)
        
        for parent_module, nodes in nodes_by_parent.items():
            if parent_module == root_tag:
                # Add directly to root graph
                for node in nodes:
                    self._add_onnx_node(graph_elem, node)
                    self.placed_nodes.add(node.id)
            else:
                # Ensure parent module exists and add nodes to it
                self._ensure_module_exists_and_add_nodes(graph_elem, parent_module, nodes, graph_data)
    
    def _determine_parent_module(self, node: NodeData, root_tag: str) -> str:
        """Determine which module this node should be placed in based on hierarchy tag."""
        hierarchy_tag = getattr(node, 'hierarchy_tag', '') or ''
        
        if not hierarchy_tag or hierarchy_tag == root_tag:
            return root_tag
        
        # Extract parent module from hierarchy tag
        # Example: "/BertModel/BertEmbeddings" -> should go in "embeddings" compound node
        # The node's hierarchy tag tells us which module it belongs to
        tag_parts = hierarchy_tag.strip('/').split('/')
        if len(tag_parts) >= 2:
            # Get the second-level module (first level is usually the model class)
            return tag_parts[1].lower()  # Convert to match compound node naming
        
        return root_tag
    
    def _ensure_module_exists_and_add_nodes(self, graph_elem: ET.Element, module_name: str, nodes: list, graph_data: GraphData):
        """Ensure the compound node exists and add operation nodes to it."""
        # Find existing compound node with this module name
        compound_node = None
        for node_elem in graph_elem.findall("node"):
            if node_elem.get("id") == module_name:
                compound_node = node_elem
                break
        
        if compound_node is None:
            # Create compound node if it doesn't exist
            compound_node = ET.Element("node", attrib={"id": module_name})
            self._add_data(compound_node, "n0", module_name.title())
            self._add_data(compound_node, "n1", f"/{module_name}")
            self._add_data(compound_node, "n2", '{"module_type": "pytorch", "execution_order": -1}')
            self._add_data(compound_node, "n3", module_name)
            
            # Create nested graph for this compound node
            nested_graph = ET.Element("graph", attrib={"id": f"{module_name}::", "edgedefault": "directed"})
            self._add_data(nested_graph, "d0", module_name.title())
            self._add_data(nested_graph, "d1", "pytorch")
            self._add_data(nested_graph, "d2", "-1")
            self._add_data(nested_graph, "d3", f"/{module_name}")
            compound_node.append(nested_graph)
            
            graph_elem.append(compound_node)
        
        # Add nodes to the nested graph
        nested_graph = compound_node.find("graph")
        if nested_graph is not None:
            for node in nodes:
                self._add_onnx_node(nested_graph, node)
                self.placed_nodes.add(node.id)
    
    def _add_io_nodes(self, graph_elem: ET.Element, graph_data: GraphData):
        """Add input and output nodes."""
        # Get model class for hierarchy tag
        model_info = self.htp_data.get("model", {})
        model_class = model_info.get("class_name", "Model")
        
        # Add inputs
        for input_node in graph_data.inputs:
            node_elem = ET.Element("node", attrib={"id": input_node.id})
            self._add_data(node_elem, GC.NODE_OP_TYPE, "Input")
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, f"/{model_class}")
            self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, "{}")
            self._add_data(node_elem, GC.NODE_NAME, input_node.name)
            
            # Add shape comment if available
            if "shape" in input_node.attributes:
                comment = ET.Comment(f" {input_node.name}: {input_node.attributes['shape']} ")
                node_elem.append(comment)
            
            graph_elem.append(node_elem)
        
        # Add outputs
        for output_node in graph_data.outputs:
            node_elem = ET.Element("node", attrib={"id": output_node.id})
            self._add_data(node_elem, GC.NODE_OP_TYPE, "Output")
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, f"/{model_class}")
            self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, "{}")
            self._add_data(node_elem, GC.NODE_NAME, output_node.name)
            
            # Add shape comment if available
            if "shape" in output_node.attributes:
                comment = ET.Comment(f" {output_node.name}: {output_node.attributes['shape']} ")
                node_elem.append(comment)
            
            graph_elem.append(node_elem)
    
    def _add_edges(self, graph_elem: ET.Element, graph_data: GraphData):
        """Add edges with v1.1 tensor information."""
        for edge in graph_data.edges:
            edge_elem = ET.Element("edge", attrib={
                "source": edge.source_id,
                "target": edge.target_id
            })
            
            # Basic tensor name
            self._add_data(edge_elem, GC.EDGE_TENSOR_NAME, edge.tensor_name)
            
            # v1.1: Tensor type and shape information
            tensor_type = getattr(edge, 'tensor_type', '')
            tensor_shape = getattr(edge, 'tensor_shape', [])
            tensor_data_ref = getattr(edge, 'tensor_data_ref', None)
            
            self._add_data(edge_elem, "t0", tensor_type)
            self._add_data(edge_elem, "t1", json.dumps(tensor_shape))
            self._add_data(edge_elem, "t2", tensor_data_ref or "")
            
            graph_elem.append(edge_elem)
    
    def _create_graphml_root(self) -> ET.Element:
        """Create root GraphML element."""
        return ET.Element("graphml", attrib={
            "xmlns": GC.GRAPHML_NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                "http://graphml.graphdrawing.org/xmlns "
                "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
            )
        })
    
    def _define_v11_keys(self, graphml: ET.Element):
        """Define GraphML v1.1 key schema."""
        # Define all keys needed for v1.1 format
        keys = [
            # Graph keys (compound nodes)
            ("d0", "graph", "class_name", "string"),
            ("d1", "graph", "module_type", "string"),
            ("d2", "graph", "execution_order", "int"),
            ("d3", "graph", "traced_tag", "string"),
            
            # Node keys (enhanced for v1.1)
            ("n0", "node", "op_type", "string"),
            ("n1", "node", "hierarchy_tag", "string"),
            ("n2", "node", "onnx_attributes", "string"),
            ("n3", "node", "name", "string"),
            ("n4", "node", "input_names", "string"),
            ("n5", "node", "output_names", "string"),
            ("n6", "node", "domain", "string"),
            
            # Edge keys (enhanced for v1.1)
            ("e0", "edge", "tensor_name", "string"),
            ("t0", "edge", "tensor_type", "string"),
            ("t1", "edge", "tensor_shape", "string"),
            ("t2", "edge", "tensor_data_ref", "string"),
            
            # Model metadata keys
            ("m0", "graph", "source_onnx_text", "string"),
            ("m1", "graph", "source_htp", "string"),
            ("m2", "graph", "format_version", "string"),
            ("m3", "graph", "export_timestamp", "string"),
            ("m4", "graph", "opset_imports", "string"),
            ("m5", "graph", "producer_name", "string"),
            ("m6", "graph", "producer_version", "string"),
            ("m7", "graph", "model_version", "string"),
            ("m8", "graph", "doc_string", "string"),
            
            # Parameter storage keys
            ("p0", "graph", "parameter_strategy", "string"),
            ("p1", "graph", "parameter_file", "string"),
            ("p2", "graph", "parameter_checksum", "string"),
            
            # Graph structure keys
            ("g0", "graph", "graph_inputs", "string"),
            ("g1", "graph", "graph_outputs", "string"),
            ("g2", "graph", "value_info", "string"),
            ("g3", "graph", "initializers_ref", "string"),
        ]
        
        for key_id, for_type, attr_name, attr_type in keys:
            self._add_key(graphml, key_id, for_type, attr_name, attr_type)
    
    def _add_key(self, parent: ET.Element, id: str, for_type: str, attr_name: str, attr_type: str):
        """Add a key definition."""
        ET.SubElement(parent, "key", attrib={
            "id": id,
            "for": for_type,
            "attr.name": attr_name,
            "attr.type": attr_type
        })
    
    def _add_graph_metadata(
        self,
        graph: ET.Element,
        model_class: str,
        onnx_model: onnx.ModelProto,
        parameter_info: dict[str, Any]
    ):
        """Add comprehensive graph metadata for v1.1."""
        # Basic graph attributes
        self._add_data(graph, GC.GRAPH_CLASS_NAME, model_class)
        self._add_data(graph, GC.GRAPH_MODULE_TYPE, "huggingface")
        self._add_data(graph, GC.GRAPH_EXECUTION_ORDER, "0")
        self._add_data(graph, GC.GRAPH_TRACED_TAG, f"/{model_class}")
        
        # Format and timestamp
        self._add_data(graph, GC.META_FORMAT_VERSION, self.format_version)
        self._add_data(graph, GC.META_TIMESTAMP, datetime.now().isoformat())
        
        # ONNX model metadata
        opset_imports = []
        for imp in onnx_model.opset_import:
            opset_imports.append({
                "domain": imp.domain or "",
                "version": imp.version
            })
        self._add_data(graph, "m4", json.dumps(opset_imports))
        
        self._add_data(graph, "m5", onnx_model.producer_name or "")
        self._add_data(graph, "m6", onnx_model.producer_version or "")
        self._add_data(graph, "m7", str(onnx_model.model_version))
        self._add_data(graph, "m8", onnx_model.doc_string or "")
        
        # Parameter storage information
        self._add_data(graph, "p0", self.parameter_strategy)
        self._add_data(graph, "p1", parameter_info.get("parameter_file", ""))
        self._add_data(graph, "p2", parameter_info.get("checksum", ""))
        
        # Graph structure information
        self._add_input_output_metadata(graph, onnx_model)
        
        # Initializers reference
        self._add_data(graph, "g3", parameter_info.get("parameter_file", ""))
    
    def _add_input_output_metadata(self, graph: ET.Element, onnx_model: onnx.ModelProto):
        """Add input/output metadata to graph."""
        # Extract input metadata
        inputs_metadata = []
        for input_info in onnx_model.graph.input:
            # Skip initializers
            if any(init.name == input_info.name for init in onnx_model.graph.initializer):
                continue
                
            tensor_info = self._extract_tensor_info(input_info)
            inputs_metadata.append({
                "name": input_info.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        
        # Extract output metadata
        outputs_metadata = []
        for output_info in onnx_model.graph.output:
            tensor_info = self._extract_tensor_info(output_info)
            outputs_metadata.append({
                "name": output_info.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        
        # Value info (intermediate tensors)
        value_info = []
        for vi in onnx_model.graph.value_info:
            tensor_info = self._extract_tensor_info(vi)
            value_info.append({
                "name": vi.name,
                "type": tensor_info["type"],
                "shape": tensor_info["shape"]
            })
        
        # Add as JSON metadata
        self._add_data(graph, GC.GRAPH_INPUTS, json.dumps(inputs_metadata))
        self._add_data(graph, GC.GRAPH_OUTPUTS, json.dumps(outputs_metadata))
        self._add_data(graph, "g2", json.dumps(value_info))
    
    def _extract_tensor_info(self, tensor_info) -> dict[str, Any]:
        """Extract tensor type and shape information."""
        info = {"type": "", "shape": []}
        
        if tensor_info.type.tensor_type:
            # Get data type
            elem_type = tensor_info.type.tensor_type.elem_type
            info["type"] = self._get_tensor_type_name(elem_type)
            
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
    
    def _get_tensor_type_name(self, elem_type: int) -> str:
        """Convert ONNX tensor type to string name."""
        type_map = {
            1: "float32",   # FLOAT
            2: "uint8",     # UINT8
            3: "int8",      # INT8
            4: "uint16",    # UINT16
            5: "int16",     # INT16
            6: "int32",     # INT32
            7: "int64",     # INT64
            8: "string",    # STRING
            9: "bool",      # BOOL
            10: "float16",  # FLOAT16
            11: "float64",  # DOUBLE
            12: "uint32",   # UINT32
            13: "uint64",   # UINT64
        }
        return type_map.get(elem_type, f"unknown_{elem_type}")
    
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
    
    def _encode_tensor_attribute(self, tensor: TensorProto) -> dict[str, Any]:
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
    
    def _add_data(self, parent: ET.Element, key: str, value: str):
        """Add a data element."""
        data = ET.SubElement(parent, "data", attrib={"key": key})
        data.text = str(value) if value is not None else ""
    
    def save(self, onnx_model_path: str, output_path: str) -> None:
        """
        Convert ONNX model and save to GraphML file.
        
        Args:
            onnx_model_path: Path to ONNX model file
            output_path: Path for output GraphML file
        """
        # Convert to GraphML
        result = self.convert(onnx_model_path)
        
        # Handle different return types
        if isinstance(result, str):
            # Flat mode returns string
            graphml_content = result
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Write to file
            output_file.write_text(graphml_content, encoding='utf-8')
        else:
            # Hierarchical mode already saved files
            pass
    
    def get_statistics(self) -> dict:
        """
        Get statistics from the last conversion.
        
        Returns:
            Dictionary with conversion statistics
        """
        return {
            "nodes": self.parser.last_node_count,
            "edges": self.parser.last_edge_count,
            "excluded_initializers": self.parser.last_initializer_count
        }