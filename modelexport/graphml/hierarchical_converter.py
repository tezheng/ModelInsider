"""
Enhanced Hierarchical GraphML Converter

This module extends the base ONNX to GraphML converter to create hierarchical
structures that match the baseline GraphML format with proper compound nodes
for all PyTorch modules. Now supports hybrid approach combining execution
tracing with complete structural module discovery.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any

import onnx
import torch.nn as nn
from transformers import AutoModel

from .converter import ONNXToGraphMLConverter
from .metadata_reader import MetadataReader
from .onnx_parser import ONNXGraphParser
from .utils import GraphData, NodeData
from .utils import GraphMLConstants as GC
from ..core.structural_hierarchy_builder import StructuralHierarchyBuilder


class EnhancedHierarchicalConverter(ONNXToGraphMLConverter):
    """
    Enhanced converter that creates hierarchical GraphML matching baseline format.
    
    Key differences from v1:
    - No module_root wrapper
    - Direct scope names as node IDs
    - Creates compound nodes for all modules in metadata
    - Matches baseline structure exactly
    """
    
    def __init__(
        self,
        htp_metadata_path: str,
        exclude_initializers: bool = True,
        exclude_attributes: set[str] | None = None,
        use_hybrid_hierarchy: bool = True
    ):
        """Initialize with HTP metadata and optional hybrid hierarchy support."""
        super().__init__(exclude_initializers, exclude_attributes)
        self.metadata_reader = MetadataReader(htp_metadata_path)
        self.htp_metadata_path = htp_metadata_path
        self.use_hybrid_hierarchy = use_hybrid_hierarchy
        
        # Load metadata to get model info
        with open(htp_metadata_path) as f:
            self.htp_data = json.load(f)
        
        # Initialize structural hierarchy builder if using hybrid approach
        if self.use_hybrid_hierarchy:
            self.structural_builder = StructuralHierarchyBuilder(
                exceptions=["Linear", "LayerNorm", "Embedding", "Dropout", "Tanh"]
            )
    
    def convert(self, onnx_model_path: str) -> str:
        """Convert ONNX model to hierarchical GraphML matching baseline format."""
        # Load ONNX model
        model_path = Path(onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        onnx_model = onnx.load(str(model_path))
        
        # Parse basic structure
        parser = ONNXGraphParser(self.exclude_initializers, self.exclude_attributes)
        graph_data = parser.parse(onnx_model)
        
        # Add metadata
        graph_data.metadata["source_file"] = model_path.name
        graph_data.metadata["htp_file"] = Path(self.htp_metadata_path).name
        
        # Create hierarchical structure
        graphml = self._create_hierarchical_graphml(graph_data, onnx_model)
        
        # Convert to string
        ET.indent(graphml, space="  ")
        return ET.tostring(graphml, encoding='unicode', xml_declaration=True)
    
    def _create_hierarchical_graphml(self, graph_data: GraphData, onnx_model: onnx.ModelProto) -> ET.Element:
        """Create GraphML with hierarchical structure matching baseline."""
        # Create root
        graphml = self._create_graphml_root()
        
        # Define keys matching baseline
        self._define_baseline_keys(graphml)
        
        # Get model info
        model_info = self.htp_data.get("model", {})
        model_class = model_info.get("class_name", "Model")
        
        # Create main graph with model name as ID (matching baseline)
        main_graph = ET.Element("graph", attrib={
            "id": model_class,
            "edgedefault": "directed"
        })
        
        # Add graph attributes
        self._add_data(main_graph, GC.GRAPH_CLASS_NAME, model_class)
        self._add_data(main_graph, GC.GRAPH_MODULE_TYPE, "huggingface")
        self._add_data(main_graph, GC.GRAPH_EXECUTION_ORDER, "0")
        self._add_data(main_graph, GC.GRAPH_TRACED_TAG, f"/{model_class}")
        self._add_data(main_graph, GC.META_FORMAT_VERSION, "1.1")
        
        from datetime import datetime
        self._add_data(main_graph, GC.META_TIMESTAMP, datetime.now().isoformat())
        
        # Build module hierarchy from metadata with optional hybrid enhancement
        modules_data = self.htp_data.get("modules", {})
        
        if self.use_hybrid_hierarchy:
            # Enhance with structural discovery to match baseline 44 compound nodes
            enhanced_modules_data = self._enhance_with_structural_hierarchy(modules_data)
            self._create_module_hierarchy(main_graph, enhanced_modules_data, graph_data)
        else:
            self._create_module_hierarchy(main_graph, modules_data, graph_data)
        
        # Add root-level ONNX nodes (e.g., /Constant_6 tagged with /BertModel)
        self._add_root_level_onnx_nodes(main_graph, f"/{model_class}", graph_data)
        
        # Add input/output nodes at the main level
        self._add_io_nodes(main_graph, graph_data)
        
        # Add edges at the main level
        self._add_edges(main_graph, graph_data)
        
        graphml.append(main_graph)
        return graphml
    
    def _enhance_with_structural_hierarchy(self, traced_modules_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance traced module hierarchy with structural discovery to achieve 44 compound nodes.
        
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
            
            # Merge the hierarchies - traced takes priority, structural fills gaps
            enhanced_hierarchy = self._merge_hierarchies(traced_modules_data, structural_hierarchy)
            
            print(f"📊 Enhanced hierarchy has: {len(list(self._flatten_hierarchy(enhanced_hierarchy)))} modules")
            
            return enhanced_hierarchy
            
        except Exception as e:
            print(f"⚠️ Error in structural discovery: {e}")
            print("⚠️ Falling back to traced hierarchy only")
            return traced_modules_data
    
    def _merge_hierarchies(self, traced: Dict[str, Any], structural: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge traced and structural hierarchies.
        
        Priority: traced modules keep their execution order and details,
        structural modules are added to fill gaps.
        """
        # Start with traced hierarchy as base
        merged = traced.copy()
        
        # Add missing modules from structural hierarchy
        if "children" in structural:
            if "children" not in merged:
                merged["children"] = {}
            
            for struct_key, struct_data in structural["children"].items():
                if struct_key not in merged["children"]:
                    # Add missing structural module
                    merged["children"][struct_key] = struct_data
                else:
                    # Recursively merge children
                    merged["children"][struct_key] = self._merge_hierarchies(
                        merged["children"][struct_key], 
                        struct_data
                    )
        
        return merged
    
    def _flatten_hierarchy(self, hierarchy: Dict[str, Any]):
        """Flatten hierarchy to count total modules."""
        yield hierarchy
        if "children" in hierarchy:
            for child in hierarchy["children"].values():
                yield from self._flatten_hierarchy(child)

    def _create_module_hierarchy(self, parent_elem: ET.Element, modules_data: dict, graph_data: GraphData):
        """Recursively create module hierarchy matching baseline structure."""
        # Handle root module's children
        if "children" in modules_data:
            for child_name, child_data in modules_data["children"].items():
                self._create_compound_node(parent_elem, child_data, graph_data)
    
    def _create_compound_node(self, parent_elem: ET.Element, module_data: dict, graph_data: GraphData):
        """Create a compound node for a module."""
        # Use scope as node ID (matching baseline)
        scope = module_data.get("scope", "")
        if not scope:
            return
        
        class_name = module_data.get("class_name", "")
        traced_tag = module_data.get("traced_tag", "")
        execution_order = module_data.get("execution_order", 0)
        
        # Determine module type
        module_type = "pytorch" if class_name in ["Linear", "LayerNorm", "Embedding", "Dropout"] else "huggingface"
        
        # Extract basename from scope for node ID (e.g., "/embeddings" -> "embeddings")
        # But use full path for nested nodes (e.g., "embeddings.word_embeddings")
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
        
        # Create nested graph - use node_id for graph ID
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
            for child_name, child_data in module_data["children"].items():
                self._create_compound_node(nested_graph, child_data, graph_data)
        
        compound_node.append(nested_graph)
        parent_elem.append(compound_node)
    
    def _add_module_onnx_nodes(self, graph_elem: ET.Element, module_tag: str, graph_data: GraphData):
        """Add ONNX operation nodes that belong to a specific module."""
        # Find nodes with matching hierarchy tag
        for node in graph_data.nodes:
            if node.hierarchy_tag and node.hierarchy_tag.startswith(module_tag):
                # Check if this node belongs directly to this module (not a child module)
                relative_path = node.hierarchy_tag[len(module_tag):].lstrip('/')
                if '/' not in relative_path:  # Direct child
                    self._add_onnx_node(graph_elem, node)
    
    def _add_onnx_node(self, graph_elem: ET.Element, node: NodeData):
        """Add an ONNX operation node."""
        node_elem = ET.Element("node", attrib={"id": node.id})
        
        # Add attributes - include the actual ONNX operation type
        self._add_data(node_elem, GC.NODE_OP_TYPE, node.op_type or "")
        self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, node.hierarchy_tag or "")
        self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, "{}")
        self._add_data(node_elem, GC.NODE_NAME, node.name)
        
        graph_elem.append(node_elem)
    
    def _add_root_level_onnx_nodes(self, graph_elem: ET.Element, root_tag: str, graph_data: GraphData):
        """Add ONNX operation nodes that belong to the root model level."""
        # Find nodes with root-level hierarchy tag (e.g., /BertModel)
        for node in graph_data.nodes:
            if node.hierarchy_tag and node.hierarchy_tag == root_tag:
                # These are root-level operations like /Constant_6
                self._add_onnx_node(graph_elem, node)
    
    def _add_io_nodes(self, graph_elem: ET.Element, graph_data: GraphData):
        """Add input and output nodes."""
        # Add inputs
        for input_node in graph_data.inputs:
            node_elem = ET.Element("node", attrib={"id": input_node.id})
            self._add_data(node_elem, GC.NODE_OP_TYPE, "Input")
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, "")
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
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, "")
            self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, "{}")
            self._add_data(node_elem, GC.NODE_NAME, output_node.name)
            
            # Add shape comment if available
            if "shape" in output_node.attributes:
                comment = ET.Comment(f" {output_node.name}: {output_node.attributes['shape']} ")
                node_elem.append(comment)
            
            graph_elem.append(node_elem)
    
    def _add_edges(self, graph_elem: ET.Element, graph_data: GraphData):
        """Add all edges to the graph."""
        for edge in graph_data.edges:
            edge_elem = ET.Element("edge", attrib={
                "source": edge.source_id,
                "target": edge.target_id
            })
            self._add_data(edge_elem, GC.EDGE_TENSOR_NAME, edge.tensor_name)
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
    
    def _define_baseline_keys(self, graphml: ET.Element):
        """Define keys matching the baseline format."""
        # Graph attributes
        self._add_key(graphml, GC.GRAPH_CLASS_NAME, "graph", "class_name", "string")
        self._add_key(graphml, GC.GRAPH_MODULE_TYPE, "graph", "module_type", "string")
        self._add_key(graphml, GC.GRAPH_EXECUTION_ORDER, "graph", "execution_order", "int")
        self._add_key(graphml, GC.GRAPH_TRACED_TAG, "graph", "traced_tag", "string")
        
        # Node attributes
        self._add_key(graphml, GC.NODE_OP_TYPE, "node", "op_type", "string")
        self._add_key(graphml, GC.NODE_HIERARCHY_TAG, "node", "hierarchy_tag", "string")
        self._add_key(graphml, GC.NODE_ATTRIBUTES_JSON, "node", "node_attributes", "string")
        self._add_key(graphml, GC.NODE_NAME, "node", "name", "string")
        
        # Edge attributes
        self._add_key(graphml, GC.EDGE_TENSOR_NAME, "edge", "tensor_name", "string")
        
        # Metadata
        self._add_key(graphml, GC.META_SOURCE_ONNX, "graph", "source_onnx_text", "string")
        self._add_key(graphml, GC.META_SOURCE_HTP, "graph", "source_htp", "string")
        self._add_key(graphml, GC.META_FORMAT_VERSION, "graph", "format_version", "string")
        self._add_key(graphml, GC.META_TIMESTAMP, "graph", "export_timestamp", "string")
    
    def _add_key(self, parent: ET.Element, id: str, for_type: str, attr_name: str, attr_type: str):
        """Add a key definition."""
        ET.SubElement(parent, "key", attrib={
            "id": id,
            "for": for_type,
            "attr.name": attr_name,
            "attr.type": attr_type
        })
    
    def _add_data(self, parent: ET.Element, key: str, value: str):
        """Add a data element."""
        data = ET.SubElement(parent, "data", attrib={"key": key})
        data.text = value


# Backward compatibility alias
HierarchicalGraphMLConverter = EnhancedHierarchicalConverter