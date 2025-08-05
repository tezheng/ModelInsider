"""
GraphML XML Writer

This module generates GraphML XML from internal graph data structures,
following the GraphML v1.3 specification for compatibility with visualization tools.
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime

from .constants import GRAPHML_FORMAT_VERSION
from .utils import EdgeData, GraphData, NodeData, NodeType
from .utils import GraphMLConstants as GC


class GraphMLWriter:
    """
    Generate GraphML XML from graph data.
    
    This writer creates standard GraphML output that can be read by
    visualization tools like yEd, Gephi, and others.
    """
    
    def __init__(self):
        # Register namespace
        ET.register_namespace("", GC.GRAPHML_NS)
    
    def write(self, graph_data: GraphData) -> ET.Element:
        """
        Convert graph data to GraphML XML structure.
        
        Args:
            graph_data: Internal graph representation
            
        Returns:
            Root GraphML XML element
        """
        # Create root element
        graphml = self._create_graphml_root()
        
        # Define attribute keys
        self._define_attribute_keys(graphml)
        
        # Create main graph
        graph = self._create_graph_element(graph_data)
        graphml.append(graph)
        
        return graphml
    
    def to_string(self, element: ET.Element, pretty: bool = True) -> str:
        """
        Convert XML element to string.
        
        Args:
            element: XML element tree
            pretty: Whether to format with indentation
            
        Returns:
            XML as string
        """
        if pretty:
            ET.indent(element, space="  ")
        
        return ET.tostring(element, encoding='unicode', xml_declaration=True)
    
    def _create_graphml_root(self) -> ET.Element:
        """Create root GraphML element with namespace."""
        return ET.Element("graphml", attrib={
            "xmlns": GC.GRAPHML_NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                "http://graphml.graphdrawing.org/xmlns "
                "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
            )
        })
    
    def _define_attribute_keys(self, graphml: ET.Element) -> None:
        """Define GraphML v1.3 attribute keys (31 total)."""
        # Graph attributes (for compound nodes) - 4 keys
        self._add_key(graphml, GC.GRAPH_CLASS_NAME, "graph", "class_name", "string")
        self._add_key(graphml, GC.GRAPH_MODULE_TYPE, "graph", "module_type", "string")
        self._add_key(graphml, GC.GRAPH_EXECUTION_ORDER, "graph", "execution_order", "int")
        self._add_key(graphml, GC.GRAPH_TRACED_TAG, "graph", "traced_tag", "string")
        
        # Node attributes - 7 keys
        self._add_key(graphml, GC.NODE_OP_TYPE, "node", "op_type", "string")
        self._add_key(graphml, GC.NODE_HIERARCHY_TAG, "node", "hierarchy_tag", "string")
        self._add_key(graphml, GC.NODE_ATTRIBUTES_JSON, "node", "onnx_attributes", "string")  # v1.3 name
        self._add_key(graphml, GC.NODE_NAME, "node", "name", "string")
        self._add_key(graphml, GC.NODE_INPUT_NAMES, "node", "input_names", "string")  # v1.3
        self._add_key(graphml, GC.NODE_OUTPUT_NAMES, "node", "output_names", "string")  # v1.3
        self._add_key(graphml, GC.NODE_DOMAIN, "node", "domain", "string")  # v1.3
        
        # Edge attributes - 4 keys (v1.3)
        self._add_key(graphml, GC.EDGE_TENSOR_NAME, "edge", "tensor_name", "string")
        self._add_key(graphml, GC.EDGE_TENSOR_TYPE, "edge", "tensor_type", "string")  # v1.3
        self._add_key(graphml, GC.EDGE_TENSOR_SHAPE, "edge", "tensor_shape", "string")  # v1.3
        self._add_key(graphml, GC.EDGE_TENSOR_DATA_REF, "edge", "tensor_data_ref", "string")  # v1.3
        
        # Model metadata - 9 keys (v1.3 naming)
        self._add_key(graphml, GC.META_SOURCE_ONNX, "graph", "source_onnx_file", "string")  # v1.3 name
        self._add_key(graphml, GC.META_SOURCE_HTP, "graph", "source_htp_file", "string")  # v1.3 name
        self._add_key(graphml, GC.META_FORMAT_VERSION, "graph", "format_version", "string")
        self._add_key(graphml, GC.META_TIMESTAMP, "graph", "export_timestamp", "string")
        self._add_key(graphml, GC.META_OPSET_IMPORTS, "graph", "opset_imports", "string")  # v1.3
        self._add_key(graphml, GC.META_PRODUCER_NAME, "graph", "producer_name", "string")  # v1.3
        self._add_key(graphml, GC.META_PRODUCER_VERSION, "graph", "producer_version", "string")  # v1.3
        self._add_key(graphml, GC.META_MODEL_VERSION, "graph", "model_version", "string")  # v1.3
        self._add_key(graphml, GC.META_DOC_STRING, "graph", "doc_string", "string")  # v1.3
        
        # Parameter storage - 3 keys (v1.3)
        self._add_key(graphml, GC.PARAM_STRATEGY, "graph", "parameter_strategy", "string")  # v1.3
        self._add_key(graphml, GC.PARAM_FILE, "graph", "parameter_file", "string")  # v1.3
        self._add_key(graphml, GC.PARAM_CHECKSUM, "graph", "parameter_checksum", "string")  # v1.3
        
        # Graph I/O - 4 keys (v1.3)
        self._add_key(graphml, GC.GRAPH_INPUTS, "graph", "graph_inputs", "string")  # v1.3
        self._add_key(graphml, GC.GRAPH_OUTPUTS, "graph", "graph_outputs", "string")  # v1.3
        self._add_key(graphml, GC.GRAPH_VALUE_INFO, "graph", "value_info", "string")  # v1.3
        self._add_key(graphml, GC.GRAPH_INITIALIZERS_REF, "graph", "initializers_ref", "string")  # v1.3
    
    def _add_key(
        self,
        parent: ET.Element,
        id: str,
        for_type: str,
        attr_name: str,
        attr_type: str,
        desc: str | None = None
    ) -> None:
        """Add a key definition to GraphML."""
        key = ET.SubElement(parent, "key", attrib={
            "id": id,
            "for": for_type,
            "attr.name": attr_name,
            "attr.type": attr_type
        })
        if desc:
            desc_elem = ET.SubElement(key, "desc")
            desc_elem.text = desc
    
    def _create_graph_element(self, graph_data: GraphData) -> ET.Element:
        """Create the main graph element."""
        graph = ET.Element("graph", attrib={
            "id": "G",
            "edgedefault": "directed"
        })
        
        # Add metadata
        self._add_graph_metadata(graph, graph_data)
        
        # Add input nodes
        for input_node in graph_data.inputs:
            node_elem = self._create_node_element(input_node)
            graph.append(node_elem)
        
        # Add operation nodes
        for node in graph_data.nodes:
            node_elem = self._create_node_element(node)
            graph.append(node_elem)
        
        # Add output nodes
        for output_node in graph_data.outputs:
            node_elem = self._create_node_element(output_node)
            graph.append(node_elem)
        
        # Add edges
        for edge in graph_data.edges:
            edge_elem = self._create_edge_element(edge)
            graph.append(edge_elem)
        
        return graph
    
    def _add_graph_metadata(self, graph: ET.Element, graph_data: GraphData) -> None:
        """Add v1.3 metadata to graph element."""
        metadata = graph_data.metadata
        
        # Required metadata fields
        self._add_data(graph, GC.META_SOURCE_ONNX, metadata.get("source_file", ""))
        self._add_data(graph, GC.META_SOURCE_HTP, metadata.get("htp_file", ""))
        self._add_data(graph, GC.META_FORMAT_VERSION, metadata.get("format_version", GRAPHML_FORMAT_VERSION))
        self._add_data(graph, GC.META_TIMESTAMP, datetime.now().isoformat())
        
        # ONNX model metadata (v1.3)
        self._add_data(graph, GC.META_OPSET_IMPORTS, metadata.get("opset_imports", "[]"))
        self._add_data(graph, GC.META_PRODUCER_NAME, metadata.get("producer_name", ""))
        self._add_data(graph, GC.META_PRODUCER_VERSION, metadata.get("producer_version", ""))
        self._add_data(graph, GC.META_MODEL_VERSION, metadata.get("model_version", "0"))
        self._add_data(graph, GC.META_DOC_STRING, metadata.get("doc_string", ""))
        
        # Parameter metadata (v1.3)
        self._add_data(graph, GC.PARAM_STRATEGY, metadata.get("parameter_strategy", "embedded"))
        self._add_data(graph, GC.PARAM_FILE, metadata.get("parameter_file", ""))
        self._add_data(graph, GC.PARAM_CHECKSUM, metadata.get("parameter_checksum", ""))
        
        # Graph I/O metadata (v1.3)
        self._add_data(graph, GC.GRAPH_INPUTS, metadata.get("graph_inputs", "[]"))
        self._add_data(graph, GC.GRAPH_OUTPUTS, metadata.get("graph_outputs", "[]"))
        self._add_data(graph, GC.GRAPH_VALUE_INFO, metadata.get("value_info", "[]"))
        self._add_data(graph, GC.GRAPH_INITIALIZERS_REF, metadata.get("initializers_ref", '{}'))
    
    def _create_node_element(self, node: NodeData) -> ET.Element:
        """Create a node element."""
        node_elem = ET.Element("node", attrib={"id": node.id})
        
        # Add operation type (can be empty for operations)
        self._add_data(node_elem, GC.NODE_OP_TYPE, node.op_type if node.op_type else "")
        
        # Add hierarchy tag if present
        if node.hierarchy_tag:
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, node.hierarchy_tag)
        
        # Add node attributes as JSON (v1.3 format)
        node_attrs = {}
        if node.module_type:
            node_attrs["module_type"] = node.module_type
        if node.execution_order is not None:
            node_attrs["execution_order"] = node.execution_order
        # Add any ONNX-specific attributes
        if hasattr(node, 'attributes') and node.attributes:
            node_attrs.update(node.attributes)
        self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, json.dumps(node_attrs))
        
        # Add v1.3 fields for bidirectional conversion
        if hasattr(node, 'inputs') and node.inputs:
            self._add_data(node_elem, GC.NODE_INPUT_NAMES, json.dumps(node.inputs))
        if hasattr(node, 'outputs') and node.outputs:
            self._add_data(node_elem, GC.NODE_OUTPUT_NAMES, json.dumps(node.outputs))
        if hasattr(node, 'domain'):
            self._add_data(node_elem, GC.NODE_DOMAIN, getattr(node, 'domain', ''))
        
        # Add node name
        self._add_data(node_elem, GC.NODE_NAME, node.name)
        
        # Add shape information for inputs/outputs
        if node.node_type in (NodeType.INPUT, NodeType.OUTPUT) and "shape" in node.attributes:
            # Add as comment for visibility
            comment = ET.Comment(f" {node.name}: {node.attributes.get('shape', '')} ")
            node_elem.append(comment)
        
        return node_elem
    
    def _create_edge_element(self, edge: EdgeData) -> ET.Element:
        """Create an edge element."""
        edge_elem = ET.Element("edge", attrib={
            "source": edge.source_id,
            "target": edge.target_id
        })
        
        # Add tensor name
        self._add_data(edge_elem, GC.EDGE_TENSOR_NAME, edge.tensor_name)
        
        # Add v1.3 tensor information if available
        if hasattr(edge, 'tensor_dtype') and edge.tensor_dtype:
            self._add_data(edge_elem, GC.EDGE_TENSOR_TYPE, edge.tensor_dtype)
        if hasattr(edge, 'tensor_shape') and edge.tensor_shape:
            self._add_data(edge_elem, GC.EDGE_TENSOR_SHAPE, json.dumps(edge.tensor_shape))
        if hasattr(edge, 'tensor_data_ref'):
            self._add_data(edge_elem, GC.EDGE_TENSOR_DATA_REF, getattr(edge, 'tensor_data_ref', ''))
        
        return edge_elem
    
    def _add_data(self, parent: ET.Element, key: str, value: str) -> None:
        """Add a data element to a node or edge."""
        data = ET.SubElement(parent, "data", attrib={"key": key})
        data.text = value