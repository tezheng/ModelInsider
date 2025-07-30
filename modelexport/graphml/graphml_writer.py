"""
GraphML XML Writer

This module generates GraphML XML from internal graph data structures,
following the GraphML specification for compatibility with visualization tools.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

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
        """Define GraphML attribute keys."""
        # Graph attributes (for compound nodes)
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
        
        # Graph metadata
        self._add_key(graphml, GC.META_SOURCE_ONNX, "graph", "source_onnx_text", "string")
        self._add_key(graphml, GC.META_SOURCE_HTP, "graph", "source_htp", "string")
        self._add_key(graphml, GC.META_FORMAT_VERSION, "graph", "format_version", "string")
        self._add_key(graphml, GC.META_TIMESTAMP, "graph", "export_timestamp", "string")
    
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
        """Add metadata to graph element."""
        metadata = graph_data.metadata
        
        # Source file
        if "source_file" in metadata:
            self._add_data(graph, GC.META_SOURCE_ONNX, metadata["source_file"])
        
        # HTP file (if present)
        if "htp_file" in metadata:
            self._add_data(graph, GC.META_SOURCE_HTP, metadata["htp_file"])
        
        # Format version
        self._add_data(graph, GC.META_FORMAT_VERSION, "1.1")
        
        # Timestamp
        self._add_data(graph, GC.META_TIMESTAMP, datetime.now().isoformat())
    
    def _create_node_element(self, node: NodeData) -> ET.Element:
        """Create a node element."""
        node_elem = ET.Element("node", attrib={"id": node.id})
        
        # Add operation type (can be empty for operations)
        self._add_data(node_elem, GC.NODE_OP_TYPE, node.op_type if node.op_type else "")
        
        # Add hierarchy tag if present
        if node.hierarchy_tag:
            self._add_data(node_elem, GC.NODE_HIERARCHY_TAG, node.hierarchy_tag)
        
        # Add node attributes as JSON
        import json
        node_attrs = {}
        if node.module_type:
            node_attrs["module_type"] = node.module_type
        if node.execution_order is not None:
            node_attrs["execution_order"] = node.execution_order
        self._add_data(node_elem, GC.NODE_ATTRIBUTES_JSON, json.dumps(node_attrs))
        
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
        
        return edge_elem
    
    def _add_data(self, parent: ET.Element, key: str, value: str) -> None:
        """Add a data element to a node or edge."""
        data = ET.SubElement(parent, "data", attrib={"key": key})
        data.text = value