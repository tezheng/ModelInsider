"""
ONNX Graph Parser

This module extracts graph structure from ONNX models, focusing on the
computational graph while optionally excluding parameter tensors.
"""

from typing import Any

import onnx

from .utils import (
    EdgeData,
    GraphData,
    NodeData,
    NodeType,
    format_tensor_shape,
    get_tensor_dtype_name,
    sanitize_node_id,
)


class ONNXGraphParser:
    """
    Parser for extracting graph structure from ONNX models.
    
    This parser focuses on the computational graph, extracting nodes
    (operations) and edges (tensor connections) while optionally
    excluding initializers (parameters/weights).
    """
    
    def __init__(
        self,
        exclude_initializers: bool = True,
        exclude_attributes: set[str] | None = None
    ):
        self.exclude_initializers = exclude_initializers
        self.exclude_attributes = exclude_attributes or set()
        
        # Statistics
        self.last_node_count = 0
        self.last_edge_count = 0
        self.last_initializer_count = 0
    
    def parse(self, onnx_model: onnx.ModelProto) -> GraphData:
        """
        Parse ONNX model into internal graph representation.
        
        Args:
            onnx_model: Loaded ONNX model
            
        Returns:
            GraphData structure with nodes, edges, and metadata
        """
        graph = onnx_model.graph
        graph_data = GraphData()
        
        # Extract metadata
        graph_data.metadata.update(self._extract_metadata(onnx_model))
        
        # Build initializer set for quick lookup
        initializer_names = {init.name for init in graph.initializer}
        self.last_initializer_count = len(initializer_names)
        
        # Extract nodes
        node_map = {}
        for idx, node in enumerate(graph.node):
            node_data = self._extract_node(node, idx)
            graph_data.nodes.append(node_data)
            node_map[node_data.name] = node_data
        
        # Extract inputs (excluding initializers)
        for input_proto in graph.input:
            if input_proto.name not in initializer_names:
                input_node = self._extract_input(input_proto)
                graph_data.inputs.append(input_node)
                node_map[input_node.name] = input_node
        
        # Extract outputs
        for output_proto in graph.output:
            output_node = self._extract_output(output_proto)
            graph_data.outputs.append(output_node)
            node_map[output_node.name] = output_node
        
        # Extract edges
        graph_data.edges = self._extract_edges(graph, node_map, initializer_names)
        
        # Update statistics
        self.last_node_count = len(graph_data.nodes)
        self.last_edge_count = len(graph_data.edges)
        
        return graph_data
    
    def _extract_metadata(self, model: onnx.ModelProto) -> dict[str, Any]:
        """Extract model metadata."""
        metadata = {
            "model_version": model.model_version if hasattr(model, 'model_version') else 0,
            "producer_name": model.producer_name if model.producer_name else "",
            "producer_version": model.producer_version if model.producer_version else "",
            "domain": model.domain if model.domain else "",
            "graph_name": model.graph.name if model.graph.name else "main_graph",
            "doc_string": model.doc_string if model.doc_string else ""
        }
        
        # Extract opset versions
        opset_imports = []
        for opset in model.opset_import:
            opset_imports.append({
                "domain": opset.domain if opset.domain else "ai.onnx",
                "version": opset.version
            })
        metadata["opset_imports"] = opset_imports
        
        return metadata
    
    def _extract_node(self, node: Any, index: int) -> NodeData:
        """Extract data from an ONNX node."""
        # Use original node name as ID (with forward slashes) to match baseline
        node_id = node.name if node.name else f"node_{index}"
        node_name = node.name if node.name else f"node_{index}"
        
        # Extract attributes (filtering excluded ones)
        attributes = {}
        if node.attribute:
            for attr in node.attribute:
                if attr.name not in self.exclude_attributes:
                    attributes[attr.name] = self._extract_attribute_value(attr)
        
        # Extract special attributes if present
        hierarchy_tag = None
        if 'hierarchy_tag' in attributes:
            hierarchy_tag = attributes.pop('hierarchy_tag')
        
        module_type = None
        if 'module_type' in attributes:
            module_type = attributes.pop('module_type')
        
        execution_order = None
        if 'execution_order' in attributes:
            execution_order = attributes.pop('execution_order')
        
        return NodeData(
            id=node_id,
            name=node_name,
            op_type=node.op_type,
            node_type=NodeType.OPERATION,
            inputs=list(node.input),
            outputs=list(node.output),
            attributes=attributes,
            hierarchy_tag=hierarchy_tag,
            module_type=module_type,
            execution_order=execution_order
        )
    
    def _extract_input(self, input_proto: Any) -> NodeData:
        """Extract data from a graph input."""
        input_id = f"input_{sanitize_node_id(input_proto.name)}"
        
        # Extract type information
        type_info = {}
        if input_proto.type and input_proto.type.tensor_type:
            tensor_type = input_proto.type.tensor_type
            type_info["dtype"] = get_tensor_dtype_name(tensor_type.elem_type)
            if tensor_type.shape:
                type_info["shape"] = format_tensor_shape(
                    list(tensor_type.shape.dim)
                )
        
        return NodeData(
            id=input_id,
            name=input_proto.name,
            op_type="Input",
            node_type=NodeType.INPUT,
            outputs=[input_proto.name],
            attributes=type_info
        )
    
    def _extract_output(self, output_proto: Any) -> NodeData:
        """Extract data from a graph output."""
        output_id = f"output_{sanitize_node_id(output_proto.name)}"
        
        # Extract type information
        type_info = {}
        if output_proto.type and output_proto.type.tensor_type:
            tensor_type = output_proto.type.tensor_type
            type_info["dtype"] = get_tensor_dtype_name(tensor_type.elem_type)
            if tensor_type.shape:
                type_info["shape"] = format_tensor_shape(
                    list(tensor_type.shape.dim)
                )
        
        return NodeData(
            id=output_id,
            name=output_proto.name,
            op_type="Output",
            node_type=NodeType.OUTPUT,
            inputs=[output_proto.name],
            attributes=type_info
        )
    
    def _extract_edges(
        self,
        graph: Any,
        node_map: dict[str, NodeData],
        initializer_names: set[str]
    ) -> list[EdgeData]:
        """Extract edges (tensor connections) from the graph."""
        edges = []
        
        # Build tensor producer map
        tensor_producers = {}
        
        # Graph inputs produce tensors
        for input_proto in graph.input:
            if input_proto.name not in initializer_names:
                input_id = f"input_{sanitize_node_id(input_proto.name)}"
                tensor_producers[input_proto.name] = input_id
        
        # Node outputs produce tensors
        for idx, node in enumerate(graph.node):
            # Use original node name as ID to match baseline
            node_id = node.name if node.name else f"node_{idx}"
            for output in node.output:
                tensor_producers[output] = node_id
        
        # Create edges from producers to consumers
        for idx, node in enumerate(graph.node):
            # Use original node name as ID to match baseline
            node_id = node.name if node.name else f"node_{idx}"
            
            for input_tensor in node.input:
                # Skip initializers if requested
                if self.exclude_initializers and input_tensor in initializer_names:
                    continue
                
                # Find producer
                producer_id = tensor_producers.get(input_tensor)
                if producer_id:
                    edge = EdgeData(
                        source_id=producer_id,
                        target_id=node_id,
                        tensor_name=input_tensor
                    )
                    edges.append(edge)
        
        # Connect to output nodes
        for output_proto in graph.output:
            output_id = f"output_{sanitize_node_id(output_proto.name)}"
            producer_id = tensor_producers.get(output_proto.name)
            if producer_id:
                edge = EdgeData(
                    source_id=producer_id,
                    target_id=output_id,
                    tensor_name=output_proto.name
                )
                edges.append(edge)
        
        return edges
    
    def _extract_attribute_value(self, attr: Any) -> Any:
        """Extract value from ONNX attribute."""
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8') if attr.s else ""
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return f"<type_{attr.type}>"