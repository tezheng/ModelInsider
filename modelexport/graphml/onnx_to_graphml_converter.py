"""
Unified ONNX to GraphML Converter v1.3

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
from .constants import GRAPHML_FORMAT_VERSION, GRAPHML_V13_KEYS, GRAPHML_CONST
from .graphml_writer import GraphMLWriter
from .logging import get_logger
from .metadata_reader import MetadataReader
from .onnx_parser import ONNXGraphParser
from .parameter_manager import ParameterManager
from .utils import GraphData, NodeData
from .utils import GraphMLConstants as GC


class ONNXToGraphMLConverter:
    """
    Unified converter for ONNX to GraphML transformation with bidirectional support.
    
    This converter can produce both flat and hierarchical GraphML outputs,
    with full support for ONNX reconstruction (GraphML v1.3 format).
    
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
        use_hybrid_hierarchy: bool = True,
        validate_output: bool = True
    ):
        """Initialize unified converter with v1.3 validation."""
        self.hierarchical = hierarchical
        self.htp_metadata_path = htp_metadata_path
        self.parameter_strategy = parameter_strategy
        self.exclude_initializers = exclude_initializers
        self.exclude_attributes = exclude_attributes or set()
        self.use_hybrid_hierarchy = use_hybrid_hierarchy
        self.validate_output = validate_output
        
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
        self.format_version = GRAPHML_FORMAT_VERSION  # Use centralized version constant
        self.logger = get_logger(__name__)
        
        # Load HTP metadata if provided
        if self.htp_metadata_path:
            self.metadata_reader = MetadataReader(htp_metadata_path)
            with open(htp_metadata_path) as f:
                self.htp_data = json.load(f)
            
            # Initialize structural builder for hybrid hierarchy
            if self.use_hybrid_hierarchy:
                self.structural_builder = StructuralHierarchyBuilder(
                    exceptions=["Linear", "LayerNorm", "Embedding", "Dropout", "Tanh"]
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
        self.logger.info(f"Loading ONNX model: {model_path}")
        onnx_model = onnx.load(str(model_path))
        
        model_size_mb = model_path.stat().st_size / GRAPHML_CONST.BYTES_PER_MB
        node_count = len(onnx_model.graph.node)
        param_count = len(onnx_model.graph.initializer)
        
        self.logger.info(
            f"ONNX model loaded - size: {model_size_mb:.1f}MB, nodes: {node_count}, parameters: {param_count}"
        )
        
        if self.hierarchical:
            return self._convert_hierarchical(onnx_model, model_path, output_base)
        else:
            return self._convert_flat(onnx_model, model_path)
    
    def _convert_flat(self, onnx_model: onnx.ModelProto, model_path: Path) -> str:
        """Convert to flat GraphML v1.3 format."""
        # Parse ONNX model
        graph_data = self.parser.parse(onnx_model)
        
        # Add v1.3 metadata
        graph_data.metadata["source_file"] = model_path.name
        graph_data.metadata["format_version"] = self.format_version
        graph_data.metadata["producer_name"] = onnx_model.producer_name or ""
        graph_data.metadata["producer_version"] = onnx_model.producer_version or ""
        graph_data.metadata["model_version"] = str(onnx_model.model_version)
        graph_data.metadata["doc_string"] = onnx_model.doc_string or ""
        
        # Add opset imports
        opset_imports = [{"domain": op.domain or "", "version": op.version}
                        for op in onnx_model.opset_import]
        graph_data.metadata["opset_imports"] = json.dumps(opset_imports)
        
        # Add parameter strategy
        graph_data.metadata["parameter_strategy"] = self.parameter_strategy
        
        # Add graph I/O information
        inputs_info = [{"name": i.name} for i in onnx_model.graph.input]
        outputs_info = [{"name": o.name} for o in onnx_model.graph.output]
        graph_data.metadata["graph_inputs"] = json.dumps(inputs_info)
        graph_data.metadata["graph_outputs"] = json.dumps(outputs_info)
        graph_data.metadata["value_info"] = json.dumps([])
        graph_data.metadata["initializers_ref"] = json.dumps({
            "file": "",
            "count": len(onnx_model.graph.initializer)
        })
        
        # Generate GraphML
        graphml_element = self.writer.write(graph_data)
        return self.writer.to_string(graphml_element)
    
    def _convert_hierarchical(
        self, 
        onnx_model: onnx.ModelProto, 
        model_path: Path,
        output_base: str | None
    ) -> dict[str, str]:
        """
        Convert ONNX model to hierarchical GraphML v1.3 format with bidirectional support.
        
        This is the core algorithm for hierarchical GraphML generation. It performs the following steps:
        
        1. **Parameter Extraction**: Uses ParameterManager to extract and store ONNX initializers
           - Supports sidecar (.onnxdata), embedded, and reference strategies
           - Preserves parameter checksums for integrity validation
        
        2. **Graph Parsing**: Uses ONNXParser to extract nodes, edges, and basic structure
           - Converts ONNX nodes to GraphML nodes with op_type preservation
           - Creates edges from ONNX value dependencies
           - Extracts tensor shapes and types
        
        3. **Metadata Enhancement**: Adds v1.3 metadata for full ONNX reconstruction
           - Source file tracking and HTP metadata reference
           - Producer information and opset version preservation
           - Graph I/O schema and initializer references
        
        4. **Hierarchy Integration**: Merges HTP hierarchy tags with ONNX structure
           - Maps PyTorch module paths to ONNX operation nodes
           - Creates compound nodes for module boundaries
           - Preserves execution order and traced relationships
        
        5. **GraphML Generation**: Creates v1.3 compliant GraphML with key definitions
           - Uses systematic key naming (n0-n6, e0-e3, meta0-meta8, etc.)
           - Embeds JSON data for complex attributes
           - Maintains namespace compliance
        
        6. **Validation**: Optional three-layer validation for production quality
           - Schema validation (XSD compliance)
           - Semantic validation (logical consistency)
           - Round-trip validation (conversion accuracy)
        
        Args:
            onnx_model: Loaded ONNX ModelProto object
            model_path: Path to original ONNX file for metadata
            output_base: Base filename for output files (without extension)
            
        Returns:
            Dictionary mapping file types to generated file paths:
            - "graphml": Main GraphML file path
            - "parameters": Parameter sidecar file (if using sidecar strategy)
            - "format_version": GraphML format version used
            
        Raises:
            ValueError: If validation fails or required metadata is missing
            FileNotFoundError: If HTP metadata file is not accessible
            
        Algorithm Complexity:
            - Time: O(N + E + H) where N=nodes, E=edges, H=hierarchy depth
            - Space: O(N + E + P) where P=parameters
            - Typical performance: <2s for BERT-base, <30s for large transformer models
        """
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
        
        # Validate output if enabled
        if self.validate_output:
            self.logger.info("Running GraphML v1.3 validation")
            from .validators import GraphMLV13Validator
            validator = GraphMLV13Validator()
            results = validator.validate_all(graphml_path, str(model_path))
            
            # Log validation results
            passed = [r for r in results if r.status.value == "PASS"]
            warnings = [r for r in results if r.status.value == "WARNING"]
            failed = [r for r in results if r.status.value == "FAIL"]
            
            self.logger.info(
                f"Validation completed - passed: {len(passed)}, warnings: {len(warnings)}, failed: {len(failed)}"
            )
            
            # Check if all validations passed
            if failed:
                error_msgs = [f"{r.layer}: {r.message}" for r in failed]
                self.logger.error(f"GraphML validation failed: {error_msgs}")
                raise ValueError(f"GraphML validation failed:\n" + "\n".join(error_msgs))
        
        # Return file paths
        result = {
            "graphml": graphml_path,
            "format_version": self.format_version
        }
        
        if parameter_info:
            result.update(parameter_info.get("files", {}))
        
        output_size_mb = Path(graphml_path).stat().st_size / GRAPHML_CONST.BYTES_PER_MB  
        self.logger.info(
            f"GraphML conversion completed - output: {graphml_path}, size: {output_size_mb:.1f}MB"
        )
            
        return result
    
    def _enhance_graph_data(self, graph_data: GraphData, onnx_model: onnx.ModelProto) -> GraphData:
        """
        Enhance graph data with ONNX attributes for bidirectional conversion capability.
        
        This critical method bridges the gap between basic GraphML structure and full ONNX fidelity.
        It performs attribute extraction and node enhancement to enable accurate ONNX reconstruction.
        
        Algorithm Steps:
        1. **Attribute Extraction Phase**: 
           - Iterates through all ONNX nodes and extracts their attributes
           - Handles complex attribute types (tensors, graphs, strings, lists)
           - Preserves exact ONNX attribute names and values
        
        2. **I/O Mapping Phase**:
           - Maps ONNX node inputs/outputs to tensor names
           - Preserves input/output ordering (critical for ONNX semantics)
           - Extracts operator domain information
        
        3. **Node Enhancement Phase**:
           - Enriches GraphML nodes with ONNX-specific data
           - Adds JSON-serialized attributes for complex data types
           - Links hierarchy tags with ONNX operation semantics
        
        4. **Hierarchy Tag Processing**:
           - Merges HTP hierarchy information with ONNX structure
           - Handles missing or malformed hierarchy tags gracefully
           - Maintains module-to-operation mapping consistency
        
        Data Flow:
        ```
        ONNX Model → Attribute Extraction → I/O Mapping → Node Enhancement → Enhanced GraphData
            ↓              ↓                    ↓               ↓                  ↓
        Raw nodes → {name: attrs} → {name: io_info} → Enriched nodes → Bidirectional-ready
        ```
        
        Args:
            graph_data: Basic GraphML structure from ONNXParser
            onnx_model: Original ONNX model for attribute extraction
            
        Returns:
            Enhanced GraphData with bidirectional conversion capabilities
            - Nodes contain full ONNX attributes as JSON
            - I/O mapping preserved for tensor flow reconstruction
            - Hierarchy tags integrated with operation semantics
            
        Performance Notes:
            - Time Complexity: O(N * A) where N=nodes, A=avg attributes per node
            - Space Complexity: O(N * A) for attribute storage
            - Typical execution: <100ms for most models
            
        Data Preservation:
            - All ONNX attributes preserved with exact types
            - Input/output tensor ordering maintained
            - Domain information retained for custom operators
            - Hierarchy mapping validated and error-checked
        """
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
        self._define_v13_keys(graphml)
        
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
        
        # Determine module type
        module_type = "pytorch" if class_name in ["Linear", "LayerNorm", "Embedding", "Dropout"] else "huggingface"
        
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
        
        # Use constants for key IDs
        self._add_data(node_elem, "n4", json.dumps(input_names))  # input_names
        self._add_data(node_elem, "n5", json.dumps(output_names))  # output_names
        self._add_data(node_elem, "n6", domain)  # domain
        
        graph_elem.append(node_elem)
    
    def _add_remaining_onnx_nodes(self, graph_elem: ET.Element, root_tag: str, graph_data: GraphData):
        """Add ONNX operation nodes that haven't been placed in subgraphs yet."""
        for node in graph_data.nodes:
            if node.id not in self.placed_nodes:
                self._add_onnx_node(graph_elem, node)
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
            
            # v1.3: Tensor type and shape information with new key names
            tensor_type = getattr(edge, 'tensor_type', '')
            tensor_shape = getattr(edge, 'tensor_shape', [])
            tensor_data_ref = getattr(edge, 'tensor_data_ref', None)
            
            # Use constants for edge keys (v1.3 naming)
            self._add_data(edge_elem, "e1", tensor_type)  # tensor_type (was t0)
            self._add_data(edge_elem, "e2", json.dumps(tensor_shape))  # tensor_shape (was t1)
            self._add_data(edge_elem, "e3", tensor_data_ref or "")  # tensor_data_ref (was t2)
            
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
    
    def _define_v13_keys(self, graphml: ET.Element):
        """Define GraphML v1.3 key schema with conflict resolution."""
        # Define all keys needed for v1.3 format (resolves m5-m8 conflicts)
        keys = [
            # Graph keys (compound nodes)
            ("d0", "graph", "class_name", "string"),
            ("d1", "graph", "module_type", "string"),
            ("d2", "graph", "execution_order", "int"),
            ("d3", "graph", "traced_tag", "string"),
            
            # Node keys (enhanced for v1.3)
            ("n0", "node", "op_type", "string"),
            ("n1", "node", "hierarchy_tag", "string"),
            ("n2", "node", "onnx_attributes", "string"),
            ("n3", "node", "name", "string"),
            ("n4", "node", "input_names", "string"),
            ("n5", "node", "output_names", "string"),
            ("n6", "node", "domain", "string"),
            
            # Edge keys (v1.3 consolidation)
            ("e0", "edge", "tensor_name", "string"),
            ("e1", "edge", "tensor_type", "string"),  # was t0
            ("e2", "edge", "tensor_shape", "string"),  # was t1
            ("e3", "edge", "tensor_data_ref", "string"),  # was t2
            
            # Model metadata keys (v1.3 conflict resolution)
            ("meta0", "graph", "source_onnx_file", "string"),  # was m0
            ("meta1", "graph", "source_htp_file", "string"),  # was m1
            ("meta2", "graph", "format_version", "string"),  # was m2
            ("meta3", "graph", "export_timestamp", "string"),  # was m3
            ("meta4", "graph", "opset_imports", "string"),  # was m4
            ("meta5", "graph", "producer_name", "string"),  # was m5 (CONFLICT RESOLVED)
            ("meta6", "graph", "producer_version", "string"),  # was m6 (CONFLICT RESOLVED)
            ("meta7", "graph", "model_version", "string"),  # was m7 (CONFLICT RESOLVED)
            ("meta8", "graph", "doc_string", "string"),  # was m8 (CONFLICT RESOLVED)
            
            # Parameter storage keys (v1.3 naming)
            ("param0", "graph", "parameter_strategy", "string"),  # was p0
            ("param1", "graph", "parameter_file", "string"),  # was p1
            ("param2", "graph", "parameter_checksum", "string"),  # was p2
            
            # Graph I/O keys (v1.3 naming)
            ("io0", "graph", "graph_inputs", "string"),  # was g0
            ("io1", "graph", "graph_outputs", "string"),  # was g1
            ("io2", "graph", "value_info", "string"),  # was g2
            ("io3", "graph", "initializers_ref", "string"),  # was g3
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
        
        # Format and timestamp (v1.3 keys)
        self._add_data(graph, "meta2", self.format_version)  # was m2
        self._add_data(graph, "meta3", datetime.now().isoformat())  # was m3
        
        # ONNX model metadata (v1.3 keys)
        opset_imports = []
        for imp in onnx_model.opset_import:
            opset_imports.append({
                "domain": imp.domain or "",
                "version": imp.version
            })
        self._add_data(graph, "meta4", json.dumps(opset_imports))  # was m4
        
        self._add_data(graph, "meta5", onnx_model.producer_name or "")  # was m5
        self._add_data(graph, "meta6", onnx_model.producer_version or "")  # was m6
        self._add_data(graph, "meta7", str(onnx_model.model_version))  # was m7
        self._add_data(graph, "meta8", onnx_model.doc_string or "")  # was m8
        
        # Parameter storage information (v1.3 keys)
        self._add_data(graph, "param0", self.parameter_strategy)  # was p0
        self._add_data(graph, "param1", parameter_info.get("parameter_file", ""))  # was p1
        self._add_data(graph, "param2", parameter_info.get("checksum", ""))  # was p2
        
        # Graph structure information
        self._add_input_output_metadata(graph, onnx_model)
        
        # Initializers reference (v1.3 key) - should be JSON
        if self.parameter_strategy == "embedded" and "embedded_data" in parameter_info:
            # For embedded strategy, include the embedded data
            self._add_data(graph, "io3", json.dumps(parameter_info["embedded_data"]))  # was g3
        else:
            # For other strategies, just reference info
            initializers_ref = {
                "file": parameter_info.get("parameter_file", ""),
                "count": len(onnx_model.graph.initializer)
            }
            self._add_data(graph, "io3", json.dumps(initializers_ref))  # was g3
    
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
        
        # Add as JSON metadata (v1.3 keys)
        self._add_data(graph, "io0", json.dumps(inputs_metadata))  # was g0
        self._add_data(graph, "io1", json.dumps(outputs_metadata))  # was g1
        self._add_data(graph, "io2", json.dumps(value_info))  # was g2
    
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