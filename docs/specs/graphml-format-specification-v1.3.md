# GraphML Format Specification v1.3 (Schema-Driven Engineering)

**Status**: 🚀 **SCHEMA-DRIVEN SPECIFICATION**  
**Format Version**: 1.3  
**Specification Date**: 2025-08-04  
**Engineering Approach**: TEZ-135 Six Pillars Implementation  
**Supersedes**: v1.1, v1.1.1, v1.2 (eliminates all conflicts)

## Executive Summary

GraphML v1.3 is the definitive specification implementing **schema-driven engineering** that eliminates the chaos of conflicting specifications. This specification provides:

1. **Formal Schema Definition**: XSD/YAML schemas that ARE the specification
2. **Conflict Resolution**: Eliminates m5-m8 key conflicts definitively
3. **100% Round-Trip Target**: Engineering for perfect fidelity
4. **Performance Constraints**: <2s export, <500MB memory
5. **Governance Model**: Change management and version control
6. **Quality Gates**: Multi-layer validation with automated migration

## The Six Pillars of Schema-Driven Engineering

### Pillar 1: Schema Engine Architecture
The schema IS the specification. All validation, conversion, and tooling derives from formal schema definitions.

### Pillar 2: Multi-Layer Validation System
Three-tier validation: Schema Compliance → Semantic Consistency → Round-Trip Accuracy

### Pillar 3: Unified Converter Architecture
Single converter architecture with schema-driven field mapping and validation

### Pillar 4: Automated Migration System
Automatic migration between versions with validation and rollback capability

### Pillar 5: Governance Framework
Formal change management with version control and backward compatibility guarantees

### Pillar 6: Prevention & Quality Gates
Proactive validation preventing specification conflicts and implementation drift

---

## I. Formal Schema Definition

### 1.1 XSD Schema (Primary Definition)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="http://modelexport.ai/graphml/v1.3"
           xmlns:gml="http://modelexport.ai/graphml/v1.3"
           elementFormDefault="qualified">

  <!-- Root GraphML Element -->
  <xs:element name="graphml" type="gml:GraphMLType"/>
  
  <xs:complexType name="GraphMLType">
    <xs:sequence>
      <!-- Key Definitions (Required) -->
      <xs:element name="key" type="gml:KeyType" minOccurs="19" maxOccurs="unbounded"/>
      <!-- Main Graph (Required) -->
      <xs:element name="graph" type="gml:GraphType"/>
    </xs:sequence>
    <xs:attribute name="xmlns" type="xs:anyURI" fixed="http://graphml.graphdrawing.org/xmlns"/>
    <xs:attribute name="version" type="xs:string" fixed="1.3"/>
  </xs:complexType>

  <!-- Key Definition Type -->
  <xs:complexType name="KeyType">
    <xs:sequence>
      <xs:element name="desc" type="xs:string" minOccurs="0"/>
    </xs:sequence>
    <xs:attribute name="id" type="gml:KeyIdType" use="required"/>
    <xs:attribute name="for" type="gml:KeyForType" use="required"/>
    <xs:attribute name="attr.name" type="xs:string" use="required"/>
    <xs:attribute name="attr.type" type="gml:AttributeTypeType" use="required"/>
  </xs:complexType>

  <!-- Valid Key IDs (Eliminates Conflicts) -->
  <xs:simpleType name="KeyIdType">
    <xs:restriction base="xs:string">
      <!-- Graph Keys (Compound Nodes) -->
      <xs:enumeration value="d0"/><!-- class_name -->
      <xs:enumeration value="d1"/><!-- module_type -->
      <xs:enumeration value="d2"/><!-- execution_order -->
      <xs:enumeration value="d3"/><!-- traced_tag -->
      
      <!-- Node Keys -->
      <xs:enumeration value="n0"/><!-- op_type -->
      <xs:enumeration value="n1"/><!-- hierarchy_tag -->
      <xs:enumeration value="n2"/><!-- onnx_attributes -->
      <xs:enumeration value="n3"/><!-- name -->
      <xs:enumeration value="n4"/><!-- input_names -->
      <xs:enumeration value="n5"/><!-- output_names -->
      <xs:enumeration value="n6"/><!-- domain -->
      
      <!-- Edge Keys -->
      <xs:enumeration value="e0"/><!-- tensor_name -->
      <xs:enumeration value="e1"/><!-- tensor_type -->
      <xs:enumeration value="e2"/><!-- tensor_shape -->
      <xs:enumeration value="e3"/><!-- tensor_data_ref -->
      
      <!-- Model Metadata Keys (CONFLICT RESOLVED) -->
      <xs:enumeration value="meta0"/><!-- source_onnx_file -->
      <xs:enumeration value="meta1"/><!-- source_htp_file -->
      <xs:enumeration value="meta2"/><!-- format_version -->
      <xs:enumeration value="meta3"/><!-- export_timestamp -->
      <xs:enumeration value="meta4"/><!-- opset_imports -->
      <xs:enumeration value="meta5"/><!-- producer_name -->
      <xs:enumeration value="meta6"/><!-- producer_version -->
      <xs:enumeration value="meta7"/><!-- model_version -->
      <xs:enumeration value="meta8"/><!-- doc_string -->
      
      <!-- Parameter Storage Keys -->
      <xs:enumeration value="param0"/><!-- parameter_strategy -->
      <xs:enumeration value="param1"/><!-- parameter_file -->
      <xs:enumeration value="param2"/><!-- parameter_checksum -->
      
      <!-- Graph I/O Keys -->
      <xs:enumeration value="io0"/><!-- graph_inputs -->
      <xs:enumeration value="io1"/><!-- graph_outputs -->
      <xs:enumeration value="io2"/><!-- value_info -->
      <xs:enumeration value="io3"/><!-- initializers_ref -->
    </xs:restriction>
  </xs:simpleType>

  <!-- Key Target Types -->
  <xs:simpleType name="KeyForType">
    <xs:restriction base="xs:string">
      <xs:enumeration value="node"/>
      <xs:enumeration value="edge"/>
      <xs:enumeration value="graph"/>
    </xs:restriction>
  </xs:simpleType>

  <!-- Attribute Data Types -->
  <xs:simpleType name="AttributeTypeType">
    <xs:restriction base="xs:string">
      <xs:enumeration value="string"/>
      <xs:enumeration value="int"/>
      <xs:enumeration value="long"/>
      <xs:enumeration value="float"/>
      <xs:enumeration value="double"/>
      <xs:enumeration value="boolean"/>
    </xs:restriction>
  </xs:simpleType>

  <!-- Graph Type -->
  <xs:complexType name="GraphType">
    <xs:sequence>
      <!-- Graph Metadata -->
      <xs:element name="data" type="gml:DataType" minOccurs="3" maxOccurs="unbounded"/>
      <!-- Nodes and Edges -->
      <xs:choice minOccurs="0" maxOccurs="unbounded">
        <xs:element name="node" type="gml:NodeType"/>
        <xs:element name="edge" type="gml:EdgeType"/>
      </xs:choice>
    </xs:sequence>
    <xs:attribute name="id" type="xs:string" use="required"/>
    <xs:attribute name="edgedefault" type="xs:string" fixed="directed"/>
  </xs:complexType>

  <!-- Node Type -->
  <xs:complexType name="NodeType">
    <xs:sequence>
      <xs:element name="data" type="gml:DataType" minOccurs="4" maxOccurs="7"/>
      <xs:element name="graph" type="gml:GraphType" minOccurs="0"/><!-- For compound nodes -->
    </xs:sequence>
    <xs:attribute name="id" type="xs:string" use="required"/>
  </xs:complexType>

  <!-- Edge Type -->
  <xs:complexType name="EdgeType">
    <xs:sequence>
      <xs:element name="data" type="gml:DataType" minOccurs="1" maxOccurs="4"/>
    </xs:sequence>
    <xs:attribute name="source" type="xs:string" use="required"/>
    <xs:attribute name="target" type="xs:string" use="required"/>
  </xs:complexType>

  <!-- Data Element Type -->
  <xs:complexType name="DataType">
    <xs:simpleContent>
      <xs:extension base="xs:string">
        <xs:attribute name="key" type="gml:KeyIdType" use="required"/>
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>

</xs:schema>
```

### 1.2 YAML Schema (Alternative Definition)

```yaml
$schema: "http://json-schema.org/draft-07/schema#"
$id: "http://modelexport.ai/graphml/v1.3/schema"
title: "GraphML v1.3 Schema"
description: "Schema-driven GraphML specification for ONNX model export"

type: object
required: [graphml]
properties:
  graphml:
    type: object
    required: [xmlns, version, keys, graph]
    properties:
      xmlns:
        const: "http://graphml.graphdrawing.org/xmlns"
      version:
        const: "1.3"
      keys:
        type: array
        minItems: 19
        items:
          $ref: "#/definitions/Key"
      graph:
        $ref: "#/definitions/Graph"

definitions:
  Key:
    type: object
    required: [id, for, attr_name, attr_type]
    properties:
      id:
        enum: [
          # Graph keys
          "d0", "d1", "d2", "d3",
          # Node keys  
          "n0", "n1", "n2", "n3", "n4", "n5", "n6",
          # Edge keys
          "e0", "e1", "e2", "e3",
          # Metadata keys (CONFLICT RESOLVED)
          "meta0", "meta1", "meta2", "meta3", "meta4", "meta5", "meta6", "meta7", "meta8",
          # Parameter keys
          "param0", "param1", "param2",
          # I/O keys
          "io0", "io1", "io2", "io3"
        ]
      for:
        enum: ["node", "edge", "graph"]
      attr_name:
        type: string
      attr_type:
        enum: ["string", "int", "long", "float", "double", "boolean"]
      desc:
        type: string

  Graph:
    type: object
    required: [id, edgedefault, metadata, nodes]
    properties:
      id:
        type: string
      edgedefault:
        const: "directed"
      metadata:
        type: object
        required: [format_version, export_timestamp, parameter_strategy]
        additionalProperties: true
      nodes:
        type: array
        items:
          $ref: "#/definitions/Node"
      edges:
        type: array
        items:
          $ref: "#/definitions/Edge"

  Node:
    type: object
    required: [id, op_type, hierarchy_tag, onnx_attributes, name]
    properties:
      id:
        type: string
      op_type:
        type: string
      hierarchy_tag:
        type: string
      onnx_attributes:
        type: string  # JSON string
      name:
        type: string
      input_names:
        type: string  # JSON array
      output_names:
        type: string  # JSON array
      domain:
        type: string
      compound_graph:
        $ref: "#/definitions/Graph"

  Edge:
    type: object
    required: [source, target, tensor_name]
    properties:
      source:
        type: string
      target:
        type: string
      tensor_name:
        type: string
      tensor_type:
        type: string
      tensor_shape:
        type: string  # JSON array
      tensor_data_ref:
        type: string

# Validation Rules (Semantic Consistency)
validation_rules:
  node_connectivity:
    description: "All edge sources and targets must reference valid nodes"
    rule: "edge.source IN nodes.id AND edge.target IN nodes.id"
    
  json_fields:
    description: "JSON string fields must be valid JSON"
    rule: "JSON.parse(node.onnx_attributes) AND JSON.parse(node.input_names) AND JSON.parse(node.output_names)"
    
  compound_nodes:
    description: "Compound nodes must have nested graphs"
    rule: "IF node.op_type STARTS_WITH 'module_' THEN node.compound_graph IS_REQUIRED"
    
  parameter_strategy:
    description: "Parameter strategy must be consistent"
    rule: "IF metadata.parameter_strategy = 'sidecar' THEN metadata.parameter_file IS_REQUIRED"
    
  round_trip_fidelity:
    description: "ONNX reconstruction fidelity targets"
    rule: "node_count_preservation >= 0.85 AND parameter_preservation = 1.0"
```

---

## II. Key Schema Changes (v1.3)

### 2.1 Conflict Resolution

**ELIMINATED**: m5-m8 key conflicts  
**SOLUTION**: Renamed metadata keys with clear prefixes

| Old Key | New Key | Purpose | Conflict Resolved |
|---------|---------|---------|-------------------|
| m5 | meta5 | producer_name | ✅ Different meaning in v1.1 vs v1.2 |
| m6 | meta6 | producer_version | ✅ Different meaning in v1.1 vs v1.2 |
| m7 | meta7 | model_version | ✅ Different meaning in v1.1 vs v1.2 |
| m8 | meta8 | doc_string | ✅ Different meaning in v1.1 vs v1.2 |
| t0 | e1 | tensor_type | ✅ Logical grouping with edge keys |
| t1 | e2 | tensor_shape | ✅ Logical grouping with edge keys |
| t2 | e3 | tensor_data_ref | ✅ Logical grouping with edge keys |
| p0 | param0 | parameter_strategy | ✅ Clear parameter namespace |
| p1 | param1 | parameter_file | ✅ Clear parameter namespace |
| p2 | param2 | parameter_checksum | ✅ Clear parameter namespace |
| g0 | io0 | graph_inputs | ✅ Clear I/O namespace |
| g1 | io1 | graph_outputs | ✅ Clear I/O namespace |
| g2 | io2 | value_info | ✅ Clear I/O namespace |
| g3 | io3 | initializers_ref | ✅ Clear I/O namespace |

### 2.2 Schema-Driven Field Definitions

**Complete Key Specification** (19 required keys):

```xml
<!-- Graph Keys (Compound Nodes) -->
<key id="d0" for="graph" attr.name="class_name" attr.type="string"/>
<key id="d1" for="graph" attr.name="module_type" attr.type="string"/>
<key id="d2" for="graph" attr.name="execution_order" attr.type="int"/>
<key id="d3" for="graph" attr.name="traced_tag" attr.type="string"/>

<!-- Node Keys -->
<key id="n0" for="node" attr.name="op_type" attr.type="string"/>
<key id="n1" for="node" attr.name="hierarchy_tag" attr.type="string"/>
<key id="n2" for="node" attr.name="onnx_attributes" attr.type="string"/>
<key id="n3" for="node" attr.name="name" attr.type="string"/>
<key id="n4" for="node" attr.name="input_names" attr.type="string"/>
<key id="n5" for="node" attr.name="output_names" attr.type="string"/>
<key id="n6" for="node" attr.name="domain" attr.type="string"/>

<!-- Edge Keys -->
<key id="e0" for="edge" attr.name="tensor_name" attr.type="string"/>
<key id="e1" for="edge" attr.name="tensor_type" attr.type="string"/>
<key id="e2" for="edge" attr.name="tensor_shape" attr.type="string"/>
<key id="e3" for="edge" attr.name="tensor_data_ref" attr.type="string"/>

<!-- Model Metadata Keys -->
<key id="meta0" for="graph" attr.name="source_onnx_file" attr.type="string"/>
<key id="meta1" for="graph" attr.name="source_htp_file" attr.type="string"/>
<key id="meta2" for="graph" attr.name="format_version" attr.type="string"/>
<key id="meta3" for="graph" attr.name="export_timestamp" attr.type="string"/>
<key id="meta4" for="graph" attr.name="opset_imports" attr.type="string"/>
<key id="meta5" for="graph" attr.name="producer_name" attr.type="string"/>
<key id="meta6" for="graph" attr.name="producer_version" attr.type="string"/>
<key id="meta7" for="graph" attr.name="model_version" attr.type="string"/>
<key id="meta8" for="graph" attr.name="doc_string" attr.type="string"/>

<!-- Parameter Storage Keys -->
<key id="param0" for="graph" attr.name="parameter_strategy" attr.type="string"/>
<key id="param1" for="graph" attr.name="parameter_file" attr.type="string"/>
<key id="param2" for="graph" attr.name="parameter_checksum" attr.type="string"/>

<!-- Graph I/O Keys -->
<key id="io0" for="graph" attr.name="graph_inputs" attr.type="string"/>
<key id="io1" for="graph" attr.name="graph_outputs" attr.type="string"/>
<key id="io2" for="graph" attr.name="value_info" attr.type="string"/>
<key id="io3" for="graph" attr.name="initializers_ref" attr.type="string"/>
```

---

## III. Multi-Layer Validation System

### 3.1 Layer 1: Schema Compliance Validation

**XSD Validation**:
```python
def validate_schema_compliance(graphml_file: str) -> ValidationResult:
    """Layer 1: Validate against XSD schema."""
    import xmlschema
    
    schema = xmlschema.XMLSchema('/schemas/graphml-v1.3.xsd')
    try:
        schema.validate(graphml_file)
        return ValidationResult(
            layer="Schema",
            status="PASS",
            message="Valid GraphML v1.3 structure"
        )
    except xmlschema.XMLSchemaException as e:
        return ValidationResult(
            layer="Schema", 
            status="FAIL",
            message=f"Schema violation: {e}",
            error_code="SCHEMA_001"
        )
```

**Key Validation Rules**:
- Exactly 19 required keys must be present
- Key IDs must match enumerated values (no m5-m8 allowed)
- Key types must match attribute usage
- Graph must have required metadata keys

### 3.2 Layer 2: Semantic Consistency Validation

**Connectivity Validation**:
```python
def validate_semantic_consistency(graphml_data: dict) -> ValidationResult:
    """Layer 2: Validate semantic consistency."""
    errors = []
    
    # Rule 1: Edge connectivity
    node_ids = {node['id'] for node in graphml_data['nodes']}
    for edge in graphml_data['edges']:
        if edge['source'] not in node_ids:
            errors.append(f"Edge source '{edge['source']}' not found")
        if edge['target'] not in node_ids:
            errors.append(f"Edge target '{edge['target']}' not found")
    
    # Rule 2: JSON field validation
    for node in graphml_data['nodes']:
        try:
            json.loads(node.get('onnx_attributes', '{}'))
            json.loads(node.get('input_names', '[]'))
            json.loads(node.get('output_names', '[]'))
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in node {node['id']}: {e}")
    
    # Rule 3: Parameter strategy consistency
    metadata = graphml_data['metadata']
    if metadata.get('parameter_strategy') == 'sidecar':
        if not metadata.get('parameter_file'):
            errors.append("Sidecar strategy requires parameter_file")
    
    if errors:
        return ValidationResult(
            layer="Semantic",
            status="FAIL", 
            message=f"Consistency violations: {'; '.join(errors)}",
            error_code="SEMANTIC_001"
        )
    
    return ValidationResult(
        layer="Semantic",
        status="PASS",
        message="Semantic consistency validated"
    )
```

### 3.3 Layer 3: Round-Trip Accuracy Validation

**Fidelity Targets**:
- **Node Count Preservation**: ≥85% (compound node filtering acceptable)
- **Parameter Preservation**: 100% (bit-exact)
- **Attribute Preservation**: 100% (ONNX operators)
- **Topology Preservation**: ≥90% (edge connectivity)

```python
def validate_round_trip_accuracy(
    original_onnx: str, 
    graphml_file: str,
    temp_dir: str
) -> ValidationResult:
    """Layer 3: Validate round-trip conversion accuracy."""
    
    # Step 1: Convert GraphML back to ONNX
    reconstructed_onnx = f"{temp_dir}/reconstructed.onnx"
    converter = GraphMLToONNXConverter()
    result = converter.convert(graphml_file, reconstructed_onnx)
    
    # Step 2: Load models for comparison
    original = onnx.load(original_onnx)
    reconstructed = onnx.load(reconstructed_onnx)
    
    # Step 3: Calculate preservation metrics
    metrics = {
        'node_count_original': len(original.graph.node),
        'node_count_reconstructed': len(reconstructed.graph.node),
        'parameter_count_original': len(original.graph.initializer),
        'parameter_count_reconstructed': len(reconstructed.graph.initializer),
    }
    
    node_preservation = metrics['node_count_reconstructed'] / metrics['node_count_original']
    param_preservation = metrics['parameter_count_reconstructed'] / metrics['parameter_count_original']
    
    # Step 4: Validate against targets
    if node_preservation < 0.85:
        return ValidationResult(
            layer="RoundTrip",
            status="FAIL",
            message=f"Node preservation {node_preservation:.2%} below 85% target",
            error_code="ROUNDTRIP_001",
            metrics=metrics
        )
    
    if param_preservation < 1.0:
        return ValidationResult(
            layer="RoundTrip", 
            status="FAIL",
            message=f"Parameter preservation {param_preservation:.2%} below 100% target",
            error_code="ROUNDTRIP_002",
            metrics=metrics
        )
    
    return ValidationResult(
        layer="RoundTrip",
        status="PASS", 
        message=f"Round-trip validated: {node_preservation:.2%} nodes, {param_preservation:.2%} params",
        metrics=metrics
    )
```

---

## IV. Unified Converter Architecture

### 4.1 Schema-Driven Converter

```python
class GraphMLV13Converter:
    """Schema-driven GraphML v1.3 converter."""
    
    def __init__(self, schema_path: str = None):
        self.schema = self._load_schema(schema_path)
        self.key_mappings = self._build_key_mappings()
        self.validators = [
            SchemaComplianceValidator(self.schema),
            SemanticConsistencyValidator(),
            RoundTripAccuracyValidator()
        ]
    
    def _build_key_mappings(self) -> Dict[str, KeyDefinition]:
        """Build key mappings from schema definition."""
        return {
            # Graph keys
            'd0': KeyDefinition('graph', 'class_name', 'string', required=True),
            'd1': KeyDefinition('graph', 'module_type', 'string', required=True),
            'd2': KeyDefinition('graph', 'execution_order', 'int', required=True),
            'd3': KeyDefinition('graph', 'traced_tag', 'string', required=True),
            
            # Node keys
            'n0': KeyDefinition('node', 'op_type', 'string', required=True),
            'n1': KeyDefinition('node', 'hierarchy_tag', 'string', required=True),
            'n2': KeyDefinition('node', 'onnx_attributes', 'string', required=True),
            'n3': KeyDefinition('node', 'name', 'string', required=True),
            'n4': KeyDefinition('node', 'input_names', 'string', required=True),
            'n5': KeyDefinition('node', 'output_names', 'string', required=True),
            'n6': KeyDefinition('node', 'domain', 'string', required=False),
            
            # Edge keys
            'e0': KeyDefinition('edge', 'tensor_name', 'string', required=True),
            'e1': KeyDefinition('edge', 'tensor_type', 'string', required=True),
            'e2': KeyDefinition('edge', 'tensor_shape', 'string', required=False),
            'e3': KeyDefinition('edge', 'tensor_data_ref', 'string', required=False),
            
            # Metadata keys (CONFLICT RESOLVED)
            'meta0': KeyDefinition('graph', 'source_onnx_file', 'string', required=False),
            'meta1': KeyDefinition('graph', 'source_htp_file', 'string', required=False),
            'meta2': KeyDefinition('graph', 'format_version', 'string', required=True),
            'meta3': KeyDefinition('graph', 'export_timestamp', 'string', required=True),
            'meta4': KeyDefinition('graph', 'opset_imports', 'string', required=True),
            'meta5': KeyDefinition('graph', 'producer_name', 'string', required=True),
            'meta6': KeyDefinition('graph', 'producer_version', 'string', required=True),
            'meta7': KeyDefinition('graph', 'model_version', 'string', required=False),
            'meta8': KeyDefinition('graph', 'doc_string', 'string', required=False),
            
            # Parameter keys
            'param0': KeyDefinition('graph', 'parameter_strategy', 'string', required=True),
            'param1': KeyDefinition('graph', 'parameter_file', 'string', required=False),
            'param2': KeyDefinition('graph', 'parameter_checksum', 'string', required=False),
            
            # I/O keys
            'io0': KeyDefinition('graph', 'graph_inputs', 'string', required=True),
            'io1': KeyDefinition('graph', 'graph_outputs', 'string', required=True),
            'io2': KeyDefinition('graph', 'value_info', 'string', required=False),
            'io3': KeyDefinition('graph', 'initializers_ref', 'string', required=False),
        }
    
    def convert_onnx_to_graphml(
        self,
        onnx_file: str,
        graphml_file: str,
        htp_metadata: str = None,
        parameter_strategy: str = "sidecar",
        validate: bool = True
    ) -> ConversionResult:
        """Convert ONNX to GraphML v1.3 with validation."""
        
        try:
            # Step 1: Load and parse ONNX
            onnx_model = onnx.load(onnx_file)
            
            # Step 2: Create GraphML structure
            graphml_root = self._create_graphml_root()
            self._add_key_definitions(graphml_root)
            main_graph = self._create_main_graph(graphml_root, onnx_model)
            
            # Step 3: Add metadata using new key names
            self._add_metadata(main_graph, onnx_model, parameter_strategy)
            
            # Step 4: Process nodes and edges
            self._process_nodes(main_graph, onnx_model, htp_metadata)
            self._process_edges(main_graph, onnx_model)
            
            # Step 5: Handle parameters
            self._handle_parameters(main_graph, onnx_model, parameter_strategy, graphml_file)
            
            # Step 6: Write GraphML
            self._write_graphml(graphml_root, graphml_file)
            
            # Step 7: Validate if requested
            validation_results = []
            if validate:
                for validator in self.validators:
                    result = validator.validate(graphml_file, onnx_file)
                    validation_results.append(result)
                    if result.status == "FAIL":
                        raise ValidationError(f"Validation failed: {result.message}")
            
            return ConversionResult(
                status="SUCCESS",
                graphml_file=graphml_file,
                validation_results=validation_results,
                metrics=self._calculate_metrics(onnx_model, graphml_file)
            )
            
        except Exception as e:
            return ConversionResult(
                status="ERROR",
                error_message=str(e),
                error_code="CONVERSION_001"
            )
    
    def _add_metadata(self, graph: ET.Element, onnx_model, parameter_strategy: str):
        """Add metadata using v1.3 key names."""
        timestamp = datetime.now().isoformat()
        
        # Required metadata
        self._add_data(graph, "meta2", "1.3")  # format_version
        self._add_data(graph, "meta3", timestamp)  # export_timestamp
        self._add_data(graph, "param0", parameter_strategy)  # parameter_strategy
        
        # ONNX metadata
        if onnx_model.producer_name:
            self._add_data(graph, "meta5", onnx_model.producer_name)
        if onnx_model.producer_version:
            self._add_data(graph, "meta6", onnx_model.producer_version)
        if onnx_model.model_version:
            self._add_data(graph, "meta7", str(onnx_model.model_version))
        if onnx_model.doc_string:
            self._add_data(graph, "meta8", onnx_model.doc_string)
        
        # Opset imports
        opsets = [{"domain": imp.domain or "", "version": imp.version} 
                 for imp in onnx_model.opset_import]
        self._add_data(graph, "meta4", json.dumps(opsets))
        
        # Graph I/O specifications
        inputs = [{"name": inp.name, "type": self._get_tensor_type(inp.type), 
                  "shape": self._get_tensor_shape(inp.type)} 
                 for inp in onnx_model.graph.input]
        outputs = [{"name": out.name, "type": self._get_tensor_type(out.type),
                   "shape": self._get_tensor_shape(out.type)} 
                  for out in onnx_model.graph.output]
        
        self._add_data(graph, "io0", json.dumps(inputs))
        self._add_data(graph, "io1", json.dumps(outputs))
        
        # Value info (optional)
        if onnx_model.graph.value_info:
            value_info = [{"name": vi.name, "type": self._get_tensor_type(vi.type),
                          "shape": self._get_tensor_shape(vi.type)} 
                         for vi in onnx_model.graph.value_info]
            self._add_data(graph, "io2", json.dumps(value_info))
```

---

## V. Automated Migration System

### 5.1 Version Detection and Migration

```python
class GraphMLMigrationEngine:
    """Automated migration between GraphML versions."""
    
    def __init__(self):
        self.version_detectors = {
            '1.1': self._detect_v1_1,
            '1.2': self._detect_v1_2,
            '1.3': self._detect_v1_3
        }
        self.migrators = {
            ('1.1', '1.3'): self._migrate_v1_1_to_v1_3,
            ('1.2', '1.3'): self._migrate_v1_2_to_v1_3,
        }
    
    def detect_version(self, graphml_file: str) -> str:
        """Detect GraphML format version."""
        tree = ET.parse(graphml_file)
        root = tree.getroot()
        
        # Check for version in metadata
        version_data = root.find(".//data[@key='m2']") or root.find(".//data[@key='meta2']")
        if version_data is not None and version_data.text:
            return version_data.text
            
        # Fallback to key signature detection
        existing_keys = {k.get("id") for k in root.findall(".//key")}
        
        for version, detector in self.version_detectors.items():
            if detector(existing_keys):
                return version
                
        raise ValueError("Unable to detect GraphML version")
    
    def migrate_to_v1_3(
        self, 
        source_file: str, 
        target_file: str,
        backup: bool = True
    ) -> MigrationResult:
        """Migrate any version to v1.3."""
        
        try:
            source_version = self.detect_version(source_file)
            
            if source_version == '1.3':
                return MigrationResult(
                    status="NO_MIGRATION_NEEDED",
                    source_version=source_version,
                    target_version='1.3'
                )
            
            # Create backup if requested
            if backup:
                backup_file = f"{source_file}.backup.{source_version}"
                shutil.copy2(source_file, backup_file)
            
            # Perform migration
            migrator_key = (source_version, '1.3')
            if migrator_key not in self.migrators:
                raise ValueError(f"No migration path from {source_version} to 1.3")
                
            migrator = self.migrators[migrator_key]
            migration_result = migrator(source_file, target_file)
            
            # Validate migrated file
            validator = GraphMLV13Validator()
            validation_result = validator.validate_all_layers(target_file)
            
            if validation_result.status == "FAIL":
                raise ValidationError(f"Migration validation failed: {validation_result.message}")
            
            return MigrationResult(
                status="SUCCESS",
                source_version=source_version,
                target_version='1.3',
                backup_file=backup_file if backup else None,
                validation_result=validation_result
            )
            
        except Exception as e:
            return MigrationResult(
                status="ERROR",
                error_message=str(e),
                error_code="MIGRATION_001"
            )
    
    def _migrate_v1_1_to_v1_3(self, source_file: str, target_file: str) -> dict:
        """Migrate v1.1 to v1.3 (key remapping)."""
        tree = ET.parse(source_file)
        root = tree.getroot()
        
        # Key remapping rules
        key_remappings = {
            # Edge type keys moved to edge namespace
            't0': 'e1',  # tensor_type
            't1': 'e2',  # tensor_shape
            't2': 'e3',  # tensor_data_ref
            
            # Parameter keys get param prefix
            'p0': 'param0',  # parameter_strategy
            'p1': 'param1',  # parameter_file
            'p2': 'param2',  # parameter_checksum
            
            # Graph I/O keys get io prefix
            'g0': 'io0',  # graph_inputs
            'g1': 'io1',  # graph_outputs
            'g2': 'io2',  # value_info
            'g3': 'io3',  # initializers_ref
            
            # Metadata keys get meta prefix (CONFLICT RESOLUTION)
            'm0': 'meta0',  # source_onnx_file
            'm1': 'meta1',  # source_htp_file
            'm2': 'meta2',  # format_version
            'm3': 'meta3',  # export_timestamp
            'm4': 'meta4',  # opset_imports
            'm5': 'meta5',  # producer_name (v1.1 meaning)
            'm6': 'meta6',  # producer_version (v1.1 meaning)
            'm7': 'meta7',  # model_version (v1.1 meaning)
            'm8': 'meta8',  # doc_string (v1.1 meaning)
        }
        
        # Update key definitions
        for key_elem in root.findall(".//key"):
            old_id = key_elem.get("id")
            if old_id in key_remappings:
                key_elem.set("id", key_remappings[old_id])
        
        # Update data references
        for data_elem in root.findall(".//data"):
            old_key = data_elem.get("key")
            if old_key in key_remappings:
                data_elem.set("key", key_remappings[old_key])
        
        # Update version number
        version_elem = root.find(".//data[@key='meta2']")
        if version_elem is not None:
            version_elem.text = "1.3"
        
        # Write migrated file
        tree.write(target_file, encoding='utf-8', xml_declaration=True)
        
        return {
            "remapped_keys": len(key_remappings),
            "migration_type": "key_remapping"
        }
    
    def _migrate_v1_2_to_v1_3(self, source_file: str, target_file: str) -> dict:
        """Migrate v1.2 to v1.3 (minimal changes)."""
        tree = ET.parse(source_file)
        root = tree.getroot()
        
        # v1.2 already has most v1.3 structure, just need key prefix updates
        key_remappings = {
            't0': 'e1', 't1': 'e2', 't2': 'e3',  # Edge keys
            'p0': 'param0', 'p1': 'param1', 'p2': 'param2',  # Parameter keys
            'g0': 'io0', 'g1': 'io1', 'g2': 'io2', 'g3': 'io3',  # I/O keys
            # Metadata keys in v1.2 may already have conflicts - resolve them
            'm4': 'meta4', 'm5': 'meta5', 'm6': 'meta6', 'm7': 'meta7', 'm8': 'meta8'
        }
        
        # Apply remappings
        for key_elem in root.findall(".//key"):
            old_id = key_elem.get("id")
            if old_id in key_remappings:
                key_elem.set("id", key_remappings[old_id])
        
        for data_elem in root.findall(".//data"):
            old_key = data_elem.get("key")
            if old_key in key_remappings:
                data_elem.set("key", key_remappings[old_key])
        
        # Update version
        version_elem = root.find(".//data[@key='meta2']")
        if version_elem is not None:
            version_elem.text = "1.3"
        
        tree.write(target_file, encoding='utf-8', xml_declaration=True)
        
        return {
            "remapped_keys": len([k for k in key_remappings if k in [elem.get("id") for elem in root.findall(".//key")]]),
            "migration_type": "prefix_normalization"
        }
```

---

## VI. Governance Framework

### 6.1 Change Management Process

**Specification Change Workflow**:

1. **Proposal Phase**
   - RFC (Request for Comments) document required
   - Impact assessment on existing implementations
   - Backward compatibility analysis
   - Migration path definition

2. **Review Phase**
   - Technical review committee approval
   - Schema validation of proposed changes
   - Implementation feasibility assessment
   - User impact evaluation

3. **Implementation Phase**
   - Schema updates with version increment
   - Migration tool development
   - Comprehensive testing
   - Documentation updates

4. **Release Phase**
   - Staged rollout with validation
   - User notification and migration support
   - Performance monitoring
   - Rollback capability

### 6.2 Version Management

**Semantic Versioning**:
- **Major** (X.0): Breaking changes, incompatible schema changes
- **Minor** (X.Y): Backward-compatible additions
- **Patch** (X.Y.Z): Bug fixes, clarifications

**Compatibility Matrix**:
```yaml
version_compatibility:
  v1.3:
    reads: [v1.1, v1.2, v1.3]  # Can read and migrate
    writes: [v1.3]             # Only writes v1.3
    migrates_from: [v1.1, v1.2]
    
  future_versions:
    v1.4:
      breaking_changes: false
      migration_required: false
    v2.0:
      breaking_changes: true
      migration_required: true
      deprecation_timeline: "6 months"
```

### 6.3 Governance Bodies

**Technical Steering Committee**:
- Schema architecture decisions
- Breaking change approvals
- Version roadmap planning
- Conflict resolution

**Implementation Working Group**:
- Converter development
- Tool compatibility
- Performance optimization
- Quality assurance

**User Advisory Board**:
- Feature requirements
- Usability feedback
- Migration support
- Documentation needs

---

## VII. Performance Constraints & SLAs

### 7.1 Performance Targets

**Export Performance**:
- **Small Models** (<100 nodes): <1s export time, <50MB peak memory
- **Medium Models** (100-1K nodes): <2s export time, <200MB peak memory  
- **Large Models** (1K-10K nodes): <10s export time, <500MB peak memory
- **Enterprise Models** (>10K nodes): <60s export time, <2GB peak memory

**Import Performance**:
- **Round-trip Time**: ≤2x export time
- **Memory Efficiency**: ≤1.5x model size peak memory
- **Validation Time**: ≤10% of conversion time

**Quality Targets**:
- **Schema Validation**: 100% compliance required
- **Semantic Validation**: 100% consistency required
- **Round-trip Accuracy**: ≥85% node preservation, 100% parameter preservation

### 7.2 Performance Monitoring

```python
class PerformanceMonitor:
    """Performance monitoring for GraphML v1.3 operations."""
    
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'export_time_small': 1.0,     # seconds
            'export_time_medium': 2.0,    # seconds
            'export_time_large': 10.0,    # seconds
            'memory_peak_ratio': 1.5,     # relative to model size
            'round_trip_accuracy': 0.85,  # node preservation
            'parameter_accuracy': 1.0     # exact preservation
        }
    
    def monitor_export(self, onnx_file: str, graphml_file: str) -> PerformanceResult:
        """Monitor export performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Perform export
        try:
            converter = GraphMLV13Converter()
            result = converter.convert_onnx_to_graphml(onnx_file, graphml_file)
            
            end_time = time.time()
            peak_memory = psutil.Process().memory_info().rss
            
            # Calculate metrics
            model_size = os.path.getsize(onnx_file)
            node_count = self._count_onnx_nodes(onnx_file)
            
            metrics = {
                'export_time': end_time - start_time,
                'memory_used': peak_memory - start_memory,
                'memory_ratio': (peak_memory - start_memory) / model_size,
                'node_count': node_count,
                'model_size': model_size
            }
            
            # Check against thresholds
            performance_class = self._classify_model_size(node_count)
            threshold_key = f'export_time_{performance_class}'
            
            if metrics['export_time'] > self.thresholds.get(threshold_key, float('inf')):
                return PerformanceResult(
                    status="SLA_VIOLATION",
                    metric="export_time",
                    actual=metrics['export_time'],
                    threshold=self.thresholds[threshold_key],
                    metrics=metrics
                )
            
            return PerformanceResult(
                status="PASS",
                metrics=metrics
            )
            
        except Exception as e:
            return PerformanceResult(
                status="ERROR",
                error_message=str(e)
            )
```

---

## VIII. Quality Gates & Prevention System

### 8.1 Automated Quality Gates

**Pre-Release Validation Pipeline**:
```yaml
quality_gates:
  gate_1_schema_validation:
    description: "All generated GraphML must pass XSD validation"
    tools: [xmlschema, lxml]
    success_criteria: "100% schema compliance"
    blocking: true
    
  gate_2_semantic_validation:
    description: "Semantic consistency and connectivity validation"
    tools: [custom_validators]
    success_criteria: "100% semantic consistency"
    blocking: true
    
  gate_3_round_trip_validation:
    description: "Bidirectional conversion accuracy"
    tools: [round_trip_tester]
    success_criteria: "≥85% node preservation, 100% parameter preservation"
    blocking: true
    
  gate_4_performance_validation:
    description: "Performance SLA compliance"
    tools: [performance_monitor]
    success_criteria: "All SLA targets met"
    blocking: false  # Warning only
    
  gate_5_tool_compatibility:
    description: "Compatibility with major visualization tools"
    tools: [networkx, gephi_validator]
    success_criteria: "Load successfully in 90% of tested tools"
    blocking: false
    
  gate_6_regression_testing:
    description: "No regression in existing functionality"
    tools: [pytest]
    success_criteria: "100% test suite pass rate"
    blocking: true
```

### 8.2 Conflict Prevention System

**Automated Conflict Detection**:
```python
class ConflictPreventionSystem:
    """Prevent specification conflicts before they occur."""
    
    def __init__(self):
        self.reserved_keys = set()
        self.key_registry = {}
        self.version_history = {}
    
    def validate_key_proposal(self, key_id: str, purpose: str, version: str) -> ValidationResult:
        """Validate proposed key addition against conflicts."""
        
        # Check for existing usage
        if key_id in self.key_registry:
            existing = self.key_registry[key_id]
            if existing['purpose'] != purpose:
                return ValidationResult(
                    status="CONFLICT",
                    message=f"Key {key_id} conflict: existing purpose '{existing['purpose']}' vs proposed '{purpose}'",
                    error_code="KEY_CONFLICT_001"
                )
        
        # Check for reserved key patterns
        if self._is_reserved_pattern(key_id):
            return ValidationResult(
                status="RESERVED",
                message=f"Key {key_id} matches reserved pattern",
                error_code="KEY_RESERVED_001"
            )
        
        # Check version compatibility
        if not self._is_version_compatible(key_id, version):
            return ValidationResult(
                status="VERSION_CONFLICT",
                message=f"Key {key_id} not compatible with version {version}",
                error_code="KEY_VERSION_001"
            )
        
        return ValidationResult(
            status="APPROVED",
            message=f"Key {key_id} approved for {purpose} in {version}"
        )
    
    def register_key_usage(self, key_id: str, purpose: str, version: str):
        """Register key usage to prevent future conflicts."""
        self.key_registry[key_id] = {
            'purpose': purpose,
            'version': version,
            'registered_at': datetime.now().isoformat()
        }
    
    def generate_key_report(self) -> dict:
        """Generate comprehensive key usage report."""
        return {
            'total_keys': len(self.key_registry),
            'by_prefix': self._count_by_prefix(),
            'conflicts_detected': self._detect_all_conflicts(),
            'reserved_violations': self._check_reserved_violations(),
            'version_compatibility': self._check_version_compatibility()
        }
```

---

## IX. Implementation Requirements

### 9.1 Required Components

**Core Components**:
1. **SchemaValidator**: XSD/YAML validation engine
2. **SemanticValidator**: Consistency and connectivity validation
3. **RoundTripValidator**: Accuracy measurement and validation
4. **MigrationEngine**: Automated version migration
5. **PerformanceMonitor**: SLA compliance monitoring
6. **ConflictPrevention**: Specification conflict detection

**Converter Requirements**:
```python
class GraphMLV13ConverterInterface(ABC):
    """Required interface for GraphML v1.3 converters."""
    
    @abstractmethod
    def convert_onnx_to_graphml(
        self,
        onnx_file: str,
        graphml_file: str,
        htp_metadata: Optional[str] = None,
        parameter_strategy: str = "sidecar",
        validate: bool = True
    ) -> ConversionResult:
        """Convert ONNX to GraphML v1.3."""
        pass
    
    @abstractmethod
    def convert_graphml_to_onnx(
        self,
        graphml_file: str,
        onnx_file: str,
        validate: bool = True
    ) -> ConversionResult:
        """Convert GraphML v1.3 to ONNX."""
        pass
    
    @abstractmethod
    def validate_schema(self, graphml_file: str) -> ValidationResult:
        """Validate against v1.3 schema."""
        pass
    
    @abstractmethod
    def migrate_from_previous(
        self,
        old_graphml_file: str,
        new_graphml_file: str
    ) -> MigrationResult:
        """Migrate from previous versions."""
        pass
```

### 9.2 CLI Interface Requirements

```bash
# Export with v1.3 format (default)
modelexport export MODEL_NAME output.graphml --format v1.3

# Export with validation
modelexport export MODEL_NAME output.graphml --format v1.3 --validate

# Import with validation  
modelexport import-graphml input.graphml output.onnx --validate

# Migrate from previous versions
modelexport migrate-graphml old.graphml new.graphml --from-version 1.1 --to-version 1.3

# Validate existing GraphML
modelexport validate-graphml input.graphml --strict

# Performance benchmark
modelexport benchmark MODEL_NAME --format v1.3 --report performance.json

# Schema validation only
modelexport validate-schema input.graphml --schema-path graphml-v1.3.xsd
```

### 9.3 Testing Requirements

**Test Categories**:
1. **Schema Compliance Tests**: Validate all generated GraphML against XSD
2. **Migration Tests**: Test migration from v1.1 and v1.2 to v1.3
3. **Round-trip Tests**: Validate conversion accuracy across model types
4. **Performance Tests**: Ensure SLA compliance across model sizes
5. **Conflict Tests**: Verify conflict resolution and prevention
6. **Tool Compatibility Tests**: Validate with NetworkX, yEd, Gephi

**Test Coverage Requirements**:
- **Unit Tests**: 100% coverage of converter components
- **Integration Tests**: End-to-end conversion workflows  
- **Performance Tests**: All model size categories
- **Compatibility Tests**: All supported visualization tools
- **Regression Tests**: No degradation from previous versions

---

## X. Migration Guide

### 10.1 From v1.1 to v1.3

**Breaking Changes**:
- Key IDs changed (m5-m8 → meta5-meta8, etc.)
- Schema namespace updated
- Validation requirements stricter

**Migration Steps**:
```bash
# 1. Backup existing files
cp model_v1.1.graphml model_v1.1.graphml.backup

# 2. Auto-migrate using CLI
modelexport migrate-graphml model_v1.1.graphml model_v1.3.graphml --from-version 1.1

# 3. Validate migrated file
modelexport validate-graphml model_v1.3.graphml --strict

# 4. Test round-trip conversion
modelexport import-graphml model_v1.3.graphml reconstructed.onnx --validate
```

**Key Remapping Table**:
| v1.1 Key | v1.3 Key | Purpose | Auto-Migrated |
|----------|----------|---------|---------------|
| m5 | meta5 | producer_name | ✅ |
| m6 | meta6 | producer_version | ✅ |
| m7 | meta7 | model_version | ✅ |
| m8 | meta8 | doc_string | ✅ |
| t0 | e1 | tensor_type | ✅ |
| t1 | e2 | tensor_shape | ✅ |
| p0 | param0 | parameter_strategy | ✅ |
| g0 | io0 | graph_inputs | ✅ |

### 10.2 From v1.2 to v1.3

**Breaking Changes**:
- Minimal - mostly key prefix normalization
- Schema validation becomes mandatory
- Performance requirements formalized

**Migration Steps**: Same as v1.1, but with `--from-version 1.2`

### 10.3 Backward Compatibility

**Reading Older Versions**:
- v1.3 converters can read v1.1 and v1.2 files
- Automatic migration during read operations
- Warning messages for deprecated features

**Writing Older Versions**:
- Not supported - v1.3 only writes v1.3 format
- Use migration tools to downgrade if absolutely necessary
- Loss of v1.3-specific features in downgrade

---

## XI. Conclusion

GraphML v1.3 represents the definitive solution to the specification chaos through **schema-driven engineering**. By implementing the Six Pillars approach, this specification:

1. **Eliminates Conflicts**: Resolves m5-m8 key conflicts definitively
2. **Enforces Quality**: Multi-layer validation ensures correctness
3. **Enables Evolution**: Governance framework supports controlled change
4. **Prevents Regression**: Automated quality gates prevent future conflicts
5. **Ensures Performance**: SLA targets with monitoring and enforcement
6. **Facilitates Migration**: Automated migration from all previous versions

**The Schema IS the Specification**: All implementation, validation, and tooling derives from the formal XSD and YAML schema definitions. This approach eliminates ambiguity and ensures consistent implementation across all tools and converters.

**Implementation Timeline**:
- **Phase 1** (Week 1): Schema definition and validation framework
- **Phase 2** (Week 2): Converter implementation with key remapping
- **Phase 3** (Week 3): Migration engine and CLI integration
- **Phase 4** (Week 4): Quality gates and performance monitoring
- **Phase 5** (Week 5): Documentation, testing, and release

This specification ends the era of conflicting GraphML formats and establishes a solid foundation for future evolution of ONNX model visualization and interchange.

---

**Document Status**: FORMAL SPECIFICATION  
**Implementation Requirement**: MANDATORY for all GraphML v1.3 implementations  
**Schema Compliance**: 100% required  
**Quality Gates**: All must pass before release  

*This specification is the authoritative definition of GraphML v1.3. All implementations must conform to these requirements.*