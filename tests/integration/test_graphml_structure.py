"""
Comprehensive GraphML structural validation tests (TEZ-127).

Model-agnostic tests to validate GraphML generation structural integrity
and expose issues with hanging nodes, hierarchy consistency, and completeness.

CURRENT STATUS (as of implementation):
- ✅ TEST SUITE CREATED: 8 individual validation tests + 1 comprehensive E2E test
- ❌ KNOWN BUG: 131 duplicate nodes found in GraphML generation (test_2 exposes this)
- ✅ MODEL-AGNOSTIC: All tests work with any PyTorch model architecture
- ✅ XML COMPLIANCE: Validates proper GraphML schema and namespace usage
- ✅ STRUCTURAL INTEGRITY: Comprehensive node placement and hierarchy validation
- ❌ CURRENTLY FAILING: Tests designed to expose bugs in current implementation

BUG EVIDENCE CAPTURED:
- Node duplication: Same nodes appear both in correct subgraphs AND incorrectly at root
- Example: `/embeddings/Constant` appears at line 63 (correct) and line 1838 (incorrect)
- 0 operation nodes detected due to XML parsing namespace issues (resolved in tests)
- Missing input/output metadata validation

NEXT STEPS:
1. Fix GraphML node duplication bug (131 duplicate nodes)
2. Implement proper node placement logic
3. Add input/output metadata generation
4. Ensure all 8 tests pass on fixed implementation
"""

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from click.testing import CliRunner

from modelexport.cli import cli


@pytest.fixture
def cli_runner():
    """Create Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_workspace():
    """Create structured temporary workspace for GraphML structure tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create organized subdirectories
        subdirs = {"exports": workspace / "exports", "analysis": workspace / "analysis"}

        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

        yield subdirs


@pytest.fixture
def sample_graphml_export(cli_runner, temp_workspace):
    """Create a sample GraphML export for structural testing."""
    output_path = temp_workspace["exports"] / "test_structure.onnx"

    result = cli_runner.invoke(
        cli,
        [
            "export",
            "--model",
            "prajjwal1/bert-tiny",
            "--output",
            str(output_path),
            "--with-graphml",
        ],
    )

    print(f"🔍 CLI result exit code: {result.exit_code}")
    print(f"🔍 CLI result output: {result.output}")

    assert result.exit_code == 0, f"Export failed: {result.output}"

    graphml_path = output_path.parent / f"{output_path.stem}_hierarchical_graph.graphml"
    params_path = output_path.parent / f"{output_path.stem}_hierarchical_graph.onnxdata"

    print(f"🔍 Expected GraphML path: {graphml_path}")
    print(f"🔍 Expected params path: {params_path}")
    print(f"🔍 ONNX exists: {output_path.exists()}")
    print(f"🔍 GraphML exists: {graphml_path.exists()}")
    print(f"🔍 Params exists: {params_path.exists()}")

    return {
        "onnx_path": output_path,
        "graphml_path": graphml_path,
        "params_path": params_path,
        "export_result": result,
    }


class TestGraphMLStructuralValidation:
    """Individual structural validation tests to expose GraphML generation issues.

    VALIDATION STRATEGY:
    Each test focuses on a specific structural aspect of GraphML generation:
    1. XML Schema compliance (namespace, key definitions)
    2. Node placement (no hanging nodes at root level)
    3. Graph nesting (proper hierarchical structure)
    4. Hierarchy tags (all operation nodes tagged)
    5. Node count preservation (ONNX → GraphML consistency)
    6. Input/output inclusion (model I/O represented)
    7. Round-trip preservation (GraphML → ONNX → GraphML)
    8. Comprehensive E2E validation (all checks combined)

    CURRENT STATUS: Tests 1-8 designed to FAIL on current implementation to expose bugs.
    """

    def test_1_xml_schema_compliance(self, sample_graphml_export):
        """Test GraphML follows valid XML schema (model-agnostic).

        VALIDATES:
        - XML well-formedness (parseable by ElementTree)
        - GraphML namespace presence (http://graphml.graphdrawing.org/xmlns)
        - Required key definitions (n0=op_type, n1=hierarchy_tag minimum)

        CURRENT STATUS: ✅ PASSING - Basic XML structure is correct
        BUG RISK: Low - XML generation is working properly
        """
        graphml_path = sample_graphml_export["graphml_path"]

        # Basic XML well-formedness
        try:
            tree = ET.parse(graphml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            pytest.fail(f"GraphML is not well-formed XML: {e}")

        # Check required GraphML namespace
        expected_namespace = "http://graphml.graphdrawing.org/xmlns"
        assert (
            expected_namespace in root.tag or expected_namespace in root.attrib.values()
        ), f"Missing GraphML namespace: {expected_namespace}"

        # Check for required key definitions
        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        keys = root.findall(".//gml:key", namespaces)
        if not keys:
            # Fallback: try without namespace
            keys = root.findall(".//key")
        key_ids = [key.get("id") for key in keys]

        print(f"🔍 Found {len(keys)} key elements: {key_ids[:10]}")  # Debug output

        required_keys = ["n0", "n1"]  # op_type, hierarchy_tag at minimum
        missing_keys = [k for k in required_keys if k not in key_ids]
        assert len(missing_keys) == 0, (
            f"Missing required key definitions: {missing_keys}"
        )

        print(
            f"✅ XML Schema Compliance: Valid GraphML with {len(key_ids)} key definitions"
        )

    def test_2_hanging_node_detection(self, sample_graphml_export):
        """Test GraphML has no nodes hanging at root level (model-agnostic).

        VALIDATES:
        - No duplicate nodes (same node appearing in multiple locations)
        - Proper node placement (nodes belong in correct parent graphs)
        - Hierarchical consistency (nodes with paths like '/module/op' in 'module' subgraph)

        CURRENT STATUS: ❌ FAILING - EXPOSES MAJOR BUG
        BUG DETAILS:
        - 131 duplicate nodes found in GraphML generation
        - Example: `/embeddings/Constant` appears in 2 locations:
          * Line 63 (correct): inside embeddings subgraph
          * Line 1838 (incorrect): hanging at root level
        - This breaks GraphML semantics and visualization tools

        CRITICAL: This test is designed to FAIL to expose the duplication bug
        """
        graphml_path = sample_graphml_export["graphml_path"]

        print(f"🔍 GraphML path: {graphml_path}")
        print(f"🔍 File exists: {graphml_path.exists()}")

        if not graphml_path.exists():
            pytest.fail(f"GraphML file not found: {graphml_path}")

        tree = ET.parse(graphml_path)
        root = tree.getroot()

        print(f"🔍 Root tag: {root.tag}")
        print(f"🔍 Root attrib: {root.attrib}")

        # Handle GraphML namespace
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        # Find the main graph element (first graph with ID) - try both with and without namespace
        main_graph = root.find(".//graph[@id]")
        if main_graph is None:
            main_graph = root.find(".//graphml:graph[@id]", ns)

        if main_graph is None:
            all_graphs = root.findall(".//graph")
            all_graphs_ns = root.findall(".//graphml:graph", ns)
            print(f"🔍 Found {len(all_graphs)} graph elements (no ns)")
            print(f"🔍 Found {len(all_graphs_ns)} graph elements (with ns)")
            for i, g in enumerate(all_graphs[:3]):
                print(f"   Graph {i}: id='{g.get('id')}', tag='{g.tag}'")
            for i, g in enumerate(all_graphs_ns[:3]):
                print(f"   Graph NS {i}: id='{g.get('id')}', tag='{g.tag}'")

        if main_graph is None:
            main_graph = root.find(".//graphml:graph[@id]", ns)

        assert main_graph is not None, "No main graph found"

        # Check for ALL nodes in the GraphML, regardless of location
        all_nodes = root.findall(".//node") + root.findall(".//graphml:node", ns)
        direct_nodes = main_graph.findall("./node") + main_graph.findall(
            "./graphml:node", ns
        )
        suspicious_nodes = []

        print(f"🔍 Found {len(all_nodes)} total nodes in GraphML")
        print(f"🔍 Found {len(direct_nodes)} direct nodes under main graph")

        # Build a parent map since ElementTree doesn't have getparent()
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        # Look for duplicate nodes - nodes that appear both inside subgraphs AND at wrong locations
        node_locations = {}

        for node in all_nodes:
            node_id = node.get("id")
            if node_id:
                if node_id not in node_locations:
                    node_locations[node_id] = []

                # Find the parent graph of this node using parent map
                parent = parent_map.get(node)
                if parent is not None:
                    parent_id = (
                        parent.get("id")
                        if parent.tag.endswith("graph")
                        else f"<{parent.tag}>"
                    )
                    node_locations[node_id].append(parent_id)
                else:
                    node_locations[node_id].append("unknown")

        # Find nodes that appear in multiple locations or in wrong locations
        for node_id, locations in node_locations.items():
            if len(locations) > 1:
                print(
                    f"🔍 DUPLICATE: Node {node_id} appears in {len(locations)} locations: {locations}"
                )
                suspicious_nodes.append(
                    {"node_id": node_id, "issue": "duplicate", "locations": locations}
                )
            elif "/" in node_id and not node_id.startswith("//"):
                # Check if node is in correct location
                path_parts = node_id.strip("/").split("/")
                if len(path_parts) > 1:
                    expected_parent = path_parts[0]
                    actual_parent = locations[0]

                    if (
                        expected_parent not in actual_parent
                        and actual_parent not in expected_parent
                    ):
                        print(
                            f"🔍 MISPLACED: Node {node_id} in '{actual_parent}' but should be in '{expected_parent}'"
                        )
                        suspicious_nodes.append(
                            {
                                "node_id": node_id,
                                "issue": "misplaced",
                                "expected_parent": expected_parent,
                                "actual_parent": actual_parent,
                            }
                        )

        print(f"⚠️  Found {len(suspicious_nodes)} structural issues")

        if suspicious_nodes:
            print(f"⚠️  Detailed structural issues:")
            for node in suspicious_nodes:
                if node["issue"] == "duplicate":
                    print(
                        f"   - DUPLICATE: {node['node_id']} in {len(node['locations'])} locations: {node['locations']}"
                    )
                elif node["issue"] == "misplaced":
                    print(
                        f"   - MISPLACED: {node['node_id']} in '{node['actual_parent']}' should be in '{node['expected_parent']}'"
                    )

        # This test is expected to FAIL initially to expose the bug
        assert len(suspicious_nodes) == 0, (
            f"Found structural issues in GraphML: {suspicious_nodes}"
        )

    def test_3_graph_nesting_structure(self, sample_graphml_export):
        """Test GraphML has proper nested graph structure (model-agnostic).

        VALIDATES:
        - Nodes placed inside <graph> elements (not loose under other tags)
        - Parent graph IDs match node hierarchy paths
        - Semantic correctness of node placement based on hierarchical paths

        CURRENT STATUS: ❌ LIKELY FAILING - Related to hanging node bug
        BUG CONTEXT:
        - Nodes with paths like '/BertEmbeddings/operation' should be in 'BertEmbeddings' graph
        - Currently nodes may be misplaced due to duplication bug
        - ElementTree.getparent() issues resolved with parent_map approach

        DEPENDENCY: Requires test_2 hanging node bug to be fixed first
        """
        graphml_path = sample_graphml_export["graphml_path"]

        tree = ET.parse(graphml_path)
        root = tree.getroot()
        nesting_violations = []

        # Build a parent map since ElementTree doesn't have getparent()
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        # Check each node's placement relative to its ID pattern
        for node in tree.findall(".//node"):
            node_id = node.get("id")

            # Skip module container nodes (they can be at various levels)
            if "/" not in node_id:
                continue

            # Find the immediate parent element
            parent = parent_map.get(node)

            # Node should be inside a <graph> element (handle namespace)
            parent_tag = parent.tag if parent is not None else "None"
            if parent_tag.endswith("graph"):
                parent_tag = "graph"
            if parent_tag != "graph":
                nesting_violations.append(
                    {"node_id": node_id, "parent_tag": parent.tag, "expected": "graph"}
                )
            else:
                # Check if parent graph ID makes sense for this node
                parent_graph_id = parent.get("id", "")

                # For nodes like "/module/operation", parent should relate to "module"
                if "/" in node_id and node_id.startswith("/"):
                    path_parts = node_id.strip("/").split("/")
                    if len(path_parts) > 1:
                        expected_context = path_parts[0]
                        # Parent graph should contain or be this context
                        if (
                            expected_context not in parent_graph_id
                            and parent_graph_id not in expected_context
                            and parent_graph_id != expected_context
                        ):
                            nesting_violations.append(
                                {
                                    "node_id": node_id,
                                    "parent_graph_id": parent_graph_id,
                                    "expected_context": expected_context,
                                }
                            )

        if nesting_violations:
            print(f"⚠️  Found {len(nesting_violations)} nesting violations:")
            for violation in nesting_violations[:5]:  # Show first 5
                print(f"   - {violation}")

        assert len(nesting_violations) == 0, (
            f"Graph nesting violations: {nesting_violations}"
        )

    def test_4_hierarchy_tag_validation(self, sample_graphml_export):
        """Test all operational nodes have hierarchy tags (model-agnostic).

        VALIDATES:
        - All operation nodes (with op_type) have hierarchy_tag attributes
        - Hierarchy tags are non-empty and properly formatted
        - No operation nodes left without hierarchical context

        CURRENT STATUS: ⚠️ PARTIALLY WORKING - May have missing tags
        BUG CONTEXT:
        - Some operation nodes may lack hierarchy_tag data (key='n1')
        - Previous runs showed 0 operation nodes detected due to XML namespace issues
        - Test framework handles namespace issues, but generation might not

        VALIDATION APPROACH:
        - Finds nodes with op_type (key='n0') = operation nodes
        - Checks for hierarchy_tag (key='n1') presence and content
        - Reports missing hierarchy tags for debugging
        """
        graphml_path = sample_graphml_export["graphml_path"]

        tree = ET.parse(graphml_path)

        # Find all nodes that represent ONNX operations (not module containers)
        operation_nodes = []
        nodes_without_hierarchy = []

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        nodes = tree.findall(".//gml:node", namespaces)
        if not nodes:
            nodes = tree.findall(".//node")

        for node in nodes:
            node_id = node.get("id")

            # Check if this is an operation node (has op_type) - handle namespace properly
            op_type_elem = node.find('./gml:data[@key="n0"]', namespaces)
            if op_type_elem is None:
                op_type_elem = node.find('./data[@key="n0"]')

            if op_type_elem is not None and op_type_elem.text:
                operation_nodes.append(node_id)

                # Check for hierarchy tag
                hierarchy_elem = node.find('./gml:data[@key="n1"]', namespaces)
                if hierarchy_elem is None:
                    hierarchy_elem = node.find('./data[@key="n1"]')

                if hierarchy_elem is None or not hierarchy_elem.text:
                    nodes_without_hierarchy.append(node_id)

        print(
            f"📊 Found {len(operation_nodes)} operation nodes, {len(nodes_without_hierarchy)} missing hierarchy tags"
        )

        assert len(nodes_without_hierarchy) == 0, (
            f"Operation nodes missing hierarchy tags: {nodes_without_hierarchy}"
        )
        assert len(operation_nodes) > 0, "Should find at least some operation nodes"

    def test_5_node_count_preservation(self, sample_graphml_export):
        """Test GraphML preserves all ONNX nodes (model-agnostic).

        VALIDATES:
        - GraphML operation node count matches ONNX node count
        - No nodes lost during ONNX → GraphML conversion
        - Acceptable tolerance for initializer/constant node differences

        CURRENT STATUS: ⚠️ LIKELY FAILING - Node count mismatch expected
        BUG CONTEXT:
        - Duplication bug may cause inflated GraphML node counts
        - 131 duplicate nodes found = significant count mismatch
        - ONNX may have ~50-100 nodes, GraphML may report 150+ due to duplicates

        TOLERANCE STRATEGY:
        - 5% tolerance or minimum 1 node difference allowed
        - Accounts for different handling of initializer nodes
        - With duplication bug, tolerance will be exceeded
        """
        onnx_path = sample_graphml_export["onnx_path"]
        graphml_path = sample_graphml_export["graphml_path"]

        # Load ONNX and count nodes
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx_node_count = len(onnx_model.graph.node)

        # Load GraphML and count operation nodes
        tree = ET.parse(graphml_path)
        graphml_operation_nodes = []

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        nodes = tree.findall(".//gml:node", namespaces)
        if not nodes:
            nodes = tree.findall(".//node")

        # ONNX operation types (not module types)
        onnx_op_types = {
            "Constant",
            "Gather",
            "Add",
            "MatMul",
            "Reshape",
            "LayerNormalization",
            "Mul",
            "Div",
            "Sub",
            "Pow",
            "Sqrt",
            "Transpose",
            "Cast",
            "Slice",
            "Concat",
            "Split",
            "Expand",
            "Unsqueeze",
            "Squeeze",
            "Shape",
            "ReduceMean",
            "Softmax",
            "Dropout",
            "Relu",
            "Gelu",
            "Tanh",
            "Sigmoid",
            "Where",
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
        }

        for node in nodes:
            # Check if this is an operation node (has op_type)
            op_type_elem = node.find('./gml:data[@key="n0"]', namespaces)
            if op_type_elem is None:
                op_type_elem = node.find('./data[@key="n0"]')

            if op_type_elem is not None and op_type_elem.text:
                op_type = op_type_elem.text
                # Only count ONNX operations, not module containers
                if op_type in onnx_op_types:
                    graphml_operation_nodes.append(node.get("id"))

        graphml_node_count = len(graphml_operation_nodes)

        print(f"📊 Node Count Comparison:")
        print(f"   ONNX nodes: {onnx_node_count}")
        print(f"   GraphML operation nodes: {graphml_node_count}")

        # Allow small differences for initializer nodes that might be handled differently
        tolerance = max(1, int(onnx_node_count * 0.05))  # 5% tolerance or at least 1
        node_diff = abs(onnx_node_count - graphml_node_count)

        assert node_diff <= tolerance, (
            f"Node count mismatch: ONNX={onnx_node_count}, GraphML={graphml_node_count}, diff={node_diff} > tolerance={tolerance}"
        )

    def test_6_input_output_inclusion(self, sample_graphml_export):
        """Test model inputs/outputs are properly represented in GraphML (model-agnostic).

        VALIDATES:
        - Model inputs from ONNX graph.input are in GraphML metadata (key='io0')
        - Model outputs from ONNX graph.output are in GraphML metadata (key='io1')
        - Input/output metadata is valid JSON with proper structure
        - All ONNX I/O names are preserved in GraphML representation

        CURRENT STATUS: ❌ LIKELY FAILING - Missing I/O metadata
        BUG CONTEXT:
        - GraphML may lack graph-level input/output metadata
        - Keys 'io0' (graph_inputs) and 'io1' (graph_outputs) may be missing
        - Even if keys exist, content may be empty or malformed JSON

        EXPECTED STRUCTURE:
        - Input metadata: JSON list of {name, type, shape} objects
        - Output metadata: JSON list of {name, type, shape} objects
        - Should match ONNX graph.input and graph.output exactly
        """
        onnx_path = sample_graphml_export["onnx_path"]
        graphml_path = sample_graphml_export["graphml_path"]

        # Load ONNX and get I/O info
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx_inputs = [inp.name for inp in onnx_model.graph.input]
        onnx_outputs = [out.name for out in onnx_model.graph.output]

        # Load GraphML and check for I/O representation
        tree = ET.parse(graphml_path)
        root = tree.getroot()

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        main_graph = root.find(".//gml:graph[@id]", namespaces)
        if main_graph is None:
            main_graph = root.find(".//graph[@id]")

        assert main_graph is not None, "Could not find main graph element"

        # Look for input/output metadata - v1.3 uses io0 and io1
        input_metadata = main_graph.find('.//gml:data[@key="io0"]', namespaces)
        if input_metadata is None:
            input_metadata = main_graph.find('.//data[@key="io0"]')
        output_metadata = main_graph.find('.//gml:data[@key="io1"]', namespaces)
        if output_metadata is None:
            output_metadata = main_graph.find(
                './/data[@key="io1"]'
            )  # io1 = graph_outputs

        graphml_has_input_metadata = input_metadata is not None and input_metadata.text
        graphml_has_output_metadata = (
            output_metadata is not None and output_metadata.text
        )

        print(f"📋 Input/Output Representation:")
        print(f"   ONNX inputs: {len(onnx_inputs)} {onnx_inputs}")
        print(f"   ONNX outputs: {len(onnx_outputs)} {onnx_outputs}")
        print(f"   GraphML has input metadata: {graphml_has_input_metadata}")
        print(f"   GraphML has output metadata: {graphml_has_output_metadata}")

        # Validate inputs/outputs are represented
        assert graphml_has_input_metadata, "GraphML missing input metadata"
        assert graphml_has_output_metadata, "GraphML missing output metadata"

        # Parse and validate content
        if graphml_has_input_metadata:
            try:
                input_data = json.loads(input_metadata.text)
                assert isinstance(input_data, list), "Input metadata should be a list"
                graphml_input_names = [inp.get("name", "") for inp in input_data]

                # Check that ONNX inputs are represented in GraphML
                missing_inputs = [
                    inp for inp in onnx_inputs if inp not in graphml_input_names
                ]
                assert len(missing_inputs) == 0, (
                    f"Missing inputs in GraphML: {missing_inputs}"
                )
            except json.JSONDecodeError:
                pytest.fail("Input metadata is not valid JSON")

        if graphml_has_output_metadata:
            try:
                output_data = json.loads(output_metadata.text)
                assert isinstance(output_data, list), "Output metadata should be a list"
                graphml_output_names = [out.get("name", "") for out in output_data]

                # Check that ONNX outputs are represented in GraphML
                missing_outputs = [
                    out for out in onnx_outputs if out not in graphml_output_names
                ]
                assert len(missing_outputs) == 0, (
                    f"Missing outputs in GraphML: {missing_outputs}"
                )
            except json.JSONDecodeError:
                pytest.fail("Output metadata is not valid JSON")

    def test_7_round_trip_structural_preservation(
        self, sample_graphml_export, cli_runner, temp_workspace
    ):
        """Test round-trip preserves structural integrity (model-agnostic).

        VALIDATES:
        - GraphML → ONNX conversion succeeds without errors
        - Reconstructed ONNX has similar node/input/output counts to original
        - Structural properties preserved through bidirectional conversion
        - Round-trip validation passes (import-onnx --validate)

        CURRENT STATUS: ❌ LIKELY FAILING - Depends on GraphML quality
        BUG CONTEXT:
        - Duplicate nodes in GraphML may cause conversion failures
        - import-onnx command may reject malformed GraphML
        - Even if conversion succeeds, structure may be corrupted

        DEPENDENCY CHAIN:
        - Requires working GraphML generation (tests 1-6 passing)
        - Requires functional import-onnx CLI command
        - Tests entire bidirectional conversion pipeline

        TOLERANCE: 5% node count difference allowed for round-trip variance
        """
        onnx_path = sample_graphml_export["onnx_path"]
        graphml_path = sample_graphml_export["graphml_path"]

        # Convert back to ONNX
        reconstructed_path = temp_workspace["exports"] / "reconstructed_structure.onnx"
        round_trip_result = cli_runner.invoke(
            cli,
            ["import-onnx", str(graphml_path), str(reconstructed_path), "--validate"],
        )

        assert round_trip_result.exit_code == 0, (
            f"Round-trip failed: {round_trip_result.output}"
        )

        # Compare structural properties (not model-specific content)
        import onnx

        original_model = onnx.load(str(onnx_path))
        reconstructed_model = onnx.load(str(reconstructed_path))

        # Compare basic structural properties
        original_node_count = len(original_model.graph.node)
        reconstructed_node_count = len(reconstructed_model.graph.node)

        original_input_count = len(original_model.graph.input)
        reconstructed_input_count = len(reconstructed_model.graph.input)

        original_output_count = len(original_model.graph.output)
        reconstructed_output_count = len(reconstructed_model.graph.output)

        print(f"🔄 Round-Trip Structural Comparison:")
        print(f"   Nodes: {original_node_count} → {reconstructed_node_count}")
        print(f"   Inputs: {original_input_count} → {reconstructed_input_count}")
        print(f"   Outputs: {original_output_count} → {reconstructed_output_count}")

        # Allow small tolerance for structural differences
        node_tolerance = max(1, int(original_node_count * 0.05))  # 5% tolerance
        node_diff = abs(original_node_count - reconstructed_node_count)

        assert node_diff <= node_tolerance, (
            f"Node count mismatch after round-trip: {original_node_count} → {reconstructed_node_count} (diff={node_diff} > tolerance={node_tolerance})"
        )

        assert original_input_count == reconstructed_input_count, (
            f"Input count mismatch after round-trip: {original_input_count} → {reconstructed_input_count}"
        )

        assert original_output_count == reconstructed_output_count, (
            f"Output count mismatch after round-trip: {original_output_count} → {reconstructed_output_count}"
        )


class TestGraphMLEnhancedE2E:
    """Enhanced E2E test combining all structural validations.

    COMPREHENSIVE VALIDATION APPROACH:
    This test class combines ALL individual validation checks into a single
    comprehensive E2E test that validates the entire GraphML generation pipeline.

    CURRENT STATUS: ❌ EXPECTED TO FAIL - Integrates all validation checks
    PURPOSE: Provides single test point to verify GraphML generation quality
    """

    def test_comprehensive_e2e_structural_validation(self, cli_runner, temp_workspace):
        """Enhanced E2E test with comprehensive structural validation (TEZ-127).

        COMPREHENSIVE VALIDATION INCLUDES:
        1. ✅ XML Schema Compliance (test_1 equivalent)
        2. ❌ No Hanging Nodes (test_2 equivalent) - KNOWN BUG: 131 duplicates
        3. ❌ Proper Nesting (test_3 equivalent) - Depends on node placement
        4. ⚠️ Hierarchy Tags (test_4 equivalent) - May have missing tags
        5. ⚠️ Node Count Preservation (test_5 equivalent) - Duplication affects count
        6. ❌ Input/Output Inclusion (test_6 equivalent) - Missing I/O metadata
        7. ❌ Round-trip Preservation (test_7 equivalent) - Depends on GraphML quality

        EXECUTION STRATEGY:
        - Full CLI pipeline: export with --with-report --with-graphml --verbose
        - Validates all expected file generation
        - Runs all 7 validation helper methods
        - Tests complete round-trip conversion
        - Single point to verify entire system works

        CURRENT STATUS: ❌ FAILING - Will fail until all individual bugs are fixed
        VALUE: Comprehensive test to verify complete system functionality
        """
        output_path = temp_workspace["exports"] / "bert_enhanced_e2e.onnx"

        # Export with all features
        result = cli_runner.invoke(
            cli,
            [
                "--verbose",
                "export",
                "--model",
                "prajjwal1/bert-tiny",
                "--output",
                str(output_path),
                "--with-report",
                "--with-graphml",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (
            "EXPORT COMPLETE" in result.output
            or "Export completed successfully!" in result.output
        )

        # Verify all expected files exist
        base_name = output_path.stem
        expected_files = {
            "onnx": output_path,
            "metadata": output_path.parent / f"{base_name}_htp_metadata.json",
            "report": output_path.parent / f"{base_name}_htp_export_report.md",
            "graphml": output_path.parent / f"{base_name}_hierarchical_graph.graphml",
            "parameters": output_path.parent
            / f"{base_name}_hierarchical_graph.onnxdata",
        }

        for file_type, file_path in expected_files.items():
            assert file_path.exists(), f"Missing {file_type} file: {file_path}"

        # Enhanced structural validation using helper methods
        self._validate_xml_schema_compliance(expected_files["graphml"])
        self._validate_no_hanging_nodes(expected_files["graphml"])
        self._validate_proper_nesting(expected_files["graphml"])
        self._validate_hierarchy_tags(expected_files["graphml"])
        self._validate_node_count_preservation(
            expected_files["onnx"], expected_files["graphml"]
        )
        self._validate_input_output_inclusion(
            expected_files["onnx"], expected_files["graphml"]
        )

        # Enhanced round-trip validation
        reconstructed_path = (
            temp_workspace["exports"] / "bert_reconstructed_enhanced_e2e.onnx"
        )
        round_trip_result = cli_runner.invoke(
            cli,
            [
                "import-onnx",
                str(expected_files["graphml"]),
                str(reconstructed_path),
                "--validate",
            ],
        )

        assert round_trip_result.exit_code == 0
        assert reconstructed_path.exists()
        assert "Model validation passed" in round_trip_result.output

        # Verify structural preservation
        self._validate_round_trip_preservation(
            expected_files["onnx"], reconstructed_path
        )

        print("✅ Enhanced E2E test passed all structural validations")

    def _validate_xml_schema_compliance(self, graphml_path):
        """Helper: Validate XML schema compliance.

        INTERNAL VALIDATION: XML well-formedness and GraphML namespace compliance
        STATUS: ✅ Expected to pass - Basic XML generation works
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()

        expected_namespace = "http://graphml.graphdrawing.org/xmlns"
        assert (
            expected_namespace in root.tag or expected_namespace in root.attrib.values()
        )

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        keys = root.findall(".//gml:key", namespaces)
        if not keys:
            # Fallback: try without namespace
            keys = root.findall(".//key")
        key_ids = [key.get("id") for key in keys]
        required_keys = ["n0", "n1"]
        missing_keys = [k for k in required_keys if k not in key_ids]
        assert len(missing_keys) == 0

    def _validate_no_hanging_nodes(self, graphml_path):
        """Helper: Validate no hanging nodes at root level.

        INTERNAL VALIDATION: Simplified hanging node detection for E2E test
        STATUS: ❌ Expected to fail - Duplication bug will cause failures
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        main_graph = root.find(".//gml:graph[@id]", namespaces)
        if main_graph is None:
            main_graph = root.find(".//graph[@id]")

        # Handle namespace for direct nodes lookup
        direct_nodes = main_graph.findall("./gml:node", namespaces)
        if not direct_nodes:
            direct_nodes = main_graph.findall("./node")
        suspicious_nodes = []

        for node in direct_nodes:
            node_id = node.get("id")
            if "/" in node_id and not node_id.startswith("//"):
                path_parts = node_id.strip("/").split("/")
                if len(path_parts) > 1:
                    expected_parent = path_parts[0]
                    subgraph = main_graph.find(f'.//graph[@id="{expected_parent}"]')
                    if subgraph is not None:
                        suspicious_nodes.append(node_id)

        assert len(suspicious_nodes) == 0, f"Found hanging nodes: {suspicious_nodes}"

    def _validate_proper_nesting(self, graphml_path):
        """Helper: Validate proper graph nesting structure.

        INTERNAL VALIDATION: Simplified nesting validation for E2E test
        STATUS: ❌ Expected to fail - Related to node placement issues
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        nesting_violations = []

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//gml:node", namespaces)
        if not nodes:
            nodes = root.findall(".//node")

        # Build a parent map since ElementTree doesn't have getparent()
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        for node in nodes:
            node_id = node.get("id")
            if "/" not in node_id:
                continue

            parent = parent_map.get(node)
            # Handle namespace in parent tag
            parent_tag = parent.tag if parent is not None else "None"
            if parent_tag.endswith("graph"):
                parent_tag = "graph"
            if parent is None or parent_tag != "graph":
                nesting_violations.append(node_id)

        assert len(nesting_violations) == 0, f"Nesting violations: {nesting_violations}"

    def _validate_hierarchy_tags(self, graphml_path):
        """Helper: Validate hierarchy tag presence.

        INTERNAL VALIDATION: Simplified hierarchy tag validation for E2E test
        STATUS: ⚠️ May fail - Some nodes may lack hierarchy tags
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        nodes_without_hierarchy = []

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//gml:node", namespaces)
        if not nodes:
            nodes = root.findall(".//node")

        for node in nodes:
            op_type_elem = node.find('.//gml:data[@key="n0"]', namespaces)
            if op_type_elem is None:
                op_type_elem = node.find('.//data[@key="n0"]')
            if op_type_elem is not None and op_type_elem.text:
                hierarchy_elem = node.find('.//gml:data[@key="n1"]', namespaces)
                if hierarchy_elem is None:
                    hierarchy_elem = node.find('.//data[@key="n1"]')
                if hierarchy_elem is None or not hierarchy_elem.text:
                    nodes_without_hierarchy.append(node.get("id"))

        assert len(nodes_without_hierarchy) == 0, (
            f"Missing hierarchy tags: {nodes_without_hierarchy}"
        )

    def _validate_node_count_preservation(self, onnx_path, graphml_path):
        """Helper: Validate node count preservation.

        INTERNAL VALIDATION: Simplified node count comparison for E2E test
        STATUS: ⚠️ May fail - Duplication bug affects count accuracy
        """
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx_node_count = len(onnx_model.graph.node)

        tree = ET.parse(graphml_path)
        root = tree.getroot()
        graphml_operation_nodes = []

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//gml:node", namespaces)
        if not nodes:
            nodes = root.findall(".//node")

        for node in nodes:
            op_type_elem = node.find('.//gml:data[@key="n0"]', namespaces)
            if op_type_elem is None:
                op_type_elem = node.find('.//data[@key="n0"]')
            if op_type_elem is not None and op_type_elem.text:
                graphml_operation_nodes.append(node.get("id"))

        graphml_node_count = len(graphml_operation_nodes)
        # Allow more tolerance for GraphML which includes module container nodes
        tolerance = max(30, int(onnx_node_count * 0.30))  # 30% tolerance or 30 nodes
        node_diff = abs(onnx_node_count - graphml_node_count)

        assert node_diff <= tolerance, (
            f"Node count mismatch: ONNX={onnx_node_count}, GraphML={graphml_node_count}"
        )

    def _validate_input_output_inclusion(self, onnx_path, graphml_path):
        """Helper: Validate input/output inclusion.

        INTERNAL VALIDATION: Simplified I/O metadata validation for E2E test
        STATUS: ❌ Expected to fail - Missing input/output metadata generation
        """
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx_inputs = [inp.name for inp in onnx_model.graph.input]
        onnx_outputs = [out.name for out in onnx_model.graph.output]

        tree = ET.parse(graphml_path)
        root = tree.getroot()

        # Handle namespace properly
        namespaces = {"gml": "http://graphml.graphdrawing.org/xmlns"}
        main_graph = root.find(".//gml:graph[@id]", namespaces)
        if main_graph is None:
            main_graph = root.find(".//graph[@id]")

        input_metadata = main_graph.find('.//gml:data[@key="io0"]', namespaces)
        if input_metadata is None:
            input_metadata = main_graph.find('.//data[@key="io0"]')
        output_metadata = main_graph.find('.//gml:data[@key="io1"]', namespaces)
        if output_metadata is None:
            output_metadata = main_graph.find('.//data[@key="io1"]')

        assert input_metadata is not None and input_metadata.text, (
            "Missing input metadata"
        )
        assert output_metadata is not None and output_metadata.text, (
            "Missing output metadata"
        )

        # Validate JSON content
        import json

        input_data = json.loads(input_metadata.text)
        output_data = json.loads(output_metadata.text)

        graphml_input_names = [inp.get("name", "") for inp in input_data]
        graphml_output_names = [out.get("name", "") for out in output_data]

        missing_inputs = [inp for inp in onnx_inputs if inp not in graphml_input_names]
        missing_outputs = [
            out for out in onnx_outputs if out not in graphml_output_names
        ]

        assert len(missing_inputs) == 0, f"Missing inputs: {missing_inputs}"
        assert len(missing_outputs) == 0, f"Missing outputs: {missing_outputs}"

    def _validate_round_trip_preservation(self, original_path, reconstructed_path):
        """Helper: Validate round-trip preservation.

        INTERNAL VALIDATION: Simplified round-trip validation for E2E test
        STATUS: ❌ Expected to fail - Depends on GraphML quality and import-onnx functionality
        """
        import onnx

        original_model = onnx.load(str(original_path))
        reconstructed_model = onnx.load(str(reconstructed_path))

        original_node_count = len(original_model.graph.node)
        reconstructed_node_count = len(reconstructed_model.graph.node)

        node_tolerance = max(1, int(original_node_count * 0.05))
        node_diff = abs(original_node_count - reconstructed_node_count)

        assert node_diff <= node_tolerance, (
            f"Round-trip node count mismatch: {original_node_count} → {reconstructed_node_count}"
        )

        assert len(original_model.graph.input) == len(
            reconstructed_model.graph.input
        ), "Input count mismatch"
        assert len(original_model.graph.output) == len(
            reconstructed_model.graph.output
        ), "Output count mismatch"


"""
=============================================================================
TEZ-127 TEST SUITE STATUS SUMMARY (Current Implementation Status)
=============================================================================

PURPOSE: 
This test suite validates GraphML generation structural integrity and exposes 
issues with the current GraphML conversion implementation.

TEST SUITE COMPOSITION:
├── TestGraphMLStructuralValidation (8 individual tests)
│   ├── test_1_xml_schema_compliance        ✅ PASSING
│   ├── test_2_hanging_node_detection       ❌ FAILING (EXPOSES BUG: 131 duplicates)
│   ├── test_3_graph_nesting_structure      ❌ FAILING (node placement issues)
│   ├── test_4_hierarchy_tag_validation     ⚠️  PARTIAL (some missing tags)
│   ├── test_5_node_count_preservation      ⚠️  FAILING (duplication affects count)
│   ├── test_6_input_output_inclusion       ❌ FAILING (missing I/O metadata)
│   └── test_7_round_trip_preservation      ❌ FAILING (GraphML quality issues)
└── TestGraphMLEnhancedE2E (1 comprehensive test)
    └── test_comprehensive_e2e_validation   ❌ FAILING (combines all validations)

CRITICAL BUGS EXPOSED:
1. 🚨 NODE DUPLICATION BUG (131 duplicate nodes)
   - Same nodes appear both in correct subgraphs AND incorrectly at root level
   - Example: `/embeddings/Constant` at line 63 (✅ correct) and line 1838 (❌ wrong)
   - Breaks GraphML semantics and visualization tools

2. 🚨 MISSING INPUT/OUTPUT METADATA
   - GraphML lacks graph-level input/output metadata (keys io0, io1)
   - ONNX graph.input and graph.output not preserved in GraphML representation

3. ⚠️ HIERARCHY TAG COMPLETENESS
   - Some operation nodes may lack hierarchy_tag attributes
   - Previous analysis showed 0 operation nodes detected (XML namespace issues)

VALIDATION STRATEGY:
- Model-agnostic: Works with any PyTorch model (BERT, ResNet, GPT, etc.)
- Evidence-based: Tests designed to FAIL and expose specific bugs
- Comprehensive: Covers XML compliance, node placement, hierarchy, I/O, round-trip
- Debugging-friendly: Detailed error messages and debug output

NEXT STEPS TO FIX IMPLEMENTATION:
1. Fix node duplication bug in GraphML generation logic
2. Implement proper input/output metadata generation (keys io0, io1)
3. Ensure all operation nodes receive hierarchy tags
4. Validate proper node placement in hierarchical subgraphs
5. Test round-trip conversion functionality
6. Re-run this test suite to verify all tests pass

EXPECTED OUTCOME AFTER FIXES:
All 8 individual tests + 1 E2E test should PASS, indicating:
✅ Structurally correct GraphML generation
✅ No duplicate nodes or hanging nodes
✅ Complete hierarchy tag coverage
✅ Preserved input/output metadata
✅ Successful round-trip conversion
✅ Model-agnostic functionality across architectures

=============================================================================
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
