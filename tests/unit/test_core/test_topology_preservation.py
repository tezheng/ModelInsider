"""
Test topology preservation between baseline and hierarchy exports.

Ensures our hierarchy exporter produces identical ONNX topology to standard torch.onnx.export.
"""

from collections import Counter
from pathlib import Path

import onnx
import pytest


class TestTopologyPreservation:
    """Test that hierarchy export preserves exact ONNX topology."""
    
    @pytest.fixture
    def baseline_model_path(self):
        """Path to baseline ONNX export."""
        return Path("temp/baseline_corrected.onnx")
    
    @pytest.fixture
    def hierarchy_model_path(self):
        """Path to hierarchy ONNX export.""" 
        return Path("temp/debug_full.onnx")
    
    @pytest.fixture
    def baseline_model(self, baseline_model_path):
        """Load baseline ONNX model."""
        if not baseline_model_path.exists():
            pytest.skip(f"Baseline model not found: {baseline_model_path}")
        return onnx.load(str(baseline_model_path))
    
    @pytest.fixture
    def hierarchy_model(self, hierarchy_model_path):
        """Load hierarchy ONNX model."""
        if not hierarchy_model_path.exists():
            pytest.skip(f"Hierarchy model not found: {hierarchy_model_path}")
        return onnx.load(str(hierarchy_model_path))
    
    def test_node_count_identical(self, baseline_model, hierarchy_model):
        """Test that both models have identical node counts."""
        baseline_count = len(baseline_model.graph.node)
        hierarchy_count = len(hierarchy_model.graph.node)
        
        assert baseline_count == hierarchy_count, (
            f"Node count mismatch: baseline={baseline_count}, hierarchy={hierarchy_count}"
        )
    
    def test_operation_types_identical(self, baseline_model, hierarchy_model):
        """Test that operation types and counts are identical."""
        baseline_ops = [node.op_type for node in baseline_model.graph.node]
        hierarchy_ops = [node.op_type for node in hierarchy_model.graph.node]
        
        baseline_counts = Counter(baseline_ops)
        hierarchy_counts = Counter(hierarchy_ops)
        
        # Check each operation type count
        all_ops = set(baseline_counts.keys()) | set(hierarchy_counts.keys())
        mismatches = []
        
        for op_type in all_ops:
            baseline_count = baseline_counts.get(op_type, 0)
            hierarchy_count = hierarchy_counts.get(op_type, 0)
            
            if baseline_count != hierarchy_count:
                mismatches.append(
                    f"{op_type}: baseline={baseline_count}, hierarchy={hierarchy_count}"
                )
        
        assert not mismatches, (
            f"Operation type count mismatches:\n" + "\n".join(mismatches)
        )
    
    def test_input_output_structure_identical(self, baseline_model, hierarchy_model):
        """Test that input/output structure is identical."""
        # Check inputs
        baseline_inputs = len(baseline_model.graph.input)
        hierarchy_inputs = len(hierarchy_model.graph.input)
        
        assert baseline_inputs == hierarchy_inputs, (
            f"Input count mismatch: baseline={baseline_inputs}, hierarchy={hierarchy_inputs}"
        )
        
        # Check outputs
        baseline_outputs = len(baseline_model.graph.output)
        hierarchy_outputs = len(hierarchy_model.graph.output)
        
        assert baseline_outputs == hierarchy_outputs, (
            f"Output count mismatch: baseline={baseline_outputs}, hierarchy={hierarchy_outputs}"
        )
    
    def test_initializer_count_identical(self, baseline_model, hierarchy_model):
        """Test that parameter counts are identical."""
        baseline_params = len(baseline_model.graph.initializer)
        hierarchy_params = len(hierarchy_model.graph.initializer)
        
        assert baseline_params == hierarchy_params, (
            f"Parameter count mismatch: baseline={baseline_params}, hierarchy={hierarchy_params}"
        )
    
    def test_node_structure_identical(self, baseline_model, hierarchy_model):
        """Test that individual node structures are identical."""
        baseline_nodes = baseline_model.graph.node
        hierarchy_nodes = hierarchy_model.graph.node
        
        assert len(baseline_nodes) == len(hierarchy_nodes), "Node count mismatch"
        
        mismatches = []
        
        for i, (baseline_node, hierarchy_node) in enumerate(zip(baseline_nodes, hierarchy_nodes, strict=False)):
            # Check operation type
            if baseline_node.op_type != hierarchy_node.op_type:
                mismatches.append(
                    f"Node {i}: op_type mismatch - {baseline_node.op_type} vs {hierarchy_node.op_type}"
                )
            
            # Check input count
            if len(baseline_node.input) != len(hierarchy_node.input):
                mismatches.append(
                    f"Node {i} ({baseline_node.op_type}): input count mismatch - "
                    f"{len(baseline_node.input)} vs {len(hierarchy_node.input)}"
                )
            
            # Check output count
            if len(baseline_node.output) != len(hierarchy_node.output):
                mismatches.append(
                    f"Node {i} ({baseline_node.op_type}): output count mismatch - "
                    f"{len(baseline_node.output)} vs {len(hierarchy_node.output)}"
                )
        
        assert not mismatches, (
            f"Node structure mismatches:\n" + "\n".join(mismatches[:10])  # Show first 10
        )
    
    def test_topology_summary(self, baseline_model, hierarchy_model):
        """Generate summary comparison for debugging."""
        baseline_summary = self._get_model_summary(baseline_model)
        hierarchy_summary = self._get_model_summary(hierarchy_model)
        
        print("\n📊 TOPOLOGY COMPARISON SUMMARY:")
        print(f"   Baseline:  {baseline_summary}")
        print(f"   Hierarchy: {hierarchy_summary}")
        
        # This test always passes but provides useful debug info
        assert True
    
    def _get_model_summary(self, model) -> dict:
        """Get model summary statistics."""
        nodes = model.graph.node
        op_types = [node.op_type for node in nodes]
        
        return {
            'total_nodes': len(nodes),
            'unique_op_types': len(set(op_types)),
            'inputs': len(model.graph.input),
            'outputs': len(model.graph.output),
            'initializers': len(model.graph.initializer),
            'most_common_ops': Counter(op_types).most_common(5)
        }


class TestTopologyPreservationDetailed:
    """Detailed topology preservation tests with specific checks."""
    
    def test_bert_specific_topology_preservation(self):
        """Test BERT-specific topology preservation."""
        baseline_path = Path("temp/baseline_export.onnx")
        hierarchy_path = Path("temp/bert_tiny_clean.onnx")
        
        if not baseline_path.exists() or not hierarchy_path.exists():
            pytest.skip("Required ONNX files not found")
        
        baseline_model = onnx.load(str(baseline_path))
        hierarchy_model = onnx.load(str(hierarchy_path))
        
        # BERT-specific checks
        baseline_ops = [node.op_type for node in baseline_model.graph.node]
        hierarchy_ops = [node.op_type for node in hierarchy_model.graph.node]
        
        # Key BERT operations should be preserved
        key_bert_ops = ['MatMul', 'Add', 'ReduceMean', 'Sqrt', 'Div', 'Gather', 'Reshape']
        
        for op in key_bert_ops:
            baseline_count = baseline_ops.count(op)
            hierarchy_count = hierarchy_ops.count(op)
            
            assert baseline_count == hierarchy_count, (
                f"BERT operation {op} count mismatch: "
                f"baseline={baseline_count}, hierarchy={hierarchy_count}"
            )
    
    def test_no_additional_operations_introduced(self, baseline_model, hierarchy_model):
        """Test that no new operations are introduced by hierarchy export."""
        baseline_ops = set(node.op_type for node in baseline_model.graph.node)
        hierarchy_ops = set(node.op_type for node in hierarchy_model.graph.node)
        
        additional_ops = hierarchy_ops - baseline_ops
        missing_ops = baseline_ops - hierarchy_ops
        
        assert not additional_ops, f"Additional operations introduced: {additional_ops}"
        assert not missing_ops, f"Operations missing: {missing_ops}"
    
    def test_tensor_flow_preservation(self, baseline_model, hierarchy_model):
        """Test that tensor flow structure is preserved."""
        # This is a more complex test that would verify tensor connections
        # For now, we'll do a basic check on tensor names and counts
        
        def get_all_tensors(model):
            tensors = set()
            for node in model.graph.node:
                tensors.update(node.input)
                tensors.update(node.output)
            return tensors
        
        baseline_tensors = get_all_tensors(baseline_model)
        hierarchy_tensors = get_all_tensors(hierarchy_model)
        
        # The tensor sets should be identical (names may differ but structure preserved)
        assert len(baseline_tensors) == len(hierarchy_tensors), (
            f"Tensor count mismatch: baseline={len(baseline_tensors)}, "
            f"hierarchy={len(hierarchy_tensors)}"
        )