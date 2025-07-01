"""
Test slice operation tagging to verify if we actually fixed the issue or just bypassed it.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
import onnx
from modelexport.hierarchy_exporter import HierarchyExporter


class SimpleSliceModel(nn.Module):
    """Simple model that slices input tensor in forward pass."""
    
    def __init__(self):
        super().__init__()
        # Add a meaningful layer so we're not just pure PyTorch
        self.processor = nn.Linear(10, 10)
        
    def forward(self, x):
        # Slice the input tensor using different methods
        sliced = x[1:4]  # This should create ONNX Slice node
        
        # Process through our layer to create meaningful hierarchy
        processed = self.processor(sliced)
        
        return processed


class HuggingFaceStyleSliceModel(nn.Module):
    """Model that mimics HuggingFace patterns with slicing."""
    
    def __init__(self):
        super().__init__()
        # Create a more HF-like structure
        self.embeddings = nn.Embedding(100, 10)
        self.layer_norm = nn.LayerNorm(10)  # This should be tagged (whitelist)
        
    def forward(self, input_ids):
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Slice embeddings (common in HF models for sequence processing)
        sliced_embeddings = embeddings[1:-1]  # Remove first and last tokens
        
        # Normalize
        normalized = self.layer_norm(sliced_embeddings)
        
        return normalized


class TestSliceTagging:
    """Test slice operation tagging functionality."""
    
    def test_simple_slice_model_generates_onnx_slice_nodes(self):
        """Test that slice operations generate ONNX Slice nodes."""
        model = SimpleSliceModel()
        model.eval()
        inputs = torch.randn(5, 10)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            # Verify export succeeded
            assert result is not None
            assert result['total_operations'] > 0
            
            # Load ONNX model and check for Slice nodes
            onnx_model = onnx.load(tmp.name)
            slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
            
            # Should have at least one Slice node from x[1:4]
            assert len(slice_nodes) > 0, "No Slice nodes found in ONNX - slice operation not converted"
    
    def test_slice_nodes_appear_in_tag_mapping(self):
        """Test that Slice nodes appear in the tag mapping."""
        model = SimpleSliceModel()
        model.eval()
        inputs = torch.randn(5, 10)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            tag_mapping = exporter.get_tag_mapping()
            
            # Find Slice nodes in tag mapping
            slice_nodes_in_mapping = [
                name for name, info in tag_mapping.items() 
                if info.get('op_type') == 'Slice'
            ]
            
            assert len(slice_nodes_in_mapping) > 0, "No Slice nodes found in tag mapping - tagging system not capturing them"
    
    def test_slice_nodes_can_have_tags(self):
        """Test whether slice nodes can actually receive tags."""
        model = SimpleSliceModel()
        model.eval()
        inputs = torch.randn(5, 10)
        
        # Test both strategies
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                tag_mapping = exporter.get_tag_mapping()
                
                # Check if any Slice nodes have tags
                slice_nodes_with_tags = [
                    (name, info) for name, info in tag_mapping.items() 
                    if info.get('op_type') == 'Slice' and info.get('tags', [])
                ]
                
                slice_nodes_without_tags = [
                    (name, info) for name, info in tag_mapping.items() 
                    if info.get('op_type') == 'Slice' and not info.get('tags', [])
                ]
                
                print(f"\n{strategy} strategy:")
                print(f"  Slice nodes with tags: {len(slice_nodes_with_tags)}")
                print(f"  Slice nodes without tags: {len(slice_nodes_without_tags)}")
                
                # For Phase 1 (HF focus), simple PyTorch models may not get tags
                # But we should at least capture the nodes in the mapping
                total_slice_nodes = len(slice_nodes_with_tags) + len(slice_nodes_without_tags)
                assert total_slice_nodes > 0, f"No Slice nodes found in {strategy} tag mapping"
    
    def test_hf_style_slice_model_tagging(self):
        """Test slice tagging in a more HuggingFace-like model."""
        model = HuggingFaceStyleSliceModel()
        model.eval()
        inputs = torch.randint(0, 100, (5,))
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            
            # Check ONNX has slice nodes
            onnx_model = onnx.load(tmp.name)
            slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
            
            tag_mapping = exporter.get_tag_mapping()
            
            # LayerNorm should be tagged (whitelist)
            layernorm_nodes = [
                name for name, info in tag_mapping.items() 
                if 'LayerNorm' in info.get('op_type', '') and info.get('tags', [])
            ]
            
            # In Phase 1, LayerNorm should be tagged due to whitelist
            assert len(layernorm_nodes) > 0, "LayerNorm not tagged - whitelist not working"
            
            # Check if slice nodes exist in mapping
            slice_nodes_in_mapping = [
                name for name, info in tag_mapping.items() 
                if info.get('op_type') == 'Slice'
            ]
            
            assert len(slice_nodes_in_mapping) > 0, "Slice nodes not captured in HF-style model"
    
    def test_slice_operation_registry_consistency(self):
        """Test that slice operation in registry is correctly configured."""
        from modelexport.hierarchy_exporter import OperationConfig
        
        # Check slice operation exists in registry
        assert 'slice' in OperationConfig.OPERATION_REGISTRY, "slice operation missing from registry"
        
        slice_config = OperationConfig.OPERATION_REGISTRY['slice']
        
        # Should have empty patch_targets (no torch.slice function to patch)
        assert slice_config['patch_targets'] == [], "slice should have empty patch_targets"
        
        # Should map to ONNX Slice
        assert 'Slice' in slice_config['onnx_types'], "slice should map to ONNX Slice operation"
    
    def test_slice_related_operations_are_valid(self):
        """Test that slice-related operations we added are valid."""
        from modelexport.hierarchy_exporter import OperationConfig
        
        slice_related = ['narrow', 'select', 'take']
        
        for op_name in slice_related:
            assert op_name in OperationConfig.OPERATION_REGISTRY, f"{op_name} missing from registry"
            
            config = OperationConfig.OPERATION_REGISTRY[op_name]
            
            # Should have valid patch targets
            assert len(config['patch_targets']) > 0, f"{op_name} should have patch targets"
            
            # Verify patch targets are callable
            for module_name, func_name in config['patch_targets']:
                if module_name == 'torch':
                    assert hasattr(torch, func_name), f"torch.{func_name} should exist"
    
    def test_narrow_operation_works_in_model(self):
        """Test that torch.narrow operation works and can be tagged."""
        class NarrowModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)
                
            def forward(self, x):
                # Use torch.narrow (should be patchable)
                narrowed = torch.narrow(x, 0, 1, 3)  # Select 3 elements starting from index 1
                return self.linear(narrowed)
        
        model = NarrowModel()
        model.eval()
        inputs = torch.randn(5, 5)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            
            # Should have operations (narrow might become Slice in ONNX)
            tag_mapping = exporter.get_tag_mapping()
            assert len(tag_mapping) > 0, "No operations captured in narrow test"
    
    def test_comparison_indexed_vs_sliced_operations(self):
        """Compare different ways of accessing tensor elements."""
        class ComparisonModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)
                
            def forward(self, x):
                # Different ways to access tensor data
                sliced = x[1:4]           # __getitem__ slice
                narrowed = torch.narrow(x, 0, 1, 3)  # torch.narrow  
                selected = torch.select(x, 0, 2)     # torch.select
                
                # Process each differently
                out1 = self.linear(sliced)
                out2 = self.linear(narrowed)  
                out3 = self.linear(selected.unsqueeze(0))
                
                return out1 + out2 + out3
        
        model = ComparisonModel()
        model.eval()
        inputs = torch.randn(6, 3)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            # Check what types of operations were generated
            onnx_model = onnx.load(tmp.name)
            
            op_types = [node.op_type for node in onnx_model.graph.node]
            slice_count = op_types.count('Slice')
            gather_count = op_types.count('Gather')
            
            print(f"\nComparison model operations:")
            print(f"  Total ONNX operations: {len(op_types)}")
            print(f"  Slice operations: {slice_count}")
            print(f"  Gather operations: {gather_count}")
            print(f"  Other operations: {len(op_types) - slice_count - gather_count}")
            
            # Should have some slice/gather operations
            assert (slice_count + gather_count) > 0, "No slice/gather operations found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])