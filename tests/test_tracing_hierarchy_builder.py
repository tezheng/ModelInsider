"""Test cases for TracingHierarchyBuilder with exceptions parameter."""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel

from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder
from modelexport.core.model_input_generator import generate_dummy_inputs


class TestTracingHierarchyBuilder:
    """Test TracingHierarchyBuilder functionality."""
    
    def test_default_no_exceptions(self):
        """Test default behavior - no torch.nn modules in hierarchy."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        tracer = TracingHierarchyBuilder()
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # Check no torch.nn modules in hierarchy
        for info in summary["module_hierarchy"].values():
            class_name = info.get("class_name", "")
            # These are common torch.nn modules that should be filtered
            assert class_name not in ["LayerNorm", "Embedding", "Linear", "Dropout"]
    
    def test_with_exceptions(self):
        """Test with exceptions - specified torch.nn modules included."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # Include LayerNorm and Embedding in hierarchy
        tracer = TracingHierarchyBuilder(exceptions=["LayerNorm", "Embedding"])
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # Count LayerNorm and Embedding modules
        layernorm_count = sum(
            1 for info in summary["module_hierarchy"].values() 
            if info.get("class_name") == "LayerNorm"
        )
        embedding_count = sum(
            1 for info in summary["module_hierarchy"].values() 
            if info.get("class_name") == "Embedding"
        )
        
        # BERT-tiny has 5 LayerNorm and 3 Embedding modules
        assert layernorm_count == 5
        assert embedding_count == 3
        
        # Linear and Dropout should still be filtered
        for info in summary["module_hierarchy"].values():
            class_name = info.get("class_name", "")
            assert class_name not in ["Linear", "Dropout"]
    
    def test_hierarchy_count_difference(self):
        """Test that exceptions increase hierarchy module count."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # Without exceptions
        tracer1 = TracingHierarchyBuilder()
        tracer1.trace_model_execution(model, inputs)
        count_without = len(tracer1.get_execution_summary()["module_hierarchy"])
        
        # With exceptions
        tracer2 = TracingHierarchyBuilder(exceptions=["LayerNorm", "Embedding"])
        tracer2.trace_model_execution(model, inputs)
        count_with = len(tracer2.get_execution_summary()["module_hierarchy"])
        
        # Should have more modules with exceptions
        assert count_with > count_without
        # Specifically, 8 more (5 LayerNorm + 3 Embedding)
        assert count_with == count_without + 8
    
    def test_resnet_with_torch_nn(self):
        """Test ResNet with torch.nn children included."""
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        inputs = generate_dummy_inputs("microsoft/resnet-50", exporter="onnx")
        
        # Include common torch.nn modules used in ResNet
        exceptions = ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"]
        tracer = TracingHierarchyBuilder(exceptions=exceptions)
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # Check that ResNetConvLayer has Conv2d, BatchNorm2d children
        conv_layer_found = False
        for path, info in summary["module_hierarchy"].items():
            if info.get("class_name") == "ResNetConvLayer":
                conv_layer_found = True
                # Look for its children
                conv2d_child = any(
                    p.startswith(path + ".") and 
                    summary["module_hierarchy"][p].get("class_name") == "Conv2d"
                    for p in summary["module_hierarchy"]
                )
                assert conv2d_child, f"ResNetConvLayer at {path} should have Conv2d child"
                break
        
        assert conv_layer_found, "Should find at least one ResNetConvLayer"
    
    def test_empty_exceptions(self):
        """Test with empty exceptions list - same as default."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # Empty list should behave like default
        tracer = TracingHierarchyBuilder(exceptions=[])
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # Check no torch.nn modules
        for info in summary["module_hierarchy"].values():
            class_name = info.get("class_name", "")
            assert class_name not in ["LayerNorm", "Embedding", "Linear", "Dropout"]