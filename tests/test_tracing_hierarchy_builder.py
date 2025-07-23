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
        """Test default behavior - ALL modules included in hierarchy per TEZ-24 fix."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        tracer = TracingHierarchyBuilder()
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # TEZ-24 Fix: ALL modules are now included for complete hierarchy reports
        # Check that torch.nn modules ARE included
        class_names = [info.get("class_name", "") for info in summary["module_hierarchy"].values()]
        
        # These torch.nn modules should now be included for complete reports
        assert "LayerNorm" in class_names
        assert "Embedding" in class_names
        assert "Linear" in class_names
        assert "Dropout" in class_names
        
        # Should have significantly more modules than before
        assert len(summary["module_hierarchy"]) > 25  # Was ~18 before fix
    
    def test_with_exceptions(self):
        """Test with exceptions - ALL modules included per TEZ-24 fix, exceptions param ignored."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # TEZ-24 Fix: exceptions parameter is now ignored, ALL modules included
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
        
        # TEZ-24 Fix: Linear and Dropout are now also included for complete reports
        linear_count = sum(
            1 for info in summary["module_hierarchy"].values() 
            if info.get("class_name") == "Linear"
        )
        dropout_count = sum(
            1 for info in summary["module_hierarchy"].values() 
            if info.get("class_name") == "Dropout"
        )
        
        # Should find these modules as well
        assert linear_count > 0
        assert dropout_count > 0
    
    def test_hierarchy_count_difference(self):
        """Test hierarchy count - ALL modules included per TEZ-24 fix."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # Without exceptions
        tracer1 = TracingHierarchyBuilder()
        tracer1.trace_model_execution(model, inputs)
        count_without = len(tracer1.get_execution_summary()["module_hierarchy"])
        
        # With exceptions (TEZ-24: exceptions param ignored, same result)
        tracer2 = TracingHierarchyBuilder(exceptions=["LayerNorm", "Embedding"])
        tracer2.trace_model_execution(model, inputs)
        count_with = len(tracer2.get_execution_summary()["module_hierarchy"])
        
        # TEZ-24 Fix: Both should have same count since ALL modules are included
        assert count_with == count_without
        # Both should have complete hierarchy
        assert count_without > 25  # Complete hierarchy includes all modules
    
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
        """Test with empty exceptions list - ALL modules included per TEZ-24 fix."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")
        
        # TEZ-24 Fix: Empty list behavior is same as any other - ALL modules included
        tracer = TracingHierarchyBuilder(exceptions=[])
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()
        
        # Check that torch.nn modules ARE included
        class_names = [info.get("class_name", "") for info in summary["module_hierarchy"].values()]
        assert "LayerNorm" in class_names
        assert "Embedding" in class_names
        assert "Linear" in class_names
        assert "Dropout" in class_names
        
        # Should have complete hierarchy
        assert len(summary["module_hierarchy"]) > 25