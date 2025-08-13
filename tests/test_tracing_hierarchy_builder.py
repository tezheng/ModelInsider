"""Test cases for TracingHierarchyBuilder with exceptions parameter."""

from transformers import AutoModel

from modelexport.core.model_input_generator import generate_dummy_inputs
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder


class TestTracingHierarchyBuilder:
    """Test TracingHierarchyBuilder functionality."""

    def test_default_no_exceptions(self):
        """Test default behavior - torch.nn modules excluded per MUST-002."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")

        tracer = TracingHierarchyBuilder()
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()

        # MUST-002: torch.nn modules should NOT be included by default
        class_names = [
            info.get("class_name", "") for info in summary["module_hierarchy"].values()
        ]

        # These torch.nn modules should NOT be included
        assert "LayerNorm" not in class_names
        assert "Embedding" not in class_names
        assert "Linear" not in class_names
        assert "Dropout" not in class_names

        # Should only have HuggingFace modules
        assert len(summary["module_hierarchy"]) >= 15  # Typical HF module count
        assert len(summary["module_hierarchy"]) < 25  # Not including torch.nn modules

    def test_with_exceptions(self):
        """Test with exceptions - specified torch.nn modules are included."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")

        # With exceptions, specified torch.nn modules should be included
        tracer = TracingHierarchyBuilder(exceptions=["LayerNorm", "Embedding"])
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()

        # Count LayerNorm and Embedding modules
        layernorm_count = sum(
            1
            for info in summary["module_hierarchy"].values()
            if info.get("class_name") == "LayerNorm"
        )
        embedding_count = sum(
            1
            for info in summary["module_hierarchy"].values()
            if info.get("class_name") == "Embedding"
        )

        # BERT-tiny has 5 LayerNorm and 3 Embedding modules
        assert layernorm_count == 5
        assert embedding_count == 3

        # Linear and Dropout are NOT in the exceptions list, so should NOT be included
        linear_count = sum(
            1
            for info in summary["module_hierarchy"].values()
            if info.get("class_name") == "Linear"
        )
        dropout_count = sum(
            1
            for info in summary["module_hierarchy"].values()
            if info.get("class_name") == "Dropout"
        )

        # Should NOT find these modules (not in exceptions)
        assert linear_count == 0
        assert dropout_count == 0

    def test_hierarchy_count_difference(self):
        """Test hierarchy count difference with and without exceptions."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")

        # Without exceptions (default - exclude torch.nn)
        tracer1 = TracingHierarchyBuilder()
        tracer1.trace_model_execution(model, inputs)
        count_without = len(tracer1.get_execution_summary()["module_hierarchy"])

        # With exceptions (include specified torch.nn modules)
        tracer2 = TracingHierarchyBuilder(exceptions=["LayerNorm", "Embedding"])
        tracer2.trace_model_execution(model, inputs)
        count_with = len(tracer2.get_execution_summary()["module_hierarchy"])

        # With exceptions should have MORE modules
        assert count_with > count_without
        # Without exceptions should have only HF modules
        assert count_without < 25  # Only HF modules
        # With exceptions should include LayerNorm and Embedding
        assert count_with > count_without + 5  # At least 5 LayerNorm + 3 Embedding

    def test_resnet_with_torch_nn(self):
        """Test ResNet with torch.nn children included."""
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        inputs = generate_dummy_inputs("microsoft/resnet-50", exporter="onnx")

        # Test 1: Default behavior - torch.nn modules excluded
        tracer_default = TracingHierarchyBuilder()
        tracer_default.trace_model_execution(model, inputs)
        summary_default = tracer_default.get_execution_summary()

        # Should NOT have torch.nn modules by default
        class_names_default = [
            info.get("class_name", "")
            for info in summary_default["module_hierarchy"].values()
        ]
        assert "Conv2d" not in class_names_default
        assert "BatchNorm2d" not in class_names_default
        assert "ReLU" not in class_names_default
        assert "MaxPool2d" not in class_names_default

        # Test 2: With exceptions - include specified torch.nn modules
        exceptions = ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"]
        tracer = TracingHierarchyBuilder(exceptions=exceptions)
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()

        # Now these torch.nn modules should be included
        class_names = [
            info.get("class_name", "") for info in summary["module_hierarchy"].values()
        ]
        assert "Conv2d" in class_names
        assert "BatchNorm2d" in class_names

        # Check that ResNetConvLayer has Conv2d, BatchNorm2d children when exceptions are used
        conv_layer_found = False
        for path, info in summary["module_hierarchy"].items():
            if info.get("class_name") == "ResNetConvLayer":
                conv_layer_found = True
                # Look for its children
                conv2d_child = any(
                    p.startswith(path + ".")
                    and summary["module_hierarchy"][p].get("class_name") == "Conv2d"
                    for p in summary["module_hierarchy"]
                )
                assert conv2d_child, (
                    f"ResNetConvLayer at {path} should have Conv2d child"
                )
                break

        assert conv_layer_found, "Should find at least one ResNetConvLayer"

    def test_empty_exceptions(self):
        """Test with empty exceptions list - torch.nn modules still excluded per MUST-002."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs("prajjwal1/bert-tiny", exporter="onnx")

        # Empty list should behave like None - exclude torch.nn modules
        tracer = TracingHierarchyBuilder(exceptions=[])
        tracer.trace_model_execution(model, inputs)
        summary = tracer.get_execution_summary()

        # Check that torch.nn modules are NOT included (MUST-002)
        class_names = [
            info.get("class_name", "") for info in summary["module_hierarchy"].values()
        ]
        assert "LayerNorm" not in class_names
        assert "Embedding" not in class_names
        assert "Linear" not in class_names
        assert "Dropout" not in class_names

        # Should only have HuggingFace modules
        assert len(summary["module_hierarchy"]) < 25
