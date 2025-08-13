"""
Real-world scenario and corner case tests from QA/PM perspective.

This test suite covers production scenarios, stress testing, edge cases,
and user experience considerations for GraphML conversion.
"""

import concurrent.futures
import json
import os
import time
import xml.etree.ElementTree as ET

import onnx
import torch
import torch.nn as nn

from modelexport.graphml import ONNXToGraphMLConverter


class TestProductionScenarios:
    """Test cases for production usage scenarios."""

    def test_transformer_model_conversion(self, tmp_path):
        """Test conversion of transformer-based models (common in production)."""

        # Create a simplified transformer model
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256),
                    num_layers=num_layers,
                )
                self.output = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                # x shape: (batch, seq_len)
                x = self.embedding(x)
                x = x + self.pos_encoding[:, : x.size(1), :]
                x = self.transformer(x)
                return self.output(x)

        model = SimpleTransformer()
        dummy_input = torch.randint(0, 1000, (1, 64))  # batch=1, seq_len=64
        onnx_path = tmp_path / "transformer.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
        )

        # Convert to GraphML
        converter = ONNXToGraphMLConverter(hierarchical=False)
        start_time = time.time()
        graphml_str = converter.convert(str(onnx_path))
        conversion_time = time.time() - start_time

        # Verify conversion succeeded
        assert graphml_str is not None
        assert len(graphml_str) > 1000  # Should be substantial output
        assert conversion_time < 30.0, f"Conversion too slow: {conversion_time}s"

        # Check for transformer-specific operations
        assert "MatMul" in graphml_str  # Attention operations
        assert "Softmax" in graphml_str  # Attention softmax
        assert "Add" in graphml_str  # Residual connections

    def test_vision_model_conversion(self, tmp_path):
        """Test conversion of vision models (ResNet-like architecture)."""

        # Create a simplified ResNet block
        class ResNetBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

                self.downsample = None
                if stride != 1 or in_channels != out_channels:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels),
                    )

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)
                return out

        class SimpleResNet(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, 2, 1)

                self.layer1 = ResNetBlock(64, 64)
                self.layer2 = ResNetBlock(64, 128, stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        model = SimpleResNet()
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = tmp_path / "resnet.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["image"],
            output_names=["predictions"],
            dynamic_axes={"image": {0: "batch"}},
        )

        # Convert and verify
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))

        # Check for CNN operations
        assert "Conv" in graphml_str
        # BatchNorm might be fused or have different names in ONNX
        assert (
            "BatchNormalization" in graphml_str
            or "BatchNorm" in graphml_str
            or "Add" in graphml_str
        )
        assert "MaxPool" in graphml_str
        assert "GlobalAveragePool" in graphml_str or "ReduceMean" in graphml_str

    def test_batch_conversion_workflow(self, tmp_path):
        """Test batch conversion of multiple models (common in MLOps)."""
        models = {
            "small": nn.Linear(10, 5),
            "medium": nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 50)),
            "large": nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
            ),
        }

        converter = ONNXToGraphMLConverter(hierarchical=False)
        results = {}
        total_start = time.time()

        for name, model in models.items():
            # Export to ONNX
            input_size = 10 if name == "small" else (50 if name == "medium" else 100)
            dummy_input = torch.randn(1, input_size)
            onnx_path = tmp_path / f"{name}.onnx"

            torch.onnx.export(model, dummy_input, str(onnx_path))

            # Convert to GraphML
            start = time.time()
            output_path = tmp_path / f"{name}.graphml"
            converter.save(str(onnx_path), str(output_path))
            duration = time.time() - start

            results[name] = {"duration": duration, "size": output_path.stat().st_size}

        total_duration = time.time() - total_start

        # Verify batch processing
        assert len(results) == 3
        assert all(r["size"] > 0 for r in results.values())
        assert total_duration < 10.0, f"Batch conversion too slow: {total_duration}s"

        # Check relative sizes make sense
        assert results["large"]["size"] > results["medium"]["size"]
        assert results["medium"]["size"] > results["small"]["size"]

    def test_model_with_custom_metadata_workflow(self, tmp_path):
        """Test production workflow with custom metadata injection."""
        # Create model
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "model_with_metadata.onnx"

        # Export with metadata
        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Load and inject production metadata
        onnx_model = onnx.load(str(onnx_path))

        # Add model-level metadata (common in production)
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="model_version", value="1.2.3")
        )
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="training_date", value="2024-01-15")
        )
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="framework", value="pytorch-2.0.1")
        )

        # Add node-level metadata for traceability
        for i, node in enumerate(onnx_model.graph.node):
            # Add layer mapping info
            attr = onnx.AttributeProto()
            attr.name = "original_layer_name"
            attr.type = onnx.AttributeProto.STRING
            attr.s = f"model.layer_{i}".encode()
            node.attribute.append(attr)

        onnx.save(onnx_model, str(onnx_path))

        # Convert and verify metadata preserved
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))

        # Node-level custom attributes are included in onnx_attributes
        # In v1.3, all ONNX attributes are preserved in the JSON
        assert "original_layer_name" in graphml_str

        # But basic GraphML structure should be valid
        root = ET.fromstring(graphml_str)
        assert root is not None

        # And we should have the standard metadata - v1.3 uses meta0
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        source_file = root.find(".//data[@key='meta0']", ns)
        assert source_file is not None
        assert source_file.text == "model_with_metadata.onnx"


class TestStressAndScale:
    """Stress tests for scalability and performance."""

    def test_very_large_model(self, tmp_path):
        """Test conversion of very large models (memory and performance)."""
        # Create a large model with many layers
        layers = []
        for _i in range(50):  # 50 layer pairs = 100 layers total
            layers.extend([nn.Linear(256, 256), nn.ReLU()])

        large_model = nn.Sequential(*layers)
        dummy_input = torch.randn(1, 256)
        onnx_path = tmp_path / "very_large.onnx"

        # Export (this itself might take time)
        torch.onnx.export(
            large_model,
            dummy_input,
            str(onnx_path),
            do_constant_folding=False,  # Keep all ops
        )

        # Monitor conversion
        converter = ONNXToGraphMLConverter(hierarchical=False)
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        graphml_str = converter.convert(str(onnx_path))

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        duration = end_time - start_time
        memory_increase = end_memory - start_memory

        # Should complete in reasonable time
        assert duration < 60.0, f"Conversion of large model too slow: {duration}s"
        assert graphml_str is not None
        assert len(graphml_str) > 10000  # Should be very large output

        # Check node count
        root = ET.fromstring(graphml_str)
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//node", ns)
        assert len(nodes) >= 100, f"Expected many nodes, got {len(nodes)}"

    def test_wide_model(self, tmp_path):
        """Test model with very wide layers (many parameters)."""
        # Model with very wide layers
        model = nn.Sequential(
            nn.Linear(10000, 5000),  # 50M parameters
            nn.ReLU(),
            nn.Linear(5000, 1000),  # 5M parameters
            nn.ReLU(),
            nn.Linear(1000, 10),  # 10K parameters
        )

        dummy_input = torch.randn(1, 10000)
        onnx_path = tmp_path / "wide_model.onnx"

        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Convert with initializer exclusion (default)
        converter = ONNXToGraphMLConverter(
            hierarchical=False, exclude_initializers=True
        )
        graphml_str = converter.convert(str(onnx_path))

        # Should handle wide models efficiently
        assert graphml_str is not None

        # GraphML should not be too large (initializers excluded)
        assert len(graphml_str) < 1000000, (
            "GraphML too large, might include initializers"
        )

    def test_deeply_nested_model(self, tmp_path):
        """Test model with very deep nesting (stress hierarchical converter)."""

        # Create deeply nested model
        def create_nested_module(depth, width=32):
            if depth == 0:
                return nn.Linear(width, width)
            else:
                return nn.Sequential(
                    nn.Linear(width, width),
                    create_nested_module(depth - 1, width),
                    nn.ReLU(),
                )

        deep_model = create_nested_module(depth=15)  # 15 levels deep
        dummy_input = torch.randn(1, 32)
        onnx_path = tmp_path / "deep_nested.onnx"

        torch.onnx.export(deep_model, dummy_input, str(onnx_path))

        # Create mock HTP metadata for deeply nested structure
        htp_metadata = {"modules": {}, "node_mappings": {}}

        def add_module_hierarchy(path, depth):
            htp_metadata["modules"][path] = {
                "class_name": "Sequential" if depth > 0 else "Linear",
                "execution_order": len(htp_metadata["modules"]),
            }
            if depth > 0:
                add_module_hierarchy(f"{path}/0", 0)  # Linear
                add_module_hierarchy(f"{path}/1", depth - 1)  # Nested
                add_module_hierarchy(f"{path}/2", 0)  # ReLU

        add_module_hierarchy("/Model", 15)

        htp_path = tmp_path / "deep_htp.json"
        htp_path.write_text(json.dumps(htp_metadata))

        # Convert with hierarchical converter
        converter = ONNXToGraphMLConverter(htp_metadata_path=str(htp_path))
        start_time = time.time()
        graphml_str = converter.convert(str(onnx_path))
        duration = time.time() - start_time

        # Should handle deep nesting
        assert graphml_str is not None
        assert duration < 30.0, f"Deep nesting conversion too slow: {duration}s"

    def test_concurrent_conversions(self, tmp_path):
        """Test thread safety with many concurrent conversions."""
        # Create multiple different models
        models = []
        for i in range(10):
            model = nn.Sequential(
                nn.Linear(20 + i, 30 + i), nn.ReLU(), nn.Linear(30 + i, 10)
            )
            models.append((f"model_{i}", model, 20 + i))

        # Export all to ONNX
        onnx_paths = []
        for name, model, input_size in models:
            dummy_input = torch.randn(1, input_size)
            onnx_path = tmp_path / f"{name}.onnx"
            torch.onnx.export(model, dummy_input, str(onnx_path))
            onnx_paths.append((name, str(onnx_path)))

        # Convert concurrently
        converter = ONNXToGraphMLConverter(hierarchical=False)

        def convert_model(args):
            name, onnx_path = args
            try:
                graphml_str = converter.convert(onnx_path)
                return name, len(graphml_str), None
            except Exception as e:
                return name, 0, str(e)

        # Use thread pool for concurrent conversion
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(convert_model, path_info) for path_info in onnx_paths
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all succeeded
        for name, length, error in results:
            assert error is None, f"Model {name} failed: {error}"
            assert length > 0, f"Model {name} produced empty output"

        assert len(results) == 10, "Not all models converted"


class TestCornerCases:
    """Tests for corner cases and unusual scenarios."""

    def test_model_with_no_operations(self, tmp_path):
        """Test model that's just an identity (no operations)."""

        class IdentityModel(nn.Module):
            def forward(self, x):
                return x

        model = IdentityModel()
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "identity.onnx"

        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Should handle models with minimal operations
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None

        # Should have at least input/output nodes
        root = ET.fromstring(graphml_str)
        ns = {"": "http://graphml.graphdrawing.org/xmlns"}
        nodes = root.findall(".//node", ns)
        assert len(nodes) >= 2  # At least input and output

    def test_model_with_loops(self, tmp_path):
        """Test model with loop operations (RNN-like)."""

        class LoopModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x, steps=3):
                # Simple loop (gets unrolled in ONNX or uses Loop op)
                for _ in range(steps):
                    x = self.linear(x)
                    x = torch.relu(x)
                return x

        model = LoopModel()
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "loop_model.onnx"

        # Export with static loop unrolling
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
        )

        # Should handle loop structures
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None

        # Should have multiple matrix operations (unrolled loop)
        # Linear layers might be exported as Gemm or MatMul
        matmul_count = graphml_str.count("MatMul") + graphml_str.count("Gemm")
        assert matmul_count >= 3, f"Expected multiple matrix ops, got {matmul_count}"

    def test_model_with_conditional(self, tmp_path):
        """Test model with conditional operations."""

        class ConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x):
                # Use operations that create conditionals in ONNX
                mask = x > 0
                positive = self.linear1(x)
                negative = self.linear2(x)
                return torch.where(mask, positive, negative)

        model = ConditionalModel()
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "conditional.onnx"

        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Should handle conditional operations
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None

        # Should have Where operation
        assert "Where" in graphml_str

    def test_model_with_dynamic_control_flow(self, tmp_path):
        """Test model with dynamic shapes and control flow."""

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 20, batch_first=True)
                self.linear = nn.Linear(20, 5)

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last output
                last_output = lstm_out[:, -1, :]
                return self.linear(last_output)

        model = DynamicModel()
        dummy_input = torch.randn(2, 15, 10)  # batch=2, seq=15, features=10
        onnx_path = tmp_path / "dynamic_lstm.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch", 1: "sequence"}, "output": {0: "batch"}},
        )

        # Should handle LSTM and dynamic shapes
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))
        assert graphml_str is not None

        # Should have LSTM operations
        assert "LSTM" in graphml_str or "Scan" in graphml_str

    def test_corrupted_metadata_handling(self, tmp_path):
        """Test handling of corrupted or malformed metadata."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "corrupted.onnx"

        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Load and add corrupted metadata
        onnx_model = onnx.load(str(onnx_path))

        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]

            # Add attribute with invalid UTF-8
            attr = onnx.AttributeProto()
            attr.name = "corrupted_string"
            attr.type = onnx.AttributeProto.STRING
            attr.s = b"\x80\x81\x82\x83"  # Invalid UTF-8
            node.attribute.append(attr)

            # Add attribute with wrong type annotation
            attr2 = onnx.AttributeProto()
            attr2.name = "wrong_type"
            attr2.type = onnx.AttributeProto.FLOAT
            # But set string value (mismatch)
            attr2.s = b"not_a_float"
            node.attribute.append(attr2)

        onnx.save(onnx_model, str(onnx_path))

        # Should handle corrupted data gracefully
        converter = ONNXToGraphMLConverter(hierarchical=False)
        try:
            graphml_str = converter.convert(str(onnx_path))
            # If it succeeds, verify output is valid
            assert graphml_str is not None
            root = ET.fromstring(graphml_str)  # Should be valid XML
        except Exception as e:
            # If it fails, should be a clear error
            assert "UTF-8" in str(e) or "decode" in str(e) or "attribute" in str(e)


class TestUserExperience:
    """Tests focused on user experience and usability."""

    def test_helpful_error_messages(self, tmp_path):
        """Test that error messages are helpful for users."""
        converter = ONNXToGraphMLConverter(hierarchical=False)

        # Test various error scenarios
        error_cases = [
            ("non_existent.onnx", FileNotFoundError, "not found"),
            (str(tmp_path / "empty.onnx"), Exception, ""),  # Empty file
            (str(tmp_path), Exception, ""),  # Directory instead of file
        ]

        # Create empty file
        (tmp_path / "empty.onnx").write_text("")

        for path, expected_error, expected_message in error_cases:
            try:
                converter.convert(path)
                raise AssertionError(f"Expected error for {path}")
            except expected_error as e:
                error_msg = str(e)
                assert error_msg, "Empty error message"
                if expected_message:
                    assert expected_message in error_msg.lower(), (
                        f"Error message '{error_msg}' not helpful for {path}"
                    )

    def test_progress_indication_for_large_models(self, tmp_path):
        """Test that large model conversion provides progress feedback."""
        # Create a large model
        layers = []
        for _i in range(30):
            layers.extend([nn.Linear(128, 128), nn.ReLU()])

        large_model = nn.Sequential(*layers)
        dummy_input = torch.randn(1, 128)
        onnx_path = tmp_path / "large_progress.onnx"

        torch.onnx.export(large_model, dummy_input, str(onnx_path))

        # Convert and check statistics
        converter = ONNXToGraphMLConverter(hierarchical=False)
        graphml_str = converter.convert(str(onnx_path))

        # Check that statistics are available
        stats = converter.get_statistics()
        assert "nodes" in stats
        assert "edges" in stats
        assert stats["nodes"] > 50  # Should have many nodes

    def test_output_file_safety(self, tmp_path):
        """Test safe handling of output file operations."""
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        onnx_path = tmp_path / "test.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))

        converter = ONNXToGraphMLConverter(hierarchical=False)

        # Test overwriting existing file
        output_path = tmp_path / "output.graphml"
        output_path.write_text("existing content")

        # Should overwrite without error
        converter.save(str(onnx_path), str(output_path))
        assert output_path.read_text() != "existing content"

        # Test creating nested directories
        nested_output = tmp_path / "deep" / "nested" / "dir" / "output.graphml"
        converter.save(str(onnx_path), str(nested_output))
        assert nested_output.exists()

        # Test handling read-only directory (skip on Windows)
        if os.name != "nt":
            readonly_dir = tmp_path / "readonly"
            readonly_dir.mkdir()
            os.chmod(readonly_dir, 0o444)

            try:
                converter.save(str(onnx_path), str(readonly_dir / "output.graphml"))
                raise AssertionError("Expected permission error")
            except (PermissionError, OSError):
                pass  # Expected
            finally:
                os.chmod(readonly_dir, 0o755)  # Restore permissions

    def test_internationalization_support(self, tmp_path):
        """Test support for international characters in paths and metadata."""
        # Model with unicode in various places
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)

        # Unicode in filename
        onnx_path = tmp_path / "模型_mô_hình_モデル.onnx"
        torch.onnx.export(model, dummy_input, str(onnx_path))

        # Add unicode metadata
        onnx_model = onnx.load(str(onnx_path))

        # Unicode in metadata
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="作者", value="测试用户")
        )
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key="説明", value="テストモデル")
        )

        # Unicode in node attributes
        if onnx_model.graph.node:
            node = onnx_model.graph.node[0]
            attr = onnx.AttributeProto()
            attr.name = "layer_name_中文"
            attr.type = onnx.AttributeProto.STRING
            attr.s = "第一层_พระชั้น_レイヤー".encode()
            node.attribute.append(attr)

        onnx.save(onnx_model, str(onnx_path))

        # Convert with unicode handling
        converter = ONNXToGraphMLConverter(hierarchical=False)
        output_path = tmp_path / "输出_output_出力.graphml"
        converter.save(str(onnx_path), str(output_path))

        # Verify unicode preserved in filename
        graphml_content = output_path.read_text(encoding="utf-8")
        # The Unicode filename should be preserved
        assert "模型_mô_hình_モデル.onnx" in graphml_content

        # Parse to verify valid XML with unicode
        root = ET.fromstring(graphml_content)
        assert root is not None

        # Note: Model metadata_props and custom node attributes are not currently
        # extracted by the parser, but unicode in filenames is preserved
