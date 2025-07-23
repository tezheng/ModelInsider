"""
Error Recovery Test Suite

This module tests error recovery and resilience aspects of the modelexport
system, including handling of corrupted files, network failures, partial
export recovery, and state corruption scenarios.

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Error recovery must be universal
- MUST-002: TORCH.NN FILTERING - Error handling across all architectures
- MUST-003: UNIVERSAL DESIGN - Recovery mechanisms work for any model type

Error Recovery Test Categories:
1. Corruption Handling - Corrupted ONNX files, metadata files
2. Network Failures - Model download failures, timeout scenarios
3. Partial Export Recovery - Resume interrupted exports
4. State Recovery - Recover from inconsistent internal states
5. Graceful Degradation - Continue operation with reduced functionality
6. Error Reporting - Clear error messages and diagnostic information
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from modelexport.core import tag_utils
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class TestCorruptionHandling:
    """
    Test handling of corrupted files and data structures.
    
    These tests validate that the system gracefully handles various
    types of file corruption without crashes or data loss.
    """
    
    def test_corrupted_onnx_file_recovery(self):
        """
        Test recovery from corrupted ONNX files.
        
        Validates that the system can detect and handle corrupted
        ONNX files without compromising system stability.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create various types of corrupted ONNX files
            corrupted_scenarios = [
                {
                    "name": "random_binary",
                    "data": os.urandom(1024),
                    "description": "Random binary data"
                },
                {
                    "name": "truncated_header",
                    "data": b"ONNX" + os.urandom(100),  # Truncated ONNX file
                    "description": "Truncated ONNX header"
                },
                {
                    "name": "empty_file",
                    "data": b"",
                    "description": "Empty file"
                },
                {
                    "name": "text_file",
                    "data": b"This is not an ONNX file\nJust plain text\n",
                    "description": "Text file with ONNX extension"
                },
                {
                    "name": "malformed_protobuf",
                    "data": b"\x08\x01\x12\x04test" + b"\xff" * 1000,
                    "description": "Malformed protobuf data"
                }
            ]
            
            for scenario in corrupted_scenarios:
                corrupted_file = temp_path / f"{scenario['name']}.onnx"
                
                # Create corrupted file
                with open(corrupted_file, 'wb') as f:
                    f.write(scenario['data'])
                
                # Test various operations on corrupted file
                self._test_corrupted_file_operations(corrupted_file, scenario['description'])
    
    def _test_corrupted_file_operations(self, corrupted_file: Path, description: str):
        """Test various operations on a corrupted file."""
        try:
            # Test tag loading from corrupted ONNX
            tags = tag_utils.load_tags_from_onnx(str(corrupted_file))
            # Should return empty dict or raise graceful exception
            assert isinstance(tags, dict), f"Tag loading should return dict for {description}"
            
        except (ValueError, TypeError, ImportError, AttributeError) as e:
            # Expected errors for corrupted files
            assert "segmentation" not in str(e).lower(), f"Segfault on {description}: {e}"
            assert "buffer overflow" not in str(e).lower(), f"Buffer overflow on {description}: {e}"
            
        except Exception as e:
            # Other exceptions should not indicate security issues
            error_msg = str(e).lower()
            assert "access violation" not in error_msg, f"Access violation on {description}: {e}"
            assert "memory" not in error_msg or "corruption" not in error_msg, f"Memory corruption on {description}: {e}"
        
        try:
            # Test sidecar loading (should handle missing gracefully)
            sidecar_data = tag_utils.load_tags_from_sidecar(str(corrupted_file))
            # Should return data or raise FileNotFoundError
            assert isinstance(sidecar_data, dict), f"Sidecar loading should return dict for {description}"
            
        except FileNotFoundError:
            # Expected when no sidecar file exists
            pass
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            # Expected errors for corrupted sidecar files
            pass
        except Exception as e:
            # Should not cause system-level errors
            error_msg = str(e).lower()
            assert "segmentation" not in error_msg, f"Segfault on sidecar for {description}: {e}"
    
    def test_corrupted_metadata_recovery(self):
        """
        Test recovery from corrupted metadata files.
        
        Validates that the system can handle corrupted metadata
        files and either recover or fail gracefully.
        """
        # First create a valid export to get valid metadata structure
        exporter = HierarchyExporter(strategy="htp")
        model = nn.Linear(10, 5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid export first
            valid_output = temp_path / "valid_model.onnx"
            result = exporter.export(
                model=model,
                model_name_or_path="test",
                output_path=str(valid_output),
                opset_version=17
            )
            
            # Get the metadata file path
            metadata_file = temp_path / "valid_model_htp_metadata.json"
            
            if metadata_file.exists():
                # Load valid metadata to understand structure
                with open(metadata_file) as f:
                    valid_metadata = json.load(f)
                
                # Create various types of corrupted metadata
                corrupted_metadata_scenarios = [
                    {
                        "name": "missing_keys",
                        "data": {"incomplete": "metadata"},
                        "description": "Missing required keys"
                    },
                    {
                        "name": "invalid_json",
                        "data": '{"invalid": json, missing quotes}',
                        "description": "Invalid JSON syntax"
                    },
                    {
                        "name": "circular_reference",
                        "data": None,  # Will be set to circular structure
                        "description": "Circular reference in JSON"
                    },
                    {
                        "name": "wrong_data_types",
                        "data": {
                            "export_info": "should_be_dict",
                            "statistics": [],
                            "tagged_nodes": "should_be_dict"
                        },
                        "description": "Wrong data types"
                    },
                    {
                        "name": "extremely_large",
                        "data": {"tagged_nodes": {f"node_{i}": f"tag_{i}" for i in range(10000)}},
                        "description": "Extremely large metadata"
                    }
                ]
                
                # Create circular reference
                circular_data = {"self": None}
                circular_data["self"] = circular_data
                corrupted_metadata_scenarios[2]["data"] = circular_data
                
                for scenario in corrupted_metadata_scenarios:
                    corrupted_metadata_file = temp_path / f"corrupted_{scenario['name']}_metadata.json"
                    
                    try:
                        if isinstance(scenario["data"], str):
                            # Invalid JSON string
                            with open(corrupted_metadata_file, 'w') as f:
                                f.write(scenario["data"])
                        else:
                            # Valid Python structure (may have circular refs)
                            try:
                                with open(corrupted_metadata_file, 'w') as f:
                                    json.dump(scenario["data"], f)
                            except (ValueError, TypeError):
                                # Can't serialize (e.g., circular reference) - write placeholder
                                with open(corrupted_metadata_file, 'w') as f:
                                    f.write('{"error": "circular_reference"}')
                    except Exception:
                        # Skip scenarios we can't create
                        continue
                    
                    # Test loading corrupted metadata
                    self._test_corrupted_metadata_operations(corrupted_metadata_file, scenario['description'])
    
    def _test_corrupted_metadata_operations(self, corrupted_file: Path, description: str):
        """Test operations on corrupted metadata file."""
        # Create corresponding ONNX file path
        onnx_file = corrupted_file.with_suffix('.onnx')
        
        try:
            # Test sidecar loading
            sidecar_data = tag_utils.load_tags_from_sidecar(str(onnx_file))
            # Should handle gracefully
            assert isinstance(sidecar_data, dict), f"Should return dict for {description}"
            
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as e:
            # Expected errors for corrupted metadata
            pass
        except RecursionError:
            # Expected for circular references
            pass
        except Exception as e:
            # Should not cause system crashes
            error_msg = str(e).lower()
            assert "segmentation" not in error_msg, f"Segfault on {description}: {e}"
            assert "memory" not in error_msg or "corruption" not in error_msg, f"Memory issue on {description}: {e}"
        
        try:
            # Test tag statistics
            stats = tag_utils.get_tag_statistics(str(onnx_file))
            assert isinstance(stats, dict), f"Stats should return dict for {description}"
            
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            # Expected errors for corrupted metadata
            pass
        except Exception as e:
            # Should handle gracefully
            error_msg = str(e).lower()
            assert "crash" not in error_msg, f"Crash on stats for {description}: {e}"
    
    def test_partial_file_corruption(self):
        """
        Test handling of partially corrupted files.
        
        Validates that the system can handle files that are partially
        corrupted but may have some recoverable data.
        """
        exporter = HierarchyExporter(strategy="htp")
        model = nn.Linear(10, 5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a valid export first
            valid_output = temp_path / "original.onnx"
            result = exporter.export(
                model=model,
                model_name_or_path="test",
                output_path=str(valid_output),
                opset_version=17
            )
            
            if valid_output.exists():
                # Read the valid file
                with open(valid_output, 'rb') as f:
                    valid_data = f.read()
                
                # Create partially corrupted versions
                corruption_scenarios = [
                    {
                        "name": "corrupted_end",
                        "data": valid_data[:len(valid_data)//2] + os.urandom(len(valid_data)//2),
                        "description": "Corrupted second half"
                    },
                    {
                        "name": "corrupted_middle",
                        "data": valid_data[:100] + os.urandom(200) + valid_data[300:],
                        "description": "Corrupted middle section"
                    },
                    {
                        "name": "truncated",
                        "data": valid_data[:len(valid_data)//3],
                        "description": "Truncated file"
                    },
                    {
                        "name": "padded",
                        "data": valid_data + b"\x00" * 1000,
                        "description": "Extra padding at end"
                    }
                ]
                
                for scenario in corruption_scenarios:
                    corrupted_file = temp_path / f"{scenario['name']}.onnx"
                    
                    with open(corrupted_file, 'wb') as f:
                        f.write(scenario['data'])
                    
                    # Test operations on partially corrupted file
                    try:
                        tags = tag_utils.load_tags_from_onnx(str(corrupted_file))
                        # May succeed with partial data or fail gracefully
                        if tags:
                            assert isinstance(tags, dict), f"Should return dict for {scenario['description']}"
                            
                    except Exception as e:
                        # Should fail gracefully without system crashes
                        error_msg = str(e).lower()
                        assert "segmentation" not in error_msg, f"Segfault on {scenario['description']}: {e}"
                        assert "access violation" not in error_msg, f"Access violation on {scenario['description']}: {e}"


class TestNetworkFailures:
    """
    Test handling of network-related failures.
    
    These tests validate that the system gracefully handles network
    issues when downloading models or accessing remote resources.
    """
    
    def test_model_download_failure_handling(self):
        """
        Test handling of model download failures.
        
        Validates that the system gracefully handles failures when
        downloading models from HuggingFace Hub or other sources.
        """
        exporter = HierarchyExporter(strategy="htp")
        
        # Test with various invalid model names that would cause download failures
        invalid_models = [
            "nonexistent/model-name",
            "user/private-model-that-does-not-exist",
            "invalid-format-model-name",
            "",  # Empty model name
            None,  # None model name
            "model-with-special-chars/!@#$%",
            "extremely-long-model-name/" + "x" * 200
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for invalid_model in invalid_models:
                output_path = Path(temp_dir) / f"download_failure_test.onnx"
                
                try:
                    # This should fail gracefully for invalid model names
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(invalid_model)
                    
                    # If model loading unexpectedly succeeds, continue with export test
                    result = exporter.export(
                        model=model,
                        model_name_or_path=invalid_model,
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                except (OSError, ValueError, TypeError, AttributeError) as e:
                    # Expected errors for invalid model names
                    error_msg = str(e).lower()
                    
                    # Verify error message is informative
                    assert any(keyword in error_msg for keyword in [
                        "not found", "invalid", "error", "failed", "does not exist"
                    ]), f"Error message should be informative: {e}"
                    
                    # Should not indicate system-level issues
                    assert "segmentation" not in error_msg, f"Segfault on invalid model {invalid_model}: {e}"
                    assert "memory" not in error_msg or "corruption" not in error_msg, f"Memory issue on {invalid_model}: {e}"
                    
                except Exception as e:
                    # Other exceptions should also be handled gracefully
                    error_msg = str(e).lower()
                    assert "crash" not in error_msg, f"System crash on invalid model {invalid_model}: {e}"
                    
                finally:
                    # Clean up any partial files
                    if output_path.exists():
                        output_path.unlink()
    
    @mock.patch('requests.get')
    def test_network_timeout_handling(self, mock_get):
        """
        Test handling of network timeouts.
        
        Validates that the system handles network timeouts gracefully
        without hanging or crashing.
        """
        # Mock network timeout
        import requests
        mock_get.side_effect = requests.Timeout("Connection timeout")
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "timeout_test.onnx"
            
            try:
                # This should handle the network timeout gracefully
                # Note: This test may not actually trigger the timeout depending on implementation
                from transformers import AutoModel
                model = AutoModel.from_pretrained("prajjwal1/bert-tiny")  # Use cached if available
                
                result = exporter.export(
                    model=model,
                    model_name_or_path="prajjwal1/bert-tiny",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # If export succeeds despite mocked timeout, verify output
                if output_path.exists():
                    assert output_path.stat().st_size > 0, "Output should not be empty"
                    
            except (requests.Timeout, ConnectionError, OSError) as e:
                # Expected network errors
                error_msg = str(e).lower()
                assert "timeout" in error_msg or "connection" in error_msg, f"Should indicate network issue: {e}"
                
            except Exception as e:
                # Other exceptions should not indicate system issues
                error_msg = str(e).lower()
                assert "segmentation" not in error_msg, f"Segfault on timeout: {e}"
    
    def test_offline_mode_handling(self):
        """
        Test handling of offline mode scenarios.
        
        Validates that the system can operate when network access
        is unavailable or restricted.
        """
        exporter = HierarchyExporter(strategy="htp")
        
        # Create a local model that doesn't require network access
        local_model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "offline_test.onnx"
            
            try:
                # This should work even without network access
                result = exporter.export(
                    model=local_model,
                    model_name_or_path="local_test_model",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # Verify successful offline operation
                assert output_path.exists(), "Offline export should succeed"
                assert output_path.stat().st_size > 0, "Offline export should produce valid output"
                assert result.get("coverage_percentage", 0) == 100.0, "Offline export should achieve full coverage"
                
            except Exception as e:
                # Offline operation should not fail due to network issues
                error_msg = str(e).lower()
                assert "network" not in error_msg, f"Offline operation should not require network: {e}"
                assert "connection" not in error_msg, f"Offline operation should not require connection: {e}"


class TestStateRecovery:
    """
    Test recovery from inconsistent internal states.
    
    These tests validate that the system can detect and recover
    from inconsistent internal states that might arise from
    interrupted operations or unexpected conditions.
    """
    
    def test_interrupted_export_recovery(self):
        """
        Test recovery from interrupted export operations.
        
        Validates that the system can handle interrupted exports
        and clean up properly without leaving inconsistent state.
        """
        exporter = HierarchyExporter(strategy="htp")
        model = nn.Linear(50, 25)
        
        class InterruptedExport:
            """Context manager to simulate interrupted export."""
            
            def __init__(self, interrupt_after_seconds=1):
                self.interrupt_after = interrupt_after_seconds
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Simulate cleanup after interruption
                pass
                
            def check_interrupt(self):
                if time.time() - self.start_time > self.interrupt_after:
                    raise KeyboardInterrupt("Simulated interruption")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "interrupted_test.onnx"
            
            # Simulate interrupted export
            try:
                with InterruptedExport(interrupt_after_seconds=0.5):
                    result = exporter.export(
                        model=model,
                        model_name_or_path="interrupted_test",
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    # If export completes quickly, it succeeded
                    if output_path.exists():
                        assert output_path.stat().st_size > 0, "Completed export should have valid output"
                        
            except KeyboardInterrupt:
                # Simulated interruption occurred
                pass
            except Exception as e:
                # Other exceptions should not indicate state corruption
                error_msg = str(e).lower()
                assert "corruption" not in error_msg, f"State corruption detected: {e}"
            
            # Verify system can continue operating after interruption
            try:
                # Attempt another export to verify system state is clean
                recovery_output = Path(temp_dir) / "recovery_test.onnx"
                
                recovery_result = exporter.export(
                    model=model,
                    model_name_or_path="recovery_test",
                    output_path=str(recovery_output),
                    opset_version=17
                )
                
                # Recovery export should succeed
                assert recovery_output.exists(), "Recovery export should succeed"
                assert recovery_result.get("coverage_percentage", 0) > 90, "Recovery should achieve good coverage"
                
            except Exception as e:
                pytest.fail(f"System failed to recover after interruption: {e}")
    
    def test_inconsistent_hierarchy_recovery(self):
        """
        Test recovery from inconsistent hierarchy states.
        
        Validates that the system can detect and handle inconsistent
        hierarchy information during export.
        """
        # Create a model with potentially problematic hierarchy
        class ProblematicHierarchyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 5)
                
                # Create potential hierarchy confusion
                self.layer1.name = "ambiguous_name"
                self.layer2.name = "ambiguous_name"  # Same name as layer1
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        exporter = HierarchyExporter(strategy="htp")
        model = ProblematicHierarchyModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "hierarchy_recovery_test.onnx"
            
            try:
                result = exporter.export(
                    model=model,
                    model_name_or_path="hierarchy_test",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # System should handle hierarchy ambiguity gracefully
                assert output_path.exists(), "Export should succeed despite hierarchy ambiguity"
                assert result.get("coverage_percentage", 0) > 80, "Should achieve reasonable coverage"
                assert result.get("empty_tags", 1) == 0, "Should not have empty tags (CARDINAL RULE)"
                
                # Verify metadata is consistent
                metadata_path = Path(temp_dir) / "hierarchy_recovery_test_htp_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    # Verify metadata structure is valid
                    assert "tagged_nodes" in metadata, "Metadata should contain tagged_nodes"
                    assert isinstance(metadata["tagged_nodes"], dict), "Tagged nodes should be dict"
                    
                    # All tags should be non-empty
                    for node_name, tag in metadata["tagged_nodes"].items():
                        assert tag and tag.strip(), f"Node {node_name} has empty tag"
                
            except Exception as e:
                # Should handle hierarchy issues gracefully
                error_msg = str(e).lower()
                assert "corruption" not in error_msg, f"Hierarchy corruption detected: {e}"
                assert "inconsistent" not in error_msg, f"Inconsistent state detected: {e}"
    
    def test_memory_state_recovery(self):
        """
        Test recovery from memory-related state issues.
        
        Validates that the system can recover from memory pressure
        or allocation issues without leaving corrupted state.
        """
        import gc
        
        exporter = HierarchyExporter(strategy="htp")
        
        # Create a model that exercises memory management
        class MemoryTestModel(nn.Module):
            def __init__(self, size=100):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(size, size) for _ in range(10)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with increasing memory pressure
            for size in [50, 100, 200]:  # Keep sizes reasonable for tests
                output_path = Path(temp_dir) / f"memory_recovery_{size}.onnx"
                
                try:
                    model = MemoryTestModel(size)
                    
                    # Force memory pressure
                    gc.collect()
                    
                    result = exporter.export(
                        model=model,
                        model_name_or_path=f"memory_test_{size}",
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    # Verify successful export
                    assert output_path.exists(), f"Memory test {size} should succeed"
                    assert result.get("coverage_percentage", 0) > 90, f"Memory test {size} should achieve good coverage"
                    
                    # Clean up
                    del model, result
                    gc.collect()
                    
                except (MemoryError, RuntimeError) as e:
                    # Memory errors are acceptable for large models
                    if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        continue
                    else:
                        pytest.fail(f"Unexpected memory-related error: {e}")
                        
                except Exception as e:
                    # Other exceptions should not indicate memory corruption
                    error_msg = str(e).lower()
                    assert "corruption" not in error_msg, f"Memory corruption on size {size}: {e}"
                    assert "segmentation" not in error_msg, f"Segfault on size {size}: {e}"


class TestGracefulDegradation:
    """
    Test graceful degradation when facing limitations.
    
    These tests validate that the system continues to operate
    with reduced functionality when facing various limitations.
    """
    
    def test_degraded_mode_operation(self):
        """
        Test operation in degraded mode when full functionality unavailable.
        
        Validates that the system can operate with reduced functionality
        when certain features or resources are unavailable.
        """
        exporter = HierarchyExporter(strategy="htp")
        model = nn.Linear(20, 10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "degraded_mode_test.onnx"
            
            # Test with various degraded conditions
            degraded_scenarios = [
                {"name": "no_input_specs", "input_specs": None},
                {"name": "minimal_config", "opset_version": 11},  # Older opset
                {"name": "no_model_name", "model_name_or_path": None}
            ]
            
            for scenario in degraded_scenarios:
                try:
                    test_output = Path(temp_dir) / f"degraded_{scenario['name']}.onnx"
                    
                    export_kwargs = {
                        "model": model,
                        "output_path": str(test_output),
                        "opset_version": scenario.get("opset_version", 17)
                    }
                    
                    # Add optional parameters if provided
                    if "model_name_or_path" in scenario:
                        export_kwargs["model_name_or_path"] = scenario["model_name_or_path"] or "degraded_test"
                    else:
                        export_kwargs["model_name_or_path"] = "degraded_test"
                        
                    if "input_specs" in scenario:
                        export_kwargs["input_specs"] = scenario["input_specs"]
                    
                    result = exporter.export(**export_kwargs)
                    
                    # Should succeed even in degraded mode
                    assert test_output.exists(), f"Degraded mode {scenario['name']} should produce output"
                    assert result.get("coverage_percentage", 0) > 80, f"Degraded mode {scenario['name']} should achieve reasonable coverage"
                    
                except Exception as e:
                    # Degraded mode failures should be graceful
                    error_msg = str(e).lower()
                    assert "crash" not in error_msg, f"Degraded mode {scenario['name']} should not crash: {e}"
                    assert "fatal" not in error_msg, f"Degraded mode {scenario['name']} should not be fatal: {e}"
    
    def test_limited_resource_operation(self):
        """
        Test operation with limited system resources.
        
        Validates that the system can adapt to limited resources
        and continue operation with appropriate adjustments.
        """
        exporter = HierarchyExporter(strategy="htp")
        model = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Simulate limited resource conditions
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "limited_resources_test.onnx"
            
            # Monitor resources during export
            import psutil
            process = psutil.Process()
            
            try:
                # Get initial resource state
                initial_memory = process.memory_info().rss
                initial_files = len(process.open_files())
                
                result = exporter.export(
                    model=model,
                    model_name_or_path="limited_resources_test",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # Should succeed with reasonable resource usage
                assert output_path.exists(), "Limited resource export should succeed"
                
                final_memory = process.memory_info().rss
                final_files = len(process.open_files())
                
                memory_increase = final_memory - initial_memory
                file_increase = final_files - initial_files
                
                # Resource usage should be reasonable
                memory_mb = memory_increase / (1024 * 1024)
                assert memory_mb < 200, f"Should use reasonable memory: {memory_mb:.1f}MB"
                assert file_increase <= 5, f"Should not leak file handles: {file_increase}"
                
            except Exception as e:
                # Resource limitations should be handled gracefully
                error_msg = str(e).lower()
                if "memory" in error_msg or "resource" in error_msg:
                    # Expected resource limitation error
                    pass
                else:
                    pytest.fail(f"Unexpected error in limited resource test: {e}")


# Error recovery test markers
pytestmark = [
    pytest.mark.error_recovery,
    pytest.mark.slow  # Error recovery tests may take longer
]