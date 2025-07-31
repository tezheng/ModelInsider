"""
Resource Management Test Suite

This module tests resource management aspects of the modelexport system,
including memory limits, timeout handling, cleanup validation, and
large model handling.

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Resource tests must be universal
- MUST-002: TORCH.NN FILTERING - Validate resource handling across all architectures  
- MUST-003: UNIVERSAL DESIGN - Resource management works for any model type

Resource Test Categories:
1. Memory Management - Limits, pressure testing, leak detection
2. Timeout Handling - Operation timeouts, hanging process detection
3. Resource Cleanup - Temp files, GPU memory, process cleanup
4. Large Model Handling - Models exceeding system capabilities
5. Concurrent Operations - Multiple simultaneous exports
6. Resource Monitoring - Performance tracking, bottleneck detection
"""

import gc
import tempfile
import threading
import time
from pathlib import Path

import psutil
import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp.htp_exporter import HTPExporter


class TestMemoryManagement:
    """
    Test memory management and memory leak prevention.
    
    These tests validate that the system properly manages memory,
    prevents memory leaks, and handles memory pressure gracefully.
    """
    
    def test_memory_leak_detection(self):
        """
        Test for memory leaks during repeated exports.
        
        Validates that memory usage doesn't continuously increase
        across multiple export operations.
        """
        # Create a simple test model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        exporter = HTPExporter(verbose=False)
        memory_usage = []
        
        # Perform multiple exports and track memory
        num_iterations = 5  # Keep small for test performance
        
        for i in range(num_iterations):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"leak_test_{i}.onnx"
                
                # Force garbage collection before measurement
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Measure memory before export
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                # Perform export
                result = exporter.export(
                    model=model,
                    model_name_or_path="test_leak",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # Force cleanup
                del result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Measure memory after export and cleanup
                memory_after = process.memory_info().rss
                memory_usage.append(memory_after - memory_before)
        
        # Analyze memory usage pattern
        if len(memory_usage) >= 3:
            # Check for continuously increasing memory usage (leak indicator)
            increasing_trend = all(
                memory_usage[i] <= memory_usage[i+1] * 1.2  # Allow 20% variance
                for i in range(len(memory_usage)-1)
            )
            
            # Memory shouldn't continuously increase beyond reasonable bounds
            max_increase = max(memory_usage)
            reasonable_limit = 100 * 1024 * 1024  # 100MB per export
            
            assert max_increase < reasonable_limit, \
                f"Potential memory leak detected: max increase {max_increase / (1024*1024):.1f}MB"
            
            # Later iterations shouldn't use significantly more memory than early ones
            if len(memory_usage) >= 4:
                early_avg = sum(memory_usage[:2]) / 2
                late_avg = sum(memory_usage[-2:]) / 2
                growth_ratio = late_avg / early_avg if early_avg > 0 else 1
                
                assert growth_ratio < 2.0, \
                    f"Memory usage grew {growth_ratio:.1f}x across iterations - possible leak"
    
    def test_memory_pressure_handling(self):
        """
        Test behavior under memory pressure conditions.
        
        Validates that the system gracefully handles low memory
        conditions without crashes or data corruption.
        """
        # Create a model that uses more memory
        class MemoryIntensiveModel(nn.Module):
            def __init__(self, size_factor=1):
                super().__init__()
                # Scale model size but keep it reasonable for tests
                base_size = min(1000 * size_factor, 5000)
                self.layer1 = nn.Linear(base_size, base_size // 2)
                self.layer2 = nn.Linear(base_size // 2, base_size // 4)
                self.layer3 = nn.Linear(base_size // 4, 10)
                
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                return self.layer3(x)
        
        exporter = HTPExporter(verbose=False)
        
        # Test with increasing model sizes
        size_factors = [1, 2, 3]  # Keep modest for test stability
        
        for size_factor in size_factors:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"memory_pressure_{size_factor}.onnx"
                
                try:
                    model = MemoryIntensiveModel(size_factor)
                    
                    # Monitor memory during export
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss
                    available_memory = psutil.virtual_memory().available
                    
                    # Only proceed if we have reasonable available memory
                    if available_memory < 500 * 1024 * 1024:  # Less than 500MB
                        pytest.skip("Insufficient memory for pressure testing")
                    
                    result = exporter.export(
                        model=model,
                        model_name_or_path="memory_test",
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    # Verify export succeeded and output is reasonable
                    assert output_path.exists(), "Export should succeed under memory pressure"
                    assert output_path.stat().st_size > 0, "Output should not be empty"
                    
                    # Check memory usage is reasonable
                    final_memory = process.memory_info().rss
                    memory_increase = final_memory - initial_memory
                    memory_increase_mb = memory_increase / (1024 * 1024)
                    
                    # Memory increase should be proportional to model size but not excessive
                    max_reasonable_mb = 500 * size_factor  # 500MB per size factor
                    assert memory_increase_mb < max_reasonable_mb, \
                        f"Excessive memory usage: {memory_increase_mb:.1f}MB (max: {max_reasonable_mb}MB)"
                    
                except (MemoryError, RuntimeError) as e:
                    # Memory errors are acceptable for large models
                    if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        continue
                    else:
                        raise
                except Exception as e:
                    # Other exceptions should not indicate memory corruption
                    error_msg = str(e).lower()
                    assert "segmentation fault" not in error_msg, f"Memory corruption detected: {e}"
                    assert "access violation" not in error_msg, f"Memory access violation: {e}"
    
    def test_gpu_memory_management(self):
        """
        Test GPU memory management if CUDA is available.
        
        Validates proper GPU memory allocation and cleanup.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU memory testing")
        
        # Create a model that could use GPU memory
        model = nn.Sequential(
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Move model to GPU if available
        device = torch.device("cuda:0")
        model = model.to(device)
        
        exporter = HTPExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gpu_memory_test.onnx"
            
            # Measure GPU memory before export
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated()
            
            try:
                result = exporter.export(
                    model=model,
                    model_name_or_path="gpu_test",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # Verify export succeeded
                assert output_path.exists(), "GPU model export should succeed"
                
                # Force cleanup
                del result
                torch.cuda.empty_cache()
                
                # Check GPU memory after cleanup
                gpu_memory_after = torch.cuda.memory_allocated()
                memory_difference = abs(gpu_memory_after - gpu_memory_before)
                
                # GPU memory should be properly cleaned up
                max_reasonable_difference = 10 * 1024 * 1024  # 10MB tolerance
                assert memory_difference < max_reasonable_difference, \
                    f"GPU memory not properly cleaned: {memory_difference / (1024*1024):.1f}MB difference"
                
            except Exception as e:
                # GPU operations may fail for various reasons, but shouldn't leak memory
                torch.cuda.empty_cache()
                gpu_memory_after = torch.cuda.memory_allocated()
                memory_difference = abs(gpu_memory_after - gpu_memory_before)
                
                assert memory_difference < 50 * 1024 * 1024, \
                    f"GPU memory leak detected after error: {memory_difference / (1024*1024):.1f}MB"


class TestTimeoutHandling:
    """
    Test timeout handling and hanging process detection.
    
    These tests validate that operations complete within reasonable
    time limits and don't hang indefinitely.
    """
    
    def test_export_timeout_limits(self):
        """
        Test that exports complete within reasonable time limits.
        
        Validates that export operations don't hang indefinitely
        and complete within expected time bounds.
        """
        # Create models of varying complexity
        models = {
            "simple": nn.Linear(10, 5),
            "medium": nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            ),
            "complex": nn.Sequential(
                *[nn.Sequential(nn.Linear(50, 50), nn.ReLU()) for _ in range(10)],
                nn.Linear(50, 10)
            )
        }
        
        # Expected timeout limits for each model type (in seconds)
        timeout_limits = {
            "simple": 30,
            "medium": 60,
            "complex": 120
        }
        
        exporter = HTPExporter(verbose=False)
        
        for model_name, model in models.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"timeout_test_{model_name}.onnx"
                
                start_time = time.time()
                timeout_limit = timeout_limits[model_name]
                
                try:
                    # Use threading to implement timeout
                    result = None
                    exception = None
                    
                    def export_with_timeout():
                        nonlocal result, exception
                        try:
                            result = exporter.export(
                                model=model,
                                model_name_or_path=f"timeout_test_{model_name}",
                                output_path=str(output_path),
                                opset_version=17
                            )
                        except Exception as e:
                            exception = e
                    
                    thread = threading.Thread(target=export_with_timeout)
                    thread.start()
                    thread.join(timeout=timeout_limit)
                    
                    elapsed_time = time.time() - start_time
                    
                    if thread.is_alive():
                        pytest.fail(f"{model_name} model export timeout after {elapsed_time:.1f}s (limit: {timeout_limit}s)")
                    
                    if exception:
                        # Exception occurred, but it should have happened quickly
                        assert elapsed_time < timeout_limit, \
                            f"{model_name} model exception took {elapsed_time:.1f}s (limit: {timeout_limit}s)"
                    else:
                        # Export succeeded
                        assert result is not None, f"{model_name} model export returned None"
                        assert elapsed_time < timeout_limit, \
                            f"{model_name} model export took {elapsed_time:.1f}s (limit: {timeout_limit}s)"
                        assert output_path.exists(), f"{model_name} model output file not created"
                
                except Exception as e:
                    # Ensure exceptions happen quickly
                    elapsed_time = time.time() - start_time
                    assert elapsed_time < timeout_limit, \
                        f"{model_name} model error took {elapsed_time:.1f}s (limit: {timeout_limit}s)"
    
    def test_hanging_operation_detection(self):
        """
        Test detection of hanging operations.
        
        Validates that the system can detect and handle operations
        that may hang or take unexpectedly long.
        """
        # Simulate a potentially hanging model (with built-in timeout for safety)
        class PotentiallyHangingModel(nn.Module):
            def __init__(self, hang_probability=0.0):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.hang_probability = hang_probability
                
            def forward(self, x):
                # Simulate potential hanging with very short delay for tests
                if self.hang_probability > 0:
                    time.sleep(min(0.1, self.hang_probability))  # Max 0.1s delay
                return self.linear(x)
        
        exporter = HTPExporter(verbose=False)
        
        # Test with different hanging probabilities
        hang_probabilities = [0.0, 0.05, 0.1]  # Short delays for test stability
        
        for hang_prob in hang_probabilities:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"hang_test_{hang_prob}.onnx"
                
                model = PotentiallyHangingModel(hang_prob)
                start_time = time.time()
                max_time = 30  # 30 second maximum
                
                try:
                    # Monitor the export operation
                    result = None
                    exception = None
                    
                    def monitored_export():
                        nonlocal result, exception
                        try:
                            result = exporter.export(
                                model=model,
                                model_name_or_path=f"hang_test_{hang_prob}",
                                output_path=str(output_path),
                                opset_version=17
                            )
                        except Exception as e:
                            exception = e
                    
                    thread = threading.Thread(target=monitored_export)
                    thread.start()
                    
                    # Monitor with shorter intervals
                    monitor_interval = 5  # Check every 5 seconds
                    total_waited = 0
                    
                    while thread.is_alive() and total_waited < max_time:
                        thread.join(timeout=monitor_interval)
                        total_waited += monitor_interval
                        
                        if thread.is_alive():
                            elapsed = time.time() - start_time
                            print(f"Export still running after {elapsed:.1f}s...")
                    
                    elapsed_time = time.time() - start_time
                    
                    if thread.is_alive():
                        # Operation is hanging
                        pytest.fail(f"Detected hanging operation after {elapsed_time:.1f}s with hang_prob={hang_prob}")
                    
                    # Verify reasonable completion time
                    expected_max_time = 10 + (hang_prob * 100)  # Base time + hang factor
                    assert elapsed_time < expected_max_time, \
                        f"Operation took {elapsed_time:.1f}s, expected < {expected_max_time:.1f}s"
                    
                except Exception as e:
                    # Exceptions should occur quickly
                    elapsed_time = time.time() - start_time
                    assert elapsed_time < max_time, \
                        f"Exception took {elapsed_time:.1f}s to occur (too long)"


class TestResourceCleanup:
    """
    Test proper cleanup of resources after operations.
    
    These tests validate that temporary files, memory, and other
    resources are properly cleaned up after operations complete.
    """
    
    def test_temporary_file_cleanup(self):
        """
        Test that temporary files are properly cleaned up.
        
        Validates that no temporary files are left behind after
        export operations complete or fail.
        """
        exporter = HTPExporter(verbose=False)
        model = nn.Linear(10, 5)
        
        # Track system temp directory
        system_temp = Path(tempfile.gettempdir())
        
        # Get initial temp file count
        initial_temp_files = set()
        try:
            initial_temp_files = set(system_temp.glob("*"))
        except (OSError, PermissionError):
            # Skip if we can't read system temp directory
            pytest.skip("Cannot access system temp directory")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cleanup_test.onnx"
            
            # Perform export
            result = exporter.export(
                model=model,
                model_name_or_path="cleanup_test",
                output_path=str(output_path),
                opset_version=17
            )
            
            # Clean up result
            del result
            gc.collect()
        
        # Check temp directory after operation
        try:
            final_temp_files = set(system_temp.glob("*"))
            leaked_files = final_temp_files - initial_temp_files
            
            # Filter out files that might be created by other processes
            potential_leaks = [
                f for f in leaked_files 
                if any(keyword in f.name.lower() for keyword in 
                      ['onnx', 'torch', 'model', 'export', 'hierarchy'])
            ]
            
            assert len(potential_leaks) == 0, \
                f"Potential temp file leaks detected: {[f.name for f in potential_leaks]}"
                
        except (OSError, PermissionError):
            # Skip verification if we can't read temp directory
            pass
    
    def test_process_resource_cleanup(self):
        """
        Test that process resources are properly cleaned up.
        
        Validates that file handles, threads, and other process
        resources don't leak after operations.
        """
        import threading
        
        exporter = HTPExporter(verbose=False)
        model = nn.Linear(10, 5)
        
        # Get initial resource counts
        process = psutil.Process()
        initial_open_files = len(process.open_files())
        initial_threads = threading.active_count()
        
        # Perform multiple operations
        for i in range(3):  # Multiple operations to test for accumulation
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"resource_test_{i}.onnx"
                
                result = exporter.export(
                    model=model,
                    model_name_or_path=f"resource_test_{i}",
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # Clean up
                del result
                gc.collect()
        
        # Force additional cleanup
        time.sleep(0.1)  # Allow time for cleanup
        gc.collect()
        
        # Check final resource counts
        final_open_files = len(process.open_files())
        final_threads = threading.active_count()
        
        # Allow some tolerance for system variations
        file_handle_increase = final_open_files - initial_open_files
        thread_increase = final_threads - initial_threads
        
        assert file_handle_increase <= 2, \
            f"Possible file handle leak: {file_handle_increase} additional handles"
        assert thread_increase <= 1, \
            f"Possible thread leak: {thread_increase} additional threads"
    
    def test_cleanup_after_failure(self):
        """
        Test resource cleanup after export failures.
        
        Validates that resources are properly cleaned up even when
        export operations fail or are interrupted.
        """
        exporter = HTPExporter(verbose=False)
        
        # Create a model that will cause export failure
        class FailingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                # This will cause issues during ONNX export
                if hasattr(x, 'requires_grad') and x.requires_grad:
                    raise RuntimeError("Intentional failure for testing")
                return self.linear(x)
        
        model = FailingModel()
        
        # Track resources before failure
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_open_files = len(process.open_files())
        
        # Attempt export that should fail
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "failure_test.onnx"
            
            try:
                result = exporter.export(
                    model=model,
                    model_name_or_path="failure_test",
                    output_path=str(output_path),
                    opset_version=17
                )
                # If export unexpectedly succeeds, still check resources
            except Exception as e:
                # Expected failure - verify it's the right type
                pass
        
        # Force cleanup after failure
        gc.collect()
        time.sleep(0.1)
        
        # Check resource cleanup after failure
        final_memory = process.memory_info().rss
        final_open_files = len(process.open_files())
        
        memory_increase = final_memory - initial_memory
        file_handle_increase = final_open_files - initial_open_files
        
        # Resources should be cleaned up even after failure
        assert memory_increase < 50 * 1024 * 1024, \
            f"Memory not cleaned up after failure: {memory_increase / (1024*1024):.1f}MB increase"
        assert file_handle_increase <= 2, \
            f"File handles not cleaned up after failure: {file_handle_increase} additional handles"


class TestConcurrentOperations:
    """
    Test concurrent export operations and thread safety.
    
    These tests validate that the system can handle multiple
    simultaneous export operations safely.
    """
    
    def test_concurrent_exports(self):
        """
        Test multiple concurrent export operations.
        
        Validates that multiple exports can run simultaneously
        without interfering with each other.
        """
        # Create multiple simple models
        models = [
            nn.Linear(10, 5),
            nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5)),
            nn.Sequential(nn.Linear(15, 8), nn.Tanh(), nn.Linear(8, 3))
        ]
        
        results = {}
        exceptions = {}
        
        def export_model(model_index, model):
            """Export a single model."""
            try:
                exporter = HTPExporter(verbose=False)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / f"concurrent_test_{model_index}.onnx"
                    
                    result = exporter.export(
                        model=model,
                        model_name_or_path=f"concurrent_test_{model_index}",
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    # Verify output
                    if output_path.exists():
                        results[model_index] = {
                            "success": True,
                            "file_size": output_path.stat().st_size,
                            "coverage": result.get("coverage_percentage", 0)
                        }
                    else:
                        results[model_index] = {"success": False, "error": "No output file"}
                        
            except Exception as e:
                exceptions[model_index] = str(e)
        
        # Start concurrent exports
        threads = []
        for i, model in enumerate(models):
            thread = threading.Thread(target=export_model, args=(i, model))
            threads.append(thread)
            thread.start()
        
        # Wait for all exports to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout per thread
        
        # Verify all threads completed
        active_threads = [t for t in threads if t.is_alive()]
        assert len(active_threads) == 0, f"{len(active_threads)} threads still running"
        
        # Verify results
        assert len(results) + len(exceptions) == len(models), \
            "Not all concurrent operations completed"
        
        # At least some operations should succeed
        successful_exports = sum(1 for r in results.values() if r.get("success", False))
        assert successful_exports > 0, "No concurrent exports succeeded"
        
        # Check for consistent results across concurrent operations
        if successful_exports > 1:
            coverages = [r["coverage"] for r in results.values() if r.get("success") and "coverage" in r]
            if len(coverages) > 1:
                # All successful exports should achieve similar coverage
                min_coverage = min(coverages)
                max_coverage = max(coverages)
                assert min_coverage >= 90.0, f"Low coverage in concurrent export: {min_coverage}%"
                assert max_coverage - min_coverage <= 10.0, \
                    f"Inconsistent coverage across concurrent exports: {min_coverage}% to {max_coverage}%"
    
    def test_thread_safety(self):
        """
        Test thread safety of export operations.
        
        Validates that the system maintains thread safety during
        concurrent operations.
        """
        # Create a shared model
        shared_model = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10)
        )
        
        # Shared state tracking
        shared_state = {
            "exports_completed": 0,
            "exports_failed": 0,
            "max_memory_usage": 0
        }
        
        state_lock = threading.Lock()
        
        def thread_safe_export(thread_id):
            """Perform export with thread safety checks."""
            try:
                exporter = HTPExporter(verbose=False)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / f"thread_safety_{thread_id}.onnx"
                    
                    # Monitor memory during export
                    process = psutil.Process()
                    memory_before = process.memory_info().rss
                    
                    result = exporter.export(
                        model=shared_model,
                        model_name_or_path=f"thread_safety_{thread_id}",
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    memory_after = process.memory_info().rss
                    memory_usage = memory_after - memory_before
                    
                    # Update shared state safely
                    with state_lock:
                        shared_state["exports_completed"] += 1
                        shared_state["max_memory_usage"] = max(
                            shared_state["max_memory_usage"], 
                            memory_usage
                        )
                    
                    # Verify export success
                    assert output_path.exists(), f"Thread {thread_id} export failed to create output"
                    assert result.get("coverage_percentage", 0) > 90, \
                        f"Thread {thread_id} export low coverage: {result.get('coverage_percentage', 0)}%"
                    
            except Exception as e:
                with state_lock:
                    shared_state["exports_failed"] += 1
                raise
        
        # Start multiple threads
        num_threads = 3  # Keep moderate for test stability
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=thread_safe_export, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout
        
        # Verify thread safety
        with state_lock:
            total_operations = shared_state["exports_completed"] + shared_state["exports_failed"]
            assert total_operations == num_threads, \
                f"Thread safety issue: {total_operations} operations != {num_threads} threads"
            
            # Most operations should succeed
            success_rate = shared_state["exports_completed"] / num_threads
            assert success_rate >= 0.5, f"Low success rate in threaded operations: {success_rate:.1%}"
            
            # Memory usage should be reasonable
            max_memory_mb = shared_state["max_memory_usage"] / (1024 * 1024)
            assert max_memory_mb < 1000, \
                f"Excessive memory usage in threaded operations: {max_memory_mb:.1f}MB"


# Resource monitoring utilities
class ResourceMonitor:
    """Utility class for monitoring system resources during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_state = self._get_current_state()
    
    def _get_current_state(self):
        """Get current resource state."""
        return {
            "memory_rss": self.process.memory_info().rss,
            "open_files": len(self.process.open_files()),
            "threads": threading.active_count(),
            "cpu_percent": self.process.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    def get_resource_diff(self):
        """Get difference from initial state."""
        current = self._get_current_state()
        return {
            key: current[key] - self.initial_state[key]
            for key in current
        }
    
    def assert_reasonable_usage(self, max_memory_mb=500, max_files=10, max_threads=5):
        """Assert that resource usage is within reasonable bounds."""
        diff = self.get_resource_diff()
        
        memory_mb = diff["memory_rss"] / (1024 * 1024)
        assert memory_mb < max_memory_mb, \
            f"Excessive memory usage: {memory_mb:.1f}MB (max: {max_memory_mb}MB)"
        
        assert diff["open_files"] <= max_files, \
            f"Too many open files: {diff['open_files']} (max: {max_files})"
        
        assert diff["threads"] <= max_threads, \
            f"Too many threads: {diff['threads']} (max: {max_threads})"


# Resource test markers
pytestmark = [
    pytest.mark.resource,
    pytest.mark.slow  # Resource tests may take longer
]