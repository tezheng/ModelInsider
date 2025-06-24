"""
Performance and Stress Testing - Round 6

These tests validate the hierarchy exporter under various performance and stress conditions,
ensuring robust operation with large models, high memory usage, and concurrent operations.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import time
import gc
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from modelexport.hierarchy_exporter import HierarchyExporter


class TestLargeModelPerformance:
    """Test performance with large models."""
    
    def test_large_parameter_count_model(self):
        """Test model with large number of parameters."""
        class LargeParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Create model with many parameters
                self.layers = nn.ModuleList([
                    nn.Linear(1024, 1024) for _ in range(20)  # ~20M parameters
                ])
                self.output = nn.Linear(1024, 10)
                
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return self.output(x)
        
        model = LargeParamModel()
        model.eval()
        inputs = torch.randn(4, 1024)
        
        # Count actual parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model has {param_count:,} parameters")
        
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            start_time = time.time()
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                export_time = time.time() - start_time
                print(f"{strategy} export took {export_time:.2f} seconds")
                
                assert result is not None
                assert result['total_operations'] > 15  # Should have many operations
                
                # Export should complete in reasonable time (< 30 seconds)
                assert export_time < 30.0
    
    def test_very_deep_model(self):
        """Test model with very deep architecture."""
        class VeryDeepModel(nn.Module):
            def __init__(self, depth=100):
                super().__init__()
                self.depth = depth
                self.layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.BatchNorm1d(64)
                    ) for _ in range(depth)
                ])
                self.output = nn.Linear(64, 1)
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)
        
        model = VeryDeepModel(depth=50)  # 50 layers deep
        model.eval()
        inputs = torch.randn(2, 64)
        
        exporter = HierarchyExporter(strategy="htp")
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            export_time = time.time() - start_time
            print(f"Deep model export took {export_time:.2f} seconds")
            
            assert result is not None
            assert result['total_operations'] > 30
            
            # Should handle deep models efficiently
            assert export_time < 20.0
    
    def test_wide_model_performance(self):
        """Test model with very wide layers."""
        class WideModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Wide layers
                self.layer1 = nn.Linear(2048, 4096)
                self.layer2 = nn.Linear(4096, 4096)
                self.layer3 = nn.Linear(4096, 2048)
                self.layer4 = nn.Linear(2048, 1024)
                self.output = nn.Linear(1024, 10)
                
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                x = torch.relu(self.layer3(x))
                x = torch.relu(self.layer4(x))
                return self.output(x)
        
        model = WideModel()
        model.eval()
        inputs = torch.randn(1, 2048)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 4


class TestHighOperationCountStress:
    """Test stress conditions with high operation counts."""
    
    def test_many_operations_model(self):
        """Test model that generates many ONNX operations."""
        class ManyOpsModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Each operation creates multiple ONNX ops
                self.ops = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(32, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Dropout2d(0.1)
                    ) for _ in range(25)  # 25 blocks
                ])
                self.input_conv = nn.Conv2d(3, 32, 3, padding=1)
                self.output_conv = nn.Conv2d(32, 1, 1)
                
            def forward(self, x):
                x = self.input_conv(x)
                for op in self.ops:
                    x = op(x)
                return self.output_conv(x)
        
        model = ManyOpsModel()
        model.eval()
        inputs = torch.randn(1, 3, 32, 32)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            print(f"Model generated {result['total_operations']} operations")
            assert result['total_operations'] > 50  # Should have many operations
            
            # Check tag mapping size
            tag_mapping = exporter.get_tag_mapping()
            assert len(tag_mapping) == result['total_operations']
    
    def test_complex_computation_graph(self):
        """Test model with complex computation graph."""
        class ComplexGraphModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_layer = nn.Linear(64, 64)
                
                # Multiple branches that use shared layer
                self.branch1 = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16)
                )
                
                self.branch2 = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.Tanh(),
                    nn.Linear(32, 16)
                )
                
                self.branch3 = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.Sigmoid(),
                    nn.Linear(32, 16)
                )
                
                self.combine = nn.Linear(48, 10)  # 16 * 3 = 48
                
            def forward(self, x):
                # Shared processing
                shared = self.shared_layer(x)
                
                # Multiple branches
                b1 = self.branch1(shared)
                b2 = self.branch2(shared)
                b3 = self.branch3(shared)
                
                # Complex combinations
                combined = torch.cat([b1, b2, b3], dim=1)
                
                # Additional operations
                combined = combined * 0.5 + torch.mean(combined, dim=1, keepdim=True)
                
                return self.combine(combined)
        
        model = ComplexGraphModel()
        model.eval()
        inputs = torch.randn(4, 64)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 10


class TestMemoryStress:
    """Test memory usage and cleanup under stress conditions."""
    
    def test_memory_cleanup_after_export(self):
        """Test that memory is properly cleaned up after export."""
        class MemoryTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(512, 512) for _ in range(10)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = MemoryTestModel()
        model.eval()
        inputs = torch.randn(8, 512)
        
        # Measure memory before
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        initial_objects = len(gc.get_objects())
        
        # Perform multiple exports
        for i in range(5):
            exporter = HierarchyExporter(strategy="htp")
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                assert result is not None
            
            # Manually clean up
            del exporter
        
        # Measure memory after
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        print(f"Object count grew by {object_growth}")
        
        # Allow some growth but not excessive
        assert object_growth < 1000, f"Too much memory growth: {object_growth} objects"
    
    def test_large_input_size_handling(self):
        """Test handling of large input sizes."""
        class LargeInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                self.fc = nn.Linear(32 * 64, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        model = LargeInputModel()
        model.eval()
        
        # Large input size
        inputs = torch.randn(1, 3, 512, 512)  # Large image
        
        exporter = HierarchyExporter(strategy="usage_based")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
    
    def test_batch_size_stress(self):
        """Test with large batch sizes."""
        class BatchTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)
                self.norm = nn.BatchNorm1d(64)
                self.output = nn.Linear(64, 10)
                
            def forward(self, x):
                x = torch.relu(self.norm(self.linear(x)))
                return self.output(x)
        
        model = BatchTestModel()
        model.eval()
        
        # Large batch size
        inputs = torch.randn(64, 128)  # Large batch
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None


class TestConcurrentOperations:
    """Test concurrent export operations."""
    
    def test_concurrent_exports_different_models(self):
        """Test concurrent exports of different models (with locking due to PyTorch ONNX limitations)."""
        def create_test_model(size):
            class TestModel(nn.Module):
                def __init__(self, size):
                    super().__init__()
                    self.linear = nn.Linear(size, size // 2)
                    self.output = nn.Linear(size // 2, 10)
                    
                def forward(self, x):
                    return self.output(torch.relu(self.linear(x)))
            return TestModel(size)
        
        # Use a lock to serialize ONNX exports due to PyTorch's global state limitation
        export_lock = threading.Lock()
        
        def export_model(model_size, thread_id):
            model = create_test_model(model_size)
            model.eval()
            inputs = torch.randn(2, model_size)
            
            exporter = HierarchyExporter(strategy="usage_based")
            
            with export_lock:  # Serialize ONNX exports
                with tempfile.NamedTemporaryFile(suffix=f'_thread_{thread_id}.onnx') as tmp:
                    result = exporter.export(
                        model=model,
                        example_inputs=inputs,
                        output_path=tmp.name
                    )
                    
                    return result is not None, thread_id
        
        # Run concurrent exports (serialized by lock)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                model_size = 64 + i * 32  # Different sizes: 64, 96, 128, 160
                future = executor.submit(export_model, model_size, i)
                futures.append(future)
            
            # Wait for all to complete
            results = []
            for future in as_completed(futures):
                success, thread_id = future.result()
                results.append(success)
                print(f"Thread {thread_id} completed: {success}")
        
        # All exports should succeed
        assert all(results), "Some concurrent exports failed"
    
    def test_concurrent_exports_same_model(self):
        """Test concurrent exports of the same model (with locking due to PyTorch ONNX limitations)."""
        class SharedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc = nn.Linear(16 * 16, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        shared_model = SharedModel()
        shared_model.eval()
        shared_inputs = torch.randn(1, 3, 32, 32)
        
        # Use a lock to serialize ONNX exports
        export_lock = threading.Lock()
        
        def export_shared_model(thread_id):
            exporter = HierarchyExporter(strategy="htp")
            
            with export_lock:  # Serialize ONNX exports
                with tempfile.NamedTemporaryFile(suffix=f'_shared_{thread_id}.onnx') as tmp:
                    result = exporter.export(
                        model=shared_model,
                        example_inputs=shared_inputs,
                        output_path=tmp.name
                    )
                    
                    return result is not None, result['total_operations'] if result else 0
        
        # Run concurrent exports of same model (serialized by lock)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(export_shared_model, i) for i in range(3)]
            
            results = []
            operation_counts = []
            for future in as_completed(futures):
                success, op_count = future.result()
                results.append(success)
                operation_counts.append(op_count)
        
        # All should succeed with same operation count
        assert all(results), "Some concurrent exports of shared model failed"
        assert len(set(operation_counts)) == 1, "Operation counts should be identical"


class TestStressEdgeCases:
    """Test edge cases under stress conditions."""
    
    def test_rapid_successive_exports(self):
        """Test rapid successive exports without cleanup time."""
        class QuickTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
                self.output = nn.Linear(16, 5)
                
            def forward(self, x):
                return self.output(torch.relu(self.linear(x)))
        
        model = QuickTestModel()
        model.eval()
        inputs = torch.randn(1, 32)
        
        # Rapid successive exports
        results = []
        start_time = time.time()
        
        for i in range(10):
            exporter = HierarchyExporter(strategy="usage_based")
            
            with tempfile.NamedTemporaryFile(suffix=f'_rapid_{i}.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                results.append(result is not None)
        
        total_time = time.time() - start_time
        print(f"10 rapid exports took {total_time:.2f} seconds")
        
        # All should succeed
        assert all(results), "Some rapid exports failed"
        
        # Should complete reasonably quickly
        assert total_time < 10.0, "Rapid exports took too long"
    
    def test_export_with_memory_pressure(self):
        """Test export under simulated memory pressure."""
        class MemoryPressureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(256, 256) for _ in range(8)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = MemoryPressureModel()
        model.eval()
        inputs = torch.randn(16, 256)  # Larger batch
        
        # Create some memory pressure by allocating large tensors
        memory_pressure = []
        try:
            # Allocate some memory to create pressure
            for i in range(5):
                memory_pressure.append(torch.randn(1000, 1000))
            
            exporter = HierarchyExporter(strategy="htp")
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                assert result is not None
                
        finally:
            # Clean up memory pressure
            del memory_pressure
            gc.collect()


if __name__ == "__main__":
    # Run performance and stress tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])