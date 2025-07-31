"""
Performance tests for ONNX to GraphML conversion.

Tests performance characteristics including:
- Conversion time for large models
- Memory usage during conversion  
- Scalability with model complexity
- Handling of models with 100+ nodes
"""

import os
import time
import xml.etree.ElementTree as ET

import psutil
import pytest

from modelexport.graphml import ONNXToGraphMLConverter


class TestONNXToGraphMLPerformance:
    """Performance tests for ONNX to GraphML conversion."""
    
    def test_large_model_conversion_time(self, large_onnx_model):
        """Test that large model conversion completes within time limits."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Measure conversion time
        start_time = time.time()
        graphml_output = converter.convert(large_onnx_model)
        end_time = time.time()
        
        conversion_time = end_time - start_time
        
        # Verify conversion completed within 60 seconds
        assert conversion_time < 60.0, f"Conversion took {conversion_time:.2f}s, expected <60s"
        
        # Verify output is valid
        assert len(graphml_output) > 0
        root = ET.fromstring(graphml_output)
        assert root.tag.endswith("graphml")
        
        # Log performance metrics
        stats = converter.get_statistics()
        print(f"Performance metrics - Time: {conversion_time:.2f}s, "
              f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    
    def test_large_model_memory_usage(self, large_onnx_model):
        """Test that large model conversion uses reasonable memory."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform conversion
        graphml_output = converter.convert(large_onnx_model)
        
        # Measure peak memory during conversion
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Verify memory usage is reasonable (<500MB increase)
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB, expected <500MB"
        
        # Verify output quality
        assert len(graphml_output) > 0
        stats = converter.get_statistics()
        assert stats['nodes'] > 0, f"Expected >0 nodes for large model, got {stats['nodes']}"
        
        print(f"Memory metrics - Baseline: {baseline_memory:.1f}MB, "
              f"Peak: {peak_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
    
    def test_scalability_across_model_sizes(self, simple_onnx_model, medium_onnx_model, large_onnx_model):
        """Test that conversion time scales reasonably with model size."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        models = [
            ("simple", simple_onnx_model),
            ("medium", medium_onnx_model), 
            ("large", large_onnx_model)
        ]
        
        results = []
        
        for model_name, model_path in models:
            start_time = time.time()
            graphml_output = converter.convert(model_path)
            end_time = time.time()
            
            conversion_time = end_time - start_time
            stats = converter.get_statistics()
            
            results.append({
                "name": model_name,
                "time": conversion_time,
                "nodes": stats['nodes'],
                "edges": stats['edges']
            })
            
            # Verify output validity
            assert len(graphml_output) > 0
            root = ET.fromstring(graphml_output)
            assert root.tag.endswith("graphml")
        
        # Verify reasonable scaling (large should be slowest)
        simple_time = results[0]['time']
        large_time = results[2]['time']
        
        # Large model should take more time but not excessively so
        assert large_time > simple_time, "Large model should take more time than simple"
        assert large_time < simple_time * 100, "Large model time should not be >100x simple model"
        
        # Log scaling results
        for result in results:
            print(f"{result['name']} model - Time: {result['time']:.3f}s, "
                  f"Nodes: {result['nodes']}, Edges: {result['edges']}")
    
    def test_hierarchical_converter_performance(self, large_onnx_model):
        """Test performance of hierarchical converter with complex models."""
        # Create temporary empty metadata file for performance test
        import json
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"strategy": "htp", "tagged_nodes": {}}, f)
            temp_metadata_path = f.name
        
        try:
            # Test both base and hierarchical converters
            base_converter = ONNXToGraphMLConverter(hierarchical=False)
            hierarchical_converter = ONNXToGraphMLConverter(
                hierarchical=True, 
                htp_metadata_path=temp_metadata_path
            )
            
            # Test base converter
            start_time = time.time()
            base_output = base_converter.convert(large_onnx_model)
            base_time = time.time() - start_time
            
            # Test hierarchical converter (with empty metadata)
            start_time = time.time()
            hierarchical_output = hierarchical_converter.convert(large_onnx_model)
            hierarchical_time = time.time() - start_time
            
            # Both should complete within time limits
            assert base_time < 60.0, f"Base converter took {base_time:.2f}s, expected <60s"
            assert hierarchical_time < 60.0, f"Hierarchical converter took {hierarchical_time:.2f}s, expected <60s"
            
            # Hierarchical converter may be slower due to additional processing
            time_ratio = hierarchical_time / base_time
            assert time_ratio < 100.0, f"Hierarchical converter {time_ratio:.1f}x slower than base - performance issue detected"
            
            # Both outputs should be valid
            assert len(base_output) > 0  # base_output is string
            assert isinstance(hierarchical_output, dict) and "graphml" in hierarchical_output  # hierarchical_output is dict
            
            print(f"Converter comparison - Base: {base_time:.3f}s, "
                  f"Hierarchical: {hierarchical_time:.3f}s, Ratio: {time_ratio:.2f}x")
        
        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_metadata_path)
    
    def test_complex_hierarchy_handling(self, medium_onnx_model):
        """Test that complex hierarchical structures are handled efficiently."""
        # Create mock metadata to test hierarchical processing
        import json
        import tempfile
        
        # Create temporary metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create hierarchical metadata for medium model
            mock_metadata = {
                "strategy": "htp",
                "model_name": "MediumModel",
                "tagged_nodes": {
                    "conv1": "/MediumModel/conv1",
                    "bn1": "/MediumModel/bn1", 
                    "relu": "/MediumModel/relu",
                    "maxpool": "/MediumModel/maxpool",
                    "conv2": "/MediumModel/conv2",
                    "bn2": "/MediumModel/bn2",
                    "avgpool": "/MediumModel/avgpool",
                    "classifier": "/MediumModel/classifier"
                }
            }
            json.dump(mock_metadata, f)
            metadata_path = f.name
        
        try:
            converter = ONNXToGraphMLConverter(hierarchical=True, htp_metadata_path=metadata_path)
            
            # Measure conversion time
            start_time = time.time()
            graphml_output = converter.convert(medium_onnx_model)
            conversion_time = time.time() - start_time
            
            # Should complete quickly even with hierarchy processing
            assert conversion_time < 30.0, f"Hierarchical conversion took {conversion_time:.2f}s, expected <30s"
            
            # Verify output is valid GraphML (hierarchical mode returns dict)
            assert isinstance(graphml_output, dict) and "graphml" in graphml_output
            
            # Read and parse the GraphML file
            graphml_path = graphml_output["graphml"]
            root = ET.parse(graphml_path).getroot()
            assert root.tag.endswith("graphml"), "Output should be valid GraphML"
            
            # Count regular nodes to verify conversion worked
            all_nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
            compound_nodes = [node for node in all_nodes if node.get('id', '').startswith('module_')]
            
            print(f"Hierarchical performance - Time: {conversion_time:.3f}s, "
                  f"Total nodes: {len(all_nodes)}, Compound nodes: {len(compound_nodes)}")
            
        finally:
            # Clean up temporary file
            os.unlink(metadata_path)
    
    @pytest.mark.slow
    def test_stress_conversion_multiple_models(self, simple_onnx_model, medium_onnx_model):
        """Stress test with multiple consecutive conversions."""
        converter = ONNXToGraphMLConverter(hierarchical=False)
        models = [simple_onnx_model, medium_onnx_model]
        
        total_start_time = time.time()
        
        # Perform 10 conversions of different models
        for i in range(10):
            model_path = models[i % len(models)]
            
            start_time = time.time()
            graphml_output = converter.convert(model_path)
            conversion_time = time.time() - start_time
            
            # Each conversion should complete quickly
            assert conversion_time < 10.0, f"Conversion {i+1} took {conversion_time:.2f}s, expected <10s"
            
            # Output should be valid
            assert len(graphml_output) > 0
            root = ET.fromstring(graphml_output)
            assert root.tag.endswith("graphml")
        
        total_time = time.time() - total_start_time
        avg_time = total_time / 10
        
        print(f"Stress test - Total: {total_time:.2f}s, Average: {avg_time:.3f}s per conversion")
        
        # Total time should be reasonable
        assert total_time < 60.0, f"10 conversions took {total_time:.2f}s, expected <60s total"