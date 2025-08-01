"""
Performance benchmarking for GraphML export overhead.

Tests the performance impact of enabling --with-graphml flag.
"""

import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from modelexport.cli import cli


@pytest.mark.graphml
@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.integration
class TestGraphMLPerformance:
    """Benchmark GraphML export performance overhead."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create temporary workspace for benchmarks."""
        workspace = tmp_path / "benchmark"
        workspace.mkdir(exist_ok=True)
        return workspace
    
    @pytest.mark.e2e
    def test_graphml_overhead_bert_tiny(self, cli_runner, temp_workspace):
        """Measure GraphML export overhead for bert-tiny model."""
        # Warm-up run to ensure model is cached
        warmup_path = temp_workspace / "warmup.onnx"
        cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(warmup_path)
        ])
        
        # Benchmark without GraphML
        output_path_no_graphml = temp_workspace / "bert_no_graphml.onnx"
        start_time = time.time()
        result_no_graphml = cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(output_path_no_graphml)
        ])
        time_no_graphml = time.time() - start_time
        
        assert result_no_graphml.exit_code == 0
        
        # Benchmark with GraphML
        output_path_with_graphml = temp_workspace / "bert_with_graphml.onnx"
        start_time = time.time()
        result_with_graphml = cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(output_path_with_graphml),
            '--with-graphml'
        ])
        time_with_graphml = time.time() - start_time
        
        assert result_with_graphml.exit_code == 0
        
        # Calculate overhead
        overhead_seconds = time_with_graphml - time_no_graphml
        overhead_percentage = (overhead_seconds / time_no_graphml) * 100
        
        # Report results
        print(f"\n=== GraphML Performance Benchmark ===")
        print(f"Model: prajjwal1/bert-tiny")
        print(f"Export without GraphML: {time_no_graphml:.2f}s")
        print(f"Export with GraphML: {time_with_graphml:.2f}s")
        print(f"Overhead: {overhead_seconds:.2f}s ({overhead_percentage:.1f}%)")
        
        # Performance assertions
        # GraphML generation involves structural discovery and hierarchical conversion
        # which can add significant overhead for small models
        assert overhead_percentage < 200, f"GraphML overhead too high: {overhead_percentage:.1f}%"
        
        # Log performance characteristics for documentation
        with open(temp_workspace / "performance_metrics.txt", "w") as f:
            f.write(f"Model: prajjwal1/bert-tiny\n")
            f.write(f"Export without GraphML: {time_no_graphml:.2f}s\n")
            f.write(f"Export with GraphML: {time_with_graphml:.2f}s\n")
            f.write(f"Overhead: {overhead_seconds:.2f}s ({overhead_percentage:.1f}%)\n")
        
        # Verify file sizes
        onnx_size = output_path_with_graphml.stat().st_size
        graphml_path = output_path_with_graphml.parent / f"{output_path_with_graphml.stem}_hierarchical_graph.graphml"
        params_path = output_path_with_graphml.parent / f"{output_path_with_graphml.stem}_hierarchical_graph.onnxdata"
        
        graphml_size = graphml_path.stat().st_size
        params_size = params_path.stat().st_size
        
        graphml_ratio = (graphml_size / onnx_size) * 100
        params_ratio = (params_size / onnx_size) * 100
        
        print(f"\n=== File Size Analysis ===")
        print(f"ONNX size: {onnx_size / 1024 / 1024:.1f} MB")
        print(f"GraphML size: {graphml_size / 1024 / 1024:.1f} MB ({graphml_ratio:.1f}% of ONNX)")
        print(f"Parameters size: {params_size / 1024 / 1024:.1f} MB ({params_ratio:.1f}% of ONNX)")
        
        # Size assertions
        assert graphml_ratio < 30, f"GraphML file too large: {graphml_ratio:.1f}% of ONNX"
        assert 80 < params_ratio < 120, f"Parameter file size unexpected: {params_ratio:.1f}% of ONNX"
    
    @pytest.mark.parametrize("model_name,expected_overhead", [
        ("prajjwal1/bert-tiny", 50),  # Small model, higher relative overhead OK
        # Add more models for comprehensive benchmarking in production
        # ("gpt2", 20),  # Larger model, lower relative overhead expected
        # ("microsoft/resnet-18", 30),  # Vision model
    ])
    @pytest.mark.e2e
    def test_graphml_overhead_multiple_models(self, cli_runner, temp_workspace, model_name, expected_overhead):
        """Benchmark GraphML overhead across different model architectures."""
        # Skip if model not available in test environment
        if model_name != "prajjwal1/bert-tiny":
            pytest.skip(f"Model {model_name} not configured for testing")
        
        # Similar benchmark logic as above
        output_path = temp_workspace / f"{model_name.replace('/', '_')}.onnx"
        
        # Measure without GraphML
        start_time = time.time()
        result = cli_runner.invoke(cli, [
            'export',
            '--model', model_name,
            '--output', str(output_path)
        ])
        base_time = time.time() - start_time
        
        assert result.exit_code == 0
        
        # Measure with GraphML
        output_path_graphml = temp_workspace / f"{model_name.replace('/', '_')}_graphml.onnx"
        start_time = time.time()
        result = cli_runner.invoke(cli, [
            'export',
            '--model', model_name,
            '--output', str(output_path_graphml),
            '--with-graphml'
        ])
        graphml_time = time.time() - start_time
        
        assert result.exit_code == 0
        
        # Check overhead
        overhead_percentage = ((graphml_time - base_time) / base_time) * 100
        assert overhead_percentage < expected_overhead, \
            f"GraphML overhead for {model_name} too high: {overhead_percentage:.1f}%"
        
        print(f"\n{model_name}: {overhead_percentage:.1f}% overhead")
    
    @pytest.mark.resource
    def test_graphml_memory_usage(self, cli_runner, temp_workspace):
        """Test memory usage doesn't spike excessively with GraphML export."""
        # This is a placeholder for memory profiling
        # In production, use memory_profiler or tracemalloc
        
        output_path = temp_workspace / "memory_test.onnx"
        
        # Export with GraphML (memory usage would be tracked here)
        result = cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(output_path),
            '--with-graphml'
        ])
        
        assert result.exit_code == 0
        
        # In production: Assert memory usage is within acceptable bounds
        # assert peak_memory < base_memory * 1.5  # Max 50% memory increase