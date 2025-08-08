"""
Test cases for GraphML performance profiling and monitoring.

This test suite validates the performance profiling system including:
- Operation timing and throughput measurement
- Memory usage tracking and analysis
- System resource monitoring
- Performance issue detection and recommendations
- Metrics export functionality

Linear Task: TEZ-133 (Code Quality Improvements)
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from modelexport.graphml.profiling import (
    PerformanceProfiler,
    PerformanceMetrics,
    ResourceUsage,
    get_profiler,
    profile_operation,
    profile_function
)

# Add timeout for potentially long-running tests
pytestmark = pytest.mark.timeout(30)


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            operation="test_op",
            start_time=1000.0,
            end_time=1001.0,
            duration_ms=1000.0,
            cpu_percent=25.5,
            memory_mb=128.0,
            peak_memory_mb=150.0,
            thread_count=4,
            node_count=1000,
            edge_count=2000
        )
        
        assert metrics.operation == "test_op"
        assert metrics.duration_ms == 1000.0
        assert metrics.node_count == 1000
        assert metrics.edge_count == 2000
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = PerformanceMetrics(
            operation="test_op",
            start_time=1000.0,
            end_time=1001.0,
            duration_ms=2000.0,  # 2 seconds
            cpu_percent=25.5,
            memory_mb=128.0,
            peak_memory_mb=150.0,
            thread_count=4,
            node_count=4000  # 4000 nodes in 2 seconds = 2000 nodes/sec
        )
        
        assert metrics.throughput_nodes_per_sec == 2000.0
    
    def test_memory_efficiency_calculation(self):
        """Test memory efficiency calculation."""
        metrics = PerformanceMetrics(
            operation="test_op",
            start_time=1000.0,
            end_time=1001.0,
            duration_ms=1000.0,
            cpu_percent=25.5,
            memory_mb=200.0,  # 200MB for 100 nodes = 2MB per node
            peak_memory_mb=220.0,
            thread_count=4,
            node_count=100
        )
        
        assert metrics.memory_efficiency_mb_per_node == 2.0
    
    def test_throughput_with_no_nodes(self):
        """Test throughput when node count is None."""
        metrics = PerformanceMetrics(
            operation="test_op",
            start_time=1000.0,
            end_time=1001.0,
            duration_ms=1000.0,
            cpu_percent=25.5,
            memory_mb=128.0,
            peak_memory_mb=150.0,
            thread_count=4,
            node_count=None
        )
        
        assert metrics.throughput_nodes_per_sec is None
        assert metrics.memory_efficiency_mb_per_node is None


class TestResourceUsage:
    """Test cases for ResourceUsage dataclass."""
    
    def test_resource_usage_creation(self):
        """Test creating resource usage snapshot."""
        usage = ResourceUsage(
            timestamp=1234567890.0,
            cpu_percent=45.2,
            memory_mb=512.0,
            memory_percent=25.0,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            thread_count=8
        )
        
        assert usage.timestamp == 1234567890.0
        assert usage.cpu_percent == 45.2
        assert usage.memory_mb == 512.0
        assert usage.thread_count == 8


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler class."""
    
    @pytest.fixture
    def profiler(self):
        """Create a test profiler instance."""
        return PerformanceProfiler(
            enable_monitoring=True,
            sampling_interval=0.01,  # Fast sampling for testing
            history_size=100
        )
    
    @pytest.fixture
    def disabled_profiler(self):
        """Create a disabled profiler for testing disabled state."""
        return PerformanceProfiler(enable_monitoring=False)
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.enable_monitoring is True
        assert profiler.sampling_interval == 0.01
        assert profiler.history_size == 100
        assert len(profiler.metrics_history) == 0
        assert len(profiler.operation_stats) == 0
    
    def test_disabled_profiler(self, disabled_profiler):
        """Test that disabled profiler doesn't monitor."""
        assert disabled_profiler.enable_monitoring is False
        assert disabled_profiler._monitoring_active is False
    
    @pytest.mark.timeout(5)
    def test_profile_operation_context_manager(self, profiler):
        """Test profiling an operation using context manager."""
        operation_name = "test_operation"
        
        with profiler.profile_operation(
            operation_name,
            file_size_bytes=1024,
            node_count=100,
            edge_count=200
        ) as metrics:
            # Simulate some work
            time.sleep(0.1)
            
            # Verify metrics object exists
            assert metrics is not None
            assert metrics.operation == operation_name
            assert metrics.file_size_bytes == 1024
            assert metrics.node_count == 100
            assert metrics.edge_count == 200
        
        # Verify metrics were recorded
        assert len(profiler.metrics_history) == 1
        assert operation_name in profiler.operation_stats
        
        recorded_metrics = profiler.metrics_history[0]
        assert recorded_metrics.operation == operation_name
        assert recorded_metrics.duration_ms >= 100  # At least 100ms from sleep
        assert recorded_metrics.node_count == 100
        assert recorded_metrics.edge_count == 200
    
    def test_profile_operation_disabled(self, disabled_profiler):
        """Test profiling when disabled."""
        with disabled_profiler.profile_operation("test_op") as metrics:
            assert metrics is None
        
        assert len(disabled_profiler.metrics_history) == 0
    
    @pytest.mark.timeout(5)
    def test_profile_function_decorator(self, profiler):
        """Test the profile_function decorator."""
        
        @profiler.profile_function("decorated_function")
        def test_function(x, y):
            time.sleep(0.05)  # Simulate work
            return x + y
        
        result = test_function(5, 3)
        assert result == 8
        
        # Verify profiling occurred
        assert len(profiler.metrics_history) == 1
        assert "decorated_function" in profiler.operation_stats
        
        metrics = profiler.metrics_history[0]
        assert metrics.operation == "decorated_function"
        assert metrics.duration_ms >= 50  # At least 50ms from sleep
    
    @pytest.mark.timeout(5)
    def test_operation_summary(self, profiler):
        """Test getting operation performance summary."""
        operation_name = "summary_test"
        
        # Perform multiple operations
        for i in range(3):
            with profiler.profile_operation(operation_name, node_count=100 + i * 10):
                time.sleep(0.02)  # Small delay
        
        summary = profiler.get_operation_summary(operation_name)
        
        assert summary["operation"] == operation_name
        assert summary["total_executions"] == 3
        assert "duration_ms" in summary
        assert "memory_mb" in summary
        assert summary["duration_ms"]["min"] >= 20  # At least 20ms
        assert summary["duration_ms"]["max"] >= summary["duration_ms"]["min"]
        assert summary["duration_ms"]["avg"] > 0
    
    def test_operation_summary_no_data(self, profiler):
        """Test operation summary with no data."""
        summary = profiler.get_operation_summary("nonexistent_operation")
        assert "error" in summary
        assert "No data for operation" in summary["error"]
    
    @pytest.mark.timeout(5) 
    def test_system_health(self, profiler):
        """Test system health monitoring."""
        # Wait a bit for samples to be collected
        time.sleep(0.2)
        
        health = profiler.get_system_health()
        
        assert "timestamp" in health
        assert "current" in health
        assert "recent_avg" in health
        assert "monitoring_active" in health
        assert health["monitoring_active"] is True
        
        # Check current metrics structure
        current = health["current"]
        assert "cpu_percent" in current
        assert "memory_mb" in current
        assert "thread_count" in current
        
        # Values should be reasonable
        assert current["cpu_percent"] >= 0
        assert current["memory_mb"] > 0
        assert current["thread_count"] > 0
    
    def test_performance_analysis_high_memory(self, profiler):
        """Test performance analysis for high memory usage."""
        with patch.object(profiler.logger, 'warning') as mock_warning:
            # Create metrics with high memory usage
            high_memory_metrics = PerformanceMetrics(
                operation="memory_test",
                start_time=1000.0,
                end_time=1001.0,
                duration_ms=500.0,
                cpu_percent=25.0,
                memory_mb=500.0,
                peak_memory_mb=1500.0,  # > 1GB threshold
                thread_count=4
            )
            
            profiler._analyze_performance(high_memory_metrics)
            
            # Should log warning about high memory usage
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args
            assert "performance_issues_detected" in call_args[0][0]
    
    def test_performance_analysis_low_throughput(self, profiler):
        """Test performance analysis for low throughput."""
        with patch.object(profiler.logger, 'warning') as mock_warning:
            # Create metrics with low throughput
            low_throughput_metrics = PerformanceMetrics(
                operation="throughput_test",
                start_time=1000.0,
                end_time=1010.0,
                duration_ms=10000.0,  # 10 seconds
                cpu_percent=25.0,
                memory_mb=100.0,
                peak_memory_mb=120.0,
                thread_count=4,
                node_count=500  # 500 nodes in 10 seconds = 50 nodes/sec (< 100 threshold)
            )
            
            profiler._analyze_performance(low_throughput_metrics)
            
            # Should log warning about low throughput
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args
            assert "performance_issues_detected" in call_args[0][0]
    
    def test_metrics_export_json(self, profiler, tmp_path):
        """Test exporting metrics to JSON format."""
        # Generate some test metrics
        with profiler.profile_operation("export_test", node_count=50):
            time.sleep(0.01)
        
        output_file = tmp_path / "metrics.json"
        profiler.export_metrics(str(output_file), format="json")
        
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert "export_timestamp" in data
        assert "profiler_config" in data
        assert "operation_summaries" in data
        assert "system_health" in data
        assert "raw_metrics" in data
        
        # Check raw metrics
        assert len(data["raw_metrics"]) == 1
        raw_metric = data["raw_metrics"][0]
        assert raw_metric["operation"] == "export_test"
        assert raw_metric["node_count"] == 50
    
    def test_metrics_export_csv(self, profiler, tmp_path):
        """Test exporting metrics to CSV format."""
        # Generate some test metrics
        with profiler.profile_operation("csv_test", node_count=25, edge_count=50):
            time.sleep(0.01)
        
        output_file = tmp_path / "metrics.csv"
        profiler.export_metrics(str(output_file), format="csv")
        
        assert output_file.exists()
        
        # Verify CSV content
        with open(output_file) as f:
            lines = f.readlines()
        
        assert len(lines) >= 2  # Header + at least one data row
        
        # Check header
        header = lines[0].strip().split(',')
        expected_columns = [
            "operation", "duration_ms", "memory_mb", "peak_memory_mb",
            "cpu_percent", "node_count", "edge_count", "throughput_nodes_per_sec",
            "start_time", "end_time"
        ]
        assert header == expected_columns
        
        # Check data row
        data_row = lines[1].strip().split(',')
        assert data_row[0] == "csv_test"  # operation
        assert data_row[5] == "25"  # node_count
        assert data_row[6] == "50"  # edge_count
    
    def test_metrics_export_invalid_format(self, profiler, tmp_path):
        """Test exporting with invalid format."""
        output_file = tmp_path / "metrics.invalid"
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            profiler.export_metrics(str(output_file), format="invalid")


class TestGlobalProfilerFunctions:
    """Test cases for global profiler functions."""
    
    def test_get_profiler_singleton(self):
        """Test that get_profiler returns the same instance."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        
        assert profiler1 is profiler2
        assert isinstance(profiler1, PerformanceProfiler)
    
    @pytest.mark.timeout(5)
    def test_profile_operation_function(self):
        """Test the global profile_operation function."""
        with profile_operation("global_test", node_count=75) as metrics:
            time.sleep(0.01)
            assert metrics is not None
            assert metrics.operation == "global_test"
            assert metrics.node_count == 75
    
    @pytest.mark.timeout(5)
    def test_profile_function_decorator(self):
        """Test the global profile_function decorator."""
        
        @profile_function("global_decorated")
        def test_func():
            time.sleep(0.01)
            return "test_result"
        
        result = test_func()
        assert result == "test_result"
        
        # Check that profiling occurred
        profiler = get_profiler()
        assert len(profiler.metrics_history) > 0
        
        # Find our operation in the metrics
        found_operation = False
        for metrics in profiler.metrics_history:
            if metrics.operation == "global_decorated":
                found_operation = True
                break
        
        assert found_operation


class TestEnvironmentConfiguration:
    """Test cases for environment-based configuration."""
    
    def test_profiler_disabled_by_environment(self):
        """Test disabling profiler via environment variable."""
        with patch.dict('os.environ', {'GRAPHML_ENABLE_PROFILING': 'false'}):
            # Clear global profiler to test fresh creation
            import modelexport.graphml.profiling
            modelexport.graphml.profiling._global_profiler = None
            
            profiler = get_profiler()
            assert profiler.enable_monitoring is False
    
    def test_profiler_custom_config_by_environment(self):
        """Test custom profiler configuration via environment."""
        with patch.dict('os.environ', {
            'GRAPHML_ENABLE_PROFILING': 'true',
            'GRAPHML_PROFILING_INTERVAL': '0.5',
            'GRAPHML_PROFILING_HISTORY': '2000'
        }):
            # Clear global profiler to test fresh creation
            import modelexport.graphml.profiling
            modelexport.graphml.profiling._global_profiler = None
            
            profiler = get_profiler()
            assert profiler.enable_monitoring is True
            assert profiler.sampling_interval == 0.5
            assert profiler.history_size == 2000


class TestIntegrationScenarios:
    """Integration test cases simulating real usage."""
    
    @pytest.mark.timeout(10)
    def test_complete_profiling_workflow(self):
        """Test a complete profiling workflow."""
        profiler = PerformanceProfiler(
            enable_monitoring=True,
            sampling_interval=0.01,
            history_size=50
        )
        
        # Simulate multiple operations
        operations = [
            ("convert_onnx_to_graphml", 1000, 500),
            ("validate_graphml", 1000, 500),
            ("convert_graphml_to_onnx", 800, 400)
        ]
        
        for op_name, node_count, edge_count in operations:
            with profiler.profile_operation(
                op_name,
                file_size_bytes=1024000,  # 1MB file
                node_count=node_count,
                edge_count=edge_count
            ):
                # Simulate processing time proportional to node count
                time.sleep(node_count / 50000.0)  # 0.02s for 1000 nodes
        
        # Verify all operations were recorded
        assert len(profiler.metrics_history) == 3
        assert len(profiler.operation_stats) == 3
        
        # Check operation summaries
        for op_name, _, _ in operations:
            summary = profiler.get_operation_summary(op_name)
            assert summary["total_executions"] == 1
            assert summary["duration_ms"]["avg"] > 0
        
        # Check system health
        health = profiler.get_system_health()
        assert health["monitoring_active"] is True
        assert "current" in health
        
        # Export metrics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            profiler.export_metrics(f.name, format="json")
            
            # Verify export file exists and contains data
            export_path = Path(f.name)
            assert export_path.exists()
            
            with open(export_path) as export_file:
                export_data = json.load(export_file)
                assert len(export_data["raw_metrics"]) == 3
                assert len(export_data["operation_summaries"]) == 3
        
        # Cleanup
        export_path.unlink()
    
    @pytest.mark.timeout(5)
    def test_exception_handling_during_profiling(self):
        """Test that profiling handles exceptions gracefully."""
        profiler = PerformanceProfiler(enable_monitoring=True)
        
        with pytest.raises(ValueError, match="Test exception"):
            with profiler.profile_operation("exception_test") as metrics:
                assert metrics is not None
                raise ValueError("Test exception")
        
        # Verify metrics were still recorded despite exception
        assert len(profiler.metrics_history) == 1
        recorded_metrics = profiler.metrics_history[0]
        assert recorded_metrics.operation == "exception_test"
        assert recorded_metrics.duration_ms > 0