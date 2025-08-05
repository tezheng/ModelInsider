"""
Performance profiling and monitoring for GraphML operations.

This module provides comprehensive performance monitoring with metrics collection,
bottleneck detection, and resource usage tracking for all GraphML operations.
It integrates with the logging system to provide actionable insights.

Linear Task: TEZ-133 (Code Quality Improvements)
"""

import os
import time
import psutil
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from .constants import GRAPHML_CONST
from .logging import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    cpu_percent: float
    memory_mb: float
    peak_memory_mb: float
    thread_count: int
    file_size_bytes: Optional[int] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    
    @property
    def throughput_nodes_per_sec(self) -> Optional[float]:
        """Calculate nodes processed per second."""
        if self.node_count and self.duration_ms > 0:
            return (self.node_count * 1000) / self.duration_ms
        return None
    
    @property
    def memory_efficiency_mb_per_node(self) -> Optional[float]:
        """Calculate memory usage per node."""
        if self.node_count and self.node_count > 0:
            return self.memory_mb / self.node_count
        return None


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    thread_count: int


class PerformanceProfiler:
    """
    Performance profiler with real-time monitoring and bottleneck detection.
    
    This class provides comprehensive performance monitoring including:
    - Operation timing and throughput measurement
    - Memory usage tracking and leak detection
    - CPU utilization monitoring
    - Bottleneck identification and recommendations
    - Performance regression detection
    """
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        sampling_interval: float = 0.1,
        history_size: int = 1000
    ):
        """
        Initialize performance profiler.
        
        Args:
            enable_monitoring: Whether to enable real-time monitoring
            sampling_interval: Interval between resource samples (seconds)
            history_size: Number of metrics to keep in memory
        """
        self.enable_monitoring = enable_monitoring
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.resource_samples: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Process reference for system metrics
        self._process = psutil.Process()
        
        # Logger
        self.logger = get_logger(__name__)
        
        # Performance baselines (will be updated as we collect data)
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        if self.enable_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="GraphML-Profiler"
        )
        self._monitor_thread.start()
        self.logger.debug("performance_monitoring_started")
    
    def _stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        self.logger.debug("performance_monitoring_stopped")
    
    def _monitor_resources(self) -> None:
        """Background thread for resource monitoring."""
        while not self._stop_event.wait(self.sampling_interval):
            try:
                # Get current resource usage
                cpu_percent = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / GRAPHML_CONST.BYTES_PER_MB
                memory_percent = self._process.memory_percent()
                
                # Get disk I/O (if available)
                try:
                    io_counters = self._process.io_counters()
                    disk_read_mb = io_counters.read_bytes / GRAPHML_CONST.BYTES_PER_MB
                    disk_write_mb = io_counters.write_bytes / GRAPHML_CONST.BYTES_PER_MB
                except (psutil.AccessDenied, AttributeError):
                    disk_read_mb = disk_write_mb = 0.0
                
                thread_count = self._process.num_threads()
                
                # Store sample
                sample = ResourceUsage(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    thread_count=thread_count
                )
                
                self.resource_samples.append(sample)
                
            except Exception as e:
                self.logger.error(f"resource_monitoring_error: {e}")
                break
    
    @contextmanager
    def profile_operation(
        self,
        operation: str,
        file_size_bytes: Optional[int] = None,
        node_count: Optional[int] = None,
        edge_count: Optional[int] = None
    ):
        """
        Context manager for profiling operations.
        
        Args:
            operation: Name of the operation being profiled
            file_size_bytes: Size of input file
            node_count: Number of nodes being processed
            edge_count: Number of edges being processed
            
        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        if not self.enable_monitoring:
            yield None
            return
        
        # Get initial state
        start_time = time.perf_counter()
        initial_memory = self._process.memory_info().rss / GRAPHML_CONST.BYTES_PER_MB
        initial_cpu = self._process.cpu_percent()
        
        # Track peak memory during operation
        peak_memory = initial_memory
        
        try:
            # Create metrics object
            metrics = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=0.0,  # Will be set in finally
                duration_ms=0.0,  # Will be calculated in finally
                cpu_percent=initial_cpu,
                memory_mb=initial_memory,
                peak_memory_mb=peak_memory,
                thread_count=self._process.num_threads(),
                file_size_bytes=file_size_bytes,
                node_count=node_count,
                edge_count=edge_count
            )
            
            self.logger.info(
                f"operation_started: {operation} - "
                f"memory: {initial_memory:.1f}MB, file: {file_size_bytes} bytes, nodes: {node_count}"
            )
            
            yield metrics
            
        finally:
            # Calculate final metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            final_memory = self._process.memory_info().rss / GRAPHML_CONST.BYTES_PER_MB
            final_cpu = self._process.cpu_percent()
            
            # Update metrics
            metrics.end_time = end_time
            metrics.duration_ms = duration_ms
            metrics.cpu_percent = max(initial_cpu, final_cpu)
            metrics.memory_mb = final_memory
            
            # Calculate peak memory from samples during operation
            operation_samples = [
                s for s in self.resource_samples 
                if start_time <= s.timestamp <= end_time
            ]
            if operation_samples:
                metrics.peak_memory_mb = max(s.memory_mb for s in operation_samples)
            else:
                metrics.peak_memory_mb = max(initial_memory, final_memory)
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.operation_stats[operation].append(metrics)
            
            # Log completion with performance data
            self.logger.info(
                f"operation_completed: {operation} - "
                f"duration: {duration_ms:.1f}ms, memory: {final_memory:.1f}MB, "
                f"peak: {metrics.peak_memory_mb:.1f}MB, throughput: {metrics.throughput_nodes_per_sec} nodes/sec"
            )
            
            # Check for performance issues
            self._analyze_performance(metrics)
    
    def _analyze_performance(self, metrics: PerformanceMetrics) -> None:
        """Analyze performance metrics and log recommendations."""
        issues = []
        recommendations = []
        
        # Check memory usage
        if metrics.peak_memory_mb > 1000:  # > 1GB
            issues.append(f"High memory usage: {metrics.peak_memory_mb:.1f}MB")
            recommendations.append("Consider processing in smaller batches")
        
        # Check duration against thresholds
        if metrics.duration_ms > GRAPHML_CONST.CONVERSION_TIMEOUT * 1000:
            issues.append(f"Long operation duration: {metrics.duration_ms:.1f}ms")
            recommendations.append("Consider optimizing the operation or increasing timeout")
        
        # Check throughput for node processing
        if metrics.throughput_nodes_per_sec and metrics.throughput_nodes_per_sec < 100:
            issues.append(f"Low throughput: {metrics.throughput_nodes_per_sec:.1f} nodes/sec")
            recommendations.append("Check for bottlenecks in node processing logic")
        
        # Check memory efficiency
        if metrics.memory_efficiency_mb_per_node and metrics.memory_efficiency_mb_per_node > 1.0:
            issues.append(f"High memory per node: {metrics.memory_efficiency_mb_per_node:.2f}MB/node")
            recommendations.append("Optimize data structures and memory usage")
        
        # Log performance analysis
        if issues:
            # Use simple logging to avoid structured logging conflicts
            issues_str = "; ".join(issues)
            recommendations_str = "; ".join(recommendations)
            
            self.logger.warning(
                f"performance_issues_detected: {metrics.operation} - "
                f"Issues: {issues_str} - Recommendations: {recommendations_str}"
            )
    
    def profile_function(
        self,
        operation: Optional[str] = None,
        include_args: bool = False
    ) -> Callable:
        """
        Decorator for automatic function profiling.
        
        Args:
            operation: Operation name (defaults to function name)
            include_args: Whether to include function arguments in logging
            
        Returns:
            Decorated function with automatic profiling
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Extract potential profiling hints from arguments
                file_size = None
                node_count = None
                edge_count = None
                
                # Try to extract common parameters
                if 'file_path' in kwargs:
                    file_path = kwargs['file_path']
                    try:
                        file_size = os.path.getsize(file_path)
                    except (OSError, TypeError):
                        pass
                
                if 'node_count' in kwargs:
                    node_count = kwargs['node_count']
                if 'edge_count' in kwargs:
                    edge_count = kwargs['edge_count']
                
                with self.profile_operation(
                    op_name,
                    file_size_bytes=file_size,
                    node_count=node_count,
                    edge_count=edge_count
                ) as metrics:
                    result = func(*args, **kwargs)
                    
                    # Try to extract counts from result if it's a statistics object
                    if hasattr(result, 'node_count') and metrics:
                        metrics.node_count = result.node_count
                    if hasattr(result, 'edge_count') and metrics:
                        metrics.edge_count = result.edge_count
                    
                    return result
            
            return wrapper
        return decorator
    
    def get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with performance statistics
        """
        if operation not in self.operation_stats:
            return {"error": f"No data for operation: {operation}"}
        
        metrics_list = self.operation_stats[operation]
        if not metrics_list:
            return {"error": f"No metrics for operation: {operation}"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in metrics_list]
        memory_peaks = [m.peak_memory_mb for m in metrics_list]
        throughputs = [m.throughput_nodes_per_sec for m in metrics_list if m.throughput_nodes_per_sec]
        
        summary = {
            "operation": operation,
            "total_executions": len(metrics_list),
            "duration_ms": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations)
            },
            "memory_mb": {
                "min_peak": min(memory_peaks),
                "max_peak": max(memory_peaks),
                "avg_peak": sum(memory_peaks) / len(memory_peaks)
            }
        }
        
        if throughputs:
            summary["throughput_nodes_per_sec"] = {
                "min": min(throughputs),
                "max": max(throughputs),
                "avg": sum(throughputs) / len(throughputs)
            }
        
        return summary
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health metrics.
        
        Returns:
            Dictionary with current system state
        """
        if not self.resource_samples:
            return {"error": "No resource samples available"}
        
        recent_samples = list(self.resource_samples)[-10:]  # Last 10 samples
        
        current_sample = recent_samples[-1]
        avg_cpu = sum(s.cpu_percent for s in recent_samples) / len(recent_samples)
        avg_memory = sum(s.memory_mb for s in recent_samples) / len(recent_samples)
        
        return {
            "timestamp": current_sample.timestamp,
            "current": {
                "cpu_percent": current_sample.cpu_percent,
                "memory_mb": current_sample.memory_mb,
                "memory_percent": current_sample.memory_percent,
                "thread_count": current_sample.thread_count
            },
            "recent_avg": {
                "cpu_percent": avg_cpu,
                "memory_mb": avg_memory
            },
            "monitoring_active": self._monitoring_active,
            "sample_count": len(self.resource_samples)
        }
    
    def export_metrics(
        self,
        output_file: str,
        format: str = "json"
    ) -> None:
        """
        Export collected metrics to file.
        
        Args:
            output_file: Path to output file
            format: Export format (json, csv)
        """
        import json
        import csv
        from datetime import datetime
        
        if format.lower() == "json":
            # Export as JSON
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "profiler_config": {
                    "sampling_interval": self.sampling_interval,
                    "history_size": self.history_size
                },
                "operation_summaries": {
                    op: self.get_operation_summary(op)
                    for op in self.operation_stats.keys()
                },
                "system_health": self.get_system_health(),
                "raw_metrics": [
                    {
                        "operation": m.operation,
                        "duration_ms": m.duration_ms,
                        "memory_mb": m.memory_mb,
                        "peak_memory_mb": m.peak_memory_mb,
                        "cpu_percent": m.cpu_percent,
                        "node_count": m.node_count,
                        "edge_count": m.edge_count,
                        "throughput_nodes_per_sec": m.throughput_nodes_per_sec,
                        "start_time": m.start_time,
                        "end_time": m.end_time
                    }
                    for m in self.metrics_history
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format.lower() == "csv":
            # Export as CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "operation", "duration_ms", "memory_mb", "peak_memory_mb",
                    "cpu_percent", "node_count", "edge_count", "throughput_nodes_per_sec",
                    "start_time", "end_time"
                ])
                
                # Data rows
                for m in self.metrics_history:
                    writer.writerow([
                        m.operation, m.duration_ms, m.memory_mb, m.peak_memory_mb,
                        m.cpu_percent, m.node_count, m.edge_count, m.throughput_nodes_per_sec,
                        m.start_time, m.end_time
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(
            f"metrics_exported: {output_file} - format: {format}, count: {len(self.metrics_history)}"
        )
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        if hasattr(self, '_monitoring_active') and self._monitoring_active:
            self._stop_monitoring()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create the global performance profiler."""
    global _global_profiler
    
    if _global_profiler is None:
        # Check environment variables for configuration
        enable_profiling = os.getenv("GRAPHML_ENABLE_PROFILING", "true").lower() == "true"
        sampling_interval = float(os.getenv("GRAPHML_PROFILING_INTERVAL", "0.1"))
        history_size = int(os.getenv("GRAPHML_PROFILING_HISTORY", "1000"))
        
        _global_profiler = PerformanceProfiler(
            enable_monitoring=enable_profiling,
            sampling_interval=sampling_interval,
            history_size=history_size
        )
    
    return _global_profiler


def profile_operation(
    operation: str,
    file_size_bytes: Optional[int] = None,
    node_count: Optional[int] = None,
    edge_count: Optional[int] = None
):
    """Convenience function for profiling operations."""
    return get_profiler().profile_operation(
        operation=operation,
        file_size_bytes=file_size_bytes,
        node_count=node_count,
        edge_count=edge_count
    )


def profile_function(
    operation: Optional[str] = None,
    include_args: bool = False
) -> Callable:
    """Convenience decorator for profiling functions."""
    return get_profiler().profile_function(
        operation=operation,
        include_args=include_args
    )