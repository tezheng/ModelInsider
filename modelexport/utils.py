"""
Utility functions for ModelExport including performance measurement and profiling.
"""

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

import psutil


class PerformanceMeasurement:
    """
    Performance measurement and profiling utilities for ModelExport operations.
    """
    
    def __init__(self):
        self.measurements = {}
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }
    
    def timing_decorator(self, func_name: str | None = None):
        """
        Decorator to measure function execution time and memory usage.
        
        Args:
            func_name: Optional custom name for the function measurement
        """
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Pre-execution measurements
                start_time = time.perf_counter()
                start_memory = self.get_memory_usage()
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Post-execution measurements
                    end_time = time.perf_counter()
                    end_memory = self.get_memory_usage()
                    
                    # Calculate metrics
                    execution_time = end_time - start_time
                    memory_delta = {
                        'rss_delta': end_memory['rss'] - start_memory['rss'],
                        'vms_delta': end_memory['vms'] - start_memory['vms'],
                        'peak_rss': end_memory['rss'],
                        'peak_vms': end_memory['vms'],
                    }
                    
                    # Store measurement
                    if name not in self.measurements:
                        self.measurements[name] = []
                    
                    measurement = {
                        'execution_time': execution_time,
                        'memory_start': start_memory,
                        'memory_end': end_memory,
                        'memory_delta': memory_delta,
                        'timestamp': time.time(),
                        'success': True
                    }
                    
                    self.measurements[name].append(measurement)
                    
                    # Log performance
                    self._log_performance(name, measurement)
                    
                    return result
                    
                except Exception as e:
                    # Log failed execution
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    measurement = {
                        'execution_time': execution_time,
                        'memory_start': start_memory,
                        'memory_end': self.get_memory_usage(),
                        'error': str(e),
                        'timestamp': time.time(),
                        'success': False
                    }
                    
                    if name not in self.measurements:
                        self.measurements[name] = []
                    self.measurements[name].append(measurement)
                    
                    raise
                    
            return wrapper
        return decorator
    
    def _log_performance(self, func_name: str, measurement: dict[str, Any]):
        """Log performance measurement."""
        execution_time = measurement['execution_time']
        memory_delta = measurement['memory_delta']
        
        print(f"⏱️ {func_name}: {execution_time:.3f}s "
              f"(Memory: {memory_delta['rss_delta']:+.1f}MB RSS, "
              f"{memory_delta['vms_delta']:+.1f}MB VMS)")
    
    def get_summary(self, func_name: str | None = None) -> dict[str, Any]:
        """
        Get performance summary for a function or all functions.
        
        Args:
            func_name: Optional function name to get summary for. If None, returns all.
        """
        if func_name:
            if func_name not in self.measurements:
                return {}
            measurements = {func_name: self.measurements[func_name]}
        else:
            measurements = self.measurements
        
        summary = {}
        
        for name, records in measurements.items():
            successful_records = [r for r in records if r.get('success', True)]
            if not successful_records:
                continue
                
            times = [r['execution_time'] for r in successful_records]
            memory_deltas = [r['memory_delta']['rss_delta'] for r in successful_records]
            
            summary[name] = {
                'call_count': len(successful_records),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_memory_delta': sum(memory_deltas),
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta': max(memory_deltas),
                'first_call': min(r['timestamp'] for r in successful_records),
                'last_call': max(r['timestamp'] for r in successful_records),
            }
        
        return summary
    
    def save_measurements(self, filepath: str):
        """Save all measurements to a JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert measurements to JSON-serializable format
        json_data = {
            'initial_memory': self.initial_memory,
            'measurements': self.measurements,
            'summary': self.get_summary(),
            'timestamp': time.time(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"📊 Performance measurements saved to: {filepath}")
    
    def print_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_summary()
        
        if not summary:
            print("📊 No performance measurements recorded.")
            return
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        for func_name, stats in summary.items():
            print(f"\n🔧 {func_name}")
            print(f"   Calls: {stats['call_count']}")
            print(f"   Total time: {stats['total_time']:.3f}s")
            print(f"   Avg time: {stats['avg_time']:.3f}s")
            print(f"   Min/Max time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            print(f"   Avg memory delta: {stats['avg_memory_delta']:+.1f}MB")
            print(f"   Max memory delta: {stats['max_memory_delta']:+.1f}MB")
        
        print("\n" + "=" * 50)


# Global performance measurement instance
perf_monitor = PerformanceMeasurement()


def profile_function(func_name: str | None = None):
    """
    Decorator for profiling function performance.
    
    Usage:
        @profile_function("auxiliary_operations_processing")
        def my_function():
            pass
    """
    return perf_monitor.timing_decorator(func_name)


def get_performance_summary(func_name: str | None = None) -> dict[str, Any]:
    """Get performance summary."""
    return perf_monitor.get_summary(func_name)


def save_performance_measurements(filepath: str):
    """Save performance measurements to file."""
    perf_monitor.save_measurements(filepath)


def print_performance_summary():
    """Print performance summary."""
    perf_monitor.print_summary()


class ModelSizeAnalyzer:
    """
    Utility to analyze model characteristics for performance testing.
    """
    
    @staticmethod
    def analyze_onnx_model(onnx_model) -> dict[str, Any]:
        """Analyze ONNX model characteristics."""
        total_nodes = len(onnx_model.graph.node)
        
        # Count operation types
        op_type_counts = {}
        for node in onnx_model.graph.node:
            op_type = node.op_type
            op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1
        
        # Identify auxiliary operations
        auxiliary_ops = ['Shape', 'Constant', 'Cast', 'Reshape', 'Transpose', 
                        'Unsqueeze', 'Squeeze', 'Where', 'Gather', 'ReduceMean',
                        'Slice', 'Concat', 'Add', 'Sub', 'Mul', 'Div']
        
        auxiliary_count = sum(op_type_counts.get(op, 0) for op in auxiliary_ops)
        
        # Calculate complexity metrics
        total_parameters = len(onnx_model.graph.initializer)
        total_inputs = len(onnx_model.graph.input)
        total_outputs = len(onnx_model.graph.output)
        
        return {
            'total_nodes': total_nodes,
            'auxiliary_operations': auxiliary_count,
            'auxiliary_percentage': (auxiliary_count / total_nodes * 100) if total_nodes > 0 else 0,
            'operation_types': len(op_type_counts),
            'op_type_distribution': op_type_counts,
            'parameters': total_parameters,
            'inputs': total_inputs,
            'outputs': total_outputs,
            'complexity_score': total_nodes + total_parameters  # Simple complexity metric
        }
    
    @staticmethod
    def print_model_analysis(analysis: dict[str, Any], model_name: str = "Model"):
        """Print formatted model analysis."""
        print(f"\n📋 {model_name.upper()} ANALYSIS")
        print("=" * 40)
        print(f"Total operations: {analysis['total_nodes']}")
        print(f"Auxiliary operations: {analysis['auxiliary_operations']} ({analysis['auxiliary_percentage']:.1f}%)")
        print(f"Operation types: {analysis['operation_types']}")
        print(f"Parameters: {analysis['parameters']}")
        print(f"Complexity score: {analysis['complexity_score']}")
        
        print("\nTop operation types:")
        sorted_ops = sorted(analysis['op_type_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_ops[:5]:
            print(f"  {op_type}: {count}")
        print("=" * 40)