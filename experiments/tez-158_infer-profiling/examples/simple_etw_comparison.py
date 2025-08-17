"""
Simple ETW vs PSUtil Monitor Comparison Example

This example demonstrates the difference between PSUtil (polling-based) 
and ETW (event-driven) monitoring on Windows systems.

Usage:
    python simple_etw_comparison.py [--duration SECONDS] [--rate HZ]

Author: TEZ-165 Implementation
Date: 2025-08-16
"""

import sys
import os
import time
import platform
import argparse
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our monitors
try:
    from monitors.psutil_monitor import PSUtilMonitor, PSUtilMonitorConfig
    PSUTIL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PSUtil monitor not available: {e}")
    PSUTIL_AVAILABLE = False

try:
    from monitors.etw_monitor import ETWMonitor, ETWMonitorConfig
    ETW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ETW monitor not available: {e}")
    ETW_AVAILABLE = False


def print_separator(title: str = "", width: int = 60):
    """Print a formatted separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def format_bytes(bytes_value: float) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}TB"


def run_psutil_monitoring(duration: int = 5, sampling_rate: float = 10.0):
    """Run PSUtil-based monitoring"""
    print_separator("PSUtil Monitor (Polling-Based)")
    print(f"Platform: {platform.system()}")
    print(f"Sampling Rate: {sampling_rate}Hz")
    print(f"Duration: {duration} seconds")
    print(f"Expected Overhead: ~{sampling_rate * 0.35:.1f}% CPU")
    print()
    
    # Configure PSUtil monitor
    config = PSUtilMonitorConfig(
        sampling_rate_hz=sampling_rate,
        buffer_size=int(sampling_rate * duration * 2),
        max_processes=20,
        include_system_processes=False,
        enable_io_counters=True,
        enable_handle_count=(platform.system() == "Windows"),
        rolling_window_size=10
    )
    
    monitor = PSUtilMonitor(config)
    
    # Track top CPU processes
    top_processes = {}
    sample_count = 0
    
    def process_metric(sample):
        nonlocal sample_count
        if "cpu_percent" in sample.metric_name and "total" not in sample.metric_name:
            pid = sample.metadata.get('pid')
            name = sample.metadata.get('process_name')
            if pid and sample.value > 0:
                if pid not in top_processes:
                    top_processes[pid] = {
                        'name': name,
                        'cpu_samples': [],
                        'memory_samples': []
                    }
                top_processes[pid]['cpu_samples'].append(sample.value)
        elif "memory_rss" in sample.metric_name:
            pid = sample.metadata.get('pid')
            if pid in top_processes:
                top_processes[pid]['memory_samples'].append(sample.value)
        sample_count += 1
    
    monitor.register_callback(process_metric)
    
    # Start monitoring
    print("Starting PSUtil monitoring...")
    start_time = time.time()
    monitor.start_monitoring()
    
    # Run for specified duration
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    # Stop monitoring
    monitor.stop_monitoring()
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Get statistics
    stats = monitor.get_statistics()
    summary = monitor.get_process_summary()
    
    # Print results
    print(f"\n{'-' * 40}")
    print("PSUtil Monitoring Results:")
    print(f"  Actual Duration: {actual_duration:.2f}s")
    print(f"  Samples Collected: {sample_count}")
    print(f"  Effective Rate: {sample_count / actual_duration:.1f} samples/sec")
    print(f"  Processes Monitored: {summary['monitored_processes']}")
    
    if 'collection_time_ms' in stats:
        print(f"  Avg Collection Time: {stats['collection_time_ms']['mean']:.2f}ms")
        print(f"  Max Collection Time: {stats['collection_time_ms']['max']:.2f}ms")
        overhead = (stats['collection_time_ms']['mean'] / 1000) * sampling_rate * 100
        print(f"  Estimated Overhead: {overhead:.1f}%")
    
    # Top processes by CPU
    print(f"\nTop 5 Processes by CPU (PSUtil):")
    sorted_procs = sorted(
        [(pid, data) for pid, data in top_processes.items()],
        key=lambda x: sum(x[1]['cpu_samples']) / max(1, len(x[1]['cpu_samples'])),
        reverse=True
    )[:5]
    
    for pid, data in sorted_procs:
        avg_cpu = sum(data['cpu_samples']) / max(1, len(data['cpu_samples']))
        avg_mem = sum(data['memory_samples']) / max(1, len(data['memory_samples'])) if data['memory_samples'] else 0
        print(f"  {data['name']:<20} (PID: {pid:6}): CPU={avg_cpu:6.2f}%, Memory={avg_mem:8.2f}MB")
    
    # Cleanup
    monitor.cleanup()
    return stats


def run_etw_monitoring(duration: int = 5, sampling_rate: float = 10.0):
    """Run ETW-based monitoring (Windows only)"""
    print_separator("ETW Monitor (Event-Driven)")
    
    if platform.system() != "Windows":
        print("ETW monitoring is only available on Windows!")
        return None
    
    print(f"Platform: Windows (Kernel-Level)")
    print(f"Sampling Rate: {sampling_rate}Hz (up to 1000Hz supported)")
    print(f"Duration: {duration} seconds")
    print(f"Expected Overhead: <2% CPU")
    print()
    
    # Configure ETW monitor
    config = ETWMonitorConfig(
        sampling_rate_hz=sampling_rate,
        buffer_size=int(sampling_rate * duration * 2),
        session_name="OnnxInferenceExample",
        buffer_size_kb=64,
        enable_process_events=True,
        enable_thread_events=True,
        enable_cpu_sampling=True,
        enable_memory_events=True,
        enable_context_switch=True,
        cpu_sampling_interval=int(1000 / sampling_rate),  # Convert Hz to interval
        use_existing_session=True,  # Try to use existing kernel session
        exclude_system_processes=True
    )
    
    monitor = ETWMonitor(config)
    
    # Track kernel-level metrics
    kernel_metrics = {}
    event_count = 0
    
    def process_etw_event(sample):
        nonlocal event_count
        pid = sample.metadata.get('pid')
        name = sample.metadata.get('process_name')
        
        if pid and name:
            if pid not in kernel_metrics:
                kernel_metrics[pid] = {
                    'name': name,
                    'cpu_cycles': [],
                    'context_switches': [],
                    'kernel_time': [],
                    'user_time': [],
                    'working_set': [],
                    'page_faults': []
                }
            
            if "cpu_cycles" in sample.metric_name:
                kernel_metrics[pid]['cpu_cycles'].append(sample.value)
            elif "context_switches" in sample.metric_name:
                kernel_metrics[pid]['context_switches'].append(sample.value)
            elif "kernel_time" in sample.metric_name:
                kernel_metrics[pid]['kernel_time'].append(sample.value)
            elif "user_time" in sample.metric_name:
                kernel_metrics[pid]['user_time'].append(sample.value)
            elif "working_set" in sample.metric_name:
                kernel_metrics[pid]['working_set'].append(sample.value)
            elif "page_faults" in sample.metric_name:
                kernel_metrics[pid]['page_faults'].append(sample.value)
        
        event_count += 1
    
    monitor.register_callback(process_etw_event)
    
    # Start monitoring
    print("Starting ETW monitoring (kernel-level)...")
    start_time = time.time()
    monitor.start_monitoring()
    
    # Run for specified duration
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    # Stop monitoring
    monitor.stop_monitoring()
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Get statistics
    stats = monitor.get_statistics()
    summary = monitor.get_process_summary()
    
    # Print results
    print(f"\n{'-' * 40}")
    print("ETW Monitoring Results:")
    print(f"  Actual Duration: {actual_duration:.2f}s")
    print(f"  Events Processed: {event_count}")
    print(f"  Event Rate: {event_count / actual_duration:.1f} events/sec")
    print(f"  Processes Monitored: {summary['monitored_processes']}")
    print(f"  Session Active: {summary['session_active']}")
    print(f"  Overhead Estimate: {summary['overhead_estimate']}")
    
    # Kernel-level metrics
    print(f"\nTop 5 Processes by CPU Cycles (ETW Kernel Metrics):")
    sorted_procs = sorted(
        [(pid, data) for pid, data in kernel_metrics.items()],
        key=lambda x: sum(x[1]['cpu_cycles']) if x[1]['cpu_cycles'] else 0,
        reverse=True
    )[:5]
    
    for pid, data in sorted_procs:
        total_cycles = sum(data['cpu_cycles']) if data['cpu_cycles'] else 0
        total_switches = sum(data['context_switches']) if data['context_switches'] else 0
        avg_kernel = sum(data['kernel_time']) / max(1, len(data['kernel_time'])) if data['kernel_time'] else 0
        avg_user = sum(data['user_time']) / max(1, len(data['user_time'])) if data['user_time'] else 0
        
        print(f"  {data['name']:<20} (PID: {pid:6}):")
        print(f"    CPU Cycles: {total_cycles:,}")
        print(f"    Context Switches: {total_switches}")
        print(f"    Kernel Time: {avg_kernel:.1f}ms, User Time: {avg_user:.1f}ms")
    
    # Cleanup
    monitor.cleanup()
    return stats


def compare_monitors(duration: int = 5, sampling_rate: float = 10.0):
    """Compare PSUtil and ETW monitors side by side"""
    print_separator("Monitor Comparison Test", 60)
    print(f"Configuration:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling Rate: {sampling_rate}Hz")
    print()
    
    results = {}
    
    # Run PSUtil monitoring
    if PSUTIL_AVAILABLE:
        psutil_stats = run_psutil_monitoring(duration, sampling_rate)
        results['psutil'] = psutil_stats
        print()
        time.sleep(2)  # Brief pause between tests
    
    # Run ETW monitoring (Windows only)
    if ETW_AVAILABLE and platform.system() == "Windows":
        etw_stats = run_etw_monitoring(duration, sampling_rate)
        results['etw'] = etw_stats
        print()
    
    # Print comparison summary
    print_separator("Comparison Summary", 60)
    
    if 'psutil' in results and results['psutil']:
        psutil_overhead = results['psutil'].get('collection_time_ms', {}).get('mean', 0) / 10
        print(f"PSUtil Monitor:")
        print(f"  Technology: Polling-based (cross-platform)")
        print(f"  Overhead: ~{psutil_overhead:.1f}%")
        print(f"  Pros: Simple, cross-platform, well-documented")
        print(f"  Cons: Higher overhead, sampling gaps possible")
    
    if 'etw' in results and results['etw']:
        print(f"\nETW Monitor:")
        print(f"  Technology: Event-driven (Windows kernel)")
        print(f"  Overhead: <2%")
        print(f"  Pros: Low overhead, exact timing, kernel-level")
        print(f"  Cons: Windows-only, complex, requires privileges")
    
    print("\nRecommendation:")
    if platform.system() == "Windows":
        if sampling_rate > 50:
            print("  Use ETW for high-frequency monitoring (>50Hz)")
            print("  PSUtil overhead becomes significant at high rates")
        else:
            print("  PSUtil is sufficient for low-frequency monitoring (<50Hz)")
            print("  Simpler to implement and maintain")
    else:
        print("  Use PSUtil (ETW not available on this platform)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare PSUtil and ETW monitoring on Windows"
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=5,
        help='Monitoring duration in seconds (default: 5)'
    )
    parser.add_argument(
        '--rate', '-r',
        type=float,
        default=10.0,
        help='Sampling rate in Hz (default: 10)'
    )
    parser.add_argument(
        '--psutil-only',
        action='store_true',
        help='Run only PSUtil monitoring'
    )
    parser.add_argument(
        '--etw-only',
        action='store_true',
        help='Run only ETW monitoring (Windows only)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("Error: Duration must be positive")
        return 1
    
    if args.rate <= 0 or args.rate > 1000:
        print("Error: Sampling rate must be between 0 and 1000 Hz")
        return 1
    
    try:
        if args.psutil_only:
            if not PSUTIL_AVAILABLE:
                print("Error: PSUtil monitor not available")
                return 1
            run_psutil_monitoring(args.duration, args.rate)
        elif args.etw_only:
            if not ETW_AVAILABLE:
                print("Error: ETW monitor not available")
                return 1
            if platform.system() != "Windows":
                print("Error: ETW monitor only works on Windows")
                return 1
            run_etw_monitoring(args.duration, args.rate)
        else:
            compare_monitors(args.duration, args.rate)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        return 0
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())