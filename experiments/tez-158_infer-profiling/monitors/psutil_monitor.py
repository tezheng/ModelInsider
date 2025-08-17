"""
PSUtil-based Process Monitor Implementation

Provides cross-platform process monitoring using the psutil library.
Focuses on per-process CPU and memory metrics with configurable sampling rates.

Author: TEZ-165 Implementation  
Date: 2025-08-16
"""

import os
import sys
import time
import psutil
import platform
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import threading

try:
    from .base_monitor import BaseMonitor, MetricSample, MonitorConfig
except ImportError:
    # For direct execution
    from base_monitor import BaseMonitor, MetricSample, MonitorConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProcessMetrics:
    """Metrics for a single process"""
    pid: int
    name: str
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    num_threads: int
    create_time: float
    io_read_bytes: Optional[int] = None
    io_write_bytes: Optional[int] = None
    num_handles: Optional[int] = None  # Windows only
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "cpu_percent": self.cpu_percent,
            "memory_rss_mb": self.memory_rss_mb,
            "memory_vms_mb": self.memory_vms_mb,
            "memory_percent": self.memory_percent,
            "num_threads": self.num_threads,
            "create_time": self.create_time,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "num_handles": self.num_handles
        }


@dataclass
class PSUtilMonitorConfig(MonitorConfig):
    """Configuration specific to PSUtil monitoring"""
    target_processes: Optional[Set[int]] = None  # PIDs to monitor (None = all)
    target_process_names: Optional[Set[str]] = None  # Process names to monitor
    include_system_processes: bool = False  # Include system processes
    cpu_sample_interval: float = 0.1  # CPU sampling interval for psutil
    max_processes: int = 100  # Maximum number of processes to track
    enable_io_counters: bool = True  # Track I/O counters if available
    enable_handle_count: bool = True  # Track handle count on Windows
    rolling_window_size: int = 10  # Size of rolling window for CPU averaging


class PSUtilMonitor(BaseMonitor):
    """
    Cross-platform process monitor using psutil library
    
    Provides per-process CPU and memory monitoring with configurable sampling rates.
    Works on Windows, Linux, and macOS with consistent API.
    """
    
    def __init__(self, config: Optional[PSUtilMonitorConfig] = None):
        """
        Initialize PSUtil monitor
        
        Args:
            config: PSUtil monitor configuration
        """
        self.config = config or PSUtilMonitorConfig()
        
        # Process tracking
        self._process_cache: Dict[int, psutil.Process] = {}
        self._cpu_percentages: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.rolling_window_size)
        )
        self._last_cpu_times: Dict[int, float] = {}
        
        # Platform detection
        self.platform = platform.system()
        self.is_windows = self.platform == "Windows"
        self.is_linux = self.platform == "Linux"
        self.is_macos = self.platform == "Darwin"
        
        # Initialize base class
        super().__init__(self.config)
        
        logger.info(f"PSUtilMonitor initialized on {self.platform}")
    
    def _initialize(self) -> None:
        """Initialize PSUtil monitoring resources"""
        logger.info(f"Initializing PSUtilMonitor (Platform: {self.platform})")
        
        # Initialize process cache
        self._refresh_process_cache()
        
        # Initialize CPU percent calculation for all processes
        for pid, process in self._process_cache.items():
            try:
                process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def _refresh_process_cache(self) -> None:
        """Refresh the cache of monitored processes"""
        try:
            # Get all processes or filtered set
            if self.config.target_processes:
                # Monitor specific PIDs
                for pid in self.config.target_processes:
                    try:
                        if pid not in self._process_cache:
                            self._process_cache[pid] = psutil.Process(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            else:
                # Monitor all processes (up to max_processes)
                count = 0
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pid = proc.info['pid']
                        name = proc.info['name']
                        
                        # Filter by name if specified
                        if self.config.target_process_names:
                            if name not in self.config.target_process_names:
                                continue
                        
                        # Skip system processes if requested
                        if not self.config.include_system_processes:
                            # Platform-specific system process filtering
                            if self.is_windows:
                                if pid <= 4 or name in ['System', 'Registry', 'Idle']:
                                    continue
                            elif self.is_linux:
                                if pid <= 2 or name in ['kernel', 'kthreadd']:
                                    continue
                            elif self.is_macos:
                                if pid <= 1 or name in ['kernel_task', 'launchd']:
                                    continue
                        
                        if pid not in self._process_cache:
                            self._process_cache[pid] = proc
                            count += 1
                            
                            # Limit number of processes
                            if count >= self.config.max_processes:
                                break
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            # Clean up dead processes
            dead_pids = []
            for pid, proc in self._process_cache.items():
                if not proc.is_running():
                    dead_pids.append(pid)
            
            for pid in dead_pids:
                del self._process_cache[pid]
                if pid in self._cpu_percentages:
                    del self._cpu_percentages[pid]
                if pid in self._last_cpu_times:
                    del self._last_cpu_times[pid]
        
        except Exception as e:
            logger.error(f"Error refreshing process cache: {e}")
    
    def _collect_process_metrics(self, process: psutil.Process) -> Optional[ProcessMetrics]:
        """Collect metrics for a single process"""
        try:
            pid = process.pid
            
            # Get process info with timeout using oneshot context
            with process.oneshot():
                # Basic info
                name = process.name()
                create_time = process.create_time()
                num_threads = process.num_threads()
                
                # Memory metrics
                mem_info = process.memory_info()
                memory_rss_mb = mem_info.rss / (1024 * 1024)
                memory_vms_mb = mem_info.vms / (1024 * 1024)
                memory_percent = process.memory_percent()
                
                # CPU metrics - use interval-based measurement
                cpu_percent = process.cpu_percent(interval=None)
                
                # Store CPU percentage in rolling buffer for smoothing
                self._cpu_percentages[pid].append(cpu_percent)
                
                # Average CPU over recent samples for stability
                if len(self._cpu_percentages[pid]) > 0:
                    avg_cpu = sum(self._cpu_percentages[pid]) / len(self._cpu_percentages[pid])
                else:
                    avg_cpu = cpu_percent
                
                # I/O counters (if available and enabled)
                io_read_bytes = None
                io_write_bytes = None
                if self.config.enable_io_counters:
                    try:
                        io_counters = process.io_counters()
                        io_read_bytes = io_counters.read_bytes
                        io_write_bytes = io_counters.write_bytes
                    except (psutil.AccessDenied, AttributeError):
                        pass
                
                # Handle count (Windows only)
                num_handles = None
                if self.is_windows and self.config.enable_handle_count:
                    try:
                        num_handles = process.num_handles()
                    except (psutil.AccessDenied, AttributeError):
                        pass
            
            return ProcessMetrics(
                pid=pid,
                name=name,
                cpu_percent=avg_cpu,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                memory_percent=memory_percent,
                num_threads=num_threads,
                create_time=create_time,
                io_read_bytes=io_read_bytes,
                io_write_bytes=io_write_bytes,
                num_handles=num_handles
            )
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            return None
        except Exception as e:
            logger.debug(f"Error collecting metrics for PID {process.pid}: {e}")
            return None
    
    def collect_metrics(self) -> List[MetricSample]:
        """
        Collect per-process CPU and memory metrics
        
        Returns:
            List of metric samples for all monitored processes
        """
        samples = []
        timestamp = time.time()
        
        # Refresh process cache periodically (every 10 samples)
        if len(self.samples) % 10 == 0:
            self._refresh_process_cache()
        
        # Collect metrics for each process
        for pid, process in list(self._process_cache.items()):
            metrics = self._collect_process_metrics(process)
            
            if metrics:
                # Create samples for each metric type
                
                # CPU usage sample
                samples.append(MetricSample(
                    timestamp=timestamp,
                    metric_name=f"process_cpu_percent",
                    value=metrics.cpu_percent,
                    unit="percent",
                    metadata={
                        "pid": metrics.pid,
                        "process_name": metrics.name
                    }
                ))
                
                # Memory RSS sample
                samples.append(MetricSample(
                    timestamp=timestamp,
                    metric_name=f"process_memory_rss",
                    value=metrics.memory_rss_mb,
                    unit="MB",
                    metadata={
                        "pid": metrics.pid,
                        "process_name": metrics.name
                    }
                ))
                
                # Memory VMS sample
                samples.append(MetricSample(
                    timestamp=timestamp,
                    metric_name=f"process_memory_vms",
                    value=metrics.memory_vms_mb,
                    unit="MB",
                    metadata={
                        "pid": metrics.pid,
                        "process_name": metrics.name
                    }
                ))
                
                # Memory percentage sample
                samples.append(MetricSample(
                    timestamp=timestamp,
                    metric_name=f"process_memory_percent",
                    value=metrics.memory_percent,
                    unit="percent",
                    metadata={
                        "pid": metrics.pid,
                        "process_name": metrics.name
                    }
                ))
                
                # Thread count sample
                samples.append(MetricSample(
                    timestamp=timestamp,
                    metric_name=f"process_threads",
                    value=metrics.num_threads,
                    unit="count",
                    metadata={
                        "pid": metrics.pid,
                        "process_name": metrics.name
                    }
                ))
                
                # I/O samples (if available)
                if metrics.io_read_bytes is not None:
                    samples.append(MetricSample(
                        timestamp=timestamp,
                        metric_name=f"process_io_read_bytes",
                        value=metrics.io_read_bytes,
                        unit="bytes",
                        metadata={
                            "pid": metrics.pid,
                            "process_name": metrics.name
                        }
                    ))
                
                if metrics.io_write_bytes is not None:
                    samples.append(MetricSample(
                        timestamp=timestamp,
                        metric_name=f"process_io_write_bytes",
                        value=metrics.io_write_bytes,
                        unit="bytes",
                        metadata={
                            "pid": metrics.pid,
                            "process_name": metrics.name
                        }
                    ))
                
                # Handle count (Windows only)
                if metrics.num_handles is not None:
                    samples.append(MetricSample(
                        timestamp=timestamp,
                        metric_name=f"process_handle_count",
                        value=metrics.num_handles,
                        unit="count",
                        metadata={
                            "pid": metrics.pid,
                            "process_name": metrics.name
                        }
                    ))
        
        # Add system-wide summary metrics
        if self._process_cache:
            total_cpu = sum(
                self._cpu_percentages[pid][-1] if self._cpu_percentages[pid] else 0
                for pid in self._process_cache
            )
            
            samples.append(MetricSample(
                timestamp=timestamp,
                metric_name="total_monitored_cpu_percent",
                value=total_cpu,
                unit="percent",
                metadata={"process_count": len(self._process_cache)}
            ))
        
        return samples
    
    def get_process_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored processes"""
        summary = {
            "platform": self.platform,
            "backend": "psutil",
            "monitored_processes": len(self._process_cache),
            "total_samples": len(self.samples),
            "processes": []
        }
        
        for pid, process in self._process_cache.items():
            try:
                metrics = self._collect_process_metrics(process)
                if metrics:
                    summary["processes"].append(metrics.to_dict())
            except:
                continue
        
        # Sort by CPU usage
        summary["processes"].sort(key=lambda x: x["cpu_percent"], reverse=True)
        
        return summary
    
    def add_process(self, pid: int) -> bool:
        """Add a process to monitoring"""
        try:
            if pid not in self._process_cache:
                self._process_cache[pid] = psutil.Process(pid)
                logger.info(f"Added process {pid} to monitoring")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning(f"Failed to add process {pid}")
        return False
    
    def remove_process(self, pid: int) -> bool:
        """Remove a process from monitoring"""
        if pid in self._process_cache:
            del self._process_cache[pid]
            if pid in self._cpu_percentages:
                del self._cpu_percentages[pid]
            if pid in self._last_cpu_times:
                del self._last_cpu_times[pid]
            logger.info(f"Removed process {pid} from monitoring")
            return True
        return False
    
    def set_target_processes(self, pids: Optional[Set[int]] = None,
                           names: Optional[Set[str]] = None) -> None:
        """Set target processes to monitor"""
        self.config.target_processes = pids
        self.config.target_process_names = names
        self._refresh_process_cache()
    
    def cleanup(self) -> None:
        """Cleanup PSUtil monitoring resources"""
        logger.info("Cleaning up PSUtil monitor")
        
        # Clear caches
        self._process_cache.clear()
        self._cpu_percentages.clear()
        self._last_cpu_times.clear()


def test_psutil_monitor():
    """Test PSUtil monitor functionality"""
    import json
    
    print("=" * 60)
    print("PSUtil Monitor Test")
    print("=" * 60)
    
    # Create monitor with 10Hz sampling
    config = PSUtilMonitorConfig(
        sampling_rate_hz=10.0,
        buffer_size=1000,
        max_processes=10,
        include_system_processes=False,
        enable_io_counters=True,
        enable_handle_count=True
    )
    
    monitor = PSUtilMonitor(config)
    
    # Register a callback to print metrics
    def print_metric(sample: MetricSample):
        if "cpu" in sample.metric_name and "total" not in sample.metric_name:
            print(f"[{sample.timestamp:.2f}] {sample.metadata.get('process_name', 'Unknown')} "
                  f"(PID: {sample.metadata.get('pid')}): "
                  f"CPU={sample.value:.1f}%")
    
    monitor.register_callback(print_metric)
    
    # Monitor for 5 seconds
    print(f"\nStarting monitoring on {platform.system()} for 5 seconds...")
    monitor.start_monitoring()
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    
    monitor.stop_monitoring()
    
    # Print statistics
    stats = monitor.get_statistics()
    print("\nMonitoring Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Print process summary
    summary = monitor.get_process_summary()
    print(f"\nPlatform: {summary['platform']}")
    print(f"Backend: {summary['backend']}")
    print(f"Monitored {summary['monitored_processes']} processes")
    print("\nTop 5 processes by CPU:")
    for proc in summary["processes"][:5]:
        print(f"  {proc['name']} (PID: {proc['pid']}): "
              f"CPU={proc['cpu_percent']:.1f}%, "
              f"Memory={proc['memory_rss_mb']:.1f}MB")
    
    # Cleanup
    monitor.cleanup()
    print("\nTest completed!")


if __name__ == "__main__":
    test_psutil_monitor()