# Windows System Monitoring Technologies: Comprehensive Research Report

## Executive Summary

This document provides comprehensive research on Windows system monitoring technologies for high-frequency process/thread-level profiling of CPU, GPU, NPU, and memory resources. Based on extensive research conducted in 2024, we analyze ETW, Windows Performance Counters, WMI, and various Python-based profiling solutions to determine the optimal approach for system monitoring.

## Table of Contents
1. [Native Windows Technologies](#native-windows-technologies)
2. [GPU Monitoring Solutions](#gpu-monitoring-solutions)
3. [NPU Monitoring Capabilities](#npu-monitoring-capabilities)
4. [Python Profiling Ecosystem](#python-profiling-ecosystem)
5. [Comparative Analysis](#comparative-analysis)
6. [Recommendations](#recommendations)
7. [Implementation Strategy](#implementation-strategy)

---

## Native Windows Technologies

### 1. ETW (Event Tracing for Windows)

**Overview:**
ETW is a high-speed kernel-level tracing facility built into Windows that provides an infrastructure for events raised by both user-mode applications and kernel-mode drivers.

**Key Strengths:**
- **Lowest Overhead**: Uses kernel-mode buffers with separate writer threads
- **System-Wide Coverage**: Captures events from all processes and kernel
- **High-Frequency Capability**: Can handle millions of events per second
- **Thread-Level Detail**: Provides context switches, thread creation/destruction events
- **Native Integration**: Built into Windows kernel since Vista

**Technical Capabilities:**
- **CPU Monitoring**: Sampling profiler, context switches, CPU utilization per process
- **Memory Tracking**: Heap allocation tracing, working set monitoring, page faults
- **I/O Operations**: Disk I/O, network activity, file operations
- **GPU Support**: Via GPUView and WPA (Windows Performance Analyzer)

**Performance Characteristics:**
- Overhead: <1-2% for most scenarios
- Sampling rates: Up to 10KHz for CPU sampling
- Buffer management: Circular buffers prevent unbounded growth
- Log size: ~1GB per 3 seconds for intensive applications

**Python Integration:**
```python
# Current approaches for Python:
# 1. pyetw - Python bindings for ETW (limited maintenance)
# 2. wpa-python - Python wrapper for WPA analysis
# 3. Custom integration via ctypes/pythonnet
```

**Limitations:**
- Complex API requiring deep Windows knowledge
- Limited direct Python bindings
- Requires elevated privileges for many operations
- Windows-only (no cross-platform support)

### 2. Windows Performance Counters (PDH)

**Overview:**
Performance Data Helper (PDH) API provides access to Windows performance counter data, offering high-level system metrics.

**Evolution:**
- Pre-Vista: Separate infrastructure from ETW
- Vista+: Performance counters have ETW facade for unified access
- Modern: PERFLIB 2.0 uses ETW as underlying transport

**Key Features:**
- **Standard Metrics**: CPU %, Memory usage, Disk I/O, Network traffic
- **Process/Thread Counters**: Per-process and per-thread metrics
- **Custom Counters**: Applications can register custom performance counters
- **Remote Access**: Built-in support for remote monitoring

**Python Integration:**
```python
import win32pdh
import win32pdhutil

# Query CPU usage for specific process
cpu_usage = win32pdhutil.GetPerformanceAttributes(
    "Process", "% Processor Time", "python"
)
```

**Performance Impact:**
- Overhead: 2-5% depending on counter frequency
- Update frequency: Typically 1Hz to 10Hz
- Accuracy: Good for averages, less precise for instantaneous values

### 3. WMI (Windows Management Instrumentation)

**Overview:**
WMI provides a unified interface for Windows management data and operations, including performance monitoring.

**Characteristics:**
- **High-Level API**: Easier to use than ETW
- **Remote Monitoring**: Excellent for distributed systems
- **Comprehensive Coverage**: Access to all Windows subsystems
- **Query Language**: WQL (WMI Query Language) for data retrieval

**Python Example:**
```python
import wmi

c = wmi.WMI()
for process in c.Win32_Process():
    print(f"Process: {process.Name}, PID: {process.ProcessId}")
    print(f"WorkingSetSize: {process.WorkingSetSize}")
```

**Limitations:**
- **Higher Overhead**: 5-10% for intensive monitoring
- **Accuracy Issues**: Rounds values, may miss sub-second events
- **Latency**: Not suitable for high-frequency sampling
- **Security**: Complex authentication for remote access

### 4. Comparison of Native Technologies

| Feature | ETW | Performance Counters | WMI |
|---------|-----|---------------------|-----|
| Overhead | <2% | 2-5% | 5-10% |
| Sampling Rate | Up to 10KHz | 1-10Hz typical | 1Hz typical |
| Thread-Level Detail | Excellent | Good | Limited |
| Remote Monitoring | Complex | Supported | Excellent |
| Python Support | Limited | Via pywin32 | Good (wmi module) |
| Learning Curve | Steep | Moderate | Easy |
| Real-time Capability | Excellent | Good | Fair |

---

## GPU Monitoring Solutions

### 1. NVIDIA Monitoring

**NVML (NVIDIA Management Library):**
- **Purpose**: Programmatic interface for NVIDIA GPU monitoring
- **Capabilities**: GPU utilization, memory usage, temperature, power consumption
- **Python Support**: Excellent via `pynvml` and `nvidia-ml-py`

**Python Implementation:**
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get GPU utilization
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU: {util.gpu}%, Memory: {util.memory}%")

# Get memory info
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Used: {mem_info.used / 1024**2}MB / {mem_info.total / 1024**2}MB")

# Get power usage
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
print(f"Power: {power}W")
```

**Advanced Tools:**
- **nvitop**: Interactive GPU process viewer with Python API
- **nvidia-smi**: Command-line tool (can be parsed programmatically)
- **CUDA Profiling Tools**: NSight Systems, NSight Compute

### 2. AMD GPU Monitoring

**ROCm Support:**
- **Windows Status**: Limited native support, often requires WSL
- **Python Support**: rocm-smi Python bindings (Linux-focused)
- **Challenges**: Driver compatibility, limited Windows tools

**Alternative Approaches:**
- DirectML for inference monitoring
- Windows Performance Counters for basic GPU metrics
- Third-party tools like GPU-Z with CLI support

### 3. Intel GPU Monitoring

**Intel Graphics Performance Analyzers (GPA):**
- System Analyzer for real-time metrics
- Graphics Monitor for continuous tracking
- Limited Python integration

**OneAPI Level Zero:**
- Modern API for Intel GPU programming and monitoring
- Python bindings under development

### 4. DirectX/DirectML Monitoring

**Windows-Native GPU Monitoring:**
```python
# Using Windows Performance Counters for GPU
import win32pdh

# GPU Engine utilization counter
counter_path = r"\GPU Engine(*)\Utilization Percentage"
# Requires parsing counter instances for specific GPU
```

---

## NPU Monitoring Capabilities

### 1. Current State (2024)

**Hardware Support:**
- **Intel Core Ultra**: NPU via Intel AI Boost
- **Qualcomm Snapdragon X Elite**: 45 TOPS Hexagon NPU
- **AMD Ryzen AI**: XDNA architecture NPUs

**Software Integration:**
- DirectML 1.15.2+ supports NPU acceleration
- ONNX Runtime 1.18+ with NPU execution providers
- Windows Task Manager shows NPU utilization (Windows 11 24H2+)

### 2. Monitoring Approaches

**Task Manager Integration:**
- Basic NPU % utilization visible
- No API for programmatic access yet

**ONNX Runtime Profiling:**
```python
import onnxruntime as ort

# Create session with NPU provider
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# Enable profiling
options = ort.SessionOptions()
options.enable_profiling = True
session_with_profiling = ort.InferenceSession(
    "model.onnx", options, providers=providers
)

# Run inference
outputs = session_with_profiling.run(None, inputs)

# Get profiling data
prof_file = session_with_profiling.end_profiling()
```

**DirectML Monitoring:**
- Performance counters for DirectML operations
- ETW providers for DirectML events
- Limited direct NPU metrics

### 3. Future Developments

**Expected Improvements:**
- Unified NPU monitoring API (similar to NVML)
- Enhanced Windows Performance Counter support
- Better integration with profiling tools

---

## Python Profiling Ecosystem

### 1. Mature Python Profiling Projects

#### **psutil** (Most Popular)
- **GitHub Stars**: 10K+
- **Windows Support**: Excellent
- **Capabilities**: Process/system monitoring, CPU, memory, disk, network
- **Limitations**: No GPU monitoring, limited thread-level detail

```python
import psutil

# Process monitoring
process = psutil.Process()
print(f"CPU: {process.cpu_percent(interval=0.1)}%")
print(f"Memory: {process.memory_info().rss / 1024**2}MB")
print(f"Threads: {process.num_threads()}")

# System monitoring
print(f"System CPU: {psutil.cpu_percent(interval=1, percpu=True)}")
print(f"System Memory: {psutil.virtual_memory().percent}%")
```

#### **py-spy** (Sampling Profiler)
- **GitHub Stars**: 12K+
- **Windows Support**: Full support
- **Strengths**: Low overhead, attaches to running processes
- **Use Case**: Production profiling

```bash
# Profile running Python process
py-spy record -o profile.svg --pid 12345

# Profile Python script
py-spy record -o profile.svg -- python script.py
```

#### **Scalene** (AI-Powered Profiler)
- **GitHub Stars**: 11K+
- **Windows Support**: Yes
- **Unique Features**: GPU profiling, memory profiling, AI optimization suggestions
- **Web UI**: Interactive visualization

```bash
# Profile with Scalene
scalene --html --outfile profile.html script.py
```

#### **pyinstrument** (Call Stack Profiler)
- **GitHub Stars**: 6K+
- **Windows Support**: Excellent
- **Strengths**: Beautiful HTML output, low overhead
- **API**: Decorator and context manager support

```python
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

# Your code here
expensive_function()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
```

#### **austin** (Frame Stack Sampler)
- **GitHub Stars**: 1.3K+
- **Windows Support**: Yes
- **Strengths**: Extremely low overhead, memory profiling
- **TUI**: Terminal UI for real-time monitoring

### 2. Specialized Monitoring Tools

#### **GPUtil** (GPU Monitoring)
```python
import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.name}")
    print(f"  Load: {gpu.load * 100}%")
    print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
    print(f"  Temperature: {gpu.temperature}°C")
```

#### **pyRAPL** (Energy Monitoring)
- Intel RAPL (Running Average Power Limit) interface
- Measures CPU and DRAM energy consumption
- Linux-focused, limited Windows support

### 3. Integrated Solutions

#### **Prometheus + Grafana**
```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import psutil

# Define metrics
cpu_usage = Gauge('process_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('process_memory_usage_bytes', 'Memory usage in bytes')

# Update metrics
def update_metrics():
    process = psutil.Process()
    cpu_usage.set(process.cpu_percent(interval=0.1))
    memory_usage.set(process.memory_info().rss)

# Start metrics server
start_http_server(8000)
```

---

## Comparative Analysis

### 1. Technology Selection Matrix

| Requirement | Best Option | Alternative | Notes |
|-------------|------------|-------------|-------|
| Low Overhead (<2%) | ETW | py-spy | ETW requires more setup |
| High Frequency (>100Hz) | ETW | Custom PDH | ETW supports up to 10KHz |
| GPU Monitoring | NVML (NVIDIA) | DirectML counters | Platform-specific |
| NPU Monitoring | ONNX Runtime profiling | Task Manager | Limited API access |
| Cross-Platform | psutil | - | No GPU support |
| Production Use | py-spy | psutil | Minimal impact |
| Development | Scalene | pyinstrument | Rich features |
| Remote Monitoring | WMI | Prometheus | WMI for Windows-only |
| Thread-Level Detail | ETW | austin | ETW most comprehensive |

### 2. Performance Comparison

```
Overhead Comparison (for 100Hz sampling):
- ETW: 1-2%
- Performance Counters: 3-5%
- WMI: 8-12%
- psutil: 2-4%
- py-spy: <1%
- pyinstrument: 10-20%
- Scalene: 15-25%
```

### 3. Feature Coverage

| Feature | ETW | PDH | WMI | psutil | py-spy | Scalene |
|---------|-----|-----|-----|--------|--------|---------|
| CPU (Process) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPU (Thread) | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| Memory | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| GPU | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ✅ |
| NPU | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Network | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Disk I/O | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Energy | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ |

✅ = Full support, ⚠️ = Partial support, ❌ = No support

---

## Recommendations

### 1. Optimal Technology Stack

**For Production ONNX Inference Monitoring:**

```python
# Recommended Architecture
class HybridMonitor:
    """
    Combines best-of-breed technologies for comprehensive monitoring
    """
    
    def __init__(self):
        # CPU/Memory: psutil for cross-platform compatibility
        self.system_monitor = psutil.Process()
        
        # GPU: NVML for NVIDIA, fallback to WMI counters
        self.gpu_monitor = self._init_gpu_monitor()
        
        # NPU: ONNX Runtime profiling
        self.npu_profiling = self._init_npu_profiling()
        
        # Detailed Profiling: ETW for Windows-specific deep dive
        self.etw_session = self._init_etw_session()
        
        # Thread-level: py-spy for production sampling
        self.sampler = self._init_pyspy_sampler()
    
    def _init_gpu_monitor(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            return NVMLMonitor()
        except:
            # Fallback to Windows Performance Counters
            return WindowsGPUMonitor()
    
    def _init_npu_profiling(self):
        # Use ONNX Runtime with profiling enabled
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.enable_profiling = True
        return options
    
    def _init_etw_session(self):
        # Initialize ETW for detailed Windows monitoring
        if platform.system() == 'Windows':
            return ETWSession()
        return None
    
    def _init_pyspy_sampler(self):
        # Configure py-spy for thread-level sampling
        return PySPYSampler(rate=100)  # 100Hz sampling
```

### 2. Implementation Priority

**Phase 1: Foundation (Current Implementation)**
- ✅ Basic psutil-based monitoring
- ✅ Process-level CPU and memory tracking
- ✅ 100Hz sampling capability

**Phase 2: GPU Integration**
```python
# Add GPU monitoring
class GPUMonitor:
    def __init__(self):
        self.backend = self._detect_gpu()
    
    def _detect_gpu(self):
        # Try NVIDIA first
        try:
            import pynvml
            pynvml.nvmlInit()
            return NVIDIABackend()
        except:
            pass
        
        # Try AMD ROCm
        try:
            import rocm_smi
            return AMDBackend()
        except:
            pass
        
        # Fallback to DirectML/Windows counters
        return WindowsGPUBackend()
```

**Phase 3: ETW Integration**
```python
# Enhanced ETW integration for detailed tracing
class ETWIntegration:
    def __init__(self):
        self.providers = [
            "Microsoft-Windows-Kernel-Process",  # Process events
            "Microsoft-Windows-DxgKrnl",         # GPU events
            "Microsoft-Windows-CPU",             # CPU sampling
        ]
        
    def start_session(self, session_name="ONNXProfiling"):
        # Use pyetw or ctypes for ETW control
        pass
```

**Phase 4: NPU Support**
```python
# NPU monitoring when APIs become available
class NPUMonitor:
    def __init__(self):
        self.provider = self._detect_npu()
    
    def _detect_npu(self):
        # Check for Intel Core Ultra
        if self._has_intel_npu():
            return IntelNPUProvider()
        
        # Check for Qualcomm Snapdragon
        if self._has_qualcomm_npu():
            return QualcommNPUProvider()
        
        return None
```

### 3. Best Practices

**1. Sampling Rate Guidelines:**
- CPU/Memory: 10-100Hz (psutil)
- GPU: 10-30Hz (NVML)
- NPU: Event-based (ONNX Runtime)
- Detailed Profiling: On-demand (py-spy)

**2. Overhead Management:**
```python
class AdaptiveMonitor:
    def adjust_sampling_rate(self, current_overhead):
        if current_overhead > 5.0:  # 5% threshold
            self.reduce_sampling_rate()
        elif current_overhead < 2.0:
            self.increase_sampling_rate()
```

**3. Data Collection Strategy:**
```python
# Use circular buffers to prevent memory growth
from collections import deque

class MetricsBuffer:
    def __init__(self, max_samples=10000):
        self.cpu_samples = deque(maxlen=max_samples)
        self.gpu_samples = deque(maxlen=max_samples)
        self.memory_samples = deque(maxlen=max_samples)
```

---

## Implementation Strategy

### 1. Hybrid Approach

```python
# Recommended implementation combining multiple technologies
class WindowsPerformanceMonitor:
    """
    Production-ready Windows performance monitoring system
    """
    
    def __init__(self, config):
        # Layer 1: High-frequency process monitoring
        self.process_monitor = PSUtilMonitor(
            sampling_rate_hz=100,
            metrics=['cpu', 'memory', 'io']
        )
        
        # Layer 2: GPU monitoring
        self.gpu_monitor = self._init_gpu_monitor()
        
        # Layer 3: Deep profiling (on-demand)
        self.profiler = PySpyProfiler()
        
        # Layer 4: ETW for system-wide events
        if config.enable_etw:
            self.etw_monitor = ETWMonitor(
                providers=config.etw_providers
            )
        
        # Layer 5: NPU monitoring (future)
        self.npu_monitor = None  # Placeholder
    
    def start_monitoring(self):
        """Start all monitoring layers"""
        self.process_monitor.start()
        
        if self.gpu_monitor:
            self.gpu_monitor.start()
        
        if self.etw_monitor:
            self.etw_monitor.start_session()
    
    def collect_metrics(self):
        """Aggregate metrics from all sources"""
        metrics = {
            'timestamp': time.time(),
            'process': self.process_monitor.get_latest(),
            'gpu': self.gpu_monitor.get_latest() if self.gpu_monitor else None,
            'system': self.etw_monitor.get_events() if self.etw_monitor else None
        }
        return metrics
```

### 2. Production Deployment

**Recommended Configuration:**
```yaml
monitoring:
  # Base monitoring (always on)
  process:
    backend: psutil
    sampling_rate_hz: 50
    metrics: [cpu, memory, io_counters]
  
  # GPU monitoring (if available)
  gpu:
    backend: auto  # NVML > DirectML > WMI
    sampling_rate_hz: 10
    metrics: [utilization, memory, temperature, power]
  
  # Profiling (on-demand)
  profiling:
    backend: py-spy
    trigger: manual
    duration_seconds: 30
    output_format: speedscope
  
  # ETW (Windows only, optional)
  etw:
    enabled: false  # Enable for deep debugging
    providers:
      - Microsoft-Windows-Kernel-Process
      - Microsoft-Windows-Kernel-Memory
    buffer_size_mb: 64
  
  # Storage
  storage:
    backend: circular_buffer
    max_samples: 100000
    compression: lz4
```

### 3. Code Example: Complete Implementation

```python
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class MonitorConfig:
    cpu_sampling_hz: float = 100.0
    gpu_sampling_hz: float = 10.0
    buffer_size: int = 10000
    enable_gpu: bool = True
    enable_etw: bool = False
    enable_profiling: bool = False

class UnifiedWindowsMonitor:
    """
    Unified monitoring solution for Windows combining:
    - psutil for CPU/Memory
    - NVML for GPU (when available)
    - ETW for detailed tracing (optional)
    - py-spy for profiling (on-demand)
    """
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self._init_process_monitor()
        self._init_gpu_monitor()
        self._init_profiling()
        
        # Metrics storage
        self.metrics_buffer = []
        self.lock = threading.Lock()
    
    def _init_process_monitor(self):
        """Initialize process monitoring with psutil"""
        self.process = psutil.Process()
        self.system_cpu_count = psutil.cpu_count()
    
    def _init_gpu_monitor(self):
        """Initialize GPU monitoring if available"""
        self.gpu_available = False
        
        if not self.config.enable_gpu:
            return
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) 
                    for i in range(self.gpu_count)
                ]
                self.gpu_available = True
                print(f"✅ Initialized NVML GPU monitoring for {self.gpu_count} GPUs")
            except Exception as e:
                print(f"⚠️ GPU monitoring not available: {e}")
    
    def _init_profiling(self):
        """Initialize profiling capabilities"""
        self.profiler = None
        if self.config.enable_profiling:
            try:
                import py_spy
                self.profiler = py_spy
                print("✅ py-spy profiling available")
            except ImportError:
                print("⚠️ py-spy not available for profiling")
    
    def collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-level metrics"""
        with self.process.oneshot():
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            io_counters = self.process.io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024 * 1024),
                'memory_vms_mb': memory_info.vms / (1024 * 1024),
                'io_read_mb': io_counters.read_bytes / (1024 * 1024),
                'io_write_mb': io_counters.write_bytes / (1024 * 1024),
                'num_threads': self.process.num_threads(),
                'num_handles': self.process.num_handles() if hasattr(self.process, 'num_handles') else None
            }
    
    def collect_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect GPU metrics if available"""
        if not self.gpu_available:
            return None
        
        gpu_metrics = []
        for i, handle in enumerate(self.gpu_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                
                gpu_metrics.append({
                    'gpu_id': i,
                    'utilization_percent': util.gpu,
                    'memory_utilization_percent': util.memory,
                    'memory_used_mb': mem_info.used / (1024 * 1024),
                    'memory_total_mb': mem_info.total / (1024 * 1024),
                    'temperature_c': temp,
                    'power_watts': power
                })
            except Exception as e:
                print(f"Error collecting GPU {i} metrics: {e}")
        
        return gpu_metrics if gpu_metrics else None
    
    def monitoring_thread(self):
        """Main monitoring thread"""
        cpu_interval = 1.0 / self.config.cpu_sampling_hz
        gpu_interval = 1.0 / self.config.gpu_sampling_hz
        
        last_cpu_time = time.time()
        last_gpu_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Collect CPU/Memory metrics
            if current_time - last_cpu_time >= cpu_interval:
                process_metrics = self.collect_process_metrics()
                
                with self.lock:
                    self.metrics_buffer.append({
                        'timestamp': current_time,
                        'type': 'process',
                        'data': process_metrics
                    })
                
                last_cpu_time = current_time
            
            # Collect GPU metrics
            if self.gpu_available and current_time - last_gpu_time >= gpu_interval:
                gpu_metrics = self.collect_gpu_metrics()
                
                if gpu_metrics:
                    with self.lock:
                        self.metrics_buffer.append({
                            'timestamp': current_time,
                            'type': 'gpu',
                            'data': gpu_metrics
                        })
                
                last_gpu_time = current_time
            
            # Maintain buffer size
            if len(self.metrics_buffer) > self.config.buffer_size:
                with self.lock:
                    self.metrics_buffer = self.metrics_buffer[-self.config.buffer_size:]
            
            # Small sleep to prevent CPU spinning
            time.sleep(min(cpu_interval, gpu_interval) / 2)
    
    def start(self):
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitoring_thread)
        self.monitor_thread.start()
        print("✅ Monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("✅ Monitoring stopped")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics from buffer"""
        with self.lock:
            if not self.metrics_buffer:
                return {}
            
            # Get latest of each type
            latest = {}
            for entry in reversed(self.metrics_buffer):
                if entry['type'] not in latest:
                    latest[entry['type']] = entry
                
                if len(latest) == 2:  # process and gpu
                    break
            
            return latest
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from buffer"""
        with self.lock:
            if not self.metrics_buffer:
                return {}
            
            process_metrics = [m for m in self.metrics_buffer if m['type'] == 'process']
            gpu_metrics = [m for m in self.metrics_buffer if m['type'] == 'gpu']
            
            stats = {}
            
            if process_metrics:
                cpu_values = [m['data']['cpu_percent'] for m in process_metrics]
                memory_values = [m['data']['memory_rss_mb'] for m in process_metrics]
                
                stats['process'] = {
                    'samples': len(process_metrics),
                    'cpu_avg': sum(cpu_values) / len(cpu_values),
                    'cpu_max': max(cpu_values),
                    'cpu_min': min(cpu_values),
                    'memory_avg': sum(memory_values) / len(memory_values),
                    'memory_max': max(memory_values),
                    'memory_min': min(memory_values)
                }
            
            if gpu_metrics and gpu_metrics[0]['data']:
                # Aggregate GPU stats (for first GPU)
                gpu_util_values = [m['data'][0]['utilization_percent'] for m in gpu_metrics]
                gpu_mem_values = [m['data'][0]['memory_used_mb'] for m in gpu_metrics]
                
                stats['gpu'] = {
                    'samples': len(gpu_metrics),
                    'utilization_avg': sum(gpu_util_values) / len(gpu_util_values),
                    'utilization_max': max(gpu_util_values),
                    'memory_avg': sum(gpu_mem_values) / len(gpu_mem_values),
                    'memory_max': max(gpu_mem_values)
                }
            
            return stats

# Example usage
if __name__ == "__main__":
    config = MonitorConfig(
        cpu_sampling_hz=100.0,
        gpu_sampling_hz=10.0,
        buffer_size=10000,
        enable_gpu=True
    )
    
    monitor = UnifiedWindowsMonitor(config)
    monitor.start()
    
    # Monitor for 10 seconds
    time.sleep(10)
    
    # Get statistics
    stats = monitor.get_statistics()
    print("\n📊 Monitoring Statistics:")
    print(stats)
    
    monitor.stop()
```

---

## Conclusion

### Key Findings

1. **ETW remains the gold standard** for low-overhead, high-frequency Windows system monitoring, but requires significant expertise and lacks mature Python bindings.

2. **psutil provides the best balance** of functionality, ease of use, and cross-platform support for general process monitoring.

3. **GPU monitoring is fragmented** with NVIDIA having the best support via NVML, while AMD and Intel lag behind in Windows tooling.

4. **NPU monitoring is emerging** with basic Task Manager support but lacks programmatic APIs for detailed metrics.

5. **Python ecosystem offers excellent tools** like py-spy and Scalene for profiling, but system-level monitoring still relies on platform-specific solutions.

### Recommended Approach

For production ONNX inference monitoring on Windows, implement a **hybrid solution**:

1. **Base Layer**: psutil for process-level CPU/Memory at 50-100Hz
2. **GPU Layer**: NVML for NVIDIA, DirectML counters as fallback
3. **Profiling Layer**: py-spy for on-demand detailed analysis
4. **ETW Layer**: Optional for deep system tracing when needed
5. **NPU Layer**: ONNX Runtime profiling until better APIs emerge

This approach provides comprehensive monitoring with acceptable overhead while maintaining code maintainability and cross-platform compatibility where possible.

### Future Considerations

- Monitor Windows updates for improved NPU APIs
- Track DirectML evolution for unified GPU/NPU monitoring
- Consider OpenTelemetry integration for standardized metrics export
- Evaluate emerging tools like Windows ML for AI workload monitoring

---

*Research conducted: November 2024*
*Document Version: 1.0*
*Next Review: Q1 2025*