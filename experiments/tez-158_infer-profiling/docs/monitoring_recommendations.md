# Windows Monitoring Implementation Recommendations

## Executive Summary

Based on comprehensive research of Windows monitoring technologies, here are actionable recommendations for implementing high-frequency process/thread monitoring for ONNX inference profiling.

## Immediate Actions (Phase 1)

### 1. Enhance Current Implementation

**Current State:** Basic psutil-based monitoring at 100Hz

**Recommended Improvements:**

```python
# Enhanced PSUtil Monitor with better Windows support
class EnhancedWindowsMonitor(BaseMonitor):
    def __init__(self, config):
        super().__init__(config)
        
        # Use Windows-specific optimizations
        if platform.system() == 'Windows':
            # Enable fast process handle caching
            self.process = psutil.Process()
            self.process.cpu_percent()  # Initialize CPU monitoring
            
            # Pre-cache Windows Performance Counters
            self._init_performance_counters()
    
    def _init_performance_counters(self):
        """Initialize Windows Performance Counters for better accuracy"""
        try:
            import win32pdh
            import win32pdhutil
            
            # Setup counters for precise metrics
            self.pdh_counters = {
                'cpu': r'\Process(python)\% Processor Time',
                'private_bytes': r'\Process(python)\Private Bytes',
                'working_set': r'\Process(python)\Working Set',
                'io_read': r'\Process(python)\IO Read Bytes/sec',
                'io_write': r'\Process(python)\IO Write Bytes/sec'
            }
        except ImportError:
            self.pdh_counters = None
```

### 2. Add GPU Monitoring Support

**Priority:** High (GPU is critical for ML inference)

```python
# GPU Monitor Implementation
class GPUMonitor:
    @staticmethod
    def create():
        """Factory method to create appropriate GPU monitor"""
        # Try NVIDIA first (most common for ML)
        try:
            import pynvml
            pynvml.nvmlInit()
            return NVIDIAGPUMonitor()
        except:
            pass
        
        # Try Windows Performance Counters
        try:
            import win32pdh
            return WindowsGPUMonitor()
        except:
            pass
        
        return None

class NVIDIAGPUMonitor:
    def __init__(self):
        import pynvml
        self.pynvml = pynvml
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                       for i in range(self.device_count)]
    
    def get_metrics(self):
        metrics = []
        for handle in self.handles:
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.append({
                'gpu_util': util.gpu,
                'mem_util': util.memory,
                'mem_used_mb': mem.used / 1024**2,
                'mem_total_mb': mem.total / 1024**2
            })
        return metrics
```

**Installation Requirements:**
```bash
# For NVIDIA GPU monitoring
pip install pynvml nvidia-ml-py

# For Windows Performance Counters
pip install pywin32
```

## Medium-Term Enhancements (Phase 2)

### 1. Implement ETW Integration

**Why ETW?**
- Lowest overhead (<2% at 100Hz)
- Thread-level detail
- Kernel-level accuracy
- Can capture system-wide events

**Implementation Approach:**

```python
# ETW Integration using Python
class ETWProvider:
    def __init__(self):
        # Option 1: Use pyetw (if available)
        try:
            import pyetw
            self.backend = 'pyetw'
        except:
            # Option 2: Use ctypes for direct Windows API
            self.backend = 'ctypes'
            self._init_ctypes_etw()
    
    def _init_ctypes_etw(self):
        """Direct ETW API access via ctypes"""
        import ctypes
        from ctypes import wintypes
        
        # Load Windows DLLs
        self.advapi32 = ctypes.WinDLL('advapi32')
        self.tdh = ctypes.WinDLL('tdh')
        
        # Define ETW structures
        class EVENT_TRACE_PROPERTIES(ctypes.Structure):
            _fields_ = [
                ('Wnode', ctypes.c_byte * 48),
                ('BufferSize', wintypes.ULONG),
                ('MinimumBuffers', wintypes.ULONG),
                ('MaximumBuffers', wintypes.ULONG),
                ('MaximumFileSize', wintypes.ULONG),
                ('LogFileMode', wintypes.ULONG),
                ('FlushTimer', wintypes.ULONG),
                ('EnableFlags', wintypes.ULONG),
                ('AgeLimit', wintypes.LONG),
                ('NumberOfBuffers', wintypes.ULONG),
                ('FreeBuffers', wintypes.ULONG),
                ('EventsLost', wintypes.ULONG),
                ('BuffersWritten', wintypes.ULONG),
                ('LogBuffersLost', wintypes.ULONG),
                ('RealTimeBuffersLost', wintypes.ULONG),
                ('LoggerThreadId', wintypes.HANDLE),
                ('LogFileNameOffset', wintypes.ULONG),
                ('LoggerNameOffset', wintypes.ULONG)
            ]
```

**Alternative: Use WPA Python Wrapper**
```python
# Use Windows Performance Analyzer via Python
import subprocess
import json

class WPAProfiler:
    def start_trace(self, duration=10):
        """Start WPA trace for specified duration"""
        cmd = [
            'wpr', '-start', 'CPU', '-start', 'GPU',
            '-start', 'Memory', '-filemode'
        ]
        subprocess.run(cmd, check=True)
        
        time.sleep(duration)
        
        cmd = ['wpr', '-stop', 'trace.etl']
        subprocess.run(cmd, check=True)
    
    def analyze_trace(self, trace_file):
        """Analyze ETL trace file"""
        # Use wpaexporter to extract data
        cmd = ['wpaexporter', trace_file, '-profile', 'cpu_usage.wpaProfile']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_wpa_output(result.stdout)
```

### 2. Add NPU Monitoring

**Current Status:** Limited API support, but growing rapidly

```python
class NPUMonitor:
    def __init__(self):
        self.backend = self._detect_npu()
    
    def _detect_npu(self):
        # Check Windows version (requires Windows 11 24H2+)
        import sys
        if sys.getwindowsversion().build < 26100:
            return None
        
        # Try ONNX Runtime with NPU provider
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'DmlExecutionProvider' in providers:
                return 'DirectML'
            elif 'QnnExecutionProvider' in providers:
                return 'Qualcomm'
            elif 'OpenVINOExecutionProvider' in providers:
                return 'Intel'
        except:
            pass
        
        return None
    
    def get_metrics(self):
        """Get NPU metrics from Task Manager API (when available)"""
        # Currently limited to Task Manager visibility
        # Future: Use WMI or Performance Counters when available
        pass
```

## Long-Term Strategy (Phase 3)

### 1. Unified Monitoring Framework

```yaml
# Proposed Architecture
monitoring_stack:
  core:
    - psutil      # Base process monitoring
    - py-spy      # Sampling profiler
    
  gpu:
    - pynvml      # NVIDIA GPUs
    - rocm-smi    # AMD GPUs (via WSL if needed)
    - intel-gpu   # Intel GPUs (future)
    
  windows_specific:
    - etw         # Low-level tracing
    - pdh         # Performance counters
    - wmi         # Remote monitoring
    
  npu:
    - onnx_runtime # NPU profiling via ONNX
    - directml    # Windows ML acceleration
    
  export:
    - prometheus  # Metrics export
    - opentelemetry # Distributed tracing
```

### 2. Production Configuration

```python
# Production-Ready Configuration
PRODUCTION_CONFIG = {
    'monitoring': {
        'cpu': {
            'backend': 'psutil',  # or 'etw' for lower overhead
            'sampling_hz': 50,    # Balance between detail and overhead
            'aggregation': 'avg',  # avg, max, p95, p99
        },
        'memory': {
            'backend': 'psutil',
            'sampling_hz': 10,
            'track_allocations': False,  # Enable only for debugging
        },
        'gpu': {
            'backend': 'auto',  # Auto-detect available GPU
            'sampling_hz': 10,
            'metrics': ['utilization', 'memory', 'temperature'],
        },
        'npu': {
            'backend': 'onnx_runtime',
            'enable_profiling': True,
        },
        'storage': {
            'type': 'circular_buffer',
            'max_samples': 100000,
            'persist_to_disk': False,
        },
        'overhead_limit': 5.0,  # Maximum acceptable overhead %
    }
}
```

## Specific Recommendations

### For TEZ-158 Implementation

1. **Keep Current psutil Base** ✅
   - Already implemented and working
   - Good balance of functionality and ease
   - Cross-platform compatibility

2. **Add GPU Monitoring** 🔄
   ```bash
   pip install pynvml
   # Add to monitors/gpu_monitor.py
   ```

3. **Prepare ETW Infrastructure** 📋
   ```python
   # Create monitors/etw_advanced.py
   # Start with read-only ETW consumer
   # Use existing ETW sessions from Windows
   ```

4. **Implement Adaptive Sampling** 💡
   ```python
   class AdaptiveSampler:
       def adjust_rate(self, overhead_percent):
           if overhead_percent > 5:
               self.sampling_hz *= 0.8
           elif overhead_percent < 2:
               self.sampling_hz *= 1.2
   ```

### Tool Selection Matrix

| Use Case | Recommended Tool | Reason |
|----------|-----------------|---------|
| General CPU/Memory | psutil | Easy, reliable, cross-platform |
| High-frequency CPU | ETW | Lowest overhead, kernel accuracy |
| GPU Monitoring | pynvml | Most mature, best documentation |
| Production Profiling | py-spy | Low overhead, no code changes |
| Development Profiling | Scalene | Rich features, AI suggestions |
| Remote Monitoring | WMI + Prometheus | Standard tools, good integration |
| NPU Monitoring | ONNX Runtime | Only current option |

### Performance Targets

```python
# Recommended sampling rates by metric
SAMPLING_RATES = {
    'cpu_percent': 50,      # Hz - Good balance
    'memory': 10,           # Hz - Changes slowly  
    'gpu_utilization': 10,  # Hz - Hardware limited
    'disk_io': 1,          # Hz - Low frequency needed
    'network': 1,          # Hz - Low frequency needed
    'npu': 'event_based',  # On inference events
}

# Maximum acceptable overhead
MAX_OVERHEAD = {
    'development': 10.0,   # % - Can be higher
    'testing': 5.0,        # % - Moderate
    'production': 2.0,     # % - Must be minimal
}
```

## Implementation Timeline

### Week 1-2: Foundation
- [x] Basic psutil monitoring (DONE)
- [ ] Add GPU monitoring via pynvml
- [ ] Implement circular buffer optimization
- [ ] Add overhead tracking

### Week 3-4: Enhancement
- [ ] ETW consumer implementation
- [ ] Performance counter integration
- [ ] Adaptive sampling implementation
- [ ] Profiling integration (py-spy)

### Week 5-6: Advanced Features
- [ ] NPU monitoring exploration
- [ ] Remote monitoring setup
- [ ] Metrics export (Prometheus)
- [ ] Dashboard creation

### Week 7-8: Production Hardening
- [ ] Performance optimization
- [ ] Error handling improvement
- [ ] Documentation completion
- [ ] Integration testing

## Code Repository Structure

```
experiments/tez-158_infer-profiling/
├── monitors/
│   ├── base_monitor.py          ✅ Implemented
│   ├── etw_monitor.py           ✅ Basic implementation
│   ├── gpu_monitor.py           📋 TODO: Add NVML
│   ├── npu_monitor.py           📋 TODO: Future
│   ├── unified_monitor.py       📋 TODO: Combine all
│   └── adaptive_sampler.py      📋 TODO: Smart sampling
├── profilers/
│   ├── pyspy_profiler.py        📋 TODO
│   ├── scalene_profiler.py      📋 TODO
│   └── etw_profiler.py          📋 TODO
├── exporters/
│   ├── prometheus_exporter.py   📋 TODO
│   ├── csv_exporter.py          📋 TODO
│   └── otlp_exporter.py         📋 TODO
└── tests/
    └── perf/
        ├── test_overhead.py      📋 TODO
        ├── test_accuracy.py      📋 TODO
        └── test_stability.py     📋 TODO
```

## Conclusion

The current psutil-based implementation is a solid foundation. The next priority should be:

1. **Add GPU monitoring** (immediate value for ML workloads)
2. **Implement adaptive sampling** (reduce overhead automatically)
3. **Prepare ETW integration** (for future deep profiling needs)
4. **Design for NPU support** (emerging technology, be ready)

The hybrid approach combining psutil + GPU monitoring + on-demand ETW will provide the best balance of functionality, ease of use, and performance for ONNX inference profiling on Windows.