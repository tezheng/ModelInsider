# ETW Monitor Examples

This directory contains examples demonstrating the use of ETW (Event Tracing for Windows) and PSUtil monitors for system profiling during ONNX inference.

## Examples

### 1. Simple ETW Comparison (`simple_etw_comparison.py`)

Compares PSUtil (polling-based) and ETW (event-driven) monitoring side by side.

**Usage:**
```bash
# Compare both monitors (Windows only for ETW)
python simple_etw_comparison.py

# Run with custom duration and sampling rate
python simple_etw_comparison.py --duration 10 --rate 100

# Run only PSUtil monitoring (cross-platform)
python simple_etw_comparison.py --psutil-only

# Run only ETW monitoring (Windows only)
python simple_etw_comparison.py --etw-only
```

**Key Features:**
- Side-by-side comparison of monitoring technologies
- Shows overhead differences (PSUtil: ~35% at 100Hz, ETW: <2%)
- Displays top processes by CPU usage
- Works on all platforms (ETW features on Windows only)

### 2. ETW ONNX Inference Monitor (`etw_onnx_inference_monitor.py`)

Profiles ONNX model inference with kernel-level ETW monitoring.

**Usage:**
```bash
# Profile ONNX model with ETW monitoring (Windows)
python etw_onnx_inference_monitor.py --model-path model.onnx

# Run with custom parameters
python etw_onnx_inference_monitor.py \
    --model-path model.onnx \
    --num-inferences 200 \
    --warmup 20

# Use PSUtil instead of ETW (cross-platform)
python etw_onnx_inference_monitor.py \
    --model-path model.onnx \
    --use-psutil
```

**Key Features:**
- Profiles ONNX inference with system monitoring
- Collects inference latency statistics (mean, P95, P99)
- Tracks CPU and memory usage during inference
- ETW provides kernel-level metrics (CPU cycles, context switches)
- Saves results to JSON file

## Monitor Comparison

### PSUtil Monitor
- **Technology**: Polling-based process monitoring
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **Overhead**: ~35% CPU at 100Hz sampling
- **Metrics**: CPU%, memory, I/O counters
- **Pros**: Simple, well-documented, cross-platform
- **Cons**: Higher overhead, can miss short events

### ETW Monitor
- **Technology**: Event-driven kernel monitoring
- **Platform**: Windows only
- **Overhead**: <2% CPU at 100Hz sampling
- **Metrics**: CPU cycles, context switches, kernel/user time, page faults
- **Pros**: Very low overhead, exact timing, kernel-level detail
- **Cons**: Windows-only, complex, may require admin rights

## Requirements

```bash
# Basic requirements
pip install psutil numpy

# For ONNX inference examples
pip install onnxruntime

# For Windows ETW (optional, for better integration)
pip install pywin32
```

## Sample Output

### PSUtil Monitoring
```
PSUtil Monitor (Polling-Based)
================================
Platform: Windows
Sampling Rate: 100.0Hz
Expected Overhead: ~35.0% CPU

Top 5 Processes by CPU (PSUtil):
  python.exe           (PID:  12345): CPU= 45.23%, Memory=  256.45MB
  chrome.exe           (PID:  23456): CPU= 12.34%, Memory=  512.12MB
```

### ETW Monitoring (Windows)
```
ETW Monitor (Event-Driven)
==========================
Platform: Windows (Kernel-Level)
Expected Overhead: <2% CPU

Top 5 Processes by CPU Cycles (ETW Kernel Metrics):
  python.exe           (PID:  12345):
    CPU Cycles: 1,234,567,890
    Context Switches: 1234
    Kernel Time: 45.2ms, User Time: 123.4ms
```

## Recommendations

- **Development/Testing**: Use PSUtil for simplicity
- **Production Windows (≤50Hz)**: PSUtil is sufficient
- **Production Windows (>50Hz)**: Use ETW for low overhead
- **Cross-platform**: PSUtil is the only option
- **Kernel-level metrics**: ETW provides unique insights

## Troubleshooting

### ETW Not Available
- ETW only works on Windows
- Some features may require admin rights
- Fallback to PSUtil is automatic

### High PSUtil Overhead
- Reduce sampling rate (e.g., 10Hz instead of 100Hz)
- Use ETW on Windows for better performance
- Consider adaptive sampling

### ONNX Model Loading Issues
- Ensure model file exists and is valid ONNX format
- Check ONNX Runtime version compatibility
- Verify input shape requirements