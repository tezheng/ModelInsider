# QNN Profiling Integration POC

This directory contains a proof-of-concept implementation for integrating Qualcomm QNN profiling capabilities with the modelexport project to enable NPU performance monitoring and optimization.

## Overview

The POC demonstrates how to:
- Enable QNN profiling during model inference on Qualcomm NPU/HTP
- Capture hardware-level metrics (HVX/HMX/VTCM usage)
- Generate Chrome trace visualizations for performance analysis
- Extract and analyze performance bottlenecks
- Compare metrics against baselines

## Key Files

- `docs/QNN_PROFILING_DEEP_DIVE.md` - Comprehensive technical documentation of QNN profiling capabilities
- `qnn_profiling_poc.py` - Basic POC demonstrating profiling concepts and metrics extraction
- `qnn_profiling_integration.py` - Advanced integration with actual QNN Python SDK APIs
- `README.md` - This file

## Features Demonstrated

### 1. Profiling Configuration
- Multiple profiling levels (basic, detailed, backend, client, linting*)
- Performance profiles (balanced, high_performance, extreme_performance)
- OpTrace generation for Chrome visualization
- QHAS (QNN HTP Analysis Summary) reports

*Note: Linting level requires native C++ SDK

### 2. Metrics Captured

#### Available via Python SDK
- Total inference time
- HTP execution time  
- VTCM acquisition time
- Resource power-up time (HMX + HVX)
- DDR bandwidth usage
- Node-level execution times
- Hardware resource utilization percentages

#### Native SDK Exclusive
- Real-time hardware counter access
- Linting-level node analysis
- Fine-grained RPC timing splits
- Multi-graph yielding details
- Custom profiling event handlers

### 3. Visualization Outputs
- Chrome trace JSON (viewable at chrome://tracing)
- QHAS summary reports (JSON/HTML)
- Performance bottleneck analysis
- Baseline comparison reports

## Usage

### Basic POC (No Dependencies)
```bash
# Run the basic POC (simulates profiling without QNN SDK)
python qnn_profiling_poc.py
```

### Advanced Integration (Requires QNN SDK)
```bash
# Ensure QNN SDK is installed at /mnt/c/Qualcomm/AIStack/qairt/
# Or set QNN_SDK_ROOT environment variable

# Run the advanced integration
python qnn_profiling_integration.py
```

### Integration with ModelExport

```python
from experiments.tez_159_qnn_infer_perf.qnn_profiling_integration import ModelExportQNNProfiler

# Initialize profiler
profiler = ModelExportQNNProfiler(
    profiling_level="detailed",
    perf_profile="high_performance", 
    enable_optrace=True
)

# Profile model execution
outputs, metrics = profiler.profile_model_execution(
    model_path="model.dlc",
    inputs={"input": input_tensor},
    backend="htp"
)

# Analyze bottlenecks
bottlenecks = profiler.analyze_bottlenecks(metrics)
```

## Output Examples

### Chrome Trace Visualization
- Shows parallel execution timeline
- Hardware resource utilization over time
- Node-level execution details
- Critical path analysis

### QHAS Summary
```json
{
  "execution_summary": {
    "total_inference_time_ms": 12.5,
    "htp_execution_time_ms": 11.2,
    "overhead_ms": 1.3,
    "throughput_fps": 80.0
  },
  "resource_utilization": {
    "hvx_utilization_percent": 72.5,
    "hmx_utilization_percent": 65.3,
    "vtcm_peak_usage_kb": 768,
    "ddr_bandwidth_mbps": 1250.5
  }
}
```

### Bottleneck Analysis
- VTCM wait time analysis
- Hardware utilization gaps
- Slow node identification
- Optimization recommendations

## Next Steps for Integration

1. **CLI Integration**
   ```bash
   modelexport profile MODEL.onnx \
       --backend htp \
       --profiling-level detailed \
       --output-chrome-trace
   ```

2. **Automated Performance Regression Testing**
   - Profile models during CI/CD
   - Compare against baseline metrics
   - Flag performance regressions

3. **Model Optimization Workflow**
   - Profile → Identify bottlenecks → Optimize → Re-profile
   - Automated suggestions for optimization

4. **Multi-Device Profiling**
   - Profile across different Snapdragon chipsets
   - Generate device-specific optimization strategies

## Requirements

### Python SDK Path
- Supported: `qairt.api.profiler`, `qti.aisw.core.model_level_api`
- Requires: QNN SDK 2.34.0+ installed

### Native SDK (Optional)
- For linting-level profiling
- Custom event handlers
- Real-time hardware counters

## Limitations

1. **Python SDK**: ~80% of native SDK functionality
2. **Linting level**: Requires native C++ SDK
3. **Platform support**: Linux and Windows (WOS) only
4. **Profiling overhead**: 5-15% runtime increase

## References

- [QNN SDK Documentation](https://developer.qualcomm.com/)
- [Chrome Tracing](chrome://tracing)
- [Qualcomm AI Hub](https://app.aihub.qualcomm.com/docs/hub/)
- See `docs/QNN_PROFILING_DEEP_DIVE.md` for comprehensive technical details

## License

This POC is part of the modelexport project and follows the same licensing terms.