# System Monitors for ONNX Inference Profiling

This directory contains system-level monitoring implementations for TEZ-158.

## Structure

```
monitors/
├── README.md                  # This file
├── __init__.py               # Package initialization
├── base_monitor.py           # Base monitor abstract class
├── etw_monitor.py           # ETW integration for Windows (TEZ-165)
├── cpu_monitor.py           # CPU metrics collection (TEZ-164)  
├── memory_monitor.py        # Memory metrics collection (TEZ-164)
├── energy_monitor.py        # Energy and thermal monitoring (TEZ-166)
└── system_monitor.py        # Main system monitor orchestrator (TEZ-164)
```

## Implementation Status

- [ ] TEZ-164: Core System Metrics Collection
- [🚧] TEZ-165: ETW Integration for Windows (In Progress)
- [ ] TEZ-166: Energy and Thermal Monitoring
- [ ] TEZ-167: Model Metrics (Inference Performance)
- [ ] TEZ-168: Operator Metrics (Operation-Level)
- [ ] TEZ-169: Dashboard Integration

## ETW Monitor (TEZ-165)

The ETW (Event Tracing for Windows) monitor provides kernel-level process monitoring with minimal overhead.

### Key Features
- Per-process CPU usage tracking
- Per-process memory monitoring
- High-frequency sampling (10Hz to 100Hz)
- Minimal performance overhead (<0.5ms)

### Implementation Approach
1. **Modern ETW via pythonnet**: Using Microsoft.Diagnostics.Tracing.TraceEvent
2. **Process Performance Counters**: Direct access to Windows performance data
3. **Kernel Provider Events**: Process, thread, and memory events
4. **Real-time Processing**: Event streaming with buffering

### Dependencies
- pythonnet: .NET interop for TraceEvent library
- Microsoft.Diagnostics.Tracing.TraceEvent: Modern ETW consumption
- Windows 10/11: Required for ETW functionality