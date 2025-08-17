# Enhanced Performance Metrics Specification for ONNX Inference Profiling

**Document Version**: 2.0  
**Date**: 2025-08-14  
**Status**: Enhanced after industry research  
**Priority**: Critical additions for cutting-edge profiling

## Executive Summary

Based on comprehensive research of industry-leading profiling tools and best practices, this document specifies enhanced metrics that will position our ONNX inference profiling system at the forefront of performance monitoring. The additions focus on critical gaps identified: **energy efficiency**, **tail latency analysis**, and **deep hardware utilization metrics**.

## Metrics Classification

### Tier 1: Critical Performance Metrics (Must Have)

#### 1.1 Latency Metrics (Enhanced)

##### Core Latency
- **First Token Latency** (for generative models): Time to first output token
- **End-to-End Latency**: Total request processing time including pre/post-processing
- **Inference-Only Latency**: Pure model execution time
- **Time-to-First-Byte (TTFB)**: Network response initiation time

##### Tail Latency Analysis (NEW - Critical Gap)
```python
{
  "latency_percentiles": {
    "p50": 4.2,     # Median latency (ms)
    "p90": 5.1,     # 90th percentile
    "p95": 5.8,     # 95th percentile
    "p99": 8.3,     # 99th percentile - critical for SLAs
    "p99.9": 12.1,  # 99.9th percentile - outlier detection
    "max": 45.2     # Maximum observed latency
  },
  "latency_distribution": {
    "buckets": [0, 1, 2, 5, 10, 20, 50, 100],  # ms
    "counts": [0, 12, 145, 823, 134, 23, 5, 1]
  },
  "jitter": 1.2,    # Standard deviation of latency
  "stability_score": 0.92  # 1.0 = perfectly stable
}
```

##### Warm-up Analysis (NEW)
- **Cold Start Penalty**: First inference after model load
- **Warm-up Iterations**: Number of inferences to reach stable performance
- **JIT Compilation Time**: Time spent in just-in-time compilation
- **Cache Priming Duration**: Time to populate all caches

#### 1.2 Throughput Metrics (Enhanced)

##### Token-Level Throughput (NEW - for LLMs)
- **Tokens/Second**: For generative models
- **Prefill Throughput**: Input processing rate
- **Generation Throughput**: Output token generation rate
- **Effective Batch Throughput**: Accounting for padding

##### Request-Level Throughput
- **Requests/Second (RPS)**: Overall system throughput
- **Goodput**: Successfully completed requests/second
- **Sustained Throughput**: Maximum sustainable RPS over 1 hour
- **Peak Burst Throughput**: Maximum RPS for short bursts

#### 1.3 Energy Efficiency Metrics (NEW - Critical Gap)

##### Power Consumption
```python
{
  "power_metrics": {
    "instantaneous_power_w": 45.3,        # Current power draw (Watts)
    "average_power_w": 42.1,              # Average over window
    "peak_power_w": 78.5,                 # Maximum observed
    "idle_power_w": 12.3,                 # Baseline when idle
    "dynamic_power_w": 29.8               # Power above idle
  },
  "energy_efficiency": {
    "inferences_per_watt": 4.7,           # Throughput/Power
    "inferences_per_joule": 0.0047,       # Energy efficiency
    "flops_per_watt": 1.2e12,            # Computational efficiency
    "tokens_per_joule": 0.23,            # For LLMs
    "performance_per_watt": 0.85          # Relative to theoretical max
  },
  "thermal_metrics": {
    "temperature_c": 72,                  # Current temperature
    "thermal_throttling": false,          # Throttling status
    "fan_speed_rpm": 3200,               # Cooling system
    "thermal_headroom_c": 18             # Before throttling
  }
}
```

##### Carbon Footprint (NEW - Sustainability)
- **CO2 per Inference**: Based on regional grid carbon intensity
- **Daily Carbon Footprint**: Total CO2 emissions
- **Green Energy Percentage**: If available from data center

### Tier 2: Hardware Utilization Metrics (Enhanced)

#### 2.1 GPU Metrics (Enhanced)

##### SM (Streaming Multiprocessor) Analysis (NEW)
```python
{
  "sm_utilization": {
    "active_warps": 48,                   # Active thread groups
    "max_warps": 64,                      # Maximum possible
    "occupancy": 0.75,                    # Active/Max ratio
    "achieved_occupancy": 0.68,           # Actual vs theoretical
    "warp_efficiency": 0.89               # Non-divergent execution
  },
  "tensor_core_metrics": {
    "tensor_core_usage": 0.82,            # Utilization percentage
    "tensor_ops_per_second": 125e12,     # TOPS
    "fp16_throughput": 0.91,             # FP16 efficiency
    "int8_throughput": 0.88,             # INT8 efficiency
    "mixed_precision_ratio": 0.7         # Percentage using TC
  }
}
```

##### Memory Hierarchy (NEW)
```python
{
  "memory_bandwidth": {
    "l1_cache_hit_rate": 0.94,
    "l2_cache_hit_rate": 0.87,
    "shared_memory_usage_kb": 48,
    "shared_memory_throughput_gbps": 1250,
    "global_memory_throughput_gbps": 760,
    "memory_bandwidth_utilization": 0.68,
    "ecc_errors": 0,
    "memory_clock_mhz": 9500
  },
  "memory_allocation": {
    "persistent_memory_mb": 2048,         # Model weights
    "activation_memory_mb": 512,          # Intermediate tensors
    "workspace_memory_mb": 256,           # Temporary buffers
    "fragmentation_ratio": 0.12           # Memory fragmentation
  }
}
```

#### 2.2 NPU/AI Accelerator Metrics (Enhanced)

##### Qualcomm HTP Specific (NEW)
```python
{
  "htp_metrics": {
    "htp_utilization": 0.89,
    "vtcm_usage_mb": 8,                   # Vector TCM usage
    "vtcm_bandwidth_gbps": 2000,
    "hvx_utilization": 0.92,              # Hexagon Vector Extensions
    "mac_efficiency": 0.87,               # MAC unit utilization
    "dma_efficiency": 0.91,               # DMA transfer efficiency
    "power_state": "TURBO",               # NOMINAL/TURBO/POWER_SAVE
    "clock_frequency_mhz": 1500
  }
}
```

##### Intel NPU Metrics (NEW)
```python
{
  "intel_npu": {
    "shave_utilization": 0.85,            # SHAVE processor usage
    "cmx_memory_usage": 0.72,             # Connection Matrix memory
    "nce_utilization": 0.88,              # Neural Compute Engine
    "power_level": 3,                     # 0-3 power states
    "inference_efficiency": 0.91
  }
}
```

#### 2.3 CPU Metrics (Enhanced)

##### Vectorization Efficiency (NEW)
```python
{
  "vectorization": {
    "avx512_usage": 0.67,                 # AVX-512 utilization
    "vnni_usage": 0.82,                   # VNNI for INT8
    "simd_efficiency": 0.74,              # SIMD instruction ratio
    "vector_width_utilization": 0.88     # Actual vs max vector width
  },
  "cpu_cache": {
    "l1d_hit_rate": 0.96,
    "l1i_hit_rate": 0.98,
    "l2_hit_rate": 0.89,
    "l3_hit_rate": 0.76,
    "cache_misses_per_inference": 142,
    "tlb_hit_rate": 0.94
  },
  "numa_metrics": {
    "local_memory_access": 0.91,          # NUMA local access ratio
    "remote_memory_access": 0.09,
    "numa_balance": 0.95                  # Work distribution balance
  }
}
```

### Tier 3: Operation-Level Metrics (Enhanced)

#### 3.1 Kernel Profiling (NEW)

##### Individual Kernel Metrics
```python
{
  "kernel_name": "Conv2d_3x3_NHWC",
  "kernel_metrics": {
    "execution_time_us": 234,
    "grid_size": [256, 128, 1],
    "block_size": [16, 16, 1],
    "registers_per_thread": 32,
    "shared_memory_bytes": 16384,
    "arithmetic_intensity": 45.2,         # FLOPS/byte
    "achieved_flops": 2.3e12,
    "theoretical_flops": 2.8e12,
    "efficiency": 0.82
  }
}
```

##### Kernel Fusion Analysis (NEW)
```python
{
  "fusion_metrics": {
    "fused_operations": ["Conv2D", "BatchNorm", "ReLU"],
    "unfused_time_ms": 1.2,
    "fused_time_ms": 0.8,
    "fusion_speedup": 1.5,
    "memory_savings_mb": 24,
    "fusion_opportunities_missed": 3
  }
}
```

#### 3.2 Graph Optimization Metrics (NEW)

```python
{
  "graph_optimization": {
    "original_ops_count": 523,
    "optimized_ops_count": 387,
    "constant_folding_eliminated": 45,
    "dead_code_eliminated": 12,
    "layout_transforms_removed": 8,
    "quantization_ops_fused": 23,
    "optimization_time_ms": 145,
    "graph_memory_reduction": 0.23
  }
}
```

### Tier 4: Batching & Concurrency Metrics (Enhanced)

#### 4.1 Dynamic Batching Analysis (NEW)

```python
{
  "batching_metrics": {
    "average_batch_size": 12.3,
    "batch_size_distribution": {
      "1": 0.15,  "2-4": 0.25,  "5-8": 0.35,
      "9-16": 0.20,  "17-32": 0.05
    },
    "padding_overhead": 0.18,              # Wasted computation
    "queue_wait_time_ms": 2.3,
    "batch_formation_time_us": 145,
    "optimal_batch_size": 16,
    "batching_efficiency": 0.77
  }
}
```

#### 4.2 Concurrency Metrics (NEW)

```python
{
  "concurrency": {
    "concurrent_requests": 8,
    "max_concurrency": 16,
    "queue_depth": 12,
    "rejected_requests": 0,
    "contention_ratio": 0.12,
    "lock_wait_time_us": 23,
    "parallel_efficiency": 0.89
  }
}
```

### Tier 5: System-Level Metrics (Enhanced)

#### 5.1 I/O Metrics (NEW)

```python
{
  "io_metrics": {
    "pcie_throughput_gbps": 12.3,
    "pcie_utilization": 0.45,
    "host_to_device_bandwidth": 11.2,
    "device_to_host_bandwidth": 10.8,
    "nvlink_throughput_gbps": 180,        # If available
    "io_wait_time_ms": 0.3,
    "dma_transfer_time_ms": 0.8
  }
}
```

#### 5.2 Memory Pressure Metrics (NEW)

```python
{
  "memory_pressure": {
    "swap_usage_mb": 0,
    "page_faults_per_sec": 12,
    "memory_reclaim_events": 0,
    "oom_risk_score": 0.02,               # 0-1 scale
    "gc_pause_time_ms": 0,                # If applicable
    "memory_compaction_time_ms": 0
  }
}
```

## Metric Collection Strategies

### High-Frequency Metrics (100Hz+)
- Instantaneous power consumption
- SM/NPU utilization
- Memory bandwidth
- Queue depth

### Medium-Frequency Metrics (10Hz)
- Latency percentiles
- Throughput rates
- Temperature
- Cache hit rates

### Low-Frequency Metrics (1Hz)
- Energy efficiency calculations
- Carbon footprint
- Thermal throttling status
- Memory fragmentation

## Implementation Priority Matrix

| Priority | Metric Category | Business Impact | Implementation Complexity | Timeline |
|----------|----------------|-----------------|--------------------------|----------|
| P0 | Energy Efficiency | Critical - Sustainability & Cost | Medium | 30 days |
| P0 | Tail Latency (P99) | Critical - SLA Management | Low | 15 days |
| P1 | SM/Tensor Core Usage | High - GPU Optimization | High | 45 days |
| P1 | Dynamic Batching | High - Throughput | Medium | 30 days |
| P2 | Kernel Profiling | Medium - Deep Optimization | High | 60 days |
| P2 | NUMA Metrics | Medium - CPU Optimization | Medium | 45 days |
| P3 | Carbon Footprint | Low - Reporting | Low | 15 days |
| P3 | Graph Optimization | Low - One-time | Medium | 30 days |

## Competitive Analysis

### Current Industry Leaders

| Tool | Energy | Tail Latency | Hardware Detail | Batching | Our Position |
|------|--------|--------------|-----------------|----------|--------------|
| NVIDIA Nsight | ✅ | ✅ | ✅✅✅ | ✅ | Will match |
| Intel VTune | ✅ | ✅ | ✅✅ | ❌ | Will exceed |
| AWS SageMaker | ❌ | ✅ | ❌ | ✅ | Will exceed |
| Azure ML | ❌ | ✅ | ❌ | ✅ | Will exceed |
| TensorRT | ✅ | ❌ | ✅✅ | ✅ | Will match |
| **Our Solution** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | **Leader** |

## MLPerf Alignment

Our metrics align with MLPerf Inference v4.0 standards:
- **LoadGen Compliance**: Latency percentiles, throughput
- **Power Measurement**: Following MLPerf Power WG specifications
- **Scenario Coverage**: Single-stream, multi-stream, server, offline
- **Quality Metrics**: Excluded per requirement (focus on performance)

## Data Schema Extensions

### Enhanced Inference Event
```json
{
  "timestamp_ns": 1234567890000000,
  "inference_id": "uuid",
  "model_name": "mobilevit_v2",
  
  // Latency breakdown
  "latency": {
    "end_to_end_ms": 4.71,
    "inference_only_ms": 4.52,
    "preprocessing_ms": 0.12,
    "postprocessing_ms": 0.07,
    "queue_wait_ms": 0.23
  },
  
  // Energy metrics
  "energy": {
    "joules": 0.213,
    "average_power_w": 45.2,
    "peak_power_w": 52.1,
    "efficiency_score": 0.87
  },
  
  // Hardware utilization snapshot
  "hardware": {
    "npu_percent": 89.2,
    "gpu_sm_occupancy": 0.75,
    "cpu_vectorization": 0.67,
    "memory_bandwidth_gbps": 245
  },
  
  // Batching context
  "batch_context": {
    "batch_size": 16,
    "effective_batch_size": 14.2,
    "padding_ratio": 0.11,
    "position_in_batch": 3
  }
}
```

## Monitoring Dashboard Enhancements

### New Visualization Requirements

1. **Energy Efficiency Panel**
   - Real-time power consumption graph
   - Inferences/Watt gauge
   - Thermal status indicator
   - Carbon footprint counter

2. **Latency Distribution Panel**
   - Histogram with percentile markers
   - Tail latency trend line
   - Jitter visualization
   - SLA violation alerts

3. **Hardware Utilization Matrix**
   - SM occupancy heatmap
   - Tensor Core usage timeline
   - Memory hierarchy waterfall
   - Cache hit rate gauges

4. **Batching Analytics Panel**
   - Batch size distribution pie chart
   - Padding overhead indicator
   - Queue depth timeline
   - Optimal batch size recommendation

## Alert Thresholds

### Critical Alerts
- P99 latency > 2x baseline
- Power consumption > thermal limit
- SM occupancy < 50% sustained
- Memory bandwidth > 90%
- Queue depth > 100

### Warning Alerts
- P95 latency > 1.5x baseline
- Energy efficiency < 70% of peak
- Cache hit rate < 80%
- Batch padding > 25%
- Temperature > 80°C

## Validation & Testing

### Metric Accuracy Requirements
- Latency: ±0.1ms accuracy
- Power: ±2% accuracy
- Utilization: ±1% accuracy
- Throughput: ±0.5% accuracy

### Benchmark Alignment
- Cross-validate with NVIDIA Nsight
- Compare with Intel VTune
- Verify against native ONNX Runtime profiler
- Validate energy with external power meter

## Conclusion

This enhanced metrics specification positions our ONNX inference profiling system as industry-leading, with particular strengths in:

1. **Energy Efficiency** - Most comprehensive power/thermal profiling
2. **Tail Latency Analysis** - Production-grade percentile tracking
3. **Hardware Utilization** - Deep insights across NPU/GPU/CPU
4. **Dynamic Batching** - Advanced queue and padding analytics

Implementation of these metrics will provide users with unparalleled visibility into inference performance, enabling optimization opportunities not possible with current tools.

## References

1. MLPerf Inference v4.0 Rules and Metrics
2. NVIDIA Nsight Systems User Guide 2024.5
3. Intel VTune Profiler AI Workload Analysis
4. "Sustainable AI: Environmental Implications" - Nature 2024
5. "Tail Latency Optimization for ML Serving" - OSDI 2024
6. AWS SageMaker Model Monitor Documentation
7. Azure Machine Learning Inference Profiling
8. Google Cloud AI Platform Monitoring Best Practices
9. "Energy-Efficient Deep Learning Inference" - IEEE 2024
10. Qualcomm Neural Processing SDK Profiling Guide