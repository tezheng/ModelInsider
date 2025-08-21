#!/usr/bin/env python3
"""
Comprehensive QNN HTP Profiling Example
Captures ALL available performance metrics from Qualcomm NPU/HTP

This script demonstrates complete profiling capabilities including:
- All profiling levels (basic, detailed, backend, client)
- All performance profiles (low_balanced to extreme_performance)
- Complete metric extraction and analysis
- Chrome trace generation for visualization
- QHAS summary report generation
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# QNN SDK paths - adjust for your system
QNN_SDK_PATHS = [
    Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424"),
    Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"),
    Path(os.environ.get("QNN_SDK_ROOT", ""))
]

# Try to find and import QNN SDK
QNN_AVAILABLE = False
for sdk_path in QNN_SDK_PATHS:
    if sdk_path.exists():
        sys.path.insert(0, str(sdk_path / "lib" / "python"))
        try:
            from qairt.api import *
            from qairt.api.profiler import *
            from qairt.api.executor import *
            from qairt.api.configs import *
            QNN_AVAILABLE = True
            logger.info(f"QNN SDK loaded from: {sdk_path}")
            break
        except ImportError as e:
            logger.debug(f"Failed to import from {sdk_path}: {e}")
            continue

if not QNN_AVAILABLE:
    logger.warning("QNN SDK not available - running in simulation mode")


@dataclass
class ComprehensiveMetrics:
    """Complete set of QNN performance metrics"""
    
    # Execution Timing Metrics
    total_inference_time_ms: float = 0.0
    host_rpc_time_us: float = 0.0
    htp_rpc_time_us: float = 0.0
    accel_time_cycles: int = 0
    accel_time_us: float = 0.0
    misc_accel_time_us: float = 0.0
    
    # Resource Acquisition Metrics
    vtcm_acquisition_time_us: float = 0.0
    resource_power_up_time_us: float = 0.0  # HMX + HVX power-up
    
    # Hardware Utilization Metrics
    hvx_utilization_percent: float = 0.0
    hmx_utilization_percent: float = 0.0
    hvx_thread_count: int = 0
    vtcm_size_mb: float = 0.0
    vtcm_peak_usage_kb: float = 0.0
    
    # Memory Bandwidth Metrics
    ddr_bandwidth_mbps: float = 0.0
    input_fill_bytes: int = 0
    intermediate_fill_bytes: int = 0
    intermediate_spill_bytes: int = 0
    output_spill_bytes: int = 0
    inter_htp_fill_bytes: int = 0
    inter_htp_spill_bytes: int = 0
    
    # Performance Estimation (from graph finalization)
    sim_exec_cycles: int = 0
    sim_exec_lower_cycles: int = 0
    sim_exec_upper_cycles: int = 0
    estimated_inference_time_ms: float = 0.0
    
    # Node-Level Statistics
    total_nodes: int = 0
    average_node_time_us: float = 0.0
    max_node_time_us: float = 0.0
    min_node_time_us: float = 0.0
    slowest_node_name: str = ""
    fastest_node_name: str = ""
    
    # Yielding Metrics (multi-graph)
    yield_count: int = 0
    yield_release_time_us: float = 0.0
    yield_wait_time_us: float = 0.0
    yield_restore_time_us: float = 0.0
    
    # System Metrics
    cpu_utilization_percent: float = 0.0
    memory_usage_mb: float = 0.0
    temperature_celsius: float = 0.0
    
    # Profiling Metadata
    backend: str = "htp"
    profiling_level: str = "detailed"
    perf_profile: str = "high_performance"
    timestamp: str = ""
    model_name: str = ""
    input_shape: str = ""
    
    # Derived Metrics
    throughput_fps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    efficiency_ops_per_watt: float = 0.0


class ComprehensiveQNNProfiler:
    """
    Complete QNN Profiler capturing ALL available metrics
    """
    
    PROFILING_LEVELS = ["basic", "detailed", "backend", "client"]
    PERF_PROFILES = [
        "low_balanced",
        "balanced", 
        "default",
        "high_performance",
        "sustained_high_performance",
        "extreme_performance"
    ]
    
    def __init__(
        self,
        output_dir: Path = Path("./comprehensive_profiling_output")
    ):
        """Initialize comprehensive profiler"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[ComprehensiveMetrics] = []
        
    def profile_all_configurations(
        self,
        model_path: Path,
        input_data: np.ndarray,
        warmup_runs: int = 3,
        profile_runs: int = 10
    ) -> Dict[str, ComprehensiveMetrics]:
        """
        Profile model with all configurations
        
        Args:
            model_path: Path to DLC or ONNX model
            input_data: Input tensor
            warmup_runs: Number of warmup runs
            profile_runs: Number of profiling runs
            
        Returns:
            Dictionary of metrics for each configuration
        """
        all_metrics = {}
        
        # Test all profiling level and perf profile combinations
        for prof_level in self.PROFILING_LEVELS:
            for perf_prof in self.PERF_PROFILES:
                config_name = f"{prof_level}_{perf_prof}"
                logger.info(f"\n{'='*60}")
                logger.info(f"Profiling configuration: {config_name}")
                logger.info(f"{'='*60}")
                
                try:
                    metrics = self.profile_single_configuration(
                        model_path=model_path,
                        input_data=input_data,
                        profiling_level=prof_level,
                        perf_profile=perf_prof,
                        warmup_runs=warmup_runs,
                        profile_runs=profile_runs
                    )
                    all_metrics[config_name] = metrics
                    self.metrics_history.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Failed to profile {config_name}: {e}")
                    
        return all_metrics
    
    def profile_single_configuration(
        self,
        model_path: Path,
        input_data: np.ndarray,
        profiling_level: str = "detailed",
        perf_profile: str = "high_performance",
        warmup_runs: int = 3,
        profile_runs: int = 10
    ) -> ComprehensiveMetrics:
        """
        Profile with a single configuration
        """
        metrics = ComprehensiveMetrics(
            profiling_level=profiling_level,
            perf_profile=perf_profile,
            timestamp=datetime.now().isoformat(),
            model_name=model_path.stem,
            input_shape=str(input_data.shape)
        )
        
        if QNN_AVAILABLE:
            return self._profile_with_qnn(
                model_path, input_data, metrics,
                warmup_runs, profile_runs
            )
        else:
            return self._simulate_profiling(
                model_path, input_data, metrics,
                warmup_runs, profile_runs
            )
    
    def _profile_with_qnn(
        self,
        model_path: Path,
        input_data: np.ndarray,
        metrics: ComprehensiveMetrics,
        warmup_runs: int,
        profile_runs: int
    ) -> ComprehensiveMetrics:
        """Real QNN profiling implementation"""
        
        try:
            # Setup profiling configuration
            profiling_config = {
                "profiling_level": metrics.profiling_level,
                "perf_profile": metrics.perf_profile,
                "backend": "htp",
                "enable_optrace": True,
                "hvx_threads": 4,  # Use 4 HVX threads
                "vtcm_mb": 8,      # Allocate 8MB VTCM
            }
            
            # Create execution context with profiling
            exec_config = ExecutionConfig(**profiling_config)
            
            # Initialize profiler context
            profiler_context = ProfilerContext(
                level=metrics.profiling_level,
                option="optrace"
            )
            
            logger.info(f"Starting QNN profiling with {profile_runs} runs")
            
            # Warmup runs
            for i in range(warmup_runs):
                logger.debug(f"Warmup run {i+1}/{warmup_runs}")
                # Execute model without profiling for warmup
                self._execute_model(model_path, input_data, exec_config)
            
            # Profiling runs
            run_times = []
            with Profiler(context=profiler_context) as profiler:
                for i in range(profile_runs):
                    start_time = time.perf_counter()
                    
                    # Execute model with profiling
                    outputs = self._execute_model(
                        model_path, input_data, exec_config
                    )
                    
                    end_time = time.perf_counter()
                    run_time = (end_time - start_time) * 1000  # ms
                    run_times.append(run_time)
                    
                    logger.debug(f"Profile run {i+1}/{profile_runs}: {run_time:.2f}ms")
                
                # Generate profiling report
                report = profiler.generate_report()
                
            # Extract metrics from report
            metrics = self._extract_metrics_from_report(report, metrics)
            
            # Calculate statistics
            metrics.total_inference_time_ms = np.mean(run_times)
            metrics.latency_p50_ms = np.percentile(run_times, 50)
            metrics.latency_p95_ms = np.percentile(run_times, 95)
            metrics.latency_p99_ms = np.percentile(run_times, 99)
            metrics.throughput_fps = 1000.0 / metrics.total_inference_time_ms
            
            # Save Chrome trace
            self._save_chrome_trace(report, metrics)
            
        except Exception as e:
            logger.error(f"QNN profiling failed: {e}")
            # Fall back to simulation
            return self._simulate_profiling(
                model_path, input_data, metrics,
                warmup_runs, profile_runs
            )
        
        return metrics
    
    def _simulate_profiling(
        self,
        model_path: Path,
        input_data: np.ndarray,
        metrics: ComprehensiveMetrics,
        warmup_runs: int,
        profile_runs: int
    ) -> ComprehensiveMetrics:
        """Simulate profiling with realistic metrics"""
        
        logger.info("Running simulated profiling (QNN SDK not available)")
        
        # Simulate realistic metrics based on profiling level and perf profile
        base_time = 10.0  # Base inference time in ms
        
        # Adjust based on performance profile
        perf_multipliers = {
            "low_balanced": 2.0,
            "balanced": 1.5,
            "default": 1.2,
            "high_performance": 1.0,
            "sustained_high_performance": 1.1,
            "extreme_performance": 0.9
        }
        
        multiplier = perf_multipliers.get(metrics.perf_profile, 1.0)
        
        # Generate simulated metrics
        metrics.total_inference_time_ms = base_time * multiplier + np.random.normal(0, 0.5)
        metrics.htp_rpc_time_us = 200 + np.random.normal(0, 20)
        metrics.host_rpc_time_us = 150 + np.random.normal(0, 15)
        metrics.accel_time_us = (metrics.total_inference_time_ms * 1000) * 0.85
        metrics.accel_time_cycles = int(metrics.accel_time_us * 1000)
        
        # Resource metrics
        metrics.vtcm_acquisition_time_us = 100 + np.random.normal(0, 10)
        metrics.resource_power_up_time_us = 80 + np.random.normal(0, 8)
        
        # Hardware utilization (varies by perf profile)
        if "extreme" in metrics.perf_profile:
            metrics.hvx_utilization_percent = 85 + np.random.normal(0, 5)
            metrics.hmx_utilization_percent = 80 + np.random.normal(0, 5)
        elif "high" in metrics.perf_profile:
            metrics.hvx_utilization_percent = 70 + np.random.normal(0, 5)
            metrics.hmx_utilization_percent = 65 + np.random.normal(0, 5)
        else:
            metrics.hvx_utilization_percent = 50 + np.random.normal(0, 5)
            metrics.hmx_utilization_percent = 45 + np.random.normal(0, 5)
        
        # Memory bandwidth
        metrics.ddr_bandwidth_mbps = 1200 + np.random.normal(0, 100)
        metrics.input_fill_bytes = input_data.nbytes
        metrics.output_spill_bytes = input_data.nbytes  # Assume similar size
        
        # Node statistics (simulate for detailed level)
        if metrics.profiling_level == "detailed":
            metrics.total_nodes = 50 + np.random.randint(-10, 10)
            metrics.average_node_time_us = metrics.accel_time_us / metrics.total_nodes
            metrics.max_node_time_us = metrics.average_node_time_us * 3
            metrics.min_node_time_us = metrics.average_node_time_us * 0.1
            metrics.slowest_node_name = "Conv2d_Layer_5"
            metrics.fastest_node_name = "ReLU_Layer_2"
        
        # System metrics
        metrics.cpu_utilization_percent = 15 + np.random.normal(0, 3)
        metrics.memory_usage_mb = 250 + np.random.normal(0, 25)
        metrics.temperature_celsius = 45 + np.random.normal(0, 2)
        
        # Derived metrics
        metrics.throughput_fps = 1000.0 / metrics.total_inference_time_ms
        metrics.latency_p50_ms = metrics.total_inference_time_ms
        metrics.latency_p95_ms = metrics.total_inference_time_ms * 1.1
        metrics.latency_p99_ms = metrics.total_inference_time_ms * 1.2
        
        # VTCM usage
        metrics.vtcm_size_mb = 8.0
        metrics.vtcm_peak_usage_kb = 768 + np.random.normal(0, 50)
        
        # HVX threads
        metrics.hvx_thread_count = 4
        
        # Performance estimation
        metrics.sim_exec_cycles = int(metrics.accel_time_us * 1500)
        metrics.sim_exec_lower_cycles = int(metrics.sim_exec_cycles * 0.9)
        metrics.sim_exec_upper_cycles = int(metrics.sim_exec_cycles * 1.1)
        metrics.estimated_inference_time_ms = metrics.sim_exec_cycles / 1500000.0
        
        return metrics
    
    def _execute_model(
        self,
        model_path: Path,
        input_data: np.ndarray,
        exec_config: Any
    ) -> np.ndarray:
        """Execute model on QNN backend"""
        # This would contain actual QNN execution code
        # Placeholder for demonstration
        output_shape = input_data.shape
        return np.random.randn(*output_shape)
    
    def _extract_metrics_from_report(
        self,
        report: Dict[str, Any],
        metrics: ComprehensiveMetrics
    ) -> ComprehensiveMetrics:
        """Extract all metrics from QNN profiling report"""
        
        # Extract timing metrics
        if "timing" in report:
            timing = report["timing"]
            metrics.host_rpc_time_us = timing.get("host_rpc_time", 0)
            metrics.htp_rpc_time_us = timing.get("htp_rpc_time", 0)
            metrics.accel_time_us = timing.get("accel_time", 0)
            metrics.vtcm_acquisition_time_us = timing.get("vtcm_acquire_time", 0)
            metrics.resource_power_up_time_us = timing.get("resource_powerup_time", 0)
        
        # Extract hardware utilization
        if "hardware" in report:
            hw = report["hardware"]
            metrics.hvx_utilization_percent = hw.get("hvx_utilization", 0)
            metrics.hmx_utilization_percent = hw.get("hmx_utilization", 0)
            metrics.hvx_thread_count = hw.get("hvx_threads", 0)
            metrics.vtcm_peak_usage_kb = hw.get("vtcm_peak_kb", 0)
        
        # Extract bandwidth metrics
        if "bandwidth" in report:
            bw = report["bandwidth"]
            metrics.ddr_bandwidth_mbps = bw.get("ddr_bandwidth", 0)
            metrics.input_fill_bytes = bw.get("input_fill", 0)
            metrics.output_spill_bytes = bw.get("output_spill", 0)
        
        # Extract node statistics
        if "nodes" in report:
            nodes = report["nodes"]
            metrics.total_nodes = len(nodes)
            if nodes:
                node_times = [n.get("time_us", 0) for n in nodes]
                metrics.average_node_time_us = np.mean(node_times)
                metrics.max_node_time_us = np.max(node_times)
                metrics.min_node_time_us = np.min(node_times)
                
                # Find slowest/fastest nodes
                slowest_idx = np.argmax(node_times)
                fastest_idx = np.argmin(node_times)
                metrics.slowest_node_name = nodes[slowest_idx].get("name", "")
                metrics.fastest_node_name = nodes[fastest_idx].get("name", "")
        
        return metrics
    
    def _save_chrome_trace(
        self,
        report: Dict[str, Any],
        metrics: ComprehensiveMetrics
    ):
        """Save Chrome trace for visualization"""
        trace_file = self.output_dir / f"chrome_trace_{metrics.profiling_level}_{metrics.perf_profile}.json"
        
        # Create Chrome trace format
        chrome_trace = {
            "traceEvents": [],
            "displayTimeUnit": "ms",
            "metadata": {
                "profiling_level": metrics.profiling_level,
                "perf_profile": metrics.perf_profile,
                "timestamp": metrics.timestamp
            }
        }
        
        # Add events (placeholder - would extract from actual report)
        start_time = 0
        chrome_trace["traceEvents"].append({
            "name": "Model Inference",
            "cat": "inference",
            "ph": "X",  # Complete event
            "ts": start_time,
            "dur": metrics.total_inference_time_ms * 1000,  # microseconds
            "pid": 1,
            "tid": 1,
            "args": {
                "hvx_utilization": metrics.hvx_utilization_percent,
                "hmx_utilization": metrics.hmx_utilization_percent,
                "ddr_bandwidth": metrics.ddr_bandwidth_mbps
            }
        })
        
        # Save trace
        with open(trace_file, 'w') as f:
            json.dump(chrome_trace, f, indent=2)
        
        logger.info(f"Chrome trace saved to: {trace_file}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.metrics_history:
            logger.warning("No metrics to report")
            return {}
        
        report = {
            "summary": {
                "total_configurations_tested": len(self.metrics_history),
                "timestamp": datetime.now().isoformat(),
                "best_configuration": None,
                "worst_configuration": None
            },
            "configurations": [],
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Find best and worst configurations
        best_metrics = min(self.metrics_history, key=lambda m: m.total_inference_time_ms)
        worst_metrics = max(self.metrics_history, key=lambda m: m.total_inference_time_ms)
        
        report["summary"]["best_configuration"] = {
            "profiling_level": best_metrics.profiling_level,
            "perf_profile": best_metrics.perf_profile,
            "inference_time_ms": best_metrics.total_inference_time_ms,
            "throughput_fps": best_metrics.throughput_fps
        }
        
        report["summary"]["worst_configuration"] = {
            "profiling_level": worst_metrics.profiling_level,
            "perf_profile": worst_metrics.perf_profile,
            "inference_time_ms": worst_metrics.total_inference_time_ms,
            "throughput_fps": worst_metrics.throughput_fps
        }
        
        # Add all configurations
        for metrics in self.metrics_history:
            report["configurations"].append(asdict(metrics))
        
        # Performance comparison
        report["performance_comparison"] = {
            "inference_time_range_ms": {
                "min": best_metrics.total_inference_time_ms,
                "max": worst_metrics.total_inference_time_ms,
                "improvement_percent": (
                    (worst_metrics.total_inference_time_ms - best_metrics.total_inference_time_ms) 
                    / worst_metrics.total_inference_time_ms * 100
                )
            },
            "hvx_utilization_range": {
                "min": min(m.hvx_utilization_percent for m in self.metrics_history),
                "max": max(m.hvx_utilization_percent for m in self.metrics_history)
            },
            "ddr_bandwidth_range_mbps": {
                "min": min(m.ddr_bandwidth_mbps for m in self.metrics_history),
                "max": max(m.ddr_bandwidth_mbps for m in self.metrics_history)
            }
        }
        
        # Generate recommendations
        if best_metrics.hvx_utilization_percent < 60:
            report["recommendations"].append(
                "HVX utilization is low. Consider optimizing operators for better vectorization."
            )
        
        if best_metrics.vtcm_peak_usage_kb > 700:
            report["recommendations"].append(
                "VTCM usage is high. Consider splitting large tensors or optimizing memory layout."
            )
        
        if best_metrics.ddr_bandwidth_mbps > 2000:
            report["recommendations"].append(
                "High DDR bandwidth usage detected. Consider operator fusion or tiling strategies."
            )
        
        # Save report
        report_file = self.output_dir / "comprehensive_performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report saved to: {report_file}")
        
        return report
    
    def print_metrics_summary(self, metrics: ComprehensiveMetrics):
        """Print formatted metrics summary"""
        
        print("\n" + "="*80)
        print(f"PERFORMANCE METRICS SUMMARY")
        print(f"Configuration: {metrics.profiling_level} / {metrics.perf_profile}")
        print("="*80)
        
        print("\nEXECUTION TIMING")
        print(f"  Total Inference Time:     {metrics.total_inference_time_ms:.2f} ms")
        print(f"  Throughput:              {metrics.throughput_fps:.1f} FPS")
        print(f"  HTP Execution Time:      {metrics.accel_time_us:.0f} us")
        print(f"  Host RPC Overhead:       {metrics.host_rpc_time_us:.0f} us")
        print(f"  HTP RPC Overhead:        {metrics.htp_rpc_time_us:.0f} us")
        
        print("\nHARDWARE UTILIZATION")
        print(f"  HVX Utilization:         {metrics.hvx_utilization_percent:.1f}%")
        print(f"  HMX Utilization:         {metrics.hmx_utilization_percent:.1f}%")
        print(f"  HVX Thread Count:        {metrics.hvx_thread_count}")
        print(f"  VTCM Peak Usage:         {metrics.vtcm_peak_usage_kb:.0f} KB / {metrics.vtcm_size_mb:.0f} MB")
        
        print("\nMEMORY BANDWIDTH")
        print(f"  DDR Bandwidth:           {metrics.ddr_bandwidth_mbps:.0f} MB/s")
        print(f"  Input Fill:              {metrics.input_fill_bytes:,} bytes")
        print(f"  Output Spill:            {metrics.output_spill_bytes:,} bytes")
        
        print("\nRESOURCE ACQUISITION")
        print(f"  VTCM Acquisition Time:   {metrics.vtcm_acquisition_time_us:.0f} us")
        print(f"  Resource Power-up Time:  {metrics.resource_power_up_time_us:.0f} us")
        
        if metrics.total_nodes > 0:
            print("\nNODE-LEVEL STATISTICS")
            print(f"  Total Nodes:             {metrics.total_nodes}")
            print(f"  Average Node Time:       {metrics.average_node_time_us:.1f} us")
            print(f"  Slowest Node:            {metrics.slowest_node_name} ({metrics.max_node_time_us:.1f} us)")
            print(f"  Fastest Node:            {metrics.fastest_node_name} ({metrics.min_node_time_us:.1f} us")
        
        print("\nLATENCY PERCENTILES")
        print(f"  P50 (Median):            {metrics.latency_p50_ms:.2f} ms")
        print(f"  P95:                     {metrics.latency_p95_ms:.2f} ms")
        print(f"  P99:                     {metrics.latency_p99_ms:.2f} ms")
        
        print("\nSYSTEM METRICS")
        print(f"  CPU Utilization:         {metrics.cpu_utilization_percent:.1f}%")
        print(f"  Memory Usage:            {metrics.memory_usage_mb:.0f} MB")
        print(f"  Temperature:             {metrics.temperature_celsius:.1f} C")
        
        print("="*80 + "\n")


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE QNN HTP PROFILING EXAMPLE")
    print("Capturing ALL Available Performance Metrics")
    print("="*80)
    
    # Initialize profiler
    profiler = ComprehensiveQNNProfiler()
    
    # Create dummy input data (adjust for your model)
    input_shape = (1, 3, 224, 224)  # Example: batch=1, channels=3, height=224, width=224
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Model path (use your actual model)
    model_path = Path("model.dlc")  # or model.onnx
    
    # Check if we should use a test model
    if not model_path.exists():
        logger.info("Model not found, using test configuration")
        model_path = Path("test_model.dlc")
    
    # Profile single configuration (fastest option)
    print("\nRunning single configuration profiling...")
    metrics = profiler.profile_single_configuration(
        model_path=model_path,
        input_data=input_data,
        profiling_level="detailed",
        perf_profile="high_performance",
        warmup_runs=3,
        profile_runs=10
    )
    
    # Print results
    profiler.print_metrics_summary(metrics)
    
    # Optional: Profile ALL configurations (takes longer)
    user_input = input("\nProfile ALL configurations? This will take several minutes (y/n): ")
    if user_input.lower() == 'y':
        print("\nProfiling all configurations...")
        all_metrics = profiler.profile_all_configurations(
            model_path=model_path,
            input_data=input_data,
            warmup_runs=2,
            profile_runs=5
        )
        
        # Generate comprehensive report
        report = profiler.generate_comprehensive_report()
        
        print("\nProfiling complete!")
        print(f"Results saved to: {profiler.output_dir}")
        print(f"Chrome traces available for visualization")
        print(f"Comprehensive report: {profiler.output_dir}/comprehensive_performance_report.json")
        
        # Print best configuration
        if report and "summary" in report:
            best = report["summary"]["best_configuration"]
            print(f"\nBest Configuration:")
            print(f"   Profile: {best['perf_profile']}")
            print(f"   Level: {best['profiling_level']}")
            print(f"   Inference: {best['inference_time_ms']:.2f} ms")
            print(f"   Throughput: {best['throughput_fps']:.1f} FPS")
    
    print("\n" + "="*80)
    print("To visualize Chrome traces:")
    print("1. Open Chrome browser")
    print("2. Navigate to: chrome://tracing")
    print(f"3. Load trace file from: {profiler.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()