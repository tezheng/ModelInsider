#!/usr/bin/env python3
"""
QNN Profiling POC - Demonstrates integration of QNN profiling capabilities
with modelexport for NPU performance analysis.

This POC shows how to:
1. Enable QNN profiling during model inference
2. Capture hardware metrics (HVX/HMX/VTCM usage)
3. Generate Chrome trace visualization
4. Extract and analyze performance metrics
"""

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProfilingLevel(str, Enum):
    """QNN Profiling levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    LINTING = "linting"  # Native SDK only
    BACKEND = "backend"
    CLIENT = "client"


class PerfProfile(str, Enum):
    """Performance profiles for HTP execution"""
    LOW_BALANCED = "low_balanced"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
    EXTREME_PERFORMANCE = "extreme_performance"


@dataclass
class ProfilingConfig:
    """Configuration for QNN profiling"""
    level: ProfilingLevel = ProfilingLevel.DETAILED
    option: str = "optrace"  # For Chrome trace generation
    perf_profile: PerfProfile = PerfProfile.HIGH_PERFORMANCE
    output_dir: Path = Path("./profiling_output")
    generate_chrome_trace: bool = True
    generate_qhas_summary: bool = True
    qhas_output_type: str = "json"  # "json" or "html"


@dataclass
class ProfilingMetrics:
    """Extracted profiling metrics"""
    total_inference_time_us: float
    htp_execution_time_us: float
    vtcm_acquisition_time_us: float
    resource_power_up_time_us: float
    ddr_bandwidth_mbps: float
    hvx_utilization_percent: float
    hmx_utilization_percent: float
    vtcm_used_kb: int
    node_metrics: List[Dict[str, Any]]


class QNNProfiler:
    """
    QNN Profiling integration for modelexport.
    
    This class demonstrates how to integrate QNN profiling capabilities
    with model inference to capture NPU performance metrics.
    """
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if QNN SDK is available
        self.qnn_sdk_root = self._find_qnn_sdk()
        if not self.qnn_sdk_root:
            logger.warning("QNN SDK not found. Some features may be limited.")
    
    def _find_qnn_sdk(self) -> Optional[Path]:
        """Locate QNN SDK installation"""
        # Check environment variable
        qnn_sdk_env = os.environ.get("QNN_SDK_ROOT")
        if qnn_sdk_env:
            return Path(qnn_sdk_env)
        
        # Check common locations
        common_paths = [
            Path("/opt/qcom/aistack/qairt"),
            Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"),
            Path.home() / "Qualcomm/AIStack/qairt"
        ]
        
        for path in common_paths:
            if path.exists():
                # Find latest version
                versions = sorted([d for d in path.iterdir() if d.is_dir()])
                if versions:
                    return versions[-1]
        
        return None
    
    def profile_inference(
        self,
        model_path: Union[str, Path],
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        backend: str = "htp"
    ) -> Tuple[Any, ProfilingMetrics]:
        """
        Run inference with profiling enabled.
        
        Args:
            model_path: Path to model (ONNX, DLC, or QNN context binary)
            inputs: Input data for inference
            backend: Backend to use (htp, dsp, gpu, cpu)
        
        Returns:
            Tuple of (inference_results, profiling_metrics)
        """
        logger.info(f"Starting profiled inference on {backend} backend")
        logger.info(f"Profiling level: {self.config.level}, Performance profile: {self.config.perf_profile}")
        
        # In a real implementation, this would:
        # 1. Load the model using QNN SDK
        # 2. Create execution context with profiling enabled
        # 3. Run inference
        # 4. Collect profiling data
        
        # For POC, we'll simulate the process
        inference_results = self._simulate_inference(inputs)
        profiling_data = self._simulate_profiling_data()
        
        # Process profiling data
        metrics = self._extract_metrics(profiling_data)
        
        # Generate visualization outputs
        if self.config.generate_chrome_trace:
            self._generate_chrome_trace(profiling_data)
        
        if self.config.generate_qhas_summary:
            self._generate_qhas_summary(profiling_data)
        
        return inference_results, metrics
    
    def _simulate_inference(self, inputs: Any) -> Any:
        """Simulate inference execution"""
        # In real implementation, this would call QNN execution APIs
        logger.info("Simulating model inference...")
        return np.random.randn(1, 1000)  # Dummy output
    
    def _simulate_profiling_data(self) -> Dict[str, Any]:
        """Simulate profiling data collection"""
        # In real implementation, this would parse actual profiling logs
        return {
            "events": [
                {
                    "name": "Conv2d_1",
                    "cat": "node",
                    "ph": "X",
                    "ts": 1000,
                    "dur": 245,
                    "tid": 1,
                    "pid": 1,
                    "args": {
                        "op_type": "Conv2d",
                        "hvx_threads": 4,
                        "vtcm_used_kb": 512,
                        "resource_mask": "0x0011"  # HVX + HMX active
                    }
                },
                {
                    "name": "MatMul_2",
                    "cat": "node",
                    "ph": "X",
                    "ts": 1250,
                    "dur": 180,
                    "tid": 1,
                    "pid": 1,
                    "args": {
                        "op_type": "MatMul",
                        "hvx_threads": 4,
                        "vtcm_used_kb": 256,
                        "resource_mask": "0x0010"  # HMX active
                    }
                }
            ],
            "metadata": {
                "total_inference_time_us": 2500,
                "htp_execution_time_us": 2000,
                "vtcm_acquisition_time_us": 150,
                "resource_power_up_time_us": 100,
                "ddr_bandwidth_mbps": 1200
            }
        }
    
    def _extract_metrics(self, profiling_data: Dict[str, Any]) -> ProfilingMetrics:
        """Extract key metrics from profiling data"""
        metadata = profiling_data.get("metadata", {})
        events = profiling_data.get("events", [])
        
        # Calculate hardware utilization
        total_time = metadata.get("total_inference_time_us", 1)
        hvx_active_time = sum(e["dur"] for e in events if "0x0001" in e.get("args", {}).get("resource_mask", ""))
        hmx_active_time = sum(e["dur"] for e in events if "0x0010" in e.get("args", {}).get("resource_mask", ""))
        
        # Extract node-level metrics
        node_metrics = []
        for event in events:
            if event.get("cat") == "node":
                node_metrics.append({
                    "name": event["name"],
                    "duration_us": event["dur"],
                    "op_type": event.get("args", {}).get("op_type"),
                    "hvx_threads": event.get("args", {}).get("hvx_threads"),
                    "vtcm_used_kb": event.get("args", {}).get("vtcm_used_kb"),
                    "resource_mask": event.get("args", {}).get("resource_mask")
                })
        
        return ProfilingMetrics(
            total_inference_time_us=metadata.get("total_inference_time_us", 0),
            htp_execution_time_us=metadata.get("htp_execution_time_us", 0),
            vtcm_acquisition_time_us=metadata.get("vtcm_acquisition_time_us", 0),
            resource_power_up_time_us=metadata.get("resource_power_up_time_us", 0),
            ddr_bandwidth_mbps=metadata.get("ddr_bandwidth_mbps", 0),
            hvx_utilization_percent=(hvx_active_time / total_time * 100) if total_time > 0 else 0,
            hmx_utilization_percent=(hmx_active_time / total_time * 100) if total_time > 0 else 0,
            vtcm_used_kb=max((e.get("args", {}).get("vtcm_used_kb", 0) for e in events), default=0),
            node_metrics=node_metrics
        )
    
    def _generate_chrome_trace(self, profiling_data: Dict[str, Any]) -> Path:
        """Generate Chrome trace JSON for visualization"""
        output_path = self.config.output_dir / "chrome_trace.json"
        
        # Format data for Chrome tracing
        trace_data = {
            "traceEvents": profiling_data.get("events", []),
            "displayTimeUnit": "ms",
            "metadata": profiling_data.get("metadata", {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        logger.info(f"Chrome trace saved to: {output_path}")
        logger.info("View in Chrome: chrome://tracing -> Load -> Select file")
        return output_path
    
    def _generate_qhas_summary(self, profiling_data: Dict[str, Any]) -> Path:
        """Generate QNN HTP Analysis Summary"""
        output_path = self.config.output_dir / f"qhas_summary.{self.config.qhas_output_type}"
        
        # Extract summary statistics
        metadata = profiling_data.get("metadata", {})
        events = profiling_data.get("events", [])
        
        summary = {
            "execution_summary": {
                "total_inference_time_us": metadata.get("total_inference_time_us", 0),
                "htp_execution_time_us": metadata.get("htp_execution_time_us", 0),
                "overhead_us": metadata.get("total_inference_time_us", 0) - metadata.get("htp_execution_time_us", 0)
            },
            "resource_utilization": {
                "vtcm_acquisition_time_us": metadata.get("vtcm_acquisition_time_us", 0),
                "resource_power_up_time_us": metadata.get("resource_power_up_time_us", 0),
                "ddr_bandwidth_mbps": metadata.get("ddr_bandwidth_mbps", 0)
            },
            "node_statistics": {
                "total_nodes": len(events),
                "average_node_time_us": sum(e["dur"] for e in events) / len(events) if events else 0,
                "longest_node": max(events, key=lambda e: e["dur"])["name"] if events else "N/A"
            }
        }
        
        if self.config.qhas_output_type == "json":
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
        else:  # HTML output
            html_content = self._generate_html_summary(summary)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        logger.info(f"QHAS summary saved to: {output_path}")
        return output_path
    
    def _generate_html_summary(self, summary: Dict[str, Any]) -> str:
        """Generate HTML summary report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QNN HTP Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>QNN HTP Performance Analysis</h1>
            <h2>Execution Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Inference Time</td><td>{summary['execution_summary']['total_inference_time_us']} μs</td></tr>
                <tr><td>HTP Execution Time</td><td>{summary['execution_summary']['htp_execution_time_us']} μs</td></tr>
                <tr><td>Overhead</td><td>{summary['execution_summary']['overhead_us']} μs</td></tr>
            </table>
            <h2>Resource Utilization</h2>
            <table>
                <tr><th>Resource</th><th>Value</th></tr>
                <tr><td>VTCM Acquisition Time</td><td>{summary['resource_utilization']['vtcm_acquisition_time_us']} μs</td></tr>
                <tr><td>Power-up Time</td><td>{summary['resource_utilization']['resource_power_up_time_us']} μs</td></tr>
                <tr><td>DDR Bandwidth</td><td>{summary['resource_utilization']['ddr_bandwidth_mbps']} MB/s</td></tr>
            </table>
        </body>
        </html>
        """
        return html
    
    def analyze_bottlenecks(self, metrics: ProfilingMetrics) -> Dict[str, Any]:
        """Analyze performance bottlenecks from metrics"""
        bottlenecks = []
        
        # Check VTCM acquisition overhead
        if metrics.vtcm_acquisition_time_us > 200:
            bottlenecks.append({
                "type": "VTCM_WAIT",
                "severity": "HIGH",
                "description": f"High VTCM acquisition time: {metrics.vtcm_acquisition_time_us}μs",
                "recommendation": "Consider reducing VTCM usage or optimizing allocation"
            })
        
        # Check hardware utilization
        if metrics.hvx_utilization_percent < 50:
            bottlenecks.append({
                "type": "LOW_HVX_UTIL",
                "severity": "MEDIUM",
                "description": f"Low HVX utilization: {metrics.hvx_utilization_percent:.1f}%",
                "recommendation": "Consider vectorizing more operations"
            })
        
        if metrics.hmx_utilization_percent < 30:
            bottlenecks.append({
                "type": "LOW_HMX_UTIL",
                "severity": "MEDIUM",
                "description": f"Low HMX utilization: {metrics.hmx_utilization_percent:.1f}%",
                "recommendation": "Consider using more matrix operations"
            })
        
        # Check for slow nodes
        slow_nodes = [n for n in metrics.node_metrics if n["duration_us"] > 500]
        if slow_nodes:
            bottlenecks.append({
                "type": "SLOW_NODES",
                "severity": "HIGH",
                "description": f"Found {len(slow_nodes)} slow nodes (>500μs)",
                "nodes": [n["name"] for n in slow_nodes],
                "recommendation": "Optimize or replace slow operations"
            })
        
        return {
            "bottlenecks": bottlenecks,
            "performance_score": 100 - len(bottlenecks) * 20,  # Simple scoring
            "optimization_potential": "HIGH" if len(bottlenecks) > 2 else "MEDIUM" if bottlenecks else "LOW"
        }


def main():
    """Main function demonstrating QNN profiling POC"""
    logger.info("QNN Profiling POC - Starting")
    
    # Configure profiling
    config = ProfilingConfig(
        level=ProfilingLevel.DETAILED,
        perf_profile=PerfProfile.HIGH_PERFORMANCE,
        output_dir=Path("./profiling_output"),
        generate_chrome_trace=True,
        generate_qhas_summary=True,
        qhas_output_type="json"
    )
    
    # Initialize profiler
    profiler = QNNProfiler(config)
    
    # Simulate model and input
    model_path = Path("model.onnx")  # Would be actual model in real use
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run profiled inference
    results, metrics = profiler.profile_inference(model_path, dummy_input, backend="htp")
    
    # Display metrics
    logger.info("\n=== Profiling Metrics ===")
    logger.info(f"Total Inference Time: {metrics.total_inference_time_us:.2f} μs")
    logger.info(f"HTP Execution Time: {metrics.htp_execution_time_us:.2f} μs")
    logger.info(f"VTCM Acquisition: {metrics.vtcm_acquisition_time_us:.2f} μs")
    logger.info(f"Resource Power-up: {metrics.resource_power_up_time_us:.2f} μs")
    logger.info(f"DDR Bandwidth: {metrics.ddr_bandwidth_mbps:.1f} MB/s")
    logger.info(f"HVX Utilization: {metrics.hvx_utilization_percent:.1f}%")
    logger.info(f"HMX Utilization: {metrics.hmx_utilization_percent:.1f}%")
    logger.info(f"Max VTCM Used: {metrics.vtcm_used_kb} KB")
    
    # Analyze bottlenecks
    analysis = profiler.analyze_bottlenecks(metrics)
    logger.info("\n=== Bottleneck Analysis ===")
    logger.info(f"Performance Score: {analysis['performance_score']}/100")
    logger.info(f"Optimization Potential: {analysis['optimization_potential']}")
    
    for bottleneck in analysis["bottlenecks"]:
        logger.warning(f"[{bottleneck['severity']}] {bottleneck['type']}: {bottleneck['description']}")
        logger.info(f"  Recommendation: {bottleneck['recommendation']}")
    
    logger.info(f"\nProfiling outputs saved to: {config.output_dir}")
    logger.info("POC completed successfully!")


if __name__ == "__main__":
    main()