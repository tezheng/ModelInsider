#!/usr/bin/env python3
"""
Real QNN HTP Profiling POC with Mock Computing Graph
This script performs ACTUAL profiling on Qualcomm HTP hardware using the installed QNN SDK.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# QNN SDK Configuration - Using the actual installed path
QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")

# Verify QNN SDK exists
if not QNN_SDK_ROOT.exists():
    logger.error(f"QNN SDK not found at {QNN_SDK_ROOT}")
    logger.error("Please ensure QNN SDK is installed at the correct location")
    sys.exit(1)

# Add QNN Python bindings to path
qnn_python_path = QNN_SDK_ROOT / "lib" / "python"
if qnn_python_path.exists():
    sys.path.insert(0, str(qnn_python_path))
    logger.info(f"Added QNN Python path: {qnn_python_path}")

# Set environment variables for QNN
os.environ["QNN_SDK_ROOT"] = str(QNN_SDK_ROOT)
os.environ["PATH"] = f"{QNN_SDK_ROOT}/bin/aarch64-windows-msvc;{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = f"{qnn_python_path};{os.environ.get('PYTHONPATH', '')}"

logger.info(f"QNN SDK Root: {QNN_SDK_ROOT}")
logger.info(f"QNN binaries: {QNN_SDK_ROOT}/bin/aarch64-windows-msvc")


class RealQNNProfiler:
    """
    Real QNN HTP Profiler using actual Qualcomm NPU hardware
    """
    
    def __init__(self, output_dir: Path = Path("./real_qnn_profiling_output")):
        """Initialize real QNN profiler with hardware access"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # QNN tool paths for Windows ARM64
        # Note: Converters are in arm64x-windows-msvc, runtime tools in aarch64-windows-msvc
        self.qnn_tools = {
            "model_converter": QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter",
            "model_runner": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe",
            "profile_viewer": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-profile-viewer.exe",
            "context_binary_generator": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-generator.exe",
            "snpe_converter": QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "snpe-onnx-to-dlc",
        }
        
        # Verify tools exist
        for tool_name, tool_path in self.qnn_tools.items():
            if not tool_path.exists():
                logger.warning(f"{tool_name} not found at {tool_path}")
            else:
                logger.info(f"Found {tool_name}: {tool_path}")
        
        # HTP backend library
        self.htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "libQnnHtp.dll"
        if not self.htp_backend.exists():
            # Try alternative path
            self.htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
        
        logger.info(f"HTP Backend: {self.htp_backend}")
    
    def create_mock_onnx_model(self) -> Path:
        """
        Create a simple mock ONNX model for profiling
        This creates a small CNN-like computation graph
        """
        try:
            import onnx
            from onnx import helper, TensorProto
            
            logger.info("Creating mock ONNX model for profiling...")
            
            # Input
            input_tensor = helper.make_tensor_value_info(
                'input', TensorProto.FLOAT, [1, 3, 224, 224]
            )
            
            # Output
            output_tensor = helper.make_tensor_value_info(
                'output', TensorProto.FLOAT, [1, 1000]
            )
            
            # Create Conv weight (3x3x3x64)
            conv_weight = np.random.randn(64, 3, 3, 3).astype(np.float32)
            conv_weight_tensor = helper.make_tensor(
                'conv_weight',
                TensorProto.FLOAT,
                [64, 3, 3, 3],
                conv_weight.flatten().tolist()
            )
            
            # Create Conv bias
            conv_bias = np.random.randn(64).astype(np.float32)
            conv_bias_tensor = helper.make_tensor(
                'conv_bias',
                TensorProto.FLOAT,
                [64],
                conv_bias.tolist()
            )
            
            # Create FC weight (simplified)
            fc_weight = np.random.randn(1000, 64 * 55 * 55).astype(np.float32)
            fc_weight_tensor = helper.make_tensor(
                'fc_weight',
                TensorProto.FLOAT,
                [1000, 64 * 55 * 55],
                fc_weight.flatten().tolist()
            )
            
            # Create FC bias
            fc_bias = np.random.randn(1000).astype(np.float32)
            fc_bias_tensor = helper.make_tensor(
                'fc_bias',
                TensorProto.FLOAT,
                [1000],
                fc_bias.tolist()
            )
            
            # Create nodes
            conv_node = helper.make_node(
                'Conv',
                inputs=['input', 'conv_weight', 'conv_bias'],
                outputs=['conv_output'],
                kernel_shape=[3, 3],
                strides=[4, 4],
                pads=[1, 1, 1, 1]
            )
            
            relu_node = helper.make_node(
                'Relu',
                inputs=['conv_output'],
                outputs=['relu_output']
            )
            
            pool_node = helper.make_node(
                'GlobalAveragePool',
                inputs=['relu_output'],
                outputs=['pool_output']
            )
            
            flatten_node = helper.make_node(
                'Flatten',
                inputs=['pool_output'],
                outputs=['flatten_output'],
                axis=1
            )
            
            # For simplicity, using a MatMul instead of full Gemm
            reshape_const = helper.make_tensor(
                'reshape_shape',
                TensorProto.INT64,
                [2],
                [1, -1]
            )
            
            reshape_node = helper.make_node(
                'Reshape',
                inputs=['flatten_output', 'reshape_shape'],
                outputs=['reshaped']
            )
            
            # Simple output projection
            output_node = helper.make_node(
                'Add',
                inputs=['reshaped', 'fc_bias'],
                outputs=['output']
            )
            
            # Create the graph
            graph_def = helper.make_graph(
                [conv_node, relu_node, pool_node, flatten_node, reshape_node, output_node],
                'mock_model',
                [input_tensor],
                [output_tensor],
                [conv_weight_tensor, conv_bias_tensor, fc_bias_tensor, reshape_const]
            )
            
            # Create the model
            model_def = helper.make_model(graph_def, producer_name='qnn_profiling_poc')
            model_def.opset_import[0].version = 13
            
            # Save the model
            model_path = self.output_dir / "mock_model.onnx"
            onnx.save(model_def, str(model_path))
            
            logger.info(f"Mock ONNX model saved to: {model_path}")
            return model_path
            
        except ImportError:
            logger.error("ONNX not installed. Please install: pip install onnx")
            # Return a placeholder path
            return self.output_dir / "mock_model.onnx"
    
    def convert_onnx_to_dlc(self, onnx_path: Path) -> Optional[Path]:
        """
        Convert ONNX model to QNN DLC format using qnn-onnx-converter or snpe-onnx-to-dlc
        """
        dlc_path = self.output_dir / f"{onnx_path.stem}.dlc"
        
        # Try QNN converter first, then SNPE converter as fallback
        converter_tool = self.qnn_tools["model_converter"]
        if not converter_tool.exists():
            converter_tool = self.qnn_tools["snpe_converter"]
            if not converter_tool.exists():
                # Try with Python
                converter_tool = "python"
                converter_script = str(self.qnn_tools["model_converter"])
                converter_cmd = [
                    converter_tool,
                    converter_script,
                    "--input_network", str(onnx_path),
                    "--output_path", str(dlc_path),
                    "--input_dim", "input", "1,3,224,224",
                    "--out_names", "output"
                ]
            else:
                # Use SNPE converter
                converter_cmd = [
                    "python",
                    str(converter_tool),
                    "--input_network", str(onnx_path),
                    "--output", str(dlc_path),
                    "--input_dim", "input", "1,3,224,224"
                ]
        else:
            # Use QNN converter
            converter_cmd = [
                "python",
                str(converter_tool),
                "--input_network", str(onnx_path),
                "--output_path", str(dlc_path),
                "--input_dim", "input", "1,3,224,224",
                "--out_names", "output"
            ]
        
        logger.info(f"Converting ONNX to DLC: {onnx_path} -> {dlc_path}")
        logger.debug(f"Command: {' '.join(converter_cmd)}")
        
        try:
            result = subprocess.run(
                converter_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully converted to DLC: {dlc_path}")
                return dlc_path
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Model conversion timed out")
            return None
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            return None
    
    def generate_context_binary(self, dlc_path: Path) -> Optional[Path]:
        """
        Generate HTP context binary for optimized execution
        """
        context_path = self.output_dir / f"{dlc_path.stem}_htp_context.bin"
        
        # Build context generator command
        context_cmd = [
            str(self.qnn_tools["context_binary_generator"]),
            "--model", str(dlc_path),
            "--backend", str(self.htp_backend),
            "--binary_file", str(context_path),
            "--output_dir", str(self.output_dir)
        ]
        
        logger.info(f"Generating HTP context binary: {context_path}")
        
        try:
            result = subprocess.run(
                context_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Context binary generated: {context_path}")
                return context_path
            else:
                logger.warning(f"Context generation failed: {result.stderr}")
                # Can still run without context binary
                return None
                
        except Exception as e:
            logger.warning(f"Error generating context: {e}")
            return None
    
    def profile_on_htp(
        self,
        model_path: Path,
        profiling_level: str = "detailed",
        perf_profile: str = "high_performance",
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Run actual profiling on HTP hardware using qnn-net-run
        """
        # Prepare input data
        input_list_file = self.output_dir / "input_list.txt"
        input_data_file = self.output_dir / "input_data.raw"
        
        # Generate random input data
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_data.tofile(str(input_data_file))
        
        # Create input list file
        with open(input_list_file, 'w') as f:
            f.write(f"input:0 {input_data_file}\n")
        
        # Profiling output file
        profile_output = self.output_dir / f"profile_{profiling_level}_{perf_profile}.json"
        
        # Build qnn-net-run command with profiling
        run_cmd = [
            str(self.qnn_tools["model_runner"]),
            "--model", str(model_path),
            "--backend", str(self.htp_backend),
            "--input_list", str(input_list_file),
            "--output_dir", str(self.output_dir),
            "--profiling_level", profiling_level,
            "--perf_profile", perf_profile,
            "--profiling_file", str(profile_output),
            "--log_level", "info"
        ]
        
        # Add performance options
        if perf_profile == "high_performance":
            run_cmd.extend(["--use_hvx", "1"])
        elif perf_profile == "extreme_performance":
            run_cmd.extend(["--use_hvx", "1", "--vtcm_mb", "8"])
        
        logger.info(f"Running profiling on HTP with level={profiling_level}, profile={perf_profile}")
        logger.debug(f"Command: {' '.join(run_cmd)}")
        
        metrics = {
            "profiling_level": profiling_level,
            "perf_profile": perf_profile,
            "num_runs": num_runs,
            "inference_times_ms": [],
            "metrics": {}
        }
        
        try:
            # Run multiple times to get average
            for run in range(num_runs):
                start_time = time.perf_counter()
                
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000
                metrics["inference_times_ms"].append(inference_time)
                
                if result.returncode != 0:
                    logger.error(f"Run {run+1} failed: {result.stderr}")
                else:
                    logger.info(f"Run {run+1}/{num_runs}: {inference_time:.2f}ms")
                    
                    # Parse output for metrics
                    if "Inference time" in result.stdout:
                        for line in result.stdout.split('\n'):
                            if "Inference time" in line:
                                # Extract actual inference time
                                try:
                                    time_str = line.split(':')[-1].strip()
                                    if 'ms' in time_str:
                                        actual_time = float(time_str.replace('ms', '').strip())
                                        metrics["metrics"]["measured_inference_ms"] = actual_time
                                except:
                                    pass
            
            # Calculate statistics
            if metrics["inference_times_ms"]:
                metrics["avg_inference_ms"] = np.mean(metrics["inference_times_ms"])
                metrics["min_inference_ms"] = np.min(metrics["inference_times_ms"])
                metrics["max_inference_ms"] = np.max(metrics["inference_times_ms"])
                metrics["std_inference_ms"] = np.std(metrics["inference_times_ms"])
                
                logger.info(f"Average inference time: {metrics['avg_inference_ms']:.2f}ms")
            
            # Try to load profiling data if generated
            if profile_output.exists():
                try:
                    with open(profile_output, 'r') as f:
                        profile_data = json.load(f)
                        metrics["profile_data"] = profile_data
                        logger.info(f"Loaded profiling data from {profile_output}")
                except:
                    logger.warning(f"Could not load profiling data from {profile_output}")
            
        except subprocess.TimeoutExpired:
            logger.error("Profiling timed out")
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
        
        return metrics
    
    def analyze_profiling_results(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and extract key metrics from profiling results
        """
        analysis = {
            "configuration": {
                "profiling_level": metrics.get("profiling_level"),
                "perf_profile": metrics.get("perf_profile"),
                "num_runs": metrics.get("num_runs")
            },
            "performance": {
                "avg_inference_ms": metrics.get("avg_inference_ms", 0),
                "min_inference_ms": metrics.get("min_inference_ms", 0),
                "max_inference_ms": metrics.get("max_inference_ms", 0),
                "std_deviation_ms": metrics.get("std_inference_ms", 0),
                "throughput_fps": 1000.0 / metrics.get("avg_inference_ms", 1) if metrics.get("avg_inference_ms") else 0
            }
        }
        
        # Extract detailed metrics from profile data if available
        if "profile_data" in metrics:
            profile = metrics["profile_data"]
            
            # Try to extract HTP-specific metrics
            if isinstance(profile, dict):
                if "execution_time_us" in profile:
                    analysis["htp_metrics"] = {
                        "execution_time_us": profile["execution_time_us"]
                    }
                
                if "nodes" in profile:
                    # Analyze node-level performance
                    nodes = profile["nodes"]
                    if isinstance(nodes, list):
                        node_times = [n.get("time_us", 0) for n in nodes if "time_us" in n]
                        if node_times:
                            analysis["node_metrics"] = {
                                "total_nodes": len(nodes),
                                "avg_node_time_us": np.mean(node_times),
                                "max_node_time_us": np.max(node_times),
                                "min_node_time_us": np.min(node_times)
                            }
        
        return analysis
    
    def run_comprehensive_profiling(self) -> Dict[str, Any]:
        """
        Run complete profiling workflow with real HTP hardware
        """
        logger.info("="*60)
        logger.info("REAL QNN HTP PROFILING POC")
        logger.info("="*60)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sdk_version": str(QNN_SDK_ROOT),
            "configurations": []
        }
        
        # Step 1: Create mock ONNX model
        logger.info("\nStep 1: Creating mock ONNX model...")
        onnx_model = self.create_mock_onnx_model()
        
        if not onnx_model.exists():
            logger.error("Failed to create ONNX model")
            return results
        
        # Step 2: Convert to DLC
        logger.info("\nStep 2: Converting ONNX to DLC...")
        dlc_model = self.convert_onnx_to_dlc(onnx_model)
        
        if not dlc_model:
            logger.error("Failed to convert to DLC")
            return results
        
        # Step 3: Generate HTP context (optional but recommended)
        logger.info("\nStep 3: Generating HTP context binary...")
        context_binary = self.generate_context_binary(dlc_model)
        
        # Step 4: Profile with different configurations
        logger.info("\nStep 4: Running profiling on HTP hardware...")
        
        # Test configurations
        test_configs = [
            ("basic", "balanced"),
            ("detailed", "high_performance"),
            ("detailed", "extreme_performance")
        ]
        
        for prof_level, perf_prof in test_configs:
            logger.info(f"\nProfiling: {prof_level} / {perf_prof}")
            logger.info("-"*40)
            
            # Use context binary if available, otherwise use DLC
            model_to_run = context_binary if context_binary else dlc_model
            
            metrics = self.profile_on_htp(
                model_path=model_to_run,
                profiling_level=prof_level,
                perf_profile=perf_prof,
                num_runs=5
            )
            
            # Analyze results
            analysis = self.analyze_profiling_results(metrics)
            results["configurations"].append(analysis)
            
            # Print summary
            if analysis["performance"]["avg_inference_ms"] > 0:
                logger.info(f"Results:")
                logger.info(f"  Average: {analysis['performance']['avg_inference_ms']:.2f}ms")
                logger.info(f"  Min: {analysis['performance']['min_inference_ms']:.2f}ms")
                logger.info(f"  Max: {analysis['performance']['max_inference_ms']:.2f}ms")
                logger.info(f"  Throughput: {analysis['performance']['throughput_fps']:.1f} FPS")
        
        # Save complete results
        results_file = self.output_dir / "real_profiling_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return results


def main():
    """Main execution"""
    logger.info("Starting Real QNN HTP Profiling POC")
    
    # Check if we're on Windows ARM64
    import platform
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Initialize profiler
    profiler = RealQNNProfiler()
    
    # Run comprehensive profiling
    results = profiler.run_comprehensive_profiling()
    
    logger.info("\n" + "="*60)
    logger.info("PROFILING COMPLETE")
    logger.info("="*60)
    
    if results["configurations"]:
        logger.info("\nSummary of all configurations:")
        for config in results["configurations"]:
            cfg = config["configuration"]
            perf = config["performance"]
            logger.info(f"\n{cfg['profiling_level']} / {cfg['perf_profile']}:")
            logger.info(f"  Inference: {perf['avg_inference_ms']:.2f}ms")
            logger.info(f"  Throughput: {perf['throughput_fps']:.1f} FPS")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())