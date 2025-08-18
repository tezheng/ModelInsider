#!/usr/bin/env python3
"""
Concrete QNN Profiling Example for Windows ARM64 with Qualcomm NPU
This script will actually profile model inference on your device.

Prerequisites:
1. Install QNN SDK for Windows ARM64
2. Set QNN_SDK_ROOT environment variable
3. Have a model ready (ONNX or pre-compiled DLC)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Windows-specific QNN SDK paths
QNN_SDK_PATHS = [
    r"C:\Qualcomm\AIStack\qairt\2.34.0.250424",
    r"C:\Qualcomm\AIStack\QAIRT\2.34.0.250424",
    r"C:\Program Files\Qualcomm\AIStack\qairt",
]

def setup_qnn_sdk() -> Optional[Path]:
    """Setup QNN SDK for Windows ARM64"""
    # Check environment variable first
    qnn_root = os.environ.get("QNN_SDK_ROOT")
    if qnn_root and Path(qnn_root).exists():
        sdk_path = Path(qnn_root)
        print(f"Found QNN SDK via environment: {sdk_path}")
    else:
        # Search common Windows locations
        sdk_path = None
        for path in QNN_SDK_PATHS:
            if Path(path).exists():
                sdk_path = Path(path)
                print(f"Found QNN SDK at: {sdk_path}")
                break
    
    if not sdk_path:
        print("ERROR: QNN SDK not found. Please install and set QNN_SDK_ROOT")
        return None
    
    # Add Python packages to path
    python_path = sdk_path / "lib" / "python"
    if python_path.exists():
        sys.path.insert(0, str(python_path))
        print(f"Added to Python path: {python_path}")
    
    return sdk_path


def prepare_sample_model(sdk_path: Path) -> Optional[Path]:
    """
    Prepare a sample model for testing.
    We'll use MobileNet from QNN SDK examples or convert a simple ONNX model.
    """
    # Check for sample models in SDK
    sample_models = sdk_path / "models"
    if sample_models.exists():
        # Look for MobileNet or other sample models
        dlc_files = list(sample_models.glob("*.dlc"))
        if dlc_files:
            print(f"Found sample model: {dlc_files[0]}")
            return dlc_files[0]
    
    # Alternative: Convert a simple ONNX model
    print("Creating a simple test model...")
    try:
        import onnx
        import torch
        import torch.nn as nn
        
        # Create a tiny model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x
        
        # Export to ONNX
        model = TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)
        onnx_path = Path("tiny_model.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )
        
        print(f"Created test model: {onnx_path}")
        return onnx_path
        
    except ImportError:
        print("Note: Install torch and onnx to create test models")
        return None


def convert_onnx_to_dlc(sdk_path: Path, onnx_path: Path) -> Optional[Path]:
    """Convert ONNX model to DLC format for QNN"""
    converter = sdk_path / "bin" / "aarch64-windows-msvc" / "qnn-onnx-converter.exe"
    if not converter.exists():
        converter = sdk_path / "bin" / "x86_64-windows-msvc" / "qnn-onnx-converter.exe"
    
    if not converter.exists():
        print(f"Converter not found: {converter}")
        return None
    
    dlc_path = onnx_path.with_suffix(".dlc")
    
    cmd = f'"{converter}" --input_network "{onnx_path}" --output_path "{dlc_path}"'
    print(f"Converting: {cmd}")
    
    result = os.system(cmd)
    if result == 0 and dlc_path.exists():
        print(f"Converted to DLC: {dlc_path}")
        return dlc_path
    else:
        print("Conversion failed")
        return None


def run_profiled_inference_native(sdk_path: Path, model_path: Path) -> Dict[str, Any]:
    """
    Run inference with profiling using QNN native tools.
    This uses qnn-net-run.exe with profiling enabled.
    """
    # Prepare paths
    if sdk_path:
        qnn_net_run = sdk_path / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
        if not qnn_net_run.exists():
            qnn_net_run = sdk_path / "bin" / "x86_64-windows-msvc" / "qnn-net-run.exe"
        
        htp_backend = sdk_path / "lib" / "aarch64-windows-msvc" / "libQnnHtp.dll"
        if not htp_backend.exists():
            htp_backend = sdk_path / "lib" / "x86_64-windows-msvc" / "QnnHtp.dll"
    else:
        print("SDK path not available")
        return {}
    
    if not qnn_net_run.exists():
        print(f"qnn-net-run not found: {qnn_net_run}")
        return {}
    
    if not htp_backend.exists():
        print(f"HTP backend not found: {htp_backend}")
        # Try CPU backend as fallback
        cpu_backend = sdk_path / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
        if cpu_backend.exists():
            print(f"Using CPU backend instead: {cpu_backend}")
            htp_backend = cpu_backend
        else:
            return {}
    
    # Create output directory
    output_dir = Path("profiling_output_real")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare input data
    input_list = output_dir / "input_list.txt"
    input_data = output_dir / "input.raw"
    
    # Create dummy input (32x32x3 image)
    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    dummy_input.tofile(str(input_data))
    
    with open(input_list, 'w') as f:
        f.write(f"{input_data}\n")
    
    # Run with profiling
    cmd = f'''"{qnn_net_run}" ^
        --model "{model_path}" ^
        --backend "{htp_backend}" ^
        --input_list "{input_list}" ^
        --output_dir "{output_dir}" ^
        --profiling_level detailed ^
        --profiling_option optrace ^
        --perf_profile high_performance'''
    
    # Clean up Windows command formatting
    cmd = cmd.replace("^\n", "")
    cmd = " ".join(cmd.split())
    
    print(f"Running profiled inference:")
    print(cmd)
    
    start_time = time.time()
    result = os.system(cmd)
    inference_time = time.time() - start_time
    
    if result == 0:
        print(f"Inference completed in {inference_time:.3f} seconds")
        
        # Parse profiling results
        profiling_log = output_dir / "qnn-profiling-data_0.log"
        if profiling_log.exists():
            return parse_profiling_log(sdk_path, profiling_log, output_dir)
        else:
            print("No profiling log generated")
            return {"inference_time": inference_time}
    else:
        print(f"Inference failed with code {result}")
        return {}


def run_profiled_inference_python(model_path: Path) -> Dict[str, Any]:
    """
    Run inference with profiling using Python SDK.
    This is the preferred method if Python SDK is available.
    """
    try:
        # Import QNN Python modules
        from qairt import qnn
        from qairt.api import Model, Runtime, ProfilerContext
        from qairt.constants import PerfProfile, ProfilingLevel
        
        print("Using QNN Python SDK for profiling")
        
        # Create output directory
        output_dir = Path("profiling_output_python")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize profiler context
        profiler_context = ProfilerContext(
            level=ProfilingLevel.DETAILED,
            option="optrace"
        )
        
        # Load model
        model = Model(str(model_path))
        
        # Create runtime with HTP backend
        runtime = Runtime(
            model=model,
            backend="htp",
            perf_profile=PerfProfile.HIGH_PERFORMANCE,
            profiler_context=profiler_context
        )
        
        # Prepare input
        input_shape = model.get_input_shapes()[0]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference with profiling
        start_time = time.time()
        outputs = runtime.execute({"input": dummy_input})
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {inference_time:.3f} seconds")
        
        # Get profiling report
        profiling_report = runtime.get_profiling_report()
        
        # Save Chrome trace
        if profiling_report and hasattr(profiling_report, 'chrome_trace'):
            chrome_trace_path = output_dir / "chrome_trace.json"
            with open(chrome_trace_path, 'w') as f:
                json.dump(profiling_report.chrome_trace, f, indent=2)
            print(f"Chrome trace saved to: {chrome_trace_path}")
        
        # Extract metrics
        metrics = {
            "inference_time_ms": inference_time * 1000,
            "backend": "htp",
            "profiling_level": "detailed"
        }
        
        if profiling_report:
            metrics.update(extract_metrics_from_report(profiling_report))
        
        return metrics
        
    except ImportError as e:
        print(f"Python SDK not available: {e}")
        print("Falling back to native tool execution")
        return {}


def parse_profiling_log(sdk_path: Path, log_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Parse QNN profiling log using qnn-profile-viewer"""
    viewer = sdk_path / "bin" / "aarch64-windows-msvc" / "qnn-profile-viewer.exe"
    if not viewer.exists():
        viewer = sdk_path / "bin" / "x86_64-windows-msvc" / "qnn-profile-viewer.exe"
    
    if not viewer.exists():
        print(f"Profile viewer not found: {viewer}")
        return {}
    
    # Convert to CSV
    csv_output = output_dir / "profiling.csv"
    cmd = f'"{viewer}" --input_log "{log_path}" --output "{csv_output}"'
    
    print(f"Parsing profiling log: {cmd}")
    result = os.system(cmd)
    
    metrics = {}
    if result == 0 and csv_output.exists():
        # Parse CSV for key metrics
        with open(csv_output, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "ACCEL_TIME_MICROSEC" in line:
                    parts = line.split(',')
                    if len(parts) > 2:
                        try:
                            metrics["htp_execution_time_us"] = float(parts[2])
                        except:
                            pass
                elif "VTCM_ACQUIRE_TIME" in line:
                    parts = line.split(',')
                    if len(parts) > 2:
                        try:
                            metrics["vtcm_acquisition_time_us"] = float(parts[2])
                        except:
                            pass
    
    # Try to generate Chrome trace with OpTrace reader
    optrace_reader = sdk_path / "lib" / "aarch64-windows-msvc" / "QnnHtpOptraceProfilingReader.dll"
    if not optrace_reader.exists():
        optrace_reader = sdk_path / "lib" / "x86_64-windows-msvc" / "QnnHtpOptraceProfilingReader.dll"
    
    if optrace_reader.exists():
        chrome_trace = output_dir / "chrome_trace.json"
        cmd = f'"{viewer}" --input_log "{log_path}" --output "{chrome_trace}" --reader "{optrace_reader}"'
        result = os.system(cmd)
        if result == 0:
            print(f"Chrome trace saved to: {chrome_trace}")
            metrics["chrome_trace"] = str(chrome_trace)
    
    return metrics


def extract_metrics_from_report(report: Any) -> Dict[str, Any]:
    """Extract metrics from Python SDK profiling report"""
    metrics = {}
    
    if hasattr(report, 'execution_time_ms'):
        metrics["execution_time_ms"] = report.execution_time_ms
    
    if hasattr(report, 'hardware_metrics'):
        hw = report.hardware_metrics
        if hasattr(hw, 'hvx_utilization'):
            metrics["hvx_utilization_percent"] = hw.hvx_utilization
        if hasattr(hw, 'hmx_utilization'):
            metrics["hmx_utilization_percent"] = hw.hmx_utilization
        if hasattr(hw, 'vtcm_usage_kb'):
            metrics["vtcm_usage_kb"] = hw.vtcm_usage_kb
    
    return metrics


def main():
    """Main execution flow"""
    print("=" * 60)
    print("QNN Profiling on Windows ARM64 with Qualcomm NPU")
    print("=" * 60)
    
    # Setup SDK
    sdk_path = setup_qnn_sdk()
    if not sdk_path:
        print("\nPlease install QNN SDK and set QNN_SDK_ROOT environment variable")
        print("Download from: https://developer.qualcomm.com/software/qualcomm-ai-stack")
        return
    
    # Prepare model
    print("\n--- Model Preparation ---")
    model_path = prepare_sample_model(sdk_path)
    
    if not model_path:
        print("No model available. Please provide an ONNX or DLC model.")
        # Allow user to specify their own model
        user_model = input("Enter path to your model (ONNX/DLC) or press Enter to exit: ").strip()
        if user_model:
            model_path = Path(user_model)
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                return
        else:
            return
    
    # Convert ONNX to DLC if needed
    if model_path.suffix.lower() == ".onnx":
        print("\n--- Converting ONNX to DLC ---")
        dlc_path = convert_onnx_to_dlc(sdk_path, model_path)
        if dlc_path:
            model_path = dlc_path
        else:
            print("Failed to convert model to DLC format")
            return
    
    # Try Python SDK first
    print("\n--- Attempting Python SDK Profiling ---")
    metrics = run_profiled_inference_python(model_path)
    
    # Fallback to native tools if Python SDK fails
    if not metrics:
        print("\n--- Using Native Tool Profiling ---")
        metrics = run_profiled_inference_native(sdk_path, model_path)
    
    # Display results
    if metrics:
        print("\n" + "=" * 60)
        print("PROFILING RESULTS")
        print("=" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        if "chrome_trace" in metrics:
            print("\n📊 View detailed timeline:")
            print("1. Open Chrome browser")
            print("2. Navigate to: chrome://tracing")
            print("3. Click 'Load' and select:", metrics["chrome_trace"])
    else:
        print("\nNo profiling metrics collected. Please check your setup.")
    
    print("\n" + "=" * 60)
    print("Profiling complete!")


if __name__ == "__main__":
    main()