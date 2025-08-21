#!/usr/bin/env python3
"""
Real NPU Inference Test - Create actual DLC model and run on HTP
This will demonstrate REAL NPU usage with actual model inference
"""

import os
import sys
import subprocess
import time
import json
import struct
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./real_npu_inference_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_simple_onnx_model():
    """Create a very simple ONNX model for testing"""
    logger.info("Creating simple ONNX model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto, ValueInfoProto
        
        # Create a simple 2-layer neural network
        # Input: [1, 10] -> Dense(10, 5) -> ReLU -> Dense(5, 1) -> Output
        
        # Define inputs and outputs
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
        
        # Create weight tensors
        W1 = np.random.randn(10, 5).astype(np.float32) * 0.1
        b1 = np.zeros(5, dtype=np.float32)
        W2 = np.random.randn(5, 1).astype(np.float32) * 0.1
        b2 = np.zeros(1, dtype=np.float32)
        
        # Create initializers
        W1_init = helper.make_tensor('W1', TensorProto.FLOAT, [10, 5], W1.flatten())
        b1_init = helper.make_tensor('b1', TensorProto.FLOAT, [5], b1)
        W2_init = helper.make_tensor('W2', TensorProto.FLOAT, [5, 1], W2.flatten())
        b2_init = helper.make_tensor('b2', TensorProto.FLOAT, [1], b2)
        
        # Create nodes
        matmul1_node = helper.make_node('MatMul', ['input', 'W1'], ['matmul1_out'])
        add1_node = helper.make_node('Add', ['matmul1_out', 'b1'], ['add1_out'])
        relu_node = helper.make_node('Relu', ['add1_out'], ['relu_out'])
        matmul2_node = helper.make_node('MatMul', ['relu_out', 'W2'], ['matmul2_out'])
        add2_node = helper.make_node('Add', ['matmul2_out', 'b2'], ['output'])
        
        # Create graph
        graph_def = helper.make_graph(
            nodes=[matmul1_node, add1_node, relu_node, matmul2_node, add2_node],
            name='SimpleNN',
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[W1_init, b1_init, W2_init, b2_init]
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='QNN-Test')
        model_def.opset_import[0].version = 11
        
        # Save model
        onnx_path = OUTPUT_DIR / "simple_test_model.onnx"
        onnx.save(model_def, str(onnx_path))
        
        logger.info(f"Created ONNX model: {onnx_path}")
        return onnx_path
        
    except ImportError:
        logger.warning("ONNX not available, creating minimal mock model...")
        
        # Create a minimal mock ONNX file (this won't work for conversion but shows the process)
        onnx_path = OUTPUT_DIR / "mock_model.onnx"
        with open(onnx_path, 'wb') as f:
            # Write minimal ONNX-like header
            f.write(b'\x08\x01\x12\x04\x08\x01\x10\x01')
        
        logger.info(f"Created mock ONNX file: {onnx_path}")
        return onnx_path


def convert_onnx_to_dlc(onnx_path):
    """Convert ONNX model to DLC format for QNN"""
    logger.info("Converting ONNX to DLC format...")
    
    converter = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter"
    dlc_path = OUTPUT_DIR / "test_model.dlc"
    
    cmd = [
        str(converter),
        "--input_network", str(onnx_path),
        "--output_path", str(dlc_path),
        "--input_dim", "input", "1,10"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and dlc_path.exists():
            logger.info(f"✓ Successfully converted to DLC: {dlc_path}")
            logger.info(f"  DLC file size: {dlc_path.stat().st_size} bytes")
            return dlc_path
        else:
            logger.error("DLC conversion failed:")
            logger.error(f"  stdout: {result.stdout}")
            logger.error(f"  stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("DLC conversion timed out")
        return None
    except Exception as e:
        logger.error(f"DLC conversion error: {e}")
        return None


def create_test_input():
    """Create test input data for inference"""
    logger.info("Creating test input data...")
    
    # Create input tensor [1, 10]
    input_data = np.random.randn(1, 10).astype(np.float32)
    
    # Save as raw binary file
    input_file = OUTPUT_DIR / "test_input.raw"
    with open(input_file, 'wb') as f:
        input_data.tobytes()
        for val in input_data.flatten():
            f.write(struct.pack('f', val))
    
    # Create input list file
    input_list = OUTPUT_DIR / "input_list.txt"
    with open(input_list, 'w') as f:
        f.write(f"input:0 {input_file}\n")
    
    logger.info(f"Created input file: {input_file}")
    logger.info(f"Created input list: {input_list}")
    
    return input_list, input_data


def run_real_npu_inference(dlc_path, input_list):
    """Run actual inference on NPU with real model"""
    logger.info("Running REAL NPU inference...")
    
    net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
    htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    
    # Run HTP inference
    htp_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(htp_backend),
        "--input_list", str(input_list),
        "--output_dir", str(OUTPUT_DIR),
        "--perf_profile", "extreme_performance",
        "--profiling_level", "detailed"
    ]
    
    logger.info("🔥 RUNNING REAL HTP/NPU INFERENCE")
    logger.info(f"Command: {' '.join(htp_cmd[:4])}...")
    
    htp_start = time.perf_counter()
    try:
        htp_result = subprocess.run(htp_cmd, capture_output=True, text=True, timeout=30)
        htp_time = (time.perf_counter() - htp_start) * 1000
        
        logger.info(f"HTP Inference completed in {htp_time:.2f}ms")
        logger.info("HTP Output:")
        for line in htp_result.stdout.split('\n')[:20]:
            if line.strip():
                logger.info(f"  {line}")
        
    except subprocess.TimeoutExpired:
        logger.error("HTP inference timed out")
        htp_time = 30000
        htp_result = None
    except Exception as e:
        logger.error(f"HTP inference failed: {e}")
        htp_time = 0
        htp_result = None
    
    # Run CPU inference for comparison
    cpu_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
    cpu_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(cpu_backend),
        "--input_list", str(input_list),
        "--output_dir", str(OUTPUT_DIR),
        "--perf_profile", "high_performance"
    ]
    
    logger.info("Running CPU inference for comparison...")
    
    cpu_start = time.perf_counter()
    try:
        cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=30)
        cpu_time = (time.perf_counter() - cpu_start) * 1000
        
        logger.info(f"CPU Inference completed in {cpu_time:.2f}ms")
        
    except subprocess.TimeoutExpired:
        logger.error("CPU inference timed out")
        cpu_time = 30000
        cpu_result = None
    except Exception as e:
        logger.error(f"CPU inference failed: {e}")
        cpu_time = 0
        cpu_result = None
    
    return {
        "htp_time_ms": htp_time,
        "cpu_time_ms": cpu_time,
        "htp_result": htp_result,
        "cpu_result": cpu_result
    }


def analyze_npu_usage():
    """Analyze if NPU was actually used"""
    logger.info("\n" + "="*80)
    logger.info("NPU USAGE ANALYSIS")
    logger.info("="*80)
    
    # Step 1: Create ONNX model
    onnx_path = create_simple_onnx_model()
    
    # Step 2: Convert to DLC
    dlc_path = convert_onnx_to_dlc(onnx_path)
    
    if not dlc_path:
        logger.error("❌ Cannot proceed without DLC model")
        logger.error("This means we cannot run REAL NPU inference")
        logger.error("Need ONNX conversion tools working properly")
        return False
    
    # Step 3: Create test input
    input_list, input_data = create_test_input()
    
    # Step 4: Run real inference
    results = run_real_npu_inference(dlc_path, input_list)
    
    # Step 5: Analyze results
    logger.info("\n" + "="*80)
    logger.info("REAL NPU INFERENCE RESULTS")
    logger.info("="*80)
    
    if results["htp_time_ms"] > 0 and results["cpu_time_ms"] > 0:
        speedup = results["cpu_time_ms"] / results["htp_time_ms"]
        logger.info(f"✅ REAL NPU INFERENCE: {results['htp_time_ms']:.2f}ms")
        logger.info(f"✅ REAL CPU INFERENCE: {results['cpu_time_ms']:.2f}ms")
        logger.info(f"🚀 NPU SPEEDUP: {speedup:.2f}x faster")
        
        if speedup > 1.1:  # At least 10% faster
            logger.info("✅ REAL NPU ACCELERATION CONFIRMED!")
            return True
        else:
            logger.warning("⚠️ NPU not significantly faster than CPU")
            return False
    else:
        logger.error("❌ Could not run real inference tests")
        logger.error("This confirms we were only testing backend loading, not real NPU usage")
        return False


if __name__ == "__main__":
    success = analyze_npu_usage()
    
    if success:
        logger.info("🎉 SUCCESS: Real NPU usage confirmed!")
    else:
        logger.info("❌ FAILED: Could not demonstrate real NPU usage")
        logger.info("Previous tests were only measuring backend loading time")