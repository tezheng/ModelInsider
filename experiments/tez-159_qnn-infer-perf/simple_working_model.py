#!/usr/bin/env python3
"""
Simple Working Model - Create the simplest possible model that can demonstrate NPU usage
Focus on getting REAL NPU inference working rather than complex architectures
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

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./simple_working_model_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_minimal_working_model():
    """Create the simplest possible ONNX model that might work"""
    logger.info("Creating minimal working model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto, ValueInfoProto, numpy_helper
        
        # Ultra-simple model: Input[1,4] -> MatMul -> Add -> ReLU -> Output[1,2]
        # This should be supported by almost any neural network accelerator
        
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
        
        # Create simple weights
        weight = np.array([[0.5, -0.3], [0.2, 0.8], [-0.1, 0.4], [0.3, -0.2]], dtype=np.float32)
        bias = np.array([0.1, -0.05], dtype=np.float32)
        
        weight_init = numpy_helper.from_array(weight, name='weight')
        bias_init = numpy_helper.from_array(bias, name='bias')
        
        # Create nodes
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['matmul_out'])
        add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['add_out'])
        relu_node = helper.make_node('Relu', ['add_out'], ['output'])
        
        # Create graph
        graph_def = helper.make_graph(
            nodes=[matmul_node, add_node, relu_node],
            name='MinimalModel',
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[weight_init, bias_init]
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='QNN-Minimal-Test')
        model_def.opset_import[0].version = 11
        
        # Save model
        onnx_path = OUTPUT_DIR / "minimal_model.onnx"
        onnx.save(model_def, str(onnx_path))
        
        # Verify model
        onnx.checker.check_model(model_def)
        
        logger.info(f"✓ Created minimal ONNX model: {onnx_path}")
        logger.info(f"  Model size: {onnx_path.stat().st_size} bytes")
        logger.info(f"  Input shape: [1, 4]")
        logger.info(f"  Output shape: [1, 2]")
        logger.info(f"  Operations: MatMul -> Add -> ReLU")
        
        return onnx_path
        
    except ImportError as e:
        logger.error(f"ONNX not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create minimal model: {e}")
        return None


def try_python_converter():
    """Try using Python-based conversion tools"""
    logger.info("Attempting Python-based conversion...")
    
    # Check if we can access QNN Python tools
    qnn_python_path = QNN_SDK_ROOT / "lib" / "python"
    
    if qnn_python_path.exists():
        logger.info(f"Found QNN Python path: {qnn_python_path}")
        
        # Add QNN Python to sys.path
        sys.path.insert(0, str(qnn_python_path))
        
        try:
            # Try to import QNN Python modules
            import qti.aisw.converters.onnx.onnx_to_dlc as onnx_converter
            logger.info("✓ Successfully imported QNN Python converter")
            return True
        except ImportError as e:
            logger.warning(f"Could not import QNN Python converter: {e}")
            
    return False


def create_simple_input():
    """Create simple test input"""
    logger.info("Creating simple test input...")
    
    # Simple input: [1, 4] 
    input_data = np.array([[1.0, -0.5, 0.8, -0.2]], dtype=np.float32)
    
    # Save as raw binary
    input_file = OUTPUT_DIR / "simple_input.raw"
    with open(input_file, 'wb') as f:
        for val in input_data.flatten():
            f.write(struct.pack('f', val))
    
    # Create input list
    input_list = OUTPUT_DIR / "input_list.txt"
    with open(input_list, 'w') as f:
        f.write(f"input:0 {input_file}\n")
    
    logger.info(f"✓ Created simple input: {input_file}")
    logger.info(f"  Data: {input_data}")
    logger.info(f"  Input list: {input_list}")
    
    return input_list, input_data


def test_with_existing_dlc():
    """Try to find and test with any existing DLC files in QNN SDK"""
    logger.info("Looking for existing DLC models in QNN SDK...")
    
    # Common locations for sample models
    sample_dirs = [
        QNN_SDK_ROOT / "examples" / "Models",
        QNN_SDK_ROOT / "share" / "QNN" / "samples",
        QNN_SDK_ROOT / "models",
        QNN_SDK_ROOT / "benchmarks" / "QNN" / "models"
    ]
    
    dlc_files = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            logger.info(f"Checking: {sample_dir}")
            for dlc_file in sample_dir.rglob("*.dlc"):
                dlc_files.append(dlc_file)
                logger.info(f"  Found DLC: {dlc_file}")
    
    if dlc_files:
        # Use the first DLC file found
        test_dlc = dlc_files[0]
        logger.info(f"Testing with existing DLC: {test_dlc}")
        
        # Create compatible input (we'll guess the input shape)
        input_shapes_to_try = [
            ([1, 3, 224, 224], "imagenet_input.raw"),  # ImageNet format
            ([1, 224, 224, 3], "imagenet_nhwc_input.raw"),  # NHWC format
            ([1, 1000], "feature_input.raw"),  # Feature vector
            ([1, 4], "simple_input.raw")  # Simple input
        ]
        
        for shape, filename in input_shapes_to_try:
            logger.info(f"Trying input shape: {shape}")
            
            # Create random input of this shape
            input_data = np.random.randn(*shape).astype(np.float32) * 0.1
            
            input_file = OUTPUT_DIR / filename
            with open(input_file, 'wb') as f:
                for val in input_data.flatten():
                    f.write(struct.pack('f', val))
            
            input_list = OUTPUT_DIR / f"{filename}_list.txt"
            with open(input_list, 'w') as f:
                f.write(f"input:0 {input_file}\n")
            
            # Try running inference
            success = run_inference_test(test_dlc, input_list)
            if success:
                logger.info(f"✅ SUCCESS with input shape: {shape}")
                return True
            else:
                logger.info(f"Failed with input shape: {shape}")
        
        return False
    else:
        logger.info("No existing DLC files found")
        return False


def run_inference_test(dlc_path, input_list):
    """Run inference test with given DLC and input"""
    logger.info(f"Testing inference with: {dlc_path}")
    
    net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
    htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    cpu_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
    
    # Try NPU first
    htp_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(htp_backend),
        "--input_list", str(input_list),
        "--output_dir", str(OUTPUT_DIR),
        "--perf_profile", "extreme_performance"
    ]
    
    logger.info("🚀 Testing NPU inference...")
    try:
        htp_start = time.perf_counter()
        htp_result = subprocess.run(htp_cmd, capture_output=True, text=True, timeout=30)
        htp_time = (time.perf_counter() - htp_start) * 1000
        
        if htp_result.returncode == 0:
            logger.info(f"✅ NPU inference successful: {htp_time:.2f}ms")
            
            # Try CPU for comparison
            cpu_cmd = [
                str(net_run),
                "--model", str(dlc_path),
                "--backend", str(cpu_backend),
                "--input_list", str(input_list),
                "--output_dir", str(OUTPUT_DIR)
            ]
            
            logger.info("🖥️ Testing CPU inference...")
            try:
                cpu_start = time.perf_counter()
                cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=30)
                cpu_time = (time.perf_counter() - cpu_start) * 1000
                
                if cpu_result.returncode == 0:
                    logger.info(f"✅ CPU inference successful: {cpu_time:.2f}ms")
                    
                    speedup = cpu_time / htp_time if htp_time > 0 else 1.0
                    logger.info(f"🚀 NPU vs CPU: {speedup:.2f}x speedup")
                    
                    if speedup > 1.1:
                        logger.info("✅ REAL NPU ACCELERATION CONFIRMED!")
                        return True
                    else:
                        logger.info("⚠️ NPU performance similar to CPU")
                        return True  # Still counts as working
                else:
                    logger.info("CPU inference failed, but NPU worked - still success")
                    return True
                    
            except Exception as e:
                logger.info(f"CPU test failed: {e}, but NPU worked - still success")
                return True
                
        else:
            logger.info(f"NPU inference failed: {htp_result.stderr}")
            return False
            
    except Exception as e:
        logger.info(f"NPU test failed: {e}")
        return False


def main_workflow():
    """Main workflow to get real NPU usage working"""
    logger.info("="*80)
    logger.info("SIMPLE WORKING MODEL - REAL NPU USAGE ATTEMPT")
    logger.info("="*80)
    
    # Strategy 1: Try with existing DLC models
    logger.info("Strategy 1: Testing with existing DLC models...")
    if test_with_existing_dlc():
        logger.info("🎉 SUCCESS: Found working configuration with existing model!")
        return True
    
    # Strategy 2: Try Python converter
    logger.info("\nStrategy 2: Attempting Python-based conversion...")
    if try_python_converter():
        onnx_path = create_minimal_working_model()
        if onnx_path:
            # TODO: Implement Python-based conversion here
            logger.info("Python converter available but not implemented yet")
    
    # Strategy 3: Try alternative tools
    logger.info("\nStrategy 3: Looking for alternative conversion tools...")
    
    # Check for SNPE tools (older but might work)
    snpe_converter = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "snpe-onnx-to-dlc"
    if snpe_converter.exists():
        logger.info(f"Found SNPE converter: {snpe_converter}")
        # TODO: Try SNPE conversion
    
    logger.info("❌ All strategies failed")
    logger.info("Need working QNN model conversion tools for real NPU usage")
    return False


if __name__ == "__main__":
    success = main_workflow()
    
    if success:
        logger.info("🎉 SUCCESS: Real NPU usage demonstrated!")
    else:
        logger.info("❌ FAILED: Could not achieve real NPU usage")
        logger.info("The main blocker is QNN model conversion tools compatibility")