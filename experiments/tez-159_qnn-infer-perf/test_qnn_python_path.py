#!/usr/bin/env python3
"""
Test QNN Python Path Setup - Try to get QNN Python tools working
Since the converter is a Python script, we should be able to run it with proper setup
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./qnn_python_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def setup_qnn_python_environment():
    """Set up QNN Python environment properly"""
    logger.info("Setting up QNN Python environment...")
    
    # Add QNN Python paths
    qnn_python_paths = [
        QNN_SDK_ROOT / "lib" / "python",
        QNN_SDK_ROOT / "lib" / "python" / "qti" / "aisw",
        QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc",
    ]
    
    for path in qnn_python_paths:
        if path.exists():
            logger.info(f"Adding to Python path: {path}")
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        else:
            logger.warning(f"Path not found: {path}")
    
    # Set QNN environment variables
    qnn_env_vars = {
        'QNN_SDK_ROOT': str(QNN_SDK_ROOT),
        'PYTHONPATH': os.pathsep.join([str(p) for p in qnn_python_paths if p.exists()]),
        'PATH': str(QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc") + os.pathsep + os.environ.get('PATH', '')
    }
    
    for key, value in qnn_env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value[:100]}...")


def test_qnn_imports():
    """Test if we can import QNN modules"""
    logger.info("Testing QNN Python imports...")
    
    try:
        import qti.aisw.converters.onnx as onnx_frontend
        logger.info("✓ Successfully imported qti.aisw.converters.onnx")
        return True
    except ImportError as e:
        logger.error(f"Failed to import QNN ONNX converter: {e}")
        
    try:
        import qti.aisw.converters.common.utils.converter_utils as converter_utils
        logger.info("✓ Successfully imported converter utils")
        return True
    except ImportError as e:
        logger.error(f"Failed to import converter utils: {e}")
        
    try:
        # Try more basic imports
        import qti
        logger.info("✓ Successfully imported qti base module")
        return True
    except ImportError as e:
        logger.error(f"Failed to import qti base: {e}")
    
    return False


def run_qnn_converter_directly():
    """Try to run the QNN converter as a Python script"""
    logger.info("Attempting to run QNN converter directly...")
    
    converter_script = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter"
    
    # Create a simple test ONNX file first
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        
        # Super simple model: input[1,2] -> output[1,1] 
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
        
        weight = np.array([[0.5], [0.3]], dtype=np.float32)
        weight_init = numpy_helper.from_array(weight, name='weight')
        
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['output'])
        
        graph_def = helper.make_graph(
            nodes=[matmul_node],
            name='SuperSimpleModel', 
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[weight_init]
        )
        
        model_def = helper.make_model(graph_def, producer_name='QNN-Test')
        model_def.opset_import[0].version = 11
        
        test_onnx = OUTPUT_DIR / "super_simple.onnx"
        onnx.save(model_def, str(test_onnx))
        
        logger.info(f"Created test ONNX: {test_onnx}")
        
    except Exception as e:
        logger.error(f"Failed to create test ONNX: {e}")
        return False
    
    # Try running the converter script with Python
    test_dlc = OUTPUT_DIR / "super_simple.dlc"
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(converter_script),
        "--input_network", str(test_onnx),
        "--output_path", str(test_dlc),
        "--input_dim", "input", "1,2"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Set up environment for subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(p) for p in [
            QNN_SDK_ROOT / "lib" / "python",
            QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc"
        ] if p.exists()])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
        logger.info(f"Return code: {result.returncode}")
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.info(f"STDERR:\n{result.stderr}")
            
        if result.returncode == 0 and test_dlc.exists():
            logger.info(f"✅ SUCCESS! Created DLC file: {test_dlc}")
            logger.info(f"DLC size: {test_dlc.stat().st_size} bytes")
            return test_dlc
        else:
            logger.error("Conversion failed")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out")
        return None
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return None


def test_real_npu_with_dlc(dlc_path):
    """Test actual NPU inference with the created DLC"""
    logger.info(f"Testing real NPU inference with: {dlc_path}")
    
    # Create simple input for [1,2] model
    input_data = np.array([[1.0, -0.5]], dtype=np.float32)
    
    input_file = OUTPUT_DIR / "test_input.raw"
    with open(input_file, 'wb') as f:
        for val in input_data.flatten():
            f.write(struct.pack('f', val))
    
    input_list = OUTPUT_DIR / "input_list.txt"
    with open(input_list, 'w') as f:
        f.write(f"input:0 {input_file}\n")
    
    logger.info(f"Created input: {input_data} -> {input_file}")
    
    # Test NPU inference
    net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
    htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    cpu_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
    
    results = {}
    
    # Test HTP/NPU
    htp_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(htp_backend), 
        "--input_list", str(input_list),
        "--output_dir", str(OUTPUT_DIR),
        "--perf_profile", "extreme_performance"
    ]
    
    logger.info("🚀 Testing REAL NPU INFERENCE...")
    htp_start = time.perf_counter()
    try:
        htp_result = subprocess.run(htp_cmd, capture_output=True, text=True, timeout=30)
        htp_time = (time.perf_counter() - htp_start) * 1000
        
        logger.info(f"NPU result: {htp_result.returncode}, time: {htp_time:.2f}ms")
        if htp_result.stdout:
            logger.info(f"NPU stdout: {htp_result.stdout[:500]}")
        if htp_result.stderr:
            logger.info(f"NPU stderr: {htp_result.stderr[:500]}")
            
        results['htp_success'] = htp_result.returncode == 0
        results['htp_time'] = htp_time
        
    except Exception as e:
        logger.error(f"NPU test failed: {e}")
        results['htp_success'] = False
        results['htp_time'] = 0
    
    # Test CPU
    cpu_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(cpu_backend),
        "--input_list", str(input_list), 
        "--output_dir", str(OUTPUT_DIR)
    ]
    
    logger.info("🖥️ Testing CPU inference...")
    cpu_start = time.perf_counter()
    try:
        cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=30)
        cpu_time = (time.perf_counter() - cpu_start) * 1000
        
        logger.info(f"CPU result: {cpu_result.returncode}, time: {cpu_time:.2f}ms")
        results['cpu_success'] = cpu_result.returncode == 0
        results['cpu_time'] = cpu_time
        
    except Exception as e:
        logger.error(f"CPU test failed: {e}")
        results['cpu_success'] = False
        results['cpu_time'] = 0
    
    # Analyze results
    if results['htp_success'] and results['cpu_success']:
        speedup = results['cpu_time'] / results['htp_time'] if results['htp_time'] > 0 else 1.0
        logger.info("=" * 60)
        logger.info("🎉 REAL NPU INFERENCE RESULTS:")
        logger.info(f"NPU: {results['htp_time']:.2f}ms")
        logger.info(f"CPU: {results['cpu_time']:.2f}ms") 
        logger.info(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            logger.info("✅ REAL NPU ACCELERATION CONFIRMED!")
        else:
            logger.info("⚠️ NPU performance similar to CPU")
            
        return True
    else:
        logger.error("❌ Inference tests failed")
        return False


def main():
    """Main workflow to test real NPU usage"""
    logger.info("=" * 80)
    logger.info("QNN PYTHON PATH TEST - REAL NPU USAGE ATTEMPT")
    logger.info("=" * 80)
    
    # Step 1: Set up environment
    setup_qnn_python_environment()
    
    # Step 2: Test imports
    if not test_qnn_imports():
        logger.error("❌ QNN Python imports failed")
        return False
    
    # Step 3: Try conversion
    dlc_path = run_qnn_converter_directly()
    if not dlc_path:
        logger.error("❌ DLC conversion failed")
        return False
    
    # Step 4: Test real NPU inference
    if test_real_npu_with_dlc(dlc_path):
        logger.info("🎉 SUCCESS: Real NPU usage achieved!")
        return True
    else:
        logger.error("❌ NPU inference failed")
        return False


if __name__ == "__main__":
    import struct
    import numpy as np
    
    success = main()
    
    if success:
        logger.info("🎉 BREAKTHROUGH: Real NPU inference working!")
    else:
        logger.info("❌ Still blocked on real NPU usage")