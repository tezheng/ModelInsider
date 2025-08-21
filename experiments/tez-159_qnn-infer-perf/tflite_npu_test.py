#!/usr/bin/env python3
"""
TFLite NPU Test - Use TFLite models instead of ONNX for NPU testing
This bypasses the ONNX converter issues and uses pre-built TFLite models
"""

import os
import sys
import subprocess
import time
import struct
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./tflite_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def setup_qnn_environment():
    """Set up QNN environment"""
    logger.info("Setting up QNN environment...")
    
    # Add QNN paths
    qnn_paths = [
        QNN_SDK_ROOT / "lib" / "python",
        QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc"
    ]
    
    for path in qnn_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            
    # Set environment variables
    os.environ['QNN_SDK_ROOT'] = str(QNN_SDK_ROOT)
    os.environ['PATH'] = f"{QNN_SDK_ROOT / 'lib' / 'aarch64-windows-msvc'};{os.environ.get('PATH', '')}"
    
    logger.info("QNN environment configured")


def convert_tflite_to_dlc():
    """Convert the example TFLite model to DLC"""
    logger.info("Converting TFLite model to DLC...")
    
    # Use the example TFLite model we found
    tflite_path = QNN_SDK_ROOT / "examples" / "QNN" / "TFLiteDelegate" / "SkipNodeExample" / "model" / "mix_precision_sample.tflite"
    
    if not tflite_path.exists():
        logger.error(f"TFLite model not found: {tflite_path}")
        return None
        
    logger.info(f"Using TFLite model: {tflite_path}")
    logger.info(f"  Model size: {tflite_path.stat().st_size} bytes")
    
    # Output DLC path
    dlc_path = OUTPUT_DIR / "tflite_model.dlc"
    
    # TFLite converter
    converter = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-tflite-converter"
    
    cmd = [
        sys.executable,
        str(converter),
        "--input_network", str(tflite_path),
        "--output_path", str(dlc_path)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        logger.info(f"Return code: {result.returncode}")
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            if "Error" in result.stderr or result.returncode != 0:
                logger.error(f"STDERR:\n{result.stderr}")
            else:
                logger.info(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0 and dlc_path.exists():
            logger.info(f"✅ SUCCESS: Created DLC from TFLite!")
            logger.info(f"  DLC path: {dlc_path}")
            logger.info(f"  DLC size: {dlc_path.stat().st_size} bytes")
            return dlc_path
        else:
            logger.error("❌ TFLite conversion failed")
            
    except subprocess.TimeoutExpired:
        logger.error("TFLite conversion timed out")
    except Exception as e:
        logger.error(f"TFLite conversion error: {e}")
    
    return None


def create_simple_tflite_model():
    """Create a simple TFLite model using TensorFlow Lite"""
    logger.info("Creating simple TFLite model...")
    
    try:
        # Try importing TensorFlow Lite
        import tensorflow as tf
        
        # Create a simple model: y = 2x + 1
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[1])
        ])
        
        # Set weights manually
        model.layers[0].set_weights([
            np.array([[2.0]]),  # weight
            np.array([1.0])     # bias
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save model
        tflite_path = OUTPUT_DIR / "simple_linear.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"✓ Created TFLite model: {tflite_path}")
        logger.info(f"  Model size: {tflite_path.stat().st_size} bytes")
        return tflite_path
        
    except ImportError:
        logger.warning("TensorFlow not available, skipping custom model creation")
        return None
    except Exception as e:
        logger.error(f"Failed to create TFLite model: {e}")
        return None


def run_npu_inference_with_dlc(dlc_path):
    """Run actual NPU inference using the DLC model"""
    logger.info(f"Running NPU inference with: {dlc_path}")
    
    # Create dummy input (we don't know the exact input shape, so we'll try a few)
    input_sizes = [4, 16, 64, 256, 1024]  # Try different sizes
    
    for size in input_sizes:
        logger.info(f"Trying input size: {size}")
        
        # Create input data
        input_file = OUTPUT_DIR / f"input_{size}.raw"
        with open(input_file, 'wb') as f:
            for i in range(size):
                f.write(struct.pack('f', 0.5))  # Write float values
        
        # Create input list
        input_list = OUTPUT_DIR / f"input_list_{size}.txt"
        with open(input_list, 'w') as f:
            # We don't know the input name, so try common ones
            for input_name in ['input', 'input_0', 'input:0', 'data']:
                f.write(f"{input_name} {input_file}\n")
        
        # Run with NPU backend
        net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
        htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
        
        cmd = [
            str(net_run),
            "--model", str(dlc_path),
            "--backend", str(htp_backend),
            "--input_list", str(input_list),
            "--output_dir", str(OUTPUT_DIR),
            "--perf_profile", "extreme_performance",
            "--profiling_level", "detailed",
            "--log_level", "debug"
        ]
        
        logger.info(f"Running NPU inference...")
        
        try:
            # Monitor NPU before
            check_npu_usage()
            
            start = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Monitor NPU after
            check_npu_usage()
            
            logger.info(f"Result: {result.returncode}, Time: {elapsed:.2f}ms")
            
            if result.returncode == 0:
                logger.info(f"✅ NPU inference successful with input size {size}!")
                
                # Check for output files
                output_files = list(OUTPUT_DIR.glob("Result_*"))
                if output_files:
                    logger.info(f"  Found {len(output_files)} output files")
                    for out_file in output_files[:3]:
                        logger.info(f"    {out_file.name} ({out_file.stat().st_size} bytes)")
                
                # Check for profiling data
                profile_files = list(OUTPUT_DIR.glob("*.json"))
                for pf in profile_files:
                    with open(pf) as f:
                        data = json.load(f)
                        if 'htp' in str(data).lower() or 'hexagon' in str(data).lower():
                            logger.info(f"  ✓ HTP/Hexagon profiling data found in {pf.name}")
                
                return True
                
            elif "input" in result.stderr.lower() and "not found" in result.stderr.lower():
                logger.warning(f"Input name mismatch for size {size}, trying next size...")
                continue
            else:
                logger.warning(f"NPU inference failed with size {size}")
                if result.stderr:
                    logger.debug(f"Error: {result.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            logger.warning(f"NPU inference timed out with size {size}")
        except Exception as e:
            logger.error(f"NPU inference error: {e}")
    
    return False


def check_npu_usage():
    """Check NPU usage indicators"""
    try:
        # Check Task Manager for NPU usage
        result = subprocess.run(
            ["powershell", "-Command", 
             "Get-Counter -Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage' -ErrorAction SilentlyContinue | Select -ExpandProperty CounterSamples | Select InstanceName, CookedValue"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout:
            if "NPU" in result.stdout or float(re.search(r'[\d.]+', result.stdout).group()) > 0:
                logger.info("📊 NPU activity detected!")
                logger.info(result.stdout)
    except:
        pass
    
    # Check for Hexagon processes
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process | Where {$_.ProcessName -like '*hexagon*' -or $_.ProcessName -like '*fastrpc*'}"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout:
            logger.info("Hexagon processes found:")
            logger.info(result.stdout)
    except:
        pass


def main():
    """Main workflow for TFLite NPU test"""
    logger.info("=" * 80)
    logger.info("🎯 TFLITE NPU TEST - ALTERNATIVE APPROACH")
    logger.info("=" * 80)
    
    # Step 1: Setup environment
    setup_qnn_environment()
    
    # Step 2: Try to convert TFLite to DLC
    dlc_path = convert_tflite_to_dlc()
    
    if not dlc_path:
        # Try creating a custom TFLite model
        logger.info("Trying to create custom TFLite model...")
        tflite_path = create_simple_tflite_model()
        
        if tflite_path:
            # Try converting custom model
            converter = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-tflite-converter"
            dlc_path = OUTPUT_DIR / "custom_model.dlc"
            
            cmd = [
                sys.executable,
                str(converter),
                "--input_network", str(tflite_path),
                "--output_path", str(dlc_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and dlc_path.exists():
                    logger.info(f"✅ Converted custom TFLite to DLC!")
                else:
                    dlc_path = None
            except:
                dlc_path = None
    
    # Step 3: Run NPU inference if we have a DLC
    if dlc_path:
        success = run_npu_inference_with_dlc(dlc_path)
        
        if success:
            logger.info("🎉 BREAKTHROUGH: Real NPU inference via TFLite!")
            logger.info("Successfully bypassed ONNX converter issues!")
            return True
        else:
            logger.error("❌ NPU inference failed even with TFLite DLC")
    else:
        logger.error("❌ Could not create DLC from TFLite either")
        logger.info("The conversion tools have fundamental compatibility issues")
    
    return False


if __name__ == "__main__":
    import numpy as np
    import re
    
    success = main()
    
    if success:
        logger.info("🎉 SUCCESS: TFLite approach worked!")
        logger.info("Real NPU hardware inference achieved!")
    else:
        logger.info("❌ TFLite approach also blocked")
        logger.info("Need to find pre-built DLC models or fix Python dependencies")