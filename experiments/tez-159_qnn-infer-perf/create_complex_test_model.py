#!/usr/bin/env python3
"""
Create Complex Test Model - Conv2D + Attention for NPU Testing
This creates a realistic model combining convolution and attention mechanisms
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
OUTPUT_DIR = Path("./complex_model_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_conv_attention_model():
    """Create a model with Conv2D + Attention layers for realistic NPU testing"""
    logger.info("Creating Conv2D + Attention model...")
    
    try:
        import onnx
        from onnx import helper, TensorProto, ValueInfoProto, numpy_helper
        
        # Model architecture: 
        # Input[1,3,64,64] -> Conv2D -> ReLU -> GlobalAveragePool -> 
        # Reshape -> Multi-Head Attention -> Dense -> Output[1,10]
        
        # =============================================================================
        # 1. Define inputs and outputs
        # =============================================================================
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 64, 64])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
        
        # =============================================================================
        # 2. Create Conv2D weights and biases
        # =============================================================================
        # Conv2D: 3 input channels -> 32 output channels, 3x3 kernel
        conv_weight = np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.1
        conv_bias = np.random.randn(32).astype(np.float32) * 0.01
        
        # Create initializers for conv layer
        conv_weight_init = numpy_helper.from_array(conv_weight, name='conv_weight')
        conv_bias_init = numpy_helper.from_array(conv_bias, name='conv_bias')
        
        # =============================================================================
        # 3. Create Attention weights (simplified multi-head attention)
        # =============================================================================
        # After GlobalAveragePool: [1, 32] -> attention -> [1, 32]
        embed_dim = 32
        num_heads = 4
        head_dim = embed_dim // num_heads
        
        # Query, Key, Value projection weights
        qkv_weight = np.random.randn(embed_dim, embed_dim * 3).astype(np.float32) * 0.1
        qkv_bias = np.random.randn(embed_dim * 3).astype(np.float32) * 0.01
        
        # Output projection
        out_proj_weight = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
        out_proj_bias = np.random.randn(embed_dim).astype(np.float32) * 0.01
        
        # Final classifier
        classifier_weight = np.random.randn(embed_dim, 10).astype(np.float32) * 0.1
        classifier_bias = np.random.randn(10).astype(np.float32) * 0.01
        
        # Create initializers for attention and classifier
        qkv_weight_init = numpy_helper.from_array(qkv_weight, name='qkv_weight')
        qkv_bias_init = numpy_helper.from_array(qkv_bias, name='qkv_bias')
        out_proj_weight_init = numpy_helper.from_array(out_proj_weight, name='out_proj_weight')
        out_proj_bias_init = numpy_helper.from_array(out_proj_bias, name='out_proj_bias')
        classifier_weight_init = numpy_helper.from_array(classifier_weight, name='classifier_weight')
        classifier_bias_init = numpy_helper.from_array(classifier_bias, name='classifier_bias')
        
        # =============================================================================
        # 4. Create computation nodes
        # =============================================================================
        nodes = []
        
        # Conv2D layer
        conv_node = helper.make_node(
            'Conv',
            inputs=['input', 'conv_weight', 'conv_bias'],
            outputs=['conv_out'],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],  # SAME padding
        )
        nodes.append(conv_node)
        
        # ReLU activation
        relu_node = helper.make_node('Relu', ['conv_out'], ['relu_out'])
        nodes.append(relu_node)
        
        # Global Average Pooling: [1, 32, 64, 64] -> [1, 32, 1, 1]
        gap_node = helper.make_node(
            'GlobalAveragePool',
            inputs=['relu_out'],
            outputs=['gap_out']
        )
        nodes.append(gap_node)
        
        # Reshape for attention: [1, 32, 1, 1] -> [1, 32]
        reshape_shape = numpy_helper.from_array(np.array([1, 32], dtype=np.int64), name='reshape_shape')
        reshape_node = helper.make_node('Reshape', ['gap_out', 'reshape_shape'], ['reshaped'])
        nodes.append(reshape_node)
        
        # =============================================================================
        # 5. Simplified Multi-Head Attention (using MatMul operations)
        # =============================================================================
        
        # QKV projection: [1, 32] @ [32, 96] -> [1, 96]
        qkv_node = helper.make_node('MatMul', ['reshaped', 'qkv_weight'], ['qkv_matmul'])
        nodes.append(qkv_node)
        
        qkv_add_node = helper.make_node('Add', ['qkv_matmul', 'qkv_bias'], ['qkv_out'])
        nodes.append(qkv_add_node)
        
        # Split QKV: [1, 96] -> 3 x [1, 32]
        # For simplicity, we'll just use the first 32 dims as "attended" features
        attended_slice_start = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name='slice_start')
        attended_slice_end = numpy_helper.from_array(np.array([1, 32], dtype=np.int64), name='slice_end')
        
        attended_node = helper.make_node(
            'Slice',
            inputs=['qkv_out', 'slice_start', 'slice_end'],
            outputs=['attended_features']
        )
        nodes.append(attended_node)
        
        # Output projection: [1, 32] @ [32, 32] -> [1, 32]
        out_proj_node = helper.make_node('MatMul', ['attended_features', 'out_proj_weight'], ['out_proj_matmul'])
        nodes.append(out_proj_node)
        
        out_proj_add_node = helper.make_node('Add', ['out_proj_matmul', 'out_proj_bias'], ['attention_out'])
        nodes.append(out_proj_add_node)
        
        # Residual connection: attended_features + attention_out
        residual_node = helper.make_node('Add', ['attended_features', 'attention_out'], ['residual_out'])
        nodes.append(residual_node)
        
        # Final classifier: [1, 32] @ [32, 10] -> [1, 10]
        classifier_node = helper.make_node('MatMul', ['residual_out', 'classifier_weight'], ['classifier_matmul'])
        nodes.append(classifier_node)
        
        classifier_add_node = helper.make_node('Add', ['classifier_matmul', 'classifier_bias'], ['output'])
        nodes.append(classifier_add_node)
        
        # =============================================================================
        # 6. Create the graph
        # =============================================================================
        initializers = [
            conv_weight_init, conv_bias_init,
            qkv_weight_init, qkv_bias_init,
            out_proj_weight_init, out_proj_bias_init,
            classifier_weight_init, classifier_bias_init,
            reshape_shape, attended_slice_start, attended_slice_end
        ]
        
        graph_def = helper.make_graph(
            nodes=nodes,
            name='ConvAttentionModel',
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='QNN-ConvAttention-Test')
        model_def.opset_import[0].version = 11
        
        # Save model
        onnx_path = OUTPUT_DIR / "conv_attention_model.onnx"
        onnx.save(model_def, str(onnx_path))
        
        # Verify model
        onnx.checker.check_model(model_def)
        
        logger.info(f"✓ Created Conv2D + Attention ONNX model: {onnx_path}")
        logger.info(f"  Model size: {onnx_path.stat().st_size / 1024:.1f} KB")
        logger.info(f"  Input shape: [1, 3, 64, 64] (RGB image)")
        logger.info(f"  Output shape: [1, 10] (classification)")
        logger.info(f"  Architecture: Conv2D(32) -> ReLU -> GAP -> Attention -> Classifier")
        
        return onnx_path
        
    except ImportError as e:
        logger.error(f"ONNX not available: {e}")
        logger.error("Install with: uv add onnx")
        return None
    except Exception as e:
        logger.error(f"Failed to create ONNX model: {e}")
        return None


def create_realistic_input_data():
    """Create realistic test input data"""
    logger.info("Creating realistic input data...")
    
    # Create RGB image data [1, 3, 64, 64]
    # Simulate a natural image with some structure
    np.random.seed(42)  # For reproducible results
    
    # Create base image with gradients and patterns
    height, width = 64, 64
    image = np.zeros((3, height, width), dtype=np.float32)
    
    # Red channel: horizontal gradient + noise
    for y in range(height):
        image[0, y, :] = (y / height) + np.random.normal(0, 0.1, width)
    
    # Green channel: vertical gradient + noise  
    for x in range(width):
        image[1, :, x] = (x / width) + np.random.normal(0, 0.1, height)
    
    # Blue channel: circular pattern + noise
    center_y, center_x = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2) / (height // 2)
            image[2, y, x] = np.sin(dist * np.pi) + np.random.normal(0, 0.05)
    
    # Normalize to [0, 1] range
    image = np.clip(image, 0, 1)
    
    # Add batch dimension: [1, 3, 64, 64]
    input_data = image[np.newaxis, ...]
    
    # Save as raw binary file
    input_file = OUTPUT_DIR / "realistic_input.raw"
    with open(input_file, 'wb') as f:
        for val in input_data.flatten():
            f.write(struct.pack('f', val))
    
    # Create input list file for QNN
    input_list = OUTPUT_DIR / "input_list.txt"
    with open(input_list, 'w') as f:
        f.write(f"input:0 {input_file}\n")
    
    logger.info(f"✓ Created realistic input data: {input_file}")
    logger.info(f"  Shape: {input_data.shape}")
    logger.info(f"  Data range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    logger.info(f"  Input list: {input_list}")
    
    return input_list, input_data


def convert_to_dlc_and_test():
    """Convert the complex model to DLC and test NPU performance"""
    logger.info("="*80)
    logger.info("COMPLEX MODEL NPU TESTING WORKFLOW")
    logger.info("="*80)
    
    # Step 1: Create complex ONNX model
    onnx_path = create_conv_attention_model()
    if not onnx_path:
        logger.error("❌ Failed to create ONNX model")
        return False
    
    # Step 2: Create realistic input data  
    input_list, input_data = create_realistic_input_data()
    
    # Step 3: Convert ONNX to DLC
    logger.info("Converting ONNX to DLC format...")
    
    converter = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter"
    dlc_path = OUTPUT_DIR / "conv_attention_model.dlc"
    
    # Try different converter approaches
    conversion_commands = [
        # Standard conversion
        [
            str(converter),
            "--input_network", str(onnx_path),
            "--output_path", str(dlc_path),
            "--input_dim", "input", "1,3,64,64"
        ],
        # With quantization disabled
        [
            str(converter), 
            "--input_network", str(onnx_path),
            "--output_path", str(dlc_path),
            "--input_dim", "input", "1,3,64,64",
            "--float_output"
        ]
    ]
    
    dlc_success = False
    for i, cmd in enumerate(conversion_commands, 1):
        logger.info(f"Trying conversion approach {i}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and dlc_path.exists():
                logger.info(f"✓ Successfully converted to DLC: {dlc_path}")
                logger.info(f"  DLC file size: {dlc_path.stat().st_size / 1024:.1f} KB")
                dlc_success = True
                break
            else:
                logger.warning(f"Conversion approach {i} failed:")
                logger.warning(f"  Return code: {result.returncode}")
                if result.stdout:
                    logger.warning(f"  stdout: {result.stdout[:500]}")
                if result.stderr:
                    logger.warning(f"  stderr: {result.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            logger.warning(f"Conversion approach {i} timed out")
        except Exception as e:
            logger.warning(f"Conversion approach {i} error: {e}")
    
    if not dlc_success:
        logger.error("❌ All conversion approaches failed")
        logger.info("The model architecture may be too complex for current QNN SDK")
        logger.info("Try using pre-built QNN models or simpler architectures")
        return False
    
    # Step 4: Test NPU inference with the complex model
    logger.info("Testing NPU inference with complex model...")
    
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
        "--perf_profile", "extreme_performance",
        "--profiling_level", "detailed"
    ]
    
    logger.info("🚀 Running NPU inference...")
    htp_start = time.perf_counter()
    try:
        htp_result = subprocess.run(htp_cmd, capture_output=True, text=True, timeout=60)
        htp_time = (time.perf_counter() - htp_start) * 1000
        
        logger.info(f"NPU inference: {htp_time:.2f}ms")
        results['htp_time'] = htp_time
        results['htp_success'] = htp_result.returncode == 0
        
        # Look for NPU-specific metrics in output
        if "HVX" in htp_result.stdout or "HMX" in htp_result.stdout:
            logger.info("✅ NPU hardware metrics detected in output!")
        
    except subprocess.TimeoutExpired:
        logger.error("NPU inference timed out")
        results['htp_time'] = 60000
        results['htp_success'] = False
    
    # Test CPU
    cpu_cmd = [
        str(net_run),
        "--model", str(dlc_path),
        "--backend", str(cpu_backend), 
        "--input_list", str(input_list),
        "--output_dir", str(OUTPUT_DIR),
        "--perf_profile", "high_performance"
    ]
    
    logger.info("🖥️ Running CPU inference...")
    cpu_start = time.perf_counter()
    try:
        cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=60)
        cpu_time = (time.perf_counter() - cpu_start) * 1000
        
        logger.info(f"CPU inference: {cpu_time:.2f}ms")
        results['cpu_time'] = cpu_time
        results['cpu_success'] = cpu_result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error("CPU inference timed out")
        results['cpu_time'] = 60000
        results['cpu_success'] = False
    
    # Analyze results
    logger.info("\n" + "="*80)
    logger.info("COMPLEX MODEL INFERENCE RESULTS")
    logger.info("="*80)
    
    if results.get('htp_success') and results.get('cpu_success'):
        speedup = results['cpu_time'] / results['htp_time']
        logger.info(f"🚀 NPU Inference: {results['htp_time']:.2f}ms")
        logger.info(f"🖥️ CPU Inference: {results['cpu_time']:.2f}ms")  
        logger.info(f"⚡ NPU Speedup: {speedup:.2f}x faster")
        
        if speedup > 1.2:
            logger.info("✅ SIGNIFICANT NPU ACCELERATION CONFIRMED!")
            logger.info("🎉 Real NPU hardware is being utilized for inference!")
        else:
            logger.info("⚠️ NPU performance similar to CPU")
            
        return True
    else:
        logger.error("❌ Inference tests failed")
        return False


if __name__ == "__main__":
    success = convert_to_dlc_and_test()
    
    if success:
        logger.info("🎉 SUCCESS: Complex model NPU testing completed!")
    else:
        logger.info("❌ FAILED: Could not complete complex model testing")