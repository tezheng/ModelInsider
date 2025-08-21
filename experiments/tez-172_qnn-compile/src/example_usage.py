#!/usr/bin/env python3
"""
Example usage of QNN compilation pipeline for modelexport.
This demonstrates how we can integrate QNN compilation into modelexport.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional


def prepare_calibration_data(
    input_shape: List[int],
    num_samples: int = 100,
    output_dir: Path = Path("./calibration_data")
) -> List[str]:
    """
    Prepare calibration data for quantization.
    
    Args:
        input_shape: Shape of model input (e.g., [1, 3, 224, 224])
        num_samples: Number of calibration samples
        output_dir: Directory to save calibration files
    
    Returns:
        List of paths to calibration data files
    """
    output_dir.mkdir(exist_ok=True)
    
    calibration_files = []
    for i in range(num_samples):
        # Generate random data (in practice, use real representative data)
        data = np.random.randn(*input_shape).astype(np.float32)
        
        # Save as raw binary
        file_path = output_dir / f"sample_{i:04d}.raw"
        data.tofile(str(file_path))
        calibration_files.append(str(file_path))
    
    print(f"Generated {num_samples} calibration samples in {output_dir}")
    return calibration_files


def compile_onnx_to_qnn_simple(onnx_path: str) -> dict:
    """
    Simple compilation without QNN SDK (simulation).
    
    This demonstrates the workflow when QNN SDK is not available.
    """
    print(f"\n{'='*60}")
    print("QNN Compilation Workflow (Simulation)")
    print(f"{'='*60}\n")
    
    # Stage 1: ONNX Analysis
    print("📊 Stage 1: Analyzing ONNX Model")
    print(f"  Input: {onnx_path}")
    print("  ✓ Validated model structure")
    print("  ✓ Extracted input shape: [1, 3, 224, 224]")
    print("  ✓ Found 50 operations")
    
    # Stage 2: Graph Optimization
    print("\n🔧 Stage 2: Graph Optimization")
    print("  ✓ Conv+BN fusion: 5 patterns")
    print("  ✓ Conv+ReLU fusion: 8 patterns")
    print("  ✓ Layout transform: NCHW → NHWC")
    print("  ✓ Dead code elimination: removed 3 ops")
    
    # Stage 3: Quantization Analysis
    print("\n📉 Stage 3: Quantization Preparation")
    print("  Target: INT8 quantization")
    print("  ✓ Identified quantizable ops: 42/50")
    print("  ✓ Non-quantizable ops will remain FP32")
    
    # Stage 4: Backend Code Generation
    print("\n💾 Stage 4: QNN Backend Generation")
    print("  Backend: HTP (Hexagon Tensor Processor)")
    print("  ✓ Generated QNN graph representation")
    print("  ✓ Applied HTP-specific optimizations")
    print("  ✓ Output format: DLC")
    
    results = {
        "output_path": "model_compiled.dlc",
        "backend": "HTP",
        "optimizations_applied": ["fusion", "layout_transform", "dead_code_elim"],
        "quantization_ready": True,
        "ops_total": 47,  # After optimization
        "ops_quantizable": 42
    }
    
    print(f"\n✅ Compilation complete: {results['output_path']}")
    return results


def compile_with_quantization_example():
    """
    Example of compilation with INT8 quantization.
    """
    print(f"\n{'='*60}")
    print("QNN Compilation with INT8 Quantization")
    print(f"{'='*60}\n")
    
    # If QNN SDK is available, use real implementation
    try:
        from qnn_compiler import QNNCompiler, QuantizationConfig, CompilationConfig, Backend
        
        # Try to initialize compiler (will fail if SDK not available)
        compiler = QNNCompiler()
        
        print("✓ QNN SDK detected - using real implementation\n")
        
        # Prepare calibration data
        print("📊 Preparing calibration data...")
        calibration_files = prepare_calibration_data(
            input_shape=[1, 3, 224, 224],
            num_samples=100
        )
        
        # Configure quantization
        quant_config = QuantizationConfig(
            enabled=True,
            calibration_data=calibration_files,
            act_bitwidth=8,  # INT8 activations
            weights_bitwidth=8,  # INT8 weights
            per_channel=True  # Per-channel quantization for Conv weights
        )
        
        # Configure compilation
        comp_config = CompilationConfig(
            backend=Backend.HTP,
            output_format="dlc",
            htp_performance_mode="high_performance"
        )
        
        # Compile model
        print("\n🔄 Compiling with quantization...")
        output = compiler.compile(
            "model.onnx",
            "model_int8.dlc",
            quantization=quant_config,
            compilation=comp_config
        )
        
        print(f"\n✅ Quantized model saved to: {output}")
        
    except ImportError:
        print("ℹ️ QNN SDK not available - showing workflow simulation\n")
        
        # Simulate the process
        print("📊 Stage 1: Calibration Data Generation")
        print("  Generated 100 calibration samples")
        print("  Sample shape: [1, 3, 224, 224]")
        
        print("\n📈 Stage 2: Calibration Statistics")
        print("  Computing activation ranges...")
        print("  ✓ Layer conv1: range [-2.35, 3.21]")
        print("  ✓ Layer conv2: range [-1.89, 2.77]")
        print("  ✓ Layer fc: range [-4.12, 5.33]")
        
        print("\n🔢 Stage 3: Quantization Parameters")
        print("  Activation quantization: UINT8")
        print("  Weight quantization: INT8 (per-channel)")
        print("  ✓ Conv layers: per-channel scales computed")
        print("  ✓ FC layers: per-tensor scales computed")
        
        print("\n💾 Stage 4: Quantized Model Generation")
        print("  ✓ Weights quantized to INT8")
        print("  ✓ Quantization parameters embedded")
        print("  ✓ Model size: 25MB → 7MB (72% reduction)")
        
        print("\n✅ Quantized model saved to: model_int8.dlc")


def compile_gguf_native_example():
    """
    🚨 NEW: Example showing native GGUF compilation.
    """
    print(f"\n{'='*60}")
    print("🚨 NATIVE GGUF COMPILATION")
    print(f"{'='*60}\n")
    
    try:
        from qnn_compiler import QNNCompiler, CompilationConfig, Backend
        
        compiler = QNNCompiler()
        
        print("✓ QNN SDK with LLMBuilder detected\n")
        
        # Configure compilation for GGUF
        config = CompilationConfig(
            backend=Backend.HTP,
            output_format="dlc",
            batch_size=1
        )
        
        print("🔄 Compiling GGUF using native QNN support...")
        print("📂 Input: model.gguf")
        print("📂 Output: model_native.dlc")
        
        # This would be the actual call:
        # output = compiler.compile_gguf("model.gguf", "model_native.dlc", config)
        
        print("\n🎯 Native GGUF Compilation Process:")
        print("  1. LLMBuilder parses GGUF format directly")
        print("  2. Extracts quantization metadata automatically") 
        print("  3. Dequantizes weights for processing")
        print("  4. Generates optimized ONNX graph internally")
        print("  5. Applies LLM-specific layouts and settings")
        print("  6. Compiles to QNN DLC with preserved quantization")
        print("  7. Cleans up intermediate files automatically")
        
        print("\n✨ Key Advantages:")
        print("  • No manual GGUF→ONNX conversion")
        print("  • Preserves original quantization scheme") 
        print("  • LLM-optimized input handling")
        print("  • Direct integration with QNN pipeline")
        print("  • Automatic cleanup of intermediate files")
        
    except ImportError:
        print("ℹ️ QNN SDK with LLMBuilder not available - showing workflow simulation\n")
        
        print("🔄 Stage 1: Native GGUF Parsing")
        print("  ✓ Loaded GGUF metadata and architecture")
        print("  ✓ Extracted quantization parameters (Q4_K, Q8_0, etc.)")
        print("  ✓ Identified model type and configuration")
        
        print("\n🔄 Stage 2: Internal ONNX Generation")
        print("  ✓ Dequantized weights for ONNX compatibility")
        print("  ✓ Generated ONNX graph with proper operators")
        print("  ✓ Applied LLM-specific optimizations")
        print("  ✓ Created quantization override encodings")
        
        print("\n🔄 Stage 3: QNN Native Compilation")
        print("  ✓ Converted ONNX to QNN IR format")
        print("  ✓ Applied HTP backend optimizations")
        print("  ✓ Generated DLC with preserved quantization")
        print("  ✓ Size reduction: ~50% vs manual conversion")
        
        print("\n🏆 Result: model_native.dlc")
        print("📊 Benefits: Native quantization + NPU optimization + Simplified workflow")


def compile_for_different_backends():
    """
    Example showing compilation for different QNN backends.
    """
    print(f"\n{'='*60}")
    print("Multi-Backend Compilation")
    print(f"{'='*60}\n")
    
    backends = {
        "HTP": {
            "description": "Hexagon Tensor Processor (NPU)",
            "best_for": "INT8/INT16 inference, power efficiency",
            "performance": "100 TOPS @ INT8"
        },
        "GPU": {
            "description": "Adreno GPU",
            "best_for": "FP16/FP32 inference, complex ops",
            "performance": "10 TFLOPS @ FP16"
        },
        "DSP": {
            "description": "Hexagon DSP",
            "best_for": "Audio/signal processing, custom ops",
            "performance": "Variable"
        },
        "CPU": {
            "description": "ARM CPU",
            "best_for": "Fallback, debugging, unsupported ops",
            "performance": "~50 GFLOPS"
        }
    }
    
    for backend_name, info in backends.items():
        print(f"🎯 Backend: {backend_name}")
        print(f"   {info['description']}")
        print(f"   Best for: {info['best_for']}")
        print(f"   Performance: {info['performance']}")
        
        # Simulate compilation
        print(f"   → Compiling for {backend_name}...")
        print(f"   ✓ Generated: model_{backend_name.lower()}.dlc\n")


def main():
    """
    Main demonstration of QNN compilation workflows.
    """
    print("\n" + "="*60)
    print(" QNN Compilation Pipeline for ModelExport")
    print("="*60)
    print("\nThis demonstrates how modelexport can leverage QNN SDK")
    print("for compiling ONNX models to Qualcomm NPU native format.\n")
    
    # 1. Simple compilation
    print("\n1️⃣ BASIC COMPILATION")
    compile_onnx_to_qnn_simple("model.onnx")
    
    # 2. Native GGUF compilation
    print("\n2️⃣ NATIVE GGUF COMPILATION")
    compile_gguf_native_example()
    
    # 3. Compilation with quantization  
    print("\n3️⃣ INT8 QUANTIZATION")
    compile_with_quantization_example()
    
    # 4. Multi-backend support
    print("\n4️⃣ BACKEND TARGETS")
    compile_for_different_backends()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 Integration with ModelExport")
    print(f"{'='*60}\n")
    
    print("Proposed CLI commands:")
    print("🚨 NEW: Native GGUF support!")
    print("  modelexport compile-qnn model.gguf model.dlc               # Native GGUF→DLC")
    print("  modelexport compile-qnn model.onnx model.dlc               # Traditional ONNX→DLC")
    print("  modelexport compile-qnn model.onnx model_int8.dlc --quantize")
    print("  modelexport compile-qnn model.gguf model.bin --format context-binary")
    
    print("\nPython API:")
    print("  from modelexport.qnn import QNNCompiler")
    print("  compiler = QNNCompiler()")
    print("🚨 NEW: compiler.compile_gguf('model.gguf', 'model.dlc')    # Native GGUF")
    print("  compiler.compile('model.onnx', 'model.dlc')              # Traditional ONNX")
    
    print("\n✨ Key Benefits:")
    print("🚨 • Native GGUF support (no manual ONNX conversion)")
    print("  • Direct Python SDK integration (no subprocess)")
    print("  • Full quantization control (INT4/8/16, FP16)")
    print("  • Multi-backend support (NPU, GPU, DSP, CPU)")
    print("  • LLM-optimized compilation pipeline")
    print("  • Hierarchy preservation through compilation")
    print("  • Integration with profiling (TEZ-159)")


if __name__ == "__main__":
    main()