# Converting Pre-Quantized Models to QNN Context Binary

## Overview

**🚨 MAJOR UPDATE**: QNN SDK has **native GGUF support**! The optimal pathway is now **GGUF → QNN DLC → Context Binary** using QNN SDK's built-in GGUF converter.

## Pathway Analysis

### 1. GGUF → QNN DLC (Native, Recommended)

**🎯 BREAKTHROUGH**: QNN SDK includes native GGUF support via `qairt-converter` with `build_onnx_graph_from_gguf()` function.

```mermaid
GGUF Model → QNN Native Converter → QNN DLC → Context Binary
```

**Key Discovery**: The `qairt-converter` tool automatically detects `.gguf` files and uses the `LLMBuilder` class to:
1. Parse GGUF format directly
2. Extract quantized weights and dequantize them
3. Generate ONNX graph internally
4. Apply QNN-specific optimizations
5. Output QNN DLC format directly

```python
# Native GGUF support in QNN SDK (qairt-converter:318-320)
if framework == 'gguf':
    build_onnx_graph_from_gguf(args)
    framework = 'onnx'  # Internally converted
```

**Native GGUF Conversion Workflow**:

```python
# Direct GGUF to DLC using QNN SDK
from qti.aisw.converters.llm_builder import LLMBuilder

def convert_gguf_to_dlc_native(
    gguf_path: str,
    output_dir: str,
    batch_size: int = 1
) -> tuple[str, str, list, list]:
    """
    Convert GGUF directly to QNN DLC using native SDK support.
    
    Returns:
        onnx_path: Generated ONNX model path
        encodings_path: Quantization encodings file
        input_layouts: Layout specifications
        inputs_to_preserve: Input preservation settings
    """
    
    # Initialize LLMBuilder with GGUF file
    builder = LLMBuilder(
        input_model=gguf_path,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    # Build GenAI model from GGUF (does everything internally)
    return builder.build_from_gguf()

# CLI Usage - Direct GGUF Support
qairt_cmd = [
    "python", "/path/to/qairt-converter",
    "--input_network", "model.gguf",           # QNN auto-detects GGUF
    "--output_path", "model.dlc",              # Output DLC directly
    "--input_layout", "input_ids,NONTRIVIAL",  # Set layouts for LLM
    "--preserve_io", "datatype,input_ids,attention_mask"
]
```

**Key Advantages of Native GGUF Support**:
- ✅ **No manual ONNX conversion** required
- ✅ **Preserves quantization metadata** from GGUF
- ✅ **Automatic weight dequantization** for processing
- ✅ **LLM-optimized layouts** and input handling
- ✅ **Direct integration** with QNN optimization pipeline

### 2. GGUF → ONNX → QNN (Manual, Fallback)

```mermaid
GGUF Model → Convert to ONNX → QNN Converter → Context Binary
```

**Tools Required**:
- `llama.cpp` or `transformers` for GGUF → ONNX conversion
- QNN SDK for ONNX → Context Binary

**Step-by-Step Process**:

```python
# Step 1: Convert GGUF to ONNX
import onnx
from transformers import AutoTokenizer, AutoModelForCausalLM

def convert_gguf_to_onnx(gguf_path: str, onnx_path: str):
    """Convert GGUF model to ONNX format"""
    
    # Option A: Using llama.cpp
    import subprocess
    subprocess.run([
        "python", "-m", "llama_cpp.convert_gguf_to_onnx",
        "--input", gguf_path,
        "--output", onnx_path
    ])
    
    # Option B: Using transformers (if model is HF compatible)
    model = AutoModelForCausalLM.from_pretrained(
        gguf_path, 
        torch_dtype=torch.float16
    )
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset_version=17
    )

# Step 2: Use QNN Converter with float fallback
from qnn_compiler import QNNCompiler, CompilationConfig

compiler = QNNCompiler()
context_binary = compiler.compile(
    onnx_path,
    "model_context.bin",
    compilation=CompilationConfig(
        output_format="context-binary",
        backend="htp",
        preserve_external_quantization=True  # Key parameter!
    )
)
```

### 2. QNN SDK Direct Quantization Handling

QNN SDK provides several mechanisms for handling pre-quantized models:

#### Option A: `--float_fallback` Mode

```python
# Use QNN with float fallback for incompatible quantization
args = [
    "--input_network", "pre_quantized_model.onnx",
    "--output_path", "model_fallback.dlc",
    "--float_fallback",  # Enable fallback to FP16/FP32
    "--float_bitwidth", "16"  # Use FP16 instead of INT8
]
```

**When to use**:
- GGUF quantization scheme incompatible with QNN
- Mixed precision requirements
- Preserve external quantization accuracy

#### Option B: `--ignore_encodings` Mode

```python
# Force QNN to re-quantize ignoring existing encodings
args = [
    "--input_network", "pre_quantized_model.onnx",  
    "--output_path", "model_requantized.dlc",
    "--ignore_encodings",  # Ignore existing quantization
    "--input_list", "calibration_data.txt",  # Provide new calibration
    "--act_bitwidth", "8",
    "--weights_bitwidth", "8"
]
```

**When to use**:
- Want QNN-optimized quantization instead of GGUF
- Have calibration data available
- Need HTP-specific quantization

#### Option C: `--use_native_input_files` Mode

```python
# Preserve existing quantization parameters
args = [
    "--input_network", "pre_quantized_model.onnx",
    "--output_path", "model_native.dlc", 
    "--use_native_input_files",  # Preserve INT8 data types
    "--use_native_output_files"  # Output in native format
]
```

**When to use**:
- Model already properly quantized for QNN
- Want to preserve existing INT8/INT16 precision
- No re-quantization needed

## Complete Implementation

```python
class PreQuantizedQNNConverter:
    """Converter for pre-quantized models to QNN format"""
    
    def __init__(self, qnn_sdk_root: str = None):
        self.compiler = QNNCompiler(qnn_sdk_root)
    
    def convert_gguf_to_qnn(
        self,
        gguf_path: str,
        output_path: str,
        strategy: str = "native_gguf"  # Changed default to native
    ) -> str:
        """
        Convert GGUF model to QNN context binary.
        
        Args:
            gguf_path: Path to GGUF model
            output_path: Output path for context binary
            strategy: "native_gguf", "float_fallback", "ignore_encodings", or "native"
        
        Returns:
            Path to generated context binary
        """
        
        # Step 1: Choose conversion strategy
        if strategy == "native_gguf":
            return self._convert_gguf_native(gguf_path, output_path)
        else:
            # Fallback strategies require ONNX conversion first
            onnx_path = Path(output_path).with_suffix(".onnx")
            self._convert_gguf_to_onnx(gguf_path, onnx_path)
            
            # Step 2: Apply QNN conversion based on strategy
            if strategy == "float_fallback":
                return self._convert_with_float_fallback(onnx_path, output_path)
            elif strategy == "ignore_encodings":
                return self._convert_with_requantization(onnx_path, output_path)
            elif strategy == "native":
                return self._convert_preserve_native(onnx_path, output_path)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    def _convert_gguf_native(self, gguf_path: str, output_path: str) -> str:
        """Convert GGUF directly to QNN DLC using native SDK support"""
        
        from qti.aisw.converters.llm_builder import LLMBuilder
        import subprocess
        import os
        
        # Step 1: Use LLMBuilder for native GGUF processing
        output_dir = Path(output_path).parent
        builder = LLMBuilder(
            input_model=gguf_path,
            output_dir=str(output_dir),
            batch_size=1
        )
        
        # Build model from GGUF (generates ONNX + encodings internally)
        onnx_path, encodings_path, input_layouts, inputs_to_preserve = builder.build_from_gguf()
        
        # Step 2: Convert ONNX to DLC using qairt-converter with GGUF optimizations
        qairt_converter = os.path.join(self.compiler.qnn_sdk_root, "bin", "qairt-converter")
        
        cmd = [
            "python", qairt_converter,
            "--input_network", onnx_path,
            "--output_path", output_path,
            "--quantization_overrides", encodings_path,  # Use GGUF encodings
        ]
        
        # Add LLM-specific layouts
        for layout in input_layouts:
            cmd.extend(["--input_layout", f"{layout[0]},{layout[1]}"])
        
        # Add input preservation settings  
        for preserve_setting in inputs_to_preserve:
            cmd.extend(["--preserve_io", preserve_setting])
        
        subprocess.run(cmd, check=True)
        
        # Cleanup intermediate files
        os.remove(onnx_path)
        if os.path.exists(encodings_path):
            os.remove(encodings_path)
        
        return output_path
    
    def _convert_gguf_to_onnx(self, gguf_path: str, onnx_path: str):
        """Convert GGUF to ONNX using llama.cpp or transformers"""
        
        # Try llama.cpp first
        try:
            import subprocess
            result = subprocess.run([
                "python", "-m", "llama_cpp.convert_gguf_to_onnx",
                "--input", gguf_path,
                "--output", str(onnx_path)
            ], capture_output=True, check=True)
            
            if onnx_path.exists():
                return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback to manual conversion
        logger.warning("llama.cpp not available, using manual conversion")
        self._manual_gguf_to_onnx(gguf_path, onnx_path)
    
    def _convert_with_float_fallback(self, onnx_path: str, output_path: str) -> str:
        """Convert preserving GGUF quantization via float fallback"""
        
        config = CompilationConfig(
            backend=Backend.HTP,
            output_format="context-binary"
        )
        
        # Use float fallback to preserve external quantization
        args = self.compiler._create_converter_args(
            onnx_path, output_path, None, config
        )
        
        # Add float fallback arguments
        args.float_fallback = True
        args.float_bitwidth = 16  # Use FP16
        args.ignore_encodings = False  # Use existing encodings
        
        return self.compiler._execute_conversion_with_args(args)
    
    def _convert_with_requantization(self, onnx_path: str, output_path: str) -> str:
        """Convert with QNN re-quantization, ignoring GGUF quantization"""
        
        quant_config = QuantizationConfig(
            enabled=True,
            calibration_data=self._generate_calibration_data(),
            act_bitwidth=8,
            weights_bitwidth=8
        )
        
        config = CompilationConfig(
            backend=Backend.HTP,
            output_format="context-binary"
        )
        
        # Force re-quantization
        args = self.compiler._create_converter_args(
            onnx_path, output_path, quant_config, config
        )
        args.ignore_encodings = True  # Ignore GGUF quantization
        
        return self.compiler._execute_conversion_with_args(args)
    
    def _convert_preserve_native(self, onnx_path: str, output_path: str) -> str:
        """Convert preserving native quantization data types"""
        
        config = CompilationConfig(
            backend=Backend.HTP,
            output_format="context-binary"
        )
        
        args = self.compiler._create_converter_args(
            onnx_path, output_path, None, config
        )
        
        # Preserve native data types
        args.use_native_input_files = True
        args.use_native_output_files = True
        args.ignore_encodings = False
        
        return self.compiler._execute_conversion_with_args(args)

# Usage example
converter = PreQuantizedQNNConverter()

# 🏆 RECOMMENDED: Convert GGUF using native SDK support (best of all worlds!)
context_native = converter.convert_gguf_to_qnn(
    "model.gguf",
    "model_native.dlc",
    strategy="native_gguf"  # Uses QNN's built-in GGUF converter
)

# Fallback: Convert GGUF with float fallback (preserves GGUF quantization)
context1 = converter.convert_gguf_to_qnn(
    "model.gguf",
    "model_preserved.dlc", 
    strategy="float_fallback"
)

# Fallback: Convert GGUF with QNN re-quantization (better NPU optimization)
context2 = converter.convert_gguf_to_qnn(
    "model.gguf",
    "model_optimized.dlc",
    strategy="ignore_encodings"
)
```

## Strategy Comparison

| Strategy | Preserves GGUF Quantization | NPU Optimization | Accuracy | Speed | Complexity |
|----------|------------------------------|------------------|----------|-------|------------|
| **native_gguf** | ✅ Optimal | ✅ Best | ✅ Best | ✅ Best | ✅ Simple |
| **float_fallback** | ✅ High | ⚠️ Medium | ✅ Best | ⚠️ Medium | ⚠️ Medium |
| **ignore_encodings** | ❌ No | ✅ Best | ⚠️ Good | ✅ Best | ❌ Complex |
| **native** | ⚠️ Partial | ✅ Good | ✅ Good | ✅ Good | ⚠️ Medium |

**🏆 Recommended**: Use `native_gguf` strategy for all GGUF models - it's the optimal pathway!

## Format-Specific Considerations

### GGUF Models
- **Q4_0, Q4_1**: 4-bit quantization → Use `float_fallback` or `ignore_encodings`
- **Q5_0, Q5_1**: 5-bit quantization → Use `ignore_encodings` (QNN doesn't support 5-bit)
- **Q8_0**: 8-bit quantization → Use `native` or `ignore_encodings`
- **F16**: Half precision → Use `float_fallback` with `--float_bitwidth 16`

### Other Pre-Quantized Formats
- **TensorRT INT8**: Convert to ONNX, then use `native` strategy
- **TensorFlow Lite**: Use `tflite2onnx` + `ignore_encodings`
- **PyTorch Mobile**: Export to ONNX + appropriate strategy

## Performance Considerations

### Memory Usage
- **float_fallback**: Highest memory (preserves FP16/FP32)
- **ignore_encodings**: Lowest memory (QNN INT8 optimization)
- **native**: Medium memory (depends on original quantization)

### Inference Speed
- **HTP Backend**: Best with `ignore_encodings` (native INT8)
- **GPU Backend**: Best with `float_fallback` (FP16 optimized)
- **CPU Backend**: Similar across strategies

### Accuracy
- **GGUF → float_fallback**: Usually preserves original accuracy
- **GGUF → ignore_encodings**: May improve or degrade (depends on calibration)
- **GGUF → native**: Good compromise

## Troubleshooting

### Common Issues

1. **"Unsupported quantization scheme"**
   - Use `--float_fallback` to bypass incompatible quantization
   
2. **"Missing calibration data"** 
   - Provide `--input_list` when using `--ignore_encodings`
   
3. **"Context binary generation failed"**
   - Check backend compatibility (HTP vs GPU vs DSP)
   
4. **"Accuracy loss after conversion"**
   - Try `--float_fallback` to preserve original quantization

### Best Practices

1. **Always validate accuracy** after conversion
2. **Use representative calibration data** for re-quantization
3. **Profile on target hardware** to choose best strategy
4. **Keep original model** for comparison and fallback

This approach gives you maximum flexibility in converting pre-quantized models while leveraging QNN's NPU optimization capabilities!