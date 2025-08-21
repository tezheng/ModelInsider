# QNN Compilation Pipeline Research

## Executive Summary

After deep analysis of ONNX Runtime's QNN EP and QNN SDK, we've discovered that **QNN SDK provides complete Python APIs** for model compilation. The `qairt-converter` tool itself is a Python script leveraging these APIs, meaning we can directly integrate QNN compilation into modelexport without subprocess calls.

**🚨 MAJOR BREAKTHROUGH**: QNN SDK has **native GGUF support**! The `qairt-converter` includes built-in GGUF parsing via `build_onnx_graph_from_gguf()` function and `LLMBuilder` class, eliminating the need for manual GGUF→ONNX conversion.

## Architecture Overview

### 1. ONNX Runtime QNN EP Architecture

```
ONNX Model → Graph Partitioning → QNN Backend Manager → Context Binary
                     ↓                      ↓                 ↓
              Supported Nodes        QNN Graph Build    Cached Execution
```

**Key Components**:
- `QNNExecutionProvider`: Main orchestrator
- `QnnBackendManager`: Handles backend initialization and context caching
- `OpBuilder`: Translates ONNX ops to QNN ops
- Context caching provides 50-80% faster initialization

### 2. QNN SDK Native Python Architecture

```python
# Direct Python API usage - no subprocess needed!
from qti.aisw.converters import onnx as onnx_frontend
from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
from qti.aisw.converters.backend.qnn_quantizer import QnnQuantizer
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations

# 🚨 NEW DISCOVERY: Native GGUF support!
from qti.aisw.converters.llm_builder import LLMBuilder
```

### 3. Native GGUF Processing Architecture

```python
# qairt-converter automatically detects GGUF files (line 174-188)
def infer_framework(args):
    input_model_to_framework = {'.onnx': 'onnx', '.pb': 'tensorflow', 
                               '.pt': 'pytorch', '.tflite': 'tflite', 
                               '.gguf': 'gguf'}  # Native GGUF support!

# Main processing (line 318-320)
if framework == 'gguf':
    build_onnx_graph_from_gguf(args)  # Built-in GGUF→ONNX conversion
    framework = 'onnx'  # Then process as ONNX

# LLMBuilder handles the conversion (llm_builder.py)
builder = LLMBuilder(input_model, config_file, output_dir, batch_size)
onnx_path, encodings_path, layouts, preserve_settings = builder.build_from_gguf()
```

## Compilation Pipeline

### Stage 1: ONNX to IR (Intermediate Representation)

```python
# Frontend converts ONNX to internal IR
frontend = onnx_frontend.OnnxConverterFrontend(args)
graph = frontend.convert()
```

**Features**:
- Operator translation
- Shape inference
- Graph validation
- Custom op support

### Stage 2: Graph Optimizations

```python
# Apply graph-level optimizations
optimizer = IROptimizations(args)
optimized_graph = optimizer.optimize(graph)
```

**Optimizations**:
- Operation fusion (Conv+BN, Conv+Relu)
- Layout transformations (NCHW → NHWC for HTP)
- Dead code elimination
- Constant folding

### Stage 3: Quantization

```python
# Quantization with calibration data
quantizer = QnnQuantizer(args)
quantized_graph = quantizer.quantize(
    optimized_graph,
    input_list=calibration_data,
    act_bitwidth=8,      # INT8 activations
    weights_bitwidth=8,   # INT8 weights
    use_per_channel_quantization=True
)
```

**Quantization Options**:

| Type | Activation | Weight | Use Case |
|------|------------|---------|----------|
| INT4 | ✗ | ✓ | Ultra-low memory, weights only |
| INT8 | ✓ | ✓ | Standard quantization |
| INT16 | ✓ | ✓ | Higher precision |
| FP16 | ✓ | ✓ | Half-precision float |
| Mixed | INT8 | INT4 | Memory-optimized |

**Quantization Algorithms**:
- **TF Mode** (default): Min/max quantization
- **Enhanced**: Long-tail distribution handling
- **Symmetric**: Zero-centered quantization
- **Per-channel**: Channel-wise weight quantization
- **Per-row**: Row-wise quantization for MatMul/FC

### Stage 4: Backend Code Generation

```python
# Generate QNN backend code
backend = QnnConverterBackend(args)
backend.save(
    quantized_graph,
    output_path="model.cpp",  # C++ or DLC format
    model_version="1.0.0"
)
```

**Output Formats**:
- **CPP**: Human-readable C++ code with QNN API calls
- **DLC**: Binary format (Deep Learning Container)
- **Context Binary**: Pre-compiled for specific hardware

### Stage 5: Context Binary Generation

```python
# Generate context binary for deployment
import subprocess

# Convert C++ to shared library
subprocess.run([
    "qnn-model-lib-generator",
    "-c", "model.cpp",
    "-b", "model.bin",  # Optional weights file
    "-o", "libmodel.so"
])

# Generate context binary
subprocess.run([
    "qnn-context-binary-generator",
    "--model", "libmodel.so",
    "--backend", "libQnnHtp.so",
    "--output", "model_context.bin",
    "--config", "htp_config.json"
])
```

## Python Implementation Strategy

### Option 1: Direct SDK Integration (Recommended)

```python
class QNNCompiler:
    """Direct integration with QNN SDK Python APIs"""
    
    def __init__(self, backend="HTP"):
        self.backend = backend
        
    def compile_onnx_to_dlc(
        self,
        onnx_path: str,
        output_path: str,
        quantization_config: Optional[Dict] = None,
        calibration_data: Optional[List[str]] = None
    ) -> str:
        """
        Compile ONNX to DLC using QNN Python SDK directly.
        
        Args:
            onnx_path: Input ONNX model
            output_path: Output DLC path
            quantization_config: Quantization settings
            calibration_data: Calibration dataset paths
        
        Returns:
            Path to generated DLC file
        """
        # 1. Setup arguments
        args = self._create_args(onnx_path, output_path, quantization_config)
        
        # 2. Frontend conversion
        frontend = onnx_frontend.OnnxConverterFrontend(args)
        graph = frontend.convert()
        
        # 3. Optimizations
        optimizer = IROptimizations(args)
        graph = optimizer.optimize(graph)
        
        # 4. Quantization (if enabled)
        if calibration_data:
            quantizer = QnnQuantizer(args)
            graph = quantizer.quantize(graph)
        
        # 5. Backend generation
        backend = QnnConverterBackend(args)
        backend.save(graph)
        
        return output_path
```

### Option 2: Wrapper Around CLI Tools

```python
class QNNCompilerCLI:
    """Wrapper around QNN command-line tools"""
    
    def compile_onnx_to_dlc(self, onnx_path: str, **kwargs) -> str:
        cmd = [
            "python", f"{QNN_SDK_ROOT}/bin/qnn-onnx-converter",
            "--input_network", onnx_path,
            "--output_path", output_path
        ]
        
        if kwargs.get("quantize"):
            cmd.extend([
                "--input_list", kwargs["calibration_data"],
                "--act_bitwidth", str(kwargs.get("act_bitwidth", 8)),
                "--weights_bitwidth", str(kwargs.get("weights_bitwidth", 8))
            ])
        
        subprocess.run(cmd, check=True)
        return output_path
```

## Quantization Deep Dive

### Calibration Data Requirements

```python
# Create calibration data list file
with open("calibration_list.txt", "w") as f:
    for sample in calibration_samples:
        # Each line: path to raw binary input
        f.write(f"{sample}\n")
```

**Data Format**:
- Raw binary files (`.raw` extension)
- Float32 or native quantized format
- Shape must match model input

### Per-Channel Quantization

```python
# Enable per-channel for better accuracy
args.use_per_channel_quantization = True

# Supported for:
# - Conv2D weights
# - DepthwiseConv2D weights  
# - TransposedConv2D weights
```

### Mixed Precision

```python
# Configure mixed precision
config = {
    "act_bitwidth": 8,        # INT8 activations
    "weights_bitwidth": 4,    # INT4 weights
    "bias_bitwidth": 32,      # INT32 bias
    "float_bitwidth": 16,     # FP16 for non-quantized ops
}
```

## Context Binary Benefits

### Performance Advantages
- **50-80% faster startup**: Pre-compiled graph
- **Reduced memory**: Optimized for target hardware
- **Hardware-specific**: Tuned for NPU/DSP/GPU

### Deployment Workflow

```python
# 1. Generate context during build
context_binary = compile_to_context_binary(model, target_device)

# 2. Deploy context binary
session = ort.InferenceSession(
    model_with_context,
    providers=[("QNNExecutionProvider", {
        "backend_path": "libQnnHtp.so",
        "context_cache_path": context_binary
    })]
)
```

## Integration with ModelExport

### Proposed CLI Interface

```bash
# Basic conversion
modelexport compile-qnn model.onnx model.dlc

# With INT8 quantization
modelexport compile-qnn model.onnx model_int8.dlc \
    --quantize \
    --calibration-data calibration_list.txt \
    --act-bitwidth 8 \
    --weights-bitwidth 8

# Generate context binary
modelexport compile-qnn model.onnx model_ctx.bin \
    --format context-binary \
    --backend HTP \
    --device "Snapdragon 8 Gen 3"
```

### Python API

```python
from modelexport.qnn import QNNCompiler

compiler = QNNCompiler(backend="HTP")

# Simple conversion
dlc_path = compiler.compile(
    "model.onnx",
    output_format="dlc"
)

# With quantization
quantized_dlc = compiler.compile(
    "model.onnx",
    output_format="dlc",
    quantization={
        "enabled": True,
        "calibration_data": calibration_loader,
        "act_bitwidth": 8,
        "weights_bitwidth": 8,
        "per_channel": True
    }
)

# Generate context binary
context = compiler.compile(
    "model.onnx",
    output_format="context-binary",
    target_device="snapdragon_8_gen3"
)
```

## Key Findings

### ✅ What We Can Do with Python

1. **Direct SDK Integration**: All core functionality is exposed via Python
2. **🚨 Native GGUF Support**: Built-in GGUF→ONNX conversion with `LLMBuilder`
3. **Complete Pipeline**: GGUF/ONNX → IR → Optimization → Quantization → DLC
4. **Quantization Control**: Full control over INT4/8/16 and FP16
5. **Graph Optimizations**: Fusion, layout transforms, dead code elimination
6. **Custom Ops**: Support for user-defined operations
7. **LLM-Optimized**: Specialized handling for Large Language Models

### ⚠️ Limitations

1. **Context Binary**: Requires `qnn-context-binary-generator.exe` (C++ tool)
2. **Model Library**: Needs `qnn-model-lib-generator` for .so generation
3. **Platform**: Quantization tools currently x86_64 only (ARM64 has issues)
4. **Device-Specific**: Context binaries are device-specific

### 🎯 Recommendations

1. **🏆 Prioritize Native GGUF**: Use `LLMBuilder` for direct GGUF→DLC conversion
2. **Use Direct Python API**: Avoid subprocess calls where possible
3. **Implement Caching**: Cache DLC and context binaries
4. **Profile Everything**: Use TEZ-159 profiling integration
5. **Hierarchical Preservation**: Maintain modelexport's hierarchy through compilation
6. **Incremental Development**: Start with GGUF support, add ONNX, then context generation
7. **LLM Focus**: Leverage QNN's specialized LLM optimizations via `LLMBuilder`

## Next Steps

1. **🚨 HIGH PRIORITY**: Implement native GGUF → DLC converter using `LLMBuilder`
2. Implement basic ONNX → DLC converter using Python SDK
3. Add quantization support with calibration
4. Integrate context binary generation
5. Create modelexport CLI commands (`modelexport compile-qnn`)
6. Add comprehensive testing
7. Document best practices and GGUF optimization strategies

## References

- ONNX Runtime QNN EP: `onnxruntime/core/providers/qnn/`
- QNN SDK Python: `qti.aisw.converters`
- QNN Tools: `qnn-onnx-converter`, `qnn-context-binary-generator`
- Quantization: `qti.aisw.converters.backend.qnn_quantizer`