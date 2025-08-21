# DeepSeek GGUF to QNN Context Binary - Execution Guide

## Prerequisites

### 1. QNN SDK Installation
```bash
# Verify QNN SDK is installed
export QNN_SDK_ROOT=/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/
ls -la $QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter*
```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy
pip install huggingface-hub  # Optional, for model download
```

## Step-by-Step Execution

### Option 1: Automated Full Pipeline

```bash
# Navigate to conversion directory
cd /home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/deepseek_conversion

# Run complete conversion with download
python convert_deepseek_to_qnn.py \
    --download \
    --qnn-sdk-root /mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/ \
    --output-dir ./output \
    --target-device snapdragon_8_gen3
```

### Option 2: Step-by-Step Manual Process

#### Step 1: Download the Model
```bash
# Create models directory
mkdir -p models

# Download using wget
wget -O models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf \
    https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf

# Verify download (should be ~1.1GB)
ls -lh models/
```

#### Step 2: Run Native GGUF Conversion
```bash
# Set SDK path
export QNN_SDK_ROOT=/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/

# Convert GGUF to DLC using native support
python $QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter \
    --input_network models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf \
    --output_path output/deepseek_qwen_1.5b.dlc \
    --input_layout "input_ids,NONTRIVIAL" \
    --input_layout "attention_mask,NONTRIVIAL" \
    --preserve_io datatype,input_ids,attention_mask \
    --float_fallback \
    --float_bitwidth 16
```

Expected output:
```
[INFO] Detected GGUF format
[INFO] Using LLMBuilder for native GGUF processing
[INFO] Parsing GGUF metadata...
[INFO] Architecture: Qwen
[INFO] Quantization: Q4_0
[INFO] Generating ONNX graph internally...
[INFO] Applying LLM optimizations...
[INFO] Converting to QNN IR format...
[INFO] Generating DLC...
[SUCCESS] Output saved to: output/deepseek_qwen_1.5b.dlc
```

#### Step 3: Generate Context Binary
```bash
# Generate context binary for NPU deployment
$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-context-binary-generator \
    --dlc_path output/deepseek_qwen_1.5b.dlc \
    --backend $QNN_SDK_ROOT/lib/x86_64-windows-msvc/libQnnHtp.dll \
    --binary_file output/deepseek_qwen_1.5b_ctx.bin \
    --output_dir output/ \
    --target_arch sm8650  # Snapdragon 8 Gen 3
```

Expected output:
```
[INFO] Loading DLC...
[INFO] Initializing HTP backend...
[INFO] Compiling for sm8650 (Snapdragon 8 Gen 3)...
[INFO] Generating context binary...
[SUCCESS] Context binary saved to: output/deepseek_qwen_1.5b_ctx.bin
```

### Option 3: Python API Usage

```python
#!/usr/bin/env python3
import sys
sys.path.append('../src')

from qnn_compiler import QNNCompiler, CompilationConfig, Backend

# Initialize compiler with QNN SDK
compiler = QNNCompiler(
    qnn_sdk_root="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/"
)

# Configure for DeepSeek model
config = CompilationConfig(
    backend=Backend.HTP,
    output_format="context-binary",
    batch_size=1,
    htp_performance_mode="high_performance",
    htp_precision="mixed",  # Important for Q4_0
    target_arch="sm8650"  # Snapdragon 8 Gen 3
)

# Use native GGUF compilation
output = compiler.compile_gguf(
    "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
    "output/deepseek_qwen_1.5b.bin",
    compilation=config
)

print(f"✅ Context binary ready: {output}")
```

## Verification Steps

### 1. Check Output Files
```bash
# List generated files
ls -lh output/

# Expected files:
# deepseek_qwen_1.5b.dlc        (~1.3GB) - QNN model
# deepseek_qwen_1.5b_ctx.bin    (~1.2GB) - Context binary
# deepseek_qwen_1.5b.params     (metadata file)
```

### 2. Inspect DLC Metadata
```bash
# Use QNN tools to inspect the DLC
$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-dlc-info \
    --input_dlc output/deepseek_qwen_1.5b.dlc
```

Expected output:
```
Model Information:
  Name: DeepSeek-R1-Distill-Qwen-1.5B
  Version: 1.0.0
  Architecture: Qwen
  Parameters: 1.5B
  
Quantization:
  Weights: Mixed (Q4_0 preserved)
  Activations: FP16
  
Operators: 248 total
  Supported on HTP: 241 (97%)
  CPU fallback: 7 (3%)
```

### 3. Validate Context Binary
```bash
# Validate context binary
$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-context-binary-info \
    --binary_file output/deepseek_qwen_1.5b_ctx.bin
```

## Deployment Testing

### Test on Windows ARM64 Device
```python
#!/usr/bin/env python3
"""test_inference.py - Test DeepSeek model on NPU"""

import onnxruntime as ort
import numpy as np

# Create session with QNN EP
providers = [
    ('QNNExecutionProvider', {
        'backend_path': 'libQnnHtp.dll',
        'context_cache_path': 'output/deepseek_qwen_1.5b_ctx.bin',
        'htp_performance_mode': 'high_performance',
        'rpc_control_latency': '100'
    })
]

session = ort.InferenceSession(
    'output/deepseek_qwen_1.5b.onnx',  # If ONNX was kept
    providers=providers
)

# Prepare input
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

# Run inference
outputs = session.run(
    None,
    {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
)

print(f"✅ Inference successful!")
print(f"Output shape: {outputs[0].shape}")
```

## Performance Benchmarking

### Expected Performance Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Model Load Time | <2s | TBD | ⏳ |
| First Token Latency | <100ms | TBD | ⏳ |
| Per-Token Latency | <50ms | TBD | ⏳ |
| Throughput | >20 tok/s | TBD | ⏳ |
| Memory Usage | <2GB | TBD | ⏳ |
| NPU Utilization | >80% | TBD | ⏳ |
| Power Consumption | <5W | TBD | ⏳ |

### Benchmark Script
```bash
# Run performance benchmark
python benchmark_npu.py \
    --model output/deepseek_qwen_1.5b_ctx.bin \
    --batch-size 1 \
    --sequence-length 512 \
    --iterations 100
```

## Troubleshooting

### Issue 1: GGUF Not Recognized
```
Error: Invalid model format specified
```
**Solution**: Ensure QNN SDK version is 2.34+ with LLMBuilder support

### Issue 2: Q4_0 Quantization Error
```
Error: Unsupported quantization scheme Q4_0
```
**Solution**: Use `--float_fallback` flag to handle Q4_0

### Issue 3: Operator Not Supported
```
Warning: Operator 'RoPE' not supported on HTP
```
**Solution**: Enable `--enable_cpu_fallback` for unsupported ops

### Issue 4: Memory Allocation Failed
```
Error: Failed to allocate 4GB for weights
```
**Solution**: Use streaming mode or reduce batch size

## Success Criteria Checklist

- [ ] ✅ Model downloaded successfully (~1.1GB)
- [ ] ✅ GGUF metadata extracted correctly
- [ ] ✅ Native GGUF compilation completed
- [ ] ✅ DLC generated (~1.3GB)
- [ ] ✅ Context binary created (~1.2GB)
- [ ] ✅ Inference runs on NPU
- [ ] ✅ Performance meets targets
- [ ] ✅ Memory usage within limits
- [ ] ✅ Power consumption acceptable

## Summary

This guide demonstrates the complete workflow for converting DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf to a QNN context binary ready for deployment on Qualcomm Snapdragon NPU. The native GGUF support in QNN SDK significantly simplifies the process, eliminating manual conversion steps while preserving the original Q4_0 quantization for optimal performance.