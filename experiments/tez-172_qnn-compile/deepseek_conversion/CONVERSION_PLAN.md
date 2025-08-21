# DeepSeek-R1-Distill-Qwen-1.5B GGUF to QNN Context Binary Conversion Plan

## Model Analysis

### Source Model
- **URL**: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
- **File**: DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf
- **Architecture**: Qwen (Alibaba's LLM architecture)
- **Size**: 1.5B parameters
- **Quantization**: Q4_0 (4-bit quantization, group size 32)
- **Use Case**: Distilled reasoning model from DeepSeek-R1

### Model Characteristics
- **Q4_0 Quantization Details**:
  - 4-bit weights per group of 32 values
  - Each group has a scaling factor (FP16)
  - ~75% size reduction vs FP16
  - Optimized for edge deployment
  
- **Qwen Architecture**:
  - RoPE positional encoding
  - SwiGLU activation
  - RMSNorm normalization
  - Multi-query attention (MQA) or Grouped-query attention (GQA)

## Conversion Strategy

### Phase 1: Model Download and Inspection
1. Download the GGUF model from HuggingFace
2. Inspect GGUF metadata using tools
3. Extract model configuration
4. Verify quantization scheme

### Phase 2: Native GGUF Compilation
Using QNN SDK's native GGUF support discovered earlier:
1. Use LLMBuilder to parse GGUF directly
2. Generate ONNX graph internally
3. Preserve Q4_0 quantization metadata
4. Apply Qwen-specific optimizations

### Phase 3: QNN DLC Generation
1. Convert to QNN IR format
2. Apply HTP backend optimizations
3. Configure for Snapdragon NPU
4. Generate optimized DLC

### Phase 4: Context Binary Creation
1. Generate model library (.so)
2. Create context binary for target device
3. Configure runtime settings
4. Package for deployment

## Technical Challenges

### Q4_0 Quantization Handling
- **Challenge**: QNN natively supports INT8/INT16, not 4-bit
- **Solution 1**: Use `--float_fallback` to dequantize to FP16
- **Solution 2**: Map Q4_0 to INT8 with adjusted scales
- **Solution 3**: Use native GGUF support with LLMBuilder

### Qwen Architecture Support
- **Challenge**: Qwen-specific operators (RoPE, SwiGLU)
- **Solution**: QNN SDK LLMBuilder handles common LLM architectures
- **Verification**: Check operator coverage in QNN

### Memory Optimization
- **Original**: ~1.1GB (Q4_0 GGUF)
- **Target**: <1.5GB context binary
- **Strategy**: Preserve quantization, optimize layouts

## Implementation Steps

### Step 1: Environment Setup
```bash
# Set QNN SDK path
export QNN_SDK_ROOT=/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/

# Create workspace
mkdir -p deepseek_conversion
cd deepseek_conversion
```

### Step 2: Download Model
```bash
# Using huggingface-cli
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
    DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf \
    --local-dir ./models

# Or using wget
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf
```

### Step 3: Native GGUF Compilation
```python
from qnn_compiler import QNNCompiler, CompilationConfig, Backend

# Initialize compiler
compiler = QNNCompiler(qnn_sdk_root="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/")

# Configure compilation for edge deployment
config = CompilationConfig(
    backend=Backend.HTP,  # Use NPU
    output_format="context-binary",  # Direct to context binary
    batch_size=1,
    htp_performance_mode="high_performance",
    htp_precision="mixed"  # Allow mixed precision for Q4_0
)

# Use native GGUF compilation
output = compiler.compile_gguf(
    "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
    "output/deepseek_qwen_1.5b.bin",
    compilation=config
)
```

### Step 4: Alternative CLI Approach
```bash
# Direct GGUF to DLC using qairt-converter
python $QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter \
    --input_network models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf \
    --output_path output/deepseek_qwen_1.5b.dlc \
    --input_layout "input_ids,NONTRIVIAL" \
    --input_layout "attention_mask,NONTRIVIAL" \
    --preserve_io datatype,input_ids,attention_mask

# Generate context binary
$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-context-binary-generator \
    --dlc_path output/deepseek_qwen_1.5b.dlc \
    --backend $QNN_SDK_ROOT/lib/x86_64-windows-msvc/libQnnHtp.dll \
    --binary_file output/deepseek_qwen_1.5b_ctx.bin \
    --output_dir output/
```

## Validation Strategy

### 1. Model Integrity
- Verify GGUF parsing succeeded
- Check ONNX graph generation
- Validate operator coverage

### 2. Quantization Preservation  
- Compare weight distributions
- Verify scaling factors
- Check activation ranges

### 3. Performance Testing
- Measure inference latency
- Check memory usage
- Validate accuracy vs original

### 4. Deployment Verification
- Test on target device
- Measure power consumption
- Validate real-time performance

## Expected Results

### Size Comparison
| Format | Size | Notes |
|--------|------|-------|
| Original GGUF | ~1.1GB | Q4_0 quantized |
| Generated ONNX | ~4.4GB | Dequantized for processing |
| QNN DLC | ~1.3GB | Optimized with metadata |
| Context Binary | ~1.2GB | Device-specific, ready to deploy |

### Performance Targets
- **Latency**: <50ms per token (batch=1)
- **Throughput**: >20 tokens/sec
- **Memory**: <2GB peak usage
- **Power**: <5W average on NPU

## Risk Mitigation

### Risk 1: Q4_0 Compatibility
- **Mitigation**: Use float_fallback mode if native Q4_0 fails
- **Fallback**: Convert to INT8 with recalibration

### Risk 2: Operator Support
- **Mitigation**: Check QNN operator coverage for Qwen
- **Fallback**: Use CPU fallback for unsupported ops

### Risk 3: Memory Constraints
- **Mitigation**: Stream weights, use memory mapping
- **Fallback**: Reduce batch size or sequence length

## Success Criteria

✅ Model successfully parsed from GGUF
✅ ONNX graph generated with proper structure
✅ QNN DLC compiled without errors
✅ Context binary generated for target device
✅ Inference runs successfully on NPU
✅ Performance meets targets
✅ Accuracy within 1% of original

## Next Steps

1. Execute download and inspection
2. Run native GGUF compilation
3. Test on Windows ARM64 device
4. Benchmark performance
5. Document findings
6. Optimize if needed