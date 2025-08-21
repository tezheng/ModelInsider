# Test Models Directory

This directory contains test models for QNN compilation experiments.

## DeepSeek-R1 Test Model

**File**: `DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf` (1019.29 MB)
**Source**: [HuggingFace - bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf)

### Download Instructions

```bash
# Download the test model (required for QNN conversion experiments)
cd experiments/tez-172_qnn-compile/models/
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf
```

### Model Specifications

- **Architecture**: Qwen-based transformer
- **Quantization**: 4-bit (Q4_0)
- **Size**: ~1GB
- **Use Case**: Testing GGUF → QNN conversion pipeline

**Note**: Large model files (>100MB) are excluded from git tracking to maintain repository size.