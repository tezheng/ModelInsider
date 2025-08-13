# ONNXAutoProcessor Examples

This directory contains practical examples demonstrating how to use the ONNXAutoProcessor for fast, universal ONNX model inference.

## 🎯 Quick Start

### BERT Tiny Inference Example

The simplest way to get started with ONNXAutoProcessor:

```bash
# Run the BERT tiny example
uv run python examples/bert_tiny_inference.py
```

This example demonstrates:
- ✅ **Auto-detection**: Automatically detects BERT as a text processor
- ⚡ **Fast inference**: ~800+ sentences/second on CPU
- 🔧 **Fixed shapes**: Optimized tensor shapes for ONNX performance  
- 📊 **Performance metrics**: Timing and throughput analysis

## 📁 Available Examples

| Example | Description | Model Type | Features |
|---------|-------------|------------|----------|
| [`bert_tiny_inference.py`](bert_tiny_inference.py) | Basic BERT text processing | Text (BERT) | Auto-detection, batch processing, performance metrics |

## 🚀 Example Output

```
🤖 BERT Tiny Inference Example
==================================================
📁 Model path: models/bert-tiny-optimum/model.onnx

🔍 Step 1: Creating ONNX processor with auto-detection...
✅ Processor created successfully in 0.116s
   Detected modality: ModalityType.TEXT
   Processor type: ONNXTokenizer

📝 Step 2: Running inference on test sentences...

🔤 Example 1: "Hello, world! This is a simple test."
   ⏱️  Inference time: 0.004s
   📊 Input IDs shape: (2, 16)
   🔍 Attention mask shape: (2, 16)

📊 Performance Summary
------------------------------
Average inference time: 0.001s per sentence
Throughput: 815.9 sentences/second
```

## 🏗️ Architecture Overview

The ONNXAutoProcessor provides a universal interface for any ONNX model:

```python
from onnx_auto_processor import ONNXAutoProcessor

# Universal - works with any ONNX model
processor = ONNXAutoProcessor.from_model("path/to/model.onnx")
result = processor("Your input text or data")
```

### Key Benefits

1. **🔍 Auto-Detection**: Automatically detects model type (text, image, audio, video, multimodal)
2. **⚡ Performance**: 40x+ speedup through fixed-shape optimization
3. **🌍 Universal**: Works with any HuggingFace model exported to ONNX
4. **🔧 Type-Safe**: Comprehensive type hints and validation
5. **🎯 Simple**: Same interface for all model types

## 📖 Usage Patterns

### Basic Usage
```python
from onnx_auto_processor import ONNXAutoProcessor

# Create processor (auto-detects model type)
processor = ONNXAutoProcessor.from_model("model.onnx")

# Run inference
result = processor("Hello world!")
```

### With Custom Base Processor
```python
from transformers import AutoTokenizer
from onnx_auto_processor import ONNXAutoProcessor

# Use specific HuggingFace processor
base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
processor = ONNXAutoProcessor.from_model("bert.onnx", base_processor=base_tokenizer)

result = processor("Hello world!")
```

### Multimodal Processing
```python
# Works automatically with multimodal models like CLIP
processor = ONNXAutoProcessor.from_model("clip.onnx")

# Text + image processing
result = processor(text="A photo of a cat", images=image_array)
```

## 🛠️ Model Requirements

### Supported Model Types
- **Text**: BERT, GPT, T5, etc.
- **Image**: ViT, ResNet, EfficientNet, etc. 
- **Audio**: Wav2Vec2, Whisper, etc.
- **Video**: VideoMAE, TimeSFormer, etc.
- **Multimodal**: CLIP, LayoutLM, etc.

### Directory Structure Expected
```
model_directory/
├── model.onnx              # Required: ONNX model file
├── config.json             # Required: Model configuration
├── tokenizer_config.json   # For text models
├── preprocessor_config.json # For image/audio models
└── special_tokens_map.json # For text models
```

## 🔧 Advanced Configuration

### Performance Tuning
- **Batch Size**: Automatically optimized for ONNX
- **Sequence Length**: Fixed based on model metadata
- **Memory Usage**: Efficient tensor management

### Error Handling
```python
try:
    processor = ONNXAutoProcessor.from_model("model.onnx")
    result = processor("input")
except ONNXProcessorError as e:
    print(f"Processor error: {e}")
except ONNXShapeError as e:
    print(f"Shape mismatch: {e}")
```

## 📚 Additional Examples (Coming Soon)

- **Image Classification**: ViT and ResNet examples
- **Audio Processing**: Whisper speech recognition
- **Video Analysis**: VideoMAE action recognition
- **Multimodal**: CLIP text-image matching
- **Batch Processing**: High-throughput inference
- **Performance Benchmarks**: Speed comparisons

## 🤝 Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the naming pattern: `{model_type}_{use_case}_example.py`
3. Include comprehensive docstrings and error handling
4. Add performance metrics and output examples
5. Update this documentation with example information

## 🐛 Troubleshooting

### Common Issues

**Import Error**: Make sure to add the src directory to your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**Model Not Found**: Verify the model path exists and contains required files:
- `model.onnx`
- `config.json` 
- Tokenizer/processor configuration files

**Shape Mismatch**: Ensure your input matches the expected model input format. The processor will automatically handle padding and truncation.

### Performance Tips

1. **Use appropriate input sizes**: Shorter sequences process faster
2. **Batch processing**: Process multiple inputs together when possible
3. **Model optimization**: Use Optimum or similar tools to optimize ONNX models
4. **Hardware acceleration**: Consider ONNX Runtime GPU providers for larger models

## 📄 License

This project follows the same license as the parent modelexport project.