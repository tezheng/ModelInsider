# Enhanced Processor Support for Optimum Compatibility

This experiment extends the HTP exporter's Optimum compatibility to support not just tokenizers, but all types of preprocessors used by different model modalities.

## 🎯 Overview

HuggingFace models use different preprocessors based on their modality:
- **Text Models** → `AutoTokenizer`
- **Vision Models** → `AutoImageProcessor` 
- **Multimodal Models** → `AutoProcessor`
- **Audio Models** → `AutoFeatureExtractor`

The enhanced `HTPConfigBuilder` now intelligently detects and saves the appropriate preprocessor type for full Optimum compatibility.

## 📁 Files in This Experiment

### Core Implementation
- `enhanced_config_builder.py` - Enhanced HTPConfigBuilder with multimodal support
  - `detect_preprocessor_type()` - Automatically detects the right preprocessor
  - `load_processor()` - Loads processor for multimodal models
  - `load_image_processor()` - Loads image processor for vision models
  - `load_feature_extractor()` - Loads feature extractor for audio models
  - `generate_optimum_config()` - Intelligently saves the right preprocessor

### Test Scripts
- `test_processor_support.py` - Tests preprocessor detection for various models
- `example_multimodal_export.py` - Complete example with text, vision, and multimodal models

## 🚀 Quick Start

### Test Preprocessor Detection

```bash
cd experiments/optimum-processor-support/
python test_processor_support.py
```

This will test detection for various model types:
```
Testing Preprocessor Detection
============================================================
✅ prajjwal1/bert-tiny                     -> tokenizer          (expected: tokenizer)
✅ gpt2                                     -> tokenizer          (expected: tokenizer)
✅ google/vit-base-patch16-224              -> image_processor    (expected: image_processor)
✅ openai/clip-vit-base-patch32             -> processor          (expected: processor)
✅ facebook/wav2vec2-base                   -> feature_extractor  (expected: feature_extractor)
```

### Run Complete Example

```bash
python example_multimodal_export.py
```

This demonstrates exporting different model types with their appropriate preprocessors.

## 💡 Key Features

### 1. Automatic Preprocessor Detection

The system automatically detects which type of preprocessor a model uses:

```python
from enhanced_config_builder import HTPConfigBuilder

# Text model
builder = HTPConfigBuilder("bert-base-uncased")
print(builder.detect_preprocessor_type())  # "tokenizer"

# Vision model
builder = HTPConfigBuilder("google/vit-base-patch16-224")
print(builder.detect_preprocessor_type())  # "image_processor"

# Multimodal model
builder = HTPConfigBuilder("openai/clip-vit-base-patch32")
print(builder.detect_preprocessor_type())  # "processor"

# Audio model
builder = HTPConfigBuilder("facebook/wav2vec2-base")
print(builder.detect_preprocessor_type())  # "feature_extractor"
```

### 2. Unified Config Generation

One method handles all preprocessor types:

```python
builder = HTPConfigBuilder(model_name)
results = builder.generate_optimum_config(
    output_dir="./output",
    save_preprocessor=True,  # Saves the appropriate preprocessor
)

# Results include:
# - config: True/False (config.json saved)
# - preprocessor: True/False (appropriate preprocessor saved)
# - preprocessor_type: "tokenizer", "processor", "image_processor", or "feature_extractor"
```

### 3. Backward Compatibility

For text models, the system maintains backward compatibility:
- Still saves tokenizer files as before
- Results include both `preprocessor` and `tokenizer` keys for text models

## 📊 Supported Model Types

### Text Models (Tokenizer)
- BERT, RoBERTa, GPT-2, T5, BART, etc.
- Files saved: `tokenizer_config.json`, `vocab.txt`/`vocab.json`, `special_tokens_map.json`

### Vision Models (Image Processor)
- ViT, ResNet, ConvNeXT, Swin, etc.
- Files saved: `preprocessor_config.json`

### Multimodal Models (Processor)
- CLIP, LayoutLM, ALIGN, etc.
- Files saved: `processor_config.json`, tokenizer files, image processor config

### Audio Models (Feature Extractor)
- Wav2Vec2, Whisper, HuBERT, etc.
- Files saved: `preprocessor_config.json`

## 🔧 Integration with Production Code

To integrate this enhanced support into production:

1. **Replace the existing HTPConfigBuilder**:
   ```bash
   cp enhanced_config_builder.py ../../modelexport/strategies/htp/config_builder.py
   ```

2. **Update HTPExporter** (already compatible):
   - The exporter already uses `generate_optimum_config()` which handles all preprocessor types

3. **Update CLI flags** (optional):
   - Consider renaming `--no-tokenizer` to `--no-preprocessor` for clarity
   - Add documentation about multimodal support

## 📝 Example Usage with Optimum

### Text Model (BERT)
```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained("./bert-export")
tokenizer = AutoTokenizer.from_pretrained("./bert-export")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

### Vision Model (ViT)
```python
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor
from PIL import Image

model = ORTModelForImageClassification.from_pretrained("./vit-export")
processor = AutoImageProcessor.from_pretrained("./vit-export")

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

### Multimodal Model (CLIP)
```python
from optimum.onnxruntime import ORTModel
from transformers import AutoProcessor
from PIL import Image

model = ORTModel.from_pretrained("./clip-export")
processor = AutoProcessor.from_pretrained("./clip-export")

image = Image.open("image.jpg")
inputs = processor(
    text=["a cat", "a dog"],
    images=image,
    return_tensors="pt",
    padding=True
)
outputs = model(**inputs)
```

## ✅ Benefits

1. **Universal Support**: Works with any HuggingFace model modality
2. **Automatic Detection**: No need to specify preprocessor type manually
3. **Full Optimum Compatibility**: Exported models work seamlessly with Optimum
4. **Backward Compatible**: Existing text model workflows continue to work
5. **Clean Design**: Single unified interface for all model types

## 🚦 Testing Checklist

- [x] Text models (BERT, GPT-2) - tokenizer detection and saving
- [x] Vision models (ViT, ResNet) - image processor detection and saving
- [x] Multimodal models (CLIP, LayoutLM) - processor detection and saving
- [x] Audio models (Wav2Vec2, Whisper) - feature extractor detection and saving
- [x] Backward compatibility with existing tokenizer-only code
- [x] Config generation with additional metadata
- [ ] Integration tests with actual Optimum loading
- [ ] Performance benchmarks

## 📚 References

- [HuggingFace Optimum Documentation](https://huggingface.co/docs/optimum)
- [AutoProcessor Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor)
- [ONNX Runtime Integration](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models)

## 🎯 Next Steps

1. **Merge to Production**: After testing, merge enhanced_config_builder.py to production
2. **Update CLI**: Consider updating CLI flags for clarity
3. **Add Integration Tests**: Create tests that actually load with Optimum
4. **Document in Main README**: Add multimodal examples to main documentation
5. **Consider Auto-Export**: Potentially auto-export vision/audio models with proper inputs