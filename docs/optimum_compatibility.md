# Optimum Compatibility Guide

## Overview

The HTP exporter now supports automatic generation of configuration files needed for using exported ONNX models with HuggingFace Optimum. This enables seamless integration with Optimum's ONNX Runtime models for high-performance inference.

## What is Generated

When exporting a model, the HTP exporter can generate:

1. **config.json** - Model architecture configuration (PretrainedConfig)
2. **Tokenizer files** - All necessary tokenizer files for text processing

## Usage

### Basic Export (with Optimum compatibility)

By default, the exporter generates all necessary files:

```bash
modelexport export --model prajjwal1/bert-tiny --output models/bert.onnx
```

This creates:
```
models/
├── bert.onnx                 # ONNX model with hierarchy tags
├── config.json               # Model configuration for Optimum
├── tokenizer.json           # Fast tokenizer
├── tokenizer_config.json    # Tokenizer configuration
├── special_tokens_map.json  # Special tokens
└── vocab.txt                # Vocabulary (BERT models)
```

### Export without Config Files

To export only the ONNX model:

```bash
modelexport export --model prajjwal1/bert-tiny --output bert.onnx --no-config --no-tokenizer
```

### Export with Config but No Tokenizer

```bash
modelexport export --model prajjwal1/bert-tiny --output bert.onnx --no-tokenizer
```

## Using Exported Models with Optimum

### Loading for Question Answering

```python
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer

# Load model and tokenizer from export directory
model = ORTModelForQuestionAnswering.from_pretrained("models/")
tokenizer = AutoTokenizer.from_pretrained("models/")

# Use for inference
question = "What is ONNX?"
context = "ONNX is an open format for machine learning models."
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```

### Loading for Feature Extraction

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained("models/")
tokenizer = AutoTokenizer.from_pretrained("models/")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

### Using with Pipelines

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained("models/")
tokenizer = AutoTokenizer.from_pretrained("models/")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("I love using ONNX models!")
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--no-config` | Disable config.json generation | False (config is generated) |
| `--no-tokenizer` | Disable tokenizer files generation | False (tokenizer is saved) |

## Python API

### Using HTPExporter with Config Control

```python
from modelexport.strategies.htp import HTPExporter

# Export with full Optimum compatibility
exporter = HTPExporter(
    save_config=True,      # Generate config.json
    save_tokenizer=True    # Save tokenizer files
)

exporter.export(
    model_name_or_path="bert-base-uncased",
    output_path="models/bert.onnx"
)
```

### Disabling Config Generation

```python
exporter = HTPExporter(
    save_config=False,     # Don't generate config.json
    save_tokenizer=False   # Don't save tokenizer
)
```

## Technical Details

### What is config.json?

The `config.json` file contains the model's architectural configuration (PretrainedConfig), including:
- Model type and architecture
- Hidden dimensions and layer counts
- Vocabulary size
- Special token IDs
- Task-specific configurations

This is **different** from OnnxConfig, which is an export configuration used only during ONNX conversion.

### Required Files for Different Tasks

| Task | Required Files | Optional Files |
|------|---------------|----------------|
| Feature Extraction | model.onnx, config.json | tokenizer files |
| Question Answering | model.onnx, config.json | tokenizer files |
| Sequence Classification | model.onnx, config.json | tokenizer files |
| Token Classification | model.onnx, config.json | tokenizer files |

### Compatibility Notes

1. **No Quantization Required**: Models work with Optimum without quantization
2. **Hierarchy Tags Preserved**: The ONNX model retains HTP hierarchy tags
3. **Full Optimum Support**: Works with all ORTModel classes
4. **Pipeline Compatible**: Supports HuggingFace pipeline API

## Troubleshooting

### Config Generation Fails

If config generation fails, the export continues and produces the ONNX model. You can:
1. Manually create config.json from the original model
2. Use `--no-config` to skip config generation
3. Check that the model name is correct and accessible

### Tokenizer Not Found

Some models may not have tokenizers. In this case:
1. Only config.json is generated
2. Use `--no-tokenizer` to suppress warnings
3. Load tokenizer separately if needed

### Custom Models

For custom models not on HuggingFace:
1. Export with `--no-config --no-tokenizer`
2. Manually create config.json if needed for Optimum
3. Use the model directly with ONNX Runtime if Optimum is not required

## Benefits

1. **Zero-Friction Deployment**: Exported models work immediately with Optimum
2. **Production Ready**: No additional steps needed for inference
3. **Maintains Hierarchy**: HTP tags preserved for debugging and analysis
4. **Flexible Control**: Choose what files to generate
5. **Graceful Degradation**: Export continues even if config generation fails

## See Also

- [HuggingFace Optimum Documentation](https://huggingface.co/docs/optimum)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [HTP Export Documentation](./htp_export.md)