# ONNX Metadata Schema Specification

## Overview

This document defines the approach for preserving original HuggingFace configurations in ONNX files and the schema registry for validation. We preserve configs as-is without modification, using schemas only for validation purposes.

## Metadata Version

Current Version: **2.0**

## Core Principle

**Preserve Original Configs**: We store the complete, unmodified HuggingFace configurations directly in ONNX metadata. Schemas are maintained separately for validation only, ensuring we can verify the integrity of exported data without altering it.

## Metadata Structure

### Root Level Properties

All metadata properties are stored in ONNX model's `metadata_props` field with the prefix `hf_` (HuggingFace).

| Property Key | Type | Required | Description |
|-------------|------|----------|-------------|
| `hf_metadata_version` | string | Yes | Schema version (e.g., "2.0") |
| `hf_export_timestamp` | string | Yes | ISO 8601 timestamp of export |
| `hf_exporter_version` | string | Yes | Version of modelexport used |
| `hf_model_config` | JSON string | Yes | Model configuration |
| `hf_processor_config` | JSON string | Yes | Processor configuration |
| `hf_pipeline_task` | string | Yes | Pipeline task identifier |
| `hf_pipeline_config` | JSON string | No | Pipeline-specific settings |
| `hf_export_config` | JSON string | No | Export settings used |

### Stored Configurations

The `hf_model_config` contains the **complete, unmodified** model configuration from HuggingFace:

```python
# What we store (complete original config)
model_config = model.config.to_dict()
onnx_metadata["hf_model_config"] = json.dumps(model_config)

# The config includes ALL original fields, for example:
# - All architecture parameters
# - All training hyperparameters  
# - All custom fields added by model authors
# - Version information
# - Any model-specific attributes
```

### Schema Registry (Validation Only)

We maintain schemas to validate configs without modifying them. These schemas document expected fields for each model type:

#### BERT Config Schema (for validation)

```json
{
  "type": "object",
  "required": ["model_type", "vocab_size", "hidden_size"],
  "properties": {
    "model_type": {"const": "bert"},
    "architectures": {"type": "array"},
    "vocab_size": {"type": "integer", "minimum": 1},
    "hidden_size": {"type": "integer", "minimum": 1},
    "num_hidden_layers": {"type": "integer", "minimum": 1},
    "num_attention_heads": {"type": "integer", "minimum": 1},
    "intermediate_size": {"type": "integer", "minimum": 1},
    "hidden_act": {"type": "string"},
    "hidden_dropout_prob": {"type": "number"},
    "attention_probs_dropout_prob": {"type": "number"},
    "max_position_embeddings": {"type": "integer"},
    "type_vocab_size": {"type": "integer"},
    "initializer_range": {"type": "number"},
    "layer_norm_eps": {"type": "number"},
    "pad_token_id": {"type": "integer"},
    "position_embedding_type": {"type": "string"},
    "use_cache": {"type": "boolean"},
    "classifier_dropout": {"type": ["number", "null"]}
  },
  "additionalProperties": true  // Allow any additional fields
}
```

**Important**: `additionalProperties: true` ensures we don't reject configs with extra fields.

### Processor Configuration Storage

Similar to model configs, we store **complete, unmodified** processor configurations:

```python
# Store original tokenizer config
if hasattr(processor, "tokenizer"):
    tokenizer_config = processor.tokenizer.to_dict()
    onnx_metadata["hf_tokenizer_config"] = json.dumps(tokenizer_config)

# Store original image processor config  
if hasattr(processor, "image_processor"):
    image_config = processor.image_processor.to_dict()
    onnx_metadata["hf_preprocessor_config"] = json.dumps(image_config)

# Store original feature extractor config
if hasattr(processor, "feature_extractor"):
    feature_config = processor.feature_extractor.to_dict()
    onnx_metadata["hf_feature_extractor_config"] = json.dumps(feature_config)
```

### Processor Schema Registry (Validation Only)

#### Text Processor (Tokenizer) Validation Schema

```json
{
  "processor_type": "AutoTokenizer",
  "tokenizer_class": "string",           // Tokenizer class name
  "vocab_file": "string",                // Vocabulary file path/name
  "merges_file": "string",               // Merges file (for BPE)
  
  // Tokenization parameters
  "do_lower_case": "boolean",           // Convert to lowercase
  "do_basic_tokenize": "boolean",       // Apply basic tokenization
  "never_split": ["string"],            // Tokens to never split
  "unk_token": "string",                 // Unknown token
  "sep_token": "string",                 // Separator token
  "pad_token": "string",                 // Padding token
  "cls_token": "string",                 // Classification token
  "mask_token": "string",                // Mask token
  "tokenize_chinese_chars": "boolean",  // Tokenize Chinese characters
  "strip_accents": "boolean",           // Strip accents
  
  // Processing parameters
  "max_length": "integer",               // Maximum sequence length
  "padding": "string",                   // Padding strategy
  "truncation": "boolean|string",       // Truncation strategy
  "return_tensors": "string",           // Output tensor format
  "model_max_length": "integer",        // Model's maximum length
  
  // Special tokens
  "special_tokens": {
    "unk_token": "string",
    "sep_token": "string",
    "pad_token": "string",
    "cls_token": "string",
    "mask_token": "string",
    "additional_special_tokens": ["string"]
  }
}
```

#### Vision Processor (Image Processor)

```json
{
  "processor_type": "AutoImageProcessor",
  "image_processor_type": "string",      // Processor class name
  
  // Image preprocessing
  "do_resize": "boolean",                // Resize images
  "size": "integer|object",              // Target size
  "resample": "integer",                 // Resampling filter
  "do_center_crop": "boolean",          // Center crop
  "crop_size": "integer|object",        // Crop size
  
  // Normalization
  "do_normalize": "boolean",            // Normalize pixel values
  "image_mean": ["float"],              // Normalization mean
  "image_std": ["float"],               // Normalization std
  "do_rescale": "boolean",              // Rescale pixel values
  "rescale_factor": "float",            // Rescale factor
  
  // Data format
  "image_format": "string",              // channels_first or channels_last
  "input_data_format": "string",        // Input format expectation
  "return_tensors": "string"            // Output tensor format
}
```

#### Audio Processor (Feature Extractor)

```json
{
  "processor_type": "AutoFeatureExtractor",
  "feature_extractor_type": "string",    // Extractor class name
  
  // Audio preprocessing
  "sampling_rate": "integer",            // Target sampling rate
  "padding_value": "float",              // Padding value
  "do_normalize": "boolean",            // Normalize audio
  "return_attention_mask": "boolean",   // Return attention mask
  
  // Feature extraction
  "feature_size": "integer",            // Feature dimension
  "num_mel_bins": "integer",            // Mel filterbank bins
  "hop_length": "integer",              // STFT hop length
  "win_length": "integer",              // STFT window length
  "win_function": "string",             // Window function
  "frame_length": "integer",            // Frame length
  "fft_length": "integer",              // FFT length
  
  // Output format
  "return_tensors": "string",           // Output tensor format
  "padding": "string"                   // Padding strategy
}
```

### Pipeline Task Identifiers

Standardized task identifiers for `hf_pipeline_task`:

#### Text Tasks
- `text-classification`
- `token-classification`
- `question-answering`
- `text-generation`
- `text2text-generation`
- `summarization`
- `translation`
- `fill-mask`
- `zero-shot-classification`
- `conversational`

#### Vision Tasks
- `image-classification`
- `image-segmentation`
- `semantic-segmentation`
- `instance-segmentation`
- `panoptic-segmentation`
- `object-detection`
- `image-to-image`
- `depth-estimation`

#### Audio Tasks
- `audio-classification`
- `automatic-speech-recognition`
- `audio-to-audio`
- `voice-activity-detection`
- `zero-shot-audio-classification`

#### Multimodal Tasks
- `image-to-text`
- `visual-question-answering`
- `document-question-answering`
- `zero-shot-image-classification`
- `video-classification`

### Pipeline Configuration Schema

Optional `hf_pipeline_config` for pipeline-specific settings:

```json
{
  "framework": "onnx",                   // Framework (always "onnx" for us)
  "device": "string",                    // Device type (cpu, cuda, etc.)
  "batch_size": "integer",              // Default batch size
  "num_threads": "integer",             // Number of threads
  "providers": ["string"],              // ONNX Runtime providers
  
  // Task-specific parameters
  "top_k": "integer",                   // For classification tasks
  "max_length": "integer",              // For generation tasks
  "min_length": "integer",              // For generation tasks
  "temperature": "float",               // For generation tasks
  "num_beams": "integer",               // For beam search
  
  // Performance settings
  "use_cache": "boolean",               // Enable caching
  "optimization_level": "integer",      // ONNX optimization level
  "quantization": "string"              // Quantization type if applied
}
```

### Export Configuration Schema

Optional `hf_export_config` documenting export settings:

```json
{
  "opset_version": "integer",           // ONNX opset version
  "optimization_level": "integer",      // Optimization level used
  "use_external_data": "boolean",      // External data storage
  "external_data_format": "string",    // Format of external data
  
  // HTP specific
  "hierarchy_tags_enabled": "boolean",  // Hierarchy preservation enabled
  "clean_onnx": "boolean",             // Clean ONNX export
  
  // Quantization
  "quantization_type": "string",       // Quantization method
  "quantization_bits": "integer",      // Quantization bits
  
  // Model source
  "source_model": "string",            // Original model name/path
  "source_revision": "string",         // Model revision/commit
  "source_framework": "string"         // Source framework (pytorch, tensorflow)
}
```

## Example Metadata

### BERT Text Classification (Actual Preserved Config)

```json
{
  "hf_metadata_version": "2.0",
  "hf_export_timestamp": "2024-01-15T10:30:00Z",
  "hf_exporter_version": "modelexport-1.2.0",
  "hf_transformers_version": "4.36.0",
  "hf_pipeline_task": "text-classification",
  
  "hf_model_config": {
    "_name_or_path": "bert-base-uncased",
    "architectures": ["BertForSequenceClassification"],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "gradient_checkpointing": false,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "id2label": {"0": "NEGATIVE", "1": "POSITIVE"},
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "label2id": {"NEGATIVE": 0, "POSITIVE": 1},
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "problem_type": "single_label_classification",
    "torch_dtype": "float32",
    "transformers_version": "4.36.0",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
  },
  
  "hf_tokenizer_config": {
    "clean_up_tokenization_spaces": true,
    "cls_token": "[CLS]",
    "do_basic_tokenize": true,
    "do_lower_case": true,
    "mask_token": "[MASK]",
    "max_length": 512,
    "model_max_length": 512,
    "never_split": null,
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "strip_accents": null,
    "tokenize_chinese_chars": true,
    "tokenizer_class": "BertTokenizer",
    "unk_token": "[UNK]"
  }
}
```

Note: This is the **actual, complete config** as stored by HuggingFace, not a filtered subset.

### Vision Transformer Image Classification

```json
{
  "hf_metadata_version": "2.0",
  "hf_export_timestamp": "2024-01-15T11:00:00Z",
  "hf_exporter_version": "modelexport-1.2.0",
  "hf_pipeline_task": "image-classification",
  
  "hf_model_config": {
    "architectures": ["ViTForImageClassification"],
    "model_type": "vit",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "num_labels": 1000,
    "id2label": {"0": "cat", "1": "dog", ...}
  },
  
  "hf_processor_config": {
    "processor_type": "AutoImageProcessor",
    "image_processor_type": "ViTImageProcessor",
    "do_resize": true,
    "size": {"height": 224, "width": 224},
    "do_normalize": true,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225]
  }
}
```

## Metadata Size Guidelines

### Size Targets
- **Small Models**: < 10KB metadata
- **Medium Models**: < 50KB metadata
- **Large Models**: < 100KB metadata
- **Warning Threshold**: 100KB
- **Hard Limit**: 1MB

### Compression Strategy

For metadata > 100KB:
1. Enable gzip compression
2. Store compressed JSON with `hf_metadata_compressed: true`
3. Use base64 encoding for compressed data

```python
import gzip
import base64
import json

def compress_metadata(metadata: dict) -> str:
    json_str = json.dumps(metadata, separators=(',', ':'))
    if len(json_str) > 100000:  # 100KB threshold
        compressed = gzip.compress(json_str.encode())
        return base64.b64encode(compressed).decode()
    return json_str
```

## Version History

### Version 2.0 (Current)
- Initial schema definition
- Support for text, vision, audio tasks
- Comprehensive processor configuration
- Pipeline configuration options

### Future Versions (Planned)
- Version 2.1: Multi-modal processor support
- Version 2.2: Graph neural network support
- Version 2.3: Custom architecture extensions

## Validation

### Schema Validation

Use JSON Schema for validation:

```python
import jsonschema

METADATA_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["metadata_version", "model_config", "processor_config", "pipeline_task"],
    "properties": {
        "metadata_version": {"type": "string", "pattern": "^\\d+\\.\\d+$"},
        "model_config": {"type": "object"},
        "processor_config": {"type": "object"},
        "pipeline_task": {"type": "string"}
    }
}

def validate_metadata(metadata: dict):
    jsonschema.validate(metadata, METADATA_SCHEMA)
```

### Compatibility Check

```python
def check_compatibility(metadata_version: str) -> bool:
    """Check if metadata version is compatible."""
    major, minor = map(int, metadata_version.split('.'))
    current_major, current_minor = 2, 0
    
    # Major version must match
    if major != current_major:
        return False
    
    # Minor version can be <= current
    return minor <= current_minor
```

## Schema Registry Structure

The schema registry will be organized as follows:

```
modelexport/inference/schemas/
├── model_configs/
│   ├── bert.json         # BERT model config schema
│   ├── gpt2.json         # GPT-2 model config schema
│   ├── vit.json          # Vision Transformer schema
│   ├── whisper.json      # Whisper model schema
│   └── ...               # Other model schemas
├── processor_configs/
│   ├── tokenizers/
│   │   ├── bert_tokenizer.json
│   │   ├── gpt2_tokenizer.json
│   │   └── ...
│   ├── image_processors/
│   │   ├── vit_image_processor.json
│   │   ├── clip_image_processor.json
│   │   └── ...
│   └── feature_extractors/
│       ├── whisper_feature_extractor.json
│       └── ...
└── schema_registry.py    # Registry loader and validator
```

## Best Practices

1. **Preserve original configs** - Never modify or filter HuggingFace configs
2. **Use schemas for validation only** - Schemas validate but don't transform
3. **Handle unknown configs gracefully** - Store configs even without schemas
4. **Version tracking** - Include transformers version in metadata
5. **Size management** - Compress large configs if needed
6. **Backward compatibility** - Support older config versions
7. **Schema updates** - Regularly update schemas for new model types
8. **Validation logging** - Log validation warnings without failing export