# Feature Engineering Metadata Design

## Overview

This document outlines the approach for preserving and embedding original HuggingFace configurations into ONNX models, with minimal transformation.

## Core Philosophy

**Complete Preservation Principle**: We honor ALL original configuration fields exactly as provided by HuggingFace, trusting that every field exists for a reason. No filtering, no selection of "essential" fields - complete preservation.

**ALL Feature Engineering Metadata Must Be Handled**: This includes:
- Complete tokenizer configurations
- Complete vocabulary (when --include-vocab is specified)
- Complete image processor settings
- Complete feature extractor parameters
- Any other processor-specific data

Without complete FE metadata, the model cannot perform inference correctly.

## Our Approach

### Phase 2: Direct Configuration Preservation

Store original HuggingFace configurations directly in ONNX metadata properties:

```python
def embed_feature_engineering_metadata(onnx_model, processor):
    """Embed original processor configuration in ONNX model."""
    
    # Clear existing metadata_props
    del onnx_model.metadata_props[:]
    
    # Directly preserve original configurations
    metadata = {}
    
    # Preserve tokenizer config as-is
    if hasattr(processor, "tokenizer"):
        tokenizer_config = processor.tokenizer.to_dict()
        metadata["hf_tokenizer_config"] = json.dumps(tokenizer_config)
        metadata["feature_engineering.type"] = "tokenizer"
        
    # Preserve image processor config as-is  
    elif hasattr(processor, "image_processor"):
        image_config = processor.image_processor.to_dict()
        metadata["hf_image_processor_config"] = json.dumps(image_config)
        metadata["feature_engineering.type"] = "image_processor"
        
    # Preserve feature extractor config as-is
    elif hasattr(processor, "feature_extractor"):
        audio_config = processor.feature_extractor.to_dict()
        metadata["hf_feature_extractor_config"] = json.dumps(audio_config)
        metadata["feature_engineering.type"] = "feature_extractor"
    
    # Add to ONNX model
    for key, value in metadata.items():
        prop = onnx_model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
```

### Metadata Namespace Convention

We use minimal prefixed keys for organization:

- `hf.*` - Original HuggingFace configs (preserved as-is)
- `feature_engineering.type` - Simple type indicator (tokenizer/image_processor/feature_extractor)
- `export.*` - Export metadata (timestamp, version)

### Reading and Verification

```python
def read_feature_engineering_metadata(onnx_model_path):
    """Read preserved configurations from ONNX model."""
    model = onnx.load(onnx_model_path)
    
    metadata = {}
    
    for prop in model.metadata_props:
        if prop.key.startswith("hf_"):
            # These are original HF configs - parse as JSON
            if "config" in prop.key:
                metadata[prop.key] = json.loads(prop.value)
            else:
                metadata[prop.key] = prop.value
        elif prop.key == "feature_engineering.type":
            metadata[prop.key] = prop.value
    
    return metadata

def verify_metadata(metadata):
    """Verify that essential fields are present."""
    # Just check that we have the type and corresponding config
    fe_type = metadata.get("feature_engineering.type")
    
    if fe_type == "tokenizer":
        assert "hf_tokenizer_config" in metadata
    elif fe_type == "image_processor":
        assert "hf_image_processor_config" in metadata
    elif fe_type == "feature_extractor":
        assert "hf_feature_extractor_config" in metadata
    
    return True
```

## What We Store (Examples)

### 1. Text Models - Original Tokenizer Config

```json
{
  "hf_tokenizer_config": {
    // Exact original config from tokenizer.to_dict()
    "tokenizer_class": "BertTokenizerFast",
    "vocab_size": 30522,
    "do_lower_case": true,
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "model_max_length": 512,
    // ... all other original fields preserved
  },
  "feature_engineering.type": "tokenizer"
}
```

### 2. Vision Models - Original Image Processor Config

```json
{
  "hf_image_processor_config": {
    // Exact original config from image_processor.to_dict()
    "image_processor_type": "ViTImageProcessor",
    "do_resize": true,
    "size": {"height": 224, "width": 224},
    "resample": 3,
    "do_center_crop": true,
    "crop_size": {"height": 224, "width": 224},
    "do_normalize": true,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    // ... all other original fields preserved
  },
  "feature_engineering.type": "image_processor"
}
```

### 3. Audio Models - Original Feature Extractor Config

```json
{
  "hf_feature_extractor_config": {
    // Exact original config from feature_extractor.to_dict()
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 80,
    "sampling_rate": 16000,
    "hop_length": 160,
    "chunk_length": 30,
    "n_fft": 400,
    "n_samples": 480000,
    // ... all other original fields preserved
  },
  "feature_engineering.type": "feature_extractor"
}
```

## Processor Reconstruction

```python
def create_processor_from_metadata(metadata):
    """Attempt to recreate processor from preserved configs."""
    fe_type = metadata.get("feature_engineering.type")
    
    if fe_type == "tokenizer":
        config = metadata.get("hf_tokenizer_config", {})
        # Try to reconstruct from preserved config
        # Note: May need vocab files for full reconstruction
        return create_tokenizer_proxy(config)
        
    elif fe_type == "image_processor":
        config = metadata.get("hf_image_processor_config", {})
        # Reconstruct from preserved config
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_dict(config)
        
    elif fe_type == "feature_extractor":
        config = metadata.get("hf_feature_extractor_config", {})
        # Reconstruct from preserved config
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_dict(config)
```

## CLI Integration

```bash
# Embed original configs (preserved as-is)
modelexport embed-metadata model.onnx bert-base-uncased

# The tool will:
# 1. Load the HF model/processor
# 2. Call .to_dict() to get original configs
# 3. Store them directly in ONNX with minimal namespace prefixes
```

## Size Considerations

Since we're storing complete original configs:
- Tokenizer config: 2-5 KB (without vocab)
- Image processor config: 1-3 KB  
- Feature extractor config: 1-3 KB
- Model config: 2-10 KB
- Total typical overhead: 5-20 KB (negligible vs model weights)
- With vocab (optional): 200-500 KB

## Key Principles

1. **Complete Preservation**: Store ALL fields from original HF configs
2. **No Filtering**: Never judge which fields are "essential" - preserve everything
3. **Trust Original Design**: Every field exists for a reason, honor them all
4. **Direct Storage**: Use .to_dict() and store the complete result
5. **Minimal Namespacing**: Only add prefixes for organization (hf_*, feature_engineering.type)
6. **No Custom Schemas**: We don't define our own schemas, we preserve HF's completely

## Benefits of This Approach

- **Simplicity**: No complex transformation logic
- **Compatibility**: Works with any HF version's config format
- **Maintainability**: No need to update when HF adds new config fields
- **Transparency**: What you store is exactly what HF provides
- **Future-proof**: New model types work automatically

## Phase 3 Enhancement (Future)

In the future, we could embed preprocessing directly in the ONNX graph using ONNX Runtime Extensions, but for Phase 2, we simply preserve and embed the original configurations.