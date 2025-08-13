# Processor Metadata Requirements for ONNX Export

## Overview

This document lists all required configuration and metadata for each processor type to enable universal ONNX inference with fixed-shape optimization.

## Summary Table: All 5 Processor Types

| Processor Type | Auto Class | Purpose | Key Required Metadata | Fixed Shapes |
|----------------|------------|---------|----------------------|--------------|
| **Text** | `AutoTokenizer` | NLP models | vocab_size, padding_strategy, max_length | batch_size, sequence_length |
| **Vision** | `AutoImageProcessor` | Computer vision | image_size, normalize params (mean/std) | batch_size, height, width, channels |
| **Audio** | `AutoFeatureExtractor` | Speech/audio | sampling_rate, feature_size, n_fft | batch_size, sequence_length |
| **Video** | `AutoVideoProcessor` | Video models | num_frames, frame_sampling_rate | batch_size, num_frames, height, width, channels |
| **Multimodal** | `AutoProcessor` | Multi-input models | modalities list, sub-configs per modality | Varies by modality combination |

## HuggingFace Processor Types

### 1. Text Processing (ONNXTokenizer)

#### Required Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processor_type` | string | Processor type identifier | `"tokenizer"` |
| `processor_class` | string | Original HF tokenizer class | `"BertTokenizerFast"` |
| `fixed_batch_size` | int | Fixed batch size for ONNX | `1` |
| `fixed_sequence_length` | int | Fixed sequence length | `128` |
| `vocab_size` | int | Vocabulary size | `30522` |
| `padding_strategy` | string | How to pad sequences | `"max_length"` |
| `truncation` | bool | Whether to truncate | `true` |
| `max_length` | int | Maximum sequence length | `128` |

#### Optional Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `do_lower_case` | bool | Convert to lowercase | `true` |
| `padding_side` | string | Side to pad on | `"right"` |
| `pad_token_id` | int | Padding token ID | `0` |
| `cls_token_id` | int | CLS token ID | `101` |
| `sep_token_id` | int | SEP token ID | `102` |
| `unk_token_id` | int | Unknown token ID | `100` |
| `mask_token_id` | int | Mask token ID | `103` |
| `special_tokens_mask` | list | Special tokens mask | `[1, 0, ..., 1]` |
| `add_special_tokens` | bool | Add special tokens | `true` |
| `return_token_type_ids` | bool | Return token type IDs | `true` |
| `return_attention_mask` | bool | Return attention mask | `true` |

### 2. Vision Processing (ONNXImageProcessor)

#### Required Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processor_type` | string | Processor type identifier | `"image_processor"` |
| `processor_class` | string | Original HF processor class | `"ViTImageProcessor"` |
| `fixed_batch_size` | int | Fixed batch size for ONNX | `1` |
| `fixed_height` | int | Fixed image height | `224` |
| `fixed_width` | int | Fixed image width | `224` |
| `num_channels` | int | Number of color channels | `3` |
| `image_size` | dict/list | Target image size | `{"height": 224, "width": 224}` |
| `do_resize` | bool | Whether to resize images | `true` |
| `do_normalize` | bool | Whether to normalize | `true` |

#### Optional Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `size` | dict | Resize dimensions | `{"shortest_edge": 224}` |
| `resample` | int | Resampling filter | `3` (BICUBIC) |
| `do_rescale` | bool | Whether to rescale pixel values | `true` |
| `rescale_factor` | float | Rescaling factor | `0.00392156862` (1/255) |
| `image_mean` | list | Normalization means | `[0.485, 0.456, 0.406]` |
| `image_std` | list | Normalization stds | `[0.229, 0.224, 0.225]` |
| `do_center_crop` | bool | Whether to center crop | `false` |
| `crop_size` | dict | Crop dimensions | `{"height": 224, "width": 224}` |
| `do_convert_rgb` | bool | Convert to RGB | `true` |
| `pixel_format` | string | Pixel format | `"channels_first"` or `"channels_last"` |

### 3. Audio Processing (ONNXFeatureExtractor)

#### Required Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processor_type` | string | Processor type identifier | `"feature_extractor"` |
| `processor_class` | string | Original HF extractor class | `"Wav2Vec2FeatureExtractor"` |
| `fixed_batch_size` | int | Fixed batch size for ONNX | `1` |
| `fixed_sequence_length` | int | Fixed audio sequence length | `16000` |
| `sampling_rate` | int | Audio sampling rate | `16000` |
| `feature_size` | int | Feature dimension | `1` |
| `do_normalize` | bool | Whether to normalize audio | `true` |

#### Optional Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `padding_value` | float | Value for padding | `0.0` |
| `padding_side` | string | Side to pad on | `"right"` |
| `return_attention_mask` | bool | Return attention mask | `true` |
| `feature_extractor_type` | string | Feature extraction method | `"raw_waveform"` |
| `n_fft` | int | FFT size (for spectrograms) | `400` |
| `hop_length` | int | Hop length (for spectrograms) | `160` |
| `n_mels` | int | Number of mel bands | `80` |
| `mel_filters` | list | Mel filterbank | `[[...]]` |
| `window` | string | Window function | `"hann"` |

### 4. Video Processing (ONNXVideoProcessor)

#### Required Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processor_type` | string | Processor type identifier | `"video_processor"` |
| `processor_class` | string | Original HF processor class | `"VideoMAEImageProcessor"` |
| `fixed_batch_size` | int | Fixed batch size for ONNX | `1` |
| `fixed_num_frames` | int | Fixed number of frames | `16` |
| `fixed_height` | int | Fixed frame height | `224` |
| `fixed_width` | int | Fixed frame width | `224` |
| `num_channels` | int | Number of color channels | `3` |
| `do_resize` | bool | Whether to resize frames | `true` |
| `do_normalize` | bool | Whether to normalize | `true` |

#### Optional Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `frame_sampling_rate` | int | Frame sampling rate | `4` |
| `clip_duration` | float | Clip duration in seconds | `2.0` |
| `image_mean` | list | Normalization means | `[0.485, 0.456, 0.406]` |
| `image_std` | list | Normalization stds | `[0.229, 0.224, 0.225]` |
| `do_center_crop` | bool | Whether to center crop | `true` |
| `crop_size` | dict | Crop dimensions | `{"height": 224, "width": 224}` |
| `resample` | int | Resampling filter | `3` (BICUBIC) |
| `rescale_factor` | float | Rescaling factor | `0.00392156862` (1/255) |

### 5. Multimodal Processing (ONNXProcessor)

#### Required Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `processor_type` | string | Processor type identifier | `"processor"` |
| `processor_class` | string | Original HF processor class | `"CLIPProcessor"` |
| `modalities` | list | List of modalities | `["text", "vision"]` |
| `fixed_batch_size` | int | Fixed batch size for ONNX | `1` |

#### Modality-Specific Sub-Configurations (Required based on modalities)

For each modality in the `modalities` list, include the corresponding config:

##### Text Modality Config (`text_config`)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tokenizer_class` | string | Tokenizer implementation | `"CLIPTokenizer"` |
| `fixed_sequence_length` | int | Fixed text sequence length | `77` |
| `vocab_size` | int | Vocabulary size | `49408` |
| `padding_strategy` | string | How to pad sequences | `"max_length"` |
| `truncation` | bool | Whether to truncate | `true` |
| `max_length` | int | Maximum sequence length | `77` |

##### Vision Modality Config (`vision_config`)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `image_processor_class` | string | Image processor implementation | `"CLIPImageProcessor"` |
| `fixed_height` | int | Fixed image height | `224` |
| `fixed_width` | int | Fixed image width | `224` |
| `num_channels` | int | Number of color channels | `3` |
| `image_size` | dict | Target image size | `{"height": 224, "width": 224}` |
| `do_resize` | bool | Whether to resize images | `true` |
| `do_normalize` | bool | Whether to normalize | `true` |
| `image_mean` | list | Normalization means | `[0.48145466, 0.4578275, 0.40821073]` |
| `image_std` | list | Normalization stds | `[0.26862954, 0.26130258, 0.27577711]` |

##### Audio Modality Config (`audio_config`)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `feature_extractor_class` | string | Feature extractor implementation | `"WhisperFeatureExtractor"` |
| `fixed_sequence_length` | int | Fixed audio sequence length | `3000` |
| `sampling_rate` | int | Audio sampling rate | `16000` |
| `feature_size` | int | Feature dimension | `80` |
| `n_fft` | int | FFT size | `400` |
| `hop_length` | int | Hop length | `160` |
| `n_mels` | int | Number of mel bands | `80` |
| `do_normalize` | bool | Whether to normalize audio | `true` |

##### Video Modality Config (`video_config`)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `video_processor_class` | string | Video processor implementation | `"VideoLlavaVideoProcessor"` |
| `fixed_num_frames` | int | Fixed number of frames | `8` |
| `fixed_height` | int | Fixed frame height | `224` |
| `fixed_width` | int | Fixed frame width | `224` |
| `num_channels` | int | Number of color channels | `3` |
| `frame_sampling_rate` | int | Frame sampling rate | `4` |
| `do_resize` | bool | Whether to resize frames | `true` |
| `do_normalize` | bool | Whether to normalize | `true` |

#### Optional Metadata
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `default_modality` | string | Default input modality | `"text"` |
| `alignment_strategy` | string | How to align modalities | `"pad_to_max"` |
| `cross_attention_config` | dict | Cross-attention settings | `{"enabled": true}` |
| `modality_fusion` | string | How modalities are combined | `"concatenate"` or `"cross_attention"` |
| `shared_embedding` | bool | Whether embeddings are shared | `false` |

## Metadata Storage Format

### Option 1: ONNX Model Metadata (Recommended)
```python
import onnx

model = onnx.load("model.onnx")

# Add processor metadata
metadata = {
    "processor_type": "tokenizer",
    "processor_class": "BertTokenizerFast",
    "fixed_batch_size": "1",
    "fixed_sequence_length": "128",
    "vocab_size": "30522",
    # ... additional fields
}

for key, value in metadata.items():
    meta = model.metadata_props.add()
    meta.key = f"processor.{key}"  # Namespace with "processor."
    meta.value = str(value) if not isinstance(value, str) else value

# For complex nested structures, use JSON
import json
complex_metadata = {
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225]
}
meta = model.metadata_props.add()
meta.key = "processor.normalization"
meta.value = json.dumps(complex_metadata)

onnx.save(model, "model_with_metadata.onnx")
```

### Option 2: Companion JSON File
```json
{
  "processor": {
    "type": "tokenizer",
    "class": "BertTokenizerFast",
    "fixed_shapes": {
      "batch_size": 1,
      "sequence_length": 128
    },
    "config": {
      "vocab_size": 30522,
      "do_lower_case": true,
      "padding_strategy": "max_length",
      "truncation": true,
      "max_length": 128,
      "pad_token_id": 0,
      "cls_token_id": 101,
      "sep_token_id": 102
    }
  },
  "model": {
    "type": "bert",
    "task": "text-classification",
    "original_name": "bert-base-uncased"
  },
  "export": {
    "version": "1.0.0",
    "timestamp": "2024-01-01T00:00:00Z",
    "hierarchy_preserved": true
  }
}
```

## Auto-Detection Logic

When metadata is missing, ONNXAutoProcessor can infer processor type from input tensor names:

### Input Tensor Name Patterns
| Pattern | Inferred Type | Notes |
|---------|--------------|-------|
| `input_ids`, `attention_mask`, `token_type_ids` | Text/Tokenizer | Standard text model inputs |
| `pixel_values`, `image` | Vision/ImageProcessor | Standard vision model inputs |
| `input_values`, `input_features` | Audio/FeatureExtractor | Standard audio model inputs |
| `video`, `frames` | Video/VideoProcessor | Video model inputs |
| Multiple patterns | Multimodal/Processor | Combined modalities |

### Shape-Based Inference
| Shape Pattern | Inferred Type | Notes |
|--------------|--------------|-------|
| `[batch, sequence]` | Text | 2D tensor typically text |
| `[batch, channels, height, width]` | Vision | 4D tensor typically images |
| `[batch, sequence]` or `[batch, 1, sequence]` | Audio | Audio waveform |
| `[batch, frames, channels, height, width]` | Video | 5D tensor for video |

## Notes

1. **Video Processing**: HuggingFace supports video through `AutoVideoProcessor` (available since transformers 4.37+). Video models typically process videos as sequences of frames, reusing image processing infrastructure.

2. **Fixed Shapes**: All ONNX processors require fixed shapes for optimal performance (40x+ speedup). Dynamic shapes are possible but significantly slower.

3. **Metadata Priority**: 
   - Explicit metadata in ONNX model takes highest priority
   - Companion JSON file is second priority
   - Auto-detection from tensor shapes/names is fallback

4. **Backward Compatibility**: The system should gracefully handle models without metadata by falling back to auto-detection.

5. **Version Compatibility**: Store the transformers version used during export to ensure compatibility:
   ```json
   {
     "export_info": {
       "transformers_version": "4.36.0",
       "onnx_version": "1.15.0",
       "export_tool": "modelexport",
       "export_version": "1.0.0"
     }
   }
   ```

## Concrete Examples for Popular Models

### BERT (Text)
```json
{
  "processor": {
    "type": "tokenizer",
    "class": "BertTokenizerFast",
    "fixed_shapes": {
      "batch_size": 1,
      "sequence_length": 128
    },
    "config": {
      "vocab_size": 30522,
      "do_lower_case": true,
      "padding_strategy": "max_length",
      "truncation": true,
      "max_length": 128,
      "pad_token_id": 0,
      "cls_token_id": 101,
      "sep_token_id": 102
    }
  }
}
```

### ViT (Vision)
```json
{
  "processor": {
    "type": "image_processor",
    "class": "ViTImageProcessor",
    "fixed_shapes": {
      "batch_size": 1,
      "height": 224,
      "width": 224,
      "channels": 3
    },
    "config": {
      "do_resize": true,
      "size": {"height": 224, "width": 224},
      "do_normalize": true,
      "image_mean": [0.485, 0.456, 0.406],
      "image_std": [0.229, 0.224, 0.225],
      "resample": 3
    }
  }
}
```

### Whisper (Audio)
```json
{
  "processor": {
    "type": "feature_extractor",
    "class": "WhisperFeatureExtractor",
    "fixed_shapes": {
      "batch_size": 1,
      "sequence_length": 3000
    },
    "config": {
      "sampling_rate": 16000,
      "feature_size": 80,
      "n_fft": 400,
      "hop_length": 160,
      "n_mels": 80,
      "do_normalize": true
    }
  }
}
```

### VideoMAE (Video)
```json
{
  "processor": {
    "type": "video_processor",
    "class": "VideoMAEImageProcessor",
    "fixed_shapes": {
      "batch_size": 1,
      "num_frames": 16,
      "height": 224,
      "width": 224,
      "channels": 3
    },
    "config": {
      "do_resize": true,
      "size": {"height": 224, "width": 224},
      "do_normalize": true,
      "image_mean": [0.485, 0.456, 0.406],
      "image_std": [0.229, 0.224, 0.225],
      "frame_sampling_rate": 4,
      "resample": 3
    }
  }
}
```

### CLIP (Multimodal - Text + Vision)
```json
{
  "processor": {
    "type": "processor",
    "class": "CLIPProcessor",
    "modalities": ["text", "vision"],
    "fixed_shapes": {
      "batch_size": 1
    },
    "text_config": {
      "tokenizer_class": "CLIPTokenizer",
      "fixed_sequence_length": 77,
      "vocab_size": 49408,
      "padding_strategy": "max_length",
      "truncation": true,
      "max_length": 77
    },
    "vision_config": {
      "image_processor_class": "CLIPImageProcessor",
      "fixed_height": 224,
      "fixed_width": 224,
      "num_channels": 3,
      "do_resize": true,
      "size": {"height": 224, "width": 224},
      "do_normalize": true,
      "image_mean": [0.48145466, 0.4578275, 0.40821073],
      "image_std": [0.26862954, 0.26130258, 0.27577711]
    }
  }
}
```

### Whisper (Multimodal - Audio + Text)
```json
{
  "processor": {
    "type": "processor",
    "class": "WhisperProcessor",
    "modalities": ["audio", "text"],
    "fixed_shapes": {
      "batch_size": 1
    },
    "audio_config": {
      "feature_extractor_class": "WhisperFeatureExtractor",
      "fixed_sequence_length": 3000,
      "sampling_rate": 16000,
      "feature_size": 80,
      "n_fft": 400,
      "hop_length": 160,
      "n_mels": 80,
      "do_normalize": true
    },
    "text_config": {
      "tokenizer_class": "WhisperTokenizer",
      "fixed_sequence_length": 448,
      "vocab_size": 51865,
      "padding_strategy": "max_length",
      "truncation": true,
      "max_length": 448
    }
  }
}
```