# Enhanced Pipeline Solution with data_processor Parameter

## Overview

This document describes the **Enhanced Pipeline** wrapper that adds a generic `data_processor` parameter to the Hugging Face pipeline API, providing a cleaner and more intuitive interface.

## The Problem

The standard Hugging Face pipeline requires different parameter names for different modalities:
- Text models need `tokenizer`
- Vision models need `image_processor`
- Audio models need `feature_extractor`
- Multimodal models need `processor`

This leads to confusion and less maintainable code, especially when working with custom processors like our `FixedShapeTokenizer`.

## The Solution: Enhanced Pipeline

We've created a wrapper that accepts a generic `data_processor` parameter and automatically routes it to the correct pipeline parameter.

### Installation

```python
from src.enhanced_pipeline import pipeline, create_pipeline, create_fixed_shape_pipeline
```

### Basic Usage

```python
from src.enhanced_pipeline import pipeline
from src.fixed_shape_tokenizer import FixedShapeTokenizer

# Create your custom processor
fixed_tokenizer = FixedShapeTokenizer(
    tokenizer=base_tokenizer,
    fixed_batch_size=2,
    fixed_sequence_length=16
)

# Use with the generic data_processor parameter
pipe = pipeline(
    "feature-extraction",
    model=model,
    data_processor=fixed_tokenizer  # ← Works for any processor type!
)

# Use the pipeline normally
results = pipe("Your text here")
```

## API Reference

### 1. `pipeline()` - Drop-in Replacement

```python
pipeline(
    task: str,
    model=None,
    data_processor=None,  # ← New generic parameter
    **kwargs
)
```

A drop-in replacement for `transformers.pipeline` with `data_processor` support.

**Example:**
```python
from src.enhanced_pipeline import pipeline

# Works with any processor type
pipe = pipeline("task-name", model=model, data_processor=any_processor)
```

### 2. `create_pipeline()` - Full-Featured Version

```python
create_pipeline(
    task: str,
    model: Optional[Union[str, PreTrainedModel, TFPreTrainedModel]] = None,
    data_processor: Optional[Any] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map: Optional[Union[str, dict[str, Union[int, str]]]] = None,
    torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs: Any
) -> Pipeline
```

Full-featured pipeline creation with all standard parameters plus `data_processor`.

### 3. `create_fixed_shape_pipeline()` - ONNX Convenience

```python
create_fixed_shape_pipeline(
    task: str,
    model,
    tokenizer,
    fixed_batch_size: int,
    fixed_sequence_length: int,
    **kwargs
)
```

Convenience function specifically for creating pipelines with fixed-shape tokenizers (common for ONNX models).

**Example:**
```python
pipe = create_fixed_shape_pipeline(
    "feature-extraction",
    model=onnx_model,
    tokenizer=base_tokenizer,
    fixed_batch_size=2,
    fixed_sequence_length=16
)
```

## How It Works

### Intelligent Routing

The enhanced pipeline automatically detects the processor type and routes it to the correct parameter:

```python
# Text task → tokenizer
pipe = pipeline("text-classification", model=model, data_processor=my_tokenizer)
# Internally becomes: pipeline(..., tokenizer=my_tokenizer)

# Vision task → image_processor
pipe = pipeline("image-classification", model=model, data_processor=my_image_proc)
# Internally becomes: pipeline(..., image_processor=my_image_proc)

# Audio task → feature_extractor
pipe = pipeline("automatic-speech-recognition", model=model, data_processor=my_extractor)
# Internally becomes: pipeline(..., feature_extractor=my_extractor)

# Multimodal task → processor
pipe = pipeline("image-to-text", model=model, data_processor=my_processor)
# Internally becomes: pipeline(..., processor=my_processor)
```

### Detection Logic

The processor type is detected using:

1. **Class name analysis**: Checks for "Tokenizer", "ImageProcessor", "FeatureExtractor", etc.
2. **Attribute inspection**: Looks for characteristic attributes like `tokenize`, `pixel_values`, etc.
3. **Task-based fallback**: Uses task type to determine the most likely processor type

## Use Cases

### 1. Fixed Shape ONNX Models

```python
from src.enhanced_pipeline import pipeline
from src.fixed_shape_tokenizer import FixedShapeTokenizer

# Create fixed shape tokenizer
fixed_tokenizer = FixedShapeTokenizer(
    tokenizer=base_tokenizer,
    fixed_batch_size=2,
    fixed_sequence_length=16
)

# Clean, intuitive pipeline creation
pipe = pipeline(
    "feature-extraction",
    model=onnx_model,
    data_processor=fixed_tokenizer
)
```

### 2. Custom Vision Processors

```python
# Custom image processor with fixed dimensions
class FixedShapeImageProcessor:
    def __init__(self, base_processor, height=224, width=224):
        self.base_processor = base_processor
        self.height = height
        self.width = width
    
    def __call__(self, images, **kwargs):
        # Process with fixed dimensions
        ...

# Use with enhanced pipeline
fixed_image_proc = FixedShapeImageProcessor(base_processor)
pipe = pipeline(
    "image-classification",
    model=vision_model,
    data_processor=fixed_image_proc  # Automatically routed to image_processor
)
```

### 3. Multimodal Models

```python
from transformers import AutoProcessor

# Load multimodal processor
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Use with generic parameter
pipe = pipeline(
    "zero-shot-image-classification",
    model=clip_model,
    data_processor=processor  # Works for any processor type!
)
```

## Benefits

1. **Cleaner API**: Single parameter name for all processor types
2. **Better Maintainability**: No need to remember different parameter names
3. **Type Agnostic**: Works with any custom processor implementation
4. **Backward Compatible**: All original pipeline parameters still work
5. **Intelligent Routing**: Automatically detects and routes to correct parameter

## Comparison

### Before (Standard Pipeline)

```python
# Must use different parameters for different types
text_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
vision_pipe = pipeline("image-classification", model=model, image_processor=processor)
audio_pipe = pipeline("audio-classification", model=model, feature_extractor=extractor)

# Error if you use wrong parameter name
pipe = pipeline("text-classification", model=model, processor=tokenizer)  # ❌ Error!
```

### After (Enhanced Pipeline)

```python
from src.enhanced_pipeline import pipeline

# Same parameter for all types
text_pipe = pipeline("text-classification", model=model, data_processor=tokenizer)
vision_pipe = pipeline("image-classification", model=model, data_processor=processor)
audio_pipe = pipeline("audio-classification", model=model, data_processor=extractor)

# Always works!
pipe = pipeline("any-task", model=model, data_processor=any_processor)  # ✅ Works!
```

## Implementation Details

The enhanced pipeline:
1. Accepts `data_processor` as a generic parameter
2. Analyzes the processor to determine its type
3. Routes it to the appropriate standard pipeline parameter
4. Calls the standard Hugging Face pipeline with correct parameters

This is a pure wrapper - it doesn't modify the underlying pipeline behavior, just provides a cleaner interface.

## Conclusion

The Enhanced Pipeline wrapper solves the confusion around different processor parameter names in the Hugging Face pipeline API. It provides a cleaner, more intuitive interface while maintaining full compatibility with all pipeline features.

Key takeaways:
- ✅ Use `data_processor` for any processor type
- ✅ Automatic intelligent routing to correct parameter
- ✅ Works with custom processors like `FixedShapeTokenizer`
- ✅ Drop-in replacement for standard pipeline
- ✅ Includes convenience functions for common use cases

This makes working with pipelines, especially for ONNX models with fixed shapes, much more straightforward and maintainable.