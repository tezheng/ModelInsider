# Optimum's Dummy Input Generation: Deep Dive Analysis

## Overview

Optimum uses a sophisticated multi-layered approach to generate dummy input tensors that match model requirements. The system is designed to be extensible, framework-agnostic, and model-specific when needed.

## Key Architecture Components

### 1. Default Shape Constants (`DEFAULT_DUMMY_SHAPES`)

Located in `optimum/utils/input_generators.py`:

```python
DEFAULT_DUMMY_SHAPES = {
    "batch_size": 2,
    "sequence_length": 16,
    "num_choices": 4,
    # Image dimensions
    "width": 64,
    "height": 64,
    "num_channels": 3,
    "point_batch_size": 3,
    "nb_points_per_image": 2,
    # Audio features
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}
```

These provide sensible defaults that work for most models during export.

### 2. Base Generator Class (`DummyInputGenerator`)

Abstract base class providing:
- **Framework abstraction**: Supports PyTorch, TensorFlow, and NumPy
- **Utility methods**:
  - `random_int_tensor()`: Generate integer tensors with specified ranges
  - `random_float_tensor()`: Generate float tensors with specified ranges
  - `random_mask_tensor()`: Generate attention masks with padding
  - `constant_tensor()`: Generate constant-valued tensors

### 3. Domain-Specific Generators

#### Text Models (`DummyTextInputGenerator`)
```python
SUPPORTED_INPUT_NAMES = (
    "input_ids",
    "attention_mask",
    "encoder_attention_mask",
    "global_attention_mask",
    "token_type_ids",
    "position_ids",
)
```

**Key Features**:
- Uses `vocab_size` from model config for input_ids generation
- Supports random batch size and sequence length ranges
- Generates proper attention masks with padding side awareness
- Special handling for multiple-choice tasks (3D tensors)

#### Vision Models (`DummyVisionInputGenerator`)
```python
SUPPORTED_INPUT_NAMES = (
    "pixel_values",
    "pixel_mask",
    "sample",
    "latent_sample",
)
```

**Key Features**:
- Configurable image dimensions (height, width, channels)
- Generates float tensors for pixel values
- Integer masks for pixel_mask

#### Audio Models (`DummyAudioInputGenerator`)
```python
SUPPORTED_INPUT_NAMES = (
    "input_features",
    "input_values"
)
```

**Key Features**:
- Supports both raw waveform (`input_values`) and features (`input_features`)
- Configurable feature size and sequence length
- Values normalized to [-1, 1] range

#### Past Key Values (`DummyPastKeyValuesGenerator`)
**Key Features**:
- Generates cached key-value pairs for autoregressive models
- Shape: `(batch_size, num_heads, sequence_length, head_dim)`
- Supports both encoder-decoder and decoder-only architectures

### 4. Model-Specific Overrides

#### Special Cases:
- **GPTBigCode**: Fused KV cache (single tensor instead of separate K,V)
- **Bloom**: Different shape ordering for older transformers versions
- **MultiQuery**: Handles models with multi-query attention (fewer KV heads)
- **Whisper**: Hardcoded mel_bins issue (80 vs 128)
- **Pix2Struct**: Static sequence lengths from processor config

## Shape Inference Mechanism

### 1. Configuration-Based Inference

Shapes are determined from multiple sources in priority order:

1. **User-provided shapes** (via kwargs)
2. **Model configuration attributes**:
   - `vocab_size` for token ranges
   - `hidden_size` for embeddings
   - `num_attention_heads` for attention layers
   - `num_hidden_layers` for layer count
3. **Default values** from `DEFAULT_DUMMY_SHAPES`

### 2. Dynamic Shape Calculation

```python
# Example from DummyTextInputGenerator
if random_sequence_length_range:
    low, high = random_sequence_length_range
    self.sequence_length = random.randint(low, high)
else:
    self.sequence_length = sequence_length  # From defaults or user
```

### 3. Task-Specific Shapes

Different tasks require different tensor shapes:
- **Multiple-choice**: `[batch_size, num_choices, sequence_length]`
- **Seq2Seq**: Separate encoder/decoder sequence lengths
- **Generation with past**: `sequence_length = 1` for cached models

## Integration with ONNX Export

### 1. OnnxConfig Classes

Each model architecture has its own config:
```python
class TextEncoderOnnxConfig(OnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

class TextDecoderOnnxConfig(OnnxConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyPastKeyValuesGenerator
    )
```

### 2. Generation Process

```python
def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
    dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)
    
    dummy_inputs = {}
    for input_name in self.inputs:
        for dummy_input_gen in dummy_inputs_generators:
            if dummy_input_gen.supports_input(input_name):
                dummy_inputs[input_name] = dummy_input_gen.generate(
                    input_name, 
                    framework=framework,
                    int_dtype=self.int_dtype,
                    float_dtype=self.float_dtype
                )
                break
```

### 3. Shape Override Mechanism

The `overwrite_shape_and_generate_input()` method allows runtime shape overrides:
- Checks for user-provided shapes in kwargs
- Updates generator attributes before calling generate()
- Ensures consistency across related inputs (e.g., attention_mask matches input_ids)

## Key Patterns and Best Practices

### 1. Extensibility Pattern
- Base classes for common functionality
- Model-specific subclasses for special cases
- Generator composition in OnnxConfig classes

### 2. Framework Agnosticism
- DTYPE_MAPPER for type conversion
- Framework-specific tensor creation
- Decorator for framework availability checking

### 3. Configuration Normalization
- NormalizedConfig classes abstract model differences
- Consistent attribute access across architectures
- Fallback chains for missing attributes

### 4. Validation Support
- Separate `generate_dummy_inputs_for_validation()` method
- Reference model inputs for validation
- Special handling for ONNX Runtime requirements

## Insights for Our Implementation

### 1. What We Should Adopt
- **Default shape constants** for common dimensions
- **Generator class hierarchy** for extensibility
- **Framework abstraction** for PyTorch/NumPy support
- **Configuration normalization** for universal handling

### 2. What We Can Improve
- **Auto-detection from model**: Inspect actual model input specs
- **Smart shape inference**: Use model's forward signature
- **Batch processing**: Generate multiple samples efficiently
- **Type inference**: Automatically determine int vs float inputs

### 3. Universal Approach Benefits
- No hardcoded model names (follows our cardinal rule)
- Works with any nn.Module through inspection
- Extensible for custom models
- Maintains hierarchy preservation focus

## Implementation Strategy for ModelExport

1. **Create base generator**:
   - Universal inspection of model.forward() signature
   - Extract shapes from first forward pass
   - Cache shapes for reuse

2. **Smart defaults**:
   - Use config attributes when available
   - Fall back to sensible defaults
   - Allow user overrides

3. **Integration points**:
   - Hook into HTPExporter
   - Generate inputs before export
   - Validate shapes match model expectations

4. **Testing approach**:
   - Test with multiple architectures
   - Verify shape inference accuracy
   - Ensure no hardcoded assumptions