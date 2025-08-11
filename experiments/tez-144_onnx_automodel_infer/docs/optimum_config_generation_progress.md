# UniversalOnnxConfig Implementation Progress

## ⚠️ IMPORTANT: This Implementation is Now Largely Superseded by ADR-012

> **Formal Decision**: See [ADR-013: ONNX Config for Optimum Compatibility](../../../docs/adr/ADR-013-onnx-config-for-optimum-compatibility.md) - Section "Decision on UniversalOnnxConfig"

### Current Status
UniversalOnnxConfig was implemented as TEZ-145 (subtask of TEZ-144) before ADR-012 was finalized. **Per ADR-013, UniversalOnnxConfig is NO LONGER NEEDED for 99% of use cases.**

### Do We Need UniversalOnnxConfig?
**Short Answer: NO** - For almost all practical purposes, you don't need UniversalOnnxConfig.

### Why Not?
**ADR-012 provides a simpler, better solution:**
- **HuggingFace Hub Models**: Config.json is automatically loaded from Hub at inference time
- **Local Models with config.json**: Config.json is automatically copied during export
- **No manual configuration needed**: Everything works automatically

### When Might You Need It?
**Only in these rare edge cases:**
- Custom models with NO config.json at all (very rare)
- Non-standard architectures not following HF conventions
- Research/experimental models
- Legacy model migration

## What Was Implemented

### 1. Core Components

#### `task_detector.py`
- **Purpose**: Automatically detect task from model architecture and configuration
- **Features**:
  - Extended task mappings covering 40+ task types
  - Model type to default task mapping for 80+ model families
  - Configuration hint-based detection
  - Task family categorization
  - Past key values requirement detection

#### `input_generator.py`
- **Purpose**: Generate input specifications and dummy inputs for various model types
- **Features**:
  - Support for text, vision, audio, multimodal, and document models
  - Automatic input name detection based on model type
  - Dynamic axes specification
  - Dummy input generation with proper shapes and dtypes
  - Preprocessor type detection

#### `shape_inference.py`
- **Purpose**: Infer output shapes and names based on model task
- **Features**:
  - Extended output mappings for 30+ tasks
  - Model-specific output handling
  - Configuration-based filtering
  - Shape prediction for various output types

#### `universal_config.py`
- **Purpose**: Main UniversalOnnxConfig class that brings everything together
- **Features**:
  - Automatic task detection
  - Input/output specification generation
  - Dynamic axes configuration
  - Dummy input generation
  - Past key values support for generation models
  - Preprocessor integration support

### 2. Key Design Decisions

1. **Universal Approach**: Unlike Optimum's model-specific configs, this implementation works with ANY model by detecting patterns at runtime.

2. **Comprehensive Coverage**: Supports:
   - Text models (BERT, GPT, T5, etc.)
   - Vision models (ViT, ResNet, ConvNext, etc.)
   - Audio models (Wav2Vec2, Whisper, etc.)
   - Multimodal models (CLIP, BLIP, etc.)
   - Document models (LayoutLM, etc.)
   - Object detection models (DETR, YOLOS, etc.)
   - Segmentation models (SAM, MaskFormer, etc.)

3. **Automatic Detection**: The system automatically detects:
   - Task from architecture name
   - Input requirements from model type
   - Output specifications from task
   - Dynamic axes for variable dimensions
   - Preprocessor type needed

## Usage Example

```python
from onnx_config import UniversalOnnxConfig
from transformers import AutoConfig, AutoModel
import torch

# Load any HuggingFace model
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Create universal config (automatic detection)
onnx_config = UniversalOnnxConfig(config)

print(f"Task: {onnx_config.task}")  # feature-extraction
print(f"Inputs: {onnx_config.get_input_names()}")  # ['input_ids', 'attention_mask', 'token_type_ids']
print(f"Outputs: {onnx_config.get_output_names()}")  # ['last_hidden_state', 'pooler_output']

# Generate dummy inputs
dummy_inputs = onnx_config.generate_dummy_inputs()

# Export to ONNX
torch.onnx.export(
    model,
    tuple(dummy_inputs.values()),
    "model.onnx",
    input_names=onnx_config.get_input_names(),
    output_names=onnx_config.get_output_names(),
    dynamic_axes=onnx_config.get_dynamic_axes(),
    opset_version=onnx_config.DEFAULT_ONNX_OPSET,
)
```

## Comparison with Optimum

### Optimum Approach:
- **Model-specific classes**: BertOnnxConfig, GPT2OnnxConfig, etc.
- **Limited coverage**: ~100 supported architectures
- **Maintenance burden**: New class needed for each model
- **Precise but rigid**: Hard-coded specifications

### Our Universal Approach:
- **Single universal class**: UniversalOnnxConfig
- **Unlimited coverage**: Works with any model
- **Zero maintenance**: Automatic detection
- **Flexible and adaptive**: Runtime configuration

## Benefits

1. **No Manual Configuration**: Automatically generates OnnxConfig for any model
2. **Future-Proof**: New models work without code changes
3. **Reduced Complexity**: Single class instead of 100+ model-specific classes
4. **Easy Integration**: Drop-in replacement for Optimum configs
5. **Comprehensive Support**: Handles edge cases and special models

## Files Created

```
experiments/tez-144_onnx_automodel_infer/
├── src/
│   └── onnx_config/
│       ├── __init__.py              # Module exports
│       ├── patterns.py              # Pattern mappings (existing)
│       ├── task_detector.py         # Task detection logic
│       ├── input_generator.py       # Input specification generation
│       ├── shape_inference.py       # Output shape inference
│       └── universal_config.py      # Main UniversalOnnxConfig class
├── notebooks/
│   └── understanding_onnxconfig.ipynb  # Educational notebook
└── test_universal_config.py         # Test script
```

## Integration with AutoModelForONNX

The UniversalOnnxConfig can be integrated with the existing AutoModelForONNX loader:

```python
from src.auto_model_loader import AutoModelForONNX
from src.onnx_config import UniversalOnnxConfig

# For exporting models
config = AutoConfig.from_pretrained(model_name)
onnx_config = UniversalOnnxConfig(config)
# ... use for export

# For loading exported models
model = AutoModelForONNX.from_pretrained(model_path)
# ... use for inference
```

## Next Steps

1. **Testing**: Test with diverse model architectures
2. **Integration**: Integrate with modelexport main codebase
3. **Documentation**: Add more examples and documentation
4. **Optimization**: Add caching and performance optimizations
5. **Edge Cases**: Handle more special cases as discovered

## Related Linear Tasks

- **TEZ-144**: ONNX model inference with Optimum (parent task)
- **TEZ-145**: UniversalOnnxConfig generation (this subtask - COMPLETED)

## Integration with ADR-012 (ONNX Config Strategy)

Following ADR-012 architectural decisions, the UniversalOnnxConfig serves specific use cases:

### Primary Use Cases
1. **Custom/Local Models**: Models not on HuggingFace Hub that need config generation
2. **Edge Cases**: Models with non-standard architectures not covered by standard patterns
3. **Development/Testing**: Rapid prototyping and experimentation with new model types
4. **Fallback Mechanism**: When automatic Hub-based config loading fails

### ADR-012 Integration
- **Hub Models**: Use ADR-012's metadata approach (config loaded from Hub automatically)
- **Local Models**: Can use either ADR-012's config copying OR UniversalConfig generation
- **Hybrid Approach**: UniversalConfig can supplement ADR-012 for edge cases

### Recommended Usage Pattern
```python
# Preferred for Hub models (ADR-012)
modelexport export bert-base-uncased model.onnx  # Automatic config handling

# For custom models - choose approach:
# Option 1: ADR-012 config copying (if model has config.json)
modelexport export ./my_custom_model model.onnx  

# Option 2: UniversalConfig generation (if no config.json available)
from onnx_config import UniversalOnnxConfig
config = UniversalOnnxConfig.from_model_path('./my_custom_model')
# Use config for export
```

## Conclusion

### Should You Use UniversalOnnxConfig?
**NO** - Use ADR-012's automatic config handling instead:

```bash
# For HuggingFace Hub models - just export, config handled automatically
modelexport export bert-base-uncased model.onnx

# For local models with config.json - also automatic
modelexport export ./my_model_dir model.onnx
```

### ADR-012 vs UniversalOnnxConfig
- **ADR-012**: Handles 99% of use cases automatically (RECOMMENDED)
- **UniversalOnnxConfig**: Only for rare edge cases without any config.json

### Migration Recommendation
**If you're using UniversalOnnxConfig, migrate to ADR-012's approach:**
1. For Hub models: Just use model name, config loads automatically
2. For local models: Ensure config.json exists, it will be copied automatically
3. Delete UniversalOnnxConfig code unless you have a specific edge case

The ADR-012 approach is simpler, more maintainable, and covers virtually all real-world scenarios.