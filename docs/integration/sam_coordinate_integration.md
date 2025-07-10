# SAM Coordinate Fix Integration Guide

## Overview

The SAM coordinate fix is **automatically integrated** into the `model_input_generator.py` using type-based config detection. No manual integration needed!

## How It Works

The fix is implemented via `patch_export_config()` function:

```python
def patch_export_config(export_config) -> None:
    """Apply model-specific patches to export configurations"""
    config_type = type(export_config).__name__
    
    if config_type == "SamOnnxConfig":
        # Automatically inject SemanticDummyPointsGenerator
        # Generates [0, 1024] coordinates instead of [0, 1]
        logger.info("🎯 Applied semantic coordinate fix for SamOnnxConfig")
```

## Automatic Detection

- ✅ **Type-based**: Detects `SamOnnxConfig` instances automatically
- ✅ **Transparent**: No changes needed in CLI, HTP strategy, or other components  
- ✅ **Targeted**: Only affects SAM models, other models unchanged
- ✅ **Extensible**: Easy pattern for future model-specific fixes

## Usage

Just use the standard modelexport workflow:

```bash
# Automatically applies SAM fix when needed
uv run modelexport export facebook/sam-vit-base sam.onnx --strategy htp
```

```python
# Or programmatically - fix applied automatically
from modelexport.core.model_input_generator import generate_dummy_inputs

inputs = generate_dummy_inputs(model_name_or_path="facebook/sam-vit-base")
# SAM coordinates automatically generated as [0, 1024] range
```

## Architecture

- **Location**: `modelexport/core/model_input_generator.py`
- **Function**: `patch_export_config()`
- **Inner Class**: `SemanticDummyPointsGenerator`
- **Integration**: Called automatically in `generate_dummy_inputs_from_model_path()`

## Benefits

- ✅ **Zero configuration** - works automatically
- ✅ **Type-safe detection** - based on config class type
- ✅ **No side effects** - injection per config instance
- ✅ **Maintainable** - clear separation of concerns
- ✅ **Extensible** - pattern for future model fixes

See `docs/analysis/sam_coordinate_fix_analysis.md` for technical details.