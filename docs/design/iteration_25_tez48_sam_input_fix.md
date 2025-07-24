# Iteration 25: TEZ-48 SAM Input Generation Fix

**Date**: 2025-07-23  
**Task**: TEZ-48 - SAM model export generates incorrect inputs, bypassing vision encoder  
**Status**: ✅ **COMPLETED**

## Overview

Fixed a critical issue where SAM (Segment Anything Model) exports were generating pre-computed embeddings instead of raw pixel values, causing the vision encoder to be completely bypassed during export. This resulted in incomplete model exports (23/241 modules vs 217/241 modules) and dramatically smaller ONNX files (16MB vs 359MB).

## Problem Analysis

### Root Cause
SAM models in Optimum's configuration were designed for deployment scenarios where the vision encoder and mask decoder are exported separately. The `SamOnnxConfig` used `DummyVisionEmbeddingsGenerator` which generated:

- `image_embeddings` - Pre-computed vision features
- `image_positional_embeddings` - Pre-computed positional encodings  

This bypassed the vision encoder entirely during export.

### Expected vs Actual Behavior

| Metric | Before Fix (Partial Export) | After Fix (Full Export) |
|--------|------------------------------|--------------------------|
| **Input Types** | `image_embeddings`, `image_positional_embeddings` | `pixel_values`, `input_points`, `input_labels` |
| **Modules Traced** | 23/241 (9.5%) | 217/241 (90.0%) |
| **ONNX File Size** | ~16MB | 358.37MB |
| **Vision Encoder** | ❌ Bypassed | ✅ Included |

## Solution Implementation

### Fix Location
`modelexport/core/model_input_generator.py` - `patch_export_config()` function

### Technical Approach
Replaced Optimum's default input generation for SAM models with a custom implementation that:

1. **Detects SAM models** by checking `config_type == "SamOnnxConfig"`
2. **Overrides input generation** to produce full model inputs:
   - `pixel_values`: Raw image tensor `[batch, 3, 1024, 1024]`
   - `input_points`: Semantic coordinates `[batch, 1, 1, 2]` 
   - `input_labels`: Point labels `[batch, 1, 1]`
3. **Eliminates embedding generation** to force full model tracing

### Key Code Changes

```python
def patch_export_config(export_config) -> None:
    if config_type == "SamOnnxConfig":
        def generate_full_model_inputs(framework="pt", **kwargs):
            # Generate inputs for full model export
            batch_size = shapes.get("batch_size", 1)
            
            inputs = {
                # Generate pixel_values for full model (includes vision encoder)
                "pixel_values": torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32),
                
                # Generate semantic input points (center region)
                "input_points": torch.tensor([[[[512.0, 512.0]]]] * batch_size, dtype=torch.float32),
                
                # Generate input labels (foreground point)
                "input_labels": torch.tensor([[[1]]] * batch_size, dtype=torch.long),
            }
            return inputs
        
        # Replace the generate_dummy_inputs method
        export_config.generate_dummy_inputs = generate_full_model_inputs
```

## Validation Results

### Export Success Metrics
✅ **Correct Input Generation**: `pixel_values` instead of embeddings  
✅ **Vision Encoder Included**: SamVisionEncoder layers 0-11 with attention modules  
✅ **Correct File Size**: 358.37MB (matches expected ~359MB)  
✅ **High Module Coverage**: 217/241 modules traced (90% vs previous 9.5%)  
✅ **Complete Hierarchy**: Full model hierarchy from vision encoder to mask decoder

### Test Coverage
Created comprehensive test suite in `tests/test_sam_input_generation_fix.py`:

- ✅ `test_sam_generates_pixel_values()` - Validates correct input types
- ✅ `test_sam_inputs_work_with_model()` - Ensures inputs work with actual model
- ✅ `test_sam_export_includes_vision_encoder()` - Confirms vision encoder inclusion
- ✅ `test_sam_does_not_generate_embeddings()` - Regression prevention

## Impact Assessment

### Before Fix Issues
- **Incomplete Exports**: Only mask decoder exported (23/241 modules)
- **Small File Size**: 16MB indicating missing components
- **User Confusion**: Expected full model but got partial export
- **Manual Workaround Required**: Users had to create custom input specs

### After Fix Benefits
- **Complete Model Export**: Full SAM model including vision encoder (217/241 modules)
- **Correct File Size**: 358.37MB indicating complete model
- **User Experience**: No manual intervention required
- **Proper Tracing**: Vision encoder properly traced and tagged

## Design Considerations

### Why This Approach?
1. **Minimal Impact**: Only affects SAM models, other models unchanged
2. **Maintains Compatibility**: Uses standard PyTorch tensors and shapes
3. **Semantic Inputs**: Generated coordinates are meaningful (center region)
4. **Extensible**: Pattern can be applied to other models with similar issues

### Alternative Approaches Considered
1. **Modify Optimum Configuration**: Too invasive, affects other users
2. **Input Specs Workaround**: Requires manual intervention from users
3. **Generator Class Override**: More complex, harder to maintain

## Lessons Learned

### Key Insights
- **Optimum's Design Intent**: Configurations often target deployment scenarios
- **Export vs Deployment**: Full model export needs different inputs than deployment
- **Testing Critical**: Model inference testing validates input correctness
- **Semantic Inputs**: Meaningful dummy data improves export quality

### Future Improvements
- Monitor other models that might have similar embedding-based configs
- Consider adding configuration option for deployment vs full export modes
- Document the difference between deployment and development export strategies

## Achievements

✅ **Root Cause Identified**: Optimum's deployment-focused SAM configuration  
✅ **Minimal Fix Implemented**: 20 lines of code change with maximum impact  
✅ **Complete Validation**: Export testing + model inference testing + regression tests  
✅ **Documentation Created**: Comprehensive iteration notes and inline comments  
✅ **Quality Assurance**: All tests pass, no regressions detected

## Next Steps

With TEZ-48 resolved, the SAM model export now works correctly for full model scenarios. Users can export complete SAM models without manual input specification, and the vision encoder is properly included in the traced hierarchy.

**Task Status**: ✅ **READY FOR COMMIT**