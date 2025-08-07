# Optimum Configuration Generation Discussion

## Overview

This document summarizes the investigation into ONNX configuration requirements for Optimum inference and proposes solutions for integrating config generation into the HTP exporter.

## Key Findings

### 1. ONNX Config Specification Location

The ONNX configuration specification exists in the Optimum library:
- **Documentation**: `external/optimum/docs/source/exporters/onnx/package_reference/configuration.mdx`
- **Source Code**: `external/optimum/optimum/exporters/onnx/base.py` (1019 lines)

### 2. Three-Level Class Hierarchy

Optimum uses a sophisticated 3-level configuration hierarchy:

```
Base Classes (Abstract/Generic)
    ↓
Middle-end Classes (Modality-aware)
    ↓
Model-specific Classes (e.g., BertOnnxConfig)
```

#### Base Classes
- `OnnxConfig`: Core abstract class with fundamental features
- `OnnxConfigWithPast`: Adds support for past key values (for autoregressive models)
- `OnnxSeq2SeqConfigWithPast`: Specialized for sequence-to-sequence models

#### Middle-end Classes
- **Text**: `TextEncoderOnnxConfig`, `TextDecoderOnnxConfig`, `TextSeq2SeqOnnxConfig`
- **Vision**: `VisionOnnxConfig`
- **Multi-modal**: `TextAndVisionOnnxConfig`

#### Model-specific Classes
- `BertOnnxConfig`, `GPT2OnnxConfig`, `T5OnnxConfig`, etc.
- Each implements specific input/output specifications for their architecture

### 3. Critical Distinction: Export Config vs Inference Config

**IMPORTANT**: There are THREE different types of configurations:

1. **OnnxConfig** (from `optimum.exporters.onnx`)
   - Used ONLY during PyTorch → ONNX export
   - Defines inputs, outputs, dynamic axes
   - NOT used during inference

2. **ORTConfig** (from `optimum.onnxruntime.configuration`)
   - Used for ONNX Runtime optimization and quantization
   - Optional for basic inference
   - Contains optimization settings

3. **HuggingFace Config** (e.g., `BertConfig`, `GPT2Config`)
   - The standard model configuration
   - **THIS is what's needed for Optimum inference**
   - Stored as `config.json` alongside the ONNX model

### 4. What Optimum Actually Needs for Inference

For basic ONNX inference with Optimum (no quantization/optimization):

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# Only needs config.json (HuggingFace config) in the directory
model = ORTModelForSequenceClassification.from_pretrained("./model_dir/")
```

Required files in `model_dir/`:
- `model.onnx` - The exported ONNX model
- `config.json` - HuggingFace model configuration (NOT OnnxConfig)
- (Optional) Tokenizer files for convenience

## Implementation Proposals

### Proposal 1: Lightweight Integration ✅ (Recommended)

Add config saving to existing HTP exporter with minimal changes:

```python
# In HTPExporter.export(), after ONNX export completes:
if model_name_or_path and not model_name_or_path.endswith('.pt'):
    from transformers import AutoConfig, AutoTokenizer
    from pathlib import Path
    
    # Save HuggingFace config
    config_dir = Path(output_path).parent
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.save_pretrained(config_dir)
        
        # Optionally save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.save_pretrained(config_dir)
        except:
            pass  # Some models don't have tokenizers
    except:
        # Model might be a local path without config
        pass
```

**Pros:**
- Minimal code changes (< 10 lines)
- Immediate Optimum compatibility
- Works with all HuggingFace models
- No new dependencies

**Cons:**
- Only works when model_name is provided
- No config for custom models

### Proposal 2: Config Generation from Model Analysis

Generate config by analyzing model structure:

```python
class OptimumConfigGenerator:
    def generate_config_from_model(self, model, hierarchy_data):
        # Analyze model structure to infer config
        config = self._infer_config_from_hierarchy(hierarchy_data)
        return config
```

**Pros:**
- Works with custom models
- No dependency on model_name

**Cons:**
- Complex inference logic
- May miss model-specific fields
- Requires maintenance

### Proposal 3: Deep Optimum Integration

Use Optimum's export functionality with HTP enhancements:

```python
from optimum.exporters.onnx import main_export

# Use Optimum export, then apply HTP tagging
main_export(model_name, output_dir)
# Load ONNX and add HTP metadata
```

**Pros:**
- Full Optimum compatibility guaranteed
- Leverages robust export pipeline

**Cons:**
- Requires optimum as dependency
- May conflict with custom logic

### Proposal 4: Hybrid Approach with Config Enhancement

Combine HTP export with intelligent config generation:

```python
class HTPConfigBuilder:
    def build_config(self, model=None, model_name=None):
        config = {}
        
        # Try loading from HuggingFace
        if model_name:
            config = AutoConfig.from_pretrained(model_name).to_dict()
        
        # Enhance with model analysis
        if model and hasattr(model, 'config'):
            config.update(model.config.to_dict())
        
        # Add HTP metadata
        config['htp_metadata'] = {...}
        
        return config
```

**Pros:**
- Best of both worlds
- Graceful fallbacks
- Enriched metadata

**Cons:**
- More complex implementation

## Recommendation

**Use Proposal 1 (Lightweight Integration)** for immediate implementation:

1. **Simplicity**: Only 10 lines of code to add
2. **Immediate Value**: Enables Optimum usage right away
3. **No Breaking Changes**: Fully backward compatible
4. **Future-Proof**: Can enhance later with Proposal 4 if needed

## Implementation Steps

1. Add config saving to `HTPExporter.export()` method
2. Automatic directory structure creation when exporting:
   ```
   output_dir/
   ├── model.onnx
   ├── config.json
   ├── tokenizer.json
   └── tokenizer_config.json
   ```
3. No need for `--save-config` flag (automatic and non-intrusive)
4. Document in README

## Example Workflow

After implementation:

```bash
# Export with automatic config saving
modelexport export --model bert-base-uncased --output output_dir/model.onnx

# Directory now contains everything for Optimum
ls output_dir/
# model.onnx  config.json  tokenizer.json  tokenizer_config.json

# Use with Optimum
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained("output_dir/")
```

## Testing Checklist

- [ ] Export creates config.json
- [ ] Optimum can load exported model
- [ ] Inference produces correct results
- [ ] Tokenizer compatibility verified
- [ ] Backward compatibility maintained

## Conclusion

The investigation revealed that:
1. OnnxConfig is for export only, not inference
2. Optimum inference only needs HuggingFace config.json
3. Simple config saving (Proposal 1) provides immediate value
4. The HTP exporter can be Optimum-compatible with minimal changes

This enables users to leverage the HTP hierarchy preservation while maintaining full compatibility with the Optimum ecosystem.