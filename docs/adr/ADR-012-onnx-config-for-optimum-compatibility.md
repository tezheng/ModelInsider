# ADR-012: ONNX Configuration Strategy for Optimum Compatibility

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Proposed | 2025-08-06 | TEZ-144 Implementation Team | HTP Team, Optimum Users | All ModelExport Users |

## Context and Problem Statement

The ModelExport HTP (Hierarchical Trace-and-Project) exporter [1] successfully creates ONNX models with preserved module hierarchy metadata. However, these exported models cannot be directly used with HuggingFace Optimum's inference APIs (ORTModel classes) [2] without proper configuration files. Users currently need to manually create configuration files to use exported models with Optimum, creating friction in the export-to-inference workflow.

We need to support two scenarios:
1. Models from HuggingFace Hub (public models)
2. In-house HF-compatible models (local/private models with config.json)

## Decision Drivers

- **Simplicity**: Minimize code changes and file management complexity
- **Compatibility**: Support both HF Hub models and in-house HF-compatible models
- **Universality**: Work with any HuggingFace-compatible model
- **User Experience**: Provide zero-friction path from export to inference
- **Backward Compatibility**: Don't break existing HTP export functionality
- **Performance**: Enable 2-3x inference speedup through ONNX Runtime [3]
- **Deployment Flexibility**: Support online, offline, and private model scenarios

## Considered Options

0. **Metadata-Only Approach** - Store HF model ID in ONNX metadata, load config dynamically
1. **Lightweight Config Copying** - Copy HuggingFace config and preprocessing components during export
2. **Smart Hybrid Approach** - Use metadata for HF Hub models, copy configs for local models
3. **Config Generation from Model Analysis** - Analyze model structure to generate config
4. **Deep Optimum Integration** - Use Optimum's export pipeline with HTP post-processing

## Decision Outcome

**Chosen option**: Option 2 - Smart Hybrid Approach

We will implement an intelligent system that:
- **For HF Hub models**: Store model ID/version in ONNX metadata, load config dynamically at inference
- **For local/in-house models**: Copy config.json and preprocessing components alongside ONNX export

The system automatically detects which approach to use based on whether the model is from HuggingFace Hub or a local path.

### Rationale

This approach provides the best of both worlds:
- **Minimal storage** for public models (metadata only)
- **Full support** for in-house models (config copying)
- **Automatic detection** - no user configuration needed
- **Single ONNX file** for Hub models, complete package for local models
- **Version tracking** for reproducibility

### Consequences

**Positive:**
- Works with ALL HuggingFace-compatible models (Hub and local)
- Minimal changes for Hub models (~10 lines)
- Full offline support for in-house models
- Automatic detection requires no user intervention
- Version tracking ensures reproducibility
- Clean deployment for both scenarios

**Negative:**
- Slightly more complex implementation (~50 lines total)
- Local models require config files alongside ONNX
- Need to detect whether model is from Hub or local

**Neutral:**
- Different deployment patterns for Hub vs local models
- Future Optimum versions could directly support this pattern

## Implementation Notes

### Automatic Detection Logic

```python
def is_hub_model(model_name_or_path: str) -> bool:
    """
    Detect if a model is from HuggingFace Hub or local path.
    
    Hub models typically follow patterns like:
    - "bert-base-uncased"
    - "google/flan-t5-xl"
    - "meta-llama/Llama-2-7b"
    
    Local models are paths like:
    - "./my_model"
    - "/path/to/model"
    - "path/to/model" (without org/model pattern)
    """
    import os
    from pathlib import Path
    
    # If it's an existing local path, it's definitely local
    if os.path.exists(model_name_or_path):
        return False
    
    # Check if it follows Hub naming pattern (org/model or just model-name)
    # Hub models don't have path separators except for the org/model pattern
    path_parts = model_name_or_path.replace("\\", "/").split("/")
    
    # Local paths with ./ or ../ or absolute paths
    if model_name_or_path.startswith(("./", "../", "/")) or "\\" in model_name_or_path:
        return False
    
    # Hub pattern: either "model-name" or "org/model-name"
    if len(path_parts) <= 2 and not any(p.startswith(".") for p in path_parts):
        # Try to verify it's actually on the Hub (optional)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.model_info(model_name_or_path)
            return True
        except:
            # If we can't verify, assume local
            return False
    
    return False
```

### Step 1: Enhanced Export with Smart Detection

```python
# In HTPExporter.export() method, after torch.onnx.export:
# (Reference: modelexport/strategies/htp/htp_exporter.py)

if model_name_or_path:
    import onnx
    from pathlib import Path
    
    # Load the exported ONNX model
    onnx_model = onnx.load(output_path)
    output_dir = Path(output_path).parent
    
    # Detect if this is a Hub model or local model
    if is_hub_model(model_name_or_path):
        # HF Hub Model: Store metadata only
        logger.info(f"Detected HuggingFace Hub model: {model_name_or_path}")
        
        # Add HuggingFace model source to metadata
        meta_source = onnx_model.metadata_props.add()
        meta_source.key = "hf_hub_id"
        meta_source.value = model_name_or_path
        
        # Try to get model revision/version for reproducibility
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info(model_name_or_path)
            if model_info.sha:
                meta_version = onnx_model.metadata_props.add()
                meta_version.key = "hf_hub_revision"
                meta_version.value = model_info.sha[:8]  # Short SHA
                logger.info(f"Stored model revision: {model_info.sha[:8]}")
        except:
            pass
        
        # Mark as Hub model for inference
        meta_type = onnx_model.metadata_props.add()
        meta_type.key = "hf_model_type"
        meta_type.value = "hub"
        
    else:
        # Local/In-house Model: Copy configs
        logger.info(f"Detected local/in-house model: {model_name_or_path}")
        
        # Mark as local model
        meta_type = onnx_model.metadata_props.add()
        meta_type.key = "hf_model_type"
        meta_type.value = "local"
        
        # Store original path for reference
        meta_path = onnx_model.metadata_props.add()
        meta_path.key = "hf_original_path"
        meta_path.value = model_name_or_path
        
        # Copy config and preprocessing components
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.save_pretrained(output_dir)
            logger.info(f"Saved config.json to {output_dir}")
            
            # Try to save preprocessing components
            components_saved = []
            
            # Try AutoProcessor (for multimodal)
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_name_or_path)
                processor.save_pretrained(output_dir)
                components_saved.append("processor")
            except:
                pass
            
            # Try AutoTokenizer (for text models)
            if "processor" not in components_saved:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                    tokenizer.save_pretrained(output_dir)
                    components_saved.append("tokenizer")
                except:
                    pass
            
            # Try AutoImageProcessor (for vision)
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
                image_processor.save_pretrained(output_dir)
                components_saved.append("image_processor")
            except:
                pass
            
            # Try AutoFeatureExtractor (for audio)
            try:
                from transformers import AutoFeatureExtractor
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
                feature_extractor.save_pretrained(output_dir)
                components_saved.append("feature_extractor")
            except:
                pass
            
            if components_saved:
                logger.info(f"Saved preprocessing components: {', '.join(components_saved)}")
                
        except Exception as e:
            logger.warning(f"Could not save config for local model: {e}")
            logger.warning("User will need to provide config manually for inference")
    
    # Update producer info
    onnx_model.producer_name = "ModelExport-HTP"
    onnx_model.producer_version = "1.0.0"
    onnx_model.domain = "com.modelexport.htp"
    
    # Save the model with metadata
    onnx.save(onnx_model, output_path)
```

### Step 2: Universal Config Loading Function

```python
# New file: modelexport/utils/optimum_loader.py

def load_hf_components_from_onnx(onnx_path: str) -> tuple[Any, Any]:
    """
    Load HuggingFace config and preprocessing components from ONNX.
    
    Handles both:
    1. Hub models - loads from HF Hub using metadata
    2. Local models - loads from co-located config files
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Tuple of (config, preprocessor)
    """
    import onnx
    from pathlib import Path
    from transformers import (
        AutoConfig, AutoProcessor, AutoTokenizer,
        AutoImageProcessor, AutoFeatureExtractor
    )
    
    # Load ONNX model and extract metadata
    onnx_model = onnx.load(onnx_path)
    onnx_dir = Path(onnx_path).parent
    
    # Extract metadata
    metadata = {}
    for prop in onnx_model.metadata_props:
        metadata[prop.key] = prop.value
    
    model_type = metadata.get("hf_model_type", "unknown")
    
    if model_type == "hub":
        # Hub model: Load from HuggingFace Hub
        hf_hub_id = metadata.get("hf_hub_id")
        hf_revision = metadata.get("hf_hub_revision")
        
        if not hf_hub_id:
            raise ValueError(
                "ONNX model marked as Hub model but missing hf_hub_id metadata"
            )
        
        # Load config from Hub
        config = AutoConfig.from_pretrained(hf_hub_id, revision=hf_revision)
        
        # Try to load preprocessor from Hub
        preprocessor = None
        for loader_cls in [AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor]:
            try:
                preprocessor = loader_cls.from_pretrained(hf_hub_id, revision=hf_revision)
                break
            except:
                continue
                
        return config, preprocessor
        
    elif model_type == "local":
        # Local model: Load from co-located files
        config_path = onnx_dir / "config.json"
        
        if not config_path.exists():
            raise ValueError(
                f"Local model but config.json not found at {config_path}. "
                "The model may have been moved without its config files."
            )
        
        # Load config from local file
        config = AutoConfig.from_pretrained(onnx_dir)
        
        # Try to load preprocessor from local files
        preprocessor = None
        for loader_cls in [AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor]:
            try:
                preprocessor = loader_cls.from_pretrained(onnx_dir)
                break
            except:
                continue
                
        return config, preprocessor
        
    else:
        # Unknown or legacy model
        raise ValueError(
            f"ONNX model has unknown type '{model_type}'. "
            "Was it exported with an older version of ModelExport? "
            "Please re-export the model."
        )
```

### Step 3: Seamless Usage

```python
# User code - works for both Hub and local models
from modelexport.utils.optimum_loader import load_hf_components_from_onnx
from optimum.onnxruntime import ORTModelForSequenceClassification
import tempfile
import shutil

# Load config and preprocessor (works for both Hub and local models!)
config, tokenizer = load_hf_components_from_onnx("model.onnx")

# Create temp directory for Optimum
with tempfile.TemporaryDirectory() as temp_dir:
    # Save config and tokenizer
    config.save_pretrained(temp_dir)
    if tokenizer:
        tokenizer.save_pretrained(temp_dir)
    
    # Copy ONNX model
    shutil.copy("model.onnx", f"{temp_dir}/model.onnx")
    
    # Load with Optimum
    model = ORTModelForSequenceClassification.from_pretrained(temp_dir)

# Use for inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```

## Validation/Confirmation

### Test Cases
1. **Hub Model Detection**: Verify correct identification of Hub vs local models
2. **Metadata Storage**: Confirm appropriate metadata for each model type
3. **Config Copying**: Verify configs are copied for local models
4. **Config Loading**: Test loading from both Hub and local sources
5. **Version Tracking**: Confirm revision tracking for Hub models
6. **Migration Path**: Ensure older exports can be handled gracefully

### Success Metrics
- Hub models require only single ONNX file
- Local models work completely offline
- Automatic detection works correctly 95%+ of the time
- No user configuration required for either scenario
- Clear error messages when configs are missing

## Detailed Analysis of Options

### Option 0: Metadata-Only Approach
- **Pros**: Simplest, single file
- **Cons**: Doesn't work for in-house models
- **Verdict**: Insufficient for enterprise use cases

### Option 1: Lightweight Config Copying
- **Pros**: Works offline, supports all models
- **Cons**: Unnecessary duplication for Hub models
- **Verdict**: Wasteful for common use case

### Option 2: Smart Hybrid Approach (Chosen)
- **Pros**: 
  - Optimal for both scenarios
  - Automatic detection
  - No user configuration
  - Single file for Hub models
  - Full offline for local models
- **Cons**: 
  - Slightly more complex (~50 lines)
  - Different patterns for different models
- **Verdict**: Best balance of simplicity and functionality

### Option 3: Config Generation
- **Pros**: Works without any source
- **Cons**: Complex, error-prone, cannot generate preprocessors
- **Verdict**: Over-engineered and unreliable

### Option 4: Deep Optimum Integration
- **Pros**: Guaranteed compatibility
- **Cons**: Loss of control, requires optimum dependency
- **Verdict**: Too invasive

## Related Decisions

- **ADR-007**: Root Module Hook Strategy - Foundation for HTP export [4]
- **ADR-010**: ONNX GraphML Format Specification - Alternative export format [5]
- **ADR-011**: Path-based Disambiguation - Hierarchy preservation approach [6]

## References

[1] HTP Exporter Implementation: `modelexport/strategies/htp/htp_exporter.py`  
[2] Optimum ORTModel: `external/optimum/optimum/onnxruntime/modeling_ort.py`  
[3] Performance Benchmarks: `experiments/tez-144_onnx_automodel_infer/README.md`  
[4] ADR-007: `docs/adr/ADR-007-root-module-hook-strategy.md`  
[5] ADR-010: `docs/adr/ADR-010-onnx-graphml-format-specification.md`  
[6] ADR-011: `docs/adr/ADR-011-path-based-disambiguation.md`  

## More Information

- [Linear Task TEZ-144](https://linear.app/tezheng/issue/TEZ-144)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [ONNX Python API](https://onnx.ai/onnx/api/python_api.html)
- [HuggingFace Offline Mode](https://huggingface.co/docs/transformers/installation#offline-mode)

## Notes

### Model Type Examples

**Hub Models** (use metadata only):
- `bert-base-uncased`
- `google/flan-t5-xl`
- `meta-llama/Llama-2-7b-hf`
- `openai/clip-vit-base-patch32`

**Local/In-house Models** (copy configs):
- `./my_fine_tuned_bert`
- `/mnt/models/company_bert_v2`
- `../models/custom_model`
- Local paths to HF-compatible models

### Deployment Patterns

**Hub Models**:
```
deployment/
└── model.onnx  # Single file, configs loaded dynamically
```

**Local Models**:
```
deployment/
├── model.onnx
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

### Error Handling

The implementation provides clear error messages:
- "Model not found on HuggingFace Hub" → Suggests checking model name
- "Config.json not found for local model" → Suggests re-exporting with configs
- "Cannot determine model type" → Suggests re-exporting with latest version

---
*Last updated: 2025-08-06*  
*Next review: 2025-11-06*