# ADR-015: Self-Contained ONNX Models

## Status

Accepted

## Context

Currently, deploying ONNX models requires managing multiple files (model, tokenizer configs, vocab files, etc.) and ensuring version compatibility across environments. This creates:

- **Deployment complexity**: 10+ files needed for a single model
- **Reproducibility issues**: Different preprocessing across environments
- **Offline limitations**: Cannot deploy without internet access to download configs
- **Documentation burden**: Extensive docs needed for preprocessing steps

This is analogous to the "works on my machine" problem that Docker solved for applications.

## Decision

We will implement **self-contained ONNX models** that embed all necessary metadata directly in the ONNX file using the standard `metadata_props` field.

A self-contained ONNX model includes:
- Model weights & computation graph (standard ONNX)
- Complete model configuration
- Feature engineering metadata (tokenizer/processor configs)
- Task identification for auto-detection
- Version information for compatibility

## Rationale

### Analogy with Docker

| Aspect | Docker Container | Self-Contained ONNX |
|--------|-----------------|---------------------|
| **Problem** | "Works on my machine" | "Works with my preprocessing" |
| **Solution** | Container with all dependencies | ONNX with all metadata |
| **Deployment** | Single image | Single file |
| **Portability** | Run anywhere with Docker | Run anywhere with ONNX Runtime |
| **Overhead** | ~50-200MB runtime | ~50-200KB metadata |

### Target User Experience

```python
# Current: Complex multi-file deployment
session = ort.InferenceSession("model.onnx")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Needs internet!
# Manual preprocessing, no task info, error-prone...

# Future: Single file, works offline
from modelexport.inference import pipeline
pipe = pipeline(model="model.onnx")  # Auto-detects everything!
result = pipe("This movie is great!")
# Output: [{"label": "POSITIVE", "score": 0.99}]
```

## Technical Approach

### Metadata Structure

```python
{
    "hf_model_config": {...},           # Complete model config
    "hf_tokenizer_config": {...},       # Tokenizer configuration  
    "feature_engineering.config": {...}, # Processing requirements
    "hf_pipeline_task": "text-classification",
    "hf_metadata_version": "2.0",
    "hf_transformers_version": "4.36.0"
}
```

### Configuration Preservation Strategy

We adopt **Complete Configuration Preservation** as a core principle:

- **Always preserve complete original configurations** using `.to_dict()`
- **Honor every field** in the original configuration
- **Trust that every field exists for a reason**
- **No filtering or selection** of "essential" fields

#### Rationale for Complete Preservation

1. **Respect Original Design**: Every field in HuggingFace configurations exists for a reason. The model authors included these fields deliberately.
2. **Future Compatibility**: New fields added by HuggingFace will automatically be preserved without requiring updates.
3. **Simplicity**: No need to maintain lists of "essential" fields. Implementation is just `.to_dict()`.
4. **Correctness**: Filtering risks breaking functionality that depends on fields we deemed "non-essential".
5. **Zero Maintenance**: No need to update field lists when HuggingFace adds new configuration options.

#### Implementation

```python
def discover_model_config(self, model: PreTrainedModel) -> Dict[str, Any]:
    """Always preserve complete model configuration."""
    return model.config.to_dict()  # Complete, unmodified

def discover_processor_config(self, processor: Any) -> Dict[str, Any]:
    """Always preserve complete processor configuration."""
    if hasattr(processor, 'to_dict'):
        return processor.to_dict()  # Complete, unmodified
    return {}
```

Typical metadata sizes with complete preservation:
- Model config: 2-10KB
- Tokenizer config: 2-5KB (without vocabulary)
- Image processor config: 1-3KB
- **Total overhead: 5-20KB** (negligible vs model weights)

### Implementation Phases

1. **Metadata Discovery**: Utilities to discover complete configs from HF models
2. **Embedding Infrastructure**: Complete config preservation in ONNX metadata_props
3. **Inference Support**: Auto-detection and processor reconstruction
4. **Testing & Documentation**: Validation and deployment guides

## Consequences

### Positive

- **90% reduction** in deployment complexity (10+ files → 1 file)
- **Offline capability**: No internet required after export
- **Guaranteed reproducibility**: Same behavior across all environments
- **Self-documenting**: Models explain their own requirements
- **80% reduction** in configuration-related errors

### Negative

- **File size increase**: 5-20KB typical metadata overhead (50-200KB with vocabulary)
- **Code complexity**: Additional metadata handling logic
- **Version management**: Metadata format evolution over time

### Mitigations

- Compression for configs >10KB
- Complete preservation simplifies code (no filtering logic)
- Version tracking in metadata
- Graceful fallback for missing metadata

## Alternatives Considered

1. **External metadata files**: Rejected - defeats single-file deployment
2. **Custom ONNX extensions**: Rejected - breaks compatibility
3. **Cloud metadata service**: Rejected - requires internet, adds latency
4. **Embedding in computation graph**: Deferred - complex implementation

## References

- [ONNX Metadata Properties Specification](https://github.com/onnx/onnx/blob/main/docs/MetadataProps.md)
- [Linear Task TEZ-157: P0 Self-Contained ONNX](https://linear.app/tezheng/issue/TEZ-157/)
- [Detailed Requirements Document](../experiments/tez-153_onnx-infer-2/docs/requirements.md)
- [Feature Engineering Metadata Design](../experiments/tez-153_onnx-infer-2/docs/feature-engineering-metadata.md)

## Decision

**Approved as P0 priority** - This is a fundamental goal of the modelexport project.

## Date

2024-08-14

## Review Date

2025-02-14 (6 months after implementation)