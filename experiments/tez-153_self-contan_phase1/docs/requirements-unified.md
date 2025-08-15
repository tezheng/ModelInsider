# ONNX Inference Phase 2 - Unified Requirements Document

## Executive Summary

This document defines the requirements for implementing **self-contained ONNX models** - a P0 (critical) feature that will revolutionize ML model deployment by embedding all necessary metadata directly into ONNX files, enabling single-file deployment with zero external dependencies.

Phase 2 enhances the ONNX export and inference pipeline to embed model configuration and processor metadata directly into ONNX files, enabling automatic task detection and configuration during inference without explicit parameters.

## Problem Statement

### Current State - The Deployment Nightmare

Currently, deploying ONNX models requires managing multiple files (model, tokenizer configs, vocab files, etc.) and ensuring version compatibility across environments. This creates:

- **Deployment complexity**: 10+ files needed for a single model
- **Reproducibility issues**: Different preprocessing across environments  
- **Offline limitations**: Cannot deploy without internet access to download configs
- **Documentation burden**: Extensive docs needed for preprocessing steps

```python
# CURRENT: Using ONNX model requires extensive manual setup
# User must know/provide EVERYTHING manually

# 1. Load model (just weights, no context)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")

# 2. User must KNOW this is BERT for text classification
# (Nothing in the ONNX file tells them this!)

# 3. User must MANUALLY setup preprocessing
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Requires internet!

# 4. Manual inference with exact input names
text = "This movie is great!"
inputs = tokenizer(text, return_tensors="np")
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})

# 5. Manual postprocessing - no label mapping!
import numpy as np
predictions = np.argmax(outputs[0], axis=-1)
```

### Vision - The Future State

A **self-contained ONNX model** is a single ONNX file that includes:

1. **Model Weights & Computation Graph** (standard ONNX)
2. **Complete Model Configuration** (architecture, parameters, versions)
3. **Feature Engineering Metadata** (tokenizer config, image processor settings, etc.)
4. **Task Identification** (what the model does: classification, generation, etc.)
5. **Version Information** (for compatibility verification)

```python
# FUTURE: Self-contained ONNX model "just works"
from modelexport.inference import pipeline

# Load model - task is AUTO-DETECTED from metadata!
pipe = pipeline(model="model.onnx")  # That's it!

# Use it immediately - preprocessing is automatic!
result = pipe("This movie is great!")
# Output: [{"label": "POSITIVE", "score": 0.99}]

# No internet required! No external files! No configuration!
```

## Background

Phase 1 successfully implemented:
- ONNX-first design with `ONNXAutoModel.from_model()` API
- `ONNXAutoProcessor` with factory pattern for different modalities
- Production-ready inference pipeline in `modelexport/inference/`
- Support for text, vision, and audio tasks

Phase 2 aims to make the pipeline smarter by embedding necessary metadata in the ONNX file itself.

## Important Clarification: Two-Stage Process

**"Self-contained" is for INFERENCE, not for metadata discovery**:
1. **Metadata Embedding Stage**: Requires HuggingFace model to discover and extract complete configurations
2. **Inference Stage**: Self-contained ONNX works offline with zero dependencies

This is by design - we need the authoritative source (HF model) to get correct, complete metadata.

## Functional Requirements

### FR1: Model Configuration Export

**Priority**: P0 (Critical)  
**Description**: Preserve and embed complete HuggingFace model configuration into ONNX exports

**Acceptance Criteria**:
- Extract complete model.config using `.to_dict()` from HuggingFace models
- Preserve ALL original fields without modification or filtering
- Store full configuration as JSON in ONNX metadata properties
- Support all HuggingFace model types and config versions
- Honor the principle: every field exists for a reason

**Implementation**:
```python
# Complete Configuration Preservation - no filtering
model_config = model.config.to_dict()  # ALL fields preserved
metadata["hf_model_config"] = json.dumps(model_config)
```

### FR2: Feature Engineering Metadata Export

**Priority**: P0 (Critical)  
**Description**: Preserve and embed complete processor configurations into ONNX exports

**Acceptance Criteria**:
- Extract complete tokenizer config for text models
- Extract complete image processor config for vision models  
- Extract complete feature extractor config for audio models
- Preserve ALL processor configurations without modification
- Store configs as JSON in ONNX metadata_props
- Support vocabulary embedding with `--include-vocab` flag

**Core Principle**: ALL Feature Engineering metadata must be handled - partial metadata means broken inference.

**Implementation**:
```python
# Complete processor preservation
if hasattr(processor, "tokenizer"):
    tokenizer_config = processor.tokenizer.to_dict()  # Complete config
    metadata["hf_tokenizer_config"] = json.dumps(tokenizer_config)
    metadata["feature_engineering.type"] = "tokenizer"
```

### FR3: Task Type Embedding

**Priority**: P0 (Critical)  
**Description**: Embed pipeline task type in ONNX model for automatic detection

**Acceptance Criteria**:
- Determine task type from model architecture
- Store task type in standardized format
- Support all HuggingFace pipeline tasks
- Include task-specific parameters (labels, etc.)

**Supported Tasks**:
- Text: text-classification, token-classification, question-answering, text-generation, summarization, translation, fill-mask
- Vision: image-classification, image-segmentation, object-detection
- Audio: audio-classification, automatic-speech-recognition
- Multimodal: zero-shot-image-classification, image-to-text

### FR4: Auto-Task Detection in Pipeline

**Priority**: P0 (Critical)  
**Description**: Modify inference pipeline to automatically detect and configure tasks from ONNX metadata

**Acceptance Criteria**:
- Read embedded metadata on model load
- Auto-configure pipeline based on task type
- Auto-select appropriate processor
- Maintain < 100ms overhead for metadata parsing
- Provide clear warnings when vocabulary is missing

**API Evolution**:
```python
# Current (Phase 1)
pipe = pipeline(task="text-classification", model="model.onnx")

# Target (Phase 2)
pipe = pipeline(model="model.onnx")  # Task auto-detected!

# With override capability
pipe = pipeline(model="model.onnx", task="sentiment-analysis")
```

### FR5: Backward Compatibility

**Priority**: P0 (Critical)  
**Description**: Maintain full backward compatibility with Phase 1 ONNX exports

**Acceptance Criteria**:
- Support ONNX files without embedded metadata
- Fallback to explicit task specification
- Provide clear warnings for missing metadata
- No regression in existing functionality
- 100% backward compatibility verified

### FR6: Complete Configuration Preservation

**Priority**: P0 (Critical)  
**Description**: The system SHALL preserve ALL configuration fields without filtering

**Core Principle**: 
- Honor every field - they exist for a reason
- No filtering - always use `.to_dict()` for complete preservation
- Trust original design from HuggingFace

**CLI Interface**:
```bash
# Stage 1: Export ONNX (any method)
modelexport export model_name output.onnx

# Stage 2: Embed metadata (requires HF model for discovery)
modelexport embed-metadata output.onnx model_name

# Include vocabulary for complete feature engineering
modelexport embed-metadata output.onnx model_name --include-vocab
```

### FR7: Feature Engineering Metadata Standards

**Priority**: P0 (Critical)  
**Description**: Implement feature engineering metadata following ONNX standards and best practices

**Acceptance Criteria**:
- Follow ONNX metadata_props conventions for feature engineering information
- Support embedding feature engineering as part of ONNX graph (future enhancement)
- Store feature engineering configs in standardized format
- Support image, text, and audio feature engineering metadata
- Allow optional embedding of all configs

**ONNX Standard Metadata Properties**:
```python
# Store feature engineering configs in metadata_props
metadata_props = {
    "feature_engineering.tokenizer_config": json.dumps(tokenizer_config),
    "feature_engineering.image_processor_config": json.dumps(image_config),
    "feature_engineering.feature_extractor_config": json.dumps(audio_config),
    # ONNX standard image metadata
    "Image.BitmapPixelFormat": "Rgb8",  # or "Gray8", "Rgba8"
    "Image.ColorSpaceGamma": "sRGB",    # or "Linear"
    "Image.NominalPixelRange": "Normalized_0_1",  # or "NominalRange_0_255"
}
```

### FR8: Processor Reconstruction

**Priority**: P0 (Critical)  
**Description**: Enable processor reconstruction from embedded metadata

**Acceptance Criteria**:
- If vocabulary embedded: Full tokenizer reconstruction
- If no vocabulary: Limited proxy with clear warning
- Support all processor types (tokenizer, image, audio)
- Guide users to use `--include-vocab` for production

### FR9: Metadata Validation and Schema Framework

**Priority**: High  
**Description**: Implement robust metadata validation with schema framework

**Acceptance Criteria**:
- Validate metadata completeness on embedding
- Check metadata integrity on reading
- Provide clear error messages for issues
- Handle corrupted metadata gracefully
- Maintain schema for validation while preserving complete configs

**Schema Validation Implementation**:
```python
# Maintain schema for validation only (not filtering)
config_schema = load_schema_for_model_type(model.config.model_type)
validate_config_against_schema(model_config, config_schema)
# Still preserve complete config even if validation warns
```

### FR10: HTP Integration

**Priority**: P1 (High)  
**Description**: Integrate metadata embedding with existing HTP exporter

**Acceptance Criteria**:
- Extend HTP exporter to capture metadata during export
- Maintain hierarchy tag functionality alongside metadata
- Support both `--clean-onnx` and metadata embedding
- Ensure no regression in HTP functionality
- Allow metadata embedding as post-processing step

## Non-Functional Requirements

### NFR1: Performance

**Priority**: P0 (Critical)

- **Metadata extraction**: < 500ms additional export time
- **Metadata parsing**: < 100ms overhead on model load
- **Memory usage**: < 10MB additional for metadata handling
- **Inference speed**: No degradation from metadata presence
- **Export time**: < 500ms additional for metadata embedding

### NFR2: Size Constraints

**Priority**: High

- **Standard metadata**: < 20KB overhead (without vocabulary)
- **With vocabulary**: < 500KB for 95% of models
- **Compression**: Automatic for configs > 10KB

### NFR3: Compatibility

**Priority**: High

- **ONNX Runtime**: Compatible with 1.14+
- **ONNX Spec**: Support opset 14-18
- **Transformers**: Work with all HuggingFace transformers 4.30+
- **Existing Tools**: Compatible with existing ONNX tools

### NFR4: Reliability

**Priority**: High

- **Metadata integrity**: Validation and checksums
- **Error handling**: Graceful degradation
- **Fallback mechanisms**: Work without metadata
- **Logging**: Comprehensive for debugging

### NFR5: Documentation

**Priority**: Medium

- **User guides**: Migration from Phase 1
- **API reference**: Complete documentation
- **Examples**: Common deployment scenarios
- **Troubleshooting**: Common issues and solutions

## Technical Architecture

### Metadata Structure Specification

The embedded metadata follows this structure:
```python
{
    "hf_metadata_version": "2.0",
    "hf_export_timestamp": "2024-01-15T10:30:00Z",
    "hf_transformers_version": "4.36.0",
    "hf_model_name": "bert-base-uncased",
    "hf_pipeline_task": "text-classification",
    "hf_model_config": {
        # Complete model.config.to_dict() - ALL fields preserved
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"],
        "hidden_size": 768,
        "num_hidden_layers": 12,
        # ... all other config fields ...
    },
    "hf_tokenizer_config": {
        # Complete tokenizer config - ALL fields preserved
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": true,
        "max_length": 512,
        # ... all other tokenizer fields ...
    },
    "hf_processor_type": "tokenizer",
    "hf_labels": ["NEGATIVE", "POSITIVE"]
}
```

### Implementation Architecture

```python
class MetadataManager:
    """Central manager for all metadata operations"""
    def __init__(self):
        self.discovery = MetadataDiscovery()
        self.validator = MetadataValidator()
        self.embedder = ONNXMetadataEmbedder()
        self.reader = ONNXMetadataReader()
    
    def process_model(self, onnx_path: str, model_name: str):
        # Stage 1: Discovery (requires HF model)
        metadata = self.discovery.discover_from_hf(model_name)
        
        # Stage 2: Validation
        self.validator.validate(metadata)
        
        # Stage 3: Embedding
        self.embedder.embed(onnx_path, metadata)

class ProcessorReconstructor:
    """Reconstruct processors from embedded metadata"""
    @classmethod
    def from_metadata(cls, metadata: Dict) -> Any:
        if "hf_tokenizer_config" in metadata:
            if "vocab" in metadata:  # Full reconstruction
                return cls._reconstruct_full_tokenizer(metadata)
            else:  # Limited proxy with warning
                warnings.warn("Limited tokenizer without vocabulary")
                return cls._create_proxy_tokenizer(metadata)
```

## Technical Constraints

1. **ONNX Metadata Storage**: Use ONNX `metadata_props` field for storing configuration
2. **JSON Serialization**: Store complete original configs as JSON strings without modification
3. **Size Limitations**: Keep metadata under 1MB to avoid ONNX file bloat (use compression if needed)
4. **Schema Versioning**: Track transformers library version for compatibility
5. **Config Preservation**: Never modify or filter original HuggingFace configs
6. **Two-Stage Process**: Export ONNX first, then embed metadata (requires HF model for discovery)

## Implementation Phases

### Phase 2.1: Foundation (Weeks 1-2)
- Create metadata package structure
- Implement MetadataDiscovery (complete configs)
- Implement ONNXMetadataEmbedder
- Implement ONNXMetadataReader
- Validate with bert-tiny

### Phase 2.2: Standalone Metadata Tool (Weeks 3-4)
- Create standalone metadata embedding tool
- Preserve complete configurations without filtering
- Implement compression for large metadata
- Ensure compatibility with any ONNX export method

### Phase 2.3: Processor Reconstruction (Weeks 5-6)
- Implement ProcessorReconstructor
- Handle vocabulary presence/absence
- Create processor proxies
- Implement caching

### Phase 2.4: Pipeline Enhancement (Weeks 7-8)
- Implement auto-task detection
- Add fallback mechanisms
- Ensure backward compatibility
- Update ONNXAutoModel

### Phase 2.5: Testing & Documentation (Weeks 9-10)
- Test 20+ model architectures
- Performance benchmarking
- User documentation
- Migration guide

### Phase 2.6: Production Release (Week 11)
- Final validation
- Gradual rollout
- Monitor adoption

### Phase 2.7: Stretch Goals (Future)
**Priority**: P2 (Nice-to-have)
- **Schema Validation Registry**: Centralized schema management
- **Feature Engineering Graph Embedding**: Embed preprocessing as ONNX ops
- **Custom ONNX Operators**: Define custom ops for complex preprocessing
- **End-to-End Models**: Include pre/post-processing in ONNX graph
- **Model Cards Integration**: Embed model documentation
- **Multi-Model Pipelines**: Support chained model metadata

## Success Metrics

### Quantitative Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Deployment Steps** | 90% reduction (10+ files → 1 file) | Count files required |
| **Configuration Errors** | 80% reduction | Support ticket analysis |
| **Adoption Rate** | 60% within 6 months | Download/usage statistics |
| **Metadata Overhead** | < 20KB for 95% of models (without vocab) | File size analysis |
| **Performance Impact** | < 100ms parsing time | Benchmark testing |
| **Backward Compatibility** | 100% verified | Regression testing |
| **User Satisfaction** | 30% reduction in support requests | Support metrics |
| **Export Time** | < 500ms additional overhead | Performance benchmarks |
| **Memory Usage** | < 10MB additional for metadata | Memory profiling |
| **Test Coverage** | > 90% code coverage | Coverage reports |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Metadata bloat** | Medium | Automatic compression for >10KB |
| **Version incompatibility** | Medium | Version tracking, fallbacks |
| **Breaking changes** | High | Extensive testing, gradual rollout |
| **Performance degradation** | Medium | Caching, lazy loading |
| **Missing vocabulary** | High | Clear warnings, guide to use --include-vocab |

## Dependencies

- HuggingFace Transformers 4.30+
- ONNX 1.14+
- ONNX Runtime 1.14+
- Python 3.8+
- Existing Phase 1 inference pipeline

## Acceptance Criteria

- [ ] Complete configuration preservation implemented
- [ ] All functional requirements implemented
- [ ] All non-functional requirements met
- [ ] 100% backward compatibility verified
- [ ] Documentation complete
- [ ] Test coverage > 90%
- [ ] Performance benchmarks pass
- [ ] Code review approved
- [ ] Integration tests pass

## Use Cases

### UC1: Data Scientist Sharing Model
**Before**: Share 10+ files, write documentation, provide setup scripts
**After**: Share single ONNX file that "just works"

### UC2: Production Deployment  
**Before**: Complex CI/CD to manage multiple artifacts and versions
**After**: Deploy single file with built-in configuration

### UC3: Edge Device Deployment
**Before**: Manual configuration, limited connectivity for downloads
**After**: Single offline-capable file with everything embedded

### UC4: Long-term Model Storage
**Before**: Risk of config drift, lost preprocessing steps over time
**After**: Self-documenting model preserved perfectly

## Appendix: Key Decisions

### Decision 1: Complete Configuration Preservation
We preserve ALL configuration fields exactly as provided by HuggingFace:
- No filtering or selection of "essential" fields
- Always use `.to_dict()` for complete extraction
- Trust that every field exists for a reason
- Typical overhead: 5-20KB (negligible vs model weights)

### Decision 2: Two-Stage Process
Metadata embedding is separate from ONNX export:
- Stage 1: Export ONNX using any method
- Stage 2: Add metadata (requires HF model for discovery)
- This ensures we get authoritative, complete configurations

### Decision 3: Vocabulary Handling
Feature engineering metadata requires special handling:
- Without vocabulary: Limited functionality with warnings
- With vocabulary (`--include-vocab`): Full reconstruction capability
- ALL FE metadata must be handled for correct inference