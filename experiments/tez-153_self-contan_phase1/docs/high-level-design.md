# Self-Contained ONNX Models - High-Level Design

## Executive Summary

This document presents the high-level design for implementing self-contained ONNX models - a revolutionary approach to ML model deployment that embeds all necessary metadata directly into ONNX files, enabling single-file deployment with zero external dependencies.

## Important Clarification: Two-Stage Process

**"Self-contained" is for INFERENCE, not for metadata discovery**:
1. **Metadata Embedding Stage**: Requires HuggingFace model to discover and extract complete configurations
2. **Inference Stage**: Self-contained ONNX works offline with zero dependencies

This is by design - we need the authoritative source (HF model) to get correct, complete metadata.

## System Context

### Current State Problem Analysis

```mermaid
graph LR
    subgraph "Current Deployment Complexity"
        M[model.onnx] --> D[Deployment]
        T[tokenizer files] --> D
        C[config.json] --> D
        V[vocab.txt] --> D
        P[preprocessing.py] --> D
        R[requirements.txt] --> D
        D --> E[Error-Prone Setup]
        E --> F[Failed Inference]
    end
```

**Pain Points:**

- 10+ files required for deployment
- Manual configuration needed
- Version compatibility issues
- Internet dependency for configs
- Preprocessing mismatch errors
- Documentation burden

### Target State Vision

```mermaid
graph LR
    subgraph "Self-Contained Solution"
        SC[model.onnx<br/>+metadata] --> P[Pipeline]
        P --> S[Successful Inference]
    end
```

**Benefits:**

- Single file deployment
- Zero configuration
- Offline operation
- Guaranteed reproducibility
- Self-documenting models

## Architectural Design

### System Architecture

```mermaid
graph TB
    subgraph "Stage 1: ONNX Export (Any Method)"
        HF1[HuggingFace Model] --> EX1[HTP Exporter]
        HF2[HuggingFace Model] --> EX2[Optimum Export]
        HF3[PyTorch Model] --> EX3[torch.onnx.export]
        EX1 --> ONNX1[model.onnx]
        EX2 --> ONNX1
        EX3 --> ONNX1
    end
    
    subgraph "Stage 2: Metadata Embedding (Isolated Tool)"
        HF[HuggingFace Model/Config] --> MD[Metadata Discovery]
        MD --> MM[Metadata Manager]
        MM --> TE[Tiered Embedder]
        ONNX1 --> TE
        TE --> ONNX2[Self-Contained ONNX]
    end
    
    subgraph "Storage Layer"
        ONNX2 --> MP[metadata_props]
        ONNX2 --> W[weights & graph]
    end
    
    subgraph "Inference Layer"
        ONNX2 --> MR[Metadata Reader]
        MR --> TD[Task Detector]
        MR --> PR[Processor Reconstructor]
        TD --> AP[Auto Pipeline]
        PR --> AP
        AP --> INF[Inference Engine]
    end
    
    subgraph "Compatibility Layer"
        Legacy[Legacy ONNX] --> FB[Fallback Handler]
        FB --> AP
    end
```

### Component Architecture

```mermaid
classDiagram
    class MetadataManager {
        +extract_from_hf(model, processor, level)
        +embed_in_onnx(onnx_path, metadata)
        +read_from_onnx(onnx_path)
        +validate(metadata)
    }
    
    class MetadataDiscovery {
        +discover_model_config(model)
        +discover_processor_config(processor)
        +infer_task_type(config)
        +get_embedding_level(size_constraint)
    }
    
    class ONNXMetadataEmbedder {
        +embed(onnx_model, metadata)
        +compress_if_needed(metadata)
        +calculate_overhead(metadata)
    }
    
    class ProcessorReconstructor {
        +from_metadata(metadata)
        +create_tokenizer(config)
        +create_image_processor(config)
        +create_feature_extractor(config)
    }
    
    class AutoPipeline {
        +from_onnx(path)
        +detect_task(metadata)
        +create_processor(metadata)
        +infer(inputs)
    }
    
    MetadataManager --> MetadataDiscovery
    MetadataManager --> ONNXMetadataEmbedder
    AutoPipeline --> ProcessorReconstructor
    AutoPipeline --> MetadataManager
```

## Data Model Design

### Metadata Structure

```yaml
metadata:
  # Version Control
  hf_metadata_version: "2.0"
  hf_export_timestamp: "ISO-8601"
  hf_transformers_version: "4.x.x"
  
  # Model Configuration (Complete, Unmodified)
  hf_model_config:
    # Complete output from model.config.to_dict()
    # All fields preserved without filtering
    architectures: [...]
    model_type: "..."
    task_specific_params: {...}
    # Every original HF config field preserved
  
  # Feature Engineering (Complete, Unmodified)
  feature_engineering:
    type: "tokenizer|image_processor|feature_extractor"
    # Complete output from processor.to_dict()
    config: {...}  # Complete processor config, unmodified
    
  # Task Identification
  hf_pipeline_task: "text-classification|..."
  hf_pipeline_config: {...}
  
  # Export Metadata
  export:
    hierarchy_preserved: true
    compression_used: false
    vocab_included: false
```

### Configuration Preservation Strategy

```mermaid
graph TD
    subgraph "Complete Configuration Preservation"
        MC[Model Config] --> COMPLETE[config.to_dict()]
        PC[Processor Config] --> COMPLETE
        COMPLETE --> EMBED[Embed in ONNX]
        EMBED --> |"5-20KB typical"| ONNX[Self-Contained Model]
    end
```

**Core Principle**: Honor all original configuration fields
- Always use `.to_dict()` for complete preservation
- Trust that every field exists for a reason  
- No filtering or "essential" field selection
- Typical overhead: 5-20KB (negligible vs model weights)

## Process Flow Design

### Metadata Embedding Process Flow (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant CLI as Embed CLI
    participant HF as HuggingFace Hub
    participant MD as MetadataDiscovery
    participant EMB as Embedder
    participant ONNX
    
    User->>CLI: embed-metadata model.onnx bert-base-uncased
    CLI->>HF: AutoModel.from_pretrained("bert-base-uncased")
    HF-->>CLI: model instance (with config)
    CLI->>HF: AutoProcessor.from_pretrained("bert-base-uncased")
    HF-->>CLI: processor instance
    CLI->>MD: discover_from_hf(model, processor)
    MD->>MD: model.config.to_dict()
    MD->>MD: processor.to_dict()
    MD-->>CLI: complete metadata dict
    CLI->>EMB: embed(onnx_path, metadata)
    EMB->>ONNX: Add to metadata_props
    ONNX-->>User: Self-contained model.onnx
```

### Core Loop Pseudo-Code

```python
# PSEUDO-CODE: How embed_metadata actually works internally

def embed_metadata(onnx_path: str, model_name: str):
    """
    Core workflow showing model loading requirement.
    """
    # Step 1: Load EXISTING ONNX model (from any exporter)
    onnx_model = onnx.load(onnx_path)
    
    # Step 2: Load HuggingFace model to get config
    # THIS IS THE KEY POINT - we need the model instance!
    hf_model = AutoModel.from_pretrained(model_name)
    hf_processor = AutoProcessor.from_pretrained(model_name)
    
    # Step 3: Discover metadata from loaded model
    discovery = MetadataDiscovery()
    metadata = discovery.discover_from_hf(
        model=hf_model,  # Requires loaded model with .config
        processor=hf_processor  # Optional, but needs loaded processor
    )
    # Internally does:
    #   metadata['hf_model_config'] = hf_model.config.to_dict()
    #   metadata['hf_processor_config'] = hf_processor.to_dict()
    
    # Step 4: Embed metadata into ONNX
    embedder = ONNXMetadataEmbedder()
    embedder.embed(onnx_model, metadata)
    
    # Step 5: Save enhanced ONNX
    onnx.save(onnx_model, onnx_path)
```

### Inference Process Flow

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Reader as Metadata Reader
    participant Detector as Task Detector
    participant Proc as Processor Creator
    participant Model as ONNX Runtime
    
    User->>Pipeline: pipeline(model="model.onnx")
    Pipeline->>Reader: Read metadata
    Reader-->>Pipeline: metadata dict
    Pipeline->>Detector: Detect task
    Detector-->>Pipeline: task type
    Pipeline->>Proc: Create processor
    Proc-->>Pipeline: processor instance
    User->>Pipeline: pipe("input text")
    Pipeline->>Model: Run inference
    Model-->>Pipeline: outputs
    Pipeline-->>User: results
```

## Interface Design

### CLI Interface

```bash
# Two-stage workflow

# Stage 1: Export ONNX (using any exporter)
modelexport export bert-base-uncased model.onnx  # HTP exporter
# OR use optimum-cli, or any other ONNX exporter

# Stage 2: Add metadata (standalone tool)
modelexport embed-metadata model.onnx bert-base-uncased
modelexport embed-metadata model.onnx bert-base-uncased --include-vocab

# Analyze embedded metadata
modelexport analyze model.onnx --show-metadata
modelexport validate model.onnx --check-metadata
```

### Python API

```python
# Two-Stage Process

# Stage 1: Export ONNX using any method
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("bert-base-uncased")
# Export using any method (HTP, Optimum, torch.onnx, etc.)
torch.onnx.export(model, dummy_input, "model.onnx")

# Stage 2: Add metadata as separate step
from modelexport.metadata import embed_metadata

embed_metadata(
    onnx_path="model.onnx",
    model_name="bert-base-uncased",
    include_vocab=False  # Set True to include vocabulary
)

# Inference - Zero Configuration
from modelexport.inference import pipeline

# Auto-detects everything from metadata!
pipe = pipeline(model="model.onnx")
result = pipe("This movie is great!")

# Advanced API
from modelexport.inference import ONNXAutoModel, ONNXAutoProcessor

model = ONNXAutoModel.from_model("model.onnx")
processor = ONNXAutoProcessor.from_model("model.onnx")
```

## Technology Stack

### Core Technologies

- **ONNX**: 1.14+ for metadata_props support
- **ONNX Runtime**: 1.14+ for inference
- **HuggingFace Transformers**: 4.0+ for model/processor configs
- **Python**: 3.8+ for implementation

### Storage Technologies

```mermaid
graph TD
    subgraph "Metadata Storage Strategy"
        MP[metadata_props field] --> JSON[JSON Serialization]
        JSON --> COMP{Size > 10KB?}
        COMP -->|Yes| GZ[GZIP Compression]
        COMP -->|No| STR[String Storage]
        GZ --> B64[Base64 Encoding]
        B64 --> STORE[Store in ONNX]
        STR --> STORE
    end
```

## Quality Attributes

### Performance Requirements

```mermaid
graph LR
    subgraph "Performance Targets"
        EX[Export<br/>+500ms max] --> PERF[Performance]
        PARSE[Parse<br/><100ms] --> PERF
        INF[Inference<br/>No degradation] --> PERF
        SIZE[Size<br/><200KB typical] --> PERF
    end
```

- **Export Overhead**: < 500ms for metadata extraction/embedding
- **Parse Overhead**: < 100ms for metadata reading
- **Inference Impact**: Zero degradation
- **Memory Usage**: < 10MB for metadata handling
- **File Size**: < 200KB metadata for 95% of models

### Compatibility Matrix

| Component | Version Support |
|-----------|----------------|
| ONNX Runtime | 1.14+ |
| Transformers | 4.0+ |
| Python | 3.8+ |
| ONNX Spec | 1.14+ (opset 14+) |

### Security & Privacy

- No sensitive data in metadata
- No user data collection
- Offline operation capability
- Reproducible builds

## Deployment Architecture

### Deployment Scenarios

```mermaid
graph TD
    subgraph "Deployment Targets"
        SC[Self-Contained ONNX] --> EDGE[Edge Devices]
        SC --> CLOUD[Cloud Services]
        SC --> AIR[Air-Gapped Systems]
        SC --> MOBILE[Mobile Apps]
        SC --> WEB[Web Browser]
    end
```

### Migration Strategy

```mermaid
graph LR
    subgraph "Phased Migration"
        P1[Phase 1<br/>Legacy] --> P2[Phase 2<br/>Metadata]
        P2 --> P3[Phase 3<br/>Full Adoption]
        
        P1 -.->|Backward Compatible| P2
        P2 -.->|Forward Compatible| P3
    end
```

## Risk Analysis

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Metadata bloat | High | Medium | Tiered embedding, compression |
| Compatibility break | High | Low | Extensive testing, versioning |
| Performance degradation | Medium | Low | Caching, lazy loading |
| Metadata corruption | Medium | Low | Checksums, validation |

### Business Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Adoption resistance | Medium | Low | Clear benefits, easy migration |
| Support burden | Low | Low | Self-documenting, good errors |
| Version fragmentation | Medium | Medium | Clear versioning strategy |

## Success Metrics

### Quantitative Metrics

```mermaid
graph TD
    subgraph "Success Metrics"
        S1[90% reduction<br/>in deployment steps] --> SUCCESS
        S2[80% reduction<br/>in config errors] --> SUCCESS
        S3[60% adoption<br/>in 6 months] --> SUCCESS
        S4[<200KB overhead<br/>for 95% models] --> SUCCESS
        S5[<100ms parse time<br/>for all models] --> SUCCESS
    end
```

### Qualitative Metrics

- **Developer Experience**: Simplified deployment workflow
- **User Satisfaction**: "It just works" experience
- **Operational Excellence**: Reduced deployment failures
- **Documentation**: Self-documenting models

## Implementation Roadmap

```mermaid
gantt
    title Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Phase 2.1
    Metadata Infrastructure     :2024-01-15, 14d
    section Phase 2.2
    HTP Integration            :14d
    section Phase 2.3
    Pipeline Enhancement       :14d
    section Phase 2.4
    Testing & Documentation    :14d
    section Phase 2.5
    Production Release         :7d
```

## Conclusion

Self-contained ONNX models represent a paradigm shift in ML model deployment, solving the "works with my preprocessing" problem analogous to how Docker solved "works on my machine". By embedding all necessary metadata directly in ONNX files, we enable:

1. **Single-file deployment** with zero external dependencies
2. **Guaranteed reproducibility** across all environments
3. **Offline operation** without internet access
4. **Self-documenting models** that explain their own requirements
5. **90% reduction** in deployment complexity

This design provides a clear path to implementation while maintaining backward compatibility and ensuring no performance degradation.
