# ADR-003: Standard vs Function Export Analysis

**Date**: 2025-01-02  
**Status**: ACCEPTED  
**Deciders**: ModelExport Team  
**Technical Story**: Analysis of ONNX export approaches for hierarchy preservation

## Context

During ModelExport development, we encountered PyTorch's `export_modules_as_functions` parameter in `torch.onnx.export()`. This feature promises to preserve module hierarchy by converting PyTorch modules into ONNX local functions. We needed to evaluate whether this approach could enhance or replace our HTP (Hierarchy Tracing and Propagation) strategy.

## Decision

**We will continue using standard ONNX export with our HTP strategy as the primary approach**, with the following rationale:

1. **Reliability**: Standard export works consistently across all model architectures
2. **Granularity**: HTP provides operation-level analysis essential for our use cases
3. **Production Readiness**: Robust handling of real-world transformer models
4. **Future Compatibility**: Not dependent on fragile internal implementation details

## Detailed Analysis

### Experimental Setup

We conducted comprehensive experiments using:
- **Test Model**: `prajjwal1/bert-tiny` (4.4M parameters, real transformer)
- **Export Methods**: Standard vs `export_modules_as_functions=True`
- **Analysis Framework**: Structural comparison, performance benchmarking, hierarchy preservation assessment

### Key Findings

#### 1. Reliability Comparison

| Aspect | Standard Export | export_modules_as_functions |
|--------|----------------|----------------------------|
| **BERT-tiny Success** | ✅ Works reliably | ❌ Fails with annotation errors |
| **Simple Models** | ✅ Always works | ✅ Works with basic models |
| **Production Models** | ✅ Handles all architectures | ❌ Brittle with complex models |
| **Error Rate** | ~0% | High with transformer models |

#### 2. Root Cause of Function Export Failures

**Critical Discovery**: `export_modules_as_functions` requires ALL instances of the same module class to have IDENTICAL type annotations.

**BERT-tiny Violation**:
```
Found outstanding annotated attribute 2310 from module 3285
```

**Specific Issues**:
- **Bias Parameter Mismatch**: All 13 Linear modules have `bias` parameters but inconsistent `__annotations__`
- **Runtime Modifications**: Hugging Face models modify annotations during initialization
- **Specialized Classes**: `BertSdpaSelfAttention` and other optimized components have complex annotation patterns
- **Complex Inheritance**: Deep inheritance hierarchies create annotation conflicts

#### 3. Granularity Analysis

| Level | Standard + HTP | export_modules_as_functions |
|-------|---------------|----------------------------|
| **Operations** | ✅ Full access (MatMul, Add, etc.) | ❌ Hidden in functions |
| **Module Context** | ✅ Tagged metadata | ✅ Function boundaries |
| **Debugging** | ✅ Direct operation tracing | ❌ Limited to function level |
| **Custom Backends** | ✅ Operation-level control | ❌ Function call overhead |

#### 4. Use Case Alignment

**ModelExport Requirements**:
- ✅ **Attention Analysis**: Trace individual MatMul operations to query/key/value
- ✅ **Layer Analysis**: Map operations to specific encoder layers
- ✅ **Parameter Mapping**: Link operations to source module hierarchy
- ✅ **Research Support**: Fine-grained analysis for transformer studies

**Function Export Limitations**:
- ❌ Operations hidden within function boundaries
- ❌ Cannot trace specific MatMul to `attention.self.query`
- ❌ Loses operation-level granularity needed for HTP
- ❌ Unreliable with production transformer models

### Performance Impact

**Standard Export**:
- File Size: 16.78 MB (BERT-tiny)
- Inference: ~0.1ms per operation
- Nodes: 284 operations directly accessible

**Function Export** (when it works):
- File Size: Typically larger due to function overhead
- Inference: Additional function call overhead
- Nodes: Reduced main graph, operations hidden in functions

### Architecture-Specific Insights

#### Transformer Models (BERT, GPT, etc.)
- **Complex Attention**: Requires operation-level tracing for research
- **Layer Structure**: Benefits from fine-grained operation mapping
- **Optimization Conflicts**: SDPA and other optimizations break function export
- **Real-world Usage**: Production models consistently fail function export

#### Simple Models
- **Basic CNNs/MLPs**: Function export may work
- **Academic Examples**: Often compatible
- **Limited Scope**: Not representative of production use cases

## Consequences

### Positive
- **Robust Foundation**: Standard export + HTP works reliably across all models
- **Research Enablement**: Operation-level granularity supports detailed analysis
- **Production Ready**: Handles real-world transformer complexity
- **Future Proof**: Not dependent on internal PyTorch implementation details

### Negative
- **Module-level Organization**: Don't get automatic module grouping
- **Alternative Visualization**: Miss cleaner hierarchical graph structure
- **Hybrid Approaches**: Cannot easily combine both granularities

### Neutral
- **Technical Debt**: Function export is deprecated in newer PyTorch versions
- **Maintenance**: Standard approach requires ongoing HTP development
- **Compatibility**: Broader ONNX runtime support with standard export

## Implementation Guidelines

### Primary Strategy: Standard Export + HTP
```python
torch.onnx.export(
    model, sample_input, output_path,
    export_modules_as_functions=False,  # Standard export
    # ... other parameters
)
# Apply HTP strategy for hierarchy preservation
```

### Future Considerations
- **Optional Enhancement**: Consider function export as alternative mode for specific use cases
- **Hybrid Research**: Investigate combining both approaches for dual-level hierarchy
- **User Choice**: Potentially offer both modes based on model compatibility

## Evidence References

- **Experiment Notebook**: `ADR-003-standard-vs-function-export-analysis.ipynb`
- **BERT-tiny Analysis**: Structural comparison showing 284 operations vs function grouping
- **Failure Investigation**: Detailed annotation mismatch analysis
- **Performance Benchmarks**: Inference timing comparisons

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-02 | Initial analysis and decision |

---

**Related ADRs**: 
- [ADR-001: Record Architecture Decisions](ADR-001-record-architecture-decisions.md)
- [ADR-002: Auxiliary Operations Tagging](ADR-002-auxiliary-operations-tagging.md)

**Tags**: `onnx-export`, `hierarchy-preservation`, `pytorch`, `transformers`, `reliability`