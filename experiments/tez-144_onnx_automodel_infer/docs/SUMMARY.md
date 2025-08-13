# TEZ-144: ONNX Configuration Strategy - Summary

## Executive Summary

After comprehensive analysis of HuggingFace Optimum's requirements, we've revised our approach from a "smart hybrid" strategy to a simpler "always copy configuration" approach. This ensures immediate compatibility with Optimum without requiring custom code.

## Key Findings

1. **Optimum Requires config.json Locally**: Optimum's `ORTModel.from_pretrained()` expects `config.json` in the same directory as the ONNX model. It cannot load configuration from ONNX metadata or dynamically fetch from HuggingFace Hub.

2. **Smart Hybrid Not Feasible**: The originally proposed approach of storing metadata for Hub models and loading config dynamically doesn't work with current Optimum implementation.

3. **Negligible Storage Impact**: Configuration files add only 2-5KB (~0.01% of model size), making optimization unnecessary.

## Revised Decision

**Always copy configuration files during export for ALL models.**

This ensures:
- ✅ Immediate Optimum compatibility
- ✅ No custom loader code needed
- ✅ Consistent deployment pattern
- ✅ Full offline capability
- ✅ Simple implementation

## Implementation Status

### Completed
- [x] Comprehensive review of Optimum codebase
- [x] Analysis of ADR-013 technical feasibility
- [x] Creation of revised ADR-013
- [x] Updated implementation guide

### Next Steps
1. Replace original ADR-013 with revised version
2. Update HTP exporter to include config copying
3. Update CLI to use new export function
4. Add validation tests for Optimum compatibility
5. Update documentation and examples

## File Structure After Export

```
model_directory/
├── model.onnx              # ONNX model with HTP metadata
├── config.json             # Model configuration (REQUIRED)
├── tokenizer.json          # Tokenizer (if applicable)
├── tokenizer_config.json   # Tokenizer config (if applicable)
└── export_metadata.json    # Export tracking information
```

## Usage Example

```python
# Export with modelexport
from modelexport import export_with_config

export_with_config("bert-base-uncased", "bert_exported/")

# Use with Optimum - just works!
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("bert_exported/")
```

## Documentation Structure

1. **ADR-013-revised**: Architectural decision with rationale
2. **high_level_design.md**: System architecture and flow
3. **implementation_guide_revised.md**: Detailed implementation
4. **comprehensive_review.md**: Analysis that led to revision

## Key Takeaway

By choosing pragmatism over premature optimization, we ensure that ModelExport's ONNX exports work seamlessly with Optimum today, while keeping the door open for future enhancements if truly needed.

## References

- [Linear Task TEZ-144](https://linear.app/...)
- [Optimum Documentation](https://huggingface.co/docs/optimum)
- [ADR-013 Comprehensive Review](./ADR-013-comprehensive-review.md)
- [Revised Implementation Guide](./implementation_guide_revised.md)