# Tagging Strategy Documentation

## Overview

The Universal Hierarchy-Preserving ONNX Exporter implements a **phased approach** to operation tagging, designed to maintain compatibility across different model architectures while preserving meaningful hierarchical information.

## Core Philosophy

The tagging system follows these principles:

1. **Universal Design**: No hardcoded model-specific logic
2. **Phase-Based Focus**: Different phases target different module types
3. **Conservative Tagging**: Only tag operations when meaningful hierarchy can be preserved
4. **Strategy Flexibility**: Multiple strategies for different use cases

## Current Phase: HuggingFace Module Focus

### Phase 1 Implementation (Current)

**Target**: HuggingFace model architectures and components
**Scope**: Operations within HuggingFace-specific modules are prioritized for tagging

**Why HF Modules Get Tagged**:
- HuggingFace modules represent meaningful semantic components (attention, embeddings, encoders, etc.)
- These modules provide valuable hierarchy information for model analysis and optimization
- Users working with HF models benefit most from this level of granularity

**Why Simple PyTorch Models May Show 0 Tags**:
- Simple `torch.nn.Linear`, `torch.nn.ReLU` models don't contain HF-specific hierarchy
- This is **expected behavior** in Phase 1 - the system is working correctly
- Basic PyTorch modules are filtered out to avoid generic, non-informative tags

### torch.nn Exception Whitelist

Certain `torch.nn` modules are **whitelisted** for tagging regardless of phase:

```python
TORCH_NN_EXCEPTIONS = {
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'GroupNorm', 
    'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'Embedding'
}
```

**Rationale**: These modules represent important architectural decisions and data flow patterns that provide meaningful semantic information even outside HF contexts.

## Tagging Strategies

### 1. Usage-Based Strategy (`usage_based`)

**Approach**: Conservative tagging of operations that can be reliably traced to meaningful modules

**Characteristics**:
- Filters out most `torch.nn` modules (except whitelist)
- Only tags operations with clear module lineage
- Minimal false positives
- May produce fewer tags but higher confidence

**Best For**: 
- Production environments where precision is critical
- Analysis requiring high-confidence hierarchy information
- Models where false positives would be problematic

### 2. Hierarchical Trace-and-Project Strategy (`htp`)

**Approach**: Advanced execution tracing with operation projection onto ONNX graph

**Characteristics**:
- Uses forward hooks to capture execution context
- Projects traced operations onto exported ONNX graph
- More comprehensive operation coverage
- Handles complex control flows and dynamic behaviors

**Best For**:
- Research and analysis requiring comprehensive coverage
- Complex models with dynamic behavior
- Detailed model profiling and optimization

## Expected Behavior Examples

### Simple PyTorch Model (Phase 1)
```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        
# Expected: 0 tagged operations (by design)
# Reason: No HF modules, basic torch.nn filtered out
```

### HuggingFace Model (Phase 1)
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# Expected: Many tagged operations
# Reason: Rich HF module hierarchy (BertLayer, BertAttention, etc.)
```

### Model with Whitelisted Components
```python
class ModelWithNorms(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)      # Filtered out
        self.norm = nn.LayerNorm(64)          # TAGGED (whitelisted)
        self.embedding = nn.Embedding(1000, 128)  # TAGGED (whitelisted)
```

## Future Phases

### Phase 2: torch.nn Module Expansion (Planned)

**Target**: Comprehensive `torch.nn` module coverage
**Changes**: 
- Reduce `torch.nn` filtering
- Expand whitelist significantly
- Maintain HF module priority while adding PyTorch coverage

### Phase 3: Custom Module Support (Planned)

**Target**: User-defined and third-party modules
**Features**:
- Configurable module filtering
- Custom hierarchy patterns
- Domain-specific tagging rules

## Diagnostic and Debugging

### Understanding Tag Counts

**0 Tagged Operations**:
- Expected for simple PyTorch models in Phase 1
- Model contains only filtered `torch.nn` modules
- No meaningful hierarchy detected for current phase

**Low Tag Counts**:
- Model may have few HF-specific components
- Most operations filtered due to generic nature
- Consider if model fits Phase 1 target scope

**High Tag Counts**:
- Rich model hierarchy detected
- Many operations traced to meaningful modules
- Good coverage for analysis and optimization

### Debugging Tips

1. **Check Module Types**: Use `model.named_modules()` to see what modules exist
2. **Strategy Comparison**: Try both strategies to understand coverage differences
3. **Whitelist Review**: Check if important modules are in the exception list
4. **Phase Alignment**: Ensure model type matches current phase expectations

## Implementation Details

### Tag Format
Tags follow hierarchical path format: `/ModelName/ModulePath/SubModule`

Example: `/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput`

### Operation Mapping
The system maps PyTorch operations to ONNX operations through a centralized registry, ensuring consistent handling across both tagging strategies.

### Memory and Performance
- State cleanup after each export
- Efficient hook management
- Minimal overhead on model execution
- Concurrent export support (with serialization for PyTorch limitations)

## Configuration

### Strategy Selection
```python
# Conservative, high-precision tagging
exporter = HierarchyExporter(strategy="usage_based")

# Comprehensive, research-oriented tagging  
exporter = HierarchyExporter(strategy="htp")
```

### Custom Exception Lists
```python
# Override default torch.nn filtering
custom_exceptions = ["Linear", "Conv2d", "ReLU"]
exporter = HierarchyExporter(
    strategy="htp", 
    torch_nn_exceptions=custom_exceptions
)
```

## Migration Guide

### From Other Tools
- Adjust expectations for Phase 1 HF focus
- Review tag output for semantic meaning vs. operation coverage
- Consider strategy choice based on use case requirements

### Version Updates
- Tag formats remain stable across versions
- Strategy behavior documented with each release
- Backward compatibility maintained for core functionality

## FAQ

**Q: Why does my PyTorch model show 0 tagged operations?**
A: This is expected in Phase 1. The system focuses on HuggingFace modules. Basic PyTorch models will show tags in Phase 2.

**Q: Should I use usage_based or htp strategy?**  
A: Use `usage_based` for production/precision needs, `htp` for research/comprehensive analysis.

**Q: Can I force tagging of torch.nn modules?**
A: Yes, modify the `torch_nn_exceptions` parameter to include desired modules.

**Q: How do I know if my model fits Phase 1?**
A: Check if your model uses HuggingFace transformers. If it's pure PyTorch, consider waiting for Phase 2 or using custom exceptions.