# Iteration 10 - Torch.nn Module Regression Fix

## Summary
Fixed the regression where torch.nn modules were incorrectly appearing in the hierarchy when `include_torch_nn_children=False` (the default setting).

## Issue
After the hierarchy utilities refactoring, torch.nn modules (LayerNorm, Dropout, Embedding) started appearing in the module hierarchy even when they shouldn't. This violated MUST-002 rule which states that torch.nn classes should not appear in hierarchy tags unless explicitly excepted.

## Root Cause Analysis
The `should_create_hierarchy_level` method in `TracingHierarchyBuilder` was changed to always return `True` to ensure complete hierarchy visibility in reports. However, this ignored the exceptions parameter and included all modules unconditionally.

## Solution
Updated `should_create_hierarchy_level` to use the universal `should_include_in_hierarchy` function:

```python
def should_create_hierarchy_level(self, module: nn.Module) -> bool:
    """
    Determine if module should create a new hierarchy level - UNIVERSAL.
    
    Respects the exceptions parameter to control which torch.nn modules are included.
    When exceptions=None (default), torch.nn modules are excluded per MUST-002.
    """
    # Use the universal should_include_in_hierarchy function
    # This properly filters torch.nn modules based on the exceptions list
    return should_include_in_hierarchy(module, exceptions=self.exceptions)
```

## Testing Results

### Default Behavior (include_torch_nn_children=False)
```
🌳 Module Hierarchy:
BertModel
├── BertEmbeddings: embeddings
├── BertEncoder: encoder
│   ├── BertLayer: 0
│   │   ├── BertAttention: attention
│   │   │   ├── BertSelfOutput: output
│   │   │   └── BertSdpaSelfAttention: self
│   │   ├── BertIntermediate: intermediate
│   │   │   └── GELUActivation: intermediate_act_fn
│   │   └── BertOutput: output
│   └── BertLayer: 1
        └── ... (similar structure)
└── BertPooler: pooler
```
- Shows only 18 HuggingFace modules
- No torch.nn modules appear

### With include_torch_nn_children=True
```
🌳 Module Hierarchy:
BertModel
├── BertEmbeddings: embeddings
│   ├── LayerNorm: LayerNorm
│   ├── Dropout: dropout
│   ├── Embedding: position_embeddings
│   ├── Embedding: token_type_embeddings
│   └── Embedding: word_embeddings
└── ... (includes Linear, LayerNorm, Dropout throughout)
```
- Shows 45 modules including torch.nn modules
- Properly includes torch.nn modules when requested

## Benefits
1. **MUST-002 Compliance**: Correctly excludes torch.nn modules by default
2. **Flexibility**: Still allows torch.nn inclusion when explicitly requested
3. **Clean Hierarchies**: Reports show only relevant HuggingFace modules
4. **Backward Compatibility**: Maintains expected behavior for existing users

## Linear Task TEZ-24 Status
✅ **COMPLETED** - All requirements met:
- ✅ Fixed malformed hierarchies (compound patterns)
- ✅ Created shared hierarchy utilities
- ✅ Fixed report formatting to match console
- ✅ Fixed torch.nn module regression
- ✅ All tests passing
- ✅ Documentation updated