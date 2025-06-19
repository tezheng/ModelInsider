# Test Case: Torch.nn Module Filtering

## Type
**MUST Test** ⚠️ **CRITICAL - CARDINAL RULE #5**

## Purpose
Verify that CARDINAL RULE #5 is maintained: torch.nn modules should NOT appear in tags - only model-specific modules appear in hierarchy tags.

## Test Data (Fixtures)
- BERT model with mixed torch.nn and transformers modules
- Simple model with torch.nn layers
- Debug output showing hierarchical paths

## Test Command
```bash
# Automated test validation
uv run python debug_tagging_simple.py

# Check tag distribution doesn't contain torch.nn classes
uv run python -c "
from modelexport import HierarchyExporter
import torch.nn as nn
from transformers import AutoModel

model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
# Check that no tags contain torch.nn class names like 'Linear', 'LayerNorm', 'Embedding'
"
```

## Expected Behavior
- Hierarchical paths exclude torch.nn module class names
- Tags show only model-specific modules (e.g., `BertEmbeddings`, `BertEncoder`)
- torch.nn modules like `Linear`, `LayerNorm`, `Embedding` do NOT appear in tags
- Paths stop at model-specific parent modules

## Failure Modes
- **torch.nn Classes in Tags**: Seeing `/Linear`, `/LayerNorm`, `/Embedding` in hierarchy
- **Incomplete Filtering**: Some torch.nn modules slip through filtering logic
- **Wrong Module Paths**: Paths like `/BertModel/BertEmbeddings/Embedding` instead of `/BertModel/BertEmbeddings`

## Dependencies
- transformers library
- torch.nn modules 
- BERT model for testing
- Debug scripts for validation

## Notes
- **🚨 MUST TEST**: This is CARDINAL RULE #5 - must be validated with EVERY code change
- **⚡ FREQUENTLY FORGOTTEN**: This rule was violated multiple times during development
- **👤 USER EMPHASIS**: User specifically emphasized this rule repeatedly
- Added to MEMO.md and hierarchy_exporter.py documentation
- Examples of CORRECT tags:
  - ✅ `/BertModel/BertEmbeddings`
  - ✅ `/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput`
- Examples of INCORRECT tags:
  - ❌ `/BertModel/BertEmbeddings/Embedding`
  - ❌ `/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput/Linear`
  - ❌ `/Linear`, `/LayerNorm`, `/Dropout`

## Implementation Check
Verify filtering logic in `_resolve_hierarchical_path()`:
```python
# Should filter out torch.nn modules
module_path = current_module.__class__.__module__
if not module_path.startswith('torch._C') and not module_path.startswith('torch.nn'):
    path_segments.append(current_module.__class__.__name__)
```

## Validation Checklist
- [ ] No `Linear` class names in hierarchy tags
- [ ] No `LayerNorm` class names in hierarchy tags  
- [ ] No `Embedding` class names in hierarchy tags
- [ ] No `Dropout` class names in hierarchy tags
- [ ] Paths stop at model-specific modules (e.g., `BertEmbeddings`)
- [ ] Filtering logic correctly identifies torch.nn modules