# Test Case: Hierarchy Structure Validation

## Type
**Sanity Test** 🧠

## Purpose
Verify that generated hierarchical tags accurately reflect the actual PyTorch model structure and follow expected hierarchical patterns.

## Test Data (Fixtures)
- BERT model with known hierarchical structure
- Simple model with predictable hierarchy
- Debug output showing module traversal

## Test Command
```bash
# Debug hierarchical path building
uv run python debug_path_building.py

# Validate hierarchy in tests  
uv run python -m pytest tests/test_cli_integration.py::TestHierarchicalTagging::test_module_hierarchy_inspection -v
```

## Expected Behavior
- Hierarchical paths follow actual nn.Module structure
- Parent-child relationships preserved
- Path segments correspond to actual module class names
- Depth reflects true nesting level
- Root model appears at path beginning

## Failure Modes
- **Incorrect Nesting**: Paths don't match actual module hierarchy
- **Missing Levels**: Some hierarchy levels skipped incorrectly
- **Wrong Class Names**: Module class names incorrect in paths
- **Inconsistent Structure**: Same module produces different paths

## Dependencies
- Model with known, verifiable structure
- Debug scripts for hierarchy inspection
- Module traversal validation

## Notes
### Expected BERT Hierarchy Structure:
```
BertModel
├── BertEmbeddings
├── BertEncoder  
│   └── BertLayer (repeated)
│       ├── BertAttention
│       │   ├── BertSdpaSelfAttention
│       │   └── BertSelfOutput  
│       ├── BertIntermediate
│       └── BertOutput
└── BertPooler
```

### Expected Hierarchical Tags:
- `/BertModel/BertEmbeddings`
- `/BertModel/BertEncoder/BertLayer/BertAttention/BertSdpaSelfAttention`
- `/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput`
- `/BertModel/BertEncoder/BertLayer/BertIntermediate`  
- `/BertModel/BertEncoder/BertLayer/BertOutput`
- `/BertModel/BertPooler`

### Path Building Rules:
1. Start with root model class name
2. Traverse `module_name.split('.')`
3. Add class name for each level (if not torch.nn)
4. Build path as `"/" + "/".join(segments)`

## Validation Checklist
- [ ] Root model class appears first in all paths
- [ ] Nesting depth matches actual module structure  
- [ ] Module class names accurate in hierarchy
- [ ] torch.nn modules properly filtered out
- [ ] Consistent path generation for same modules
- [ ] No orphaned or disconnected path segments