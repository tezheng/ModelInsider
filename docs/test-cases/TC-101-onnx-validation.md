# Test Case: ONNX Validation

## Type
**Sanity Test** 🧠

## Purpose
Verify that exported ONNX models pass standard ONNX validation and are compliant with ONNX specification. This ensures our hierarchy tagging doesn't break ONNX compatibility.

## Test Data (Fixtures)
- BERT model export
- Simple model export
- Generated ONNX files with hierarchy tags

## Test Command
```bash
# Run baseline comparison which includes ONNX validation
uv run python -m pytest tests/test_baseline_comparison.py::TestBaselineComparison::test_export_parameters_identical -v

# Manual validation
uv run python -c "
import onnx
model = onnx.load('path/to/exported/model.onnx')
onnx.checker.check_model(model)
print('ONNX validation passed')
"
```

## Expected Behavior
- `onnx.checker.check_model()` passes without errors
- No ONNX specification violations
- Doc_string field contains valid JSON
- Model can be loaded by ONNX runtime

## Failure Modes
- **Custom Attribute Rejection**: `Unrecognized attribute: hierarchy_tags`
- **Invalid ONNX Structure**: Malformed graph or nodes
- **JSON Parse Error**: Invalid JSON in doc_string field
- **Runtime Loading Error**: ONNX runtime can't load model

## Dependencies
- onnx library
- Exported ONNX models with hierarchy tags
- onnxruntime (optional, for runtime validation)

## Notes
- We switched from custom attributes to doc_string field for compliance
- This was a critical fix during development (Round 1)
- Doc_string field is ONNX-standard and accepted by all tools
- JSON format in doc_string must be valid

## Historical Context
Initial implementation used custom attributes:
```python
# OLD (rejected by ONNX validation)
attr = onnx.helper.make_attribute("hierarchy_tags", tags)
node.attribute.append(attr)
```

Current implementation uses doc_string:
```python
# NEW (ONNX compliant)
hierarchy_info = {"hierarchy_tags": tags, ...}
node.doc_string = json.dumps(hierarchy_info)
```

## Validation Checklist
- [ ] No ONNX checker errors
- [ ] Doc_string contains valid JSON
- [ ] No custom attributes that violate ONNX spec
- [ ] Model loads in external ONNX tools
- [ ] Hierarchy information preserved and accessible