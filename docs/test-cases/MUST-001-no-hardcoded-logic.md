# Test Case: No Hardcoded Logic

## Type
**MUST Test** ⚠️ **CRITICAL - CARDINAL RULE #1**

## Purpose
Verify that CARDINAL RULE #1 is maintained: absolutely no hardcoded model architectures, node names, operator names, or any similar model-specific patterns exist in the codebase.

## Test Data (Fixtures)
- Source code analysis
- Multiple model architectures (BERT, ResNet, Custom models)
- Code pattern detection

## Test Command
```bash
# Manual code review (primary validation)
grep -r "bert\|transformer\|resnet" modelexport/hierarchy_exporter.py

# Automated test validation
uv run python -m pytest tests/test_param_mapping.py::TestParameterMapping::test_find_parent_transformers_module -v
```

## Expected Behavior
- No hardcoded model names in core logic
- Universal module detection only
- Same logic works across different model types
- No architecture-specific branches in code

## Failure Modes
- **Hardcoded Strings**: Model names like "bert", "transformer" in logic
- **Architecture Branches**: If/else blocks for specific model types  
- **Operator Assumptions**: Hardcoded ONNX operator name patterns
- **Parameter Patterns**: Hardcoded parameter naming assumptions

## Dependencies
- Source code access
- Multiple model architectures for validation
- Static analysis tools (grep, etc.)

## Notes
- **🚨 MUST TEST**: This is CARDINAL RULE #1 - must be validated with EVERY code change
- **🚫 ZERO TOLERANCE**: Any violation breaks the entire design philosophy
- Exceptions allowed only in:
  - Test fixture names (e.g., `bert_tiny_model` fixture)
  - Comments and documentation
  - Debug output strings
- Core logic must be 100% universal
- Violating this rule breaks the entire design philosophy

## Validation Checklist
- [ ] No model names in `_should_tag_module()`
- [ ] No architecture assumptions in `_resolve_hierarchical_path()`
- [ ] No hardcoded operator patterns in `_build_tag_mapping_from_onnx()`
- [ ] No parameter naming assumptions in `_extract_module_name_from_param()`
- [ ] Universal criteria only in all core methods