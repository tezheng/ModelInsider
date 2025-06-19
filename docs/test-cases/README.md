# Test Cases Documentation

This directory contains detailed documentation for all test cases in the modelexport project.

## Test Case Types

### MUST Tests ⚠️
- **Purpose**: Validate CARDINAL RULES and core design principles
- **Scope**: Critical design constraints that must NEVER be violated
- **Frequency**: EVERY code change, EVERY commit, EVERY PR
- **Examples**: No hardcoded logic, torch.nn filtering, universal design
- **🚨 CRITICAL**: Zero tolerance for failures - any violation breaks the system

### Smoke Tests 🚭
- **Purpose**: Verify basic functionality works
- **Scope**: Critical path, minimal depth
- **Frequency**: Every build
- **Examples**: Can import modules, basic export works

### Sanity Tests 🧠  
- **Purpose**: Verify core assumptions and invariants hold
- **Scope**: Key behaviors, reasonable depth
- **Frequency**: Every commit
- **Examples**: ONNX validation passes, hierarchy structure correct

### Regression Tests 🔄
- **Purpose**: Ensure RULES and principles are maintained
- **Scope**: Design constraints, behavioral contracts
- **Frequency**: Every release
- **Examples**: No hardcoded logic, torch.nn modules filtered

### Integration Tests 🔗
- **Purpose**: End-to-end workflows function correctly  
- **Scope**: Full system behavior, real data
- **Frequency**: Daily/weekly
- **Examples**: CLI workflows, multi-model exports

### Performance Tests 🚀
- **Purpose**: Verify performance characteristics
- **Scope**: Time/memory bounds, scalability
- **Frequency**: Release cycles
- **Examples**: Export time limits, memory usage bounds

## Test Case Structure

Each test case follows this structure:

```markdown
# Test Case: [Name]

## Type
[Smoke/Sanity/Regression/Integration/Performance]

## Purpose
[Clear description of what this test validates]

## Test Data (Fixtures)
[Description of required test data and fixtures]

## Test Command
[Exact command to run the test]

## Expected Behavior
[What should happen when test passes]

## Failure Modes
[Common ways this test can fail and what they indicate]

## Dependencies
[Required fixtures, models, or setup]

## Notes
[Additional context, gotchas, or maintenance notes]
```

## Index of Test Cases

### MUST Tests ⚠️
- [MUST-001: No Hardcoded Logic](MUST-001-no-hardcoded-logic.md)
- [MUST-002: Torch.nn Filtering](MUST-002-torch-nn-filtering.md)
- [MUST-003: Universal Design](MUST-003-universal-design.md)

### Smoke Tests
- [TC-001: Basic Import](TC-001-basic-import.md)
- [TC-002: Simple Export](TC-002-simple-export.md)
- [TC-003: CLI Availability](TC-003-cli-availability.md)

### Sanity Tests  
- [TC-101: ONNX Validation](TC-101-onnx-validation.md)
- [TC-102: Hierarchy Structure](TC-102-hierarchy-structure.md)
- [TC-103: Tag Format](TC-103-tag-format.md)

### Regression Tests
- [TC-204: API Stability](TC-204-api-stability.md)

### Integration Tests
- [TC-301: BERT Export Workflow](TC-301-bert-export-workflow.md)
- [TC-302: CLI Complete Workflow](TC-302-cli-complete-workflow.md)
- [TC-303: Multi-Model Support](TC-303-multi-model-support.md)

### Performance Tests
- [TC-401: Export Time Bounds](TC-401-export-time-bounds.md)
- [TC-402: Memory Usage](TC-402-memory-usage.md)

## Running Test Categories

```bash
# Run all MUST tests (CRITICAL - run on every code change)
uv run python -m pytest -m must -v

# Run all smoke tests
uv run python -m pytest -m smoke -v

# Run all sanity tests  
uv run python -m pytest -m sanity -v

# Run all regression tests
uv run python -m pytest -m regression -v

# Run all integration tests
uv run python -m pytest -m integration -v

# Run all performance tests
uv run python -m pytest -m performance -v
```

## Test Maintenance

### Adding New Test Cases
1. Create new markdown file following naming convention
2. Update this README index
3. Add appropriate pytest markers
4. Document fixtures and dependencies

### Updating Existing Cases
1. Update markdown documentation
2. Ensure test code matches documentation
3. Update dependencies if changed
4. Note any breaking changes