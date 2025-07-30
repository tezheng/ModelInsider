# TEZ-127: GraphML Structural Validation Tests - Technical Plan

## 1. Problem Analysis

### Current State
- GraphML generation has critical bugs including node duplication (131 duplicate nodes)
- No comprehensive test coverage for GraphML structural integrity
- Violations of MUST RULE #1 (hardcoded logic) in multiple files

### Root Causes
1. **Node Duplication**: Same nodes appear in both correct subgraphs AND root level
2. **Missing Metadata**: GraphML lacks input/output metadata (keys g0, g1)
3. **Hardcoded Logic**: Test files contain model-specific hardcoded ONNX operation lists

## 2. Technical Approach

### Test Strategy
- **Model-Agnostic Design**: Use universal patterns (nested graph detection) instead of hardcoded lists
- **Comprehensive Coverage**: 8 structural validation tests + 1 E2E test
- **Evidence-Based**: Tests designed to expose specific bugs with clear failure messages

### Implementation Plan
1. Create structural validation test suite
2. Fix node duplication in GraphML converters
3. Implement input/output metadata generation
4. Remove all hardcoded ONNX type lists
5. Verify with comprehensive test execution

## 3. Architecture Decisions

### Universal Node Classification
```python
# Instead of hardcoded lists:
if op_type in ["Add", "MatMul", "Gather", ...]:  # ❌ WRONG

# Use structural detection:
has_nested_graph = node.find('.//graph') is not None  # ✅ CORRECT
if not has_nested_graph:  # It's an ONNX operation
```

### Node Placement Tracking
```python
self.placed_nodes = set()  # Track placed nodes to prevent duplication
```

## 4. Testing Philosophy
- Fail-first approach to expose bugs
- Model-agnostic validation
- Clear error messages for debugging

## 5. Success Criteria
- All 8 structural tests PASS
- E2E test validates complete pipeline
- Zero hardcoded model-specific logic
- No node duplication in GraphML output