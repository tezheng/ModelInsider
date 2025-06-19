# Test Cases Summary

## Overview
This document provides a comprehensive overview of all test cases for the modelexport project, organized by type and priority.

## Test Case Categories

### ⚠️ MUST Tests (CARDINAL RULES - ZERO TOLERANCE)
Critical design constraints that must be validated with EVERY code change. Any failure is a system-breaking violation.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| MUST-001 | No Hardcoded Logic | CARDINAL RULE #1: Zero hardcoded patterns | Manual | Code review + `grep -r "bert\\|transformer" hierarchy_exporter.py` |
| MUST-002 | Torch.nn Filtering | CARDINAL RULE #5: No torch.nn in tags | 10-30s | `python debug/debug_tagging_simple.py` |
| MUST-003 | Universal Design | Core principle: works with ANY model | 30-60s | Multiple model type tests |

### 🚭 Smoke Tests (Basic Functionality)
Critical path tests that verify basic functionality works. If these fail, nothing else will work.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| TC-001 | Basic Import | Module can be imported | <1s | `python -c "from modelexport import HierarchyExporter"` |
| TC-002 | Simple Export | Basic export functionality | <10s | `pytest test_param_mapping.py::...simple_model` |
| TC-003 | CLI Availability | CLI commands accessible | <5s | `modelexport --help` |

### 🧠 Sanity Tests (Core Assumptions)
Tests that verify core assumptions and invariants hold true.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| TC-101 | ONNX Validation | Exported models are ONNX compliant | 30-60s | `pytest test_baseline_comparison.py::...identical` |
| TC-102 | Hierarchy Structure | Tags reflect actual model structure | 5-10s | `python debug_path_building.py` |
| TC-103 | Tag Format | Hierarchical tags follow expected format | <5s | Manual validation |

### 🔄 Regression Tests (Design Rules)
Tests that ensure other design principles and backward compatibility are maintained.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| TC-204 | API Stability | Backward compatibility maintained | 10-30s | Test method signatures |

### 🔗 Integration Tests (End-to-End)
Tests that verify complete workflows function correctly.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| TC-301 | BERT Export Workflow | Complete CLI workflow with BERT | 60-120s | `modelexport export prajjwal1/bert-tiny` |
| TC-302 | CLI Complete Workflow | Export → Analyze → Validate | 90-150s | Multi-command sequence |
| TC-303 | Multi-Model Support | Works with diverse architectures | 120-180s | Multiple model exports |

### 🚀 Performance Tests (Time/Memory Bounds)
Tests that verify performance characteristics and resource usage.

| ID | Name | Purpose | Runtime | Command |
|----|------|---------|---------|---------|
| TC-401 | Export Time Bounds | Export completes within time limits | 60-600s | `timeout 600 pytest ...` |
| TC-402 | Memory Usage | Memory usage stays reasonable | Variable | Memory profiling |

## Test Execution Strategy

### By Priority (CI/CD Pipeline)
```bash
# Stage 0: MUST tests (CRITICAL - must pass or FAIL the entire pipeline)
pytest -m must --tb=short

# Stage 1: Smoke tests (must pass for pipeline to continue)
pytest -m smoke --tb=short

# Stage 2: Sanity tests (core functionality)  
pytest -m sanity --tb=short

# Stage 3: Regression tests (design compliance)
pytest -m regression --tb=short  

# Stage 4: Integration tests (full workflows)
pytest -m integration --tb=short

# Stage 5: Performance tests (optional, on schedule)
pytest -m performance --tb=short
```

### By Runtime (Developer Workflow)
```bash
# MUST validation (EVERY code change - <60s total)
pytest -m must --tb=short

# Quick validation (<30s total)
pytest -m "must or smoke or (sanity and not slow)"

# Medium validation (<5min total)  
pytest -m "smoke or sanity or (regression and not manual)"

# Full validation (10-30min total)
pytest -m "not performance"

# Complete validation (30-60min total)
pytest
```

## Special Test Categories

### Manual Tests
Some tests require human validation and cannot be fully automated:

- **TC-201 (No Hardcoded Logic)**: Code review required
- **Design Compliance**: Architectural review
- **User Experience**: CLI usability assessment

### Rule-Based Tests  
CARDINAL RULES are treated as MUST tests with zero tolerance for failure:

| Rule | Test Type | Test ID | Validation Method |
|------|-----------|---------|-------------------|
| No Hardcoded Logic | **MUST** | MUST-001 | Code analysis + multiple model tests |
| No torch.nn in Tags | **MUST** | MUST-002 | Tag content analysis |
| Universal Design | **MUST** | MUST-003 | Architecture diversity tests |
| Usage-Based Tagging | Sanity | TC-101+ | Hook execution validation |
| ONNX Compliance | Sanity | TC-101 | ONNX validation tools |

### Test Data Requirements

#### Models Used
- **BERT-tiny**: Primary integration test model (4.4M params)
- **SimpleModel**: Basic unit test model (<1M params)  
- **Custom Models**: Architecture diversity validation

#### Fixtures Required
- Temporary directories
- Model caches
- Input tensors
- Expected outputs

## Maintenance Guidelines

### Adding New Test Cases
1. Create markdown documentation in `/docs/test-cases/`
2. Add to this summary table
3. Implement pytest test with appropriate markers
4. Update CI/CD pipeline if needed

### Test Case Lifecycle
- **Creation**: Document purpose and validation criteria
- **Implementation**: Write pytest code with proper markers
- **Maintenance**: Update when requirements change
- **Retirement**: Remove obsolete tests, update documentation

### Marker Usage
```python
# Pytest markers for categorization
@pytest.mark.must        # CARDINAL RULES - ZERO TOLERANCE
@pytest.mark.smoke       # Basic functionality
@pytest.mark.sanity      # Core assumptions  
@pytest.mark.regression  # Design rule compliance
@pytest.mark.integration # End-to-end workflows
@pytest.mark.performance # Time/memory bounds
@pytest.mark.slow        # Takes >60 seconds
@pytest.mark.manual      # Requires human validation
```

## Success Metrics

### Test Coverage Goals
- **MUST**: 100% pass rate (ABSOLUTE REQUIREMENT - CI/PR blocker)
- **Smoke**: 100% pass rate (CI blocker)
- **Sanity**: >95% pass rate
- **Regression**: >95% pass rate
- **Integration**: >90% pass rate  
- **Performance**: Within established bounds

### Quality Gates
- **🚨 CRITICAL**: All MUST tests must pass for ANY code change
- All smoke tests must pass for release
- No MUST test failures allowed under ANY circumstances
- Performance tests establish baseline bounds
- Integration tests validate user workflows

### MUST Test Enforcement
- **Pre-commit hooks**: Run MUST tests automatically
- **CI/CD Pipeline**: Stage 0 blocker - fail fast on MUST violations
- **Code Review**: Manual verification of CARDINAL RULES
- **Release Gates**: Zero tolerance for MUST test failures