# Refactoring Guidelines & Best Practices

## Overview

This document provides systematic guidelines for refactoring Python code, based on real-world experience with the HTP Export Monitor refactoring. These practices ensure maintainable, universal, and high-quality code.

## 🚨 MANDATORY RULES (Zero Tolerance)

### Rule 1: No Hardcoded Logic
- **NEVER** hardcode model names, architectures, node names, or operation names
- **ALWAYS** use universal approaches (PyTorch `nn.Module` hierarchy, forward hooks)
- **BEFORE** every commit: Ask "Is this hardcoded to any specific architecture?"
- **VALIDATION**: Search codebase for model-specific strings before any release

### Rule 2: Ruff Lint Compliance
- **ALWAYS** run `ruff check` after code changes
- **FIX** all linting issues before committing
- **ZERO** tolerance for new linting violations
- **ENFORCE** via pre-commit hooks where possible

### Rule 3: No Regression Testing
- **ALWAYS** run full test suite after refactoring: `uv run pytest tests/`
- **VERIFY** all tests pass before marking tasks complete
- **UPDATE** test assertions to match refactored structure
- **MAINTAIN** backward compatibility unless explicitly breaking

## 📋 Pre-Refactoring Checklist

### Assessment Phase
- [ ] Identify all hardcoded values (magic numbers, strings, constants)
- [ ] Locate duplicated code patterns and logic
- [ ] Review large methods that need decomposition
- [ ] Check for missing error handling
- [ ] Assess test coverage and compatibility
- [ ] Run `ruff check` to identify existing issues

### Planning Phase
- [ ] Create Config class structure for constants
- [ ] Plan utility method signatures for common operations
- [ ] Design method decomposition strategy
- [ ] Plan error handling approach
- [ ] Identify type safety improvements needed

## 🔧 Refactoring Implementation Guide

### 1. Configuration Management
**Objective**: Centralize all magic values and constants

**Steps**:
- [ ] Create `Config` class with logical groupings:
  ```python
  class Config:
      # Display settings
      TOP_NODES_COUNT = 20
      SEPARATOR_LENGTH = 80
      
      # Console settings
      CONSOLE_WIDTH = 120
      
      # File suffixes
      METADATA_SUFFIX = "_htp_metadata.json"
      
      # Numeric constants
      MILLION = 1e6
      PERCENT = 100.0
  ```
- [ ] Replace all hardcoded values with `Config.CONSTANT_NAME`
- [ ] Group related constants together
- [ ] Use descriptive names that explain purpose
- [ ] Document purpose of each constant group

### 2. Code Deduplication
**Objective**: Extract repeated patterns into reusable utilities

**Steps**:
- [ ] Identify 3+ line code patterns that repeat
- [ ] Create utility methods with single responsibility:
  ```python
  def _create_report_console(self, file=None) -> Console:
      """Create console with consistent settings."""
      return Console(
          file=file,
          width=Config.CONSOLE_WIDTH,
          force_terminal=False,
          highlight=False
      )
  ```
- [ ] Replace all instances with utility calls
- [ ] Add proper type hints and docstrings
- [ ] Test utility methods independently

### 3. Method Decomposition
**Objective**: Break large methods into focused functions

**Steps**:
- [ ] Identify methods >50 lines or with multiple responsibilities
- [ ] Extract logical blocks into helper methods
- [ ] Use descriptive names for extracted methods
- [ ] Maintain single responsibility principle
- [ ] Add docstrings to all new methods
- [ ] Ensure proper parameter passing

### 4. Error Handling Enhancement
**Objective**: Add robust error handling with user-friendly messages

**Steps**:
- [ ] Wrap file I/O operations in try-except blocks
- [ ] Provide meaningful error messages to users
- [ ] Continue gracefully for non-critical failures
- [ ] Log errors appropriately
- [ ] Test error scenarios

### 5. Type Safety Improvements
**Objective**: Add proper type hints and fix type issues

**Steps**:
- [ ] Fix implicit Optional types: `int = None` → `int | None = None`
- [ ] Add type hints to all method parameters and returns
- [ ] Use modern Python type syntax (`|` for unions)
- [ ] Run `mypy` if available for additional type checking

### 6. Universal Design Enforcement
**Objective**: Ensure code works with any model/architecture

**Steps**:
- [ ] Review all logic for model-specific assumptions
- [ ] Replace hardcoded names with generic PyTorch patterns
- [ ] Use `nn.Module` hierarchy and forward hooks
- [ ] Test with multiple different model architectures
- [ ] Document universal design principles used

### 7. Code Organization
**Objective**: Improve readability and maintainability

**Steps**:
- [ ] Group related methods with clear section separators:
  ```python
  # ========================================================================
  # CONFIGURATION
  # ========================================================================
  ```
- [ ] Use consistent naming conventions (snake_case)
- [ ] Maintain logical flow from initialization to cleanup
- [ ] Add comprehensive docstrings
- [ ] Remove dead code and unused imports

## ✅ Post-Refactoring Validation

### Code Quality Checks
- [ ] Run `ruff check` and fix all issues
- [ ] Run `ruff format` for consistent formatting
- [ ] Verify no hardcoded model-specific logic remains
- [ ] Check all constants moved to Config class
- [ ] Ensure all utility methods have docstrings

### Testing Validation
- [ ] Run full test suite: `uv run pytest tests/ -v`
- [ ] Verify 100% test pass rate
- [ ] Update test assertions for refactored structure
- [ ] Remove references to deleted classes/methods
- [ ] Test with multiple model architectures

### Functionality Validation
- [ ] Test end-to-end workflows
- [ ] Verify backward compatibility maintained
- [ ] Check error handling works as expected
- [ ] Validate performance hasn't degraded
- [ ] Test edge cases and error scenarios

## 📊 Success Metrics

### Quantitative Measures
- **Constants Extracted**: Count of hardcoded values moved to Config
- **Methods Created**: Number of utility methods for deduplication
- **Method Size Reduction**: Average method line count reduction
- **Test Pass Rate**: Must remain 100%
- **Lint Issues**: Must be 0 after refactoring

### Qualitative Measures
- **Readability**: Code is easier to understand
- **Maintainability**: Changes are easier to make
- **Reusability**: Common operations are abstracted
- **Consistency**: Similar operations use same patterns
- **Documentation**: Purpose and usage is clear

## 🔄 Refactoring Workflow Template

### Phase 1: Preparation (20% of time)
1. Run initial `ruff check` and document issues
2. Run full test suite and document current state
3. Identify refactoring targets using checklist
4. Plan approach and estimate effort

### Phase 2: Implementation (60% of time)
1. Create Config class and extract constants
2. Implement utility methods for deduplication
3. Decompose large methods into focused functions
4. Add error handling and type hints
5. Improve code organization and documentation

### Phase 3: Validation (20% of time)
1. Run `ruff check` and fix all issues
2. Run full test suite and fix any failures
3. Test end-to-end functionality
4. Validate universal design compliance
5. Document changes and update guidelines

## 🎯 Best Practices Summary

### Do's ✅
- Extract ALL magic values to centralized Config class
- Create utility methods for 3+ line repeated patterns
- Add comprehensive error handling with user-friendly messages
- Use descriptive names for all variables and methods
- Maintain universal design principles (no model-specific logic)
- Run `ruff check` after every change
- Test thoroughly before marking complete
- Document purpose and usage clearly

### Don'ts ❌
- Never hardcode model names, architectures, or operation names
- Don't skip linting or testing validation
- Don't create utility methods for single-use code
- Don't break backward compatibility without explicit planning
- Don't leave TODO comments or dead code
- Don't commit code that doesn't pass all tests
- Don't sacrifice readability for brevity

## 📝 Example Refactoring Checklist

Use this checklist for each refactoring session:

```
□ Pre-refactoring ruff check completed
□ Pre-refactoring test run (100% pass)
□ Constants extracted to Config class (count: ___)
□ Utility methods created (count: ___)
□ Large methods decomposed (methods: ___)
□ Error handling added to file operations
□ Type hints added/fixed (implicit Optional fixed)
□ Universal design validated (no hardcoded logic)
□ Code organization improved (sections added)
□ Post-refactoring ruff check (0 issues)
□ Post-refactoring test run (100% pass)
□ End-to-end functionality tested
□ Documentation updated
```

## 🚀 Tools & Commands

### Essential Commands
```bash
# Linting and formatting
ruff check .                    # Check for issues
ruff check --fix .             # Auto-fix issues
ruff format .                  # Format code

# Testing
uv run pytest tests/ -v        # Run all tests
uv run pytest tests/test_file.py -v  # Run specific test

# Type checking (if available)
mypy .                         # Type checking
```

### IDE Integration
- Configure IDE to run ruff on save
- Set up pre-commit hooks for automatic validation
- Use type checking plugins for real-time feedback

---

**Last Updated**: 2025-01-21  
**Version**: 1.0  
**Based on**: HTP Export Monitor refactoring experience