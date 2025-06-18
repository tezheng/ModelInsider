# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ UNIVERSAL RULE #1 - ABSOLUTELY NO HARDCODED LOGIC

**THIS IS THE CARDINAL RULE - NO EXCEPTIONS AT ANY TIME!**

❌ **NEVER HARDCODE**:
- Model architecture names (BERT, GPT, ResNet, etc.)
- Node names or operation names  
- Input/output tensor names
- Layer naming patterns
- Class name string matching
- Model-specific logic of ANY kind

✅ **ALWAYS USE UNIVERSAL APPROACHES**:
- `nn.Module` hierarchy (`model.named_modules()`)
- Forward hooks for execution tracing
- Parameter-based and execution-based tagging only
- Dynamic analysis, never static assumptions

**Before every code change**: Ask "Is this hardcoded to any specific architecture?"  
**After every test fix**: Review the diff for model-specific assumptions  
**When tests fail**: Fix with universal logic, never hardcoded patches

## Project Overview

This is a Python project named "modelexport" for universal hierarchy-preserving ONNX export of Hugging Face models. The project uses Python 3.12+ and is configured with pyproject.toml for dependency management.

## Development Commands

**IMPORTANT**: Always use uv with virtual environment for this project.

**FOR CLAUDE CODE**: Always use uv run or activate venv first. Never run bare python commands.

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
uv pip install -e .

# Run testing workflows (use uv run or activate venv first)
uv run python bert_self_attention_test.py --debug-info
uv run python test_universal_hierarchy.py    # Test all 3 model architectures
uv run python verify_onnx_tags.py --all      # Verify ONNX tag consistency
uv run python dag_extractor.py               # Test BERT model specifically

# Run individual scripts
uv run python input_generator.py             # Test input generation
uv run python main.py                        # Basic hello world
```

## Project Structure

- `main.py` - Entry point with basic hello world functionality
- `conversion/hf_universal_hierarchy_exporter.py` - Universal hierarchy-preserving ONNX exporter for any HF model
- `conversion/` - Contains other conversion utilities and experiments
- `data/` - Contains exported ONNX models and metadata
- `tests/` - Test files for validation
- `pyproject.toml` - Project configuration and dependencies

## Key Insight

Every HF model is just a PyTorch `nn.Module` with inherent hierarchy. We leverage this universal structure instead of model-specific logic.

## Universal Approach

The exporter works with ANY model because:
1. **nn.Module Hierarchy**: Every model has `named_modules()` - this IS the hierarchy
2. **Hooks for Tracing**: Forward hooks map ONNX operations to source modules  
3. **ONNX Metadata**: Preserve hierarchy info in exported model
4. **No Model-Specific Code**: Works at the fundamental PyTorch level

See `MEMO.md` for detailed rationale.

## Core Components

1. **Structure Analysis**: Extract complete `nn.Module` hierarchy
2. **Execution Tracing**: Use hooks to map operations to modules
3. **Hierarchy Preservation**: Add module metadata to ONNX export
4. **Universal**: Works with any PyTorch model automatically

## Development Rules of Thumb

### 1. Universal Design Principles
- **Target**: Universal hierarchy-preserving ONNX export for HuggingFace models
- **NO HARDCODED LOGIC**: Absolutely no hardcoded model architectures, node names, operator names, or any similar model-specific patterns
- **Universal First**: Always design solutions that work for ANY model, not just specific architectures
- **Architecture Agnostic**: Leverage fundamental PyTorch structures (`nn.Module`, hooks, named_modules) that exist in all models

### 2. Test-Driven Development
- **Always Create Test Cases**: Every feature must have corresponding tests
- **TDD When Possible**: Write tests before implementation to define expected behavior
- **Comprehensive Testing**: Target both unit tests (individual functions) and integration tests (end-to-end workflows)
- **Test Multiple Architectures**: Verify solutions work across different model types (BERT, ResNet, GPT, etc.)

### 3. Code Quality Standards  
- **Clean Up After Each Iteration**: Refactor and clean code after implementing features
- **Use Linting Tools**: Apply tools like `black`, `ruff`, or `flake8` to maintain code standards
- **Remove Dead Code**: Delete unused functions, commented-out code, and obsolete implementations
- **Consistent Formatting**: Maintain consistent code style throughout the project

### 4. Pythonic Practices
- **Follow Python Conventions**: Use PEP 8 style guidelines and Python idioms
- **Type Hints**: Add type annotations for better code documentation and IDE support
- **Descriptive Names**: Use clear, self-documenting variable and function names
- **List/Dict Comprehensions**: Prefer Pythonic constructs over verbose loops where appropriate
- **Context Managers**: Use `with` statements for resource management
- **Exception Handling**: Use specific exception types and proper error handling patterns