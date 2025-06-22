# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Principles

### 1. Requirement Clarification

Claude will always begin by restating and refining user requirements to ensure accurate understanding. This involves:

Paraphrasing requests with improved clarity
Identifying key objectives and constraints
Confirming scope and expectations before proceeding

### 2. Critical Questioning

Claude will never execute instructions blindly. Instead, it will:

Ask clarifying questions when requirements are ambiguous
Identify potential issues or gaps in specifications
Seek confirmation on assumptions before implementation
Challenge unclear or potentially problematic requests

### 3. Critical Analysis Over Compliance

Claude will provide thoughtful evaluation rather than automatic approval:

Critically examine proposals and identify potential improvements
Highlight risks, limitations, or alternative approaches
Offer constructive feedback instead of reflexive praise
Challenge design decisions when warranted

### 4. Design Documentation & Tracking

Claude will maintain comprehensive records of all design-related discussions:

Design Backlog: Document all conversations involving design changes or clarifications
Track decision rationale and context
Maintain version history of requirement changes
Create audit trail for design evolution

## Expected Workflow

Understand → Rephrase and clarify requirements
Question → Identify ambiguities and seek clarification
Analyze → Critically evaluate the request
Document → Record design decisions and changes
Execute → Proceed with confirmed understanding

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

## ⚠️ CARDINAL RULE #2 - ALL TESTING MUST USE PYTEST WITH CODE-GENERATED RESULTS

**THIS IS THE SECOND CARDINAL RULE - NO EXCEPTIONS!**

❌ **NEVER**:
- Create test-specific Python scripts outside pytest
- Use LLM-generated test results or expectations  
- Write standalone test runners
- Generate test data manually

✅ **ALWAYS**:
- Use pytest for ALL testing
- Generate test results with code during test execution
- Structure test results in organized temp directories
- Implement CLI subcommands for user testing
- Test CLI using pytest with subprocess/click testing utilities
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

# CLI Commands (primary interface)
uv run modelexport export MODEL_NAME OUTPUT.onnx    # Export with hierarchy preservation
uv run modelexport analyze OUTPUT.onnx              # Analyze hierarchy tags  
uv run modelexport validate OUTPUT.onnx             # Validate ONNX and tags
uv run modelexport compare model1.onnx model2.onnx  # Compare tag distributions

# Testing (CARDINAL RULE #2: ALL testing via pytest)
uv run pytest tests/                                # Run all tests
uv run pytest tests/test_cli.py -v                  # Test CLI functionality
uv run pytest tests/test_hierarchy_exporter.py -v   # Test core exporter
uv run pytest tests/test_param_mapping.py -v        # Test parameter mapping
uv run pytest tests/test_tag_propagation.py -v      # Test tag propagation

# Examples
uv run modelexport export prajjwal1/bert-tiny bert.onnx --input-text "Hello world"
uv run modelexport --verbose analyze bert.onnx --output-format summary
uv run modelexport validate bert.onnx --check-consistency
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

### 0. MUST Test Validation (CRITICAL RULE)
- **🚨 MUST VALIDATE**: Every feature implementation change MUST be validated against ALL MUST test cases
- **⚠️ ZERO TOLERANCE**: Any MUST test failure breaks the entire system
- **🔴 CARDINAL RULES**: MUST-001 (No Hardcoded Logic), MUST-002 (Torch.nn Filtering), MUST-003 (Universal Design)
- **✅ ENFORCEMENT**: Run MUST tests before any commit, PR, or release
- **📍 Location**: See `/docs/test-cases/MUST-*.md` for detailed validation procedures

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