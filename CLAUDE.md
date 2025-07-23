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

## ⚠️ CARDINAL RULE #4 - MANDATORY TEST VERIFICATION

**THIS IS THE FOURTH CARDINAL RULE - NO EXCEPTIONS!**

❌ **NEVER**:
- Implement features without running test verification
- Revise test cases without confirming they pass
- Skip pytest validation after code changes
- Assume tests pass without verification

✅ **ALWAYS**:
- Run `uv run pytest tests/` after implementing features
- Run `uv run pytest tests/` after revising test cases
- Verify test results before marking tasks complete
- Use pytest to validate any code modifications

**After every implementation**: Run `uv run pytest tests/` to verify  
**After every test revision**: Run `uv run pytest tests/` to confirm  
**Before marking complete**: Ensure pytest verification passes

## MUST-RULES

Always follow MUST-RULES
- Rigorously adhere to universal design principles
- Prioritize generalizability over specific implementations
- Validate against core architectural constraints before committing any changes

## ⚠️ CARDINAL RULE #3 - MANDATORY ITERATION DOCUMENTATION

**THIS IS THE THIRD CARDINAL RULE - NO EXCEPTIONS!**

❌ **NEVER**:
- Complete an iteration without creating iteration notes
- Start a new iteration without reviewing previous iteration notes
- Skip todo updates after completing tasks
- Forget to record mistakes and insights

✅ **ALWAYS**:
- Create iteration notes immediately after each iteration using `/docs/design/iteration_note_template.md`
- Include: achievements, mistakes, insights, follow-up actions, updated todos
- Review the last 10 iteration notes before starting any new iteration
- Update todo list and append to iteration notes
- Follow the established plan and learn from recorded mistakes

**Before every iteration**: Read the last 10 iteration notes to understand context and avoid repeating mistakes  
**After every iteration**: Create comprehensive iteration notes using the template  
**When planning**: Use iteration notes to inform decisions and priorities

## Project Overview

This is a Python project named "modelexport" for universal hierarchy-preserving ONNX export of Hugging Face models. The project uses Python 3.12+ and is configured with pyproject.toml for dependency management.

## Development Commands

**IMPORTANT**: Always use uv with virtual environment for this project.

**FOR CLAUDE CODE**: Always use uv run or activate venv first. Never run bare python commands.

**TEMPORARY FILES**: Always use temp/ folder in project root to persist temporary files and test outputs.

**NODE.JS AVAILABILITY**: npm and npx are available via fnm (Fast Node Manager). Use `eval "$(fnm env)"` before npm/npx commands:
- Node.js version: v22.16.0
- npm version: 11.4.1
- Usage: `eval "$(fnm env)" && npm install` or `eval "$(fnm env)" && npx <command>`

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
uv pip install -e .

# CLI Commands (primary interface)
uv run modelexport export MODEL_NAME OUTPUT.onnx    # Export with hierarchy preservation
uv run modelexport export MODEL_NAME OUTPUT.onnx --clean-onnx  # Export without hierarchy_tag attributes (cleaner ONNX)
# Alternative: --no-hierarchy-attrs (same functionality)
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
uv run modelexport export prajjwal1/bert-tiny bert-clean.onnx --clean-onnx  # Clean ONNX without metadata
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

## Breakthrough: Built-in Module Tracking

The HTP strategy now uses PyTorch's built-in `torch.jit._trace._trace_module_map` infrastructure for direct module context capture during operation execution. This provides:

- **Better Layer Differentiation**: Perfect separation for simple models, significant improvement for complex models
- **Faster Export**: 29% performance improvement 
- **More Granular Tags**: Detailed module hierarchy like `/BertModel/Encoder/Layer.0/Attention/Self/Query`
- **Reduced Cross-Layer Contamination**: 33-50% reduction in layer tagging issues

## Core Components

1. **Structure Analysis**: Extract complete `nn.Module` hierarchy
2. **Direct Context Capture**: Use PyTorch's built-in module tracking during execution
3. **Hierarchy Preservation**: Add precise module metadata to ONNX export
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

## Memories

### Model Export Workflows
- Use `uv run modelexport export prajjwal1/bert-tiny temp/bert-tiny/model.onnx --strategy htp --config export_config_bertmodel.json` to export prajjwal1/bert-tiny
- Generate bert-tiny baseline with `uv run python tests/data/generate_test_data.py --model prajjwal1/bert-tiny --output-dir temp/baseline/bert-tiny/`
- Prefer opset_version=17 when converting to onnx model

### Critical Questioning
- Always ask question before planning and executing if you have questions or uncertainties for the requirements

### Code Quality
- Always ruff lint after revise the python code

### Git Commit Guidelines
- Never add `Co-Authored-By: Claude mailto:noreply@anthropic.com` when doing git commit