# Makefile for modelexport testing and development

# Variables
PYTHON := uv run python
PYTEST := uv run pytest
PIP := uv pip

# Default target
.PHONY: all
all: test-fast

# Help target
.PHONY: help
help:
	@echo "Modelexport Development Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  test-fast      Run fast unit tests (excludes slow tests)"
	@echo "  test-unit      Run all unit tests"
	@echo "  test-integration  Run integration tests"
	@echo "  test-strategy  Run strategy-specific tests"
	@echo "  test-fx        Run FX strategy tests only"
	@echo "  test-htp       Run HTP strategy tests only"
	@echo "  test-usage     Run usage-based strategy tests only"
	@echo "  test-cli       Run CLI integration tests"
	@echo "  test-all       Run all tests (including slow ones)"
	@echo "  test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  install        Install package in development mode"
	@echo "  install-dev    Install with all development dependencies"
	@echo "  lint           Run code linting"
	@echo "  format         Format code with black"
	@echo "  type-check     Run type checking with mypy"
	@echo "  clean          Clean temporary files and caches"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Generate documentation"
	@echo "  docs-serve     Serve documentation locally"

# Installation targets
.PHONY: install
install:
	$(PIP) install -e .

.PHONY: install-dev
install-dev:
	$(PIP) install -e ".[dev,test]"

# Testing targets
.PHONY: test-fast
test-fast:
	@echo "Running fast unit tests..."
	$(PYTEST) tests/unit/ -m "not slow" -v

.PHONY: test-unit
test-unit:
	@echo "Running all unit tests..."
	$(PYTEST) tests/unit/ -v

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(PYTEST) tests/integration/ -v

.PHONY: test-strategy
test-strategy:
	@echo "Running strategy-specific tests..."
	$(PYTEST) tests/unit/test_strategies/ -v

.PHONY: test-fx
test-fx:
	@echo "Running FX strategy tests..."
	$(PYTEST) -m fx -v

.PHONY: test-htp
test-htp:
	@echo "Running HTP strategy tests..."
	$(PYTEST) -m htp -v

.PHONY: test-usage
test-usage:
	@echo "Running usage-based strategy tests..."
	$(PYTEST) -m usage_based -v

.PHONY: test-cli
test-cli:
	@echo "Running CLI integration tests..."
	$(PYTEST) -m cli -v

.PHONY: test-all
test-all:
	@echo "Running all tests..."
	$(PYTEST) tests/ -v

.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ --cov=modelexport --cov-report=html --cov-report=term-missing -v

.PHONY: test-parallel
test-parallel:
	@echo "Running tests in parallel..."
	$(PYTEST) tests/ -n auto -v

# Development targets
.PHONY: lint
lint:
	@echo "Running linting..."
	$(PYTHON) -m ruff check modelexport/ tests/
	$(PYTHON) -m black --check modelexport/ tests/

.PHONY: format
format:
	@echo "Formatting code..."
	$(PYTHON) -m black modelexport/ tests/
	$(PYTHON) -m ruff --fix modelexport/ tests/

.PHONY: type-check
type-check:
	@echo "Running type checking..."
	$(PYTHON) -m mypy modelexport/

# Cleanup targets
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.onnx" -delete
	find . -type f -name "*_hierarchy.json" -delete
	find . -type f -name "*_module_info.json" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf temp/

.PHONY: clean-test
clean-test:
	@echo "Cleaning test artifacts..."
	find tests/ -type f -name "*.onnx" -delete
	find tests/ -type f -name "*_hierarchy.json" -delete
	find . -name "test_*.onnx" -delete
	rm -rf .pytest_cache/

# CLI testing shortcuts
.PHONY: test-cli-help
test-cli-help:
	@echo "Testing CLI help commands..."
	$(PYTHON) -m modelexport --help
	$(PYTHON) -m modelexport export --help
	$(PYTHON) -m modelexport analyze --help
	$(PYTHON) -m modelexport validate --help
	$(PYTHON) -m modelexport compare --help

# Development workflow targets
.PHONY: dev-setup
dev-setup: install-dev
	@echo "Development environment setup complete!"

.PHONY: pre-commit
pre-commit: format lint test-fast
	@echo "Pre-commit checks passed!"

.PHONY: ci-test
ci-test: test-unit test-integration
	@echo "CI tests completed!"

# Experiment targets
.PHONY: test-experiments
test-experiments:
	@echo "Running experimental tests..."
	$(PYTEST) scripts/experiment/ -v

# Strategy-specific development
.PHONY: test-fx-dev
test-fx-dev:
	@echo "Running FX development tests..."
	$(PYTEST) tests/unit/test_strategies/fx/ -v -s

.PHONY: test-htp-dev
test-htp-dev:
	@echo "Running HTP development tests..."
	$(PYTEST) tests/unit/test_strategies/htp/ -v -s

# Performance testing
.PHONY: test-performance
test-performance:
	@echo "Running performance tests..."
	$(PYTEST) -m slow -v

# Specific test patterns
test-pattern:
	@echo "Running tests matching pattern: $(PATTERN)"
	$(PYTEST) -k "$(PATTERN)" -v

# Test with specific Python version
test-python:
	@echo "Running tests with Python $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m pytest tests/ -v

# Documentation targets (placeholders)
.PHONY: docs
docs:
	@echo "Documentation generation not yet implemented"

.PHONY: docs-serve
docs-serve:
	@echo "Documentation serving not yet implemented"