# Test Case: Basic Import

## Type
**Smoke Test** 🚭

## Purpose
Verify that the core modelexport module can be imported without errors. This is the most basic functionality test - if this fails, nothing else will work.

## Test Data (Fixtures)
- None required
- Pure import test

## Test Command
```bash
uv run python -c "from modelexport import HierarchyExporter; print('Import successful')"
```

Or via pytest:
```bash
uv run python -m pytest tests/test_hierarchy_exporter.py::TestHierarchyExporterBasic::test_hierarchy_exporter_can_be_imported -v
```

## Expected Behavior
- Import completes without exceptions
- HierarchyExporter class is available
- No initialization errors

## Failure Modes
- **ImportError**: Missing dependencies, broken package structure
- **ModuleNotFoundError**: Package not installed correctly
- **SyntaxError**: Code syntax issues in core modules

## Dependencies
- Python 3.12+
- modelexport package installed via `uv pip install -e .`

## Notes
- This test should complete in <1 second
- Fastest possible validation of package integrity
- First test to run in any CI/CD pipeline