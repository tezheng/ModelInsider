# Debug Scripts

This directory contains debugging and development scripts used during the development of the hierarchy exporter.

## Scripts

- **debug_path_building.py**: Tests hierarchical path building logic
- **debug_tagging_simple.py**: Tests complete tagging workflow with simple inputs
- **debug_hook_execution.py**: Debugs forward hook execution and context capture
- **debug_simple_model.py**: Tests tagging with basic PyTorch models

## Usage

These scripts are intended for development and debugging purposes only. They are not part of the production codebase.

```bash
# Run from project root
uv run python debug/debug_tagging_simple.py
uv run python debug/debug_path_building.py
```

## Purpose

These scripts were created during the 5-round testing and fixing process to:

1. Debug tag propagation issues
2. Validate hierarchical path building
3. Test torch.nn module filtering
4. Verify hook execution order

They serve as examples of how to test the hierarchy exporter interactively.