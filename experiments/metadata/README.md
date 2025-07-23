# Metadata Experiments and Research

This folder contains experimental code and research artifacts from exploring metadata generation improvements for the HTP exporter.

## Background

The user asked: "is there a way to implement this as template+data, currently it works but look a bit noisy, not clear"

This led to research on best practices for metadata generation and advanced JSON Schema features.

## What's Here

### Alternative Implementations
- `metadata_models.py` - Pydantic-based approach (requires Pydantic dependency)
- `pydantic_builder.py` - Another Pydantic variation
- `templates/metadata.json.j2` - Jinja2 template approach

### Advanced Features (Could be moved back if needed)
- `advanced_metadata.py` - JSON Pointer, query, and patch utilities
- `metadata_cli_utils.py` - CLI query integration
- `metadata_patch_cli.py` - CLI patch/update commands

### Problematic Files (Violate Cardinal Rule #1)
- `auto_validation.py` - ❌ Has hardcoded model-specific logic
- `conditional_schemas.py` - ❌ Has hardcoded model patterns
- `metadata_schema.py` - Old/duplicate schema definition

### Fixed Versions (Universal Approach)
- `auto_validation_universal.py` - Universal validation without hardcoded logic
- `universal_schemas.py` - Universal schema definitions

### Documentation and Analysis
- `REVIEW_FINDINGS.md` - Code review findings
- `integration_plan.md` - Plan for integrating advanced features
- `practical_features_summary.md` - Summary of useful features
- `metadata_files_explanation.md` - Explanation of all files
- `simplified_overview.md` - Simple explanation
- `FILE_CLEANUP_ANALYSIS.md` - Analysis of which files to keep/delete

### Examples
- `cli_integration_example.py` - Example of CLI integration

## What Was Kept in Production

Only 3 essential files remain in `modelexport/strategies/htp/`:
1. `__init__.py` - Package marker
2. `htp_exporter.py` - Core exporter
3. `metadata_builder.py` - Clean metadata generation (actively used)

## Potential Future Use

If you want to add query/patch features to the CLI:
1. Move back `advanced_metadata.py` (core utilities)
2. Move back `metadata_cli_utils.py` (for --query support)
3. Move back `metadata_patch_cli.py` (for patch commands)

## Lessons Learned

1. The `metadata_builder.py` approach worked well for cleaning up metadata generation
2. JSON Pointer queries would be very useful for debugging
3. Pydantic with JSON Schema would be good when available
4. Must avoid any hardcoded model-specific logic (Cardinal Rule #1)