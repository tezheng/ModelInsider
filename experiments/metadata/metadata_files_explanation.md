# HTP Metadata Files Explanation

## Overview
These files were created to solve the "noisy metadata generation" problem you mentioned. They provide different approaches and utilities for cleaner metadata handling.

## Core Files (Primary Usage)

### 1. **metadata_builder.py** - The Main Builder (USE THIS)
**Purpose**: Clean, structured way to build metadata step-by-step
**When to use**: When generating metadata in HTP exporter
**Usage**:
```python
# In htp_exporter.py
from .metadata_builder import HTPMetadataBuilder

builder = HTPMetadataBuilder()
metadata = (
    builder
    .with_export_context(strategy="htp", version="1.0")
    .with_model_info(name_or_path="model", class_name=model.__class__.__name__, ...)
    .with_modules(modules_dict)
    .build()
)
```
**Status**: ✅ Integrated into HTP exporter

### 2. **metadata_models.py** - Pydantic Alternative (FUTURE USE)
**Purpose**: Alternative to metadata_builder using Pydantic models
**When to use**: When Pydantic is added as dependency
**Why it exists**: You said "i think schema is great" - Pydantic provides automatic JSON Schema generation
**Usage**:
```python
# Future usage when Pydantic is available
from .metadata_models import HTPMetadataModel

metadata = HTPMetadataModel(
    export_context=ExportContextModel(...),
    model=ModelInfoModel(...),
)
json_schema = metadata.model_json_schema()  # Auto-generated schema
```
**Status**: 🔮 Ready for future when Pydantic is added

## Advanced Features (JSON Schema 2020-12)

### 3. **advanced_metadata.py** - Core Utilities
**Purpose**: Foundational utilities for JSON Pointer, queries, and patches
**When to use**: As a library imported by other utilities
**Key Classes**:
- `MetadataPointer`: Navigate metadata using JSON Pointer syntax
- `MetadataQuery`: Search and analyze metadata  
- `MetadataPatch`: Update metadata without full regeneration

**Example**:
```python
# Get specific field
value = MetadataPointer.get(metadata, "/modules/encoder.layer.0")

# Search for patterns
query = MetadataQuery(metadata)
bert_layers = query.find_modules_by_class("BertLayer")
```
**Status**: 🔧 Utility library for other tools

### 4. **metadata_cli_utils.py** - CLI Query Features
**Purpose**: Add query capabilities to modelexport CLI
**When to use**: Integrate into CLI for metadata analysis
**Features**:
- Query with JSON Pointer: `--query "/modules/encoder.layer.0"`
- Pattern search: `--query "find:modules:*Layer"`
- Consistency validation: `--validate-consistency`

**Integration**:
```python
# In cli.py
@click.option('--query', help='JSON Pointer or pattern')
def analyze(onnx_path, query=None):
    if query:
        result = MetadataCLI.query_metadata(metadata_path, query)
```
**Status**: 📦 Ready to integrate into CLI

### 5. **metadata_patch_cli.py** - CLI Update Features  
**Purpose**: Update metadata without re-exporting the model
**When to use**: Fix metadata issues post-export
**Features**:
- Update coverage: `modelexport patch coverage metadata.json --coverage 98.5`
- Add analysis: `modelexport patch add-analysis metadata.json perf results.json`
- Mark issues: `modelexport patch mark-issues metadata.json issues.json`

**Integration**:
```python
# New CLI command group
@cli.group()
def patch():
    """Update metadata files"""
    pass
```
**Status**: 📦 Ready to integrate into CLI

## File Relationships

```
metadata_builder.py (MAIN - Currently Used)
    ↓
HTP Exporter generates metadata
    ↓
metadata.json file created
    ↓
metadata_cli_utils.py (Query the metadata)
    - Uses advanced_metadata.py internally
    ↓
metadata_patch_cli.py (Update the metadata)
    - Uses advanced_metadata.py internally

metadata_models.py (Alternative to builder - Future use with Pydantic)
```

## Simplified Naming Suggestions

If the names are confusing, here's a clearer structure:

```
Current Name              →  Clearer Name
metadata_builder.py       →  builder.py (keep as is - clear enough)
metadata_models.py        →  pydantic_models.py (indicates Pydantic dependency)
advanced_metadata.py      →  core_utilities.py (base utilities)
metadata_cli_utils.py     →  query_commands.py (CLI query features)
metadata_patch_cli.py     →  update_commands.py (CLI update features)
```

## Which Files to Use Now?

1. **For metadata generation**: Use `metadata_builder.py` ✅ (already integrated)
2. **For adding query features**: Integrate `metadata_cli_utils.py`
3. **For adding update features**: Integrate `metadata_patch_cli.py`
4. **Don't use yet**: `metadata_models.py` (waiting for Pydantic)
5. **Don't use directly**: `advanced_metadata.py` (used by other utilities)

## Simple Example Workflow

```bash
# 1. Export model (uses metadata_builder.py internally)
modelexport export bert model.onnx

# 2. Query metadata (would use metadata_cli_utils.py)
modelexport analyze model.onnx --query "/tagging/coverage/coverage_percentage"
# Output: 94.5

# 3. Fix coverage and update (would use metadata_patch_cli.py)  
modelexport patch coverage model_metadata.json --coverage 98.5 --tagged 145

# 4. Validate consistency (would use metadata_cli_utils.py)
modelexport analyze model.onnx --validate-consistency
# Output: ✅ Valid
```

## Summary

- **metadata_builder.py**: Main tool for creating metadata (currently used)
- **advanced_metadata.py**: Foundation library (JSON Pointer, queries, patches)
- **metadata_cli_utils.py**: CLI features for querying metadata
- **metadata_patch_cli.py**: CLI features for updating metadata
- **metadata_models.py**: Future Pydantic-based alternative

The core idea: Clean metadata generation (builder) + powerful query/update capabilities (CLI utils)