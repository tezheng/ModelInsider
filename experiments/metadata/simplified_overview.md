# Simple Overview: What Each File Does

## The Problem You Had
"metadata generation looks a bit noisy, not clear"

## The Solution: 3 Things

### 1. 📝 **Clean Generation** (metadata_builder.py)
**What**: Makes metadata generation clean and organized
**Status**: ✅ Already integrated into HTP exporter
```python
# Instead of messy dict creation, now it's:
builder.with_model_info(...)
      .with_modules(...)
      .build()
```

### 2. 🔍 **Easy Queries** (metadata_cli_utils.py + advanced_metadata.py)
**What**: Query any part of metadata instantly
**Status**: 📦 Ready to add to CLI
```bash
# Quick debugging
modelexport analyze model.onnx --query "/modules/encoder.layer.0"
modelexport analyze model.onnx --query "/tagging/coverage"
```

### 3. 🔧 **Quick Updates** (metadata_patch_cli.py + advanced_metadata.py) 
**What**: Fix metadata without re-exporting
**Status**: 📦 Ready to add to CLI
```bash
# Fix coverage after manual corrections
modelexport patch coverage metadata.json --coverage 98.5 --tagged 145
```

## Which File Does What?

```
metadata_builder.py     = Creates metadata (like a form builder)
advanced_metadata.py    = Core tools (like a Swiss Army knife)
metadata_cli_utils.py   = Query commands (like grep for metadata)
metadata_patch_cli_py   = Update commands (like sed for metadata)
metadata_models.py      = Future alternative with Pydantic
```

## If Names Are Confusing, Think of Them As:

- **Builder** → "Metadata Factory" (creates new metadata)
- **CLI Utils** → "Metadata Inspector" (looks inside metadata)
- **Patch CLI** → "Metadata Editor" (updates existing metadata)
- **Advanced** → "Metadata Toolkit" (tools used by Inspector & Editor)
- **Models** → "Metadata Blueprint" (future schema-based approach)

## The Big Picture

```
Stage 1: BUILD
HTP Exporter → uses Builder → creates clean metadata.json

Stage 2: INSPECT  
User → uses CLI Utils → queries metadata quickly

Stage 3: FIX
User → uses Patch CLI → updates metadata without re-export
```

That's it! The rest is implementation details.