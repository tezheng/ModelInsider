# Models Directory

This directory contains ONNX model files and test models used for development and testing.

## Important Notes

⚠️ **Model files are NOT tracked in git** - The `models/` directory is listed in `.gitignore` to prevent large binary files from being committed to the repository.

## Test Models

### bert-tiny-optimum/
Test model for ONNX inference validation. This model should be downloaded or generated locally.

To set up test models:
```bash
# Export a test model using modelexport
uv run modelexport export prajjwal1/bert-tiny models/bert-tiny-optimum/model.onnx
```

## Directory Structure
```
models/
├── README.md          # This file (tracked in git)
└── bert-tiny-optimum/ # Test model directory (NOT tracked)
    ├── model.onnx
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## Why Not in Git?

1. **Size**: ONNX model files are large binary files (often 10MB-1GB+)
2. **Version Control**: Binary files don't benefit from git's diff/merge capabilities
3. **Repository Performance**: Large files slow down clone/fetch operations
4. **Best Practice**: Models should be stored in artifact repositories or downloaded on-demand

## Alternatives for Model Storage

- Cloud storage (S3, GCS, Azure Blob)
- Model registries (HuggingFace Hub, MLflow)
- Artifact repositories (Artifactory, Nexus)
- Git LFS (for smaller models if absolutely necessary)