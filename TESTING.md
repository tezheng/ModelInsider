# Universal Hierarchy Testing Workflow

This document describes the comprehensive testing workflow for the universal hierarchy-preserving ONNX export system.

## Overview

The testing workflow validates that our universal approach works correctly across different model architectures by:

1. **Generating static JSON test data** for each nn.Module in the hierarchy
2. **Converting models to ONNX** with hierarchy tags injected as node attributes
3. **Verifying consistency** between static data and ONNX model tags
4. **Testing multiple architectures** to ensure universality

## Test Models

The workflow tests 3 different model architectures:

- **BERT** (`google/bert_uncased_L-2_H-128_A-2`) - Transformer encoder
- **ResNet-18** (`torchvision.models.resnet18`) - Convolutional vision model  
- **ViT** (`google/vit-base-patch16-224`) - Vision transformer

## Scripts

### 1. `input_generator.py`
Universal input generator that creates appropriate dummy inputs for any model architecture.

```python
from input_generator import UniversalInputGenerator

generator = UniversalInputGenerator()
inputs = generator.generate_inputs(model, model_name)
```

**Features:**
- Detects model type automatically (transformer, vision, vision_transformer)
- Generates model-specific inputs (input_ids, pixel_values, etc.)
- Works with any PyTorch nn.Module

### 2. `test_universal_hierarchy.py`
Comprehensive testing script that runs the full workflow for all model types.

```bash
python3 test_universal_hierarchy.py
```

**What it does:**
1. Load each test model
2. Generate appropriate inputs using `UniversalInputGenerator`
3. Analyze model hierarchy using `DAGExtractor`
4. Export to ONNX with hierarchy tags
5. Generate static JSON test data
6. Verify tag consistency between static data and ONNX model
7. Save all results to `temp/` directory

### 3. `verify_onnx_tags.py`
Standalone verification script for checking ONNX model tags.

```bash
# Verify all models
python3 verify_onnx_tags.py --all

# Verify specific model
python3 verify_onnx_tags.py --model "google/bert_uncased_L-2_H-128_A-2"

# Quiet mode
python3 verify_onnx_tags.py --all --quiet
```

**Features:**
- Compares static JSON metadata with ONNX node attributes
- Calculates tag coverage and consistency rates
- Shows detailed differences and validation results
- Supports both individual and batch verification

### 4. `dag_extractor.py` (Updated)
Core DAG extraction engine, now uses `UniversalInputGenerator` for cleaner separation of concerns.

## Output Structure

All test outputs are saved to the `temp/` directory:

```
temp/
├── test_outputs/              # JSON test data
│   ├── google_bert_*_operation_metadata.json
│   ├── google_bert_*_module_dags.json
│   ├── google_bert_*_hierarchy.json
│   ├── resnet18_operation_metadata.json
│   ├── resnet18_module_dags.json
│   └── comprehensive_test_results.json
├── onnx_models/               # ONNX files
│   ├── google_bert_*.onnx
│   ├── google_bert_*_with_tags.onnx
│   ├── resnet18.onnx
│   └── resnet18_with_tags.onnx
└── test_data/                 # Input data
    ├── google_bert_*_inputs.json
    └── resnet18_inputs.json
```

## File Descriptions

### Static JSON Files

**`*_operation_metadata.json`**: Complete operation metadata with tags
```json
{
  "MatMul_123": {
    "op_type": "MatMul",
    "inputs": ["input_ids", "embeddings.weight"],
    "outputs": ["embedding_output"],
    "tags": ["/BertModel/BertEmbeddings"]
  }
}
```

**`*_module_dags.json`**: DAG structure for each module
```json
{
  "/BertModel/BertEmbeddings": {
    "nodes": ["embeddings.weight", "MatMul_123", "Add_456"],
    "edges": [["embeddings.weight", "MatMul_123"], ["MatMul_123", "Add_456"]]
  }
}
```

**`*_hierarchy.json`**: Complete module hierarchy analysis
```json
{
  "embeddings": {
    "hierarchy_path": "/BertModel/BertEmbeddings",
    "type": "BertEmbeddings", 
    "depth": 1,
    "is_leaf": false,
    "parameter_count": 3584000
  }
}
```

### ONNX Files

**`*_with_tags.onnx`**: Enhanced ONNX model with hierarchy tags as node attributes
- Each node contains `source_module` attribute
- Shared operations have `hierarchy_tags` with multiple modules
- Model metadata includes complete hierarchy and operation info

## Verification Process

The verification process ensures that:

1. **Tag Coverage**: High percentage of ONNX nodes have hierarchy tags
2. **Tag Consistency**: Tags in ONNX match the static JSON metadata
3. **Module Coverage**: All meaningful modules have operations assigned
4. **Parameter Tracking**: Shared parameters are correctly tagged

### Success Criteria

- ✅ **Tag Coverage** ≥ 70% (most nodes should have hierarchy tags)
- ✅ **Tag Consistency** ≥ 90% (tags should match between static data and ONNX)
- ✅ **Module Coverage** ≥ 80% (most modules should have at least one operation)
- ✅ **Zero Critical Errors** (all model tests should complete successfully)

## Running the Tests

### Prerequisites

Install dependencies:
```bash
uv pip install -e .
# or
pip install torch transformers onnx numpy torchvision
```

### Full Test Suite

```bash
# Run complete test workflow for all models
python3 test_universal_hierarchy.py
```

### Individual Model Testing

```bash
# Test just BERT
python3 dag_extractor.py

# Test with specific model in input generator
python3 input_generator.py
```

### Verification Only

```bash
# Verify all models (run this after test_universal_hierarchy.py)
python3 verify_onnx_tags.py --all

# Get detailed verification for specific model
python3 verify_onnx_tags.py --model "google/bert_uncased_L-2_H-128_A-2"
```

## Expected Output

When everything works correctly, you should see:

```
✅ BERT: SUCCESS
   Modules: 45
   Operations: 156
   Tag coverage: 78.4%

✅ RESNET: SUCCESS  
   Modules: 68
   Operations: 142
   Tag coverage: 82.1%

✅ VIT: SUCCESS
   Modules: 52
   Operations: 198
   Tag coverage: 75.3%

VERIFICATION SUMMARY
Total models: 3
Successful: 3
Failed: 0
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU
3. **Memory Issues**: Use smaller batch sizes or models
4. **Missing Models**: Some HuggingFace models require internet access

### Debug Mode

For debugging, check individual files in `temp/test_outputs/` to inspect:
- Module hierarchy structure
- Operation metadata and tags
- DAG connectivity for specific modules

### Validation Failures

If tag consistency is low:
1. Check operation naming consistency between PyTorch and ONNX
2. Verify parameter mapping is correctly identifying shared parameters
3. Look at `*_operation_metadata.json` to see what tags were assigned

## Key Insights

The testing workflow validates these critical aspects of the universal approach:

1. **Architecture Independence**: Same codebase works for transformers, CNNs, and ViTs
2. **Tag Preservation**: Hierarchy information survives ONNX export process
3. **Parameter Sharing**: Shared parameters (like tied embeddings) are correctly identified
4. **DAG Completeness**: All meaningful operations are captured and connected

This comprehensive testing ensures the universal hierarchy-preserving ONNX export system works reliably across different model architectures.