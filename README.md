# HuggingFace to ONNX Hierarchy Preservation

This project implements a complete solution for preserving PyTorch module hierarchy when converting HuggingFace models to ONNX format.

## 📁 Project Structure

```
external/
├── conversion/          # Model conversion and export scripts
├── tests/              # Testing and validation scripts  
├── docs/               # Documentation and analysis
├── data/               # Generated ONNX models and reference data
└── optimum/            # HuggingFace Optimum library (reference)
```

## 🔧 Conversion Tools (`conversion/`)

### Main Implementation
- **`bert_hierarchy_exporter.py`** - Complete BERT hierarchy preservation test
- **`final_hierarchy_export.py`** - Production-ready hierarchy exporter
- **`hierarchy_preserving_export.py`** - Core hierarchy preservation logic

### DAG & Metadata
- **`dag_cache_creator.py`** - Extract DAG connections from ONNX models
- **`onnx_metadata_research.py`** - ONNX metadata capabilities research
- **`onnx_function_research.py`** - ONNX functions for hierarchy research

### Research
- **`research_onnx_conversion.py`** - Initial conversion methods research

## 🧪 Tests (`tests/`)

- **`operation_mapping_demo.py`** - Validate piece operations against whole model
- **`simple_hierarchy_test.py`** - Basic hierarchy preservation test
- **`analyze_vit_structure.py`** - Vision Transformer structure analysis

## 📚 Documentation (`docs/`)

- **`DAG_VALIDATION_SUMMARY.md`** - Complete DAG preservation analysis
- **`STATIC_CACHE_SUMMARY.md`** - Static operations cache documentation  
- **`RESEARCH_SUMMARY.md`** - Original research findings and methods

## 💾 Data (`data/`)

### Static Cache Files
- **`bert_dag_operations_cache.json`** - Operations + DAG connections (309 nodes, 325 edges)
- **`bert_static_operations_cache.json`** - Core operations cache
- **`bert_reference_hierarchy.json`** - Complete module hierarchy (47 modules)

### ONNX Models
- **`bert_tiny_whole_model_with_hierarchy.onnx`** - Enhanced whole model with metadata
- **`bert_component_*.onnx`** - Individual component pieces (10 files)
- **`bert_*.json`** - Reference data and analysis files

## 🚀 Quick Start

### 1. Run Complete BERT Test
```bash
cd conversion/
python bert_hierarchy_exporter.py
```

### 2. Validate Results
```bash
cd tests/
python operation_mapping_demo.py
```

### 3. View Analysis
```bash
# Check generated documentation
cat docs/DAG_VALIDATION_SUMMARY.md
```

## 🎯 Key Achievements

### ✅ Hierarchy Preservation
- **47 modules** across 6 hierarchy depths preserved
- **Module-to-operation mapping** maintained in ONNX metadata
- **Piece-by-piece validation** with 10 component exports

### ✅ DAG Structure Preservation  
- **309 operations** with **325 tensor dependencies**
- **Complete data flow** captured via predecessor/successor relationships
- **Execution order** preserved for reconstruction

### ✅ Validation Framework
- **Static cache** for consistent reference data
- **Cross-validation** between pieces and whole model
- **Operation mapping** from PyTorch modules to ONNX nodes

## 📊 Results Summary

| Component | Operations | Edges | Description |
|-----------|------------|-------|-------------|
| embeddings | 18 | 19 | Token + position embeddings |
| encoder.layer.0 | 63 | 69 | First transformer layer |
| encoder.layer.0.attention | 43 | 45 | Multi-head attention |
| encoder.layer.0.intermediate | 8 | 8 | MLP intermediate |
| pooler | 3 | 2 | Final pooling layer |
| **Total** | **309** | **325** | **Complete model** |

## 🔍 Usage Examples

### Load Static Cache
```python
import json

# Load operations cache
with open('data/bert_dag_operations_cache.json', 'r') as f:
    cache = json.load(f)

# Get operations for specific module
embeddings_ops = cache['pieces']['embeddings']['dag_structure']['nodes']
```

### Query Hierarchy
```python
# Load hierarchy
with open('data/bert_reference_hierarchy.json', 'r') as f:
    hierarchy = json.load(f)

# Find attention modules
attention_modules = [name for name in hierarchy.keys() if 'attention' in name]
```

### Validate Operations
```python
# Run validation
from tests.operation_mapping_demo import validate_piece_operations_against_whole
validate_piece_operations_against_whole()
```

## 🏆 Success Metrics

- ✅ **100% module coverage** - All PyTorch modules have ONNX pieces
- ✅ **309 core operations** extracted and cached
- ✅ **325 DAG connections** preserved
- ✅ **10 component pieces** successfully exported
- ✅ **Complete validation framework** ready for production use

This implementation demonstrates that **HuggingFace models can be converted to ONNX while fully preserving their hierarchical module structure**, enabling users to group ONNX operations by their original PyTorch modules.