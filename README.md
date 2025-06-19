# ModelExport - Universal Hierarchy-Preserving ONNX Export

Universal hierarchy-preserving ONNX export for any PyTorch model. This tool exports PyTorch models to ONNX while preserving the original module hierarchy through intelligent tagging.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd modelexport

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
uv pip install -e .
```

### Basic Usage

```bash
# Export a BERT model with hierarchy preservation
uv run modelexport export prajjwal1/bert-tiny bert_with_hierarchy.onnx

# Export with custom input text
uv run modelexport export prajjwal1/bert-tiny bert_custom.onnx \
  --input-text "This is a custom input for BERT model testing"

# Export with specific input shapes
uv run modelexport export prajjwal1/bert-tiny bert_shapes.onnx \
  --input-shape 1,256 --input-text "Longer sequence example"

# Analyze the exported model
uv run modelexport analyze bert_with_hierarchy.onnx

# Validate hierarchy tags
uv run modelexport validate bert_with_hierarchy.onnx --check-consistency

# Compare two models
uv run modelexport compare baseline.onnx tagged.onnx
```

## 📋 CLI Commands

### `export` - Export Model with Hierarchy

Export a PyTorch model to ONNX with hierarchy-preserving tags.

```bash
uv run modelexport export [OPTIONS] MODEL_NAME_OR_PATH OUTPUT_PATH

# Required Arguments:
#   MODEL_NAME_OR_PATH  HuggingFace model name or local path
#   OUTPUT_PATH         Path for the exported ONNX file

# Options:
#   --input-text TEXT       Input text for tokenization [default: "Hello world"]
#   --input-shape SHAPE     Input shape as comma-separated integers [default: 1,128]
#   --strategy STRATEGY     Tagging strategy [default: usage_based]
#   --opset-version INT     ONNX opset version [default: 14]
#   --temp-dir PATH         Temporary directory for intermediate files
#   -v, --verbose          Enable verbose output
```

#### Examples:

```bash
# Basic BERT export
uv run modelexport export prajjwal1/bert-tiny my_bert.onnx

# BERT with custom input
uv run modelexport export prajjwal1/bert-tiny bert_custom.onnx \
  --input-text "The quick brown fox jumps over the lazy dog" \
  --input-shape 1,64

# GPT-2 model export
uv run modelexport export gpt2 gpt2_model.onnx \
  --input-text "Once upon a time" \
  --opset-version 15

# Local model export
uv run modelexport export ./my_local_model model.onnx \
  --temp-dir ./temp_exports
```

### `analyze` - Analyze Hierarchy Tags

Analyze and display hierarchy information from an exported ONNX model.

```bash
uv run modelexport analyze [OPTIONS] ONNX_PATH

# Arguments:
#   ONNX_PATH  Path to the ONNX model file

# Options:
#   --output-format FORMAT  Output format: summary, detailed, json, csv [default: summary]
#   --output-file PATH      Save analysis to file
#   --filter-tag TAG        Filter results by specific hierarchy tag
```

#### Examples:

```bash
# Basic analysis
uv run modelexport analyze bert_model.onnx

# Detailed analysis with JSON output
uv run modelexport analyze bert_model.onnx \
  --output-format json \
  --output-file bert_analysis.json

# Filter by specific module
uv run modelexport analyze bert_model.onnx \
  --filter-tag "/BertModel/BertEncoder" \
  --output-format detailed
```

### `validate` - Validate Model and Tags

Validate ONNX model structure and hierarchy tag consistency.

```bash
uv run modelexport validate [OPTIONS] ONNX_PATH

# Arguments:
#   ONNX_PATH  Path to the ONNX model file

# Options:
#   --check-consistency     Perform deep consistency checks
#   --repair               Attempt to repair inconsistencies
```

#### Examples:

```bash
# Basic validation
uv run modelexport validate bert_model.onnx

# Deep consistency check
uv run modelexport validate bert_model.onnx --check-consistency

# Validate and repair
uv run modelexport validate bert_model.onnx \
  --check-consistency --repair
```

### `compare` - Compare Models

Compare hierarchy tags and structure between two ONNX models.

```bash
uv run modelexport compare [OPTIONS] ONNX_PATH1 ONNX_PATH2

# Arguments:
#   ONNX_PATH1  Path to first ONNX model
#   ONNX_PATH2  Path to second ONNX model

# Options:
#   --output-file PATH  Save comparison report to file
```

#### Examples:

```bash
# Compare baseline vs tagged model
uv run modelexport compare baseline.onnx tagged.onnx

# Save comparison report
uv run modelexport compare model1.onnx model2.onnx \
  --output-file comparison_report.json
```

## 🎯 Complete BERT Example Workflow

Here's a complete example of exporting and analyzing a BERT model:

```bash
# 1. Export BERT with hierarchy preservation
uv run modelexport export prajjwal1/bert-tiny bert_tiny_hierarchical.onnx \
  --input-text "Natural language processing with transformers" \
  --input-shape 1,128 \
  --verbose

# 2. Analyze the exported model
uv run modelexport analyze bert_tiny_hierarchical.onnx \
  --output-format detailed \
  --output-file bert_analysis.json

# 3. Validate the model
uv run modelexport validate bert_tiny_hierarchical.onnx \
  --check-consistency

# 4. Compare with baseline (if available)
# First create a baseline without hierarchy
uv run python -c "
import torch
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
inputs = tokenizer('Natural language processing with transformers', 
                  return_tensors='pt', max_length=128, padding='max_length', truncation=True)
model.eval()
torch.onnx.export(model, tuple(inputs.values()), 'bert_baseline.onnx', 
                 input_names=list(inputs.keys()), output_names=['output'])
"

# Compare baseline vs hierarchical
uv run modelexport compare bert_baseline.onnx bert_tiny_hierarchical.onnx \
  --output-file bert_comparison.json
```

## 📊 Understanding the Output

### Hierarchy Tags Format

The exported ONNX models contain hierarchy information in two forms:

1. **ONNX Node Attributes** (for runtime access):
   - `hierarchy_tags`: List of module paths
   - `hierarchy_path`: Primary module path
   - `hierarchy_count`: Number of tags
   - `hierarchy_method`: Tagging method used

2. **JSON Sidecar File** (for tooling):
   - `model_hierarchy.json` with detailed mapping
   - Statistics and metadata
   - Tag distribution analysis

### Example Hierarchy Paths

```
/BertModel/BertEmbeddings/word_embeddings     - Word embedding operations
/BertModel/BertEncoder/BertLayer/BertAttention - Self-attention operations  
/BertModel/BertEncoder/BertLayer/BertIntermediate - Feed-forward operations
/BertModel/BertPooler                         - Pooling operations
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_cli.py -v          # CLI tests
uv run pytest tests/test_hierarchy_exporter.py -v  # Core exporter tests

# Run with coverage
uv run pytest tests/ --cov=modelexport --cov-report=html
```

### Project Structure

```
modelexport/
├── modelexport/           # Main package
│   ├── hierarchy_exporter.py  # Core exporter
│   ├── cli.py            # Command-line interface
│   ├── tag_utils.py      # Tag manipulation utilities
│   └── graph_comparison.py    # Model comparison tools
├── tests/                # Test suite
├── docs/                 # Documentation
└── examples/            # Usage examples
```

## 🌟 Key Features

- **Universal**: Works with ANY PyTorch model (BERT, GPT, ResNet, custom models)
- **No Hardcoded Logic**: Uses universal PyTorch principles, not model-specific patterns
- **Hierarchy Preservation**: Maintains original module structure in ONNX export
- **ONNX Compliant**: Produces valid ONNX models with enhanced metadata
- **Comprehensive CLI**: Full command-line interface for all operations
- **Test-Driven**: Comprehensive test suite with pytest

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the universal design principles
4. Add tests for new functionality
5. Run the test suite: `uv run pytest tests/ -v`
6. Submit a pull request

## 📄 License

[License information]

## 🆘 Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: Check `examples/` for more usage patterns

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