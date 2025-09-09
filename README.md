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
# Export a ResNet model with complete features (recommended) - produces clean ONNX by default
uv run modelexport export --model microsoft/resnet-50 --output ./temp/resnet_50/model.onnx --with-report --with-graphml --verbose

# Analyze the exported model
uv run modelexport analyze bert_model.onnx

# Validate hierarchy tags
uv run modelexport validate bert_model.onnx --check-consistency
```

## 📋 CLI Commands

### `export` - Export Model with Hierarchy

Export a PyTorch model to ONNX with hierarchy-preserving tags.

```bash
uv run modelexport export [OPTIONS]

# Primary Options:
#   --model, -m TEXT               HuggingFace model name or local path (REQUIRED)
#   --output, -o PATH              Path for the exported ONNX file (REQUIRED)
#   
# Export Strategy Options:
#   --strategy [htp]               Export strategy (default: htp, currently only HTP supported)
#   --embed-hierarchy              Embed hierarchy metadata in ONNX nodes
#                                  (adds traceability but increases file size)
#   --torch-module                 Include torch.nn modules in hierarchy 
#                                  (e.g., LayerNorm, Embedding for models like ResNet)
#
# Configuration Options:
#   --input-specs PATH             JSON file with input specifications 
#                                  (optional, auto-generates if not provided)
#   --export-config PATH           ONNX export configuration file (JSON)
#                                  Controls: opset_version, do_constant_folding, etc.
#
# Output Format Options:
#   --with-report                  Generate detailed HTP export report
#                                  (creates _htp_export_report.txt file)
#   --with-graphml, --graphml      Export hierarchical GraphML v1.3 alongside ONNX
#                                  Creates: _hierarchical_graph.graphml + .onnxdata
#                                  Features: bidirectional conversion, schema validation
#
# Debugging Options:
#   --verbose, -v                  Enable verbose output with detailed progress
```

#### Examples:

```bash
# Recommended: Export with all features (clean ONNX by default)
uv run modelexport export --model microsoft/resnet-50 --output ./temp/resnet_50/model.onnx --with-report --with-graphml --verbose

# Basic BERT export
uv run modelexport export --model prajjwal1/bert-tiny --output bert_model.onnx

# Export with hierarchy metadata embedded in ONNX
uv run modelexport export --model prajjwal1/bert-tiny --output bert_with_hierarchy.onnx --embed-hierarchy --with-report

# Export clean ONNX (default behavior, no embedded metadata)
uv run modelexport export --model microsoft/resnet-50 --output clean_resnet.onnx
```

### `analyze` - Analyze Hierarchy Tags

Analyze and display hierarchy information from an exported ONNX model.

```bash
uv run modelexport analyze ONNX_PATH [OPTIONS]

# Options:
#   --output-format [json|csv|summary]   Output format for analysis (default: json)
#   --output-file PATH                   Save analysis to file (for json/csv formats)
#   --filter-tag TEXT                    Filter tags containing this string
#   --verbose, -v                        Enable verbose output
```

#### Examples:

```bash
# Basic analysis with summary output
uv run modelexport analyze bert.onnx --output-format summary

# Export analysis to JSON file
uv run modelexport analyze bert.onnx --output-format json --output-file analysis.json

# Filter specific tags
uv run modelexport analyze bert.onnx --filter-tag "Attention" --output-format summary
```

### `validate` - Validate Model and Tags

Validate ONNX model structure and hierarchy tag consistency.

```bash
uv run modelexport validate ONNX_PATH [OPTIONS]

# Options:
#   --check-consistency              Check for tag consistency issues
#   --repair                        Attempt to repair tag inconsistencies
#   --verbose, -v                   Enable verbose output
```

#### Examples:

```bash
# Basic validation
uv run modelexport validate bert.onnx

# Check tag consistency
uv run modelexport validate bert.onnx --check-consistency

# Validate and attempt repairs
uv run modelexport validate bert.onnx --check-consistency --repair --verbose
```

### `compare` - Compare Two Models

Compare hierarchy tags between two ONNX models to identify differences.

```bash
uv run modelexport compare MODEL1_PATH MODEL2_PATH [OPTIONS]

# Options:
#   --output-file PATH              Save comparison results to file
#   --verbose, -v                   Enable verbose output
```

#### Examples:

```bash
# Basic comparison
uv run modelexport compare model1.onnx model2.onnx

# Save comparison to file
uv run modelexport compare model1.onnx model2.onnx --output-file comparison.txt

# Verbose comparison
uv run modelexport compare model1.onnx model2.onnx --verbose
```

## 📊 Understanding the Output

### Hierarchy Tags

The exported ONNX models contain hierarchy information as node attributes:
- `hierarchy_tags`: List of module paths from the original PyTorch model
- `hierarchy_path`: Primary module path for each operation

### Example Hierarchy Paths

The hierarchy tags preserve the original PyTorch module structure. Here are examples from different model architectures:

#### BERT Model Hierarchy
```
/BertModel/BertEmbeddings/                           → Word embedding operations
/BertModel/BertEncoder/BertLayer/BertAttention       → Self-attention operations  
/BertModel/BertEncoder/BertLayer/BertIntermediate    → Feed-forward operations
/BertModel/BertPooler                                → Pooling operations
```

#### ResNet Model Hierarchy
```
/ResNet/Conv1                                        → Initial convolution layer
/ResNet/Layer1/BasicBlock/Conv1                      → First residual block convolutions
/ResNet/Layer2/BasicBlock/Conv2                      → Second layer convolutions
/ResNet/AvgPool                                      → Global average pooling
/ResNet/FC                                           → Final classification layer
```

#### GPT Model Hierarchy
```
/GPTModel/Embeddings/TokenEmbedding                  → Token embeddings
/GPTModel/Transformer/Block/Attention                → Multi-head attention
/GPTModel/Transformer/Block/MLP                      → Feed-forward network
/GPTModel/LMHead                                     → Language modeling head
```

## 🌟 Key Features

- **Universal**: Works with ANY PyTorch model (BERT, GPT, ResNet, custom models)
- **Hierarchy Preservation**: Maintains original module structure in ONNX export
- **Clean Export**: Option to export without hierarchy metadata for cleaner ONNX files
- **ONNX Compliant**: Produces valid ONNX models with optional enhanced metadata
- **Comprehensive CLI**: Simple command-line interface for model export

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v
```

## 📄 License

[License information]