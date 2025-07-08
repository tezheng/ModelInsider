# HTP Debugger

The HTP Debugger is a comprehensive tool for analyzing and debugging the HTP (Hierarchical Trace-and-Project) export process. It provides detailed insights into module hierarchy building, ONNX node tagging, and coverage analysis.

## Features

- **Module Hierarchy Analysis**: Shows optimized hierarchy building with execution tracing
- **ONNX Node Tagging**: Detailed analysis of node tagging with 4-priority system
- **Coverage Statistics**: Comprehensive coverage analysis and accuracy breakdown
- **Scope Bucketization**: Visual representation of how nodes are grouped by scope
- **Tag Distribution**: Analysis of tag distribution across ONNX nodes
- **CARDINAL RULES Verification**: Ensures compliance with all design principles
- **Debug Output**: Saves detailed debugging information to JSON files

## Usage

### Basic Usage
```bash
# Debug default BERT-tiny model
python debug/htp_debugger.py

# Debug specific model
python debug/htp_debugger.py --model distilbert-base-uncased

# Enable operation fallback for better coverage analysis
python debug/htp_debugger.py --enable-operation-fallback

# Save debug outputs to files
python debug/htp_debugger.py --save-outputs

# Custom output directory
python debug/htp_debugger.py --output temp/debug/my_debug --save-outputs
```

### Command Line Options

- `--model`: Model to debug (default: `prajjwal1/bert-tiny`)
- `--enable-operation-fallback`: Enable operation-based fallback in tagging
- `--save-outputs`: Save debug outputs to JSON files
- `--output`: Output directory for debug files (default: `temp/debug/htp_debugger`)

## Supported Models

The debugger automatically detects model types and prepares appropriate inputs:

- **BERT-like models**: `bert`, `roberta` (uses input_ids + attention_mask)
- **GPT-like models**: `gpt`, `dialogen` (uses input_ids only)
- **Vision models**: `resnet`, `vit` (uses image tensors)
- **Generic models**: Automatic fallback with tokenizer detection

## Output Analysis

### 1. Module Hierarchy Tree
Shows the optimized module hierarchy built by TracingHierarchyBuilder:
```
📊 Module Hierarchy Tree (up to depth 3):
├── <root>
├── embeddings
├── encoder
│   └── layer
│       ├── 0
│       └── 1
└── pooler
```

### 2. Tagging Statistics
Provides detailed breakdown of tagging accuracy:
- **Direct matches**: Nodes matched directly to hierarchy modules
- **Parent matches**: Nodes matched to parent scopes
- **Root fallbacks**: Nodes that fall back to model root
- **Operation matches**: Nodes matched via operation-based fallback (if enabled)

### 3. Scope Bucketization
Shows how ONNX nodes are grouped by their scope names:
```
📂 Found 33 unique scopes:
   1. __root__: 29 nodes
   2. encoder.layer.0.attention.self: 29 nodes
   3. encoder.layer.1.attention.self: 29 nodes
   ...
```

### 4. Tag Distribution
Analyzes the distribution of hierarchy tags:
```
🏷️ Unique tags: 13
Top 10 most common tags:
   1. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention: 35 nodes (24.1%)
   2. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention: 35 nodes (24.1%)
   ...
```

## Debug Output Files

When `--save-outputs` is enabled, the following files are created:

- `debug_model.onnx`: The exported ONNX model
- `hierarchy_data.json`: Complete module hierarchy data
- `tagged_nodes.json`: Node name to tag mappings
- `scope_buckets.json`: Scope-based node groupings
- `statistics.json`: Comprehensive statistics and metrics

## Examples

### Debug BERT-tiny with full output
```bash
python debug/htp_debugger.py --save-outputs
```

### Debug DistilBERT with operation fallback
```bash
python debug/htp_debugger.py --model distilbert-base-uncased --enable-operation-fallback --save-outputs
```

### Debug with custom output directory
```bash
python debug/htp_debugger.py --model microsoft/DialoGPT-small --output temp/debug/gpt_analysis --save-outputs
```

## Performance Insights

The debugger provides insights into HTP optimization:
- **Optimization ratio**: Shows executed modules vs total modules
- **Coverage percentage**: 100% coverage with NO EMPTY TAGS
- **Tagging accuracy**: Breakdown of how nodes were tagged
- **Processing efficiency**: Module count reduction and execution tracing

## CARDINAL RULES Verification

Every debug session verifies compliance with core design principles:
- **MUST-001**: NO HARDCODED LOGIC - Model root dynamically extracted
- **MUST-002**: NO EMPTY TAGS - Guaranteed zero empty tags
- **MUST-003**: UNIVERSAL DESIGN - Works with any model architecture

## Troubleshooting

### Model Loading Issues
- Ensure the model name is correct for HuggingFace models
- Check internet connection for downloading models
- Verify sufficient disk space for model downloads

### Input Preparation Issues
- The debugger automatically handles most model types
- For custom models, it falls back to generic tokenizer detection
- Check that the model supports the prepared input format

### Memory Issues
- Large models may require significant memory
- Consider using smaller models for debugging
- Ensure sufficient RAM for both model and ONNX export

## Integration with HTP Exporter

The debugger uses the same components as the production HTP exporter:
- `TracingHierarchyBuilder` for hierarchy building
- `ONNXNodeTagger` for node tagging
- Same CARDINAL RULES compliance verification

This ensures debugging results directly correspond to production behavior.