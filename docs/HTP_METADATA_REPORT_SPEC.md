# HTP Metadata and Report Specification

## Overview

The HTP (Hierarchy-preserving Tags Protocol) export process generates two output files in addition to the ONNX model:
1. **Metadata (JSON)**: Structured data about the export process, model hierarchy, and node mappings
2. **Report (TXT)**: Human-readable report with console output and complete mappings

## Metadata Structure

### Schema Location
The complete JSON schema is available at: `modelexport/strategies/htp_new/htp_metadata_schema.json`

### Top-Level Structure
```json
{
  "export_context": { ... },  // Export session metadata
  "model": { ... },           // Model information
  "tracing": { ... },         // Tracing execution info
  "modules": { ... },         // Module hierarchy
  "nodes": { ... },           // ONNX node to tag mappings (root level)
  "outputs": { ... },         // Output file information
  "report": { ... },          // Export process report
  "statistics": { ... }       // Export summary statistics
}
```

### Detailed Data Structures

#### 1. export_context
Export session metadata (from console output Step 1 & export timing):
```json
{
  "timestamp": "2025-07-21T12:01:35Z",    // ISO format timestamp
  "strategy": "htp",                       // Always "htp"
  "version": "1.0",                        // Schema version
  "exporter": "HTPExporter",               // Exporter class name
  "embed_hierarchy_attributes": true,      // Whether tags embedded in ONNX
  "export_time_seconds": 4.79             // Total export time from console
}
```

#### 2. model
Model information (from console output Step 1):
```json
{
  "name_or_path": "prajjwal1/bert-tiny",  // HuggingFace model ID or path
  "class": "BertModel",                    // Model class name
  "framework": "transformers",             // Framework used
  "total_modules": 48,                     // Total modules (console: 48 modules)
  "total_parameters": 4385920              // Total params (console: 4.4M parameters)
}
```

#### 3. tracing
Tracing execution information (from console output Step 3):
```json
{
  "builder": "TracingHierarchyBuilder",
  "modules_traced": 18,                    // Console: Traced 18 modules
  "execution_steps": 36,                   // Console: Total execution steps: 36
  "model_type": "bert",                    // From Step 2 auto-detection
  "task": "feature-extraction",            // From Step 2 auto-detection
  "inputs": {                              // From Step 2 generated inputs
    "input_ids": { "shape": [2, 16], "dtype": "torch.int64" },
    "attention_mask": { "shape": [2, 16], "dtype": "torch.int64" },
    "token_type_ids": { "shape": [2, 16], "dtype": "torch.int64" }
  }
}
```

#### 4. modules
Complete module hierarchy as a nested structure (mirrors actual module hierarchy):

**Key Design Principles:**
- **Hierarchical Structure**: The JSON structure mirrors the actual PyTorch module hierarchy
- **Module Keys**: Use HuggingFace module class names as keys (e.g., "BertEmbeddings", "BertLayer.0")
- **Indexed Modules**: Repeated modules use class name with index (e.g., "BertLayer.0", "BertLayer.1")
- **Scope Field**: Contains the full module path from root (e.g., "encoder.layer.0.attention")

```json
{
  "class_name": "BertModel",
  "traced_tag": "/BertModel",
  "scope": "",                             // Empty string for root module
  "execution_order": 0,
  "children": {
    "BertEmbeddings": {                    // Using module class name as key
      "class_name": "BertEmbeddings",
      "traced_tag": "/BertModel/BertEmbeddings",
      "scope": "embeddings",               // Full module path from root
      "execution_order": 1
    },
    "BertEncoder": {
      "class_name": "BertEncoder",
      "traced_tag": "/BertModel/BertEncoder",
      "scope": "encoder",
      "execution_order": 2,
      "children": {
        "BertLayer.0": {                   // Using indexed name for repeated modules
          "class_name": "BertLayer",
          "traced_tag": "/BertModel/BertEncoder/BertLayer.0",
          "scope": "encoder.layer.0",      // Full path from root
          "execution_order": 3,
          "children": {
            "BertAttention": {
              "class_name": "BertAttention",
              "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention",
              "scope": "encoder.layer.0.attention",  // Full path
              "execution_order": 4,
              "children": {
                "BertSdpaSelfAttention": {
                  "class_name": "BertSdpaSelfAttention",
                  "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
                  "scope": "encoder.layer.0.attention.self",  // Full path
                  "execution_order": 5
                },
                "BertSelfOutput": {
                  "class_name": "BertSelfOutput",
                  "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
                  "scope": "encoder.layer.0.attention.output",  // Full path
                  "execution_order": 6
                }
              }
            },
            "BertIntermediate": {
              "class_name": "BertIntermediate",
              "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
              "scope": "encoder.layer.0.intermediate",  // Full path
              "execution_order": 7
            },
            "BertOutput": {
              "class_name": "BertOutput",
              "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
              "scope": "encoder.layer.0.output",  // Full path
              "execution_order": 8
            }
          }
        },
        "BertLayer.1": {                   // Second layer with index
          "class_name": "BertLayer",
          "traced_tag": "/BertModel/BertEncoder/BertLayer.1",
          "scope": "encoder.layer.1",      // Full path from root
          "execution_order": 10,
          "children": {
            // ... similar structure for layer 1
          }
        }
      }
    },
    "BertPooler": {
      "class_name": "BertPooler",
      "traced_tag": "/BertModel/BertPooler",
      "scope": "pooler",                  // Full path from root
      "execution_order": 17
    }
  }
}
```

#### 5. nodes
ONNX node to hierarchy tag mappings (from console output Step 5):
```json
{
  "/embeddings/Constant": "/BertModel/BertEmbeddings",
  "/embeddings/Add": "/BertModel/BertEmbeddings",
  "/embeddings/LayerNorm/LayerNormalization": "/BertModel/BertEmbeddings",
  "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
  // ... all 136 node mappings
}
```

#### 6. outputs
Output file information (from console output final summary):
```json
{
  "onnx_model": {
    "path": "bert-tiny.onnx",             // Just filename, not full path
    "size_mb": 16.76,                      // From console: Model size: 16.76MB
    "opset_version": 17                    // From console: Opset version: 17
  },
  "metadata": {
    "path": "bert-tiny_htp_metadata.json"
  },
  "report": {
    "path": "bert-tiny_htp_export_report.txt"  // Optional, if report enabled
  }
}
```

#### 7. report
Export process details (from console output steps):
```json
{
  "steps": {
    "input_generation": {                  // From Step 2
      "method": "auto_generated",
      "model_type": "bert",
      "task": "feature-extraction",
      "inputs": {
        "input_ids": { "shape": [2, 16], "dtype": "torch.int64" },
        "attention_mask": { "shape": [2, 16], "dtype": "torch.int64" },
        "token_type_ids": { "shape": [2, 16], "dtype": "torch.int64" }
      }
    },
    "onnx_export": {                       // From Step 4
      "opset_version": 17,
      "do_constant_folding": true,
      "onnx_size_mb": 16.76
    },
    "node_tagging": {                      // From Step 5 - MERGED INTO STEPS
      "completed": true,
      "timestamp": "2025-07-22T02:52:28Z",
      "total_nodes": 136,
      "tagged_nodes_count": 136,
      "coverage_percentage": 100.0,
      "statistics": {
        "root_nodes": 19,                  // Nodes at root level
        "scoped_nodes": 117,               // Nodes with specific scope
        "unique_scopes": 32,               // Number of unique scopes
        "direct_matches": 83,              // Direct module matches
        "parent_matches": 34,              // Parent fallback matches
        "operation_matches": 0,            // Operation-based matches
        "root_fallbacks": 19               // Root fallback matches
      },
      "coverage": {
        "percentage": 100.0,
        "total_onnx_nodes": 136,
        "tagged_nodes": 136
      }
    },
    "tag_injection": {                     // From Step 6
      "tags_injected": true,
      "tags_stripped": false
    }
  }
}
```

#### 8. statistics
Export summary statistics (from console output final summary):
```json
{
  "export_time": 4.79,                     // Console: Total time: 4.79s
  "hierarchy_modules": 48,                 // Console: Hierarchy modules: 48
  "traced_modules": 18,                    // Console: Traced modules: 18/48
  "onnx_nodes": 136,                       // Console: ONNX nodes: 136
  "tagged_nodes": 136,                     // Console: Tagged nodes: 136
  "empty_tags": 0,                         // Always 0 with HTP
  "coverage_percentage": 100.0,            // Console: 100.0% coverage
  "module_types": [                        // Unique module class names
    "BertAttention", "BertEmbeddings", "BertEncoder",
    "BertIntermediate", "BertLayer", "BertModel",
    "BertOutput", "BertPooler", "BertSdpaSelfAttention",
    "BertSelfOutput", "GELUActivation"
  ]
}
```

## Report Structure

### Design Principles

1. **Progressive Disclosure**: Show summary first, details in collapsible sections
2. **Information Hierarchy**: Summary → Analysis → Process Details → Raw Data
3. **Consistency**: Match console output formatting where applicable
4. **Visual Clarity**: Use consistent formatting and clear section boundaries

### Format
The report is generated as a **Markdown file (.md)** with structured sections using headers, tables, and code blocks for better readability and tool compatibility.

**Important**: The report is generated independently from console output, ensuring consistent structure regardless of the `--verbose` flag.

### File Naming
- Report: `{output_name}_htp_export_report.md`
- Metadata: `{output_name}_htp_metadata.json`

### Markdown Report Template (Current Implementation)
```markdown
# HTP ONNX Export Report

**Generated**: {ISO_TIMESTAMP}
**Model**: {MODEL_NAME}
**Output**: {OUTPUT_PATH}
**Strategy**: HTP (Hierarchical Tracing and Projection)
**Export Time**: {TOTAL_TIME}s

## Export Process

### ✅ Step 1/6: Model Preparation
- **Model Class**: {MODEL_CLASS}
- **Total Modules**: {TOTAL_MODULES}
- **Total Parameters**: {TOTAL_PARAMETERS} ({HUMAN_READABLE_PARAMS})
- **Status**: Model set to evaluation mode

### ✅ Step 2/6: Input Generation
- **Method**: {INPUT_METHOD}
- **Model Type**: {MODEL_TYPE}
- **Detected Task**: {TASK}
- **Generated Inputs**:

| Input Name | Shape | Data Type |
| :--------- | :---- | :-------- |
| {INPUT_NAME} | {SHAPE} | {DTYPE} |
...

### ✅ Step 3/6: Hierarchy Building
- **Modules Traced**: {MODULE_COUNT}
- **Execution Steps**: {EXECUTION_STEPS}
- **Status**: Module hierarchy successfully traced

#### Module Hierarchy Preview

<details>
<summary>Click to expand module hierarchy</summary>

```
{ASCII_TREE_WITHOUT_COUNTS}
```

</details>

### ✅ Step 4/6: ONNX Export
- **Configuration**:
- Opset Version: {OPSET_VERSION}
- Constant Folding: {CONSTANT_FOLDING}
- Output Names: {OUTPUT_NAMES}
- **Model Size**: {MODEL_SIZE_MB} MB
- **Status**: Successfully exported

### ✅ Step 5/6: Node Tagging
- **Total ONNX Nodes**: {TOTAL_NODES}
- **Tagged Nodes**: {TAGGED_NODES} ({COVERAGE}% coverage)
- **Tagging Statistics**:

| Match Type | Count | Percentage |
| :--------- | :---- | :--------- |
| Direct Matches | {DIRECT} | {DIRECT_PCT}% |
| Parent Matches | {PARENT} | {PARENT_PCT}% |
| Root Fallbacks | {ROOT} | {ROOT_PCT}% |
| Empty Tags | {EMPTY} | {EMPTY_PCT}% |

#### Complete HF Hierarchy with ONNX Nodes

<details>
<summary>Click to expand complete hierarchy with node counts</summary>

```
{ASCII_TREE_WITH_NODE_COUNTS}
```

</details>

### ✅ Step 6/6: Tag Injection
- **Hierarchy Tags**: {EMBED_STATUS}
- **Output File**: {OUTPUT_FILE}
- **Status**: Export completed successfully

## Module Hierarchy

*Mermaid diagram disabled for stability.*

### Module List (Sorted by Execution Order)

| Execution Order | Class Name | Nodes | Tag | Scope |
| :-------------- | :--------- | :---- | :-- | :---- |
| {ORDER} | {CLASS} | {DIRECT}/{TOTAL} | {TAG} | {SCOPE} |
...

## Complete Node Mappings

<details>
<summary>Click to expand all {NODE_COUNT} node mappings</summary>

```
{NODE_NAME} -> {HIERARCHY_TAG}
...
```

</details>

## Export Summary

### Performance Metrics
- **Export Time**: {EXPORT_TIME}s
- **Module Processing**: ~{MODULE_TIME}s
- **ONNX Conversion**: ~{ONNX_TIME}s
- **Node Tagging**: ~{TAGGING_TIME}s

### Coverage Statistics
- **Hierarchy Modules**: {HIERARCHY_MODULES}
- **Traced Modules**: {TRACED_MODULES}/{HIERARCHY_MODULES}
- **ONNX Nodes**: {ONNX_NODES}  
- **Tagged Nodes**: {TAGGED_NODES} ({COVERAGE}%)
- **Empty Tags**: {EMPTY_TAGS}

### Output Files
- **ONNX Model**: `{ONNX_FILE}` ({SIZE} MB)
- **Metadata**: `{METADATA_FILE}`
- **Report**: `{REPORT_FILE}`

---
*Generated by HTP Exporter v1.0*
```

### Key Implementation Details

1. **Hierarchy Display**:
   - Module scope shows FULL path (e.g., `BertLayer: encoder.layer.0`)
   - Node counts use "X nodes" format (not "X ONNX nodes")
   - Complete hierarchy with node counts moved to Node Tagging section

2. **Shared Utilities**:
   - `build_ascii_tree()` generates both console and report trees
   - `count_nodes_per_tag()` provides consistent node counting
   - No duplication between console and report generation

3. **Table Formatting**:
   - Left-aligned columns using `:---` syntax
   - Module List includes node counts in format: `{direct}/{total}`
     - `direct` = nodes directly in this module (not in children)
     - `total` = all nodes in this module and its children
   - Example: `8/8` for leaf modules, `19/136` for BertModel root

4. **Removed Sections**:
   - "Top 20 Nodes by Hierarchy" (duplicated Module Node Distribution)
   - "Node Distribution" section (information consolidated elsewhere)
   - "Top Operations by Count" (not implemented)

5. **Collapsible Sections**:
   - Module hierarchy preview (in Step 3)
   - Complete hierarchy with node counts (in Step 5)
   - Complete node mappings

### Report Organization Flow

1. **Header**: Title and metadata
2. **Export Process**: Step-by-step details with node tagging results
3. **Module Hierarchy**: Detailed module table
4. **Complete Mappings**: Raw node-to-tag data
5. **Export Summary**: Performance and coverage metrics

### Key Improvements Over Plain Text Format

1. **Independent Generation**: Report is generated directly from ExportData, not dependent on console output capture
2. **Consistent Structure**: Same report structure regardless of `--verbose` flag setting
3. **Better Readability**: 
   - Markdown formatting with headers, tables, and lists
   - Collapsible sections for long content
   - Proper syntax highlighting in code blocks
4. **Visual Hierarchy**: Module hierarchy representation
5. **Professional Appearance**: Can be rendered in GitHub, VS Code, documentation tools
6. **Export Flexibility**: Can be converted to HTML, PDF if needed

### Report Sections Explained

#### 1. Header
- Timestamp in ISO format
- Model name/path  
- Output file path
- Strategy name and export time

#### 2. Export Process
Detailed information for each of the 6 export steps:
- Model Preparation
- Input Generation (with input table)
- Hierarchy Building
- ONNX Export (with configuration details)
- Node Tagging (with statistics table)
- Tag Injection

#### 3. Module Hierarchy
- Visual hierarchy representation
- Complete module list in collapsible table
- Shows all modules without truncation

#### 4. Node Distribution
- Top operations by count
- Module-wise node distribution
- Helps understand model composition

#### 5. Complete Node Mappings
- All ONNX node to hierarchy tag mappings
- Collapsible section to manage length
- Preserves complete information

#### 6. Export Summary
- Performance metrics breakdown
- Coverage statistics
- Output file information

## Key Design Principles

### 1. No Truncation in Files
While console output may truncate for readability, both metadata and report files contain complete information.

### 2. Structured Metadata
The JSON metadata follows a strict schema for programmatic access and validation.

### 3. Human-Readable Report
The text report is designed for human consumption with clear sections and formatting.

### 4. Nodes at Root Level
In the metadata, the `nodes` mapping is at the root level (not nested under tagging) for easier access.

### 5. Zero Empty Tags
The system guarantees that `empty_tags` is always 0 - every ONNX node gets a valid hierarchy tag.

### 6. Report Independence
The markdown report is generated independently from console output, ensuring consistency regardless of verbose settings.

## Test Cases

### Metadata Tests Should Verify:
1. Schema compliance using JSON schema validation
2. Required fields are present
3. Numeric constraints (e.g., coverage 0-100%, empty_tags = 0)
4. String patterns (e.g., strategy = "htp", version format)
5. Consistency between sections (e.g., node count matches)

### Report Tests Should Verify:
1. Markdown file has `.md` extension
2. All required sections are present
3. No truncation in module hierarchy or node mappings
4. Report is generated regardless of verbose flag
5. Proper markdown formatting (headers, tables, code blocks)
6. Report structure is valid
7. Summary statistics match metadata

### Example Test Assertions:
```python
# Metadata tests
assert metadata["export_context"]["strategy"] == "htp"
assert metadata["report"]["node_tagging"]["statistics"]["empty_tags"] == 0
assert 0 <= metadata["report"]["node_tagging"]["coverage"]["percentage"] <= 100

# Report tests
assert report_path.endswith("_htp_export_report.md")
assert "# HTP ONNX Export Report" in report
assert "## Export Process" in report
assert "## Module Hierarchy" in report
assert all_modules_in_report(report, metadata["modules"])
assert no_truncation_marker_in_sections(report)

# Test report generation without verbose
exporter = HTPExporter(verbose=False, enable_reporting=True)
# Report should still be generated with complete content
```

## Implementation Notes

### Dependencies
The markdown report generation requires:
- `snakemd` - For programmatic markdown generation
- Standard library modules (no additional dependencies for basic implementation)

### Tree Visualization Options
1. **Current**: Module hierarchy table with execution order
2. **Future Options**: 
   - ASCII tree using `anytree` or `treelib` in code blocks
   - Rich export to SVG (optional enhancement)

### Missing Steps Fix
To ensure all 6 steps appear in metadata report section:
1. Update MetadataWriter step handlers to consistently record step data
2. Call base class `_write_default()` or manually add to `_steps_data`
3. Include: model_preparation, hierarchy_building in report.steps