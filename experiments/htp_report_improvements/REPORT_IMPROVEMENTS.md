# HTP Report and Metadata Improvements

## Overview

This document describes the improvements made to HTP report generation and metadata structure based on the requirements:

1. Write full console output to txt file without truncation
2. Add complete modules and tagged_nodes to report
3. Restructure metadata tagging field for better organization
4. Compare metadata vs console output and handle differences

## Key Improvements

### 1. Full Text Report Generation

**Problem**: Console output was truncated at 50 lines for tree visualization

**Solution**: 
- Added `text_report_buffer` to capture all output
- Override `_render_tree_output()` to write full tree without truncation
- Generate `*_full_report.txt` with complete output

**Benefits**:
- Complete module hierarchy visible
- All ONNX nodes listed
- No information lost due to truncation

### 2. Complete Data in Report

**Added to Text Report**:
```
========================================
COMPLETE MODULE HIERARCHY
========================================
Module: [module_path]
  Class: [class_name]
  Tag: [hierarchy_tag]
  Execution Order: [number]

========================================
COMPLETE NODE-TO-HIERARCHY MAPPING
========================================
[onnx_node_name] -> [hierarchy_tag]
```

### 3. Metadata Structure Improvements

**OLD Structure**:
```json
{
  "tagging": {
    "tagged_nodes": {...},
    "statistics": {...},
    "coverage": {...}
  }
}
```

**NEW Structure**:
```json
{
  "nodes": {
    // Direct node->tag mapping at root level
    "onnx_node_1": "/Model/Layer1",
    "onnx_node_2": "/Model/Layer2"
  },
  "report": {
    "node_tagging": {
      "statistics": {...},
      "coverage": {...}
    },
    "full_hierarchy": {
      "modules": {...},
      "total_modules": 42,
      "module_types": [...]
    }
  }
}
```

**Changes**:
1. `tagged_nodes` → `nodes` (simpler name, root level)
2. `tagging.statistics` → `report.node_tagging.statistics`
3. `tagging.coverage` → `report.node_tagging.coverage`
4. Added `report.full_hierarchy` with complete module information

### 4. Console vs Metadata Comparison

**Console Output Sections**:
1. Step-by-step progress (1-8)
2. Tree visualization (truncated)
3. Top 20 nodes summary
4. Statistics summary

**Metadata Content**:
1. Complete data (no truncation)
2. Structured JSON format
3. All modules and nodes
4. Machine-readable

**Reconciliation**:
- Text report captures full console output
- Metadata provides structured data
- Both formats complement each other

## Usage

### Using the Improved Exporter

```python
from modelexport.strategies.htp.htp_exporter_improved import ImprovedHTPExporter

exporter = ImprovedHTPExporter(verbose=True)
exporter.export(model, "output.onnx")

# Generated files:
# - output.onnx (model)
# - output_metadata.json (structured data)
# - output_full_report.txt (complete console output)
```

### Output Files

1. **ONNX Model** (`*.onnx`)
   - Model with embedded hierarchy tags

2. **Metadata** (`*_metadata.json`)
   - Structured JSON with improved organization
   - `nodes` at root level (node->tag mappings)
   - Consolidated `report` section

3. **Full Report** (`*_full_report.txt`)
   - Complete console output
   - Full module hierarchy (no truncation)
   - Complete node mappings
   - All statistics and analysis

## Migration Guide

### For Existing Code

If you have code that reads the old metadata structure:

**Old Access**:
```python
tagged_nodes = metadata["tagging"]["tagged_nodes"]
coverage = metadata["tagging"]["coverage"]
```

**New Access**:
```python
node_mapping = metadata["nodes"]
coverage = metadata["report"]["node_tagging"]["coverage"]
```

### Backward Compatibility

The improved exporter maintains the same CLI interface:
```bash
modelexport export model_name output.onnx --strategy htp
```

## Benefits

1. **No Information Loss**: Full hierarchy and node data preserved
2. **Better Organization**: Cleaner metadata structure
3. **Complete Reports**: Text file with all console output
4. **Clear Naming**: `nodes` is simple and intuitive
5. **Consolidated Reporting**: All report data in one section

## Next Steps

1. Replace `HTPExporter` with `ImprovedHTPExporter` in production
2. Update any code that reads metadata to use new structure
3. Use full text reports for debugging and analysis