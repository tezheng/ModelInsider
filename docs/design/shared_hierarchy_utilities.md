# Shared Hierarchy Utilities Design

## Overview

This document describes the shared hierarchy utilities used across the HTP export system to ensure consistent tree representation in both console output and reports.

## Core Components

### 1. `find_immediate_children(parent_path, hierarchy)`

Finds immediate children of a module path, handling compound patterns like `layer.0` correctly.

**Key Logic**:
- Root case: Returns modules without dots in their path
- Non-root case: Returns modules that are direct children, including compound patterns
- Compound pattern detection: `layer.0`, `blocks.1` etc. are treated as immediate children

### 2. `count_nodes_per_tag(tagged_nodes)`

Counts ONNX nodes per hierarchy tag, aggregating counts up the hierarchy tree.

**Usage**:
- Takes dictionary mapping node names to hierarchy tags
- Returns dictionary mapping tags to node counts
- Used for "Complete HF Hierarchy with ONNX Nodes" display

### 3. `build_ascii_tree(hierarchy, max_depth, max_lines, show_counts, node_counts)`

Generates ASCII tree representation for both console and report output.

**Features**:
- Optional depth and line limits for console preview
- Optional node count display with `show_counts=True`
- Consistent format: `ClassName: full.module.path (N nodes)`
- Handles both dict and ModuleInfo objects

**Display Format**:
```
BertModel (136 nodes)
├── BertEmbeddings: embeddings (8 nodes)
├── BertEncoder: encoder (106 nodes)
│   ├── BertLayer: encoder.layer.0 (53 nodes)
│   │   ├── BertAttention: encoder.layer.0.attention (39 nodes)
│   │   │   ├── BertSelfOutput: encoder.layer.0.attention.output (4 nodes)
│   │   │   └── BertSdpaSelfAttention: encoder.layer.0.attention.self (35 nodes)
```

### 4. `build_rich_tree(hierarchy, show_counts, tagged_nodes)`

Generates Rich Tree object for styled console output.

**Features**:
- Uses Rich library for styled output
- Bold class names, dim scope paths
- Optional cyan-colored node counts
- Same hierarchical structure as ASCII tree

## Key Design Decisions

### 1. Unified Tree Building Logic

Both ASCII and Rich trees use the same `find_immediate_children` function to ensure consistent hierarchy traversal.

### 2. Full Scope Display

The scope (part after colon) shows the FULL module path, not just the last segment:
- ✅ `BertLayer: encoder.layer.0`
- ❌ `BertLayer: 0` 
- ❌ `BertLayer: layer.0`

This provides complete context for each module's location in the hierarchy.

### 3. Consistent Node Count Format

All displays use "(N nodes)" format without "ONNX" prefix:
- ✅ `BertModel (136 nodes)`
- ❌ `BertModel (136 ONNX nodes)`

### 4. Shared Counting Logic

The `count_nodes_per_tag` function provides consistent node counting across all displays, aggregating counts up the hierarchy tree.

## Usage Examples

### Console Module Hierarchy (without counts):
```python
tree = build_rich_tree(hierarchy, show_counts=False)
console.print(tree)
```

### Console Complete Hierarchy (with counts):
```python
tree = build_rich_tree(hierarchy, show_counts=True, tagged_nodes=tagged_nodes)
console.print(tree)
```

### Report Module Hierarchy Preview:
```python
tree_lines = build_ascii_tree(hierarchy)
# Wrap in <details> for collapsible section
```

### Report Complete Hierarchy with Counts:
```python
node_counts = count_nodes_per_tag(tagged_nodes)
tree_lines = build_ascii_tree(hierarchy, show_counts=True, node_counts=node_counts)
```

## Benefits

1. **Consistency**: Same tree structure and format across all outputs
2. **Maintainability**: Single implementation to maintain
3. **Flexibility**: Supports both simple and detailed views
4. **Extensibility**: Easy to add new features to all displays at once