# Single-Word Naming Options for node_hierarchy_mapping

## Current: `node_hierarchy_mapping` (too long)

## Single-Word Options:

### 1. **`nodes`** ✓
- Simple and clear
- Makes sense: `metadata["nodes"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Short, intuitive
- Cons: Might be too generic

### 2. **`tags`** ✓
- Describes what it contains (hierarchy tags)
- Makes sense: `metadata["tags"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Accurate, short
- Cons: Could be confused with other tagging systems

### 3. **`mappings`** ✓
- Describes the relationship
- Makes sense: `metadata["mappings"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Clear purpose
- Cons: Plural might not fit convention

### 4. **`hierarchy`** ✓
- Emphasizes the hierarchical nature
- Makes sense: `metadata["hierarchy"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Descriptive of content
- Cons: Might be confused with module hierarchy

### 5. **`tracing`** ✓
- Relates to how it was created (tracing execution)
- Makes sense: `metadata["tracing"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Technical accuracy
- Cons: Already used for tracing info

### 6. **`index`** ✓
- Like an index/lookup table
- Makes sense: `metadata["index"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Database-like clarity
- Cons: Might imply ordering

### 7. **`map`** ✓
- Short for mapping
- Makes sense: `metadata["map"]["onnx_op_123"] = "/Model/Layer1"`
- Pros: Very short, clear
- Cons: Might conflict with Python map()

## Recommendation: `nodes`

Most intuitive and follows common naming patterns where the key describes what type of items are contained.