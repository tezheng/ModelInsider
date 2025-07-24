# HTP Report Format Requirements

## Overview
The HTP export report should maintain consistency with the console output format, using the same headings, structure, and presentation style.

## Specific Requirements

### 1. Module Hierarchy Preview (Step 3)
**Current Issue**: Uses truncation with max_depth=3, max_lines=20
**Required Format**:
- Wrap the full hierarchy in `<details>` tags instead of truncating
- Show complete hierarchy without depth or line limits
- Use collapsible section for better readability

**Example**:
```markdown
#### Module Hierarchy Preview

<details>
<summary>Click to expand module hierarchy</summary>

```
BertModel
├── BertEmbeddings: embeddings
│   ├── LayerNorm: LayerNorm
│   ├── Dropout: dropout
│   ├── Embedding: position_embeddings
│   ├── Embedding: token_type_embeddings
│   └── Embedding: word_embeddings
└── BertEncoder: encoder
    ├── BertLayer: 0
    │   ├── BertAttention: attention
    │   │   ├── BertSelfOutput: output
    │   │   │   ├── LayerNorm: LayerNorm
    │   │   │   ├── Linear: dense
    │   │   │   └── Dropout: dropout
    │   │   └── BertSdpaSelfAttention: self
    │   │       ├── Linear: key
    │   │       ├── Linear: query
    │   │       └── Linear: value
    │   └── ... (complete hierarchy)
```

</details>
```

### 2. Node Distribution Preview (Step 5)
**Current Issue**: Shows only top 10 with title "Node Distribution Preview"
**Required Format**:
- Use title "Top 20 Nodes by Hierarchy" (matching console)
- Show top 20 items instead of 10
- Format as numbered list (1-20)

**Example**:
```markdown
#### Top 20 Nodes by Hierarchy

```
 1. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention: 29 nodes
 2. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention: 29 nodes
 3. /BertModel: 19 nodes
 ...
20. /BertModel/BertEmbeddings/Dropout: 1 nodes
```

### 3. Complete Module Hierarchy (After Step 6)
**Current Issue**: Title is "Complete Module Hierarchy"
**Required Format**:
- Use title "Complete HF Hierarchy with ONNX Nodes" (matching console)
- Show the hierarchy tree with node counts
- No truncation - show complete hierarchy

**Example**:
```markdown
### Complete HF Hierarchy with ONNX Nodes

<details>
<summary>Click to expand complete hierarchy with node counts</summary>

```
BertModel (136 nodes)
├── BertEmbeddings: embeddings (4 nodes)
│   ├── LayerNorm: LayerNorm (1 nodes)
│   ├── Dropout: dropout (0 nodes)
│   ├── Embedding: position_embeddings (1 nodes)
│   ├── Embedding: token_type_embeddings (1 nodes)
│   └── Embedding: word_embeddings (1 nodes)
└── BertEncoder: encoder (0 nodes)
    ├── BertLayer: 0 (0 nodes)
    │   ├── BertAttention: attention (0 nodes)
    │   │   ├── BertSelfOutput: output (0 nodes)
    │   │   │   ├── LayerNorm: LayerNorm (1 nodes)
    │   │   │   ├── Linear: dense (2 nodes)
    │   │   │   └── Dropout: dropout (0 nodes)
    │   │   └── BertSdpaSelfAttention: self (29 nodes)
    │   │       ├── Linear: key (2 nodes)
    │   │       ├── Linear: query (2 nodes)
    │   │       └── Linear: value (2 nodes)
    └── ... (complete hierarchy with counts)
```

</details>
```

## Implementation Notes

1. **Consistency**: The report should mirror the console output as closely as possible
2. **No Truncation**: Unlike console (which truncates for readability), reports should show complete information
3. **Collapsible Sections**: Use `<details>` tags for long content to keep the report manageable
4. **Exact Titles**: Use the exact same titles as console output for consistency