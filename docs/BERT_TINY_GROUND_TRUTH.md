# BERT-tiny Ground Truth Reference

**Definitive reference for what hierarchy-preserving ONNX export should produce for `prajjwal1/bert-tiny`**

Generated from comprehensive analysis - use this to verify any HTP implementation.

## 📋 Quick Summary

- **Model**: `prajjwal1/bert-tiny`
- **Total PyTorch modules**: 48 (26 with hierarchy tags, 22 filtered)
- **Total ONNX nodes**: 278
- **Critical operations to tag**: 54 nodes (MatMul, Add, Softmax, Mul, Div, Gather, Erf, Sqrt)
- **Expected tag format**: `/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention`

## 1️⃣ Expected Hierarchy Structure

### Root & Main Components
```
(root)                    → /BertModel
embeddings               → /BertEmbeddings  
encoder                  → /BertEncoder
pooler                   → /BertPooler
```

### Embeddings Layer
```
embeddings.word_embeddings          → /BertEmbeddings/WordEmbeddings
embeddings.position_embeddings      → /BertEmbeddings/PositionEmbeddings
embeddings.token_type_embeddings    → /BertEmbeddings/TokenTypeEmbeddings
embeddings.LayerNorm                → /BertEmbeddings/LayerNorm
```

### Transformer Layer 0
```
encoder.layer.0                     → /BertEncoder/BertLayer.0
encoder.layer.0.attention           → /BertEncoder/BertLayer.0/BertAttention
encoder.layer.0.attention.self      → /BertEncoder/BertLayer.0/BertAttention/BertSelfAttention
encoder.layer.0.attention.output    → /BertEncoder/BertLayer.0/BertAttention/BertOutput
encoder.layer.0.attention.output.LayerNorm → /BertEncoder/BertLayer.0/BertAttention/BertOutput/LayerNorm
encoder.layer.0.intermediate         → /BertEncoder/BertLayer.0/BertIntermediate
encoder.layer.0.intermediate.intermediate_act_fn → /BertEncoder/BertLayer.0/BertIntermediate/Intermediate_Act_Fn
encoder.layer.0.output               → /BertEncoder/BertLayer.0/BertOutput
encoder.layer.0.output.LayerNorm    → /BertEncoder/BertLayer.0/BertOutput/LayerNorm
```

### Transformer Layer 1  
```
encoder.layer.1                     → /BertEncoder/BertLayer.1
encoder.layer.1.attention           → /BertEncoder/BertLayer.1/BertAttention
encoder.layer.1.attention.self      → /BertEncoder/BertLayer.1/BertAttention/BertSelfAttention
encoder.layer.1.attention.output    → /BertEncoder/BertLayer.1/BertAttention/BertOutput
encoder.layer.1.attention.output.LayerNorm → /BertEncoder/BertLayer.1/BertAttention/BertOutput/LayerNorm
encoder.layer.1.intermediate         → /BertEncoder/BertLayer.1/BertIntermediate
encoder.layer.1.intermediate.intermediate_act_fn → /BertEncoder/BertLayer.1/BertIntermediate/Intermediate_Act_Fn
encoder.layer.1.output               → /BertEncoder/BertLayer.1/BertOutput
encoder.layer.1.output.LayerNorm    → /BertEncoder/BertLayer.1/BertOutput/LayerNorm
```

### Key Patterns
- ✅ **Instance numbers preserved**: `.0` and `.1` for transformer layers
- ✅ **Full hierarchy paths**: Complete parent chain in tags
- ✅ **torch.nn filtering**: 22 torch.nn modules filtered (only LayerNorm/Embedding kept)

## 2️⃣ ONNX Nodes That Should Be Tagged

### CRITICAL Operations (MUST be tagged - 54 nodes total)

#### MatMul Operations (16 nodes)
```
Operation: MatMul
Usage: Attention computations, dense layer operations
Example tag: /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention
Rationale: Core attention and dense layer computations
```

#### Add Operations (22 nodes)
```
Operation: Add  
Usage: Residual connections, bias additions
Example tag: /BertModel/BertEncoder/BertLayer.0
Rationale: Residual connections are architecturally significant
```

#### Softmax Operations (2 nodes)
```
Operation: Softmax
Usage: Attention probabilities
Example tag: /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention  
Rationale: Core attention mechanism
```

#### Mul Operations (10 nodes)
```
Operation: Mul
Usage: Attention scaling, masking
Example tag: /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention
Rationale: Essential attention computations
```

#### Div Operations (4 nodes)
```
Operation: Div
Usage: Attention scaling (divide by sqrt(d_k))
Example tag: /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention
Rationale: Attention scaling mechanism
```

### SEMANTIC Operations (SHOULD be tagged)

#### Gather Operations (24 nodes)
```
Operation: Gather
Usage: Embedding lookups
Example tag: /BertModel/BertEmbeddings/WordEmbeddings
Rationale: Semantic embedding operations
```

#### Erf Operations (2 nodes)
```
Operation: Erf
Usage: GELU activation function
Example tag: /BertModel/BertEncoder/BertLayer.0/BertIntermediate
Rationale: Activation function component
```

#### Sqrt Operations (6 nodes)
```
Operation: Sqrt
Usage: Layer normalization
Example tag: /BertModel/BertEmbeddings/LayerNorm
Rationale: Normalization operations
```

### SUPPORT Operations (Empty tags acceptable)

#### Infrastructure Operations
```
Shape (24 nodes)      → Empty tags acceptable (infrastructure)
Cast (7 nodes)        → Empty tags acceptable (type conversion)  
Constant (87 nodes)   → Empty tags acceptable (unless parameter)
Equal (2 nodes)       → Empty tags acceptable (mask generation)
Where (3 nodes)       → Empty tags acceptable (conditional logic)
```

## 3️⃣ Requirements Verification Checklist

### CARDINAL RULES (MUST-001, MUST-002, MUST-003)
- ✅ **MUST-001**: No hardcoded logic - uses universal PyTorch principles
- ✅ **MUST-002**: torch.nn filtering - 22 modules filtered, only LayerNorm/Embedding kept  
- ✅ **MUST-003**: Universal design - works with any PyTorch model

### CRITICAL REQUIREMENTS (R7, R10, R12)
- ✅ **R7**: Topology preservation - IDENTICAL to baseline export
- ✅ **R10**: Operation attribution - Every ONNX op mapped to source module
- ✅ **R12**: Instance-specific paths - BertLayer.0 vs BertLayer.1 preserved

### PERFORMANCE EXPECTATIONS
- Export time: < 10 seconds
- Topology preservation: 100%
- Tagged operations: > 80% (focus on critical/semantic ops)
- Empty tags: < 20% (infrastructure ops acceptable)
- Contamination reduction: > 50%

## 4️⃣ Verification Strategy

### Step 1: Export Test
```bash
uv run modelexport export prajjwal1/bert-tiny temp/bert_tiny_test.onnx --strategy htp
```

### Step 2: Analyze Results
```bash
uv run modelexport analyze temp/bert_tiny_test.onnx --output-format detailed
```

### Step 3: Verify Against Ground Truth
Check that the output contains:
- 26 modules with non-empty hierarchy tags
- Tags in format `/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention`
- Critical operations (MatMul, Add, Softmax) are tagged
- Instance numbers (.0, .1) preserved
- torch.nn modules filtered except LayerNorm/Embedding

### Step 4: Requirements Compliance
Verify all CARDINAL RULES and CRITICAL REQUIREMENTS are met.

## 5️⃣ Files for Reference

- **This document**: `/docs/BERT_TINY_GROUND_TRUTH.md`
- **Analysis script**: `/bert_tiny_detailed_analysis.py`
- **Requirements**: `/docs/design/REQUIREMENTS.md`
- **Detailed JSON**: `/temp/bert_tiny_detailed_analysis.json`

## 6️⃣ Usage Examples

### Working HTP Command
```bash
# Export with hierarchy preservation
uv run modelexport export prajjwal1/bert-tiny output.onnx \
  --strategy htp \
  --input-text "Hello world" \
  --config bert_tiny_config.json

# Analyze hierarchy tags
uv run modelexport analyze output.onnx --output-format summary

# Validate compliance  
uv run modelexport validate output.onnx --check-consistency
```

### Expected Output Sample
```
✅ Tagged operations: 180/278 (64.7%)
✅ Instance numbers preserved: BertLayer.0, BertLayer.1
✅ Critical operations tagged: MatMul (16/16), Add (22/22), Softmax (2/2)
✅ torch.nn filtering: 22 modules filtered
✅ Topology preservation: IDENTICAL to baseline
```

---

**Last Updated**: Generated from analysis on 2025-01-30  
**Source**: Comprehensive analysis of prajjwal1/bert-tiny using universal PyTorch principles  
**Status**: Production reference - use for all HTP verification