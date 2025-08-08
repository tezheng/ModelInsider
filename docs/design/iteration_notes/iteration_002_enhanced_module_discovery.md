# Iteration 002: Enhanced Module Discovery Analysis

## Date: 2025-07-28

## Achievements
1. Created EnhancedHierarchicalConverter (v2) to better match baseline structure
2. Analyzed the module discovery gap between HTP and baseline
3. Identified that baseline includes ALL torch.nn modules, not just traced ones

## Key Findings

### Module Discovery Gap
HTP only traces 18 modules during execution, but baseline includes all 48 modules:

**HTP Captures (18 modules)**:
- BertModel
- BertEmbeddings
- BertEncoder
- BertLayer (x2)
- BertAttention (x2)
- BertSdpaSelfAttention (x2)
- BertSelfOutput (x2)
- BertIntermediate (x2)
- GELUActivation (x2)
- BertOutput (x2)
- BertPooler

**Baseline Additionally Includes**:
- embeddings.word_embeddings (Embedding)
- embeddings.token_type_embeddings (Embedding)
- embeddings.position_embeddings (Embedding)
- embeddings.LayerNorm (LayerNorm)
- embeddings.dropout (Dropout)
- encoder.layer.0.attention.self.query (Linear)
- encoder.layer.0.attention.self.key (Linear)
- encoder.layer.0.attention.self.value (Linear)
- encoder.layer.0.attention.self.dropout (Dropout)
- encoder.layer.0.attention.output.dense (Linear)
- encoder.layer.0.attention.output.LayerNorm (LayerNorm)
- encoder.layer.0.attention.output.dropout (Dropout)
- encoder.layer.0.intermediate.dense (Linear)
- encoder.layer.0.output.dense (Linear)
- encoder.layer.0.output.LayerNorm (LayerNorm)
- encoder.layer.0.output.dropout (Dropout)
- (Similar for layer.1)
- pooler.dense (Linear)
- pooler.activation (Tanh)

### Root Cause
The baseline appears to use `model.named_modules()` to walk the entire module tree, creating compound nodes for every `nn.Module`, including:
- Basic PyTorch layers (Linear, LayerNorm, Embedding, Dropout)
- Activation functions when they're modules (Tanh)
- All nested modules regardless of execution

HTP's execution tracing only captures "composite" modules that have forward() methods called directly during execution. Many basic layers are called through PyTorch internals and aren't captured.

## Proposed Solution
Enhance HTP to include ALL modules:
1. After tracing, walk the model with `model.named_modules()`
2. Add any missing modules to the hierarchy
3. Mark traced vs non-traced modules differently
4. Ensure proper parent-child relationships

## Mistakes Made
1. Assumed execution tracing would capture all modules
2. Didn't realize baseline includes low-level PyTorch modules
3. Created extra wrapper nodes not in baseline

## Next Steps
1. Enhance HTP module discovery to use named_modules()
2. Update hierarchical converter to:
   - Remove module_ prefix from node IDs
   - Use model class name as main graph ID
   - Include all PyTorch modules as compound nodes
3. Test with bert-tiny to match 44 compound nodes

## Updated Todo List
- [x] Fix key ID mismatch
- [x] Fix node ID format  
- [ ] Enhance hierarchy extraction to capture all PyTorch modules (44 compound nodes)
- [x] Add JSON storage for node attributes
- [x] Update ADR-010 specification
- [x] Fix test cases
- [x] Generate bert-tiny GraphML and compare with baseline
- [ ] Fix hierarchical converter to match baseline structure
- [ ] Enhance HTP to discover all nn.Modules
- [ ] Document Phase 1 technical planning
- [ ] Generate pytest coverage reports
- [ ] Complete 10 review iterations (Currently on iteration 2/10)