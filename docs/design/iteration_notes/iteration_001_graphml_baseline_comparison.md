# Iteration 001: GraphML Baseline Comparison

## Date: 2025-07-28

## Achievements
1. Fixed key ID mismatch - changed d0-d3 to n0-n3 for nodes per baseline
2. Fixed node ID format - now using forward slashes instead of underscore prefix
3. Added JSON storage for node attributes (n2 key)
4. Updated ADR-010 specification with MUST fields
5. Fixed all test cases to match new specifications (96 tests passing)
6. Generated bert-tiny GraphML and compared with baseline

## Key Findings

### Structural Differences from Baseline
1. **Graph ID and Root Structure**:
   - Our output: `<graph id="G">` with `<node id="module_root">`
   - Baseline: `<graph id="BertModel">` without extra root node
   
2. **Compound Node Count**:
   - Our output: 19 compound nodes (20 graphs total)
   - Baseline: 44 compound nodes (45 graphs total)
   - Missing 25 compound nodes!

3. **Module Coverage**:
   - HTP traced: 18 modules
   - Model total: 48 modules  
   - Baseline creates compound nodes for ALL nn.Modules including:
     - Individual Embedding modules (word_embeddings, token_type_embeddings, position_embeddings)
     - LayerNorm modules
     - Linear modules within attention/output layers

4. **Node ID Format**:
   - Our output: `module_BertModel`, `module_BertModel_BertEmbeddings`
   - Baseline: `embeddings`, `encoder.layer.0`, etc. (direct names)

5. **Hierarchy Tag Format**:
   - Our output: Simplified tags like "BertModel"
   - Baseline: Full paths like "/BertModel/BertEmbeddings"

## Root Cause Analysis
The fundamental issue is that HTP's tracing approach only captures modules that are directly executed, missing many PyTorch nn.Module components that exist in the model structure but may be called indirectly or through PyTorch internals. The baseline appears to walk the entire module tree using `model.named_modules()` rather than relying on execution tracing.

## Next Steps
1. Enhance HTP to capture ALL PyTorch modules, not just traced ones
2. Fix hierarchical converter to match baseline structure:
   - Remove "module_root" wrapper
   - Use model name as main graph ID
   - Use direct names for node IDs without "module_" prefix
3. Ensure all 48 modules get compound nodes in the output

## Mistakes to Avoid
- Don't rely solely on execution tracing for module discovery
- Don't add extra wrapper nodes that aren't in the baseline
- Don't sanitize node IDs when baseline preserves forward slashes

## Updated Todo List
- [x] Fix key ID mismatch 
- [x] Fix node ID format
- [ ] Enhance hierarchy extraction to capture all PyTorch modules (44 compound nodes)
- [x] Add JSON storage for node attributes
- [x] Update ADR-010 specification
- [x] Fix test cases
- [x] Generate bert-tiny GraphML and compare with baseline
- [ ] Fix hierarchical converter to match baseline structure
- [ ] Document Phase 1 technical planning
- [ ] Generate pytest coverage reports
- [ ] Complete 10 review iterations