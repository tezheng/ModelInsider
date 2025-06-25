# Iteration 2: Core FX Hierarchy Exporter Implementation

**Goal**: Implement main FX-based exporter with all cardinal rules and key requirements
**Implementation**: Created `fx_hierarchy_exporter.py` with complete FX workflow

## Key Features Implemented
- ✅ FX symbolic tracing with `torch.fx.symbolic_trace()`
- ✅ CARDINAL RULE #1: No hardcoded logic - universal module filtering
- ✅ CARDINAL RULE #2: torch.nn filtering with semantic exceptions
- ✅ CARDINAL RULE #3: Universal design for any PyTorch model
- ✅ R7: Topology preservation via standard torch.onnx.export()
- ✅ R10: Direct operation-to-module attribution via FX node.target
- ✅ R12: Instance-specific hierarchy paths (BertLayer.0 vs BertLayer.1)
- ✅ R9: Module metadata extraction with forward_args, parameters, children
- ✅ R13: Subgraph extraction framework

## Technical Approach
- Phase 1: FX graph analysis and hierarchy extraction
- Phase 2: Standard ONNX export (topology preservation)
- Phase 3: FX→ONNX mapping and hierarchy injection via doc_string
- Phase 4: Analysis file generation (sidecar JSON, module info)

## FX→ONNX Mapping Strategy
- Operation type correspondence (call_module_Linear → Gemm/MatMul)
- Execution order alignment for matching
- Store hierarchy in ONNX doc_string field (compliant)

## Next Steps
1. Integrate with CLI system
2. Create test scripts to validate functionality
3. Handle edge cases and error scenarios