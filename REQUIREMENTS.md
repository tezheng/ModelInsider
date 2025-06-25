# ModelExport Requirements Specification

## Updated Requirements Summary with Design Revisions

Based on comprehensive analysis including the Design Revisions document, here are the updated requirements:

### 🎯 Core Mission
**Universal Hierarchy-Preserving ONNX Export**: Export ANY PyTorch model to ONNX while:
1. Preserving complete module hierarchy through intelligent tagging
2. **Maintaining IDENTICAL graph topology to baseline export** (R7)
3. Enabling subgraph extraction for any module (R13)
4. Supporting operation-to-module attribution (R10)

### 🚨 CARDINAL RULES

1. **MUST-001: NO HARDCODED LOGIC** ⚠️
   - NEVER hardcode model architectures, node names, operator names
   - No architecture-specific branching or string matching
   - Universal PyTorch principles only

2. **MUST-002: TORCH.NN FILTERING** ⚠️ (Updated per R11)
   - Filter out most torch.nn modules from hierarchy tags
   - **EXCEPTION LIST**: Semantically important modules allowed:
     - LayerNorm, Embedding (default exceptions)
     - BatchNorm variants, GroupNorm, InstanceNorm
   - Configurable exception list for different use cases

3. **MUST-003: UNIVERSAL DESIGN** ⚠️
   - Must work with ANY PyTorch model
   - Architecture-agnostic approach
   - No model-specific assumptions

### 🏗️ Critical Design Requirements

#### R7: Topology Preservation ✅
- **Requirement**: Export must preserve IDENTICAL graph topology to baseline
- **Implementation**: Use standard torch.onnx.export(), then add metadata
- **Validation**: Graph topology comparison tools required

#### R9: Module Information Persistence ✅
- Store module metadata as JSON:
  - forward_args (from inspect.signature)
  - parameters (names only, not tensors)
  - direct_parameters vs inherited
  - children (name→class mapping)
  - expected_hierarchy for validation

#### R10: Operation-to-Module Attribution ✅
- Map every ONNX operation to source HF module class
- Store in model metadata or sidecar JSON
- Enable operation-level debugging and optimization

#### R11: Selective torch.nn Module Inclusion ✅
- Default exclusion of torch.nn infrastructure modules
- Configurable exception list for semantically important modules
- Balance between noise reduction and architectural clarity

#### R12: Instance-Specific Hierarchy Paths ✅
- Preserve instance numbers: BertLayer.0 vs BertLayer.1
- Full parent chain in hierarchical paths
- Example: `/BertModel/BertEncoder/BertLayer.0/BertAttention`

#### R13: Multi-Consumer Tensor Tagging ✅
- Tag tensors with ALL consuming modules, not just producers
- Enable complete subgraph extraction
- Support boundary operation identification
- Implemented and tested with 100% coverage

### 📊 Enhanced Functional Requirements

#### Subgraph Extraction (R13)
```python
extract_module_subgraph(onnx_model, hierarchy_metadata, target_module)
# Returns:
# - operations: List of operation names in subgraph
# - external_inputs: Tensors from outside the module
# - internal_tensors: Tensors produced within module
# - boundary_operations: Operations providing inputs
```

#### CLI Commands
1. **export** - Export model with hierarchy preservation
   - Support HuggingFace models and local paths
   - Input shape/text customization
   - Debug options (--jit-graph, --fx-graph)

2. **analyze** - Analyze hierarchy tags in ONNX model
   - Multiple output formats (summary, detailed, json, csv)
   - Tag filtering capabilities
   - Statistical analysis

3. **validate** - Validate ONNX model and tag consistency
   - ONNX compliance checking
   - Tag format validation
   - Consistency verification

4. **compare** - Compare two ONNX models
   - Tag distribution comparison
   - Structural differences
   - Report generation

### 🧪 Testing Requirements

#### Node Type Analysis
- **Critical Operations**: Must be tagged (math, activations, NN layers)
- **Support Operations**: Context-dependent tagging acceptable
- **Input Preprocessing**: Empty tags acceptable
- 18/304 empty tags in BERT-tiny validated as correct

#### Test Categories (Priority Order)
1. **MUST Tests** - Cardinal rules, zero tolerance for failure
2. **Smoke Tests** - Basic functionality (<30s)
3. **Sanity Tests** - Core assumptions hold
4. **Regression Tests** - Design compliance maintained
5. **Integration Tests** - End-to-end workflows
6. **Performance Tests** - Time/memory bounds

#### Validation Strategy
- Topology preservation validation (identical to baseline)
- Multi-consumer coverage verification
- Subgraph extraction completeness testing

### 🎨 Implementation Status

#### Current Phase Status
- **Phase 1**: HuggingFace module focus (CURRENT)
  - HF models get rich tagging
  - Simple PyTorch models may show 0 tags (expected)
  - torch.nn modules filtered except whitelist
- **Phase 2**: Expanded torch.nn coverage (PLANNED)
- **Phase 3**: Custom module support (FUTURE)

### 🚀 Key Achievements & Validation

1. **Topology Preservation**: ✅ 100% identical to baseline
2. **Multi-Consumer Tagging**: ✅ 100% tensor coverage
3. **Subgraph Extraction**: ✅ Fully functional
4. **Contamination Reduction**: ✅ 72% reduction achieved
5. **Performance**: ✅ 29% improvement with built-in tracking
6. **Production Ready**: ✅ Comprehensive test suite (121 tests)

### 📝 Breaking Changes & Migration

#### Expected Test Updates Needed
- Multi-consumer approach creates more permissive tagging
- Operation counts changed due to efficiency improvements
- Tests should expect richer tagging behavior

#### User Guidance
- Simple PyTorch models: 0 tags expected in Phase 1
- Configure torch.nn exceptions as needed
- Refer to TAGGING_STRATEGY.md for debugging

The system now represents state-of-the-art capability in neural network hierarchy preservation with proven production readiness and significant technical innovations.

## Document History

- **Created**: Based on comprehensive analysis of project documentation
- **Sources**: CLAUDE.md, MEMO.md, DESIGN.md, DESIGN_REVISIONS.md, test case specifications
- **Validation**: Cross-referenced with implementation status and achievement reports
- **Status**: Production-ready requirements specification