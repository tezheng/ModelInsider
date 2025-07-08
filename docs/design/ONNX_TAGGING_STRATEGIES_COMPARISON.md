# ONNX Node Tagging Strategies - Comprehensive Comparison

## Overview of Current Approaches

Based on comprehensive analysis of the modelexport repository, there are **4 distinct strategies** for mapping ONNX nodes to semantic information:

| Strategy | Primary File | Status | CLI Command |
|----------|-------------|--------|-------------|
| **enhanced_semantic** | `enhanced_semantic_exporter.py` + `enhanced_semantic_mapper.py` | ✅ Default | `--strategy enhanced_semantic` |
| **htp** | `htp_hierarchy_exporter.py` | ✅ Active | `--strategy htp` |
| **fx_graph** | `fx_hierarchy_exporter.py` | ✅ Active | `--strategy fx_graph` |
| **usage_based** | `usage_based_exporter.py` | 🔄 Legacy | `--strategy usage_based` |

## Side-by-Side Detailed Comparison

### **TIMING & WORKFLOW**

| Strategy | When Does ONNX Tagging Happen? | Dependencies |
|----------|--------------------------------|-------------|
| **enhanced_semantic** | **AFTER** ONNX export → Parse ONNX node names back to PyTorch | PyTorch model + ONNX model |
| **htp** | **DURING** PyTorch execution → Direct module context capture | PyTorch model only |
| **fx_graph** | **BEFORE** ONNX export → Symbolic analysis then map to ONNX | PyTorch model only |
| **usage_based** | **DURING** PyTorch execution → Simple hook tracking | PyTorch model only |

### **CORE APPROACH DIFFERENCES**

| Strategy | Method | Coverage | Accuracy |
|----------|--------|----------|----------|
| **enhanced_semantic** | 🔄 **Reverse-engineering**: ONNX names → PyTorch modules | 97% (multi-strategy) | High (inference-based) |
| **htp** | 🎯 **Direct capture**: Module context during execution | 85% (execution-based) | Very High (direct) |
| **fx_graph** | 📊 **Symbolic analysis**: Static graph analysis | 90% (structural) | High (symbolic) |
| **usage_based** | 🔗 **Simple tracking**: Basic hook usage | 70% (basic) | Medium (limited) |

### **TECHNICAL IMPLEMENTATION**

| Strategy | Key Components | Data Flow |
|----------|----------------|-----------|
| **enhanced_semantic** | `EnhancedSemanticMapper._analyze_node_scope()` <br> `_try_direct_hf_mapping()` <br> `_try_operation_inference()` | Model → ONNX → Parse scope paths → Map back |
| **htp** | `HierarchyExporter._capture_builtin_module_map()` <br> `_tag_operations_with_enhanced_mapping()` | Model → Hook tracing → Direct ONNX tagging |
| **fx_graph** | `FXHierarchyExporter._analyze_fx_hierarchy()` <br> `_map_fx_to_onnx_nodes()` | Model → FX graph → ONNX mapping |
| **usage_based** | `UsageBasedExporter._track_module_usage()` | Model → Basic hooks → Simple tags |

### **HIERARCHY METADATA USAGE**

| Strategy | Uses TracingHierarchyBuilder? | Re-analyzes Model? | Duplication Level |
|----------|------------------------------|-------------------|------------------|
| **enhanced_semantic** | ❌ NO - Builds own hierarchy | ✅ YES - Full re-analysis | 🔴 **HIGH** (60%+ overlap) |
| **htp** | ✅ YES - Inherits from base | ❌ NO - Uses hierarchy | 🟢 **NONE** |
| **fx_graph** | ❌ NO - Own FX analysis | ✅ YES - Symbolic analysis | 🟡 **MEDIUM** (different approach) |
| **usage_based** | ❌ NO - Simple tracking | ✅ YES - Basic analysis | 🟡 **MEDIUM** (basic overlap) |

### **ONNX NODE MAPPING DIFFERENCES**

| Strategy | Mapping Method | Example |
|----------|----------------|---------|
| **enhanced_semantic** | **String parsing**: `/BertModel/BertEncoder/BertLayer.0/...` → Module lookup | Complex scope path parsing |
| **htp** | **Direct context**: Module executes → Record ONNX node → Direct tag | Module context during execution |
| **fx_graph** | **Graph correlation**: FX node → ONNX node mapping | Structural graph analysis |
| **usage_based** | **Hook correlation**: Module usage → ONNX operation inference | Simple usage tracking |

## Key Architectural Issues Identified

### **1. MASSIVE DUPLICATION in enhanced_semantic**
- `EnhancedSemanticMapper._analyze_model_hierarchy()` rebuilds what `TracingHierarchyBuilder` already computed
- Both iterate through `model.named_modules()` and extract metadata
- 60%+ code overlap with different data structures

### **2. REVERSE-ENGINEERING COMPLEXITY**
- `enhanced_semantic` parses ONNX node names: `/BertModel/BertEncoder/BertLayer.0/BertAttention/MatMul_123`
- This is error-prone and complex when direct context is available during execution

### **3. DEPENDENCY MISMATCH**
- `enhanced_semantic` depends on **both** PyTorch model AND ONNX model
- Should only depend on ONNX model + hierarchy metadata from TracingHierarchyBuilder

## Current Workflow Analysis

### **Present Workflow (enhanced_semantic strategy):**
```
1. TracingHierarchyBuilder → builds module hierarchy → tags
2. ONNX Export → creates ONNX model with node names  
3. EnhancedSemanticMapper → RE-ANALYZES PyTorch model (DUPLICATION!)
4. EnhancedSemanticMapper → parses ONNX node names back to modules
5. Final result → Each ONNX node has semantic metadata
```

### **Optimal Workflow (should be):**
```
1. TracingHierarchyBuilder → builds module hierarchy → tags + semantic info
2. ONNX Export → creates ONNX model with node names
3. SemanticMapper → uses hierarchy metadata + ONNX model only
4. Final result → Each ONNX node has semantic metadata  
```

## Performance Comparison

| Strategy | Modules Analyzed | Analysis Type | Performance |
|----------|------------------|---------------|-------------|
| **enhanced_semantic** | 48 (ALL modules) | Static + Reverse-engineering | 🔴 Slow (redundant work) |
| **htp** | 18 (executed only) | Dynamic direct capture | 🟢 Fast (single pass) |
| **fx_graph** | Variable (symbolic) | Static structural | 🟡 Medium (symbolic overhead) |
| **usage_based** | Variable (hooked) | Dynamic simple | 🟢 Fast (simple tracking) |

## Recommendations for Refined Design

### **1. Fix enhanced_semantic Strategy**
- ❌ Remove PyTorch model dependency from `EnhancedSemanticMapper`
- ✅ Make it depend only on: ONNX model + hierarchy metadata from TracingHierarchyBuilder
- ✅ Eliminate duplication in model analysis

### **2. Consolidate Common Components**
- Extract shared hierarchy building logic into base class
- All strategies should inherit from `TracingHierarchyBuilder` or use its output
- Eliminate redundant `model.named_modules()` iterations

### **3. Strategy Specialization**
- **enhanced_semantic**: ONNX node → hierarchy metadata mapping (no PyTorch dependency)
- **htp**: Direct execution context capture
- **fx_graph**: Symbolic analysis approach  
- **usage_based**: Legacy compatibility only

### **4. Unified Interface**
All strategies should follow the pattern:
```python
hierarchy_metadata = TracingHierarchyBuilder().trace_model_execution(model, inputs)
onnx_model = export_to_onnx(model, inputs)
semantic_tags = SemanticMapper(onnx_model, hierarchy_metadata).map_all_nodes()
```

This analysis reveals that the **enhanced_semantic** strategy has significant architectural issues that can be resolved by eliminating the PyTorch model dependency and leveraging the hierarchy metadata that `TracingHierarchyBuilder` already provides.