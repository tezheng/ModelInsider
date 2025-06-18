# Universal Hierarchy-Preserving ONNX Export

## Key Insight

All Hugging Face models are just PyTorch `nn.Module` instances with inherent hierarchy. We don't need model-specific implementations - we can leverage this universal structure.

## The Universal Approach

### 1. **No Model-Specific Logic**
- NO keyword matching for transformer types (BERT, GPT, T5, etc.)
- NO hardcoded architecture assumptions  
- NO special cases for different model families

### 2. **Leverage nn.Module Hierarchy**
```python
# Every HF model has this structure automatically:
for name, module in model.named_modules():
    # name: "encoder.layer.0.attention.self" 
    # module: actual PyTorch module
    # This hierarchy is ALREADY THERE
```

### 3. **Three Core Components**

#### A. **Module Hierarchy Analysis**
- Extract the complete `nn.Module` tree structure
- Track depth, parameters, children for each module
- Works for ANY PyTorch model

#### B. **Execution Tracing with Hooks**
- Register forward hooks on all modules
- Map ONNX operations back to their source modules
- Universal - hooks work on any `nn.Module`

#### C. **ONNX Function Preservation**
- Use ONNX metadata to preserve hierarchy info
- Add module attributes to ONNX nodes
- Create ONNX functions for meaningful submodules

### 4. **Why This Works Universally**

Every transformer model (BERT, GPT, T5, etc.) is built from:
```
Model
├── embeddings (nn.Module)
├── encoder/decoder (nn.Module)
│   ├── layer.0 (nn.Module)
│   │   ├── attention (nn.Module)
│   │   └── feed_forward (nn.Module)
│   └── layer.1 (nn.Module)
└── pooler/head (nn.Module)
```

The **names differ**, but the **structure pattern** is universal.

## Implementation Strategy

```python
class UniversalHierarchyExporter:
    def analyze_model_structure(self, model):
        # Works for ANY nn.Module
        for name, module in model.named_modules():
            self.hierarchy[name] = {
                'depth': len(name.split('.')),
                'children': list(module.children()),
                'parameters': module.parameters()
            }
    
    def trace_execution_with_hooks(self, model, inputs):
        # Register hooks on ALL modules
        for name, module in model.named_modules():
            module.register_forward_hook(self.hook_fn)
        
        # Run forward pass - hooks capture execution
        model(inputs)
    
    def export_with_hierarchy(self, model, inputs, output_path):
        # Standard ONNX export + hierarchy metadata
        torch.onnx.export(model, inputs, output_path)
        self.add_hierarchy_metadata(output_path)
```

## The Beauty of This Approach

1. **Zero Model-Specific Code**: Works with any `nn.Module`
2. **Future-Proof**: New architectures work automatically
3. **Simple**: Leverages what PyTorch already provides
4. **Universal**: Vision models, language models, multimodal - all work

## What We DON'T Need

❌ Architecture detection  
❌ Transformer-specific logic  
❌ Attention pattern matching  
❌ Model family classifications  
❌ Hardcoded layer naming conventions  

## What We DO Need

✅ `nn.Module` hierarchy traversal  
✅ Forward hooks for execution tracing  
✅ ONNX metadata for hierarchy preservation  
✅ Parameter-to-module mapping  

---

**Bottom Line**: Stop trying to be smart about model types. Just work with the `nn.Module` hierarchy that's already there.

---

# DAG Extraction Requirements

## Goal
Extract DAG (nodes + edges) for each nn.Module in the hierarchy, preserving both operations and connections. Generate JSON for each piece of the testing model (BERT tiny), then merge into one file.

## Test Model
- **Target**: `google/bert_uncased_L-2_H-128_A-2` (BERT tiny)
- **Output**: One merged JSON file with hierarchy labels as keys

## Hierarchy Naming Convention
- Use HF transformer class names
- Use '/' as depth separator
- Start with '/' 
- Example: `/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention`

## DAG JSON Format
Simple format with nodes and edges:
```json
{
  "/BertModel/BertEmbeddings": {
    "nodes": ["op1", "op2", "op3"],
    "edges": [["op1", "op2"], ["op2", "op3"]]
  }
}
```

## Operation Tagging Strategy

### CRITICAL TAGGING RULES
1. **Single Tag Only**: Each operation gets exactly ONE tag - the most specific transformers class module
2. **No Parent Path Tags**: Do NOT tag operations with parent module paths 
3. **Transformers Classes Only**: Only tag with transformers library classes (BertSdpaSelfAttention, BertAttention, etc.)
4. **No torch.nn Classes**: Do NOT tag with torch.nn classes (Linear, LayerNorm, Dropout, etc.)

### Examples
**CORRECT**: 
```json
{
  "/encoder/layer.0/attention/self/Mul_1": {
    "op_type": "Mul",
    "tags": ["/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSdpaSelfAttention"]
  }
}
```

**INCORRECT** (multiple tags, includes torch.nn class, includes parents):
```json
{
  "/encoder/layer.0/attention/self/Mul_1": {
    "op_type": "Mul", 
    "tags": [
      "/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSdpaSelfAttention/Linear",
      "/BertModel",
      "/BertModel/BertEncoder/ModuleList.0/BertAttention", 
      "/BertModel/BertEncoder/ModuleList.0",
      "/BertModel/BertEncoder"
    ]
  }
}
```

### Tag Selection Algorithm
1. Find all modules in execution path
2. Filter to only transformers library classes (exclude torch.nn classes)
3. Select the most specific (deepest) transformers class
4. Use only that ONE tag

### Examples

**Single tag (operation executes in one module):**
```json
{
  "MatMul_QK": {
    "op_type": "MatMul",
    "tags": ["/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"]
  }
}
```

**Multiple tags (shared parameter/operation):**
```json
{
  "word_embeddings.weight": {
    "op_type": "Initializer", 
    "tags": [
      "/BertModel/BertEmbeddings",
      "/BertModel/BertPredictionHead"
    ]
  }
}
```

**Shared input across layers:**
```json
{
  "Mul_attention_mask": {
    "op_type": "Mul",
    "tags": [
      "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention",
      "/BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfAttention"
    ]
  }
}
```

## What Gets Tagged

### Operations
- **Parameter-based ops**: Use model weights/biases (MatMul, Conv, etc.)
- **Execution-based ops**: Generated during module's forward() (Add, Reshape, etc.)
- **Control flow ops**: Execute during forward() (If, Loop, Scan) ✅
- **Initializers**: Used during forward() execution ✅

### Exclusions
- **Input/Output ops**: Model inputs, intermediate inputs/outputs between modules ❌
- **Constant nodes**: Constants should remain embedded, not extracted as separate nodes ❌

### Constant Handling Strategy
**IMPORTANT**: We distinguish between two types of constants:
1. **Embedded constants**: Scalar values, small tensors that should remain as ONNX initializers/embedded values
2. **Genuine Constant operations**: Operations that were already Constant nodes in the original computation graph

**Implementation**: 
- **DAG Extraction**: Skip tagging `Constant` node types to prevent ONNX export from extracting embedded constants into separate nodes
- **Subgraph Extraction**: When extracting subgraphs, include required Constant nodes as internal nodes (not external inputs) to preserve the same structure as standalone models
- **Result**: Extracted models have identical constant handling as standalone models (constants as embedded values/initializers, not external inputs)

## Critical Issues Found During Testing

### Issue 1: Missing Parent Module Tags
**Problem**: After removing hardcoded attention logic, operations only get tagged to immediate Linear sub-modules, not parent attention modules.
- **Expected**: `/BertModel/.../BertSdpaSelfAttention` (78 operations)
- **Actual**: `/BertModel/.../BertSdpaSelfAttention/Linear` (29 operations)

**Universal Solution**: Implement parameter-based parent module inference:
- If multiple operations use parameters from the same parent module during execution
- Create parent module tags that encompass all child module operations
- Use execution tracing to determine which operations belong to the same logical unit

### Issue 2: Incomplete Module Extraction
**Problem**: Extracting by Linear sub-module tags misses attention computation operations
**Solution**: Target parent module tags for complete functionality extraction

### Issue 3: Layer Mismatch  
**Problem**: Test targets layer 0 but extraction finds layer 1 first
**Solution**: Implement explicit layer targeting in extraction logic

## Key Insights
- **Operations** usually get single tag (where they execute)
- **Initializers** get multiple tags (shared parameters used by multiple modules)
- **Shared parameters** are common in transformers (embeddings, masks, tied weights)
- **Parameter sharing** happens at data level, not computation level
- **Each operation is unique**, even if using shared parameters

## Operation Metadata Format

Beyond simple DAG extraction, generate detailed operation metadata including inputs, outputs, and tags:

```json
{
  "word_embeddings.weight": {
    "op_type": "Initializer",
    "tags": [
      "/BertModel/BertEmbeddings",
      "/BertModel/BertPredictionHead"
    ]
  },
  "MatMul_embedding_lookup": {
    "op_type": "MatMul",
    "inputs": ["input_ids", "word_embeddings.weight"],
    "outputs": ["embedding_output"],
    "tags": ["/BertModel/BertEmbeddings"]
  },
  "MatMul_prediction_head": {
    "op_type": "MatMul", 
    "inputs": ["hidden_states", "word_embeddings.weight"],
    "outputs": ["logits"],
    "tags": ["/BertModel/BertPredictionHead"]
  }
}
```

This metadata captures:
- **op_type**: ONNX operation type
- **inputs**: Input tensor names (shows parameter sharing)
- **outputs**: Output tensor names (for connection tracking)  
- **tags**: Module hierarchy paths that use this operation

## Benefits of Operation Metadata
- **Parameter sharing analysis**: See which modules share the same parameters
- **Data flow tracking**: Follow tensor connections between operations
- **Module interaction**: Understand how modules connect via shared tensors
- **Debugging**: Detailed view of ONNX graph structure with module context

## CRITICAL REQUIREMENT: Tags Must Be In ONNX Model

**SUPER IMPORTANT**: Tags must be injected as ONNX node attributes, not just external metadata!

### The Problem
- Standard PyTorch ONNX export doesn't preserve our hierarchy tags
- Tags currently only exist in separate JSON files
- ONNX model nodes show `Tags: None` - this defeats the whole purpose!

### The Solution
**Post-export ONNX Enhancement**: After standard export, modify ONNX model to inject tags:

```python
# After torch.onnx.export()
onnx_model = onnx.load(output_path)

# Add tags as node attributes
for node in onnx_model.graph.node:
    if node.name in operation_metadata:
        tags = operation_metadata[node.name]['tags']
        if tags:
            # Add source_module attribute
            module_attr = AttributeProto()
            module_attr.name = "source_module"
            module_attr.type = AttributeProto.STRING
            module_attr.s = tags[0].encode('utf-8')  # Primary tag
            node.attribute.append(module_attr)
            
            # Add all tags if multiple
            if len(tags) > 1:
                tags_attr = AttributeProto()
                tags_attr.name = "hierarchy_tags" 
                tags_attr.type = AttributeProto.STRINGS
                tags_attr.strings = [tag.encode('utf-8') for tag in tags]
                node.attribute.append(tags_attr)

onnx.save(onnx_model, enhanced_output_path)
```

### Expected Result
Each ONNX node should show:
```
Node: /encoder/layer.0/attention/self/query/Transpose
  Op Type: Transpose
  Source Module: /BertModel/BertEncoder/ModuleList.0/BertAttention/BertSdpaSelfAttention/Linear
  Tags: ['/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSdpaSelfAttention/Linear']
```

**This is ESSENTIAL** - without tags in the ONNX model, hierarchy preservation is meaningless!

## Current Test Plan

### Test Requirements
1. **Target Model**: `google/bert_uncased_L-2_H-128_A-2` (BERT tiny)

2. **Generate JSON for each piece**: Extract DAG for **all nn.Module** in the hierarchy

3. **Preserve both op and connection (DAG)**:
   - **Nodes**: All operations belonging to each module
   - **Edges**: Connections between operations

4. **Include only internal ops**: Skip input/output ops, only ops that belong to the nn.Module

5. **Simple JSON format**:
   ```json
   {
     "nodes": ["op1", "op2", "op3"],
     "edges": [["op1", "op2"], ["op2", "op3"]]
   }
   ```

6. **Hierarchy naming**:
   - Use HF transformer class names
   - Use '/' as depth separator
   - Start with '/' (e.g., `/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention`)

7. **Merge all JSONs**: One final file with hierarchy labels as keys

8. **Operation metadata**: Include detailed ONNX ops info like:
   ```json
   {
     "word_embeddings.weight": {
       "op_type": "Initializer",
       "tags": ["/BertModel/BertEmbeddings", "/BertModel/BertPredictionHead"]
     },
     "MatMul_embedding_lookup": {
       "op_type": "MatMul",
       "inputs": ["input_ids", "word_embeddings.weight"],
       "tags": ["/BertModel/BertEmbeddings"]
     }
   }
   ```

9. **CRITICAL**: Tags must be in the ONNX model as node attributes, not just external metadata!

## Implementation Strategy
1. Hook all `nn.Module.forward()` methods for execution tracing
2. Map ONNX parameter names to PyTorch modules for parameter tagging  
3. Extract detailed operation metadata (type, inputs, outputs, tags)
4. **CRITICAL**: Inject tags as ONNX node attributes after export
5. Extract DAG for each module (only ops belonging to that module)
6. Use simple JSON format: nodes array + edges array
7. Generate comprehensive operation metadata for analysis
8. Merge all module DAGs into single file with hierarchy paths as keys