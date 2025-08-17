# HTP ONNX Export Report

**Generated**: 2025-08-15T08:08:19.129Z

**Model**: prajjwal1/bert-tiny

**Output**: experiments/tez-158_infer-profiling/models/bert.onnx

**Strategy**: HTP (Hierarchical Tracing and Projection)

**Export Time**: 4.97s

## Export Process

### ✅ Step 1/6: Model Preparation

- **Model Class**: BertModel
- **Total Modules**: 48
- **Total Parameters**: 4,385,920 (4.4M)
- **Status**: Model set to evaluation mode

### ✅ Step 2/6: Input Generation

- **Method**: auto_generated
- **Model Type**: bert
- **Detected Task**: feature-extraction
- **Generated Inputs**:

| Input Name     | Shape   | Data Type   |
| :------------- | :------ | :---------- |
| input_ids      | [2, 16] | torch.int64 |
| attention_mask | [2, 16] | torch.int64 |
| token_type_ids | [2, 16] | torch.int64 |

### ✅ Step 3/6: Hierarchy Building

- **Modules Traced**: 18
- **Execution Steps**: 36
- **Status**: Module hierarchy successfully traced

#### Module Hierarchy Preview

<details>

<summary>Click to expand module hierarchy</summary>



```

BertModel
├── BertEmbeddings: embeddings
├── BertEncoder: encoder
│   ├── BertLayer: encoder.layer.0
│   │   ├── BertAttention: encoder.layer.0.attention
│   │   │   ├── BertSelfOutput: encoder.layer.0.attention.output
│   │   │   └── BertSdpaSelfAttention: encoder.layer.0.attention.self
│   │   ├── BertIntermediate: encoder.layer.0.intermediate
│   │   │   └── GELUActivation: encoder.layer.0.intermediate.intermediate_act_fn
│   │   └── BertOutput: encoder.layer.0.output
│   └── BertLayer: encoder.layer.1
│       ├── BertAttention: encoder.layer.1.attention
│       │   ├── BertSelfOutput: encoder.layer.1.attention.output
│       │   └── BertSdpaSelfAttention: encoder.layer.1.attention.self
│       ├── BertIntermediate: encoder.layer.1.intermediate
│       │   └── GELUActivation: encoder.layer.1.intermediate.intermediate_act_fn
│       └── BertOutput: encoder.layer.1.output
└── BertPooler: pooler

```



</details>

### ✅ Step 4/6: ONNX Export

- **Configuration**:

- Opset Version: 17
- Constant Folding: True
- Output Names: ['last_hidden_state', 'pooler_output']

- **Model Size**: 16.76 MB
- **Status**: Successfully exported

### ✅ Step 5/6: Node Tagging

- **Total ONNX Nodes**: 136
- **Tagged Nodes**: 136 (100.0% coverage)
- **Tagging Statistics**:

| Match Type     | Count | Percentage |
| :------------- | :---- | :--------- |
| Direct Matches | 83    | 61.0%      |
| Parent Matches | 34    | 25.0%      |
| Root Fallbacks | 19    | 14.0%      |

#### Top 20 Nodes by Hierarchy



```

  1. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention: 35 nodes

  2. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention: 35 nodes

  3. /BertModel: 19 nodes

  4. /BertModel/BertEmbeddings: 8 nodes

  5. /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation: 8 nodes

  6. /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation: 8 nodes

  7. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput: 4 nodes

  8. /BertModel/BertEncoder/BertLayer.0/BertOutput: 4 nodes

  9. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput: 4 nodes

 10. /BertModel/BertEncoder/BertLayer.1/BertOutput: 4 nodes

 11. /BertModel/BertPooler: 3 nodes

 12. /BertModel/BertEncoder/BertLayer.0/BertIntermediate: 2 nodes

 13. /BertModel/BertEncoder/BertLayer.1/BertIntermediate: 2 nodes

```



#### Complete HF Hierarchy with ONNX Nodes

<details>

<summary>Click to expand complete hierarchy with node counts</summary>



```

BertModel (136 nodes)
├── BertEmbeddings: embeddings (8 nodes)
│   │   ├── Add (2 ops)
│   │   ├── Constant (2 ops)
│   │   ├── Gather (3 ops)
│   │   └── LayerNormalization: /embeddings/LayerNorm/LayerNormalization
├── BertEncoder: encoder (106 nodes)
│   ├── BertLayer: encoder.layer.0 (53 nodes)
│   │   ├── BertAttention: encoder.layer.0.attention (39 nodes)
│   │   │   ├── BertSelfOutput: encoder.layer.0.attention.output (4 nodes)
│   │   │   │   │   ├── Add (2 ops)
│   │   │   │   │   ├── LayerNormalization: /encoder/layer.0/attention/output/LayerNorm/LayerNormalization
│   │   │   │   │   └── MatMul: /encoder/layer.0/attention/output/dense/MatMul
│   │   │   └── BertSdpaSelfAttention: encoder.layer.0.attention.self (35 nodes)
│   │   │           ├── Add (4 ops)
│   │   │           ├── Cast (2 ops)
│   │   │           ├── Constant (7 ops)
│   │   │           ├── Div: /encoder/layer.0/attention/self/Div
│   │   │           ├── MatMul (5 ops)
│   │   │           ├── Mul (2 ops)
│   │   │           ├── Reshape (4 ops)
│   │   │           ├── Shape: /encoder/layer.0/attention/self/Shape
│   │   │           ├── Slice: /encoder/layer.0/attention/self/Slice
│   │   │           ├── Softmax: /encoder/layer.0/attention/self/Softmax
│   │   │           ├── Sqrt (3 ops)
│   │   │           └── Transpose (4 ops)
│   │   ├── BertIntermediate: encoder.layer.0.intermediate (10 nodes)
│   │   │   └── GELUActivation: encoder.layer.0.intermediate.intermediate_act_fn (8 nodes)
│   │   │           ├── Add: /encoder/layer.0/intermediate/intermediate_act_fn/Add
│   │   │           ├── Constant (3 ops)
│   │   │           ├── Div: /encoder/layer.0/intermediate/intermediate_act_fn/Div
│   │   │           ├── Erf: /encoder/layer.0/intermediate/intermediate_act_fn/Erf
│   │   │           └── Mul (2 ops)
│   │   └── BertOutput: encoder.layer.0.output (4 nodes)
│   │           ├── Add (2 ops)
│   │           ├── LayerNormalization: /encoder/layer.0/output/LayerNorm/LayerNormalization
│   │           └── MatMul: /encoder/layer.0/output/dense/MatMul
│   └── BertLayer: encoder.layer.1 (53 nodes)
│       ├── BertAttention: encoder.layer.1.attention (39 nodes)
│       │   ├── BertSelfOutput: encoder.layer.1.attention.output (4 nodes)
│       │   │   │   ├── Add (2 ops)
│       │   │   │   ├── LayerNormalization: /encoder/layer.1/attention/output/LayerNorm/LayerNormalization
│       │   │   │   └── MatMul: /encoder/layer.1/attention/output/dense/MatMul
│       │   └── BertSdpaSelfAttention: encoder.layer.1.attention.self (35 nodes)
│       │           ├── Add (4 ops)
│       │           ├── Cast (2 ops)
│       │           ├── Constant (7 ops)
│       │           ├── Div: /encoder/layer.1/attention/self/Div
│       │           ├── MatMul (5 ops)
│       │           ├── Mul (2 ops)
│       │           ├── Reshape (4 ops)
│       │           ├── Shape: /encoder/layer.1/attention/self/Shape
│       │           ├── Slice: /encoder/layer.1/attention/self/Slice
│       │           ├── Softmax: /encoder/layer.1/attention/self/Softmax
│       │           ├── Sqrt (3 ops)
│       │           └── Transpose (4 ops)
│       ├── BertIntermediate: encoder.layer.1.intermediate (10 nodes)
│       │   └── GELUActivation: encoder.layer.1.intermediate.intermediate_act_fn (8 nodes)
│       │           ├── Add: /encoder/layer.1/intermediate/intermediate_act_fn/Add
│       │           ├── Constant (3 ops)
│       │           ├── Div: /encoder/layer.1/intermediate/intermediate_act_fn/Div
│       │           ├── Erf: /encoder/layer.1/intermediate/intermediate_act_fn/Erf
│       │           └── Mul (2 ops)
│       └── BertOutput: encoder.layer.1.output (4 nodes)
│               ├── Add (2 ops)
│               ├── LayerNormalization: /encoder/layer.1/output/LayerNorm/LayerNormalization
│               └── MatMul: /encoder/layer.1/output/dense/MatMul
└── BertPooler: pooler (3 nodes)
        ├── Gather: /pooler/Gather
        ├── Gemm: /pooler/dense/Gemm
        └── Tanh: /pooler/activation/Tanh

```



</details>

### ✅ Step 6/6: Tag Injection

- **Hierarchy Tags**: Embedded in ONNX
- **Output File**: bert
- **Status**: Export completed successfully

## Module Hierarchy

*Diagram generation disabled for stability.*

### Module List (Sorted by Execution Order)

| Execution Order | Class Name            | Nodes  | Tag                                                                    | Scope                                            |
| :-------------- | :-------------------- | :----- | :--------------------------------------------------------------------- | :----------------------------------------------- |
| 0               | BertModel             | 19/136 | /BertModel                                                             | [ROOT]                                           |
| 1               | BertEmbeddings        | 8/8    | /BertModel/BertEmbeddings                                              | embeddings                                       |
| 2               | BertEncoder           | 0/106  | /BertModel/BertEncoder                                                 | encoder                                          |
| 3               | BertLayer             | 0/53   | /BertModel/BertEncoder/BertLayer.0                                     | encoder.layer.0                                  |
| 4               | BertAttention         | 0/39   | /BertModel/BertEncoder/BertLayer.0/BertAttention                       | encoder.layer.0.attention                        |
| 5               | BertSdpaSelfAttention | 35/35  | /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention | encoder.layer.0.attention.self                   |
| 6               | BertSelfOutput        | 4/4    | /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput        | encoder.layer.0.attention.output                 |
| 7               | BertIntermediate      | 2/10   | /BertModel/BertEncoder/BertLayer.0/BertIntermediate                    | encoder.layer.0.intermediate                     |
| 8               | GELUActivation        | 8/8    | /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation     | encoder.layer.0.intermediate.intermediate_act_fn |
| 9               | BertOutput            | 4/4    | /BertModel/BertEncoder/BertLayer.0/BertOutput                          | encoder.layer.0.output                           |
| 10              | BertLayer             | 0/53   | /BertModel/BertEncoder/BertLayer.1                                     | encoder.layer.1                                  |
| 11              | BertAttention         | 0/39   | /BertModel/BertEncoder/BertLayer.1/BertAttention                       | encoder.layer.1.attention                        |
| 12              | BertSdpaSelfAttention | 35/35  | /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention | encoder.layer.1.attention.self                   |
| 13              | BertSelfOutput        | 4/4    | /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput        | encoder.layer.1.attention.output                 |
| 14              | BertIntermediate      | 2/10   | /BertModel/BertEncoder/BertLayer.1/BertIntermediate                    | encoder.layer.1.intermediate                     |
| 15              | GELUActivation        | 8/8    | /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation     | encoder.layer.1.intermediate.intermediate_act_fn |
| 16              | BertOutput            | 4/4    | /BertModel/BertEncoder/BertLayer.1/BertOutput                          | encoder.layer.1.output                           |
| 17              | BertPooler            | 3/3    | /BertModel/BertPooler                                                  | pooler                                           |

## Complete Node Mappings

<details>

<summary>Click to expand all 136 node mappings</summary>



```

/Cast -> /BertModel

/Cast_1 -> /BertModel

/Cast_2 -> /BertModel

/Constant -> /BertModel

/ConstantOfShape -> /BertModel

/Constant_1 -> /BertModel

/Constant_2 -> /BertModel

/Constant_3 -> /BertModel

/Constant_4 -> /BertModel

/Constant_5 -> /BertModel

/Constant_6 -> /BertModel

/Equal -> /BertModel

/Expand -> /BertModel

/Mul -> /BertModel

/Sub -> /BertModel

/Unsqueeze -> /BertModel

/Unsqueeze_1 -> /BertModel

/Where -> /BertModel

/Where_1 -> /BertModel

/embeddings/Add -> /BertModel/BertEmbeddings

/embeddings/Add_1 -> /BertModel/BertEmbeddings

/embeddings/Constant -> /BertModel/BertEmbeddings

/embeddings/Constant_1 -> /BertModel/BertEmbeddings

/embeddings/LayerNorm/LayerNormalization -> /BertModel/BertEmbeddings

/embeddings/position_embeddings/Gather -> /BertModel/BertEmbeddings

/embeddings/token_type_embeddings/Gather -> /BertModel/BertEmbeddings

/embeddings/word_embeddings/Gather -> /BertModel/BertEmbeddings

/encoder/layer.0/attention/output/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput

/encoder/layer.0/attention/output/LayerNorm/LayerNormalization -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput

/encoder/layer.0/attention/output/dense/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput

/encoder/layer.0/attention/output/dense/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput

/encoder/layer.0/attention/self/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Cast -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Cast_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_2 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_3 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_4 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_5 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Constant_6 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Div -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/MatMul_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Mul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Mul_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Reshape -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Reshape_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Reshape_2 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Reshape_3 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Shape -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Slice -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Softmax -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Sqrt -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Sqrt_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Sqrt_2 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Transpose -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Transpose_1 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Transpose_2 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/Transpose_3 -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/key/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/key/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/query/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/query/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/value/Add -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/attention/self/value/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention

/encoder/layer.0/intermediate/dense/Add -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate

/encoder/layer.0/intermediate/dense/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate

/encoder/layer.0/intermediate/intermediate_act_fn/Add -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Constant -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Constant_1 -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Constant_2 -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Div -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Erf -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Mul -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1 -> /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation

/encoder/layer.0/output/Add -> /BertModel/BertEncoder/BertLayer.0/BertOutput

/encoder/layer.0/output/LayerNorm/LayerNormalization -> /BertModel/BertEncoder/BertLayer.0/BertOutput

/encoder/layer.0/output/dense/Add -> /BertModel/BertEncoder/BertLayer.0/BertOutput

/encoder/layer.0/output/dense/MatMul -> /BertModel/BertEncoder/BertLayer.0/BertOutput

/encoder/layer.1/attention/output/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput

/encoder/layer.1/attention/output/LayerNorm/LayerNormalization -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput

/encoder/layer.1/attention/output/dense/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput

/encoder/layer.1/attention/output/dense/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput

/encoder/layer.1/attention/self/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Cast -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Cast_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_2 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_3 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_4 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_5 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Constant_6 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Div -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/MatMul_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Mul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Mul_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Reshape -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Reshape_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Reshape_2 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Reshape_3 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Shape -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Slice -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Softmax -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Sqrt -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Sqrt_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Sqrt_2 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Transpose -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Transpose_1 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Transpose_2 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/Transpose_3 -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/key/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/key/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/query/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/query/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/value/Add -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/attention/self/value/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention

/encoder/layer.1/intermediate/dense/Add -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate

/encoder/layer.1/intermediate/dense/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate

/encoder/layer.1/intermediate/intermediate_act_fn/Add -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Constant -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Constant_1 -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Constant_2 -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Div -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Erf -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Mul -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/intermediate/intermediate_act_fn/Mul_1 -> /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation

/encoder/layer.1/output/Add -> /BertModel/BertEncoder/BertLayer.1/BertOutput

/encoder/layer.1/output/LayerNorm/LayerNormalization -> /BertModel/BertEncoder/BertLayer.1/BertOutput

/encoder/layer.1/output/dense/Add -> /BertModel/BertEncoder/BertLayer.1/BertOutput

/encoder/layer.1/output/dense/MatMul -> /BertModel/BertEncoder/BertLayer.1/BertOutput

/pooler/Gather -> /BertModel/BertPooler

/pooler/activation/Tanh -> /BertModel/BertPooler

/pooler/dense/Gemm -> /BertModel/BertPooler

```



</details>

## Export Summary

### Performance Metrics

- **Export Time**: 4.97s
- **Module Processing**: ~0.99s
- **ONNX Conversion**: ~2.48s
- **Node Tagging**: ~1.49s

### Coverage Statistics

- **Hierarchy Modules**: 48
- **Traced Modules**: 18/48
- **ONNX Nodes**: 136
- **Tagged Nodes**: 136 (100.0%)
- **Empty Tags**: 0

### Output Files

- **ONNX Model**: `experiments/tez-158_infer-profiling/models/bert` (16.76 MB)
- **Metadata**: `experiments/tez-158_infer-profiling/models/bert_htp_metadata.json`
- **Report**: `experiments/tez-158_infer-profiling/models/bert_htp_export_report.md`

***

*Generated by HTP Exporter v1.0*