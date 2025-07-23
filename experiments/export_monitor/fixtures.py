"""
Test fixtures for ExportMonitor based on real bert-tiny export data.
"""

from export_monitor import ExportData, ExportStep


def create_bert_tiny_fixture() -> ExportData:
    """Create ExportData fixture based on real bert-tiny export."""
    data = ExportData(
        # Model info
        model_name="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4385920,
        
        # Export settings
        output_path="bert-tiny.onnx",
        strategy="htp",
        
        # Files
        onnx_size_mb=16.76,
        metadata_path="bert-tiny_htp_metadata.json",
        report_path="bert-tiny_full_report.txt",
    )
    
    # Add hierarchy data (subset of actual data)
    data.hierarchy = {
        "": {
            "name": "",
            "class_name": "BertModel",
            "module_type": "huggingface",
            "traced_tag": "/BertModel",
            "execution_order": 0
        },
        "embeddings": {
            "name": "embeddings",
            "class_name": "BertEmbeddings",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEmbeddings",
            "execution_order": 1
        },
        "embeddings.word_embeddings": {
            "name": "embeddings.word_embeddings",
            "class_name": "Embedding",
            "module_type": "torch.nn",
            "traced_tag": "/BertModel/BertEmbeddings/Embedding",
            "execution_order": 10
        },
        "embeddings.position_embeddings": {
            "name": "embeddings.position_embeddings",
            "class_name": "Embedding",
            "module_type": "torch.nn",
            "traced_tag": "/BertModel/BertEmbeddings/Embedding.1",
            "execution_order": 11
        },
        "embeddings.token_type_embeddings": {
            "name": "embeddings.token_type_embeddings",
            "class_name": "Embedding",
            "module_type": "torch.nn",
            "traced_tag": "/BertModel/BertEmbeddings/Embedding.2",
            "execution_order": 12
        },
        "embeddings.LayerNorm": {
            "name": "embeddings.LayerNorm",
            "class_name": "LayerNorm",
            "module_type": "torch.nn",
            "traced_tag": "/BertModel/BertEmbeddings/LayerNorm",
            "execution_order": 13
        },
        "embeddings.dropout": {
            "name": "embeddings.dropout",
            "class_name": "Dropout",
            "module_type": "torch.nn",
            "traced_tag": "/BertModel/BertEmbeddings/Dropout",
            "execution_order": 14
        },
        "encoder": {
            "name": "encoder",
            "class_name": "BertEncoder",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder",
            "execution_order": 2
        },
        "encoder.layer.0": {
            "name": "encoder.layer.0",
            "class_name": "BertLayer",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0",
            "execution_order": 3
        },
        "encoder.layer.0.attention": {
            "name": "encoder.layer.0.attention",
            "class_name": "BertAttention",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention",
            "execution_order": 4
        },
        "encoder.layer.0.attention.self": {
            "name": "encoder.layer.0.attention.self",
            "class_name": "BertSdpaSelfAttention",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
            "execution_order": 5
        },
        "encoder.layer.0.attention.output": {
            "name": "encoder.layer.0.attention.output",
            "class_name": "BertSelfOutput",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
            "execution_order": 6
        },
        "encoder.layer.0.intermediate": {
            "name": "encoder.layer.0.intermediate",
            "class_name": "BertIntermediate",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
            "execution_order": 7
        },
        "encoder.layer.0.output": {
            "name": "encoder.layer.0.output",
            "class_name": "BertOutput",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
            "execution_order": 8
        },
        "encoder.layer.1": {
            "name": "encoder.layer.1",
            "class_name": "BertLayer",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertEncoder/BertLayer.1",
            "execution_order": 15
        },
        "pooler": {
            "name": "pooler",
            "class_name": "BertPooler",
            "module_type": "huggingface",
            "traced_tag": "/BertModel/BertPooler",
            "execution_order": 9
        }
    }
    
    # Add node tagging data (sample of actual nodes)
    data.total_nodes = 136
    data.tagged_nodes = {
        "/embeddings/Gather": "/BertModel/BertEmbeddings/Embedding",
        "/embeddings/Gather_1": "/BertModel/BertEmbeddings/Embedding.1",
        "/embeddings/Gather_2": "/BertModel/BertEmbeddings/Embedding.2",
        "/embeddings/Add": "/BertModel/BertEmbeddings",
        "/embeddings/Add_1": "/BertModel/BertEmbeddings",
        "/embeddings/ReduceMean": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Sub": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Pow": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/ReduceMean_1": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Add_2": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Sqrt": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Div": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Mul": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Add_3": "/BertModel/BertEmbeddings/LayerNorm",
        "/embeddings/Transpose": "/BertModel/BertEmbeddings/Dropout",
        "/encoder/layer.0/attention/self/Transpose": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Reshape": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Transpose_1": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Transpose_2": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Transpose_3": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/MatMul_1": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Div": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/MatMul_2": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Transpose_4": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/self/Reshape_1": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/encoder/layer.0/attention/output/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
        "/encoder/layer.0/attention/output/Add": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
        "/encoder/layer.0/attention/output/Add_1": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
        "/encoder/layer.0/intermediate/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Add": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Div": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Erf": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Add_1": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Mul": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/intermediate/Mul_1": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
        "/encoder/layer.0/output/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
        "/encoder/layer.0/output/Add": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
        "/encoder/layer.0/output/Add_1": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
        "/pooler/Gather": "/BertModel/BertPooler",
        "/pooler/MatMul": "/BertModel/BertPooler",
        "/pooler/Add": "/BertModel/BertPooler",
        "/pooler/Tanh": "/BertModel/BertPooler",
        # Root fallback nodes
        "/Constant_10": "/BertModel",
        "/Shape": "/BertModel",
        "/Gather_3": "/BertModel",
        "/Unsqueeze": "/BertModel",
        "/Concat": "/BertModel",
        "/Reshape_2": "/BertModel",
        "/Cast": "/BertModel",
        "/Constant_11": "/BertModel"
    }
    
    # Add tagging statistics
    data.tagging_stats = {
        "total_nodes": 136,
        "root_nodes": 19,
        "scoped_nodes": 117,
        "unique_scopes": 32,
        "direct_matches": 83,
        "parent_matches": 34,
        "operation_matches": 0,
        "root_fallbacks": 19,
        "empty_tags": 0
    }
    
    # Add step-specific data
    data.steps = {
        "input_generation": {
            "model_type": "bert",
            "task": "feature-extraction",
            "method": "auto",
            "inputs": {
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            },
            "outputs": ["last_hidden_state", "pooler_output"]
        },
        "hierarchy_building": {
            "builder": "TracingHierarchyBuilder",
            "modules_traced": 18,
            "execution_steps": 36
        },
        "onnx_conversion": {
            "opset_version": 11,
            "do_constant_folding": True,
            "export_params": True
        }
    }
    
    return data


def create_minimal_fixture() -> ExportData:
    """Create minimal ExportData fixture for quick tests."""
    data = ExportData(
        model_name="test-model",
        model_class="TestModel",
        total_modules=3,
        total_parameters=1000,
        output_path="test.onnx",
    )
    
    data.hierarchy = {
        "": {"class_name": "TestModel", "traced_tag": "/TestModel"},
        "layer1": {"class_name": "Linear", "traced_tag": "/TestModel/Linear"},
        "layer2": {"class_name": "Linear", "traced_tag": "/TestModel/Linear.1"}
    }
    
    data.total_nodes = 10
    data.tagged_nodes = {
        "/layer1/MatMul": "/TestModel/Linear",
        "/layer2/MatMul": "/TestModel/Linear.1",
        "/Add": "/TestModel"
    }
    
    data.tagging_stats = {
        "direct_matches": 2,
        "parent_matches": 0,
        "root_fallbacks": 1,
        "empty_tags": 0
    }
    
    return data


def create_step_timeline() -> list[tuple[ExportStep, dict]]:
    """Create a timeline of export steps with associated data updates."""
    return [
        (ExportStep.MODEL_PREP, {
            "model_name": "prajjwal1/bert-tiny",
            "model_class": "BertModel",
            "total_modules": 48,
            "total_parameters": 4385920
        }),
        (ExportStep.INPUT_GEN, {
            "model_type": "bert",
            "task": "feature-extraction",
            "inputs": {
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            }
        }),
        (ExportStep.HIERARCHY, {
            "modules_traced": 18,
            "execution_steps": 36
        }),
        (ExportStep.STRUCTURE, {
            "total_layers": 2,
            "attention_layers": 2,
            "feedforward_layers": 2
        }),
        (ExportStep.CONVERSION, {
            "opset_version": 11,
            "optimization_passes": ["constant_folding", "shape_inference"]
        }),
        (ExportStep.NODE_TAGGING, {
            "total_nodes": 136,
            "direct_matches": 83,
            "parent_matches": 34,
            "root_fallbacks": 19
        }),
        (ExportStep.VALIDATION, {
            "validation_passed": True,
            "warnings": []
        }),
        (ExportStep.COMPLETE, {
            "onnx_size_mb": 16.76,
            "export_time_seconds": 7.72
        })
    ]