#!/usr/bin/env python3
"""Test script to generate a sample markdown report with the new format."""

import tempfile
from pathlib import Path

from modelexport.strategies.htp.base_writer import ExportData, ExportStep
from modelexport.strategies.htp.markdown_report_writer import MarkdownReportWriter
from modelexport.strategies.htp.step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    ModuleInfo,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
    TensorInfo,
)


def create_sample_data():
    """Create sample export data for testing."""
    data = ExportData(
        model_name="prajjwal1/bert-tiny",
        output_path="test_model.onnx",
        embed_hierarchy=True,
    )
    
    # Model prep data
    data.model_prep = ModelPrepData(
        model_class="BertModel",
        total_modules=48,
        total_parameters=4385920,
    )
    
    # Input generation data
    data.input_gen = InputGenData(
        method="auto_generated",
        model_type="bert",
        task="feature-extraction",
        inputs={
            "input_ids": TensorInfo(shape=[2, 16], dtype="torch.int64"),
            "attention_mask": TensorInfo(shape=[2, 16], dtype="torch.int64"),
            "token_type_ids": TensorInfo(shape=[2, 16], dtype="torch.int64"),
        }
    )
    
    # Hierarchy data with more modules for testing
    data.hierarchy = HierarchyData(
        hierarchy={
            "": ModuleInfo(
                class_name="BertModel",
                traced_tag="/BertModel",
                execution_order=0,
            ),
            "embeddings": ModuleInfo(
                class_name="BertEmbeddings", 
                traced_tag="/BertModel/BertEmbeddings",
                execution_order=1,
            ),
            "embeddings.LayerNorm": ModuleInfo(
                class_name="LayerNorm",
                traced_tag="/BertModel/BertEmbeddings/LayerNorm",
                execution_order=2,
            ),
            "embeddings.dropout": ModuleInfo(
                class_name="Dropout",
                traced_tag="/BertModel/BertEmbeddings/Dropout",
                execution_order=3,
            ),
            "encoder": ModuleInfo(
                class_name="BertEncoder",
                traced_tag="/BertModel/BertEncoder",
                execution_order=4,
            ),
            "encoder.layer.0": ModuleInfo(
                class_name="BertLayer",
                traced_tag="/BertModel/BertEncoder/BertLayer.0",
                execution_order=5,
            ),
            "encoder.layer.0.attention": ModuleInfo(
                class_name="BertAttention",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention",
                execution_order=6,
            ),
            "encoder.layer.0.attention.self": ModuleInfo(
                class_name="BertSdpaSelfAttention",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
                execution_order=7,
            ),
            "encoder.layer.0.attention.output": ModuleInfo(
                class_name="BertSelfOutput",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
                execution_order=8,
            ),
            "encoder.layer.0.intermediate": ModuleInfo(
                class_name="BertIntermediate",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
                execution_order=9,
            ),
            "encoder.layer.0.output": ModuleInfo(
                class_name="BertOutput",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertOutput",
                execution_order=10,
            ),
            "encoder.layer.1": ModuleInfo(
                class_name="BertLayer",
                traced_tag="/BertModel/BertEncoder/BertLayer.1",
                execution_order=11,
            ),
            "pooler": ModuleInfo(
                class_name="BertPooler",
                traced_tag="/BertModel/BertPooler",
                execution_order=12,
            ),
        },
        execution_steps=36,
        module_list=[],
    )
    
    # ONNX export data
    data.onnx_export = ONNXExportData(
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        onnx_size_mb=16.76,
    )
    
    # Node tagging data
    data.node_tagging = NodeTaggingData(
        total_nodes=136,
        tagged_nodes={
            "/embeddings/Constant": "/BertModel/BertEmbeddings",
            "/embeddings/Add": "/BertModel/BertEmbeddings",
            "/embeddings/LayerNorm/LayerNormalization": "/BertModel/BertEmbeddings/LayerNorm",
            "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
            "/encoder/layer.0/attention/self/Softmax": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
            "/encoder/layer.0/attention/output/dense/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput",
            "/encoder/layer.0/intermediate/dense/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertIntermediate",
            "/encoder/layer.0/output/dense/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertOutput",
            "/pooler/dense/Gemm": "/BertModel/BertPooler",
            # Add more nodes for variety
            "/encoder/layer.1/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.1",
            "/encoder/layer.1/attention/self/MatMul_1": "/BertModel/BertEncoder/BertLayer.1",
            "/encoder/layer.1/attention/self/MatMul_2": "/BertModel/BertEncoder/BertLayer.1",
        },
        tagging_stats={
            "direct_matches": 83,
            "parent_matches": 34,
            "root_fallbacks": 19,
            "empty_tags": 0,
        },
        coverage=100.0,
        op_counts={
            "MatMul": 25,
            "Add": 20,
            "LayerNormalization": 15,
            "Softmax": 8,
            "Gemm": 2,
        },
    )
    
    # Tag injection data
    data.tag_injection = TagInjectionData(
        tags_injected=True,
        tags_stripped=False,
    )
    
    # Set export time
    data.export_time = 4.79
    
    return data


def main():
    """Generate a sample report with the new format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_model.onnx")
        writer = MarkdownReportWriter(output_path)
        
        # Create sample data
        data = create_sample_data()
        
        # Process all steps
        for step in ExportStep:
            writer.write(step, data)
        
        writer.flush()
        
        # Read and print the report
        report_path = Path(tmpdir) / "test_model_htp_export_report.md"
        if report_path.exists():
            content = report_path.read_text()
            print("Generated Report Preview:")
            print("=" * 80)
            print(content[:2000])  # Show first 2000 chars
            print("...")
            print("=" * 80)
            print(f"\nFull report saved to: {report_path}")
            
            # Save a copy to temp folder for inspection
            temp_copy = Path("temp/sample_report.md")
            temp_copy.parent.mkdir(exist_ok=True)
            temp_copy.write_text(content)
            print(f"Copy saved to: {temp_copy}")
        else:
            print(f"Error: Report not found at {report_path}")


if __name__ == "__main__":
    main()