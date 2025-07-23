"""
Test the unified report generator.
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_report_generator import ExportSession, UnifiedReportGenerator


def create_test_session() -> ExportSession:
    """Create a test session with sample data."""
    session = ExportSession(
        model_name_or_path="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4400000,
        output_path="bert-tiny.onnx",
        verbose=True,
        enable_reporting=True
    )
    
    # Add steps
    session.add_step("model_preparation", "completed", 
                     model_class="BertModel",
                     module_count=48,
                     parameter_count=4400000,
                     eval_mode=True)
    
    session.add_step("input_generation", "completed",
                     method="auto",
                     model_type="bert", 
                     task="feature-extraction",
                     inputs={
                         "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                         "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                         "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
                     })
    
    session.add_step("hierarchy_building", "completed",
                     builder="TracingHierarchyBuilder",
                     modules_traced=18,
                     execution_steps=36)
    
    session.add_step("onnx_export", "completed",
                     export_config={
                         "opset_version": 17,
                         "do_constant_folding": True,
                         "verbose": False
                     })
    
    session.add_step("node_tagger_creation", "completed",
                     model_root_tag="/BertModel",
                     operation_fallback=False)
    
    session.add_step("node_tagging", "completed")
    
    session.add_step("tag_injection", "completed")
    
    session.add_step("metadata_generation", "completed")
    
    # Add hierarchy data
    session.hierarchy_data = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"},
        "pooler": {"class_name": "BertPooler", "traced_tag": "/BertModel/BertPooler"}
    }
    
    # Add tagged nodes
    session.tagged_nodes = {
        "/Cast": "/BertModel",
        "/embeddings/Add": "/BertModel/BertEmbeddings",
        "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        # ... more nodes
    }
    
    # Add statistics
    session.tagging_statistics = {
        "total_nodes": 136,
        "root_nodes": 19,
        "scoped_nodes": 117,
        "unique_scopes": 32,
        "direct_matches": 83,
        "parent_matches": 34,
        "operation_matches": 0,
        "root_fallbacks": 19
    }
    
    # Set overall stats
    session.export_time = 7.72
    session.onnx_nodes_count = 136
    session.tagged_nodes_count = 136
    session.coverage_percentage = 100.0
    session.empty_tags = 0
    session.onnx_file_size_mb = 16.76
    session.opset_version = 17
    session.input_names = ["input_ids", "attention_mask", "token_type_ids"]
    session.output_names = ["last_hidden_state", "pooler_output"]
    
    # Set file paths
    session.onnx_file_path = "bert-tiny.onnx"
    session.metadata_file_path = "bert-tiny_htp_metadata.json"
    session.report_file_path = "bert-tiny_full_report.txt"
    
    return session


def test_console_output():
    """Test console output generation."""
    print("Testing console output generation...")
    
    session = create_test_session()
    generator = UnifiedReportGenerator(session)
    
    console_output = generator.generate_console_output(truncate_trees=True)
    
    # Remove ANSI codes for testing
    import re
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', console_output)
    
    # Verify key sections are present
    assert "STEP 1/8: MODEL PREPARATION" in clean_output
    assert "STEP 6/8: ONNX NODE TAGGING" in clean_output
    assert "Direct matches: 83" in clean_output
    assert "Parent matches: 34" in clean_output
    assert "Coverage: 100.0%" in clean_output
    
    print("✅ Console output test passed")
    print(f"   Generated {len(console_output.splitlines())} lines")
    
    # Save for inspection
    with open("test_console_output.txt", "w") as f:
        f.write(console_output)


def test_metadata_generation():
    """Test metadata generation."""
    print("\nTesting metadata generation...")
    
    session = create_test_session()
    generator = UnifiedReportGenerator(session)
    
    metadata = generator.generate_metadata()
    
    # Verify structure
    assert "export_context" in metadata
    assert "model" in metadata
    assert "nodes" in metadata
    assert "report" in metadata
    
    # Verify statistics are in node_tagging step
    node_tagging = metadata["report"]["steps"]["node_tagging"]
    assert "statistics" in node_tagging
    assert node_tagging["statistics"]["direct_matches"] == 83
    assert node_tagging["statistics"]["parent_matches"] == 34
    
    # Verify statistics are also in report.node_tagging
    assert "node_tagging" in metadata["report"]
    assert metadata["report"]["node_tagging"]["statistics"]["direct_matches"] == 83
    
    # Verify output files include report
    assert "report" in metadata["outputs"]
    assert metadata["outputs"]["report"]["path"] == "bert-tiny_full_report.txt"
    
    print("✅ Metadata test passed")
    
    # Save for inspection
    with open("test_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def test_text_report_generation():
    """Test text report generation."""
    print("\nTesting text report generation...")
    
    session = create_test_session()
    generator = UnifiedReportGenerator(session)
    
    text_report = generator.generate_text_report()
    
    # Verify sections
    assert "COMPLETE MODULE HIERARCHY" in text_report
    assert "COMPLETE NODE MAPPINGS" in text_report
    assert "EXPORT STATISTICS" in text_report
    assert "CONSOLE OUTPUT" in text_report
    
    # Verify no truncation in console output
    assert "(truncated for console)" not in text_report
    
    print("✅ Text report test passed")
    print(f"   Generated {len(text_report.splitlines())} lines")
    
    # Save for inspection
    with open("test_full_report.txt", "w") as f:
        f.write(text_report)


def test_consistency():
    """Test consistency across all formats."""
    print("\nTesting consistency across formats...")
    
    session = create_test_session()
    generator = UnifiedReportGenerator(session)
    
    console = generator.generate_console_output()
    metadata = generator.generate_metadata()
    report = generator.generate_text_report()
    
    # Remove ANSI codes for testing
    import re
    clean_console = re.sub(r'\x1b\[[0-9;]*m', '', console)
    
    # Check coverage percentage is consistent
    coverage_in_console = "Coverage: 100.0%"
    assert coverage_in_console in clean_console
    assert metadata["report"]["node_tagging"]["coverage"]["coverage_percentage"] == 100.0
    assert "Coverage: 100.0%" in report
    
    # Check statistics are consistent
    assert "Direct matches: 83" in clean_console
    assert metadata["report"]["steps"]["node_tagging"]["statistics"]["direct_matches"] == 83
    assert "Direct Matches: 83" in report
    
    print("✅ Consistency test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Unified Report Generator Tests...\n")
    
    test_console_output()
    test_metadata_generation()
    test_text_report_generation()
    test_consistency()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)
    
    print("\nKey improvements:")
    print("1. Single data source (ExportSession) for all formats")
    print("2. Consistent data across console, metadata, and report")
    print("3. Node tagging statistics now in metadata steps")
    print("4. Report file included in metadata outputs")
    print("5. Shared formatting logic reduces duplication")
    print("6. Easy to extend with new formats or data")


if __name__ == "__main__":
    run_all_tests()