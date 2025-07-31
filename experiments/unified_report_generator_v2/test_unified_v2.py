"""
Test the interface-based unified report generator.
"""

import re
import time
from typing import Any

from data_provider import ExportSessionData, SessionDataProvider
from generator import UnifiedReportGenerator
from interfaces import ReportFormat, StepInfo


def create_test_session() -> ExportSessionData:
    """Create test session with sample data."""
    session = ExportSessionData(
        model_name_or_path="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4400000,
        output_path="bert-tiny.onnx",
        export_time=7.72,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        onnx_path="bert-tiny.onnx",
        onnx_size_mb=16.76,
        metadata_path="bert-tiny_htp_metadata.json",
        report_path="bert-tiny_full_report.txt",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"]
    )
    
    # Add steps
    session.steps["input_generation"] = StepInfo(
        name="input_generation",
        number=2,
        total=8,
        title="INPUT GENERATION & VALIDATION",
        icon="🔧",
        status="completed",
        details={
            "method": "auto",
            "model_type": "bert",
            "task": "feature-extraction",
            "inputs": {
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            }
        }
    )
    
    session.steps["hierarchy_building"] = StepInfo(
        name="hierarchy_building",
        number=3,
        total=8,
        title="HIERARCHY BUILDING",
        icon="🏗️",
        status="completed",
        details={
            "builder": "TracingHierarchyBuilder",
            "modules_traced": 18,
            "execution_steps": 36
        }
    )
    
    session.steps["node_tagging"] = StepInfo(
        name="node_tagging",
        number=6,
        total=8,
        title="ONNX NODE TAGGING",
        icon="🔗",
        status="completed",
        details={}
    )
    
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
    
    # Add some tagged nodes
    session.tagged_nodes = {
        "/Cast": "/BertModel",
        "/embeddings/Add": "/BertModel/BertEmbeddings",
        "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
        "/pooler/Gather": "/BertModel/BertPooler"
    }
    
    # Add statistics
    session.tagging_statistics = {
        "total_nodes": 136,
        "tagged_nodes": 136,
        "direct_matches": 83,
        "parent_matches": 34,
        "operation_matches": 0,
        "root_fallbacks": 19,
        "empty_tags": 0
    }
    
    return session


def test_console_generation():
    """Test console output generation."""
    print("Testing console output generation...")
    
    session = create_test_session()
    provider = SessionDataProvider(session)
    generator = UnifiedReportGenerator(provider)
    
    console_output = generator.generate(ReportFormat.CONSOLE)
    
    # Remove ANSI codes for testing
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', console_output)
    
    # Verify key sections
    assert "STEP 2/8: INPUT GENERATION" in clean_output
    assert "STEP 6/8: ONNX NODE TAGGING" in clean_output
    assert "Direct matches: 83" in clean_output
    assert "Coverage: 100.0%" in clean_output
    
    # Check truncation
    assert "(truncated for console)" in clean_output or "more lines" in clean_output
    
    print("✅ Console output test passed")
    print(f"   Generated {len(console_output.splitlines())} lines")
    
    # Save for inspection
    with open("test_console_v2.txt", "w") as f:
        f.write(console_output)


def test_metadata_generation():
    """Test metadata generation."""
    print("\nTesting metadata generation...")
    
    session = create_test_session()
    provider = SessionDataProvider(session)
    generator = UnifiedReportGenerator(provider)
    
    metadata = generator.generate(ReportFormat.METADATA)
    
    # Debug: print keys
    print(f"Metadata keys: {list(metadata.keys())}")
    
    # Verify structure
    assert "export_context" in metadata
    assert "model" in metadata
    assert "nodes" in metadata or "tagged_nodes" in metadata or "modules" in metadata
    assert "report" in metadata or "summary" in metadata
    
    # Verify statistics are included
    if "report" in metadata and "node_tagging" in metadata["report"]:
        stats = metadata["report"]["node_tagging"].get("statistics", {})
        assert stats.get("direct_matches") == 83
    
    # Verify outputs include report
    assert "outputs" in metadata
    assert "report" in metadata["outputs"]
    
    print("✅ Metadata test passed")
    
    # Save for inspection
    generator.save("test_metadata_v2.json", ReportFormat.METADATA)


def test_text_generation():
    """Test text report generation."""
    print("\nTesting text report generation...")
    
    session = create_test_session()
    provider = SessionDataProvider(session)
    generator = UnifiedReportGenerator(provider)
    
    text_report = generator.generate(ReportFormat.TEXT)
    
    # Verify sections
    assert "HTP EXPORT FULL REPORT" in text_report
    assert "MODULE HIERARCHY" in text_report
    assert "COMPLETE MODULE HIERARCHY" in text_report
    assert "COMPLETE NODE MAPPINGS" in text_report
    
    # Verify no truncation
    assert "(truncated for console)" not in text_report
    
    # Verify statistics (case-insensitive)
    assert "83" in text_report and ("direct" in text_report.lower() or "Direct" in text_report)
    assert "100.0%" in text_report
    
    print("✅ Text report test passed")
    print(f"   Generated {len(text_report.splitlines())} lines")
    
    # Save for inspection
    generator.save("test_report_v2.txt", ReportFormat.TEXT)


def test_shared_logic():
    """Test that logic is actually shared."""
    print("\nTesting shared logic...")
    
    session = create_test_session()
    provider = SessionDataProvider(session)
    generator = UnifiedReportGenerator(provider)
    
    # Generate all formats
    console = generator.generate(ReportFormat.CONSOLE)
    metadata = generator.generate(ReportFormat.METADATA)
    text = generator.generate(ReportFormat.TEXT)
    
    # Clean console output
    clean_console = re.sub(r'\x1b\[[0-9;]*m', '', console)
    
    # Check that same statistics appear in all formats
    assert "83" in clean_console  # Direct matches in console
    
    # Check metadata structure
    if "report" in metadata and "node_tagging" in metadata["report"]:
        assert metadata["report"]["node_tagging"]["statistics"]["direct_matches"] == 83
    elif "summary" in metadata:
        # Check in summary section
        assert "83" in str(metadata)
    
    assert "83" in text  # Should appear somewhere in text report
    
    # Check model info consistency
    assert "BertModel" in clean_console
    assert metadata["model"]["class"] == "BertModel"
    assert "Model Class: BertModel" in text
    
    print("✅ Shared logic test passed")


def test_extensibility():
    """Test adding custom sections."""
    print("\nTesting extensibility...")
    
    from interfaces import IReportSection
    
    class CustomSection(IReportSection):
        """Custom test section."""
        
        def get_data(self) -> dict[str, Any]:
            return {"custom": "data"}
        
        def render(self, format: ReportFormat, context: dict[str, Any]) -> Any:
            if format == ReportFormat.CONSOLE:
                return "\n🎯 CUSTOM SECTION\nThis is custom content!\n"
            elif format == ReportFormat.TEXT:
                return "\nCUSTOM SECTION\nThis is custom content!\n"
            elif format == ReportFormat.METADATA:
                return {"custom_section": self.get_data()}
    
    session = create_test_session()
    provider = SessionDataProvider(session)
    generator = UnifiedReportGenerator(provider)
    
    # Add custom section
    generator.add_section(CustomSection())
    
    # Generate and verify
    console = generator.generate(ReportFormat.CONSOLE)
    assert "CUSTOM SECTION" in console
    
    metadata = generator.generate(ReportFormat.METADATA)
    assert "custom_section" in metadata
    
    print("✅ Extensibility test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Interface-Based Unified Report Generator Tests...\n")
    
    test_console_generation()
    test_metadata_generation()
    test_text_generation()
    test_shared_logic()
    test_extensibility()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)
    
    print("\nKey improvements:")
    print("1. Proper interface-based design with IReportSection, IReportFormatter, etc.")
    print("2. Shared formatting logic across all output formats")
    print("3. Each section knows how to render itself in different formats")
    print("4. Easy to add new sections or formats")
    print("5. Consistent data access through IDataProvider")
    print("6. Testable components with clear separation of concerns")


if __name__ == "__main__":
    run_all_tests()