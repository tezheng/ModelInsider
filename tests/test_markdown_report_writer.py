"""
Test cases for MarkdownReportWriter.

These tests verify that the markdown report is generated correctly,
independently from console output, with all required sections.
"""

import json
import tempfile
from pathlib import Path

import pytest

from modelexport.strategies.htp_new.base_writer import ExportData, ExportStep
from modelexport.strategies.htp_new.markdown_report_writer import MarkdownReportWriter
from modelexport.strategies.htp_new.step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    ModuleInfo,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
    TensorInfo,
)


@pytest.fixture
def sample_export_data():
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
    
    # Hierarchy data
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
            "encoder.layer.0.attention.self": ModuleInfo(
                class_name="BertSdpaSelfAttention",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
                execution_order=5,
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
            "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
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


class TestMarkdownReportWriter:
    """Test cases for MarkdownReportWriter."""
    
    def test_report_creation(self, sample_export_data):
        """Test that markdown report is created with correct filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Check report file exists with .md extension
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            assert report_path.exists()
            
            # Read and verify content
            content = report_path.read_text()
            assert "# HTP ONNX Export Report" in content
            assert "prajjwal1/bert-tiny" in content
    
    def test_all_sections_present(self, sample_export_data):
        """Test that all required sections are present in the report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check all main sections
            assert "## Export Process" in content
            assert "### ✅ Step 1/6: Model Preparation" in content
            assert "### ✅ Step 2/6: Input Generation" in content
            assert "### ✅ Step 3/6: Hierarchy Building" in content
            assert "### ✅ Step 4/6: ONNX Export" in content
            assert "### ✅ Step 5/6: Node Tagging" in content
            assert "### ✅ Step 6/6: Tag Injection" in content
            assert "## Module Hierarchy" in content
            # Node Distribution section has been removed in TEZ-27
            assert "## Node Distribution" not in content
            assert "## Complete Node Mappings" in content
            assert "## Export Summary" in content
    
    def test_mermaid_flowchart_disabled(self, sample_export_data):
        """Test that Mermaid flowchart is disabled (TEZ-28)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Verify Mermaid is disabled
            assert "```mermaid" not in content
            assert "flowchart LR" not in content
            assert "-->" not in content
            
            # Verify informational note is present
            assert "Hierarchy Visualization" in content
            assert "Mermaid diagram temporarily disabled" in content
            
            # Verify module data is still present in the module table
            assert "BertModel" in content
    
    def test_collapsible_sections(self, sample_export_data):
        """Test that collapsible sections are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check collapsible details tags
            assert "<details>" in content
            assert "</details>" in content
            assert "<summary>" in content
            assert "Click to expand" in content
    
    def test_input_table_format(self, sample_export_data):
        """Test that input table is formatted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check input table (with SnakeMD formatting)
            assert "| Input Name" in content
            assert "| Shape" in content  
            assert "| Data Type" in content
            assert "input_ids" in content
            assert "[2, 16]" in content
            assert "torch.int64" in content
    
    def test_statistics_table(self, sample_export_data):
        """Test that tagging statistics table is formatted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check statistics table
            assert "| Match Type" in content
            assert "| Count" in content
            assert "| Percentage" in content
            assert "Direct Matches" in content
            assert "83" in content
            # Check percentage is calculated correctly (83/136 * 100 = 61.0%)
            assert "61.0%" in content or "61.03%" in content
    
    def test_report_without_verbose(self, sample_export_data):
        """Test that report is generated correctly without verbose flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            # This simulates --enable-reporting without --verbose
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Report should still have all sections
            assert "# HTP ONNX Export Report" in content
            assert "## Export Process" in content
            assert "## Module Hierarchy" in content
            # Should not depend on console output
            assert "CONSOLE OUTPUT" not in content
    
    def test_clean_onnx_mode(self, sample_export_data):
        """Test report generation when hierarchy tags are stripped."""
        sample_export_data.embed_hierarchy = False
        sample_export_data.tag_injection.tags_injected = False
        sample_export_data.tag_injection.tags_stripped = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check tag injection section shows stripped
            assert "Stripped (clean ONNX)" in content
            assert "Embedded in ONNX" not in content
    
    def test_parameter_formatting(self, sample_export_data):
        """Test parameter count formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check parameter formatting
            assert "4,385,920" in content  # Comma-separated
            assert "4.4M" in content  # Human readable
    
    def test_no_truncation(self, sample_export_data):
        """Test that report contains complete data without truncation."""
        # Add many modules to test no truncation
        for i in range(50):
            sample_export_data.hierarchy.hierarchy[f"layer.{i}"] = ModuleInfo(
                class_name=f"TestLayer{i}",
                traced_tag=f"/Model/Layer{i}",
                execution_order=i + 10,
            )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Check no truncation indicators
            assert "... truncated ..." not in content
            assert "showing first" not in content.lower()
            assert "and more" not in content.lower()
            
            # Check actual modules are present
            assert "layer.49" in content
            assert "TestLayer49" in content
    
    def test_tez27_report_structure_changes(self, sample_export_data):
        """Test TEZ-27 report structure improvements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # 1. Check that "Top 20 Nodes by Hierarchy" is NOT in the report
            assert "Top 20 Nodes by Hierarchy" not in content
            
            # 2. Check that "Complete HF Hierarchy with ONNX Nodes" appears in Node Tagging section
            # Find the Node Tagging section
            node_tagging_idx = content.find("### ✅ Step 5/6: Node Tagging")
            step_6_idx = content.find("### ✅ Step 6/6: Tag Injection")
            
            assert node_tagging_idx != -1, "Node Tagging section not found"
            assert step_6_idx != -1, "Tag Injection section not found"
            
            # Extract Node Tagging section content
            node_tagging_section = content[node_tagging_idx:step_6_idx]
            
            # Check that hierarchy with nodes is in this section
            assert "Complete HF Hierarchy with ONNX Nodes" in node_tagging_section
            assert "Complete HF Hierarchy with ONNX Nodes" in content
            
            # 3. Check Module List table has Nodes column
            # Find Module List section
            module_list_idx = content.find("### Module List (Sorted by Execution Order)")
            assert module_list_idx != -1, "Module List section not found"
            
            # Check table headers include Nodes column
            # The table should have headers: Execution Order | Class Name | Nodes | Tag | Scope
            module_list_section = content[module_list_idx:module_list_idx + 500]
            assert "| Execution Order" in module_list_section
            assert "| Class Name" in module_list_section
            assert "| Nodes" in module_list_section
            assert "| Tag" in module_list_section
            assert "| Scope" in module_list_section
            
            # 4. Verify Node Distribution section is removed
            assert "## Node Distribution" not in content
            assert "### Module Node Distribution" not in content
            assert "### Top Operations by Count" not in content
    
    def test_module_list_nodes_column_format(self, sample_export_data):
        """Test that Module List table Nodes column has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                writer.write(step, sample_export_data)
            
            writer.flush()
            
            # Read report
            report_path = Path(tmpdir) / "test_model_htp_export_report.md"
            content = report_path.read_text()
            
            # Find Module List table
            module_list_idx = content.find("### Module List (Sorted by Execution Order)")
            assert module_list_idx != -1
            
            # Look for the table content after the headers
            # We expect to see entries like "2/2" for modules with nodes
            # Format: {direct}/{total} where direct = nodes in this module only, total = including children
            
            # Check for slash format
            import re
            node_pattern = re.compile(r'\d+/\d+')
            matches = node_pattern.findall(content[module_list_idx:])
            assert len(matches) > 0, "No node count entries found in Module List table"
            
            # For test data:
            # - Root has 0 direct nodes but 3 total (0/3)
            # - BertEmbeddings has 2 direct and 2 total (2/2) 
            # - BertSdpaSelfAttention has 1 direct and 1 total (1/1)
            table_section = content[module_list_idx:module_list_idx + 2000]
            assert "0/3" in table_section  # Root module
            assert "2/2" in table_section  # BertEmbeddings (leaf module)
            assert "1/1" in table_section  # BertSdpaSelfAttention (leaf module)


class TestMarkdownReportIntegration:
    """Integration tests with export monitor."""
    
    def test_export_monitor_integration(self, sample_export_data):
        """Test integration with HTPExportMonitor."""
        from modelexport.strategies.htp_new.export_monitor import HTPExportMonitor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            
            # Create monitor with report enabled
            monitor = HTPExportMonitor(
                output_path=output_path,
                model_name="test-model",
                verbose=False,  # Test without verbose
                enable_report=True,
                embed_hierarchy=True,
            )
            
            # Verify markdown report writer is used
            assert monitor.report_writer is not None
            assert monitor.report_writer.__class__.__name__ == "MarkdownReportWriter"
            
            # Verify correct number of writers
            assert len(monitor.writers) == 2  # metadata + report (no console)
    
    def test_metadata_report_path_sync(self, sample_export_data):
        """Test that metadata correctly records markdown report path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_model.onnx")
            
            # Create both writers
            from modelexport.strategies.htp_new.metadata_writer import MetadataWriter
            
            metadata_writer = MetadataWriter(output_path)
            report_writer = MarkdownReportWriter(output_path)
            
            # Process all steps
            for step in ExportStep:
                metadata_writer.write(step, sample_export_data)
                report_writer.write(step, sample_export_data)
            
            # Flush both
            report_writer.flush()
            metadata_writer.flush()
            
            # Check metadata points to .md file
            metadata_path = Path(tmpdir) / "test_model_htp_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            if "outputs" in metadata and "report" in metadata["outputs"]:
                assert metadata["outputs"]["report"]["path"].endswith(".md")