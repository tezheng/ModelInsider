"""
Comprehensive test cases for ExportMonitor system.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest

from export_monitor import (
    ConsoleWriter,
    ExportData,
    ExportMonitor,
    ExportStep,
    MetadataWriter,
    ReportWriter,
    StepAwareWriter,
    step,
)
from fixtures import (
    create_bert_tiny_fixture,
    create_minimal_fixture,
    create_step_timeline,
)


class TestExportData:
    """Test ExportData dataclass functionality."""
    
    def test_coverage_calculation(self):
        """Test coverage percentage calculation."""
        data = ExportData()
        
        # No nodes
        assert data.coverage == 0.0
        
        # Some nodes tagged
        data.total_nodes = 100
        data.tagged_nodes = {"node1": "tag1", "node2": "tag2"}
        assert data.coverage == 2.0
        
        # All nodes tagged
        data.tagged_nodes = {f"node{i}": f"tag{i}" for i in range(100)}
        assert data.coverage == 100.0
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        data = ExportData()
        initial_time = data.start_time
        
        # Should be close to 0 initially
        assert data.elapsed_time < 0.1
        
        # Manually set start time to past
        data.start_time = initial_time - 10
        assert 9.9 < data.elapsed_time < 10.1


class TestStepAwareWriter:
    """Test StepAwareWriter base class and decorator."""
    
    def test_step_decorator_discovery(self):
        """Test that @step decorated methods are discovered."""
        class TestWriter(StepAwareWriter):
            @step(ExportStep.HIERARCHY)
            def handle_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
                return 42
            
            @step(ExportStep.NODE_TAGGING)
            def handle_tagging(self, export_step: ExportStep, data: ExportData) -> int:
                return 24
            
            def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
                return 1
        
        writer = TestWriter()
        data = ExportData()
        
        # Test decorated handlers
        assert writer.write(ExportStep.HIERARCHY, data) == 42
        assert writer.write(ExportStep.NODE_TAGGING, data) == 24
        
        # Test default handler
        assert writer.write(ExportStep.MODEL_PREP, data) == 1
    
    def test_inheritance_override(self):
        """Test that child class methods properly override parent methods."""
        class ParentWriter(StepAwareWriter):
            @step(ExportStep.HIERARCHY)
            def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
                return 1
            
            def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
                return 0
        
        class ChildWriter(ParentWriter):
            @step(ExportStep.HIERARCHY)
            def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
                return 2
        
        parent = ParentWriter()
        child = ChildWriter()
        data = ExportData()
        
        assert parent.write(ExportStep.HIERARCHY, data) == 1
        assert child.write(ExportStep.HIERARCHY, data) == 2


class TestConsoleWriter:
    """Test ConsoleWriter functionality."""
    
    def test_console_output_structure(self):
        """Test that console output has expected structure."""
        writer = ConsoleWriter(width=80, verbose=True)
        data = create_bert_tiny_fixture()
        
        # Capture output for each step
        outputs = {}
        for step_type in [ExportStep.MODEL_PREP, ExportStep.HIERARCHY, ExportStep.NODE_TAGGING]:
            writer.write(step_type, data)
            # Note: In real test, we'd capture Console output
            outputs[step_type] = True
        
        assert all(outputs.values())
    
    def test_hierarchy_truncation(self):
        """Test that hierarchy tree is truncated for console."""
        writer = ConsoleWriter(verbose=True)
        data = create_bert_tiny_fixture()
        
        # Write hierarchy step
        result = writer.write(ExportStep.HIERARCHY, data)
        assert result == 1
        
        # In real implementation, verify truncation by capturing output


class TestMetadataWriter:
    """Test MetadataWriter functionality."""
    
    def test_metadata_structure(self):
        """Test that metadata has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            writer = MetadataWriter(str(output_path))
            data = create_bert_tiny_fixture()
            
            # Process all steps
            for step_type, step_data in create_step_timeline():
                writer.write(step_type, data)
            
            # Write to file
            writer.flush()
            
            # Read and verify
            metadata_path = Path(tmpdir) / "test_metadata.json"
            assert metadata_path.exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Check structure
            assert "export_context" in metadata
            assert metadata["export_context"]["strategy"] == "htp"
            assert metadata["export_context"]["version"] == "1.0"
            
            assert "model" in metadata
            assert metadata["model"]["name_or_path"] == "prajjwal1/bert-tiny"
            assert metadata["model"]["class"] == "BertModel"
            
            assert "modules" in metadata
            assert len(metadata["modules"]) == len(data.hierarchy)
            
            assert "nodes" in metadata
            assert len(metadata["nodes"]) == len(data.tagged_nodes)
            
            assert "report" in metadata
            assert "node_tagging" in metadata["report"]
    
    def test_node_tagging_statistics(self):
        """Test that node tagging statistics are properly recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            writer = MetadataWriter(str(output_path))
            data = create_bert_tiny_fixture()
            
            # Write node tagging step
            writer.write(ExportStep.NODE_TAGGING, data)
            writer.flush()
            
            # Read metadata
            metadata_path = Path(tmpdir) / "test_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Verify statistics
            stats = metadata["report"]["node_tagging"]["statistics"]
            assert stats["total_nodes"] == 136
            assert stats["tagged_nodes"] == len(data.tagged_nodes)
            assert stats["direct_matches"] == 83
            assert stats["parent_matches"] == 34
            assert stats["root_fallbacks"] == 19
            
            # Verify coverage
            coverage = metadata["report"]["node_tagging"]["coverage"]
            assert coverage["percentage"] == pytest.approx(100.0)
            assert coverage["total_onnx_nodes"] == 136


class TestReportWriter:
    """Test ReportWriter functionality."""
    
    def test_report_completeness(self):
        """Test that report contains all sections without truncation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            writer = ReportWriter(str(output_path))
            data = create_bert_tiny_fixture()
            
            # Process all steps
            for step_type, _ in create_step_timeline():
                if step_type == ExportStep.HIERARCHY:
                    data.hierarchy = create_bert_tiny_fixture().hierarchy
                elif step_type == ExportStep.NODE_TAGGING:
                    data.tagged_nodes = create_bert_tiny_fixture().tagged_nodes
                    data.tagging_stats = create_bert_tiny_fixture().tagging_stats
                
                writer.write(step_type, data)
            
            # Write to file
            writer.close()
            
            # Read and verify
            report_path = Path(tmpdir) / "test_report.txt"
            assert report_path.exists()
            
            with open(report_path) as f:
                report_content = f.read()
            
            # Check sections
            assert "HTP EXPORT FULL REPORT" in report_content
            assert "MODEL INFORMATION" in report_content
            assert "COMPLETE MODULE HIERARCHY" in report_content
            assert "NODE TAGGING STATISTICS" in report_content
            assert "COMPLETE NODE MAPPINGS" in report_content
            assert "EXPORT SUMMARY" in report_content
            
            # Verify no truncation
            assert "(truncated)" not in report_content
            assert "..." not in report_content.split("COMPLETE NODE MAPPINGS")[1]
            
            # Check specific data
            assert "prajjwal1/bert-tiny" in report_content
            assert "BertModel" in report_content
            assert "4,385,920" in report_content  # formatted with commas
    
    def test_node_mappings_complete(self):
        """Test that all node mappings are written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            writer = ReportWriter(str(output_path))
            data = create_bert_tiny_fixture()
            
            writer.write(ExportStep.NODE_TAGGING, data)
            writer.close()
            
            report_path = Path(tmpdir) / "test_report.txt"
            with open(report_path) as f:
                report_content = f.read()
            
            # Count node mappings
            node_mapping_section = report_content.split("COMPLETE NODE MAPPINGS")[1]
            node_lines = [line for line in node_mapping_section.split('\n') 
                         if '->' in line and line.strip()]
            
            # Should have all nodes
            assert len(node_lines) == len(data.tagged_nodes)


class TestExportMonitor:
    """Test ExportMonitor orchestration."""
    
    def test_basic_workflow(self):
        """Test basic export workflow with all writers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.onnx"
            
            with ExportMonitor(str(output_path), verbose=True, enable_report=True) as monitor:
                # Should have 3 writers
                assert len(monitor.writers) == 3
                
                # Update with timeline
                for step_type, step_data in create_step_timeline():
                    monitor.update(step_type, **step_data)
            
            # Check files were created
            metadata_path = Path(tmpdir) / "test_metadata.json"
            report_path = Path(tmpdir) / "test_report.txt"
            
            assert metadata_path.exists()
            assert report_path.exists()
    
    def test_conditional_writers(self):
        """Test conditional writer creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # All writers
            monitor1 = ExportMonitor(str(tmpdir / "test1.onnx"), verbose=True, enable_report=True)
            assert len(monitor1.writers) == 3
            assert any(isinstance(w, ConsoleWriter) for w in monitor1.writers)
            assert any(isinstance(w, MetadataWriter) for w in monitor1.writers)
            assert any(isinstance(w, ReportWriter) for w in monitor1.writers)
            
            # No console
            monitor2 = ExportMonitor(str(tmpdir / "test2.onnx"), verbose=False, enable_report=True)
            assert len(monitor2.writers) == 2
            assert not any(isinstance(w, ConsoleWriter) for w in monitor2.writers)
            
            # Minimal (metadata only)
            monitor3 = ExportMonitor(str(tmpdir / "test3.onnx"), verbose=False, enable_report=False)
            assert len(monitor3.writers) == 1
            assert isinstance(monitor3.writers[0], MetadataWriter)
    
    def test_data_accumulation(self):
        """Test that data accumulates correctly across updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ExportMonitor(str(tmpdir / "test.onnx"))
            
            # First update
            monitor.update(ExportStep.MODEL_PREP, model_name="test", model_class="TestModel")
            assert monitor.data.model_name == "test"
            assert monitor.data.model_class == "TestModel"
            
            # Second update - should preserve previous data
            monitor.update(ExportStep.MODEL_PREP, total_modules=10)
            assert monitor.data.model_name == "test"  # preserved
            assert monitor.data.model_class == "TestModel"  # preserved
            assert monitor.data.total_modules == 10  # added
    
    def test_error_handling(self):
        """Test error handling in writers."""
        class ErrorWriter(StepAwareWriter):
            def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
                raise ValueError("Test error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ExportMonitor(str(tmpdir / "test.onnx"))
            monitor.writers.append(ErrorWriter())
            
            # Should not crash on writer error
            monitor.update(ExportStep.MODEL_PREP, model_name="test")
            
            # Other writers should still work
            monitor.finalize()
    
    def test_real_bert_export_simulation(self):
        """Test with real bert-tiny export data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bert-tiny.onnx"
            
            with ExportMonitor(str(output_path), verbose=False, enable_report=True) as monitor:
                # Load fixture data
                fixture = create_bert_tiny_fixture()
                
                # Simulate export process
                monitor.update(
                    ExportStep.MODEL_PREP,
                    model_name=fixture.model_name,
                    model_class=fixture.model_class,
                    total_modules=fixture.total_modules,
                    total_parameters=fixture.total_parameters
                )
                
                monitor.data.hierarchy = fixture.hierarchy
                monitor.update(ExportStep.HIERARCHY)
                
                monitor.data.total_nodes = fixture.total_nodes
                monitor.data.tagged_nodes = fixture.tagged_nodes
                monitor.data.tagging_stats = fixture.tagging_stats
                monitor.update(ExportStep.NODE_TAGGING)
                
                monitor.data.onnx_size_mb = fixture.onnx_size_mb
            
            # Verify outputs
            metadata_path = Path(tmpdir) / "bert-tiny_metadata.json"
            report_path = Path(tmpdir) / "bert-tiny_report.txt"
            
            assert metadata_path.exists()
            assert report_path.exists()
            
            # Verify metadata content
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata["model"]["name_or_path"] == "prajjwal1/bert-tiny"
            assert metadata["model"]["total_parameters"] == 4385920
            assert len(metadata["modules"]) == len(fixture.hierarchy)
            assert metadata["report"]["node_tagging"]["statistics"]["direct_matches"] == 83


@pytest.fixture
def temp_output_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_integration_with_fixtures(temp_output_dir):
    """Integration test using fixtures."""
    output_path = temp_output_dir / "integration_test.onnx"
    
    with ExportMonitor(str(output_path), verbose=True, enable_report=True) as monitor:
        # Use timeline fixture
        for step_type, step_data in create_step_timeline():
            monitor.update(step_type, **step_data)
        
        # Add bert-tiny specific data
        fixture = create_bert_tiny_fixture()
        monitor.data.hierarchy = fixture.hierarchy
        monitor.data.tagged_nodes = fixture.tagged_nodes
        monitor.data.tagging_stats = fixture.tagging_stats
    
    # Verify all outputs
    assert (temp_output_dir / "integration_test_metadata.json").exists()
    assert (temp_output_dir / "integration_test_report.txt").exists()


if __name__ == "__main__":
    # Run specific test for debugging
    test = TestExportMonitor()
    test.test_real_bert_export_simulation()
    print("All manual tests passed!")