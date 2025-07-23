"""
Simple test to verify ExportMonitor works with fixtures.
"""

import json
import tempfile
from pathlib import Path
from io import StringIO

from rich.console import Console

from export_monitor import ExportMonitor, ExportStep, ConsoleWriter
from fixtures import create_bert_tiny_fixture, create_step_timeline


def test_console_output_capture():
    """Test capturing console output."""
    # Create console that captures to StringIO
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=80)
    
    # Create writer with custom console
    writer = ConsoleWriter()
    writer.console = console
    
    # Write some steps
    data = create_bert_tiny_fixture()
    writer.write(ExportStep.MODEL_PREP, data)
    writer.write(ExportStep.NODE_TAGGING, data)
    
    # Get output
    output = buffer.getvalue()
    print("Captured output:")
    print(output)
    
    # Verify content
    assert "MODEL PREPARATION" in output
    assert "BertModel" in output
    assert "NODE TAGGING" in output
    assert "Coverage:" in output


def test_full_export_workflow():
    """Test complete export workflow with real data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "bert-tiny.onnx"
        
        with ExportMonitor(str(output_path), verbose=False, enable_report=True) as monitor:
            # Load fixture
            fixture = create_bert_tiny_fixture()
            
            # Step 1: Model prep
            monitor.update(
                ExportStep.MODEL_PREP,
                model_name=fixture.model_name,
                model_class=fixture.model_class,
                total_modules=fixture.total_modules,
                total_parameters=fixture.total_parameters
            )
            
            # Step 2: Input gen
            monitor.update(
                ExportStep.INPUT_GEN,
                model_type="bert",
                task="feature-extraction"
            )
            
            # Step 3: Hierarchy
            monitor.data.hierarchy = fixture.hierarchy
            monitor.update(
                ExportStep.HIERARCHY,
                modules_traced=len(fixture.hierarchy),
                execution_steps=36
            )
            
            # Step 6: Node tagging
            monitor.data.total_nodes = fixture.total_nodes
            monitor.data.tagged_nodes = fixture.tagged_nodes
            monitor.data.tagging_stats = fixture.tagging_stats
            monitor.update(ExportStep.NODE_TAGGING)
            
            # Final
            monitor.data.onnx_size_mb = fixture.onnx_size_mb
        
        # Check outputs
        metadata_path = Path(tmpdir) / "bert-tiny_metadata.json"
        report_path = Path(tmpdir) / "bert-tiny_report.txt"
        
        assert metadata_path.exists(), "Metadata file should exist"
        assert report_path.exists(), "Report file should exist"
        
        # Verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print("Metadata structure:")
        print(json.dumps(metadata, indent=2)[:500] + "...")
        
        assert metadata["model"]["name_or_path"] == "prajjwal1/bert-tiny"
        assert metadata["model"]["total_parameters"] == 4385920
        assert len(metadata["modules"]) == len(fixture.hierarchy)
        
        # Verify report
        with open(report_path) as f:
            report = f.read()
        
        print("\nReport preview:")
        print(report[:500] + "...")
        
        assert "HTP EXPORT FULL REPORT" in report
        assert "prajjwal1/bert-tiny" in report
        assert "BertModel" in report
        assert "COMPLETE MODULE HIERARCHY" in report
        assert "COMPLETE NODE MAPPINGS" in report


def test_minimal_export():
    """Test minimal export without verbose output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "minimal.onnx"
        
        # Minimal - only metadata
        with ExportMonitor(str(output_path), verbose=False, enable_report=False) as monitor:
            monitor.update(ExportStep.MODEL_PREP, model_name="test", model_class="TestModel")
            monitor.update(ExportStep.COMPLETE, export_time=1.0)
        
        # Should only have metadata
        assert (Path(tmpdir) / "minimal_metadata.json").exists()
        assert not (Path(tmpdir) / "minimal_report.txt").exists()


if __name__ == "__main__":
    print("Running simple tests...\n")
    
    test_console_output_capture()
    print("\n✓ Console output capture test passed")
    
    test_full_export_workflow()
    print("\n✓ Full export workflow test passed")
    
    test_minimal_export()
    print("\n✓ Minimal export test passed")
    
    print("\nAll tests passed! ✅")