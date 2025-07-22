"""
Generate test outputs (report.txt and metadata.json) for review.
"""

import json
from pathlib import Path

from export_monitor import ExportMonitor, ExportStep
from fixtures import create_bert_tiny_fixture, create_step_timeline


def generate_bert_tiny_outputs():
    """Generate report and metadata for bert-tiny export."""
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "bert-tiny.onnx"
    
    print("Generating bert-tiny export outputs...")
    
    with ExportMonitor(str(output_path), verbose=True, enable_report=True) as monitor:
        # Load fixture data
        fixture = create_bert_tiny_fixture()
        
        # Step 1: Model preparation
        monitor.update(
            ExportStep.MODEL_PREP,
            model_name=fixture.model_name,
            model_class=fixture.model_class,
            total_modules=fixture.total_modules,
            total_parameters=fixture.total_parameters
        )
        
        # Step 2: Input generation
        if "input_generation" in fixture.steps:
            step_data = fixture.steps["input_generation"]
            monitor.update(
                ExportStep.INPUT_GEN,
                model_type=step_data["model_type"],
                task=step_data["task"],
                inputs=step_data["inputs"]
            )
            # Also update the monitor's step data
            monitor.data.steps["input_generation"] = step_data
        
        # Step 3: Hierarchy building
        monitor.data.hierarchy = fixture.hierarchy
        monitor.update(
            ExportStep.HIERARCHY,
            modules_traced=18,
            execution_steps=36
        )
        
        # Step 4: Structure analysis (optional)
        monitor.update(
            ExportStep.STRUCTURE,
            total_layers=2,
            attention_layers=2,
            feedforward_layers=2
        )
        
        # Step 5: ONNX conversion
        monitor.update(
            ExportStep.CONVERSION,
            opset_version=11,
            optimization_passes=["constant_folding", "shape_inference"]
        )
        
        # Step 6: Node tagging
        monitor.data.total_nodes = fixture.total_nodes
        monitor.data.tagged_nodes = fixture.tagged_nodes
        monitor.data.tagging_stats = fixture.tagging_stats
        monitor.update(ExportStep.NODE_TAGGING)
        
        # Step 7: Validation
        monitor.update(
            ExportStep.VALIDATION,
            validation_passed=True,
            warnings=[]
        )
        
        # Step 8: Complete
        monitor.data.onnx_size_mb = fixture.onnx_size_mb
        monitor.data.export_time = 7.72
    
    print(f"✓ Generated files in {output_dir}/:")
    print(f"  - bert-tiny_metadata.json")
    print(f"  - bert-tiny_report.txt")
    
    # Display file contents
    print("\n" + "="*80)
    print("METADATA.JSON PREVIEW:")
    print("="*80)
    
    metadata_path = output_dir / "bert-tiny_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(json.dumps(metadata, indent=2)[:2000] + "\n... (truncated for display)")
    
    print("\n" + "="*80)
    print("REPORT.TXT PREVIEW:")
    print("="*80)
    
    report_path = output_dir / "bert-tiny_report.txt"
    with open(report_path) as f:
        report = f.read()
    
    print(report[:2000] + "\n... (truncated for display)")
    
    return metadata_path, report_path


def generate_minimal_outputs():
    """Generate minimal export outputs."""
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "minimal-test.onnx"
    
    print("\n\nGenerating minimal export outputs...")
    
    with ExportMonitor(str(output_path), verbose=False, enable_report=True) as monitor:
        # Minimal data
        monitor.update(
            ExportStep.MODEL_PREP,
            model_name="test-model",
            model_class="TestModel",
            total_modules=3,
            total_parameters=1000
        )
        
        # Simple hierarchy
        monitor.data.hierarchy = {
            "": {"class_name": "TestModel", "traced_tag": "/TestModel"},
            "layer1": {"class_name": "Linear", "traced_tag": "/TestModel/Linear"},
            "layer2": {"class_name": "Linear", "traced_tag": "/TestModel/Linear.1"}
        }
        monitor.update(ExportStep.HIERARCHY)
        
        # Simple tagging
        monitor.data.total_nodes = 10
        monitor.data.tagged_nodes = {
            "/layer1/MatMul": "/TestModel/Linear",
            "/layer2/MatMul": "/TestModel/Linear.1",
            "/Add": "/TestModel"
        }
        monitor.data.tagging_stats = {
            "direct_matches": 2,
            "parent_matches": 0,
            "root_fallbacks": 1,
            "empty_tags": 0
        }
        monitor.update(ExportStep.NODE_TAGGING)
        
        monitor.data.onnx_size_mb = 0.5
        monitor.data.export_time = 0.1
    
    print(f"✓ Generated minimal files in {output_dir}/:")
    print(f"  - minimal-test_metadata.json")
    print(f"  - minimal-test_report.txt")


def main():
    """Generate all test outputs."""
    print("Generating ExportMonitor test outputs for review...\n")
    
    # Generate bert-tiny outputs
    metadata_path, report_path = generate_bert_tiny_outputs()
    
    # Generate minimal outputs
    generate_minimal_outputs()
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("Generated test outputs in test_outputs/ directory:")
    print("  1. bert-tiny_metadata.json - Full metadata for bert-tiny export")
    print("  2. bert-tiny_report.txt - Full text report for bert-tiny export")
    print("  3. minimal-test_metadata.json - Minimal test metadata")
    print("  4. minimal-test_report.txt - Minimal test report")
    print("\nYou can now review these files to see the actual output format and content.")


if __name__ == "__main__":
    main()