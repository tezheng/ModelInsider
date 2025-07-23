"""
Test the ExportMonitor system with step-aware writers.
"""

import time
from pathlib import Path

from export_monitor import (
    ConsoleWriter,
    ExportData,
    ExportMonitor,
    ExportStep,
    step,
)


def test_basic_monitor():
    """Test basic ExportMonitor functionality."""
    print("Testing basic ExportMonitor...")
    
    with ExportMonitor("test_model.onnx", verbose=True, enable_report=True) as monitor:
        # Step 1: Model preparation
        monitor.update(
            ExportStep.MODEL_PREP,
            model_name="prajjwal1/bert-tiny",
            model_class="BertModel",
            total_modules=48,
            total_parameters=4400000
        )
        
        time.sleep(0.1)  # Simulate work
        
        # Step 2: Input generation
        monitor.update(
            ExportStep.INPUT_GEN,
            inputs={
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            },
            model_type="bert",
            task="feature-extraction"
        )
        
        # Store in steps for step-specific data
        monitor.data.steps["input_generation"] = {
            "model_type": "bert",
            "task": "feature-extraction",
            "inputs": {
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            }
        }
        
        time.sleep(0.1)
        
        # Step 3: Hierarchy building
        hierarchy = {
            "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
            "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/BertEmbeddings"},
            "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
            "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
            "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"},
            "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"},
            "pooler": {"class_name": "BertPooler", "traced_tag": "/BertModel/BertPooler"}
        }
        
        monitor.update(
            ExportStep.HIERARCHY,
            hierarchy=hierarchy
        )
        
        time.sleep(0.1)
        
        # Step 6: Node tagging
        monitor.update(
            ExportStep.NODE_TAGGING,
            total_nodes=136,
            tagged_nodes={
                "/Cast": "/BertModel",
                "/embeddings/Add": "/BertModel/BertEmbeddings",
                "/encoder/layer.0/attention/self/MatMul": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
                "/pooler/Gather": "/BertModel/BertPooler",
                # ... more nodes
            },
            tagging_stats={
                "direct_matches": 83,
                "parent_matches": 34,
                "root_fallbacks": 19,
                "empty_tags": 0
            }
        )
        
        # Add fake file size
        monitor.data.onnx_size_mb = 16.76
    
    print("✅ Basic monitor test completed")
    print("   - Metadata written to: test_model_metadata.json")
    print("   - Report written to: test_model_report.txt")


def test_step_specific_handlers():
    """Test that step-specific handlers work correctly."""
    print("\nTesting step-specific handlers...")
    
    class TestWriter(ConsoleWriter):
        def __init__(self):
            super().__init__()
            self.handled_steps = []
        
        # Override the parent's hierarchy handler
        @step(ExportStep.HIERARCHY)
        def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
            self.handled_steps.append("custom_hierarchy")
            print("   ✓ Custom hierarchy handler called")
            return 1
        
        def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
            self.handled_steps.append(f"default_{export_step.value}")
            print(f"   ✓ Default handler for {export_step.value}")
            return 1
    
    writer = TestWriter()
    data = ExportData()
    
    # Test custom handler
    writer.write(ExportStep.HIERARCHY, data)
    assert "custom_hierarchy" in writer.handled_steps
    
    # Test default handler
    writer.write(ExportStep.STRUCTURE, data)
    assert "default_structure_analysis" in writer.handled_steps
    
    print("✅ Step-specific handler test passed")


def test_conditional_writers():
    """Test conditional writer creation."""
    print("\nTesting conditional writer creation...")
    
    # Test with all writers
    monitor1 = ExportMonitor("test.onnx", verbose=True, enable_report=True)
    assert len(monitor1.writers) == 3  # metadata, console, report
    print("   ✓ All writers: 3 writers created")
    
    # Test without console
    monitor2 = ExportMonitor("test.onnx", verbose=False, enable_report=True)
    assert len(monitor2.writers) == 2  # metadata, report
    print("   ✓ No console: 2 writers created")
    
    # Test minimal (metadata only)
    monitor3 = ExportMonitor("test.onnx", verbose=False, enable_report=False)
    assert len(monitor3.writers) == 1  # metadata only
    print("   ✓ Minimal: 1 writer created")
    
    print("✅ Conditional writer test passed")


def test_data_accumulation():
    """Test that data accumulates correctly."""
    print("\nTesting data accumulation...")
    
    monitor = ExportMonitor("test.onnx", verbose=False)
    
    # Add data in stages
    monitor.update(ExportStep.MODEL_PREP, model_name="test-model")
    assert monitor.data.model_name == "test-model"
    
    monitor.update(ExportStep.MODEL_PREP, model_class="TestModel")
    assert monitor.data.model_name == "test-model"  # Preserved
    assert monitor.data.model_class == "TestModel"  # Added
    
    # Test step-specific data
    monitor.update(ExportStep.INPUT_GEN, custom_field="custom_value")
    assert "input_generation" in monitor.data.steps
    assert monitor.data.steps["input_generation"]["custom_field"] == "custom_value"
    
    print("✅ Data accumulation test passed")


def test_timing_tracking():
    """Test that timing is tracked correctly."""
    print("\nTesting timing tracking...")
    
    monitor = ExportMonitor("test.onnx", verbose=False)
    
    # Simulate steps with delays
    monitor.update(ExportStep.MODEL_PREP)
    time.sleep(0.1)
    
    monitor.update(ExportStep.HIERARCHY)
    time.sleep(0.1)
    
    # Check timing
    assert ExportStep.MODEL_PREP.value in monitor.data.step_times
    assert ExportStep.HIERARCHY.value in monitor.data.step_times
    
    # Check elapsed time
    assert monitor.data.elapsed_time > 0.2
    
    print("✅ Timing tracking test passed")


def test_real_vs_buffered():
    """Demonstrate real-time console vs buffered file output."""
    print("\nDemonstrating real-time vs buffered output...")
    
    print("   Console output appears immediately:")
    with ExportMonitor("demo.onnx", verbose=True, enable_report=True) as monitor:
        for i, step_type in enumerate([ExportStep.MODEL_PREP, ExportStep.INPUT_GEN]):
            print(f"\n   [Simulating {step_type.value}...]")
            monitor.update(step_type, progress=i+1)
            time.sleep(0.5)  # Visible delay
    
    print("\n   File output written at the end:")
    print(f"   - Metadata: {Path('demo_metadata.json').exists()}")
    print(f"   - Report: {Path('demo_report.txt').exists()}")
    
    print("✅ Real-time demonstration completed")


def test_htp_integration_example():
    """Show how this would integrate with HTP exporter."""
    print("\nHTP Integration Example:")
    print("-" * 50)
    
    print("""
# In htp_exporter.py:

def export(self, model, dummy_inputs, output_path):
    with ExportMonitor(output_path, self.verbose, self.enable_report) as monitor:
        # Model preparation
        monitor.update(
            ExportStep.MODEL_PREP,
            model_name=self.model_name,
            model_class=model.__class__.__name__,
            total_modules=len(list(model.modules())),
            total_parameters=sum(p.numel() for p in model.parameters())
        )
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(model)
        monitor.update(ExportStep.HIERARCHY, hierarchy=hierarchy)
        
        # Export to ONNX
        onnx_model = self._export_onnx(model, dummy_inputs)
        monitor.update(ExportStep.CONVERSION, onnx_nodes=len(onnx_model.graph.node))
        
        # Tag nodes
        tagged = self._tag_nodes(onnx_model, hierarchy)
        monitor.update(
            ExportStep.NODE_TAGGING,
            total_nodes=len(onnx_model.graph.node),
            tagged_nodes=tagged,
            tagging_stats=self._calculate_stats(tagged)
        )
        
        # Save ONNX
        onnx.save(onnx_model, output_path)
        monitor.update(
            ExportStep.COMPLETE,
            onnx_size_mb=Path(output_path).stat().st_size / 1e6
        )
    """)
    
    print("\n✅ Integration example shown")


def run_all_tests():
    """Run all tests."""
    print("Running ExportMonitor Tests...\n")
    
    test_basic_monitor()
    test_step_specific_handlers()
    test_conditional_writers()
    test_data_accumulation()
    test_timing_tracking()
    test_real_vs_buffered()
    test_htp_integration_example()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)
    
    print("\nKey Features Demonstrated:")
    print("1. @step decorator for step-specific handling")
    print("2. Real-time console output with Rich formatting")
    print("3. Buffered metadata/report writing")
    print("4. Conditional writer creation")
    print("5. Data accumulation and timing tracking")
    print("6. Clean integration with HTP exporter")
    print("7. Context manager for automatic finalization")


if __name__ == "__main__":
    run_all_tests()