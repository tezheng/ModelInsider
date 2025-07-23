"""
Test the Pythonic report generator.
"""

import re

from report_generator import (
    ExportData,
    Format,
    ReportContext,
    ReportGenerator,
    make_header,
    truncate_lines,
)


def create_test_data() -> ExportData:
    """Create test data."""
    data = ExportData(
        model_name="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4400000,
        output_path="bert-tiny.onnx",
        onnx_size_mb=16.76,
        export_time=7.72,
        total_nodes=136,
        tagged_count=136,
        direct_matches=83,
        parent_matches=34,
        root_fallbacks=19,
        empty_tags=0
    )
    
    # Add some hierarchy
    data.hierarchy = {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/BertEmbeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
        "pooler": {"class_name": "BertPooler", "traced_tag": "/BertModel/BertPooler"}
    }
    
    # Add some nodes
    data.tagged_nodes = {
        "/Cast": "/BertModel",
        "/embeddings/Add": "/BertModel/BertEmbeddings",
        "/encoder/layer.0/MatMul": "/BertModel/BertEncoder/BertLayer.0"
    }
    
    # Add steps
    data.steps["input_generation"] = {
        "model_type": "bert",
        "task": "feature-extraction"
    }
    
    return data


def test_basic_generation():
    """Test basic report generation."""
    print("Testing basic generation...")
    
    data = create_test_data()
    generator = ReportGenerator(data)
    
    # Test console format
    console_output = generator.generate(Format.CONSOLE)
    clean = re.sub(r'\x1b\[[0-9;]*m', '', console_output)
    
    assert "Model loaded: BertModel" in clean
    assert "Coverage: 100.0%" in clean
    assert "Direct matches: 83" in clean
    
    # Test text format
    text_output = generator.generate(Format.TEXT)
    assert "HTP EXPORT FULL REPORT" in text_output
    assert "MODULE HIERARCHY" in text_output
    assert "STATISTICS" in text_output
    
    # Test JSON format
    json_output = generator.generate(Format.JSON)
    assert json_output["model"]["class"] == "BertModel"
    assert json_output["report"]["node_tagging"]["statistics"]["direct_matches"] == 83
    
    print("✅ Basic generation passed")


def test_properties():
    """Test data properties."""
    print("\nTesting data properties...")
    
    data = create_test_data()
    
    # Test coverage calculation
    assert data.coverage == 100.0
    
    # Test with partial coverage
    data.tagged_count = 100
    assert data.coverage == (100/136 * 100)
    
    print("✅ Properties test passed")


def test_context_manager():
    """Test report context manager."""
    print("\nTesting context manager...")
    
    data = create_test_data()
    
    # This would auto-save all formats on exit
    with ReportContext(data, "test_output.onnx") as ctx:
        console = ctx.generate(Format.CONSOLE)
        text = ctx.generate(Format.TEXT)
        json_data = ctx.generate(Format.JSON)
        
        assert console is not None
        assert text is not None
        assert json_data is not None
    
    print("✅ Context manager test passed")


def test_helper_functions():
    """Test helper functions."""
    print("\nTesting helper functions...")
    
    # Test header creation
    header = make_header("TEST SECTION")
    assert "TEST SECTION" in header
    assert "=" * 80 in header
    
    # Test line truncation
    long_text = "\n".join([f"Line {i}" for i in range(50)])
    truncated = truncate_lines(long_text, max_lines=10)
    
    lines = truncated.splitlines()
    assert len(lines) == 11  # 10 + truncation message
    assert "40 more lines" in lines[-1]
    
    print("✅ Helper functions test passed")


def test_pythonic_features():
    """Test Pythonic features."""
    print("\nTesting Pythonic features...")
    
    data = create_test_data()
    generator = ReportGenerator(data)
    
    # Duck typing - the generator works with any format
    for format in Format:
        output = generator.generate(format)
        assert output is not None
    
    # Dictionary-based dispatch
    assert len(generator._formatters) == 3
    
    # Simple data access
    assert data.model_class == "BertModel"
    assert len(data.hierarchy) == 5
    
    print("✅ Pythonic features test passed")


def test_real_world_usage():
    """Test real-world usage pattern."""
    print("\nTesting real-world usage...")
    
    # In actual HTP exporter, you'd do:
    data = ExportData()
    
    # During export, update data directly
    data.model_name = "bert-base-uncased"
    data.model_class = "BertModel"
    data.total_modules = 199
    
    # After tagging
    data.total_nodes = 500
    data.tagged_count = 485
    data.direct_matches = 400
    data.parent_matches = 85
    
    # Generate reports
    gen = ReportGenerator(data)
    
    # For console (verbose mode)
    if True:  # if verbose
        console = gen.generate(Format.CONSOLE, truncate=True)
        print("Console output length:", len(console))
    
    # Always generate metadata
    metadata = gen.generate(Format.JSON)
    assert metadata["model"]["class"] == "BertModel"
    
    # Optional full report
    full_report = gen.generate(Format.TEXT)
    assert "NODE MAPPINGS" in full_report
    
    print("✅ Real-world usage test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Pythonic Report Generator Tests...\n")
    
    test_basic_generation()
    test_properties()
    test_context_manager()
    test_helper_functions()
    test_pythonic_features()
    test_real_world_usage()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)
    
    print("\nPythonic features demonstrated:")
    print("1. Simple dataclass instead of complex interfaces")
    print("2. Duck typing - no need for explicit interfaces")
    print("3. Dictionary dispatch for format selection")
    print("4. Properties for computed values")
    print("5. Context managers for resource management")
    print("6. Simple functions instead of classes when appropriate")
    print("7. Direct attribute access instead of getters/setters")
    print("8. Single source of truth (ExportData) without abstractions")


if __name__ == "__main__":
    run_all_tests()