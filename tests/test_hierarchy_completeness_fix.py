"""
Test comprehensive hierarchy completeness after fixing malformed hierarchies.

This test verifies that the fix for TEZ-24 (malformed hierarchies missing nodes)
works correctly by ensuring ALL executed modules appear in hierarchy reports.
"""

import pytest
from pathlib import Path

from modelexport.strategies.htp_new.htp_exporter import HTPExporter
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder


class TestHierarchyCompletenessFix:
    """Test that hierarchy reports now include ALL executed modules."""

    def test_tracing_hierarchy_builder_includes_all_modules(self):
        """Test that TracingHierarchyBuilder now captures ALL modules including torch.nn."""
        # This directly tests the fix: should_create_hierarchy_level() returns True for all modules
        from transformers import AutoModel
        import torch
        
        model_name = "prajjwal1/bert-tiny"
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Create example inputs
        inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        }
        
        # Build hierarchy with the fixed logic
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, inputs)
        
        hierarchy = builder.get_complete_hierarchy()
        
        # Verify completeness - should include torch.nn modules now
        module_names = list(hierarchy.keys())
        class_names = [info["class_name"] for info in hierarchy.values()]
        
        # Before fix: Only ~18 modules (just HuggingFace classes)  
        # After fix: Should have significantly more modules including torch.nn classes
        assert len(hierarchy) > 25, f"Expected >25 modules, got {len(hierarchy)}"
        
        # Verify torch.nn modules are now included (these were missing before)
        expected_torch_nn_classes = ["LayerNorm", "Linear", "Dropout", "Embedding"]
        found_torch_nn_classes = [cls for cls in class_names if cls in expected_torch_nn_classes]
        
        assert len(found_torch_nn_classes) > 0, (
            f"Expected to find torch.nn classes {expected_torch_nn_classes} in hierarchy, "
            f"but only found: {found_torch_nn_classes}. All classes: {class_names}"
        )
        
        # Verify hierarchy is properly structured
        for module_name, info in hierarchy.items():
            assert "traced_tag" in info, f"Missing traced_tag for {module_name}"
            assert info["traced_tag"].startswith("/"), f"Invalid tag format: {info['traced_tag']}"
            assert "class_name" in info, f"Missing class_name for {module_name}"
            assert "module_type" in info, f"Missing module_type for {module_name}"
        
        print(f"✅ Hierarchy now includes {len(hierarchy)} modules (vs ~18 before fix)")
        print(f"✅ Found torch.nn classes: {found_torch_nn_classes}")

    def test_htp_export_generates_complete_reports(self, tmp_path):
        """Test that HTP export now generates complete hierarchy reports."""
        model_name = "prajjwal1/bert-tiny"
        output_path = tmp_path / "bert_complete.onnx"
        
        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)
        
        stats = exporter.export(
            model=None,  # Auto-load
            output_path=str(output_path),
            model_name_or_path=model_name,
        )
        
        # Verify export completed successfully
        assert output_path.exists(), "ONNX model should be created"
        assert stats["coverage_percentage"] == 100.0, "Should have 100% coverage"
        
        # Check metadata file contains complete hierarchy
        metadata_path = output_path.with_name(output_path.stem + "_htp_metadata.json")
        assert metadata_path.exists(), "Metadata file should be created"
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Verify hierarchy completeness in metadata
        assert "modules" in metadata, "Metadata should contain modules"
        modules = metadata["modules"]
        
        # Helper function to count all modules in hierarchical structure
        def count_all_modules(module_dict):
            count = 1  # Count this module
            if "children" in module_dict:
                for child in module_dict["children"].values():
                    count += count_all_modules(child)
            return count
        
        # Helper function to extract all class names from hierarchy
        def get_all_class_names(module_dict, class_names=None):
            if class_names is None:
                class_names = []
            class_names.append(module_dict.get("class_name", ""))
            if "children" in module_dict:
                for child in module_dict["children"].values():
                    get_all_class_names(child, class_names)
            return class_names
        
        # Count total modules in hierarchical structure
        total_modules = count_all_modules(modules)
        
        # Should now have significantly more modules
        assert total_modules > 25, f"Expected >25 modules in metadata, got {total_modules}"
        
        # Get all class names from hierarchical structure
        all_class_names = get_all_class_names(modules)
        
        # Verify torch.nn modules are included in metadata
        expected_torch_nn_classes = ["LayerNorm", "Linear", "Dropout", "Embedding"]
        found_torch_nn_classes = [cls for cls in all_class_names if cls in expected_torch_nn_classes]
        
        assert len(found_torch_nn_classes) > 0, (
            f"Metadata should include torch.nn classes {expected_torch_nn_classes}, "
            f"but only found: {found_torch_nn_classes}"
        )
        
        # Check report file if it was generated
        report_path = output_path.with_name(output_path.stem + "_htp_export_report.md")
        if report_path.exists():
            report_content = report_path.read_text()
            
            # Report should mention the torch.nn classes
            for torch_nn_class in found_torch_nn_classes:
                assert torch_nn_class in report_content, (
                    f"Report should mention {torch_nn_class} class"
                )
        
        print(f"✅ Export metadata now includes {total_modules} modules")
        print(f"✅ Metadata contains torch.nn classes: {found_torch_nn_classes}")

    def test_hierarchy_contains_expected_bert_structure(self):
        """Test that hierarchy contains the expected complete BERT structure."""
        from transformers import AutoModel
        import torch
        
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        model.eval()
        
        inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        }
        
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, inputs)
        hierarchy = builder.get_complete_hierarchy()
        
        # Expected structure elements that should now be present
        expected_patterns = [
            # HuggingFace components (were present before)
            "BertModel",
            "BertEmbeddings", 
            "BertEncoder",
            "BertLayer",
            "BertAttention",
            
            # torch.nn components (were missing before fix)
            "LayerNorm",
            "Linear", 
            "Dropout",
            "Embedding",
        ]
        
        class_names = [info["class_name"] for info in hierarchy.values()]
        
        missing_patterns = []
        for pattern in expected_patterns:
            if not any(pattern in cls for cls in class_names):
                missing_patterns.append(pattern)
        
        assert len(missing_patterns) == 0, (
            f"Missing expected patterns in hierarchy: {missing_patterns}. "
            f"Found classes: {class_names}"
        )
        
        # Verify hierarchical relationships are preserved
        tagged_paths = [info["traced_tag"] for info in hierarchy.values()]
        
        # Should have nested paths like /BertModel/BertEmbeddings/LayerNorm
        nested_paths = [path for path in tagged_paths if path.count("/") > 2]
        assert len(nested_paths) > 0, f"Expected nested hierarchy paths, got: {tagged_paths}"
        
        print(f"✅ Complete BERT hierarchy with {len(hierarchy)} modules")
        print(f"✅ All expected patterns found: {expected_patterns}")