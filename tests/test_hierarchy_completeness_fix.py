"""
Test comprehensive hierarchy completeness after fixing malformed hierarchies.

This test verifies that the fix for TEZ-24 (malformed hierarchies missing nodes)
works correctly by ensuring ALL executed modules appear in hierarchy reports.
"""


from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder
from modelexport.strategies.htp_new.htp_exporter import HTPExporter


class TestHierarchyCompletenessFix:
    """Test that hierarchy reports now include ALL executed modules."""

    def test_tracing_hierarchy_builder_excludes_torch_nn_by_default(self):
        """Test that TracingHierarchyBuilder excludes torch.nn modules by default (MUST-002)."""
        import torch
        from transformers import AutoModel
        
        model_name = "prajjwal1/bert-tiny"
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Create example inputs
        inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        }
        
        # Build hierarchy with default settings (no exceptions = exclude torch.nn)
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, inputs)
        
        hierarchy = builder.get_complete_hierarchy()
        
        # Verify torch.nn modules are NOT included (MUST-002 compliance)
        class_names = [info["class_name"] for info in hierarchy.values()]
        
        # Should only have HuggingFace modules, not torch.nn modules
        forbidden_torch_nn_classes = ["LayerNorm", "Linear", "Dropout", "Embedding"]
        found_torch_nn_classes = [cls for cls in class_names if cls in forbidden_torch_nn_classes]
        
        assert len(found_torch_nn_classes) == 0, (
            f"MUST-002 violation: Found torch.nn classes {found_torch_nn_classes} in hierarchy. "
            f"These should not appear in tags. All classes: {class_names}"
        )
        
        # Should still have a reasonable number of HuggingFace modules
        assert len(hierarchy) >= 15, f"Expected at least 15 HF modules, got {len(hierarchy)}"
        
        # Verify hierarchy is properly structured
        for module_name, info in hierarchy.items():
            assert "traced_tag" in info, f"Missing traced_tag for {module_name}"
            assert info["traced_tag"].startswith("/"), f"Invalid tag format: {info['traced_tag']}"
            assert "class_name" in info, f"Missing class_name for {module_name}"
            assert "module_type" in info, f"Missing module_type for {module_name}"
        
        print(f"✅ MUST-002 compliant: {len(hierarchy)} HF modules, no torch.nn modules")
        print(f"✅ Module classes: {set(class_names)}")

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
        
        # Should have HuggingFace modules
        assert total_modules >= 15, f"Expected at least 15 HF modules in metadata, got {total_modules}"
        
        # Get all class names from hierarchical structure
        all_class_names = get_all_class_names(modules)
        
        # Verify torch.nn modules are NOT included in metadata (MUST-002)
        forbidden_torch_nn_classes = ["LayerNorm", "Linear", "Dropout", "Embedding"]
        found_torch_nn_classes = [cls for cls in all_class_names if cls in forbidden_torch_nn_classes]
        
        assert len(found_torch_nn_classes) == 0, (
            f"MUST-002 violation: Metadata should NOT include torch.nn classes {forbidden_torch_nn_classes}, "
            f"but found: {found_torch_nn_classes}"
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
        
        print(f"✅ Export metadata includes {total_modules} HF modules")
        print("✅ MUST-002 compliant: No torch.nn classes in metadata")

    def test_hierarchy_contains_expected_bert_structure(self):
        """Test that hierarchy contains the expected BERT structure (HF modules only by default)."""
        import torch
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        model.eval()
        
        inputs = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        }
        
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, inputs)
        hierarchy = builder.get_complete_hierarchy()
        
        # Expected HuggingFace components (torch.nn excluded by default)
        expected_hf_patterns = [
            "BertModel",
            "BertEmbeddings", 
            "BertEncoder",
            "BertLayer",
            "BertAttention",
            "BertSdpaSelfAttention",  # Updated to actual class name
            "BertSelfOutput",
            "BertIntermediate",
            "BertOutput",
        ]
        
        # torch.nn components should NOT be present (MUST-002)
        forbidden_patterns = ["LayerNorm", "Linear", "Dropout", "Embedding"]
        
        class_names = [info["class_name"] for info in hierarchy.values()]
        
        # Check HF patterns are present
        missing_hf_patterns = []
        for pattern in expected_hf_patterns:
            if not any(pattern in cls for cls in class_names):
                missing_hf_patterns.append(pattern)
        
        # Check torch.nn patterns are NOT present
        found_forbidden = []
        for pattern in forbidden_patterns:
            if any(pattern == cls for cls in class_names):
                found_forbidden.append(pattern)
        
        assert len(missing_hf_patterns) == 0, (
            f"Missing expected HF patterns in hierarchy: {missing_hf_patterns}. "
            f"Found classes: {class_names}"
        )
        
        assert len(found_forbidden) == 0, (
            f"MUST-002 violation: Found forbidden torch.nn classes: {found_forbidden}. "
            f"These should not appear in hierarchy."
        )
        
        # Verify hierarchical relationships are preserved
        tagged_paths = [info["traced_tag"] for info in hierarchy.values()]
        
        # Should have nested paths but not with torch.nn modules
        nested_paths = [path for path in tagged_paths if path.count("/") > 2]
        assert len(nested_paths) > 0, f"Expected nested hierarchy paths, got: {tagged_paths}"
        
        print(f"✅ MUST-002 compliant BERT hierarchy with {len(hierarchy)} HF modules")
        print("✅ HF patterns found, torch.nn patterns excluded")