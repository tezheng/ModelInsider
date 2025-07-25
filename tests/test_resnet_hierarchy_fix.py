"""
Test ResNet hierarchy fix - validate compound pattern handling.

This test specifically validates that ResNetEncoder and similar compound
patterns (layer.0, layer.1, etc.) are properly detected as hierarchical 
relationships, fixing the malformed hierarchy issue.
"""

import pytest
from pathlib import Path

from modelexport.core.hierarchy_utils import find_immediate_children, build_ascii_tree


class TestResNetHierarchyFix:
    """Test that ResNet hierarchy patterns are properly handled."""

    def test_compound_pattern_detection(self):
        """Test that compound patterns like encoder.layer.0 are detected as immediate children."""
        # Mock ResNet-like hierarchy data
        hierarchy = {
            "": {"class_name": "ResNetModel"},
            "embeddings": {"class_name": "ResNetEmbeddings"},
            "encoder": {"class_name": "ResNetEncoder"},
            "encoder.layer.0": {"class_name": "ResNetLayer"},
            "encoder.layer.1": {"class_name": "ResNetLayer"},
            "encoder.layer.2": {"class_name": "ResNetLayer"},
            "encoder.layer.0.conv1": {"class_name": "Conv2d"},
            "encoder.layer.0.bn1": {"class_name": "BatchNorm2d"},
            "encoder.layer.1.conv1": {"class_name": "Conv2d"},
            "pooler": {"class_name": "AdaptiveAvgPool2d"},
        }
        
        # Test root children
        root_children = find_immediate_children("", hierarchy)
        expected_root = ["embeddings", "encoder", "pooler"]
        assert root_children == expected_root, f"Expected {expected_root}, got {root_children}"
        
        # Test encoder children - THIS IS THE KEY TEST
        # Before fix: encoder.layer.0 would NOT be detected as immediate child of encoder
        # After fix: encoder.layer.0 SHOULD be detected due to compound pattern handling
        encoder_children = find_immediate_children("encoder", hierarchy)
        expected_encoder = ["encoder.layer.0", "encoder.layer.1", "encoder.layer.2"]
        assert encoder_children == expected_encoder, (
            f"Expected encoder children {expected_encoder}, got {encoder_children}. "
            "This indicates compound pattern detection is broken."
        )
        
        # Test layer.0 children (simple case)
        layer0_children = find_immediate_children("encoder.layer.0", hierarchy) 
        expected_layer0 = ["encoder.layer.0.bn1", "encoder.layer.0.conv1"]  # Alphabetical order
        assert layer0_children == expected_layer0, f"Expected {expected_layer0}, got {layer0_children}"
        
        print("✅ Compound pattern detection working correctly")
        print(f"✅ ResNetEncoder now has children: {encoder_children}")

    def test_resnet_ascii_tree_generation(self):
        """Test that ResNet hierarchy generates proper ASCII tree."""
        # More complete ResNet-like hierarchy
        hierarchy = {
            "": {"class_name": "ResNetModel"},
            "embeddings": {"class_name": "ResNetEmbeddings"},
            "embeddings.conv": {"class_name": "Conv2d"},
            "embeddings.bn": {"class_name": "BatchNorm2d"},
            "encoder": {"class_name": "ResNetEncoder"},
            "encoder.layer.0": {"class_name": "ResNetLayer"},
            "encoder.layer.1": {"class_name": "ResNetLayer"},
            "encoder.layer.0.conv1": {"class_name": "Conv2d"},
            "encoder.layer.0.bn1": {"class_name": "BatchNorm2d"},
            "encoder.layer.1.conv1": {"class_name": "Conv2d"},
            "pooler": {"class_name": "AdaptiveAvgPool2d"},
        }
        
        # Generate ASCII tree
        tree_lines = build_ascii_tree(hierarchy)
        tree_text = "\n".join(tree_lines)
        
        print("Generated ResNet hierarchy tree:")
        print(tree_text)
        
        # Validate key structural elements
        assert "ResNetModel" in tree_text, "Missing root ResNetModel"
        assert "ResNetEncoder: encoder" in tree_text, "Missing ResNetEncoder"
        assert "ResNetLayer: encoder.layer.0" in tree_text, "Missing layer.0 under encoder"
        assert "ResNetLayer: encoder.layer.1" in tree_text, "Missing layer.1 under encoder" 
        
        # Key test: encoder should NOT be a leaf node
        lines = tree_text.split('\n')
        encoder_line_idx = None
        for i, line in enumerate(lines):
            if "ResNetEncoder: encoder" in line:
                encoder_line_idx = i
                break
        
        assert encoder_line_idx is not None, "Could not find ResNetEncoder line"
        
        # Check that there are child lines after encoder (indented with ├── or └──)
        has_children = False
        for i in range(encoder_line_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            # If next non-empty line is more indented, encoder has children
            if line.startswith("│") or line.startswith("├") or line.startswith("└"):
                # Check if it's a direct child by looking for ResNetLayer
                if "ResNetLayer" in line:
                    has_children = True
                    break
            else:
                # Hit a sibling or parent, stop looking
                break
        
        assert has_children, (
            f"ResNetEncoder appears to be a leaf node! Tree:\n{tree_text}\n"
            "This indicates the hierarchy fix is not working."
        )
        
        print("✅ ResNet hierarchy tree structure is correct")
        print("✅ ResNetEncoder properly shows children (not a leaf node)")

    def test_numeric_sorting(self):
        """Test that numeric parts are sorted correctly (0, 1, 2... not 0, 10, 11, 2)."""
        hierarchy = {
            "": {"class_name": "Model"},
            "encoder": {"class_name": "Encoder"},
            "encoder.layer.0": {"class_name": "Layer"},
            "encoder.layer.1": {"class_name": "Layer"}, 
            "encoder.layer.2": {"class_name": "Layer"},
            "encoder.layer.10": {"class_name": "Layer"},
            "encoder.layer.11": {"class_name": "Layer"},
        }
        
        children = find_immediate_children("encoder", hierarchy)
        expected_order = [
            "encoder.layer.0",
            "encoder.layer.1", 
            "encoder.layer.2",
            "encoder.layer.10",
            "encoder.layer.11"
        ]
        
        assert children == expected_order, (
            f"Expected numeric sorting {expected_order}, got {children}"
        )
        
        print("✅ Numeric sorting works correctly")
        
    def test_mixed_patterns(self):
        """Test handling of mixed simple and compound patterns."""
        hierarchy = {
            "": {"class_name": "Model"},
            "embeddings": {"class_name": "Embeddings"},          # Simple child
            "encoder": {"class_name": "Encoder"},                # Simple child
            "encoder.attention": {"class_name": "Attention"},    # Simple grandchild
            "encoder.layer.0": {"class_name": "Layer"},          # Compound grandchild
            "encoder.layer.1": {"class_name": "Layer"},          # Compound grandchild
            "pooler": {"class_name": "Pooler"},                  # Simple child
        }
        
        # Root should have simple children
        root_children = find_immediate_children("", hierarchy)
        assert root_children == ["embeddings", "encoder", "pooler"]
        
        # Encoder should have both simple and compound children  
        encoder_children = find_immediate_children("encoder", hierarchy)
        expected = ["encoder.attention", "encoder.layer.0", "encoder.layer.1"]
        assert encoder_children == expected, f"Expected {expected}, got {encoder_children}"
        
        print("✅ Mixed simple/compound patterns handled correctly")

    @pytest.mark.skipif(True, reason="Requires transformers with ResNet models - enable for full testing")
    def test_real_resnet_model_export(self, tmp_path):
        """Test actual ResNet model export (requires model download)."""
        # This would test with a real ResNet model from transformers
        # Skipped by default to avoid model downloads in CI
        from transformers import AutoModel
        from modelexport.strategies.htp_new.htp_exporter import HTPExporter
        
        model_name = "microsoft/resnet-50"  # Example ResNet model
        output_path = tmp_path / "resnet.onnx"
        
        exporter = HTPExporter(verbose=False, enable_reporting=True)
        
        stats = exporter.export(
            model=None,
            output_path=str(output_path),
            model_name_or_path=model_name,
        )
        
        # Check that ResNet hierarchy is not malformed
        metadata_path = output_path.with_name(output_path.stem + "_htp_metadata.json")
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # ResNet should have encoder with multiple layers
        modules = metadata.get("modules", {})
        encoder_children = [
            path for path in modules.keys() 
            if path.startswith("encoder.") and path.count(".") == 2
        ]
        
        assert len(encoder_children) > 0, (
            f"ResNet encoder should have layer children, but found: {list(modules.keys())}"
        )
        
        print(f"✅ Real ResNet export successful with {len(encoder_children)} encoder layers")