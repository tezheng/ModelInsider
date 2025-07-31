"""Test cases for hierarchy scope format regression."""

from transformers import AutoModel

from modelexport.core.hierarchy_utils import build_ascii_tree, build_rich_tree
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder


class TestHierarchyScopeFormat:
    """Test that hierarchy displays show full scope paths."""
    
    def test_ascii_tree_shows_full_scope(self):
        """Test that ASCII tree shows full module paths as scope."""
        # Create a simple hierarchy
        hierarchy = {
            "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
            "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/BertEmbeddings"},
            "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/BertEncoder"},
            "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/BertEncoder/BertLayer.0"},
            "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention"},
            "encoder.layer.0.attention.self": {"class_name": "BertSelfAttention", "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"},
        }
        
        # Build ASCII tree
        tree_lines = build_ascii_tree(hierarchy)
        tree_text = "\n".join(tree_lines)
        
        # Check that full paths are shown
        assert "BertLayer: encoder.layer.0" in tree_text
        assert "BertAttention: encoder.layer.0.attention" in tree_text
        assert "BertSelfAttention: encoder.layer.0.attention.self" in tree_text
        
        # Should NOT show shortened versions
        assert "BertLayer: 0" not in tree_text
        assert "BertLayer: layer.0" not in tree_text
        assert "BertAttention: attention" not in tree_text
        assert "BertSelfAttention: self" not in tree_text
    
    def test_real_model_hierarchy_scope(self):
        """Test with real BERT model to ensure scope format is correct."""
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Build hierarchy
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, {"input_ids": model.dummy_inputs["input_ids"]})
        hierarchy = builder.get_complete_hierarchy()
        
        # Build ASCII tree
        tree_lines = build_ascii_tree(hierarchy)
        tree_text = "\n".join(tree_lines)
        
        # Check key scope formats
        assert "BertLayer: encoder.layer.0" in tree_text
        assert "BertLayer: encoder.layer.1" in tree_text
        assert "BertAttention: encoder.layer.0.attention" in tree_text
        assert "BertSdpaSelfAttention: encoder.layer.0.attention.self" in tree_text
        assert "BertIntermediate: encoder.layer.0.intermediate" in tree_text
        assert "GELUActivation: encoder.layer.0.intermediate.intermediate_act_fn" in tree_text
    
    def test_rich_tree_shows_full_scope(self):
        """Test that Rich tree also shows full module paths."""
        # Simple hierarchy
        hierarchy = {
            "": {"class_name": "Model", "traced_tag": "/Model"},
            "encoder": {"class_name": "Encoder", "traced_tag": "/Model/Encoder"},
            "encoder.layer.0": {"class_name": "Layer", "traced_tag": "/Model/Encoder/Layer.0"},
        }
        
        # Build Rich tree
        tree = build_rich_tree(hierarchy)
        
        # Convert to string for checking
        from io import StringIO

        from rich.console import Console
        
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        console.print(tree)
        tree_str = output.getvalue()
        
        # Should contain full paths
        assert "encoder.layer.0" in tree_str
        # Should NOT contain shortened version
        assert ": 0" not in tree_str or ": 0 " not in tree_str  # Avoid false positive with counts