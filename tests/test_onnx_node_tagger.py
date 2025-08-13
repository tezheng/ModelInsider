"""
Tests for ONNX Node Tagger

Verifies the corrected implementation follows all CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC
- MUST-002: TORCH.NN FILTERING
- MUST-003: UNIVERSAL DESIGN
"""

from unittest.mock import MagicMock

import pytest

from modelexport.core.onnx_node_tagger import (
    ONNXNodeTagger,
    create_node_tagger_from_hierarchy,
)


class TestONNXNodeTagger:
    """Test suite for ONNX node tagger functionality."""

    @pytest.fixture
    def sample_hierarchy_data(self):
        """Sample hierarchy data from TracingHierarchyBuilder (NO HARDCODED MODEL NAMES)."""
        return {
            "embeddings": {
                "traced_tag": "/TestModel/Embeddings",
                "execution_order": 1,
                "module_type": "huggingface",
            },
            "embeddings.word_embeddings": {
                "traced_tag": "/TestModel/Embeddings/WordEmbeddings",
                "execution_order": 2,
                "module_type": "huggingface",
            },
            "encoder.layer.0.attention.self.query": {
                "traced_tag": "/TestModel/Encoder/Layer.0/Attention/Self/Query",
                "execution_order": 5,
                "module_type": "huggingface",
            },
            "encoder.layer.0.attention.self": {
                "traced_tag": "/TestModel/Encoder/Layer.0/Attention/Self",
                "execution_order": 4,
                "module_type": "huggingface",
            },
            "pooler.dense": {
                "traced_tag": "/TestModel/Pooler/Dense",
                "execution_order": 10,
                "module_type": "huggingface",
            },
        }

    @pytest.fixture
    def different_model_hierarchy_data(self):
        """Different model hierarchy to test NO HARDCODED LOGIC."""
        return {
            "features.0": {
                "traced_tag": "/ResNetModel/Features/Block.0",
                "execution_order": 1,
                "module_type": "custom",
            },
            "features.0.conv1": {
                "traced_tag": "/ResNetModel/Features/Block.0/Conv1",
                "execution_order": 2,
                "module_type": "custom",
            },
            "classifier": {
                "traced_tag": "/ResNetModel/Classifier",
                "execution_order": 8,
                "module_type": "custom",
            },
        }

    @pytest.fixture
    def mock_onnx_nodes(self):
        """Create mock ONNX nodes for testing."""
        nodes = []

        # Node with full scope path
        node1 = MagicMock()
        node1.name = "/embeddings/word_embeddings/Gather"
        node1.op_type = "Gather"
        nodes.append(node1)

        # Node with partial scope path
        node2 = MagicMock()
        node2.name = "/encoder/layer.0/attention/self/query/MatMul"
        node2.op_type = "MatMul"
        nodes.append(node2)

        # Node with unknown scope
        node3 = MagicMock()
        node3.name = "/encoder/layer.0/attention/unknown_module/Add"
        node3.op_type = "Add"
        nodes.append(node3)

        # Root node (no scope)
        node4 = MagicMock()
        node4.name = "/Softmax_123"
        node4.op_type = "Softmax"
        nodes.append(node4)

        # Node with no name
        node5 = MagicMock()
        node5.name = ""
        node5.op_type = "Constant"
        nodes.append(node5)

        return nodes

    @pytest.fixture
    def mock_onnx_model(self, mock_onnx_nodes):
        """Create mock ONNX model with test nodes."""
        model = MagicMock()
        model.graph.node = mock_onnx_nodes
        return model

    def test_no_hardcoded_logic_different_models(self, different_model_hierarchy_data):
        """Test MUST-001: NO HARDCODED LOGIC - works with different model types."""
        tagger = ONNXNodeTagger(different_model_hierarchy_data)

        # Should extract model root dynamically
        assert tagger.model_root_tag == "/ResNetModel"

        # Should work with completely different model structure
        assert "features.0" in tagger.scope_to_tag
        assert tagger.scope_to_tag["features.0"] == "/ResNetModel/Features/Block.0"

    def test_model_root_extraction_dynamic(self, sample_hierarchy_data):
        """Test dynamic model root extraction (NO HARDCODED)."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)

        # Should extract "TestModel" from the hierarchy tags
        assert tagger.model_root_tag == "/TestModel"

    def test_model_root_extraction_empty_hierarchy(self):
        """Test model root extraction with empty hierarchy."""
        tagger = ONNXNodeTagger({})
        assert tagger.model_root_tag == "/UnknownModel"

    def test_scope_extraction_from_nodes(self, sample_hierarchy_data):
        """Test scope extraction from ONNX node names."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)

        # Test cases: (node_name, expected_scope)
        test_cases = [
            ("/embeddings/word_embeddings/Gather", "embeddings.word_embeddings"),
            (
                "/encoder/layer.0/attention/self/query/MatMul",
                "encoder.layer.0.attention.self.query",
            ),
            ("/pooler/dense/Tanh", "pooler.dense"),
            ("/Softmax_123", "__root__"),
            ("MatMul_456", "__root__"),
            ("", "__root__"),
            ("/SingleComponent", "__root__"),
        ]

        for node_name, expected_scope in test_cases:
            node = MagicMock()
            node.name = node_name
            actual_scope = tagger._extract_scope_from_node(node)
            assert actual_scope == expected_scope, f"Failed for {node_name}"

    def test_bucketization_by_scope(self, sample_hierarchy_data, mock_onnx_model):
        """Test ONNX node bucketization by scope."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)
        buckets = tagger.bucketize_nodes_by_scope(mock_onnx_model)

        # Verify buckets created correctly
        assert "embeddings.word_embeddings" in buckets
        assert "encoder.layer.0.attention.self.query" in buckets
        assert "encoder.layer.0.attention.unknown_module" in buckets  # Unknown scope
        assert "__root__" in buckets

        # Verify root bucket contains no-scope nodes
        root_nodes = buckets["__root__"]
        root_node_names = [node.name for node in root_nodes]
        assert "/Softmax_123" in root_node_names
        assert "" in root_node_names  # Empty name node

    def test_priority_1_direct_scope_matching(self, sample_hierarchy_data):
        """Test PRIORITY 1: Direct scope matching."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)

        # Direct match should return exact tag
        tag = tagger._find_tag_for_scope("embeddings.word_embeddings")
        assert tag == "/TestModel/Embeddings/WordEmbeddings"

        tag = tagger._find_tag_for_scope("encoder.layer.0.attention.self.query")
        assert tag == "/TestModel/Encoder/Layer.0/Attention/Self/Query"

    def test_priority_2_parent_scope_matching(self, sample_hierarchy_data):
        """Test PRIORITY 2: Parent scope matching."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)

        # Unknown child scope should match parent
        tag = tagger._find_tag_for_scope("encoder.layer.0.attention.self.unknown_child")
        assert tag == "/TestModel/Encoder/Layer.0/Attention/Self"

        # Multiple level parent matching
        tag = tagger._find_tag_for_scope("embeddings.unknown.deep.path")
        assert tag == "/TestModel/Embeddings"

    def test_priority_3_operation_fallback_disabled(self, sample_hierarchy_data):
        """Test PRIORITY 3 disabled by default."""
        tagger = ONNXNodeTagger(sample_hierarchy_data, enable_operation_fallback=False)

        # Should skip operation-based fallback and go to root
        tag = tagger._find_tag_for_scope("completely.unknown.scope")
        assert tag == "/TestModel"  # Root fallback

    def test_priority_3_operation_fallback_enabled(self, sample_hierarchy_data):
        """Test PRIORITY 3: Operation-based fallback when enabled."""
        tagger = ONNXNodeTagger(sample_hierarchy_data, enable_operation_fallback=True)

        # Should find similar scope
        tag = tagger._find_tag_for_scope("encoder.layer.0.different_module")
        assert tag == "/TestModel/Encoder/Layer.0/Attention/Self"  # Best partial match

    def test_priority_4_root_fallback_never_empty(self, sample_hierarchy_data):
        """Test PRIORITY 4: Root fallback (NEVER EMPTY)."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)

        # Completely unknown scope should fall back to root
        tag = tagger._find_tag_for_scope("completely.unknown.nonexistent.scope")
        assert tag == "/TestModel"
        assert tag != ""  # NEVER EMPTY
        assert tag.startswith("/")  # Valid format

    def test_no_empty_tags_guaranteed(self, sample_hierarchy_data, mock_onnx_model):
        """Test that NO EMPTY TAGS are ever generated."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)
        tagged_nodes = tagger.tag_all_nodes(mock_onnx_model)

        # Verify no empty tags
        for node_name, tag in tagged_nodes.items():
            assert tag, f"Empty tag for node {node_name}"
            assert tag.strip(), f"Whitespace-only tag for node {node_name}"
            assert tag.startswith("/"), (
                f"Invalid tag format for node {node_name}: {tag}"
            )

    def test_root_nodes_get_clean_root_tag(
        self, sample_hierarchy_data, mock_onnx_model
    ):
        """Test that root nodes get clean root module tag (NOT root + operation)."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)
        tagged_nodes = tagger.tag_all_nodes(mock_onnx_model)

        # Find root nodes (those with "/Softmax_123" and "" names)
        root_node_tags = []
        for node in mock_onnx_model.graph.node:
            if node.name in ["/Softmax_123", ""]:
                node_name = node.name or f"{node.op_type}_{id(node)}"
                root_node_tags.append(tagged_nodes[node_name])

        # All root nodes should get clean root tag
        for tag in root_node_tags:
            assert tag == "/TestModel"  # Clean root tag, NO operation type

    def test_factory_function(self, sample_hierarchy_data):
        """Test factory function creates tagger correctly."""
        # Test default (operation fallback disabled)
        tagger1 = create_node_tagger_from_hierarchy(sample_hierarchy_data)
        assert not tagger1.enable_operation_fallback

        # Test with operation fallback enabled
        tagger2 = create_node_tagger_from_hierarchy(
            sample_hierarchy_data, enable_operation_fallback=True
        )
        assert tagger2.enable_operation_fallback

    def test_tagging_statistics(self, sample_hierarchy_data, mock_onnx_model):
        """Test tagging statistics generation."""
        tagger = ONNXNodeTagger(sample_hierarchy_data)
        stats = tagger.get_tagging_statistics(mock_onnx_model)

        # Verify statistics structure
        required_keys = [
            "total_nodes",
            "root_nodes",
            "scoped_nodes",
            "unique_scopes",
            "direct_matches",
            "parent_matches",
            "operation_matches",
            "root_fallbacks",
        ]

        for key in required_keys:
            assert key in stats, f"Missing statistic: {key}"
            assert isinstance(stats[key], int), f"Statistic {key} should be integer"

        # Verify totals make sense
        assert stats["total_nodes"] == len(mock_onnx_model.graph.node)
        assert stats["root_nodes"] + stats["scoped_nodes"] == stats["total_nodes"]

    def test_universal_design_different_hierarchies(self):
        """Test MUST-003: UNIVERSAL DESIGN with various hierarchy formats."""
        # Test with GPT-style hierarchy
        gpt_hierarchy = {
            "transformer.h.0.attn": {
                "traced_tag": "/GPTModel/Transformer/Block.0/Attention",
                "execution_order": 1,
                "module_type": "huggingface",
            },
            "transformer.h.0.mlp": {
                "traced_tag": "/GPTModel/Transformer/Block.0/MLP",
                "execution_order": 2,
                "module_type": "huggingface",
            },
        }

        tagger = ONNXNodeTagger(gpt_hierarchy)
        assert tagger.model_root_tag == "/GPTModel"

        # Should work with GPT-style paths
        tag = tagger._find_tag_for_scope("transformer.h.0.attn")
        assert tag == "/GPTModel/Transformer/Block.0/Attention"

        # Test with ResNet-style hierarchy
        resnet_hierarchy = {
            "layer1.0.conv1": {
                "traced_tag": "/ResNet/Layer1/Block.0/Conv1",
                "execution_order": 1,
                "module_type": "torchvision",
            }
        }

        tagger2 = ONNXNodeTagger(resnet_hierarchy)
        assert tagger2.model_root_tag == "/ResNet"


class TestIntegrationWithTracingHierarchyBuilder:
    """Integration tests with actual TracingHierarchyBuilder output."""

    def test_integration_with_real_hierarchy_data(self):
        """Test integration with realistic hierarchy data structure."""
        # Simulate realistic TracingHierarchyBuilder output
        realistic_hierarchy = {
            "embeddings": {
                "name": "embeddings",
                "class_name": "BertEmbeddings",
                "module_type": "huggingface",
                "traced_tag": "/BertModel/BertEmbeddings",
                "execution_order": 1,
            },
            "encoder.layer.0.attention.self.query": {
                "name": "encoder.layer.0.attention.self.query",
                "class_name": "Linear",
                "module_type": "huggingface",
                "traced_tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention/Query",
                "execution_order": 5,
            },
        }

        tagger = ONNXNodeTagger(realistic_hierarchy)

        # Verify it extracts model info correctly
        assert tagger.model_root_tag == "/BertModel"
        assert len(tagger.scope_to_tag) == 2

        # Test tagging with realistic node
        node = MagicMock()
        node.name = "/encoder/layer.0/attention/self/query/MatMul"
        scope = tagger._extract_scope_from_node(node)
        tag = tagger._find_tag_for_scope(scope)

        assert scope == "encoder.layer.0.attention.self.query"
        assert (
            tag
            == "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention/Query"
        )
