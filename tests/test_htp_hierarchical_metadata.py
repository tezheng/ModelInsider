"""
Test cases for HTP hierarchical metadata structure.

This module tests the hierarchical module structure implementation
in the HTP metadata, including schema validation and data integrity.
"""

import json
from pathlib import Path
import pytest
import jsonschema
import tempfile

from modelexport.strategies.htp_new.metadata_builder import HTPMetadataBuilder
from modelexport.strategies.htp_new.metadata_writer import MetadataWriter
from modelexport.strategies.htp_new.step_data import ModuleInfo


class TestHierarchicalMetadata:
    """Test hierarchical metadata structure."""
    
    @pytest.fixture
    def schema(self):
        """Load the HTP metadata schema."""
        schema_path = Path(__file__).parent.parent / "modelexport/strategies/htp_new/htp_metadata_schema.json"
        with open(schema_path) as f:
            return json.load(f)
    
    @pytest.fixture
    def sample_flat_hierarchy(self):
        """Create sample flat hierarchy data."""
        return {
            "": ModuleInfo(
                class_name="BertModel",
                traced_tag="/BertModel",
                execution_order=0
            ),
            "embeddings": ModuleInfo(
                class_name="BertEmbeddings",
                traced_tag="/BertModel/BertEmbeddings",
                execution_order=1
            ),
            "encoder": ModuleInfo(
                class_name="BertEncoder",
                traced_tag="/BertModel/BertEncoder",
                execution_order=2
            ),
            "encoder.layer.0": ModuleInfo(
                class_name="BertLayer",
                traced_tag="/BertModel/BertEncoder/BertLayer.0",
                execution_order=3
            ),
            "encoder.layer.1": ModuleInfo(
                class_name="BertLayer",
                traced_tag="/BertModel/BertEncoder/BertLayer.1",
                execution_order=10
            ),
            "encoder.layer.0.attention": ModuleInfo(
                class_name="BertAttention",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention",
                execution_order=4
            ),
            "encoder.layer.0.attention.self": ModuleInfo(
                class_name="BertSelfAttention",
                traced_tag="/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention",
                execution_order=5
            ),
            "pooler": ModuleInfo(
                class_name="BertPooler",
                traced_tag="/BertModel/BertPooler",
                execution_order=17
            ),
        }
    
    def test_hierarchical_structure_building(self, sample_flat_hierarchy):
        """Test building hierarchical structure from flat hierarchy."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Test root module
            assert hierarchical["class_name"] == "BertModel"
            assert hierarchical["scope"] == ""
            assert hierarchical["traced_tag"] == "/BertModel"
            assert "children" in hierarchical
            
            # Test first-level children
            children = hierarchical["children"]
            assert "BertEmbeddings" in children
            assert "BertEncoder" in children
            assert "BertPooler" in children
            
            # Test BertEncoder has children
            encoder = children["BertEncoder"]
            assert encoder["class_name"] == "BertEncoder"
            assert encoder["scope"] == "encoder"
            assert "children" in encoder
            
            # Test indexed modules (BertLayer.0, BertLayer.1)
            encoder_children = encoder["children"]
            assert "BertLayer.0" in encoder_children
            assert "BertLayer.1" in encoder_children
            
            # Test BertLayer.0 structure
            layer0 = encoder_children["BertLayer.0"]
            assert layer0["class_name"] == "BertLayer"
            assert layer0["scope"] == "encoder.layer.0"
            assert layer0["traced_tag"] == "/BertModel/BertEncoder/BertLayer.0"
            
            # Test nested children
            assert "children" in layer0
            assert "BertAttention" in layer0["children"]
            
            # Test deep nesting
            attention = layer0["children"]["BertAttention"]
            assert attention["scope"] == "encoder.layer.0.attention"
            assert "children" in attention
            assert "BertSelfAttention" in attention["children"]
    
    def test_module_counting(self, sample_flat_hierarchy):
        """Test counting modules in hierarchical structure."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Count modules
            count = writer._count_modules(hierarchical)
            assert count == len(sample_flat_hierarchy)  # Should count all modules
    
    def test_module_type_extraction(self, sample_flat_hierarchy):
        """Test extracting unique module types."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Extract module types
            types = writer._extract_module_types(hierarchical)
            expected_types = {
                "BertModel", "BertEmbeddings", "BertEncoder", 
                "BertLayer", "BertAttention", "BertSelfAttention", 
                "BertPooler"
            }
            assert set(types) == expected_types
    
    def test_schema_validation(self, schema, sample_flat_hierarchy):
        """Test that generated metadata validates against schema."""
        builder = HTPMetadataBuilder()
        
        # Set export context with timestamp
        builder.with_export_context(
            export_time_seconds=1.0
        )
        builder._export_context.timestamp = "2025-07-22T12:00:00Z"  # Set timestamp manually for test
        
        # Build metadata
        builder.with_model_info(
            name_or_path="test/model",
            class_name="BertModel",
            total_modules=48,
            total_parameters=1000000
        )
        
        builder.with_tracing_info(
            modules_traced=8,
            execution_steps=16,
            model_type="bert",
            task="feature-extraction"
        )
        
        # Build hierarchical modules
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            builder.with_modules(hierarchical)
        
        # Add required fields
        builder.with_tagging_info(
            tagged_nodes={"node1": "/BertModel"},
            statistics={},
            total_onnx_nodes=100,
            tagged_nodes_count=100,
            coverage_percentage=100.0,
            empty_tags=0
        )
        
        builder.with_output_files(
            onnx_path="test.onnx",
            onnx_size_mb=10.0,
            metadata_path="test_metadata.json"
        )
        
        builder.with_export_report(
            export_time_seconds=1.0,
            steps={
                "model_preparation": {
                    "completed": True,
                    "timestamp": "2025-07-22T12:00:00Z",
                    "model_class": "BertModel",
                    "total_modules": 48,
                    "total_parameters": 1000000
                },
                "input_generation": {
                    "method": "auto_generated",
                    "model_type": "bert",
                    "task": "feature-extraction",
                    "inputs": {}
                },
                "hierarchy_building": {
                    "completed": True,
                    "timestamp": "2025-07-22T12:00:01Z",
                    "modules_traced": 8,
                    "execution_steps": 16
                },
                "onnx_export": {
                    "opset_version": 17,
                    "do_constant_folding": True,
                    "onnx_size_mb": 10.0
                },
                "node_tagging": {
                    "completed": True,
                    "timestamp": "2025-07-22T12:00:02Z",
                    "total_nodes": 100,
                    "tagged_nodes_count": 100,
                    "coverage_percentage": 100.0,
                    "statistics": {},
                    "coverage": {
                        "percentage": 100.0,
                        "total_onnx_nodes": 100,
                        "tagged_nodes": 100
                    }
                },
                "tag_injection": {
                    "tags_injected": True,
                    "tags_stripped": False
                }
            },
            empty_tags_guarantee=0,
            coverage_percentage=100.0
        )
        
        builder.with_statistics(
            export_time=1.0,
            hierarchy_modules=8,
            traced_modules=8,  # Same as hierarchy_modules for this test
            onnx_nodes=100,
            tagged_nodes=100,
            empty_tags=0,
            coverage_percentage=100.0,
            module_types=["BertModel"]
        )
        
        # Build and validate
        metadata = builder.build()
        jsonschema.validate(metadata, schema)
    
    def test_indexed_module_naming(self, sample_flat_hierarchy):
        """Test that indexed modules use correct naming convention."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Check BertLayer.0 and BertLayer.1
            encoder_children = hierarchical["children"]["BertEncoder"]["children"]
            
            # Keys should be class name with index
            assert "BertLayer.0" in encoder_children
            assert "BertLayer.1" in encoder_children
            
            # Class names should be without index
            assert encoder_children["BertLayer.0"]["class_name"] == "BertLayer"
            assert encoder_children["BertLayer.1"]["class_name"] == "BertLayer"
            
            # Scopes should have full path
            assert encoder_children["BertLayer.0"]["scope"] == "encoder.layer.0"
            assert encoder_children["BertLayer.1"]["scope"] == "encoder.layer.1"
    
    def test_scope_field_format(self, sample_flat_hierarchy):
        """Test that scope fields contain full path from root."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Root scope should be empty string
            assert hierarchical["scope"] == ""
            
            # First level scopes
            assert hierarchical["children"]["BertEmbeddings"]["scope"] == "embeddings"
            assert hierarchical["children"]["BertEncoder"]["scope"] == "encoder"
            
            # Nested scopes should have full path
            layer0 = hierarchical["children"]["BertEncoder"]["children"]["BertLayer.0"]
            assert layer0["scope"] == "encoder.layer.0"
            
            attention = layer0["children"]["BertAttention"]
            assert attention["scope"] == "encoder.layer.0.attention"
            
            self_attention = attention["children"]["BertSelfAttention"]
            assert self_attention["scope"] == "encoder.layer.0.attention.self"
    
    def test_execution_order_preserved(self, sample_flat_hierarchy):
        """Test that execution order is preserved in hierarchical structure."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(sample_flat_hierarchy)
            
            # Check execution orders
            assert hierarchical["execution_order"] == 0
            assert hierarchical["children"]["BertEmbeddings"]["execution_order"] == 1
            assert hierarchical["children"]["BertEncoder"]["execution_order"] == 2
            
            layer0 = hierarchical["children"]["BertEncoder"]["children"]["BertLayer.0"]
            assert layer0["execution_order"] == 3
            
            layer1 = hierarchical["children"]["BertEncoder"]["children"]["BertLayer.1"]
            assert layer1["execution_order"] == 10
    
    def test_empty_hierarchy_handling(self):
        """Test handling of empty hierarchy."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules({})
            
            assert hierarchical == {}
    
    def test_single_module_hierarchy(self):
        """Test hierarchy with only root module."""
        flat = {
            "": ModuleInfo(
                class_name="SimpleModel",
                traced_tag="/SimpleModel",
                execution_order=0
            )
        }
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            writer = MetadataWriter(tmp.name)
            hierarchical = writer._build_hierarchical_modules(flat)
            
            assert hierarchical["class_name"] == "SimpleModel"
            assert hierarchical["scope"] == ""
            assert "children" not in hierarchical  # No children


class TestMetadataConsistency:
    """Test metadata consistency and correctness."""
    
    def test_report_steps_completeness(self):
        """Test that all 6 export steps are recorded."""
        # This would require a more complete integration test
        # For now, we verify the schema includes all steps
        schema_path = Path(__file__).parent.parent / "modelexport/strategies/htp_new/htp_metadata_schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        
        steps_def = schema["$defs"]["ExportSteps"]["properties"]
        expected_steps = [
            "model_preparation",
            "input_generation", 
            "hierarchy_building",
            "onnx_export",
            "node_tagging",
            "tag_injection"
        ]
        
        for step in expected_steps:
            assert step in steps_def, f"Missing step: {step}"
    
    def test_statistics_alignment(self):
        """Test that statistics fields align with actual data."""
        builder = HTPMetadataBuilder()
        
        # Set statistics
        builder.with_statistics(
            export_time=1.5,
            hierarchy_modules=18,
            traced_modules=18,  # Same as hierarchy_modules for this test
            onnx_nodes=136,
            tagged_nodes=136,
            empty_tags=0,
            coverage_percentage=100.0,
            module_types=["BertModel", "BertEncoder"]
        )
        
        # Build partial metadata (would fail full validation)
        from modelexport.strategies.htp_new.metadata_builder import ModelInfo
        builder._model_info = ModelInfo(
            name_or_path="test",
            class_name="TestModel",
            total_modules=48,
            total_parameters=1000
        )
        
        stats = builder._statistics
        assert stats.empty_tags == 0  # Always 0 with HTP
        assert stats.coverage_percentage == 100.0
        assert stats.tagged_nodes == stats.onnx_nodes  # Should match