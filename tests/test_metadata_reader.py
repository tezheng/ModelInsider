"""
Tests for HTP Metadata Reader

Tests the MetadataReader class functionality for parsing HTP metadata files
and extracting hierarchy information.
"""

import json

import pytest

from modelexport.graphml.metadata_reader import MetadataReader


@pytest.fixture
def sample_htp_metadata(tmp_path):
    """Create a sample HTP metadata file for testing."""
    metadata = {
        "export_data": {
            "hierarchy_data": {
                "": {
                    "class_name": "BertModel",
                    "module_type": "BertModel",
                    "execution_order": 0,
                    "traced_tag": ""
                },
                "embeddings": {
                    "class_name": "BertEmbeddings",
                    "module_type": "BertEmbeddings",
                    "execution_order": 1,
                    "traced_tag": "embeddings"
                },
                "encoder": {
                    "class_name": "BertEncoder",
                    "module_type": "BertEncoder",
                    "execution_order": 2,
                    "traced_tag": "encoder"
                },
                "encoder/layer": {
                    "class_name": "ModuleList",
                    "module_type": "ModuleList",
                    "execution_order": 3,
                    "traced_tag": "encoder/layer"
                },
                "encoder/layer/0": {
                    "class_name": "BertLayer",
                    "module_type": "BertLayer",
                    "execution_order": 4,
                    "traced_tag": "encoder/layer/0"
                },
                "encoder/layer/0/attention": {
                    "class_name": "BertAttention",
                    "module_type": "BertAttention",
                    "execution_order": 5,
                    "traced_tag": "encoder/layer/0/attention"
                }
            },
            "tagged_nodes": {
                "Add_24": {
                    "hierarchy_tag": "embeddings",
                    "op_type": "Add"
                },
                "MatMul_42": {
                    "hierarchy_tag": "encoder/layer/0/attention",
                    "op_type": "MatMul"
                },
                "Softmax_45": {
                    "hierarchy_tag": "encoder/layer/0/attention",
                    "op_type": "Softmax"
                }
            }
        },
        "model_info": {
            "model_name": "bert-tiny-test",
            "export_time": "2025-01-28T10:00:00Z"
        }
    }
    
    metadata_path = tmp_path / "test_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


@pytest.fixture
def alternative_metadata_format(tmp_path):
    """Create metadata with alternative structure (no export_data wrapper)."""
    metadata = {
        "hierarchy": {
            "": {
                "class_name": "GPT2Model",
                "execution_order": 0
            },
            "transformer": {
                "class_name": "GPT2MainLayer",
                "execution_order": 1
            }
        },
        "node_tags": {
            "Add_10": {
                "tag": "transformer",
                "op_type": "Add"
            }
        },
        "modules": {
            "": {
                "class_name": "GPT2Model",
                "module_type": "GPT2Model"
            },
            "transformer": {
                "class_name": "GPT2MainLayer",
                "module_type": "GPT2MainLayer"
            }
        }
    }
    
    metadata_path = tmp_path / "alt_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


class TestMetadataReader:
    """Test MetadataReader functionality."""
    
    def test_init_with_valid_file(self, sample_htp_metadata):
        """Test initialization with valid metadata file."""
        reader = MetadataReader(str(sample_htp_metadata))
        assert reader.metadata_path == sample_htp_metadata
        assert isinstance(reader.metadata, dict)
        assert len(reader.hierarchy_data) > 0
        assert len(reader.tagged_nodes) > 0
    
    def test_init_with_missing_file(self, tmp_path):
        """Test initialization with non-existent file."""
        missing_path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError) as exc_info:
            MetadataReader(str(missing_path))
        assert "HTP metadata file not found" in str(exc_info.value)
    
    def test_init_with_invalid_json(self, tmp_path):
        """Test initialization with invalid JSON file."""
        invalid_path = tmp_path / "invalid.json"
        with open(invalid_path, "w") as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            MetadataReader(str(invalid_path))
    
    def test_extract_hierarchy_data(self, sample_htp_metadata):
        """Test hierarchy data extraction."""
        reader = MetadataReader(str(sample_htp_metadata))
        hierarchy = reader.hierarchy_data
        
        assert "" in hierarchy  # Root module
        assert hierarchy[""]["class_name"] == "BertModel"
        assert "embeddings" in hierarchy
        assert "encoder/layer/0/attention" in hierarchy
        assert hierarchy["encoder/layer/0/attention"]["class_name"] == "BertAttention"
    
    def test_extract_tagged_nodes(self, sample_htp_metadata):
        """Test tagged nodes extraction."""
        reader = MetadataReader(str(sample_htp_metadata))
        tagged = reader.tagged_nodes
        
        assert "Add_24" in tagged
        assert tagged["Add_24"]["hierarchy_tag"] == "embeddings"
        assert "MatMul_42" in tagged
        assert tagged["MatMul_42"]["hierarchy_tag"] == "encoder/layer/0/attention"
    
    def test_get_node_hierarchy_tag(self, sample_htp_metadata):
        """Test getting hierarchy tag for specific nodes."""
        reader = MetadataReader(str(sample_htp_metadata))
        
        # Test existing nodes
        assert reader.get_node_hierarchy_tag("Add_24") == "embeddings"
        assert reader.get_node_hierarchy_tag("MatMul_42") == "encoder/layer/0/attention"
        
        # Test non-existent node
        assert reader.get_node_hierarchy_tag("NonExistent") is None
    
    def test_get_module_hierarchy(self, sample_htp_metadata):
        """Test module hierarchy extraction."""
        reader = MetadataReader(str(sample_htp_metadata))
        hierarchy = reader.get_module_hierarchy()
        
        # Check parent-child relationships
        assert "" in hierarchy  # Root has children
        assert "embeddings" in hierarchy[""]
        assert "encoder" in hierarchy[""]
        
        assert "encoder" in hierarchy
        assert "encoder/layer" in hierarchy["encoder"]
        
        assert "encoder/layer" in hierarchy
        assert "encoder/layer/0" in hierarchy["encoder/layer"]
    
    def test_get_module_info(self, sample_htp_metadata):
        """Test getting module information."""
        reader = MetadataReader(str(sample_htp_metadata))
        
        # Test root module
        root_info = reader.get_module_info("")
        assert root_info["class_name"] == "BertModel"
        assert root_info["execution_order"] == 0
        
        # Test nested module
        attention_info = reader.get_module_info("encoder/layer/0/attention")
        assert attention_info["class_name"] == "BertAttention"
        assert attention_info["execution_order"] == 5
        
        # Test non-existent module
        assert reader.get_module_info("non/existent") == {}
    
    def test_get_all_modules(self, sample_htp_metadata):
        """Test getting all modules in hierarchy order."""
        reader = MetadataReader(str(sample_htp_metadata))
        modules = reader.get_all_modules()
        
        # Check ordering - shallow modules come first
        assert modules[0] == ""
        assert "embeddings" in modules[:3]
        assert "encoder" in modules[:3]
        
        # Deeper modules come later
        assert modules.index("encoder/layer") > modules.index("encoder")
        assert modules.index("encoder/layer/0") > modules.index("encoder/layer")
        assert modules.index("encoder/layer/0/attention") > modules.index("encoder/layer/0")
    
    def test_alternative_format_handling(self, alternative_metadata_format):
        """Test handling of alternative metadata format."""
        reader = MetadataReader(str(alternative_metadata_format))
        
        # Check hierarchy extraction from "hierarchy" key
        assert reader.hierarchy_data[""]["class_name"] == "GPT2Model"
        
        # Check node tags from "node_tags" key with "tag" field
        assert reader.get_node_hierarchy_tag("Add_10") == "transformer"
        
        # Check module info from "modules" key
        assert reader.module_info["transformer"]["class_name"] == "GPT2MainLayer"
    
    def test_string_format_node_tags(self, tmp_path):
        """Test handling of string format for node tags (real HTP metadata format)."""
        metadata = {
            "nodes": {
                "/embeddings/Constant": "/BertModel/BertEmbeddings",
                "/embeddings/Add": "/BertModel/BertEmbeddings",
                "/encoder/layer/0/attention/MatMul": "/BertModel/BertEncoder/ModuleList/BertLayer/BertAttention"
            },
            "modules": {
                "/BertModel": {"class_name": "BertModel"},
                "/BertModel/BertEmbeddings": {"class_name": "BertEmbeddings"},
                "/BertModel/BertEncoder": {"class_name": "BertEncoder"}
            }
        }
        
        metadata_path = tmp_path / "string_format.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        reader = MetadataReader(str(metadata_path))
        
        # Test string format node tags
        assert reader.get_node_hierarchy_tag("/embeddings/Constant") == "/BertModel/BertEmbeddings"
        assert reader.get_node_hierarchy_tag("/embeddings/Add") == "/BertModel/BertEmbeddings"
        assert reader.get_node_hierarchy_tag("/encoder/layer/0/attention/MatMul") == "/BertModel/BertEncoder/ModuleList/BertLayer/BertAttention"
        assert reader.get_node_hierarchy_tag("NonExistent") is None
    
    def test_empty_metadata_handling(self, tmp_path):
        """Test handling of empty metadata file."""
        empty_path = tmp_path / "empty.json"
        with open(empty_path, "w") as f:
            json.dump({}, f)
        
        reader = MetadataReader(str(empty_path))
        assert reader.hierarchy_data == {}
        assert reader.tagged_nodes == {}
        assert reader.module_info == {}
        assert reader.get_all_modules() == []
    
    def test_partial_metadata_handling(self, tmp_path):
        """Test handling of metadata with only some sections."""
        partial_path = tmp_path / "partial.json"
        metadata = {
            "hierarchy_data": {
                "module1": {"class_name": "TestModule"}
            }
            # No tagged_nodes or modules sections
        }
        with open(partial_path, "w") as f:
            json.dump(metadata, f)
        
        reader = MetadataReader(str(partial_path))
        assert len(reader.hierarchy_data) == 1
        assert reader.tagged_nodes == {}
        # Module info should be built from hierarchy data
        assert len(reader.module_info) == 1
        assert reader.module_info["module1"]["class_name"] == "TestModule"