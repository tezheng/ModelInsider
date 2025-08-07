"""Tests for Optimum configuration and Hub model integration."""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp import HTPExporter
from modelexport.strategies.htp.config_builder import HTPConfigBuilder
from modelexport.utils import (
    inject_hub_metadata,
    is_hub_model,
    load_hf_components_from_onnx,
)


class TinyModel(nn.Module):
    """Simple test model."""
    
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


class TestHTPConfigBuilder:
    """Test the HTPConfigBuilder class."""
    
    def test_load_config_from_huggingface(self):
        """Test loading config from HuggingFace."""
        builder = HTPConfigBuilder("prajjwal1/bert-tiny")
        config = builder.load_config()
        
        assert config is not None
        assert config.model_type == "bert"
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_hidden_layers")
    
    def test_create_minimal_config(self):
        """Test creating minimal config for different model types."""
        # Test BERT config
        bert_config = HTPConfigBuilder.create_minimal_config(
            model_type="bert",
            hidden_size=128,
            num_hidden_layers=2
        )
        assert bert_config["model_type"] == "bert"
        assert bert_config["hidden_size"] == 128
        assert bert_config["num_hidden_layers"] == 2
        assert "intermediate_size" in bert_config
        
        # Test GPT2 config
        gpt2_config = HTPConfigBuilder.create_minimal_config(
            model_type="gpt2",
            hidden_size=768,
            num_hidden_layers=12
        )
        assert gpt2_config["model_type"] == "gpt2"
        assert gpt2_config["n_embd"] == 768
        assert gpt2_config["n_layer"] == 12
        
        # Test RoBERTa config
        roberta_config = HTPConfigBuilder.create_minimal_config(
            model_type="roberta",
            hidden_size=768,
            num_hidden_layers=12
        )
        assert roberta_config["model_type"] == "roberta"
        assert roberta_config["hidden_size"] == 768
    
    def test_save_config(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test with loaded config
            builder = HTPConfigBuilder("prajjwal1/bert-tiny")
            builder.load_config()
            
            success = builder.save_config(tmpdir)
            assert success
            
            config_path = tmpdir / "config.json"
            assert config_path.exists()
            
            # Verify content
            import json
            with open(config_path) as f:
                config_data = json.load(f)
            assert config_data["model_type"] == "bert"
    
    def test_save_minimal_config(self):
        """Test saving minimal config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_dict = HTPConfigBuilder.create_minimal_config(
                model_type="bert",
                hidden_size=256
            )
            
            success = HTPConfigBuilder.save_minimal_config(tmpdir, config_dict)
            assert success
            
            config_path = tmpdir / "config.json"
            assert config_path.exists()
            
            import json
            with open(config_path) as f:
                saved_config = json.load(f)
            assert saved_config["model_type"] == "bert"
            assert saved_config["hidden_size"] == 256
    
    def test_generate_optimum_config(self):
        """Test the full Optimum config generation flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            builder = HTPConfigBuilder("prajjwal1/bert-tiny")
            results = builder.generate_optimum_config(
                output_dir=tmpdir,
                save_tokenizer=False  # Skip tokenizer for faster test
            )
            
            assert results["config"] is True
            assert (tmpdir / "config.json").exists()
    
    def test_no_model_name(self):
        """Test behavior when no model name is provided."""
        builder = HTPConfigBuilder(None)
        
        config = builder.load_config()
        assert config is None
        
        tokenizer = builder.load_tokenizer()
        assert tokenizer is None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            success = builder.save_config(tmpdir)
            assert success is False


class TestHubModelDetection:
    """Test HuggingFace Hub model detection."""
    
    def test_hub_model_detection(self):
        """Test detection of Hub vs local models."""
        # Hub models
        hub_models = [
            "bert-base-uncased",
            "google/flan-t5-small", 
            "meta-llama/Llama-2-7b-hf",
            "facebook/bart-large",
            "openai/clip-vit-base-patch32",
        ]
        
        for model_id in hub_models:
            is_hub, metadata = is_hub_model(model_id)
            # Accept either verified or unverified Hub models
            assert is_hub or metadata.get("type") == "hub_unverified", f"Failed to detect {model_id} as Hub model"
        
        # Local models
        local_models = [
            "./local/model",
            "/absolute/path/model",
            "../relative/path",
            "~/home/model",
            "C:\\Windows\\path",
            "D:/Windows/path",
        ]
        
        for model_path in local_models:
            is_hub, metadata = is_hub_model(model_path)
            assert not is_hub, f"Incorrectly detected {model_path} as Hub model"
            assert metadata.get("type") in ["local", "invalid"], f"Wrong type for {model_path}"
    
    def test_hub_model_with_revision(self):
        """Test Hub model detection with revision."""
        model_with_revision = "bert-base-uncased@main"
        is_hub, metadata = is_hub_model(model_with_revision)
        assert is_hub or metadata.get("type") == "hub_unverified"
        
        if metadata.get("revision"):
            assert metadata["revision"] == "main"


class TestHubMetadataInjection:
    """Test metadata injection into ONNX models."""
    
    def test_metadata_injection(self):
        """Test injecting Hub metadata into ONNX model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simple ONNX model
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            onnx_path = Path(temp_dir) / "test.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
            )
            
            # Load and inject metadata
            onnx_model = onnx.load(str(onnx_path))
            
            test_metadata = {
                "model_id": "test-org/test-model",
                "sha": "abcd1234efgh5678",
                "revision": "main",
                "pipeline_tag": "text-classification",
                "library_name": "transformers",
                "private": False,
                "gated": False,
            }
            
            inject_hub_metadata(onnx_model, "test-org/test-model", test_metadata)
            onnx.save(onnx_model, str(onnx_path))
            
            # Verify metadata
            onnx_model = onnx.load(str(onnx_path))
            metadata = {}
            for prop in onnx_model.metadata_props:
                metadata[prop.key] = prop.value
            
            assert metadata.get("hf_hub_id") == "test-org/test-model"
            assert metadata.get("hf_hub_revision") == "abcd1234"
            assert metadata.get("hf_model_type") == "hub"
            assert metadata.get("hf_pipeline_tag") == "text-classification"
            assert metadata.get("hf_library_name") == "transformers"
            assert metadata.get("hf_private") == "False"
            assert metadata.get("hf_gated") == "False"
    
    def test_metadata_clearing(self):
        """Test that existing HF metadata is cleared before injection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ONNX model with existing metadata
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            onnx_path = Path(temp_dir) / "test.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
            )
            
            # Add some existing HF metadata
            onnx_model = onnx.load(str(onnx_path))
            old_prop = onnx_model.metadata_props.add()
            old_prop.key = "hf_old_property"
            old_prop.value = "should_be_removed"
            
            # Add non-HF metadata that should be preserved
            keep_prop = onnx_model.metadata_props.add()
            keep_prop.key = "other_property"
            keep_prop.value = "should_be_kept"
            
            onnx.save(onnx_model, str(onnx_path))
            
            # Inject new metadata
            onnx_model = onnx.load(str(onnx_path))
            test_metadata = {"model_id": "new-org/new-model", "sha": "12345678"}
            inject_hub_metadata(onnx_model, "new-org/new-model", test_metadata)
            onnx.save(onnx_model, str(onnx_path))
            
            # Verify old HF metadata removed, new added, non-HF kept
            onnx_model = onnx.load(str(onnx_path))
            metadata = {}
            for prop in onnx_model.metadata_props:
                metadata[prop.key] = prop.value
            
            assert "hf_old_property" not in metadata
            assert metadata.get("hf_hub_id") == "new-org/new-model"
            assert metadata.get("other_property") == "should_be_kept"


class TestHTPExporterIntegration:
    """Test HTP exporter with Hub integration."""
    
    def test_htp_export_with_hub_metadata(self):
        """Test that HTP exporter correctly handles Hub models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple model
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            output_path = Path(temp_dir) / "model.onnx"
            
            # Export with HTP
            exporter = HTPExporter(verbose=False)
            
            # We can't actually test with a real Hub model without network
            # So we'll test the local model path
            result = exporter.export(
                model=model,
                dummy_input=dummy_input,
                output_path=str(output_path),
                model_name_or_path="./local/test/model",  # Local path
            )
            
            assert result["success"]
            assert output_path.exists()
            
            # Check metadata
            onnx_model = onnx.load(str(output_path))
            metadata = {}
            for prop in onnx_model.metadata_props:
                metadata[prop.key] = prop.value
            
            # Should be marked as local model
            assert metadata.get("hf_model_type") == "local"
            assert metadata.get("hf_original_path") == "./local/test/model"
    
    def test_htp_export_without_model_path(self):
        """Test that HTP export works without model_name_or_path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            output_path = Path(temp_dir) / "model.onnx"
            
            exporter = HTPExporter(verbose=False)
            result = exporter.export(
                model=model,
                dummy_input=dummy_input,
                output_path=str(output_path),
                # No model_name_or_path provided
            )
            
            assert result["success"]
            assert output_path.exists()
            
            # Check that no HF metadata was added
            onnx_model = onnx.load(str(output_path))
            hf_metadata_count = sum(
                1 for prop in onnx_model.metadata_props 
                if prop.key.startswith("hf_")
            )
            assert hf_metadata_count == 0


class TestOptimumLoading:
    """Test loading ONNX models for Optimum inference."""
    
    def test_load_hub_model_from_metadata(self):
        """Test loading config from Hub metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ONNX with Hub metadata
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            onnx_path = Path(temp_dir) / "model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
            )
            
            # Inject Hub metadata
            onnx_model = onnx.load(str(onnx_path))
            metadata = {
                "model_id": "bert-base-uncased",
                "sha": "12345678",
                "pipeline_tag": "text-classification",
            }
            inject_hub_metadata(onnx_model, "bert-base-uncased", metadata)
            onnx.save(onnx_model, str(onnx_path))
            
            # Try to load (will fail without network, but we can test the attempt)
            with pytest.raises(Exception) as exc_info:
                config, preprocessor = load_hf_components_from_onnx(str(onnx_path))
            
            # Should fail trying to load from Hub, not because of missing metadata
            error_msg = str(exc_info.value)
            assert "bert-base-uncased" in error_msg or "HTTPError" in error_msg or "Connection" in error_msg
    
    def test_load_local_model_error(self):
        """Test appropriate error when local model config is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ONNX marked as local without config files
            model = TinyModel()
            dummy_input = torch.randn(1, 10)
            onnx_path = Path(temp_dir) / "model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=17,
            )
            
            # Mark as local model
            onnx_model = onnx.load(str(onnx_path))
            meta_type = onnx_model.metadata_props.add()
            meta_type.key = "hf_model_type"
            meta_type.value = "local"
            onnx.save(onnx_model, str(onnx_path))
            
            # Should raise error about missing config
            with pytest.raises(ValueError) as exc_info:
                config, preprocessor = load_hf_components_from_onnx(str(onnx_path))
            
            assert "config.json not found" in str(exc_info.value)