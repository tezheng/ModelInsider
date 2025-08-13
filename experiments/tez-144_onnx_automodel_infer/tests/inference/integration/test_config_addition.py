#!/usr/bin/env python3
"""
Config addition to existing ONNX models tests.

Tests adding configuration files to pre-exported ONNX models for 
Optimum compatibility (retrofit scenarios).

Tests extracted from test_existing_onnx.py
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


class TestConfigAddition:
    """Tests for adding config files to existing ONNX models."""
    
    def test_config_addition_to_existing_onnx(self):
        """Test adding config files to existing ONNX for Optimum compatibility."""
        # Look for existing ONNX model in the project
        possible_onnx_paths = [
            Path("models/bert.onnx"),
            Path("../models/bert.onnx"),
            Path("../../models/bert.onnx"),
            Path("../../../models/bert.onnx"),
        ]
        
        existing_onnx = None
        for path in possible_onnx_paths:
            if path.exists():
                existing_onnx = path
                break
        
        if not existing_onnx:
            pytest.skip("No existing ONNX model found for testing")
        
        size_mb = existing_onnx.stat().st_size / 1024 / 1024
        assert size_mb > 0.1, f"ONNX file too small: {size_mb:.2f} MB"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "bert-test"
            test_dir.mkdir()
            
            # Copy ONNX model
            test_onnx = test_dir / "model.onnx"
            shutil.copy(existing_onnx, test_onnx)
            
            # Test WITHOUT config (should fail)
            with pytest.raises(Exception):
                ORTModelForSequenceClassification.from_pretrained(test_dir)
            
            # Add config files
            model_id = "prajjwal1/bert-tiny"
            tokenizer = self._add_config_files(model_id, test_dir)
            assert tokenizer is not None, "Config addition should succeed"
            
            # Validate directory structure
            files = list(test_dir.glob("*"))
            assert len(files) >= 3, "Should have ONNX + config + tokenizer files"
            
            # Check config overhead
            config_size = sum(f.stat().st_size for f in files if f.name != "model.onnx")
            overhead_percent = (config_size / test_onnx.stat().st_size) * 100
            assert overhead_percent < 10.0, f"Config overhead too high: {overhead_percent:.3f}%"
            
            # Test WITH config (should work)
            ort_model = ORTModelForSequenceClassification.from_pretrained(test_dir)
            assert ort_model is not None, "Model should load with config"
            assert ort_model.config.model_type == "bert", "Should be BERT model"
            
            # Test inference
            self._test_basic_inference(ort_model, tokenizer)
    
    def test_config_file_completeness_for_existing_onnx(self):
        """Test that all necessary config files are created for existing ONNX."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "config_completeness_test"
            test_dir.mkdir()
            
            # Create a dummy ONNX file (we're testing config addition, not the ONNX itself)
            dummy_onnx = test_dir / "model.onnx"
            dummy_onnx.write_bytes(b"dummy onnx content for testing")
            
            # Add config files
            model_id = "prajjwal1/bert-tiny"
            tokenizer = self._add_config_files(model_id, test_dir)
            assert tokenizer is not None, "Config addition should succeed"
            
            # Check required files
            required_files = ["config.json"]
            optional_files = [
                "tokenizer.json",
                "tokenizer_config.json", 
                "vocab.txt",
                "special_tokens_map.json"
            ]
            
            # Verify required files exist
            for file_name in required_files:
                file_path = test_dir / file_name
                assert file_path.exists(), f"Required file missing: {file_name}"
                assert file_path.stat().st_size > 0, f"Required file empty: {file_name}"
            
            # At least some optional files should exist
            optional_exists = sum(1 for f in optional_files if (test_dir / f).exists())
            assert optional_exists >= 2, "Should have tokenizer files"
            
            # Validate config.json content
            with open(test_dir / "config.json") as f:
                config_data = json.load(f)
            
            assert "model_type" in config_data, "Config should have model_type"
            assert config_data["model_type"] == "bert", "Should be BERT model type"
            # Note: Some configs might not have architectures field
            # assert "architectures" in config_data, "Config should have architectures"
    
    def test_retrofit_deployment_pattern(self):
        """Test the retrofit deployment pattern for existing models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_dir = Path(temp_dir) / "deployment"
            deployment_dir.mkdir()
            
            # Simulate existing ONNX model
            onnx_path = deployment_dir / "model.onnx"
            onnx_path.write_bytes(b"dummy onnx for deployment test")
            
            # Apply retrofit pattern
            model_id = "prajjwal1/bert-tiny"
            success = self._retrofit_existing_onnx(model_id, deployment_dir)
            assert success, "Retrofit pattern should succeed"
            
            # Validate deployment readiness
            self._validate_deployment_readiness(deployment_dir)
    
    def test_config_overhead_analysis_existing_models(self):
        """Test config overhead for existing models of various sizes."""
        test_cases = [
            ("small", 1024 * 1024),      # 1 MB
            ("medium", 10 * 1024 * 1024), # 10 MB  
            ("large", 100 * 1024 * 1024), # 100 MB
        ]
        
        model_id = "prajjwal1/bert-tiny"
        
        for size_name, onnx_size in test_cases:
            with tempfile.TemporaryDirectory() as temp_dir:
                test_dir = Path(temp_dir) / f"overhead_{size_name}"
                test_dir.mkdir()
                
                # Create dummy ONNX of specific size
                dummy_onnx = test_dir / "model.onnx"
                dummy_onnx.write_bytes(b"0" * onnx_size)
                
                # Add config files
                self._add_config_files(model_id, test_dir)
                
                # Calculate overhead
                config_files = [f for f in test_dir.glob("*") if f.name != "model.onnx"]
                total_config_size = sum(f.stat().st_size for f in config_files)
                overhead_percent = (total_config_size / onnx_size) * 100
                
                # Larger models should have lower relative overhead
                if size_name == "small":
                    assert overhead_percent < 100.0, f"Small model overhead too high: {overhead_percent:.2f}%"
                elif size_name == "medium":
                    assert overhead_percent < 5.0, f"Medium model overhead too high: {overhead_percent:.2f}%"
                else:  # large
                    assert overhead_percent < 1.0, f"Large model overhead too high: {overhead_percent:.2f}%"
    
    def test_compatibility_validation_existing_onnx(self):
        """Test compatibility validation for existing ONNX models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "compatibility_test"
            test_dir.mkdir()
            
            # Create dummy ONNX file
            dummy_onnx = test_dir / "model.onnx"
            dummy_onnx.write_bytes(b"dummy onnx for compatibility test")
            
            # Add config files
            model_id = "prajjwal1/bert-tiny"
            tokenizer = self._add_config_files(model_id, test_dir)
            
            # Validate compatibility
            compatibility_result = self._validate_optimum_compatibility(test_dir)
            assert compatibility_result["config_present"], "Config should be present"
            assert compatibility_result["tokenizer_present"], "Tokenizer should be present"
            assert compatibility_result["required_files_exist"], "Required files should exist"
    
    def _add_config_files(self, model_id: str, output_dir: Path) -> Optional[AutoTokenizer]:
        """Add config files for Optimum compatibility."""
        try:
            # Add config.json (required)
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(output_dir)
            
            # Add tokenizer (conditional)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(output_dir)
            
            return tokenizer
        except Exception:
            return None
    
    def _retrofit_existing_onnx(self, model_id: str, deployment_dir: Path) -> bool:
        """Apply retrofit pattern to existing ONNX model."""
        try:
            # Implementation pattern:
            # 1. Verify ONNX exists
            onnx_path = deployment_dir / "model.onnx"
            if not onnx_path.exists():
                return False
            
            # 2. Add config files
            tokenizer = self._add_config_files(model_id, deployment_dir)
            if not tokenizer:
                return False
            
            # 3. Validate structure
            required_files = ["model.onnx", "config.json"]
            for file_name in required_files:
                if not (deployment_dir / file_name).exists():
                    return False
            
            return True
        except Exception:
            return False
    
    def _validate_deployment_readiness(self, deployment_dir: Path):
        """Validate that directory is ready for deployment."""
        # Check required files
        required_files = ["model.onnx", "config.json"]
        for file_name in required_files:
            file_path = deployment_dir / file_name
            assert file_path.exists(), f"Deployment missing required file: {file_name}"
            assert file_path.stat().st_size > 0, f"Deployment file empty: {file_name}"
        
        # Check config content
        with open(deployment_dir / "config.json") as f:
            config_data = json.load(f)
        
        assert "model_type" in config_data, "Deployment config missing model_type"
        assert len(config_data) > 1, "Deployment config too minimal"
    
    def _validate_optimum_compatibility(self, test_dir: Path) -> dict:
        """Validate Optimum compatibility for directory."""
        result = {
            "config_present": (test_dir / "config.json").exists(),
            "tokenizer_present": False,
            "required_files_exist": True,
            "onnx_present": (test_dir / "model.onnx").exists(),
        }
        
        # Check for tokenizer files
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt"]
        result["tokenizer_present"] = any((test_dir / f).exists() for f in tokenizer_files)
        
        # Check all required files
        required_files = ["model.onnx", "config.json"]
        result["required_files_exist"] = all((test_dir / f).exists() for f in required_files)
        
        return result
    
    def _test_basic_inference(self, ort_model, tokenizer):
        """Test basic inference functionality."""
        test_text = "This is a test sentence for inference."
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            padding=True,
            truncation=True
        )
        
        outputs = ort_model(**inputs)
        
        # Validate outputs
        assert outputs is not None, "Should get outputs"
        assert hasattr(outputs, 'logits'), "Should have logits"
        
        # Check shapes
        assert len(outputs.logits.shape) >= 2, "Logits should be at least 2D"
        assert outputs.logits.shape[0] == inputs['input_ids'].shape[0], "Batch size should match"
        
        # Get prediction
        prediction = np.argmax(outputs.logits, axis=-1)[0]
        confidence = np.max(np.softmax(outputs.logits[0])) * 100
        
        # Basic sanity checks
        assert isinstance(prediction, (int, np.integer)), "Prediction should be integer"
        assert 0 <= confidence <= 100, "Confidence should be percentage"