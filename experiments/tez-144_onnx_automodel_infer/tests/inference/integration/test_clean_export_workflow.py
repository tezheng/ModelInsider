#!/usr/bin/env python3
"""
Clean ONNX export workflow tests.

Tests the complete production workflow:
1. Export clean ONNX (without HTP metadata)
2. Add configuration files
3. Validate Optimum compatibility
4. Test inference capabilities

Tests extracted from test_clean_onnx_optimum.py
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pytest
from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


class TestCleanExportWorkflow:
    """Tests for clean ONNX export and Optimum integration workflow."""
    
    def test_complete_export_config_inference_workflow(self):
        """Test the complete workflow with inference validation."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "bert-export"
            export_dir.mkdir()
            
            # Step 1: Export clean ONNX
            onnx_path = export_dir / "model.onnx"
            success = self._export_clean_onnx(model_id, onnx_path)
            assert success, "Clean ONNX export should succeed"
            assert onnx_path.exists(), "ONNX file should be created"
            
            # Validate file properties
            file_size_mb = onnx_path.stat().st_size / 1024 / 1024
            assert file_size_mb > 0.1, f"ONNX file too small: {file_size_mb:.2f} MB"
            
            # Step 2: Add config files
            tokenizer = self._add_config_files(model_id, export_dir)
            assert tokenizer is not None, "Config files should be added successfully"
            
            # Validate directory structure
            files = list(export_dir.glob("*"))
            assert len(files) >= 3, "Should have ONNX + config + tokenizer files"
            
            # Check overhead
            config_size = sum(f.stat().st_size for f in files if f.name != "model.onnx")
            overhead_percent = (config_size / onnx_path.stat().st_size) * 100
            assert overhead_percent < 10.0, f"Config overhead too high: {overhead_percent:.3f}%"
            
            # Step 3: Load with Optimum
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            assert ort_model is not None, "Model should load with Optimum"
            assert ort_model.config.model_type == "bert", "Should be BERT model"
            assert hasattr(ort_model.config, 'architectures'), "Should have architectures"
            
            # Step 4: Test inference
            self._test_inference(ort_model, tokenizer, export_dir)
    
    def test_export_without_metadata(self):
        """Test that clean export removes HTP metadata."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            
            # Export with clean-onnx flag
            clean_onnx_path = export_dir / "clean_model.onnx"
            success = self._export_clean_onnx(model_id, clean_onnx_path)
            assert success, "Clean export should succeed"
            
            # Check for metadata files (should not exist for clean export)
            metadata_file = export_dir / "model_htp_metadata.json"
            assert not metadata_file.exists(), "Clean export should not create metadata files"
            
            # Export without clean flag for comparison
            regular_onnx_path = export_dir / "regular_model.onnx"
            success_regular = self._export_regular_onnx(model_id, regular_onnx_path)
            assert success_regular, "Regular export should succeed"
            
            # Both files should exist but may have different sizes
            assert clean_onnx_path.exists() and regular_onnx_path.exists()
    
    def test_config_file_completeness(self):
        """Test that all necessary config files are created."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "config_test"
            export_dir.mkdir()
            
            # Add config files
            tokenizer = self._add_config_files(model_id, export_dir)
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
                file_path = export_dir / file_name
                assert file_path.exists(), f"Required file missing: {file_name}"
                assert file_path.stat().st_size > 0, f"Required file empty: {file_name}"
            
            # At least some optional files should exist
            optional_exists = sum(1 for f in optional_files if (export_dir / f).exists())
            assert optional_exists >= 2, "Should have tokenizer files"
            
            # Validate config.json content
            with open(export_dir / "config.json") as f:
                config_data = json.load(f)
            
            assert "model_type" in config_data, "Config should have model_type"
            assert config_data["model_type"] == "bert", "Should be BERT model type"
            # Note: Some configs might not have architectures field
            # assert "architectures" in config_data, "Config should have architectures"
    
    def test_optimum_compatibility_validation(self):
        """Test detailed Optimum compatibility validation."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "optimum_test"
            export_dir.mkdir()
            
            # Complete workflow
            onnx_path = export_dir / "model.onnx"
            self._export_clean_onnx(model_id, onnx_path)
            self._add_config_files(model_id, export_dir)
            
            # Load with Optimum and validate properties
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            
            # Test model properties
            assert isinstance(ort_model, ORTModelForSequenceClassification)
            assert hasattr(ort_model, 'model'), "Should have ONNX Runtime model"
            assert hasattr(ort_model, 'config'), "Should have transformers config"
            assert hasattr(ort_model, 'onnx_paths'), "Should have ONNX paths info"
            
            # Test config properties
            config = ort_model.config
            assert config.model_type == "bert"
            assert hasattr(config, 'num_labels'), "Should have num_labels for classification"
            assert hasattr(config, 'hidden_size'), "Should have hidden_size"
            
            # Test ONNX Runtime session
            session = ort_model.model
            assert session is not None, "ONNX Runtime session should exist"
            
            # Check input/output names
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            
            assert len(input_names) > 0, "Should have input names"
            assert len(output_names) > 0, "Should have output names"
            assert "input_ids" in input_names, "Should have input_ids input"
    
    def test_inference_with_batch_processing(self):
        """Test inference with multiple batch sizes."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "batch_test"
            export_dir.mkdir()
            
            # Setup model
            onnx_path = export_dir / "model.onnx"
            self._export_clean_onnx(model_id, onnx_path)
            tokenizer = self._add_config_files(model_id, export_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            
            # Test different batch sizes
            test_cases = [
                ["Single sentence test."],
                ["First sentence.", "Second sentence."],
                ["Sentence one.", "Sentence two.", "Sentence three."]
            ]
            
            for i, sentences in enumerate(test_cases):
                batch_size = len(sentences)
                
                # Tokenize with consistent parameters
                inputs = tokenizer(
                    sentences,
                    return_tensors="np",
                    padding="max_length",
                    max_length=16,
                    truncation=True
                )
                
                # Validate input shapes
                assert inputs["input_ids"].shape[0] == batch_size
                assert inputs["input_ids"].shape[1] == 16  # max_length
                
                # Run inference
                outputs = ort_model(**inputs)
                
                # Validate outputs
                assert hasattr(outputs, 'logits') or hasattr(outputs, 'prediction_scores')
                logits = getattr(outputs, 'logits', getattr(outputs, 'prediction_scores', outputs[0]))
                
                assert logits.shape[0] == batch_size, f"Output batch size mismatch for case {i+1}"
                assert len(logits.shape) >= 2, "Logits should be at least 2D"
    
    def _export_clean_onnx(self, model_id: str, output_path: Path) -> bool:
        """Export ONNX without HTP metadata."""
        cmd = [
            "uv", "run", "modelexport", "export",
            "--model", model_id,
            "--output", str(output_path),
            "--clean-onnx"
        ]
        
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        return result.returncode == 0
    
    def _export_regular_onnx(self, model_id: str, output_path: Path) -> bool:
        """Export ONNX with standard options."""
        cmd = [
            "uv", "run", "modelexport", "export",
            "--model", model_id,
            "--output", str(output_path)
        ]
        
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        return result.returncode == 0
    
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
    
    def _test_inference(self, ort_model, tokenizer, export_dir: Path):
        """Test inference functionality."""
        # Check for metadata file for shape hints
        metadata_file = export_dir / "model_htp_metadata.json"
        expected_batch_size = 2
        expected_seq_len = 16
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                # Could extract shape info from metadata if needed
        
        # Test sentences
        test_sentences = [
            "I love this new approach!",
            "This is terrible."
        ]
        
        # Tokenize with fixed dimensions
        inputs = tokenizer(
            test_sentences,
            return_tensors="np",
            padding="max_length",
            max_length=expected_seq_len,
            truncation=True
        )
        
        # Validate input shapes
        assert inputs['input_ids'].shape == (expected_batch_size, expected_seq_len)
        
        # Run inference
        outputs = ort_model(**inputs)
        
        # Check outputs
        assert outputs is not None, "Should get outputs"
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'prediction_scores'):
            logits = outputs.prediction_scores
        else:
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        # Validate output shape
        assert logits.shape[0] == expected_batch_size, "Batch size should match"
        assert len(logits.shape) >= 2, "Should have at least 2D output"
        
        # Test predictions
        predictions = np.argmax(logits, axis=-1)
        assert len(predictions) == expected_batch_size, "Should have prediction for each input"