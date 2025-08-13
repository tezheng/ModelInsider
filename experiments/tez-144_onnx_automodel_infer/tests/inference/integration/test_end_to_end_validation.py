#!/usr/bin/env python3
"""
End-to-end validation tests for ONNX export and Optimum integration.

This module contains critical integration tests that validate the complete 
production workflow from model export to deployment with Optimum.

Tests extracted from final_validation_test.py
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


class TestEndToEndValidation:
    """End-to-end workflow validation tests."""
    
    def test_complete_export_config_optimum_workflow(self):
        """Test complete workflow: export → config → optimum loading."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "validation"
            export_dir.mkdir()
            
            # Step 1: Export clean ONNX
            onnx_path = export_dir / "model.onnx"
            success = self._export_clean_onnx(model_id, onnx_path)
            assert success, "ONNX export should succeed"
            assert onnx_path.exists(), "ONNX file should be created"
            
            # Verify file size
            file_size_mb = onnx_path.stat().st_size / 1024 / 1024
            assert file_size_mb > 0.1, "ONNX file should be substantial"
            
            # Step 2: Verify Optimum fails WITHOUT config
            with pytest.raises(Exception):
                ORTModelForSequenceClassification.from_pretrained(export_dir)
            
            # Step 3: Add config files
            config_added = self._add_config_files(model_id, export_dir)
            assert config_added, "Config files should be added successfully"
            
            # Verify directory structure
            required_files = ["model.onnx", "config.json"]
            for file_name in required_files:
                assert (export_dir / file_name).exists(), f"{file_name} should exist"
            
            # Calculate config overhead
            config_size = sum(
                f.stat().st_size 
                for f in export_dir.glob("*") 
                if f.name != "model.onnx"
            )
            onnx_size = onnx_path.stat().st_size
            overhead_percent = (config_size / onnx_size) * 100
            assert overhead_percent < 10.0, "Config overhead should be minimal"
            
            # Step 4: Load with Optimum WITH config
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            assert ort_model is not None, "Model should load successfully"
            assert hasattr(ort_model, 'config'), "Model should have config"
            assert ort_model.config.model_type == "bert", "Should be BERT model"
    
    def test_production_ready_pattern_validation(self):
        """Validate the production-ready implementation pattern."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "production_test"
            export_dir.mkdir()
            
            # Follow the validated pattern
            onnx_path = export_dir / "model.onnx"
            
            # 1. Export with clean-onnx flag
            assert self._export_clean_onnx(model_id, onnx_path)
            
            # 2. Add config using AutoConfig pattern
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(export_dir)
            
            # 3. Add tokenizer using AutoTokenizer pattern
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(export_dir)
            
            # 4. Verify Optimum loading works
            ort_model = ORTModelForSequenceClassification.from_pretrained(export_dir)
            
            # Validate model properties
            assert isinstance(ort_model, ORTModelForSequenceClassification)
            assert ort_model.config.model_type == "bert"
            assert hasattr(ort_model, 'model'), "Should have ONNX model"
    
    def test_export_command_options(self):
        """Test various export command options work correctly."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            
            # Test clean-onnx option
            clean_onnx_path = export_dir / "clean_model.onnx"
            cmd = [
                "uv", "run", "modelexport", "export",
                "--model", model_id,
                "--output", str(clean_onnx_path),
                "--clean-onnx"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent.parent
            )
            
            assert result.returncode == 0, f"Export failed: {result.stderr}"
            assert clean_onnx_path.exists(), "Clean ONNX should be exported"
            
            # Test regular export
            regular_onnx_path = export_dir / "regular_model.onnx"
            cmd_regular = [
                "uv", "run", "modelexport", "export",
                "--model", model_id,
                "--output", str(regular_onnx_path)
            ]
            
            result_regular = subprocess.run(
                cmd_regular,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent.parent
            )
            
            assert result_regular.returncode == 0, f"Regular export failed: {result_regular.stderr}"
            assert regular_onnx_path.exists(), "Regular ONNX should be exported"
    
    def test_config_overhead_analysis(self):
        """Test that config files add minimal overhead."""
        model_id = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "overhead_test"
            export_dir.mkdir()
            
            # Export ONNX
            onnx_path = export_dir / "model.onnx"
            self._export_clean_onnx(model_id, onnx_path)
            
            onnx_size = onnx_path.stat().st_size
            
            # Add config files
            self._add_config_files(model_id, export_dir)
            
            # Calculate total config size
            config_files = [f for f in export_dir.glob("*") if f.name != "model.onnx"]
            total_config_size = sum(f.stat().st_size for f in config_files)
            
            # Validate overhead
            overhead_percent = (total_config_size / onnx_size) * 100
            
            # Config files should be minimal compared to model
            assert overhead_percent < 10.0, f"Config overhead too high: {overhead_percent:.2f}%"
            assert len(config_files) >= 2, "Should have at least config.json and tokenizer files"
            
            # Log results for analysis
            print(f"ONNX size: {onnx_size / 1024 / 1024:.2f} MB")
            print(f"Config size: {total_config_size / 1024:.1f} KB")
            print(f"Overhead: {overhead_percent:.3f}%")
    
    def _export_clean_onnx(self, model_id: str, output_path: Path) -> bool:
        """Export ONNX without HTP metadata for Optimum compatibility."""
        cmd = [
            "uv", "run", "modelexport", "export",
            "--model", model_id,
            "--output", str(output_path),
            "--clean-onnx"
        ]
        
        # Use project root directory for command execution
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        return result.returncode == 0
    
    def _add_config_files(self, model_id: str, output_dir: Path) -> bool:
        """Add config files for Optimum compatibility."""
        try:
            # Add config.json (always required)
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(output_dir)
            
            # Add tokenizer (conditional)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(output_dir)
            
            return True
        except Exception:
            return False