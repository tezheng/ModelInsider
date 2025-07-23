"""
Comprehensive integration tests for HTP exporter with input generation

This test suite validates the complete integration and ensures all scenarios work correctly.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from modelexport.core.model_input_generator import generate_dummy_inputs
from modelexport.strategies.htp_new import export_with_htp_reporting


class TestComprehensiveIntegration:
    """Test comprehensive integration scenarios"""
    
    def test_config_file_vs_manual_specs(self):
        """Test that config file and manual specs produce similar results"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Test 1: Using config file style input_specs
        config_input_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 128], "dtype": "int", "range": [0, 1]}
        }
        
        # Test 2: Using manual tensor creation
        manual_inputs = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "token_type_ids": torch.randint(0, 1, (1, 128)),
            "attention_mask": torch.ones(1, 128, dtype=torch.int64)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export with config-style specs
            config_result = export_with_htp_reporting(
                model=model,
                output_path=os.path.join(temp_dir, "config_test.onnx"),
                model_name_or_path="prajjwal1/bert-tiny",
                input_specs=config_input_specs,
                verbose=False
            )
            
            # Export with auto-generation
            auto_result = export_with_htp_reporting(
                model=model,
                output_path=os.path.join(temp_dir, "auto_test.onnx"),
                model_name_or_path="prajjwal1/bert-tiny",
                verbose=False
            )
            
            # Results should be similar
            assert config_result["coverage_percentage"] == 100.0
            assert auto_result["coverage_percentage"] == 100.0
            assert config_result["empty_tags"] == 0
            assert auto_result["empty_tags"] == 0
            assert config_result["hierarchy_modules"] == auto_result["hierarchy_modules"]
            
            # Both files should exist
            assert Path(os.path.join(temp_dir, "config_test.onnx")).exists()
            assert Path(os.path.join(temp_dir, "auto_test.onnx")).exists()
    
    def test_different_sequence_lengths(self):
        """Test that different sequence lengths work correctly"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Test different sequence lengths
        test_lengths = [32, 64, 128, 256]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for seq_len in test_lengths:
                input_specs = {
                    "input_ids": {"shape": [1, seq_len], "dtype": "int", "range": [0, 1000]},
                    "token_type_ids": {"shape": [1, seq_len], "dtype": "int", "range": [0, 1]},
                    "attention_mask": {"shape": [1, seq_len], "dtype": "int", "range": [0, 1]}
                }
                
                result = export_with_htp_reporting(
                    model=model,
                    output_path=os.path.join(temp_dir, f"test_seq_{seq_len}.onnx"),
                    model_name_or_path="prajjwal1/bert-tiny",
                    input_specs=input_specs,
                    verbose=False
                )
                
                # All should succeed with full coverage
                assert result["coverage_percentage"] == 100.0
                assert result["empty_tags"] == 0
                assert result["hierarchy_modules"] == 45  # BERT tiny has 45 modules (updated count)
    
    def test_error_recovery(self):
        """Test error handling and recovery"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test 1: Invalid input specs should fail gracefully
            invalid_specs = {
                "input_ids": {"dtype": "int", "range": [0, 1000]}  # Missing shape
            }
            
            with pytest.raises(ValueError, match="Invalid input_specs"):
                export_with_htp_reporting(
                    model=model,
                    output_path=os.path.join(temp_dir, "invalid_test.onnx"),
                    model_name_or_path="prajjwal1/bert-tiny",
                    input_specs=invalid_specs,
                    verbose=False
                )
            
            # Test 2: No inputs should fail appropriately
            with pytest.raises(ValueError, match="Either input_specs or model_name_or_path must be provided"):
                export_with_htp_reporting(
                    model=model,
                    output_path=os.path.join(temp_dir, "no_inputs_test.onnx"),
                    verbose=False
                )
    
    def test_input_generator_standalone(self):
        """Test standalone input generator with various specifications"""
        
        # Test 1: BERT-style inputs
        bert_specs = {
            "input_ids": {"shape": [1, 512], "dtype": "int", "range": [0, 30000]},
            "token_type_ids": {"shape": [1, 512], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 512], "dtype": "int", "range": [0, 1]}
        }
        
        bert_inputs = generate_dummy_inputs(input_specs=bert_specs)
        
        assert len(bert_inputs) == 3
        assert bert_inputs["input_ids"].shape == (1, 512)
        assert bert_inputs["token_type_ids"].shape == (1, 512)
        assert bert_inputs["attention_mask"].shape == (1, 512)
        
        # Check value ranges
        assert torch.all(bert_inputs["input_ids"] >= 0)
        assert torch.all(bert_inputs["input_ids"] <= 30000)
        assert torch.all(bert_inputs["token_type_ids"] >= 0)
        assert torch.all(bert_inputs["token_type_ids"] <= 1)
        assert torch.all(bert_inputs["attention_mask"] >= 0)
        assert torch.all(bert_inputs["attention_mask"] <= 1)
        
        # Test 2: Vision-style inputs
        vision_specs = {
            "pixel_values": {"shape": [1, 3, 224, 224], "dtype": "float"}
        }
        
        vision_inputs = generate_dummy_inputs(input_specs=vision_specs)
        
        assert len(vision_inputs) == 1
        assert vision_inputs["pixel_values"].shape == (1, 3, 224, 224)
        assert vision_inputs["pixel_values"].dtype == torch.float32
        
        # Test 3: Mixed inputs
        mixed_specs = {
            "text_input": {"shape": [1, 256], "dtype": "int", "range": [0, 50000]},
            "image_input": {"shape": [1, 3, 224, 224], "dtype": "float"},
            "mask_input": {"shape": [1, 256], "dtype": "int", "range": [0, 1]}
        }
        
        mixed_inputs = generate_dummy_inputs(input_specs=mixed_specs)
        
        assert len(mixed_inputs) == 3
        assert mixed_inputs["text_input"].shape == (1, 256)
        assert mixed_inputs["image_input"].shape == (1, 3, 224, 224)
        assert mixed_inputs["mask_input"].shape == (1, 256)
        
        # Check dtypes
        assert mixed_inputs["text_input"].dtype == torch.int64
        assert mixed_inputs["image_input"].dtype == torch.float32
        assert mixed_inputs["mask_input"].dtype == torch.int64
    
    def test_export_config_format(self):
        """Test the exact format used in export_config_bertmodel.json"""
        
        # This should match the format in export_config_bertmodel.json
        config_format_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 128], "dtype": "int", "range": [0, 1]}
        }
        
        inputs = generate_dummy_inputs(input_specs=config_format_specs)
        
        # Should match BERT expectations
        assert len(inputs) == 3
        assert all(name in inputs for name in ["input_ids", "token_type_ids", "attention_mask"])
        assert all(tensor.shape == (1, 128) for tensor in inputs.values())
        assert all(tensor.dtype == torch.int64 for tensor in inputs.values())
        
        # Check ranges
        assert torch.all(inputs["input_ids"] >= 0) and torch.all(inputs["input_ids"] <= 1000)
        assert torch.all(inputs["token_type_ids"] >= 0) and torch.all(inputs["token_type_ids"] <= 1)
        assert torch.all(inputs["attention_mask"] >= 0) and torch.all(inputs["attention_mask"] <= 1)
    
    def test_output_consistency(self):
        """Test that outputs are consistent between runs"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Fixed input specs for reproducibility
        input_specs = {
            "input_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 64], "dtype": "int", "range": [0, 1]}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run export twice
            result1 = export_with_htp_reporting(
                model=model,
                output_path=os.path.join(temp_dir, "consistency_test1.onnx"),
                model_name_or_path="prajjwal1/bert-tiny",
                input_specs=input_specs,
                verbose=False
            )
            
            result2 = export_with_htp_reporting(
                model=model,
                output_path=os.path.join(temp_dir, "consistency_test2.onnx"),
                model_name_or_path="prajjwal1/bert-tiny",
                input_specs=input_specs,
                verbose=False
            )
            
            # Key metrics should be consistent
            assert result1["coverage_percentage"] == result2["coverage_percentage"]
            assert result1["empty_tags"] == result2["empty_tags"]
            assert result1["hierarchy_modules"] == result2["hierarchy_modules"]
            
            # Both should achieve perfect coverage
            assert result1["coverage_percentage"] == 100.0
            assert result2["coverage_percentage"] == 100.0
            assert result1["empty_tags"] == 0
            assert result2["empty_tags"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])