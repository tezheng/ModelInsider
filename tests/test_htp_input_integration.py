"""
Integration tests for HTP exporter with input generation

Tests the complete integration between model_input_generator and HTP exporter
"""

import os
import tempfile
from pathlib import Path

import pytest

from modelexport.strategies.htp.htp_exporter import HTPExporter


class TestHTPInputIntegration:
    """Test HTP exporter integration with input generation"""
    
    def test_htp_with_manual_input_specs(self):
        """Test HTP exporter with manual input specifications"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Define input specs
        input_specs = {
            "input_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 64], "dtype": "int", "range": [0, 1]}
        }
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Test export with input specs (HTPExporter handles input generation internally)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_manual_specs.onnx")
            
            result = exporter.export(
                model=model,
                output_path=output_path,
                input_specs=input_specs,
                opset_version=17
            )
            
            # Verify export success
            assert Path(output_path).exists()
            assert result["coverage_percentage"] == 100.0  # 100% coverage
            assert result.get("empty_tags", 0) == 0
    
    def test_htp_with_auto_input_generation(self):
        """Test HTP exporter with automatic input generation"""
        from transformers import AutoModel, AutoTokenizer
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Test export with auto-generation (HTPExporter generates inputs automatically)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_auto_gen.onnx")
            
            result = exporter.export(
                model=model,
                output_path=output_path,
                model_name_or_path="prajjwal1/bert-tiny",
                opset_version=17
            )
            
            # Verify export success
            assert Path(output_path).exists()
            assert result["coverage_percentage"] == 100.0  # 100% coverage
            assert result.get("empty_tags", 0) == 0
    
    def test_input_specs_priority(self):
        """Test that input_specs takes priority over model_name_or_path"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Define smaller input specs than would be auto-generated
        input_specs = {
            "input_ids": {"shape": [1, 32], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 32], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 32], "dtype": "int", "range": [0, 1]}
        }
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Test export - should use input_specs not auto-generation
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_specs_priority.onnx")
            
            result = exporter.export(
                model=model,
                output_path=output_path,
                model_name_or_path="prajjwal1/bert-tiny",
                input_specs=input_specs,  # This should take priority
                opset_version=17
            )
            
            # Verify export success with expected input size
            assert Path(output_path).exists()
            assert result["coverage_percentage"] == 100.0  # 100% coverage
    
    def test_invalid_input_specs_error(self):
        """Test error handling for invalid input specs"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Define invalid input specs (missing shape)
        invalid_input_specs = {
            "input_ids": {"dtype": "int", "range": [0, 1000]}  # Missing shape
        }
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Should raise error for invalid specs
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_invalid.onnx")
            
            with pytest.raises(ValueError, match="shape"):
                # This should fail inside the exporter when processing invalid specs
                exporter.export(
                    model=model,
                    output_path=output_path,
                    input_specs=invalid_input_specs,
                    opset_version=17
                )
    
    def test_mixed_input_types(self):
        """Test with various input tensor types"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Define mixed input types
        input_specs = {
            "input_ids": {"shape": [2, 48], "dtype": "int", "range": [0, 1000]},
            "attention_mask": {"shape": [2, 48], "dtype": "float", "range": [0.0, 1.0]}
        }
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Test export with mixed input types
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_mixed_types.onnx")
            
            result = exporter.export(
                model=model,
                output_path=output_path,
                input_specs=input_specs,
                opset_version=17
            )
            
            # Verify export success
            assert Path(output_path).exists()
            assert result["coverage_percentage"] == 100.0
    
    def test_integration_with_cli_inputs(self):
        """Test integration with CLI-style input handling"""
        from transformers import AutoModel
        
        # Load model  
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Simulate CLI input specs (as would come from JSON file)
        cli_input_specs = {
            "input_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 30000]},
            "token_type_ids": {"shape": [1, 64], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 64], "dtype": "int", "range": [0, 1]}
        }
        
        # Create exporter
        exporter = HTPExporter(verbose=False, enable_reporting=False)
        
        # Export with CLI input specs
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_cli_integration.onnx")
            
            result = exporter.export(
                model=model,
                output_path=output_path,
                input_specs=cli_input_specs,
                opset_version=17
            )
            
            # Should succeed with 100% coverage
            assert Path(output_path).exists()
            assert result["coverage_percentage"] == 100.0
            assert "htp" in result["strategy"]  # Should be htp_integrated or similar
    
    def test_edge_case_batch_sizes(self):
        """Test with various batch sizes"""
        from transformers import AutoModel
        
        # Load model
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Define input specs with different batch sizes
            input_specs = {
                "input_ids": {"shape": [batch_size, 32], "dtype": "int", "range": [0, 1000]},
                "attention_mask": {"shape": [batch_size, 32], "dtype": "int", "range": [0, 1]}
            }
            
            # Create exporter
            exporter = HTPExporter(verbose=False, enable_reporting=False)
            
            # Test export with different batch sizes
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"test_batch_{batch_size}.onnx")
                
                result = exporter.export(
                    model=model,
                    output_path=output_path,
                    input_specs=input_specs,
                    opset_version=17
                )
                
                # All batch sizes should work
                assert Path(output_path).exists()
                assert result["coverage_percentage"] == 100.0