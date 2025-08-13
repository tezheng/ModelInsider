"""
Comprehensive test cases for model_input_generator.py

This module tests the unified input generation functionality that supports:
1. Auto-detection using Optimum's TasksManager
2. Manual specification via input_specs
3. Error handling for invalid configurations
4. Different model types and architectures
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexport.core.model_input_generator import (
    _generate_from_specs,
    generate_dummy_inputs,
    generate_dummy_inputs_from_model_path,
)


class TestInputSpecs:
    """Test manual input specification via input_specs"""

    def test_valid_input_specs_basic(self):
        """Test basic valid input specs"""
        input_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int"},
            "attention_mask": {"shape": [1, 128], "dtype": "int"},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 2
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        # Check shapes
        assert inputs["input_ids"].shape == (1, 128)
        assert inputs["attention_mask"].shape == (1, 128)

        # Check dtypes
        assert inputs["input_ids"].dtype == torch.int64
        assert inputs["attention_mask"].dtype == torch.int64

    def test_valid_input_specs_with_ranges(self):
        """Test input specs with value ranges"""
        input_specs = {
            "input_ids": {"shape": [1, 10], "dtype": "int", "range": [0, 1000]},
            "attention_mask": {"shape": [1, 10], "dtype": "int", "range": [0, 1]},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        # Check value ranges
        assert torch.all(inputs["input_ids"] >= 0)
        assert torch.all(inputs["input_ids"] <= 1000)
        assert torch.all(inputs["attention_mask"] >= 0)
        assert torch.all(inputs["attention_mask"] <= 1)

    def test_input_specs_float_dtype(self):
        """Test input specs with float dtypes"""
        input_specs = {"pixel_values": {"shape": [1, 3, 224, 224], "dtype": "float"}}

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert inputs["pixel_values"].dtype == torch.float32
        assert inputs["pixel_values"].shape == (1, 3, 224, 224)

    def test_input_specs_different_shapes(self):
        """Test various tensor shapes"""
        input_specs = {
            "seq_input": {"shape": [1, 512], "dtype": "int"},
            "image_input": {"shape": [1, 3, 224, 224], "dtype": "float"},
            "scalar_input": {"shape": [1], "dtype": "float"},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert inputs["seq_input"].shape == (1, 512)
        assert inputs["image_input"].shape == (1, 3, 224, 224)
        assert inputs["scalar_input"].shape == (1,)

    def test_input_specs_missing_shape(self):
        """Test error handling for missing shape"""
        input_specs = {
            "input_ids": {"dtype": "int"}  # Missing shape
        }

        with pytest.raises(ValueError, match="Missing 'shape' in input spec"):
            generate_dummy_inputs(input_specs=input_specs)

    def test_input_specs_invalid_dtype(self):
        """Test error handling for invalid dtype"""
        input_specs = {"input_ids": {"shape": [1, 128], "dtype": "invalid_dtype"}}

        with pytest.raises(ValueError, match="Unsupported dtype"):
            generate_dummy_inputs(input_specs=input_specs)

    def test_input_specs_invalid_shape(self):
        """Test error handling for invalid shape"""
        input_specs = {"input_ids": {"shape": "invalid_shape", "dtype": "int"}}

        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=input_specs)

    def test_input_specs_invalid_range(self):
        """Test error handling for invalid range"""
        input_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int", "range": "invalid_range"}
        }

        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=input_specs)


class TestModelAutoDetection:
    """Test automatic input generation using model path"""

    @patch(
        "modelexport.core.model_input_generator.generate_dummy_inputs_from_model_path"
    )
    def test_auto_generation_success(self, mock_generate_from_model):
        """Test successful auto-generation"""
        # Mock the function that generate_dummy_inputs calls
        mock_dummy_inputs = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128, dtype=torch.int64),
        }
        mock_generate_from_model.return_value = mock_dummy_inputs

        inputs = generate_dummy_inputs(model_name_or_path="prajjwal1/bert-tiny")

        assert len(inputs) == 2
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        mock_generate_from_model.assert_called_once_with(
            model_name_or_path="prajjwal1/bert-tiny", exporter="onnx"
        )

    @patch(
        "modelexport.core.model_input_generator.generate_dummy_inputs_from_model_path"
    )
    def test_auto_generation_task_detection_failure(self, mock_generate_from_model):
        """Test handling of task detection failure"""
        # Mock function to raise exception
        mock_generate_from_model.side_effect = ValueError(
            "Could not create export config"
        )

        with pytest.raises(ValueError, match="Could not create export config"):
            generate_dummy_inputs(model_name_or_path="invalid/model")

    def test_auto_generation_no_fallback(self):
        """Test that auto-generation doesn't fall back when specs are invalid"""
        # This should not fall back to model path
        input_specs = {
            "input_ids": {"dtype": "int"}  # Missing shape
        }

        with pytest.raises(ValueError, match="Invalid input_specs"):
            generate_dummy_inputs(
                model_name_or_path="prajjwal1/bert-tiny", input_specs=input_specs
            )

    def test_no_inputs_provided(self):
        """Test error when neither input_specs nor model_name_or_path are provided"""
        with pytest.raises(
            ValueError,
            match="Either input_specs or model_name_or_path must be provided",
        ):
            generate_dummy_inputs()


class TestUnifiedFunction:
    """Test the unified generate_dummy_inputs function"""

    def test_priority_input_specs_over_model_path(self):
        """Test that input_specs takes priority over model_name_or_path"""
        input_specs = {"custom_input": {"shape": [1, 64], "dtype": "float"}}

        # Should use input_specs even if model_name_or_path is provided
        inputs = generate_dummy_inputs(
            model_name_or_path="prajjwal1/bert-tiny", input_specs=input_specs
        )

        assert len(inputs) == 1
        assert "custom_input" in inputs
        assert inputs["custom_input"].shape == (1, 64)

    def test_exporter_parameter_ignored(self):
        """Test that exporter parameter is accepted but ignored"""
        input_specs = {"input_ids": {"shape": [1, 128], "dtype": "int"}}

        # Should work regardless of exporter value
        inputs = generate_dummy_inputs(input_specs=input_specs, exporter="onnx")

        assert len(inputs) == 1
        assert "input_ids" in inputs

    def test_kwargs_handling(self):
        """Test that additional kwargs are handled properly"""
        input_specs = {"input_ids": {"shape": [1, 128], "dtype": "int"}}

        # Should handle additional kwargs without error
        inputs = generate_dummy_inputs(
            input_specs=input_specs, some_random_kwarg="ignored"
        )

        assert len(inputs) == 1
        assert "input_ids" in inputs


class TestConfigFileIntegration:
    """Test integration with config files"""

    def test_config_file_loading(self):
        """Test loading input specs from config file format"""
        config_data = {
            "input_names": ["input_ids", "token_type_ids", "attention_mask"],
            "input_specs": {
                "input_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1000]},
                "token_type_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
                "attention_mask": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
            },
        }

        # Extract input_specs as would be done in CLI
        input_specs = config_data["input_specs"]

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 3
        assert all(name in inputs for name in config_data["input_names"])

        # Check shapes match
        for name in config_data["input_names"]:
            assert inputs[name].shape == (1, 128)

    def test_bert_model_config_format(self):
        """Test format matching export_config_bertmodel.json"""
        input_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [1, 128], "dtype": "int", "range": [0, 1]},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        # Should match BERT model expected inputs
        assert inputs["input_ids"].shape == (1, 128)
        assert inputs["token_type_ids"].shape == (1, 128)
        assert inputs["attention_mask"].shape == (1, 128)

        # Check value ranges
        assert torch.all(inputs["input_ids"] >= 0)
        assert torch.all(inputs["input_ids"] <= 1000)
        assert torch.all(inputs["token_type_ids"] >= 0)
        assert torch.all(inputs["token_type_ids"] <= 1)
        assert torch.all(inputs["attention_mask"] >= 0)
        assert torch.all(inputs["attention_mask"] <= 1)


class TestLegacyCompatibility:
    """Test backward compatibility with existing functions"""

    @patch("modelexport.core.model_input_generator.get_export_config_from_model_path")
    def test_legacy_function_calls_unified(self, mock_get_config):
        """Test that legacy function generates inputs correctly"""
        # Mock the export config
        mock_config = MagicMock()
        mock_config.generate_dummy_inputs.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 128))
        }
        mock_get_config.return_value = mock_config

        # Call legacy function
        inputs = generate_dummy_inputs_from_model_path("prajjwal1/bert-tiny")

        # Should call get_export_config_from_model_path
        mock_get_config.assert_called_once()
        assert "input_ids" in inputs

    def test_internal_specs_function(self):
        """Test internal _generate_from_specs function"""
        input_specs = {
            "input_ids": {"shape": [1, 10], "dtype": "int", "range": [0, 100]}
        }

        inputs = _generate_from_specs(input_specs)

        assert len(inputs) == 1
        assert "input_ids" in inputs
        assert inputs["input_ids"].shape == (1, 10)
        assert torch.all(inputs["input_ids"] >= 0)
        assert torch.all(inputs["input_ids"] <= 100)


class TestErrorHandling:
    """Test comprehensive error handling"""

    def test_empty_input_specs(self):
        """Test error with empty input_specs"""
        # Empty dict should work - it generates no inputs
        inputs = generate_dummy_inputs(input_specs={})
        assert inputs == {}

    def test_none_input_specs(self):
        """Test error with None input_specs"""
        with pytest.raises(
            ValueError,
            match="Either input_specs or model_name_or_path must be provided",
        ):
            generate_dummy_inputs(input_specs=None)

    def test_invalid_shape_type(self):
        """Test error with invalid shape type"""
        input_specs = {
            "input_ids": {"shape": 128, "dtype": "int"}  # Should be list
        }

        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=input_specs)

    def test_negative_shape_values(self):
        """Test error with negative shape values"""
        input_specs = {"input_ids": {"shape": [1, -128], "dtype": "int"}}

        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=input_specs)

    def test_invalid_range_format(self):
        """Test error with invalid range format"""
        input_specs = {
            "input_ids": {
                "shape": [1, 128],
                "dtype": "int",
                "range": [100, 0],
            }  # max < min
        }

        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=input_specs)


class TestModelTypes:
    """Test different model architectures"""

    def test_bert_like_model(self):
        """Test BERT-like model inputs"""
        input_specs = {
            "input_ids": {"shape": [1, 512], "dtype": "int"},
            "token_type_ids": {"shape": [1, 512], "dtype": "int"},
            "attention_mask": {"shape": [1, 512], "dtype": "int"},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 3
        assert all(tensor.shape == (1, 512) for tensor in inputs.values())
        assert all(tensor.dtype == torch.int64 for tensor in inputs.values())

    def test_gpt_like_model(self):
        """Test GPT-like model inputs"""
        input_specs = {
            "input_ids": {"shape": [1, 1024], "dtype": "int"},
            "attention_mask": {"shape": [1, 1024], "dtype": "int"},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 2
        assert inputs["input_ids"].shape == (1, 1024)
        assert inputs["attention_mask"].shape == (1, 1024)

    def test_vision_model(self):
        """Test vision model inputs"""
        input_specs = {"pixel_values": {"shape": [1, 3, 224, 224], "dtype": "float"}}

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 1
        assert inputs["pixel_values"].shape == (1, 3, 224, 224)
        assert inputs["pixel_values"].dtype == torch.float32

    def test_multimodal_model(self):
        """Test multimodal model inputs"""
        input_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int"},
            "pixel_values": {"shape": [1, 3, 224, 224], "dtype": "float"},
            "attention_mask": {"shape": [1, 128], "dtype": "int"},
        }

        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 3
        assert inputs["input_ids"].shape == (1, 128)
        assert inputs["pixel_values"].shape == (1, 3, 224, 224)
        assert inputs["attention_mask"].shape == (1, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
