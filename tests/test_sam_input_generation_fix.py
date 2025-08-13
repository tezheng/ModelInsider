"""
Test SAM model input generation fix (TEZ-48).

This test validates that SAM models generate pixel_values instead of embeddings
for full model export, ensuring the vision encoder is included in the trace.
"""

import pytest
import torch

from modelexport.core.model_input_generator import generate_dummy_inputs_from_model_path


class TestSAMInputGenerationFix:
    """Test TEZ-48 fix for SAM model input generation."""

    def test_sam_generates_pixel_values(self):
        """Test that SAM models generate pixel_values for encoder-only export."""
        # Generate inputs for SAM model with feature-extraction task (encoder-only)
        inputs = generate_dummy_inputs_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        # Should generate pixel_values for encoder-only export
        assert "pixel_values" in inputs, "SAM encoder should generate pixel_values"
        assert "image_embeddings" not in inputs, (
            "SAM encoder should not have embeddings input"
        )
        assert "image_positional_embeddings" not in inputs, (
            "SAM encoder should not have positional embeddings input"
        )

        # Encoder-only should NOT have decoder inputs
        assert "input_points" not in inputs, "SAM encoder should not have input_points"
        assert "input_labels" not in inputs, "SAM encoder should not have input_labels"

        # Validate tensor shapes and types
        pixel_values = inputs["pixel_values"]
        assert pixel_values.shape == torch.Size([2, 3, 1024, 1024]), (
            f"Unexpected pixel_values shape: {pixel_values.shape}"
        )
        assert pixel_values.dtype == torch.float32, (
            f"Unexpected pixel_values dtype: {pixel_values.dtype}"
        )

    def test_sam_inputs_work_with_model(self):
        """Test that generated SAM inputs work with the respective model components."""
        from transformers import SamModel

        # Test 1: Encoder inputs with vision encoder
        encoder_inputs = generate_dummy_inputs_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        model = SamModel.from_pretrained("facebook/sam-vit-base")
        model.eval()

        with torch.no_grad():
            # Test encoder inputs work with vision encoder
            try:
                pixel_values = encoder_inputs["pixel_values"][:1]  # Use batch_size=1
                image_embeddings = model.vision_encoder(pixel_values)

                # Validate encoder output
                assert hasattr(image_embeddings, "last_hidden_state") or isinstance(
                    image_embeddings, torch.Tensor
                ), "Vision encoder should produce embeddings"

            except Exception as e:
                pytest.fail(
                    f"Generated encoder inputs failed vision encoder inference: {e}"
                )

        # Test 2: Decoder inputs structure
        decoder_inputs = generate_dummy_inputs_from_model_path(
            "facebook/sam-vit-base", task="mask-generation"
        )

        # Validate decoder inputs have correct structure
        assert "image_embeddings" in decoder_inputs, (
            "Decoder should have image_embeddings"
        )
        assert "image_positional_embeddings" in decoder_inputs, (
            "Decoder should have positional embeddings"
        )
        assert "input_points" in decoder_inputs, "Decoder should have input_points"
        assert "input_labels" in decoder_inputs, "Decoder should have input_labels"

        # Validate shapes are consistent
        img_emb = decoder_inputs["image_embeddings"]
        pos_emb = decoder_inputs["image_positional_embeddings"]
        assert img_emb.shape == pos_emb.shape, (
            "Image and positional embeddings should have same shape"
        )

    def test_sam_export_task_support(self):
        """Test that SAM export supports the two task modes via input generation."""
        # Test encoder-only export configuration
        from modelexport.core.model_input_generator import (
            get_export_config_from_model_path,
        )

        # Test 1: feature-extraction task -> encoder-only
        encoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )
        assert hasattr(encoder_config, "vision_encoder"), (
            "Should have vision_encoder attribute"
        )
        assert encoder_config.vision_encoder == True, (
            "feature-extraction should set vision_encoder=True"
        )

        # Test 2: mask-generation task -> decoder-only
        decoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="mask-generation"
        )
        assert hasattr(decoder_config, "vision_encoder"), (
            "Should have vision_encoder attribute"
        )
        assert decoder_config.vision_encoder == False, (
            "mask-generation should set vision_encoder=False"
        )

        # Test that configs generate appropriate inputs
        encoder_inputs = encoder_config.generate_dummy_inputs()
        decoder_inputs = decoder_config.generate_dummy_inputs()

        # Encoder should only have pixel_values
        assert "pixel_values" in encoder_inputs, "Encoder should have pixel_values"
        assert len(encoder_inputs) == 1, "Encoder should only have one input"

        # Decoder should have embeddings and prompts
        assert "image_embeddings" in decoder_inputs, "Decoder should have embeddings"
        assert "input_points" in decoder_inputs, "Decoder should have prompts"


class TestSAMInputGenerationRegression:
    """Regression test to ensure we don't revert to embedding generation."""

    def test_sam_does_not_generate_embeddings(self):
        """Regression test: Ensure SAM encoder doesn't generate embeddings."""
        # Test encoder-only export (feature-extraction task)
        inputs = generate_dummy_inputs_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        # These inputs would indicate wrong export mode
        decoder_inputs = [
            "image_embeddings",
            "image_positional_embeddings",
            "input_points",
            "input_labels",
        ]

        for decoder_input in decoder_inputs:
            assert decoder_input not in inputs, (
                f"Encoder should not have decoder input: {decoder_input}"
            )

        # Ensure we have the correct encoder input
        assert "pixel_values" in inputs, "Encoder should have pixel_values input"

    def test_sam_separate_component_export(self):
        """Test how to properly export SAM components separately using Optimum."""
        from optimum.exporters.onnx.model_configs import SamOnnxConfig
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("facebook/sam-vit-base")

        # Test 1: Vision encoder export (Optimum's intended way)
        encoder_config = SamOnnxConfig(
            config=config,
            task="feature-extraction",
            vision_encoder=True,  # This is the key!
        )
        encoder_inputs = encoder_config.generate_dummy_inputs(framework="pt")

        # Vision encoder should only have pixel_values input
        assert "pixel_values" in encoder_inputs, (
            "Vision encoder should have pixel_values"
        )
        assert "image_embeddings" not in encoder_inputs, (
            "Vision encoder should not have embeddings as input"
        )
        assert "input_points" not in encoder_inputs, (
            "Vision encoder should not have prompts"
        )

        # Test 2: Mask decoder export (default Optimum behavior)
        decoder_config = SamOnnxConfig(
            config=config,
            task="feature-extraction",
            vision_encoder=False,  # Default, but explicit for clarity
        )
        decoder_inputs = decoder_config.generate_dummy_inputs(framework="pt")

        # Mask decoder should have embeddings and prompts
        assert "image_embeddings" in decoder_inputs, "Mask decoder needs embeddings"
        assert "image_positional_embeddings" in decoder_inputs, (
            "Mask decoder needs positional embeddings"
        )
        assert "input_points" in decoder_inputs, "Mask decoder needs points"
        assert "input_labels" in decoder_inputs, "Mask decoder needs labels"
        assert "pixel_values" not in decoder_inputs, (
            "Mask decoder should not have pixel_values"
        )
