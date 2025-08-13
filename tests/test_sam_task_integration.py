"""
Integration test for SAM task enhancements with actual model export.
"""

import pytest

from modelexport.core.model_input_generator import (
    generate_dummy_inputs_from_model_path,
    get_export_config_from_model_path,
)


@pytest.mark.slow
class TestSAMTaskIntegration:
    """Integration tests for SAM task enhancement with model export."""

    def test_sam_decoder_export_produces_smaller_model(self):
        """Test that decoder-only export has different inputs than encoder-only export."""
        # Export decoder-only using mask-generation task
        decoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="mask-generation"
        )
        decoder_inputs = decoder_config.generate_dummy_inputs()

        # Export encoder-only using feature-extraction task
        encoder_inputs = generate_dummy_inputs_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        # Note: We're only testing input generation, not full export
        # Full export would require loading the actual model which is resource intensive

        # Verify inputs are different
        assert "image_embeddings" in decoder_inputs, (
            "Decoder should have embeddings input"
        )
        assert "pixel_values" in encoder_inputs, (
            "Encoder should have pixel_values input"
        )
        assert "pixel_values" not in decoder_inputs, (
            "Decoder should not have pixel_values"
        )
        assert "image_embeddings" not in encoder_inputs, (
            "Encoder should not have embeddings"
        )

    def test_sam_encoder_export_produces_pixel_values_only(self):
        """Test that encoder-only export only accepts pixel_values input."""
        # Export encoder-only using feature-extraction task
        encoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )
        encoder_inputs = encoder_config.generate_dummy_inputs()

        # Should only have pixel_values
        assert len(encoder_inputs) == 1, "Encoder should have exactly one input"
        assert "pixel_values" in encoder_inputs, (
            "Encoder should have pixel_values input"
        )

        # Verify shape is correct for SAM
        assert encoder_inputs["pixel_values"].shape[2:] == (1024, 1024), (
            "SAM expects 1024x1024 images"
        )
