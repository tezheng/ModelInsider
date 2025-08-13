"""
Test enhance_exporter_config function and SAM task enhancement.

This test validates the universal task enhancement system that maps semantic task names
to Optimum-supported configurations.
"""

from modelexport.core.model_input_generator import (
    enhance_exporter_config,
    get_export_config_from_model_path,
)


class TestEnhanceExporterConfig:
    """Test the universal enhance_exporter_config function."""

    def test_enhance_exporter_config_base_functionality(self):
        """Test base functionality with no enhancements needed."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="bert",
            task="feature-extraction",
            library_name="transformers",
        )

        expected = {
            "exporter": "onnx",
            "model_type": "bert",
            "task": "feature-extraction",
            "library_name": "transformers",
            "exporter_config_kwargs": None,
        }

        assert params == expected, "Base functionality should pass through unchanged"

    def test_enhance_exporter_config_with_kwargs(self):
        """Test base functionality with config kwargs."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="bert",
            task="feature-extraction",
            library_name="transformers",
            some_param="value",
        )

        expected = {
            "exporter": "onnx",
            "model_type": "bert",
            "task": "feature-extraction",
            "library_name": "transformers",
            "exporter_config_kwargs": {"some_param": "value"},
        }

        assert params == expected, "Should preserve config kwargs"

    def test_sam_mask_generation_enhancement(self):
        """Test SAM mask-generation task enhancement."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="sam",
            task="mask-generation",
            library_name="transformers",
        )

        expected = {
            "exporter": "onnx",
            "model_type": "sam",
            "task": "feature-extraction",  # Mapped
            "library_name": "transformers",
            "exporter_config_kwargs": {"vision_encoder": False},  # Enhanced
        }

        assert params == expected, "Should map mask-generation to decoder config"

    def test_sam_feature_extraction_encoder_enhancement(self):
        """Test SAM feature-extraction task enhancement for encoder-only export."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="sam",
            task="feature-extraction",
            library_name="transformers",
        )

        expected = {
            "exporter": "onnx",
            "model_type": "sam",
            "task": "feature-extraction",  # Unchanged
            "library_name": "transformers",
            "exporter_config_kwargs": {
                "vision_encoder": True
            },  # Enhanced for encoder-only
        }

        assert params == expected, (
            "Should enhance feature-extraction to encoder-only config"
        )

    def test_sam_other_task_passthrough(self):
        """Test SAM with other tasks passes through unchanged."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="sam",
            task="image-segmentation",
            library_name="transformers",
        )

        expected = {
            "exporter": "onnx",
            "model_type": "sam",
            "task": "image-segmentation",  # Unchanged
            "library_name": "transformers",
            "exporter_config_kwargs": None,  # No enhancement
        }

        assert params == expected, "Should pass through other tasks unchanged"

    def test_sam_enhancement_preserves_existing_kwargs(self):
        """Test SAM enhancements preserve existing config kwargs."""
        params = enhance_exporter_config(
            exporter="onnx",
            model_type="sam",
            task="mask-generation",
            library_name="transformers",
            existing_param="value",
        )

        expected_kwargs = {"existing_param": "value", "vision_encoder": False}

        assert params["exporter_config_kwargs"] == expected_kwargs, (
            "Should preserve existing kwargs while adding vision_encoder"
        )


class TestSAMTaskIntegration:
    """Test SAM task enhancement integration with get_export_config_from_model_path."""

    def test_sam_mask_generation_integration(self):
        """Test mask-generation task integration creates decoder config."""
        config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="mask-generation"
        )

        # Should be configured for decoder-only
        assert config.vision_encoder == False, (
            "mask-generation should set vision_encoder=False"
        )

        # Test input generation
        inputs = config.generate_dummy_inputs()

        # Decoder inputs: embeddings and prompts
        assert "image_embeddings" in inputs, "Decoder needs image embeddings"
        assert "image_positional_embeddings" in inputs, (
            "Decoder needs positional embeddings"
        )
        assert "input_points" in inputs, "Decoder needs input points"
        assert "input_labels" in inputs, "Decoder needs input labels"

        # Should NOT have vision encoder inputs
        assert "pixel_values" not in inputs, "Decoder should not have pixel_values"

    def test_sam_feature_extraction_encoder_integration(self):
        """Test feature-extraction task integration creates encoder config."""
        config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        # Should be configured for encoder-only
        assert config.vision_encoder == True, (
            "feature-extraction should set vision_encoder=True"
        )

        # Test input generation
        inputs = config.generate_dummy_inputs()

        # Encoder input: only pixel values
        assert "pixel_values" in inputs, "Encoder needs pixel_values"

        # Should NOT have decoder inputs
        assert "image_embeddings" not in inputs, (
            "Encoder should not have embeddings input"
        )
        assert "input_points" not in inputs, "Encoder should not have prompts"
        assert "input_labels" not in inputs, "Encoder should not have labels"

    def test_sam_auto_detection_defaults_to_encoder(self):
        """Test SAM auto-detection defaults to encoder-only when task is feature-extraction."""
        # When task is explicitly feature-extraction, it should map to encoder-only
        config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )

        # Should be configured for encoder-only
        assert config.vision_encoder == True, (
            "feature-extraction should map to encoder-only"
        )

        inputs = config.generate_dummy_inputs()

        # Encoder-only inputs
        assert "pixel_values" in inputs, "Encoder should have pixel_values"
        assert "image_embeddings" not in inputs, "Encoder should not have embeddings"
        assert "input_points" not in inputs, "Encoder should not have prompts"


class TestRegressionPrevention:
    """Regression tests to ensure existing functionality is preserved."""

    def test_non_sam_models_unaffected(self):
        """Test that non-SAM models are not affected by SAM enhancements."""
        # BERT should work normally
        config = get_export_config_from_model_path(
            "prajjwal1/bert-tiny", task="feature-extraction"
        )

        # Should not have vision_encoder attribute
        assert not hasattr(config, "vision_encoder") or config.vision_encoder is None, (
            "Non-SAM models should not have vision_encoder"
        )

        inputs = config.generate_dummy_inputs()

        # Should have normal BERT inputs
        assert "input_ids" in inputs, "BERT should have input_ids"
        assert "attention_mask" in inputs, "BERT should have attention_mask"
        assert "pixel_values" not in inputs, "BERT should not have pixel_values"

    def test_sam_with_no_task_auto_detection_unchanged(self):
        """Test that SAM without explicit task still works (auto-detection)."""
        # When no task is specified, Optimum auto-detects feature-extraction
        # Our enhancement should then map it to encoder-only
        config = get_export_config_from_model_path(
            "facebook/sam-vit-base"
        )  # No task specified

        # Auto-detected task should be enhanced to encoder-only
        assert config.vision_encoder == True, (
            "Auto-detected SAM should be enhanced to encoder-only"
        )

        inputs = config.generate_dummy_inputs()
        assert "pixel_values" in inputs, "Should have pixel_values for encoder"
        assert "image_embeddings" not in inputs, "Should not have decoder inputs"

    def test_enhance_exporter_config_backwards_compatibility(self):
        """Test enhance_exporter_config is backwards compatible."""
        # Should handle all parameter combinations without error

        # Minimal parameters
        params1 = enhance_exporter_config(
            "onnx", "bert", "feature-extraction", "transformers"
        )
        assert "exporter" in params1 and "model_type" in params1

        # With kwargs
        params2 = enhance_exporter_config(
            "onnx", "bert", "feature-extraction", "transformers", param="value"
        )
        assert params2["exporter_config_kwargs"] == {"param": "value"}

        # Different model types
        params3 = enhance_exporter_config(
            "onnx", "gpt2", "text-generation", "transformers"
        )
        assert params3["task"] == "text-generation"  # Should pass through unchanged


class TestSAMTaskDocumentation:
    """Test that the SAM task mapping works as documented."""

    def test_two_sam_export_modes_produce_different_inputs(self):
        """Verify the two SAM modes produce distinct input patterns."""

        # Mode 1: Encoder-only (via feature-extraction)
        encoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="feature-extraction"
        )
        encoder_inputs = encoder_config.generate_dummy_inputs()

        # Mode 2: Decoder-only (via mask-generation)
        decoder_config = get_export_config_from_model_path(
            "facebook/sam-vit-base", task="mask-generation"
        )
        decoder_inputs = decoder_config.generate_dummy_inputs()

        # Verify distinct input patterns
        assert set(encoder_inputs.keys()) == {"pixel_values"}
        assert set(decoder_inputs.keys()) == {
            "image_embeddings",
            "image_positional_embeddings",
            "input_points",
            "input_labels",
        }

        # Verify no overlap between encoder-only and decoder-only
        assert not (set(encoder_inputs.keys()) & set(decoder_inputs.keys())), (
            "Encoder and decoder should have no common inputs"
        )

        # Verify encoder has vision input
        assert "pixel_values" in encoder_inputs, "Encoder should have pixel_values"

        # Verify decoder has embeddings and prompts
        assert "image_embeddings" in decoder_inputs, "Decoder should have embeddings"
        assert "input_points" in decoder_inputs and "input_labels" in decoder_inputs, (
            "Decoder should have prompts"
        )
