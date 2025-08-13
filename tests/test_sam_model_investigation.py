"""
Test case to investigate SAM model auto-generation failure

This test reproduces the SAM model issue to understand why Optimum's
auto-generation fails and find a generic solution.
"""

import pytest

from modelexport.core.model_input_generator import (
    generate_dummy_inputs,
    generate_dummy_inputs_from_model_path,
    get_export_config_from_model_path,
    get_model_info,
    get_supported_tasks_for_model_path,
)


class TestSAMModelInvestigation:
    """Investigate SAM model auto-generation failure"""

    def test_sam_model_info(self):
        """Get basic info about SAM model"""
        model_name = "facebook/sam-vit-base"

        # Get model info to understand the structure
        info = get_model_info(model_name)

        print(f"Model info: {info}")

        # Basic assertions
        assert "model_type" in info
        assert info["model_type"] == "sam"
        assert "architectures" in info

    def test_sam_supported_tasks(self):
        """Check what tasks are supported for SAM model"""
        model_name = "facebook/sam-vit-base"

        # Get supported tasks
        supported_tasks = get_supported_tasks_for_model_path(model_name)

        print(f"Supported tasks for SAM: {supported_tasks}")

        # Check if any tasks are supported
        assert isinstance(supported_tasks, dict)

    def test_sam_export_config_creation(self):
        """Test if we can create export config for SAM"""
        model_name = "facebook/sam-vit-base"

        try:
            # Try to create export config
            export_config = get_export_config_from_model_path(model_name)
            print(f"SAM export config created successfully: {type(export_config)}")

            # If successful, try to generate dummy inputs
            dummy_inputs = export_config.generate_dummy_inputs(framework="pt")
            print(f"SAM dummy inputs generated: {list(dummy_inputs.keys())}")

            for name, tensor in dummy_inputs.items():
                print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        except Exception as e:
            print(f"SAM export config creation failed: {e}")
            print(f"Error type: {type(e)}")
            # Don't fail the test, we're investigating
            pytest.skip(f"SAM export config creation failed: {e}")

    def test_sam_auto_generation_failure(self):
        """Reproduce the exact SAM auto-generation failure"""
        model_name = "facebook/sam-vit-base"

        try:
            # This should fail with the original error
            inputs = generate_dummy_inputs_from_model_path(model_name)
            print(f"Unexpected success! SAM inputs generated: {list(inputs.keys())}")

            for name, tensor in inputs.items():
                print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        except Exception as e:
            print(f"Expected SAM failure: {e}")
            print(f"Error type: {type(e)}")

            # Capture the specific error for analysis
            assert (
                "input_points" in str(e)
                or "bounding boxes" in str(e)
                or "4D tensor" in str(e)
            )

    def test_sam_with_manual_specs_success(self):
        """Verify that manual specs work for SAM"""
        # These are the working manual specs
        input_specs = {
            "pixel_values": {"shape": [1, 3, 1024, 1024], "dtype": "float"},
            "input_points": {"shape": [1, 1, 1, 2], "dtype": "float"},
            "input_labels": {"shape": [1, 1, 1], "dtype": "int", "range": [0, 1]},
        }

        # This should work
        inputs = generate_dummy_inputs(input_specs=input_specs)

        assert len(inputs) == 3
        assert "pixel_values" in inputs
        assert "input_points" in inputs
        assert "input_labels" in inputs

        # Verify shapes match what SAM expects
        assert inputs["pixel_values"].shape == (1, 3, 1024, 1024)
        assert inputs["input_points"].shape == (1, 1, 1, 2)
        assert inputs["input_labels"].shape == (1, 1, 1)

        print("Manual SAM specs work correctly!")


if __name__ == "__main__":
    # Run tests individually for investigation
    test_instance = TestSAMModelInvestigation()

    print("=== SAM Model Investigation ===")

    print("\n1. Getting model info...")
    test_instance.test_sam_model_info()

    print("\n2. Checking supported tasks...")
    test_instance.test_sam_supported_tasks()

    print("\n3. Testing export config creation...")
    test_instance.test_sam_export_config_creation()

    print("\n4. Reproducing auto-generation failure...")
    test_instance.test_sam_auto_generation_failure()

    print("\n5. Verifying manual specs work...")
    test_instance.test_sam_with_manual_specs_success()

    print("\n=== Investigation Complete ===")
