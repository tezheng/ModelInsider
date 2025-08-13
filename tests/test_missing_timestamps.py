"""
Test to reproduce and verify the missing timestamp issue.

This test verifies that all export steps should have timestamps.
"""

import json
import tempfile
from pathlib import Path

import pytest

from modelexport.strategies.htp import HTPExporter


class TestMissingTimestamps:
    """Test missing timestamp issue in export steps."""

    def test_all_steps_should_have_timestamps(self):
        """Test that all export steps have timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.onnx"

            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
            )

            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check that all export steps have timestamps
            steps = metadata.get("report", {}).get("steps", {})

            expected_steps_with_timestamps = [
                "model_preparation",
                "input_generation",
                "hierarchy_building",
                "onnx_export",
                "node_tagging",
                "tag_injection",
            ]

            missing_timestamps = []
            for step_name in expected_steps_with_timestamps:
                if step_name in steps:
                    if "timestamp" not in steps[step_name]:
                        missing_timestamps.append(step_name)
                else:
                    missing_timestamps.append(f"{step_name} (step missing)")

            # This test should FAIL if timestamps are missing
            if missing_timestamps:
                pytest.fail(
                    f"Missing timestamps in steps: {missing_timestamps}\n"
                    f"Steps data: {json.dumps(steps, indent=2)}"
                )

    def test_resnet_export_timestamps(self):
        """Test ResNet export specifically for timestamp completeness."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "resnet.onnx"

            exporter = HTPExporter(verbose=False, enable_reporting=False)
            exporter.export(
                model_name_or_path="microsoft/resnet-50",
                output_path=str(output_path),
            )

            metadata_path = str(output_path).replace(".onnx", "_htp_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check specific steps that were missing timestamps
            steps = metadata.get("report", {}).get("steps", {})

            # These steps were found to be missing timestamps
            problematic_steps = ["input_generation", "onnx_export", "tag_injection"]

            for step_name in problematic_steps:
                assert step_name in steps, f"Step {step_name} is missing entirely"
                assert "timestamp" in steps[step_name], (
                    f"Step {step_name} is missing timestamp"
                )

                # Verify timestamp format
                timestamp = steps[step_name]["timestamp"]
                assert timestamp.endswith("Z"), (
                    f"Step {step_name} timestamp not in ISO format: {timestamp}"
                )
                assert "T" in timestamp, (
                    f"Step {step_name} timestamp missing T separator: {timestamp}"
                )
