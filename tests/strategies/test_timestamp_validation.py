"""
Timestamp validation tests using freezegun for time mocking.

These tests validate the timestamp handling implementation
following ADR-006: Timestamp Handling Best Practices.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from modelexport.strategies.htp.base_writer import ExportData, ExportStep
from modelexport.strategies.htp.step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    ModuleInfo,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
    TensorInfo,
)


class TestTimestampValidation:
    """Test timestamp validation with time mocking."""

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_export_data_has_uuid_and_timestamps(self):
        """Test that ExportData gets UUID and embedded timestamps work."""
        data = ExportData(model_name="test-model")
        
        # Verify UUID is generated
        assert data.export_id is not None
        assert len(data.export_id) == 36  # Standard UUID format
        assert uuid.UUID(data.export_id)  # Validates UUID format
        
        # Test step data creation with embedded timestamps
        with freeze_time("2024-01-15 12:30:45.123456"):
            data.model_prep = ModelPrepData(
                model_class="TestModel",
                total_modules=5,
                total_parameters=1000
            )
        
        # Verify timestamp is captured correctly as Unix epoch float
        timestamp = data.get_step_timestamp(ExportStep.MODEL_PREP)
        assert timestamp is not None
        # Convert to datetime for verification
        dt = datetime.fromtimestamp(timestamp, tz=UTC)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45
        assert dt.microsecond == 123456

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_timestamp_is_float(self):
        """Test that timestamps are stored as floats per ADR-006."""
        data = ExportData()
        data.model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        
        timestamp = data.get_step_timestamp(ExportStep.MODEL_PREP)
        
        # Verify it's a float (Unix epoch)
        assert isinstance(timestamp, float)
        assert timestamp == 1705321845.123456

    def test_multiple_steps_different_timestamps(self):
        """Test that different steps get different timestamps."""
        data = ExportData()
        
        # Create steps at different times
        with freeze_time("2024-01-15 12:30:45.123456"):
            data.model_prep = ModelPrepData(
                model_class="TestModel",
                total_modules=5,
                total_parameters=1000
            )
        
        with freeze_time("2024-01-15 12:30:46.234567"):
            data.input_gen = InputGenData(
                method="auto_generated",
                model_type="bert",
                task="classification"
            )
        
        with freeze_time("2024-01-15 12:30:47.345678"):
            data.hierarchy = HierarchyData(
                hierarchy={"": ModuleInfo(class_name="Root", traced_tag="")},
                execution_steps=10
            )
        
        # Verify each step has its own timestamp
        prep_time = data.get_step_timestamp(ExportStep.MODEL_PREP)
        input_time = data.get_step_timestamp(ExportStep.INPUT_GEN)
        hierarchy_time = data.get_step_timestamp(ExportStep.HIERARCHY)
        
        # Convert to datetime for verification
        prep_dt = datetime.fromtimestamp(prep_time, tz=UTC)
        input_dt = datetime.fromtimestamp(input_time, tz=UTC)
        hierarchy_dt = datetime.fromtimestamp(hierarchy_time, tz=UTC)
        
        assert prep_dt.second == 45
        assert input_dt.second == 46
        assert hierarchy_dt.second == 47
        
        # Verify timestamps are different floats
        assert prep_time == 1705321845.123456
        assert input_time == 1705321846.234567
        assert hierarchy_time == 1705321847.345678

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_step_duration_calculation(self):
        """Test duration calculation between steps."""
        data = ExportData()
        
        # Step 1 at 12:30:45.123
        data.model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        
        # Step 2 at 12:30:47.623 (2.5 seconds later)
        with freeze_time("2024-01-15 12:30:47.623456"):
            data.input_gen = InputGenData(
                method="auto_generated",
                model_type="bert"
            )
        
        # Calculate duration manually (YAGNI - no need for get_step_duration method)
        start = data.get_step_timestamp(ExportStep.MODEL_PREP)
        end = data.get_step_timestamp(ExportStep.INPUT_GEN)
        duration = end - start if start and end else None
        assert duration == pytest.approx(2.5, abs=0.001)

    def test_missing_steps_return_none(self):
        """Test that missing steps return None for timestamps."""
        data = ExportData()
        
        # No steps created yet
        assert data.get_step_timestamp(ExportStep.MODEL_PREP) is None
        # Duration calculation would also be None (no need for get_step_duration method)
        start = data.get_step_timestamp(ExportStep.MODEL_PREP)
        end = data.get_step_timestamp(ExportStep.INPUT_GEN)
        duration = end - start if start and end else None
        assert duration is None

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_all_step_types_get_timestamps(self):
        """Test that all step data types capture timestamps."""
        data = ExportData()
        
        # Create all step types
        data.model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        
        data.input_gen = InputGenData(
            method="auto_generated",
            inputs={"input_ids": TensorInfo(shape=[1, 128], dtype="int64")}
        )
        
        data.hierarchy = HierarchyData(
            hierarchy={"": ModuleInfo(class_name="Root", traced_tag="")},
            execution_steps=10
        )
        
        data.onnx_export = ONNXExportData(
            opset_version=17,
            onnx_size_mb=2.5
        )
        
        data.node_tagging = NodeTaggingData(
            total_nodes=100,
            tagged_nodes={"node1": "tag1"},
            tagging_stats={"direct_matches": 50},
            coverage=85.0
        )
        
        data.tag_injection = TagInjectionData(
            tags_injected=True
        )
        
        # Verify all steps have the same float timestamp
        expected_timestamp = 1705321845.123456
        assert data.get_step_timestamp(ExportStep.MODEL_PREP) == expected_timestamp
        assert data.get_step_timestamp(ExportStep.INPUT_GEN) == expected_timestamp
        assert data.get_step_timestamp(ExportStep.HIERARCHY) == expected_timestamp
        assert data.get_step_timestamp(ExportStep.ONNX_EXPORT) == expected_timestamp
        assert data.get_step_timestamp(ExportStep.NODE_TAGGING) == expected_timestamp
        assert data.get_step_timestamp(ExportStep.TAG_INJECTION) == expected_timestamp

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        data = ExportData()
        
        # Start time is captured on creation
        assert data.start_time == 1705321845.123456
        
        # Move time forward
        with freeze_time("2024-01-15 12:30:50.123456"):
            # Elapsed time should be 5 seconds
            assert data.elapsed_time == 5.0

    @patch('uuid.uuid4')
    def test_deterministic_uuid_for_testing(self, mock_uuid):
        """Test that UUID can be mocked for deterministic testing."""
        # Mock UUID for predictable testing
        mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
        
        data = ExportData()
        
        assert data.export_id == "12345678-1234-5678-1234-567812345678"

    def test_timestamp_is_utc_epoch(self):
        """Test that all timestamps are stored as UTC Unix epoch floats."""
        data = ExportData()
        data.model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        
        timestamp = data.get_step_timestamp(ExportStep.MODEL_PREP)
        
        # Verify it's a float (Unix epoch)
        assert isinstance(timestamp, float)
        
        # Verify it's a reasonable timestamp (between 2020 and 2030)
        assert 1577836800 < timestamp < 1893456000  # 2020-01-01 to 2030-01-01

    @freeze_time("2024-01-15 12:30:45.999999")
    def test_timestamp_precision(self):
        """Test that timestamp preserves microsecond precision."""
        data = ExportData()
        data.model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        
        # Verify microsecond precision is preserved in float
        timestamp = data.get_step_timestamp(ExportStep.MODEL_PREP)
        assert timestamp == 1705321845.999999
        
        # Test edge case: microseconds < 1000
        with freeze_time("2024-01-15 12:30:45.000500"):  # 500 microseconds
            data.input_gen = InputGenData(method="auto")
            timestamp = data.get_step_timestamp(ExportStep.INPUT_GEN)
            assert timestamp == 1705321845.0005

    def test_data_locality_principle(self):
        """Test that timestamps are embedded in step data (data locality)."""
        data = ExportData()
        
        # Create step data
        model_prep = ModelPrepData(
            model_class="TestModel",
            total_modules=5,
            total_parameters=1000
        )
        data.model_prep = model_prep
        
        # Verify timestamp is embedded in step data, not stored separately
        assert hasattr(model_prep, 'timestamp')
        assert model_prep.timestamp is not None
        assert isinstance(model_prep.timestamp, float)  # Unix epoch float
        
        # Verify the accessor method reads from embedded timestamp
        step_timestamp = data.get_step_timestamp(ExportStep.MODEL_PREP)
        assert step_timestamp == model_prep.timestamp


class TestTimestampConsistency:
    """Test timestamp consistency across the system."""

    @freeze_time("2024-01-15 12:30:45.123456")
    def test_export_monitor_timestamp_consistency(self):
        """Test that ExportMonitor doesn't interfere with embedded timestamps."""
        from modelexport.strategies.htp.export_monitor import HTPExportMonitor
        
        # This test ensures that the monitor correctly uses embedded timestamps
        # and doesn't try to override them with its own timestamp generation
        
        with patch('tempfile.mkdtemp', return_value='/tmp/test'):
            monitor = HTPExportMonitor(
                output_path="/tmp/test.onnx",
                model_name="test-model",
                verbose=False,
                enable_report=False
            )
            
            # Update with step data
            monitor.update(ExportStep.MODEL_PREP, 
                          model_class="TestModel",
                          total_modules=5,
                          total_parameters=1000)
            
            # Verify timestamp was captured in step data as float
            timestamp = monitor.data.get_step_timestamp(ExportStep.MODEL_PREP)
            assert timestamp == 1705321845.123456
            
            # Verify the step data has the embedded timestamp
            assert monitor.data.model_prep is not None
            assert isinstance(monitor.data.model_prep.timestamp, float)  # Unix epoch float