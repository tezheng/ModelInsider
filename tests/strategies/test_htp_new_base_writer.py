"""Unit tests for the base writer and step-aware architecture."""

import pytest

from modelexport.strategies.htp.base_writer import (
    ExportData,
    ExportStep,
    StepAwareWriter,
    step,
)
from modelexport.strategies.htp.step_data import (
    InputGenData,
    ModelPrepData,
    TensorInfo,
)


class TestStepDecorator:
    """Test the @step decorator functionality."""

    def test_step_decorator_marks_method(self):
        """Test that @step decorator properly marks methods."""

        class TestWriter(StepAwareWriter):
            @step(ExportStep.MODEL_PREP)
            def handle_model_prep(self, export_step, data):
                return 1

            def _write_default(self, export_step, data):
                return 0

        writer = TestWriter()

        # Check that the method is marked
        assert hasattr(writer.handle_model_prep, "_handles_step")
        assert writer.handle_model_prep._handles_step == ExportStep.MODEL_PREP


class TestStepAwareWriter:
    """Test the StepAwareWriter base class."""

    def test_auto_discovery_finds_step_handlers(self):
        """Test that step handlers are automatically discovered."""

        class TestWriter(StepAwareWriter):
            def __init__(self):
                self.model_prep_called = False
                self.input_gen_called = False
                super().__init__()

            @step(ExportStep.MODEL_PREP)
            def handle_model_prep(self, export_step, data):
                self.model_prep_called = True
                return 1

            @step(ExportStep.INPUT_GEN)
            def handle_input_gen(self, export_step, data):
                self.input_gen_called = True
                return 2

            def _write_default(self, export_step, data):
                return 0

        writer = TestWriter()

        # Check handlers were discovered
        assert ExportStep.MODEL_PREP in writer._step_handlers
        assert ExportStep.INPUT_GEN in writer._step_handlers
        assert len(writer._step_handlers) == 2

    def test_write_dispatches_to_correct_handler(self):
        """Test that write() dispatches to the correct handler."""

        class TestWriter(StepAwareWriter):
            def __init__(self):
                self.calls = []
                super().__init__()

            @step(ExportStep.MODEL_PREP)
            def handle_model_prep(self, export_step, data):
                self.calls.append("model_prep")
                return 1

            @step(ExportStep.INPUT_GEN)
            def handle_input_gen(self, export_step, data):
                self.calls.append("input_gen")
                return 2

            def _write_default(self, export_step, data):
                self.calls.append("default")
                return 0

        writer = TestWriter()
        data = ExportData()

        # Test dispatching
        assert writer.write(ExportStep.MODEL_PREP, data) == 1
        assert writer.calls == ["model_prep"]

        assert writer.write(ExportStep.INPUT_GEN, data) == 2
        assert writer.calls == ["model_prep", "input_gen"]

        # Test default handler for unhandled step
        assert writer.write(ExportStep.HIERARCHY, data) == 0
        assert writer.calls == ["model_prep", "input_gen", "default"]

    def test_io_protocol_methods(self):
        """Test that IO protocol methods are implemented."""

        class TestWriter(StepAwareWriter):
            def _write_default(self, export_step, data):
                return 0

        writer = TestWriter()

        # Test IO protocol
        assert not writer.readable()
        assert writer.writable()
        assert not writer.seekable()

    def test_close_calls_flush(self):
        """Test that close() calls flush()."""

        class TestWriter(StepAwareWriter):
            def __init__(self):
                self.flush_called = False
                super().__init__()

            def _write_default(self, export_step, data):
                return 0

            def flush(self):
                self.flush_called = True

        writer = TestWriter()
        writer.close()

        assert writer.flush_called


class TestExportData:
    """Test the ExportData dataclass."""

    def test_default_values(self):
        """Test that ExportData has correct default values."""
        data = ExportData()

        assert data.model_name == ""
        assert data.output_path == ""
        assert data.strategy == "htp"
        assert data.embed_hierarchy is True
        assert data.export_time == 0.0

        # Step data should be None
        assert data.model_prep is None
        assert data.input_gen is None
        assert data.hierarchy is None
        assert data.onnx_export is None
        assert data.node_tagging is None
        assert data.tag_injection is None

    def test_timestamp_property(self):
        """Test that timestamp property returns ISO format."""
        data = ExportData()
        timestamp = data.timestamp

        # Should be in ISO format
        assert "T" in timestamp
        assert timestamp.endswith("Z")

    def test_elapsed_time_property(self):
        """Test that elapsed_time calculates correctly."""
        import time

        data = ExportData()
        start = data.start_time

        # Wait a bit
        time.sleep(0.1)

        elapsed = data.elapsed_time
        assert elapsed >= 0.1
        assert elapsed < 0.2


class TestStepData:
    """Test the step data structures."""

    def test_model_prep_data(self):
        """Test ModelPrepData structure."""
        data = ModelPrepData(
            model_class="BertModel", total_modules=123, total_parameters=4400000
        )

        assert data.model_class == "BertModel"
        assert data.total_modules == 123
        assert data.total_parameters == 4400000

    def test_input_gen_data(self):
        """Test InputGenData structure."""
        data = InputGenData(
            method="auto_generated",
            model_type="bert",
            task="text-classification",
            inputs={
                "input_ids": TensorInfo(shape=[1, 128], dtype="int64"),
                "attention_mask": TensorInfo(shape=[1, 128], dtype="int64"),
            },
        )

        assert data.method == "auto_generated"
        assert data.model_type == "bert"
        assert data.task == "text-classification"
        assert len(data.inputs) == 2
        assert data.inputs["input_ids"].shape == [1, 128]
        assert data.inputs["input_ids"].dtype == "int64"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
