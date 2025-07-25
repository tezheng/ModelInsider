"""
Comprehensive test suite for HTP Export Monitor.

This module contains unit tests, smoke tests, sanity tests, and end-to-end tests
for the HTP Export Monitor functionality, based on baseline data.
"""

import io
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp_new.export_monitor import HTPExportMonitor
from modelexport.strategies.htp_new.base_writer import ExportStep as HTPExportStep, ExportData as HTPExportData


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def baseline_console_output():
    """Load baseline console output with ANSI codes."""
    baseline_path = Path("temp/baseline/console_output_with_colors.txt")
    if baseline_path.exists():
        return baseline_path.read_text()
    # Fallback for testing
    return """🔄 Loading model and exporting: prajjwal1/bert-tiny
🧠 Using HTP \033[1m(\033[0mHierarchical Trace-and-Project\033[1m)\033[0m strategy
Auto-loading model from: prajjwal1/bert-tiny
Successfully loaded BertModel
Starting HTP export for BertModel

================================================================================
📋 STEP \033[1;36m1\033[0m/\033[1;36m8\033[0m: MODEL PREPARATION
================================================================================
✅ Model loaded: BertModel \033[1m(\033[0m\033[1;36m48\033[0m modules, \033[1;36m4.4\033[0mM parameters\033[1m)\033[0m
🎯 Export target: model.onnx
⚙️ Strategy: HTP \033[1m(\033[0mHierarchy-Preserving\033[1m)\033[0m
✅ Hierarchy attributes will be embedded in ONNX
✅ Model set to evaluation mode"""


@pytest.fixture
def baseline_metadata():
    """Load baseline metadata structure."""
    baseline_path = Path("temp/baseline/model_htp_metadata.json")
    if baseline_path.exists():
        return json.loads(baseline_path.read_text())
    # Fallback structure
    return {
        "export_context": {
            "timestamp": "2025-07-19T09:56:53Z",
            "strategy": "htp",
            "version": "1.0",
            "exporter": "HTPExporter",
            "embed_hierarchy_attributes": True
        },
        "model": {
            "name_or_path": "prajjwal1/bert-tiny",
            "class_name": "BertModel",
            "total_modules": 48,
            "total_parameters": 4385536,
            "module_types": ["BertModel", "BertEmbeddings", "BertEncoder"]
        },
        "modules": {},
        "nodes": {},
        "outputs": {},
        "report": {"steps": {}}
    }


@pytest.fixture
def sample_export_data():
    """Create sample HTPExportData for testing."""
    return HTPExportData(
        model_name="prajjwal1/bert-tiny",
        model_class="BertModel",
        total_modules=48,
        total_parameters=4385536,
        output_path="test_model.onnx",
        embed_hierarchy_attributes=True,
        hierarchy={
            "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
            "embeddings": {
                "class_name": "BertEmbeddings",
                "traced_tag": "/BertModel/BertEmbeddings"
            },
            "encoder": {
                "class_name": "BertEncoder",
                "traced_tag": "/BertModel/BertEncoder"
            }
        },
        execution_steps=36,
        output_names=["last_hidden_state", "pooler_output"],
        total_nodes=136,
        tagged_nodes={f"node_{i}": "/BertModel/BertEncoder" for i in range(136)},
        tagging_stats={
            "direct_matches": 83,
            "parent_matches": 34,
            "root_fallbacks": 19,
            "empty_tags": 0
        },
        coverage=100.0,
        export_time=2.35,
        onnx_size_mb=17.5
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestTextStyler:
    """Unit tests for TextStyler utilities."""
    
    def test_bold_cyan(self):
        """Test bold cyan formatting for numbers."""
        assert TextStyler.bold_cyan(42) == "\033[1;36m42\033[0m"
        assert TextStyler.bold_cyan(3.14) == "\033[1;36m3.14\033[0m"
        assert TextStyler.bold_cyan("100") == "\033[1;36m100\033[0m"
    
    def test_bold_parens(self):
        """Test bold parentheses formatting."""
        result = TextStyler.bold_parens("content")
        assert result == "\033[1m(\033[0mcontent\033[1m)\033[0m"
    
    def test_italic_boolean(self):
        """Test italic formatting for booleans."""
        assert TextStyler.italic_green("True") == "\033[3;92mTrue\033[0m"
        assert TextStyler.italic_red("False") == "\033[3;91mFalse\033[0m"
    
    def test_path_formatting(self):
        """Test path formatting."""
        assert TextStyler.magenta("/path/") == "\033[35m/path/\033[0m"
        assert TextStyler.bright_magenta("ClassName") == "\033[95mClassName\033[0m"
    
    def test_format_boolean(self):
        """Test boolean value formatting."""
        assert TextStyler.format_boolean(True) == "\033[3;92mTrue\033[0m"
        assert TextStyler.format_boolean(False) == "\033[3;91mFalse\033[0m"


class TestConfig:
    """Unit tests for Config class."""
    
    def test_constants(self):
        """Test configuration constants."""
        assert Config.SEPARATOR_LENGTH == 80
        assert Config.SEPARATOR_CHAR == "="
        assert Config.CONSOLE_WIDTH == 100
        assert Config.COLOR_SYSTEM == "auto"
        assert Config.FORCE_TERMINAL is True
    
    def test_suffixes(self):
        """Test file suffixes."""
        assert Config.METADATA_SUFFIX == "_htp_metadata.json"
        assert Config.REPORT_SUFFIX == "_htp_export_report.txt"
        assert Config.CONSOLE_LOG_SUFFIX == "_console.log"


class TestHTPExportData:
    """Unit tests for HTPExportData dataclass."""
    
    def test_initialization(self):
        """Test data initialization."""
        data = HTPExportData(
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000
        )
        
        assert data.model_name == "test-model"
        assert data.model_class == "TestModel"
        assert data.total_modules == 10
        assert data.total_parameters == 1000
        assert data.output_path == ""
        assert data.embed_hierarchy_attributes is True
    
    def test_defaults(self):
        """Test default values."""
        data = HTPExportData()
        
        assert data.model_name == ""
        assert data.total_modules == 0
        assert data.hierarchy == {}
        assert data.tagged_nodes == {}
        assert data.coverage == 0.0


# ============================================================================
# SMOKE TESTS
# ============================================================================

class TestSmoke:
    """Smoke tests for basic functionality."""
    
    def test_monitor_creation(self, tmp_path):
        """Test that monitor can be created without errors."""
        output_path = str(tmp_path / "test.onnx")
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        assert monitor is not None
        assert monitor.output_path == output_path
        assert monitor.model_name == "test-model"
    
    def test_context_manager(self, tmp_path):
        """Test monitor works as context manager."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            assert monitor is not None
        
        # Check files were created
        assert Path(output_path.replace('.onnx', '_htp_metadata.json')).exists()
    
    def test_basic_logging(self, tmp_path):
        """Test basic logging functionality."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        data = HTPExportData(model_name="test", model_class="Test")
        
        # Should not raise
        monitor.log_step(HTPExportStep.MODEL_PREP, data)
        
        # Check console buffer has content
        assert len(monitor.console_buffer.getvalue()) > 0


# ============================================================================
# SANITY TESTS
# ============================================================================

class TestSanity:
    """Sanity tests for expected behavior."""
    
    def test_ansi_codes_in_console(self, tmp_path, sample_export_data):
        """Test that ANSI codes are present in console output."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        monitor.log_step(HTPExportStep.MODEL_PREP, sample_export_data)
        
        console_output = monitor.console_buffer.getvalue()
        
        # Check for ANSI codes
        assert "\033[" in console_output  # Any ANSI code
        assert "\033[1;36m" in console_output  # Bold cyan
        assert "\033[1m" in console_output  # Bold
        assert "\033[0m" in console_output  # Reset
    
    def test_no_ansi_in_text_report(self, tmp_path, sample_export_data):
        """Test that text report has no ANSI codes."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path, verbose=True) as monitor:
            monitor.log_step(HTPExportStep.MODEL_PREP, sample_export_data)
        
        # Read text report
        report_path = output_path.replace('.onnx', '_htp_export_report.txt')
        report_content = Path(report_path).read_text()
        
        # Check no ANSI codes
        assert "\033[" not in report_content
        assert "\\033[" not in report_content
    
    def test_metadata_structure(self, tmp_path, sample_export_data):
        """Test metadata has expected structure."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            monitor.log_step(HTPExportStep.MODEL_PREP, sample_export_data)
            monitor.log_step(HTPExportStep.NODE_TAGGING, sample_export_data)
        
        # Read metadata
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        metadata = json.loads(Path(metadata_path).read_text())
        
        # Check required sections
        assert "export_context" in metadata
        assert "model" in metadata
        assert "modules" in metadata
        assert "nodes" in metadata
        assert "outputs" in metadata
        assert "report" in metadata
        assert "steps" in metadata["report"]
    
    def test_all_steps_logged(self, tmp_path, sample_export_data):
        """Test that all export steps can be logged."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        
        # Log all steps
        for step in HTPExportStep:
            monitor.log_step(step, sample_export_data)
        
        # Check console output has all steps
        console_output = monitor.console_buffer.getvalue()
        
        assert "MODEL PREPARATION" in console_output
        assert "INPUT GENERATION" in console_output
        assert "HIERARCHY BUILDING" in console_output
        assert "ONNX EXPORT" in console_output
        assert "NODE TAGGER CREATION" in console_output
        assert "ONNX NODE TAGGING" in console_output
        assert "SAVE ONNX MODEL" in console_output
        assert "EXPORT COMPLETE" in console_output


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for component interactions."""
    
    def test_console_metadata_report_sync(self, tmp_path, sample_export_data):
        """Test that console, metadata, and report are synchronized."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path, verbose=True) as monitor:
            # Log some steps
            monitor.log_step(HTPExportStep.MODEL_PREP, sample_export_data)
            monitor.log_step(HTPExportStep.NODE_TAGGING, sample_export_data)
        
        # Read all outputs
        console_output = monitor.console_buffer.getvalue()
        
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        metadata = json.loads(Path(metadata_path).read_text())
        
        report_path = output_path.replace('.onnx', '_htp_export_report.txt')
        report = Path(report_path).read_text()
        
        # Check synchronization
        assert sample_export_data.model_name in console_output
        assert sample_export_data.model_name in report
        assert metadata["model"]["name_or_path"] == sample_export_data.model_name
        
        # Check node tagging data
        assert "Tagged nodes: 136" in console_output
        assert "Tagged nodes: 136" in report
        assert len(metadata["nodes"]) == len(sample_export_data.tagged_nodes)
    
    def test_backward_compatibility(self, tmp_path):
        """Test backward compatibility with update() method."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path)
        
        # Use old update() interface
        monitor.update(
            HTPExportStep.MODEL_PREP,
            model_name="test-model",
            model_class="TestModel",
            total_modules=10,
            total_parameters=1000
        )
        
        # Check data is stored
        assert monitor.data.model_name == "test-model"
        assert monitor.data.model_class == "TestModel"
        assert monitor.data.steps["model_preparation"]["model_name"] == "test-model"
    
    def test_file_output_consistency(self, tmp_path, sample_export_data):
        """Test that all output files are consistent."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(
            output_path=output_path,
            verbose=True,
            enable_report=True
        ) as monitor:
            # Log complete flow
            monitor.log_step(HTPExportStep.MODEL_PREP, sample_export_data)
            monitor.log_step(HTPExportStep.INPUT_GEN, sample_export_data)
            monitor.log_step(HTPExportStep.HIERARCHY, sample_export_data)
            monitor.log_step(HTPExportStep.ONNX_EXPORT, sample_export_data)
            monitor.log_step(HTPExportStep.NODE_TAGGING, sample_export_data)
            monitor.log_step(HTPExportStep.SAVE, sample_export_data)
            monitor.log_step(HTPExportStep.COMPLETE, sample_export_data)
        
        # Check all files exist
        assert Path(output_path.replace('.onnx', '_htp_metadata.json')).exists()
        assert Path(output_path.replace('.onnx', '_htp_export_report.txt')).exists()
        assert Path(output_path.replace('.onnx', '_console.log')).exists()


# ============================================================================
# END-TO-END TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end tests simulating real export scenarios."""
    
    def test_full_export_flow(self, tmp_path):
        """Test complete export flow from start to finish."""
        output_path = str(tmp_path / "bert_tiny.onnx")
        
        # Create export data matching baseline
        export_data = HTPExportData(
            model_name="prajjwal1/bert-tiny",
            model_class="BertModel",
            total_modules=48,
            total_parameters=4385536,
            output_path=output_path,
            embed_hierarchy_attributes=True,
            hierarchy={
                "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
                "embeddings": {
                    "class_name": "BertEmbeddings",
                    "traced_tag": "/BertModel/BertEmbeddings"
                },
                "encoder": {
                    "class_name": "BertEncoder",
                    "traced_tag": "/BertModel/BertEncoder"
                },
                "encoder.layer.0": {
                    "class_name": "BertLayer",
                    "traced_tag": "/BertModel/BertEncoder/BertLayer.0"
                }
            },
            execution_steps=36,
            output_names=["last_hidden_state", "pooler_output"],
            total_nodes=136,
            tagged_nodes={},
            tagging_stats={
                "direct_matches": 83,
                "parent_matches": 34,
                "root_fallbacks": 19,
                "empty_tags": 0
            },
            coverage=100.0,
            export_time=2.35,
            onnx_size_mb=17.5
        )
        
        # Simulate tagging
        for i in range(83):
            export_data.tagged_nodes[f"node_{i}"] = "/BertModel/BertEmbeddings"
        for i in range(83, 117):
            export_data.tagged_nodes[f"node_{i}"] = "/BertModel/BertEncoder"
        for i in range(117, 136):
            export_data.tagged_nodes[f"node_{i}"] = "/BertModel"
        
        # Add step data
        export_data.steps = {
            "input_generation": {
                "model_type": "bert",
                "task": "feature-extraction",
                "inputs": {
                    "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                    "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                    "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
                }
            },
            "onnx_export": {
                "config": {
                    "opset_version": 17,
                    "do_constant_folding": True,
                    "verbose": False,
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"]
                }
            }
        }
        
        # Run export monitor
        with HTPExportMonitor(
            output_path=output_path,
            model_name="prajjwal1/bert-tiny",
            verbose=True,
            enable_report=True
        ) as monitor:
            # Execute all steps
            for step in HTPExportStep:
                monitor.log_step(step, export_data)
        
        # Verify outputs
        self._verify_console_log(output_path)
        self._verify_metadata(output_path, export_data)
        self._verify_text_report(output_path, export_data)
    
    def _verify_console_log(self, output_path: str):
        """Verify console log has correct format and ANSI codes."""
        console_path = output_path.replace('.onnx', '_console.log')
        console_content = Path(console_path).read_text()
        
        # Check ANSI codes
        assert "\033[1;36m" in console_content  # Bold cyan numbers
        assert "\033[1m(\033[0m" in console_content  # Bold parentheses
        
        # Check content
        assert "🔄 Loading model and exporting: prajjwal1/bert-tiny" in console_content
        assert "🧠 Using HTP" in console_content
        assert "STEP 1/8: MODEL PREPARATION" in console_content
        assert "STEP 8/8: EXPORT COMPLETE" in console_content
    
    def _verify_metadata(self, output_path: str, export_data: HTPExportData):
        """Verify metadata has complete information."""
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        metadata = json.loads(Path(metadata_path).read_text())
        
        # Check model info
        assert metadata["model"]["name_or_path"] == export_data.model_name
        assert metadata["model"]["class_name"] == export_data.model_class
        assert metadata["model"]["total_modules"] == export_data.total_modules
        assert metadata["model"]["total_parameters"] == export_data.total_parameters
        
        # Check nodes
        assert len(metadata["nodes"]) == len(export_data.tagged_nodes)
        
        # Check report steps
        report_steps = metadata["report"]["steps"]
        assert "model_preparation" in report_steps
        assert "node_tagging" in report_steps
        assert report_steps["node_tagging"]["total_nodes"] == export_data.total_nodes
        assert report_steps["node_tagging"]["coverage_percentage"] == export_data.coverage
    
    def _verify_text_report(self, output_path: str, export_data: HTPExportData):
        """Verify text report has complete content without ANSI codes."""
        report_path = output_path.replace('.onnx', '_htp_export_report.txt')
        report_content = Path(report_path).read_text()
        
        # Check no ANSI codes
        assert "\033[" not in report_content
        
        # Check all sections
        assert "HTP EXPORT FULL REPORT" in report_content
        assert "Loading model and exporting: prajjwal1/bert-tiny" in report_content
        assert "MODEL PREPARATION" in report_content
        assert "INPUT GENERATION" in report_content
        assert "HIERARCHY BUILDING" in report_content
        assert "ONNX EXPORT" in report_content
        assert "NODE TAGGER CREATION" in report_content
        assert "ONNX NODE TAGGING" in report_content
        assert "SAVE ONNX MODEL" in report_content
        assert "EXPORT COMPLETE" in report_content
        
        # Check data
        assert "48 modules" in report_content
        assert "4.4M parameters" in report_content
        assert "Coverage: 100.0%" in report_content
    
    def test_baseline_compatibility(self, tmp_path, baseline_console_output):
        """Test that output matches baseline format."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create monitor and generate output
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="prajjwal1/bert-tiny",
            verbose=True
        )
        
        # Get console output
        console_output = monitor.console_buffer.getvalue()
        
        # Check key patterns from baseline
        baseline_patterns = [
            r"🔄 Loading model and exporting:",
            r"🧠 Using HTP.*Hierarchical Trace-and-Project.*strategy",
            r"STEP \033\[1;36m\d+\033\[0m/\033\[1;36m\d+\033\[0m",
            r"\033\[1m\(\033\[0m.*\033\[1m\)\033\[0m",  # Bold parentheses
        ]
        
        for pattern in baseline_patterns:
            assert re.search(pattern, console_output), f"Pattern not found: {pattern}"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for export monitor."""
    
    def test_large_node_count(self, tmp_path):
        """Test performance with large number of nodes."""
        output_path = str(tmp_path / "large.onnx")
        
        # Create data with many nodes
        export_data = HTPExportData(
            model_name="large-model",
            model_class="LargeModel",
            total_nodes=10000,
            tagged_nodes={f"node_{i}": f"/Model/Layer{i%100}" for i in range(10000)}
        )
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            # This should complete quickly
            monitor.log_step(HTPExportStep.NODE_TAGGING, export_data)
        
        # Verify metadata was written
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        metadata = json.loads(Path(metadata_path).read_text())
        assert len(metadata["nodes"]) == 10000
    
    def test_console_buffer_memory(self, tmp_path):
        """Test that console buffer doesn't consume excessive memory."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(output_path=output_path, verbose=True)
        
        # Log many steps
        export_data = HTPExportData(model_name="test")
        for _ in range(1000):
            monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
        
        # Check buffer size is reasonable
        buffer_size = len(monitor.console_buffer.getvalue())
        assert buffer_size < 10_000_000  # Less than 10MB


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_output_path(self):
        """Test handling of invalid output paths."""
        # Should not raise during creation
        monitor = HTPExportMonitor(
            output_path="/invalid/path/test.onnx",
            verbose=True
        )
        
        # Should handle gracefully when finalizing
        with pytest.raises(FileNotFoundError):
            monitor.finalize()
    
    def test_missing_data_fields(self, tmp_path):
        """Test handling of incomplete export data."""
        output_path = str(tmp_path / "test.onnx")
        
        # Create minimal data
        export_data = HTPExportData()
        
        # Should not raise
        with HTPExportMonitor(output_path=output_path) as monitor:
            monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
            monitor.log_step(HTPExportStep.NODE_TAGGING, export_data)
    
    def test_unicode_handling(self, tmp_path):
        """Test handling of unicode in model names and paths."""
        output_path = str(tmp_path / "测试.onnx")
        
        export_data = HTPExportData(
            model_name="模型/测试",
            model_class="TestModel",
            hierarchy={
                "层": {"class_name": "Layer", "traced_tag": "/Model/层"}
            }
        )
        
        with HTPExportMonitor(output_path=output_path, verbose=True) as monitor:
            monitor.log_step(HTPExportStep.MODEL_PREP, export_data)
            monitor.log_step(HTPExportStep.HIERARCHY, export_data)
        
        # Check outputs handle unicode
        console_output = monitor.console_buffer.getvalue()
        assert "模型/测试" in console_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])