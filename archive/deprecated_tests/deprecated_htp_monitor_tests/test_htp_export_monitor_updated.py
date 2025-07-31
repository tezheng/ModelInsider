"""
Updated test cases for HTP Export Monitor with proper mocking.

This module tests the integration of ExportMonitor with HTPExporter
ensuring proper 7-step flow, correct formatting, and proper mocking.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp.base_writer import ExportData as HTPExportData
from modelexport.strategies.htp.base_writer import ExportStep as HTPExportStep
from modelexport.strategies.htp.export_monitor import HTPExportMonitor
from modelexport.strategies.htp.htp_exporter import HTPExporter


# Test Models
class SimpleModel(nn.Module):
    """Simple test model for export testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, input=None, **kwargs):
        """Forward method that accepts keyword arguments."""
        x = input if input is not None else kwargs.get('x')
        return self.linear(x)


class ComplexModel(nn.Module):
    """Complex model with hierarchy for testing."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            ) for _ in range(2)
        ])
        self.decoder = nn.Linear(10, 5)
    
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.decoder(x)


# Fixtures
@pytest.fixture
def simple_model():
    """Fixture providing a simple test model."""
    return SimpleModel()


@pytest.fixture
def complex_model():
    """Fixture providing a complex test model."""
    return ComplexModel()


@pytest.fixture
def mock_onnx_model():
    """Mock ONNX model with proper node structure."""
    mock_model = MagicMock()
    nodes = []
    
    # Create diverse node types
    node_types = ["MatMul", "Add", "Relu", "LayerNorm", "Gather", "Softmax"]
    for i in range(15):
        node = MagicMock()
        node.name = f"node_{i}"
        node.op_type = node_types[i % len(node_types)]
        node.input = [f"input_{i}"]
        node.output = [f"output_{i}"]
        node.attribute = []
        nodes.append(node)
    
    mock_model.graph.node = nodes
    return mock_model


@pytest.fixture
def sample_hierarchy():
    """Sample hierarchy data matching bert-tiny structure."""
    return {
        "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
        "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/Embeddings"},
        "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/Encoder"},
        "encoder.layer": {"class_name": "ModuleList", "traced_tag": "/BertModel/Encoder/Layer"},
        "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/0"},
        "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention"},
        "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Self"},
        "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Output"},
        "encoder.layer.1": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/1"},
    }


@pytest.fixture
def sample_tagged_nodes():
    """Sample tagged nodes matching hierarchy."""
    return {
        "MatMul_0": "/BertModel/Embeddings",
        "Add_1": "/BertModel/Embeddings",
        "LayerNorm_2": "/BertModel/Embeddings",
        "MatMul_3": "/BertModel/Encoder/Layer/0/Attention/Self",
        "Softmax_4": "/BertModel/Encoder/Layer/0/Attention/Self",
        "MatMul_5": "/BertModel/Encoder/Layer/0/Attention/Self",
        "Add_6": "/BertModel/Encoder/Layer/0/Attention/Output",
        "LayerNorm_7": "/BertModel/Encoder/Layer/0/Attention/Output",
        "Relu_8": "/BertModel/Encoder/Layer/0",
        "MatMul_9": "/BertModel/Encoder/Layer/1/Attention/Self",
        "Gather_10": "/BertModel",
    }


class TestTextStyler:
    """Test the TextStyler class for proper formatting."""
    
    def test_bold_cyan_decimal(self):
        """Test decimal number formatting only colors before decimal point."""
        # Test with M suffix
        result = TextStyler.bold_cyan_decimal(4.4, "M")
        assert result == "\033[1;36m4.\033[0m4M"
        
        # Test with s suffix
        result = TextStyler.bold_cyan_decimal(4.83, "s")
        assert result == "\033[1;36m4.\033[0m83s"
        
        # Test with percentage
        result = TextStyler.bold_cyan_decimal(99.9, "%")
        assert result == "\033[1;36m99.\033[0m9%"
        
        # Test whole number
        result = TextStyler.bold_cyan_decimal(100.0, "%")
        assert result == "\033[1;36m100\033[0m%"
    
    def test_strip_ansi(self):
        """Test ANSI code stripping."""
        colored = "\033[1;36m4.\033[0m83s"
        plain = TextStyler.strip_ansi(colored)
        assert plain == "4.83s"


class TestHTPExportMonitor:
    """Test cases for HTP Export Monitor with proper mocking."""
    
    def test_monitor_initialization(self, tmp_path):
        """Test monitor initialization with different configurations."""
        output_path = str(tmp_path / "test.onnx")
        
        # Test with verbose and report enabled
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True,
            enable_report=True
        )
        
        assert monitor.output_path == output_path
        assert monitor.model_name == "test-model"
        assert monitor.verbose is True
        assert monitor.enable_report is True
    
    def test_seven_step_flow(self, tmp_path):
        """Test the 7-step export flow (without SAVE/COMPLETE as numbered steps)."""
        output_path = str(tmp_path / "test.onnx")
        
        # Capture console output
        console_output = []
        
        with patch('modelexport.strategies.htp.export_monitor.Console') as mock_console:
            # Mock console to capture output
            mock_file = Mock()
            mock_file.write = lambda text: console_output.append(text)
            mock_console.return_value.file = mock_file
            
            with HTPExportMonitor(output_path=output_path, verbose=True) as monitor:
                data = HTPExportData()
                
                # Step 1: Model Preparation
                data.model_name = "test-model"
                data.model_class = "SimpleModel"
                data.total_modules = 48
                data.total_parameters = 4385536
                monitor.log_step(HTPExportStep.MODEL_PREP, data)
                
                # Step 2: Input Generation
                data.steps = {
                    "input_generation": {
                        "method": "auto_generated",
                        "model_type": "bert",
                        "task": "feature-extraction",
                        "inputs": {
                            "input_ids": {"shape": [1, 512], "dtype": "int64"},
                            "attention_mask": {"shape": [1, 512], "dtype": "int64"}
                        }
                    }
                }
                monitor.log_step(HTPExportStep.INPUT_GEN, data)
                
                # Step 3: Hierarchy Building
                data.hierarchy = {
                    "": {"class_name": "BertModel", "traced_tag": "/BertModel"},
                    "embeddings": {"class_name": "BertEmbeddings", "traced_tag": "/BertModel/Embeddings"},
                    "encoder": {"class_name": "BertEncoder", "traced_tag": "/BertModel/Encoder"},
                    "encoder.layer": {"class_name": "ModuleList", "traced_tag": "/BertModel/Encoder/Layer"},
                    "encoder.layer.0": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/0"},
                    "encoder.layer.0.attention": {"class_name": "BertAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention"},
                    "encoder.layer.0.attention.self": {"class_name": "BertSdpaSelfAttention", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Self"},
                    "encoder.layer.0.attention.output": {"class_name": "BertSelfOutput", "traced_tag": "/BertModel/Encoder/Layer/0/Attention/Output"},
                    "encoder.layer.1": {"class_name": "BertLayer", "traced_tag": "/BertModel/Encoder/Layer/1"},
                }
                data.execution_steps = 36
                monitor.log_step(HTPExportStep.HIERARCHY, data)
                
                # Step 4: ONNX Export
                data.steps["onnx_export"] = {
                    "opset_version": 17,
                    "do_constant_folding": True,
                    "input_names": ["input_ids", "attention_mask"]
                }
                data.onnx_size_mb = 17.5
                monitor.log_step(HTPExportStep.ONNX_EXPORT, data)
                
                # Step 5: Node Tagger Creation
                data.steps["tagger_creation"] = {
                    "tagger_type": "HierarchyNodeTagger",
                    "enable_operation_fallback": False,
                    "root_tag": "/BertModel"
                }
                monitor.log_step(HTPExportStep.TAGGER_CREATION, data)
                
                # Step 6: Node Tagging
                data.total_nodes = 136
                data.tagged_nodes = {
                    "MatMul_0": "/BertModel/Embeddings",
                    "Add_1": "/BertModel/Embeddings",
                    "LayerNorm_2": "/BertModel/Embeddings",
                    "MatMul_3": "/BertModel/Encoder/Layer/0/Attention/Self",
                    "Softmax_4": "/BertModel/Encoder/Layer/0/Attention/Self",
                    "MatMul_5": "/BertModel/Encoder/Layer/0/Attention/Self",
                    "Add_6": "/BertModel/Encoder/Layer/0/Attention/Output",
                    "LayerNorm_7": "/BertModel/Encoder/Layer/0/Attention/Output",
                    "Relu_8": "/BertModel/Encoder/Layer/0",
                    "MatMul_9": "/BertModel/Encoder/Layer/1/Attention/Self",
                    "Gather_10": "/BertModel",
                }
                data.tagging_stats = {
                    "direct_matches": 83,
                    "parent_matches": 34,
                    "root_fallbacks": 19
                }
                data.coverage = 100.0
                monitor.log_step(HTPExportStep.NODE_TAGGING, data)
                
                # Step 7: Tag Injection (final numbered step)
                data.embed_hierarchy_attributes = True
                data.output_path = output_path
                monitor.log_step(HTPExportStep.TAG_INJECTION, data)
                
                # Finalize (metadata and summary - not numbered)
                monitor.finalize_export(
                    export_time=4.83,
                    output_path=output_path
                )
        
        # Verify console output
        full_output = "\n".join(console_output)
        
        # Check for 7 steps only
        assert "[1/7]" in full_output
        assert "[2/7]" in full_output
        assert "[3/7]" in full_output
        assert "[4/7]" in full_output
        assert "[5/7]" in full_output
        assert "[6/7]" in full_output
        assert "[7/7]" in full_output
        assert "[8/7]" not in full_output  # Should not exist
        assert "[8/8]" not in full_output  # Should not exist
        
        # Check metadata generation is not numbered
        assert "📄 METADATA GENERATION" in full_output
        assert "[" not in full_output.split("METADATA GENERATION")[0].split("\n")[-1]
        
        # Check final summary
        assert "📋 FINAL EXPORT SUMMARY" in full_output
    
    def test_hierarchy_tree_display(self, tmp_path, sample_hierarchy, sample_tagged_nodes):
        """Test hierarchy tree display with improved implementation."""
        output_path = str(tmp_path / "test.onnx")
        
        console_output = []
        
        with patch('modelexport.strategies.htp.export_monitor.Console') as mock_console:
            mock_file = Mock()
            mock_file.write = lambda text: console_output.append(text)
            mock_console.return_value.file = mock_file
            
            with HTPExportMonitor(output_path=output_path, verbose=True) as monitor:
                data = HTPExportData()
                data.hierarchy = sample_hierarchy
                data.tagged_nodes = sample_tagged_nodes
                data.total_nodes = len(sample_tagged_nodes)
                data.coverage = 100.0
                
                monitor.log_step(HTPExportStep.NODE_TAGGING, data)
        
        full_output = "\n".join(console_output)
        
        # Check hierarchy structure
        assert "BertModel (11 ONNX nodes)" in full_output
        assert "├── BertEmbeddings: embeddings (3 nodes)" in full_output
        assert "├── BertEncoder: encoder (7 nodes)" in full_output
        assert "    └── ModuleList: encoder.layer (7 nodes)" in full_output
        assert "        ├── BertLayer: encoder.layer.0 (6 nodes)" in full_output
        assert "        └── BertLayer: encoder.layer.1 (1 nodes)" in full_output
        
        # Check operations display
        assert "LayerNorm: /BertModel/Embeddings/LayerNorm_2" in full_output
        assert "Softmax: /BertModel/Encoder/Layer/0/Attention/Self/Softmax_4" in full_output
        assert "Relu" in full_output  # Should show Relu operations
        
        # Check truncation for console
        lines = full_output.split("\n")
        hierarchy_lines = [l for l in lines if "Complete HF Hierarchy" in l]
        if hierarchy_lines:
            idx = lines.index(hierarchy_lines[0])
            remaining_lines = lines[idx:]
            # Check if truncated (should be if > 30 lines)
            if len(remaining_lines) > 30:
                assert "truncated for console" in full_output
    
    def test_report_without_truncation(self, tmp_path, sample_hierarchy, sample_tagged_nodes):
        """Test that report shows full hierarchy without truncation."""
        output_path = str(tmp_path / "test.onnx")
        report_path = str(tmp_path / "test_htp_export_report.txt")
        
        with HTPExportMonitor(output_path=output_path, enable_report=True) as monitor:
            data = HTPExportData()
            
            # Create large hierarchy to test truncation
            large_hierarchy = sample_hierarchy.copy()
            for i in range(50):  # Add many modules
                large_hierarchy[f"encoder.layer.{i}"] = {
                    "class_name": "BertLayer",
                    "traced_tag": f"/BertModel/Encoder/Layer/{i}"
                }
            
            data.hierarchy = large_hierarchy
            data.tagged_nodes = sample_tagged_nodes
            data.total_nodes = len(sample_tagged_nodes)
            
            monitor.log_step(HTPExportStep.NODE_TAGGING, data)
        
        # Check report file
        assert Path(report_path).exists()
        
        with open(report_path) as f:
            report_content = f.read()
        
        # Report should NOT have truncation message
        assert "truncated for console" not in report_content
        assert "showing" not in report_content or "lines" not in report_content
        
        # Should have all layers
        for i in range(50):
            if i < 10:  # Check some layers are present
                assert f"encoder.layer.{i}" in report_content
    
    def test_metadata_structure(self, tmp_path, sample_hierarchy, sample_tagged_nodes):
        """Test metadata file structure and content."""
        output_path = str(tmp_path / "test.onnx")
        metadata_path = str(tmp_path / "test_htp_metadata.json")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            data = HTPExportData()
            
            # Simulate full export
            data.model_name = "test-model"
            data.model_class = "BertModel"
            data.total_modules = 48
            data.total_parameters = 4385536
            data.hierarchy = sample_hierarchy
            data.execution_steps = 36
            data.total_nodes = 136
            data.tagged_nodes = sample_tagged_nodes
            data.tagging_stats = {
                "direct_matches": 83,
                "parent_matches": 34,
                "root_fallbacks": 19
            }
            data.coverage = 100.0
            data.export_time = 4.83
            
            monitor.log_step(HTPExportStep.MODEL_PREP, data)
            monitor.log_step(HTPExportStep.HIERARCHY, data)
            monitor.log_step(HTPExportStep.NODE_TAGGING, data)
            
            monitor.finalize_export(export_time=4.83, output_path=output_path)
        
        # Check metadata
        assert Path(metadata_path).exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check structure
        assert "export_context" in metadata
        assert "model" in metadata
        assert "modules" in metadata
        assert "nodes" in metadata  # At root level
        assert "report" in metadata
        
        # Check node_tagging in report
        assert "node_tagging" in metadata["report"]
        assert "statistics" in metadata["report"]["node_tagging"]
        assert "coverage" in metadata["report"]["node_tagging"]
        
        # Validate values
        assert metadata["model"]["name_or_path"] == "test-model"
        assert metadata["model"]["total_parameters"] == 4385536
        assert len(metadata["nodes"]) == len(sample_tagged_nodes)
        assert metadata["report"]["node_tagging"]["coverage"]["coverage_percentage"] == 100.0


class TestHTPExporterIntegration:
    """Integration tests for HTPExporter with updated monitor."""
    
    @patch('torch.onnx.export')
    @patch('onnx.load')
    @patch('onnx.save')
    def test_full_export_flow(self, mock_save, mock_load, mock_export,
                             simple_model, mock_onnx_model, tmp_path):
        """Test complete export flow with all steps properly mocked."""
        # Configure mocks
        mock_load.return_value = mock_onnx_model
        
        # Create exporter
        exporter = HTPExporter(verbose=True, enable_reporting=True)
        
        # Export path
        output_path = str(tmp_path / "test_export.onnx")
        
        # Mock each step properly
        with patch('modelexport.core.model_input_generator.generate_dummy_inputs') as mock_inputs:
            mock_inputs.return_value = {"input_ids": torch.randint(0, 1000, (1, 128))}
            
            # Mock TracingHierarchyBuilder
            with patch('modelexport.strategies.htp.htp_exporter.TracingHierarchyBuilder') as mock_builder_class:
                mock_builder = Mock()
                mock_builder_class.return_value = mock_builder
                
                # Mock hierarchy building
                mock_builder.trace_model_execution.return_value = None
                mock_builder.get_execution_summary.return_value = {
                    "module_hierarchy": {
                        "": {"class_name": "SimpleModel", "traced_tag": "/SimpleModel"},
                        "linear": {"class_name": "Linear", "traced_tag": "/SimpleModel/Linear"}
                    },
                    "execution_steps": 5
                }
                mock_builder.get_outputs.return_value = torch.randn(1, 5)
                
                # Mock node tagger
                with patch('modelexport.strategies.htp.htp_exporter.create_node_tagger_from_hierarchy') as mock_tagger:
                    tagger = Mock()
                    mock_tagger.return_value = tagger
                    
                    # Mock tagging
                    tagger.tag_all_nodes.return_value = {
                        f"node_{i}": "/SimpleModel/Linear" for i in range(5)
                    }
                    tagger.get_tagging_statistics.return_value = {
                        "direct_matches": 5,
                        "parent_matches": 0,
                        "root_fallbacks": 0,
                        "empty_tags": 0
                    }
                    
                    # Run export
                    result = exporter.export(
                        model=simple_model,
                        output_path=output_path,
                        model_name_or_path="test-model"
                    )
        
        # Verify results
        assert result["export_time"] > 0
        assert result["hierarchy_modules"] == 2
        assert result["onnx_nodes"] == 15  # From mock_onnx_model
        assert result["tagged_nodes"] == 5
        assert result["empty_tags"] == 0
        assert result["coverage_percentage"] == 33.3  # 5/15
        
        # Check files created
        assert mock_export.called
        assert mock_save.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])