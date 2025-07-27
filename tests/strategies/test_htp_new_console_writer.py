"""Unit tests for the console writer."""

import io

import pytest
from rich.console import Console

from modelexport.strategies.htp.base_writer import ExportData, ExportStep
from modelexport.strategies.htp.console_writer import ConsoleWriter
from modelexport.strategies.htp.step_data import (
    ModelPrepData,
    InputGenData,
    TensorInfo,
    HierarchyData,
    ModuleInfo,
    ONNXExportData,
    NodeTaggingData,
    TagInjectionData,
)


class TestConsoleWriter:
    """Test the ConsoleWriter class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.buffer = io.StringIO()
        self.console = Console(
            file=self.buffer,
            width=120,
            force_terminal=True,
            legacy_windows=False,
            highlight=False,
        )
        self.writer = ConsoleWriter(console=self.console)
        self.data = ExportData(
            model_name="test-model",
            output_path="/tmp/test.onnx",
        )
    
    def get_output(self) -> str:
        """Get the console output."""
        return self.buffer.getvalue()
    
    def get_plain_output(self) -> str:
        """Get console output with ANSI codes stripped."""
        import re
        output = self.buffer.getvalue()
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', output)
    
    def test_model_prep_output(self):
        """Test MODEL_PREP step output."""
        self.data.model_prep = ModelPrepData(
            model_class="BertModel",
            total_modules=123,
            total_parameters=4400000,
        )
        
        self.writer.write(ExportStep.MODEL_PREP, self.data)
        output = self.get_plain_output()
        
        # Check key elements
        assert "STEP 1/6: MODEL PREPARATION" in output
        assert "Model loaded: BertModel" in output
        assert "123 modules" in output
        assert "4.4M parameters" in output
        assert "Export target:" in output
        assert "/tmp/test.onnx" in output
    
    def test_input_gen_auto_generated(self):
        """Test INPUT_GEN step with auto-generated inputs."""
        self.data.input_gen = InputGenData(
            method="auto_generated",
            model_type="bert",
            task="text-classification",
            inputs={
                "input_ids": TensorInfo(shape=[1, 128], dtype="int64"),
                "attention_mask": TensorInfo(shape=[1, 128], dtype="int64"),
            }
        )
        
        self.writer.write(ExportStep.INPUT_GEN, self.data)
        output = self.get_plain_output()
        
        assert "STEP 2/6: INPUT GENERATION" in output
        assert "Auto-generating inputs for: test-model" in output
        assert "Model type: bert" in output
        assert "Detected task: text-classification" in output
        assert "input_ids: shape=[1, 128], dtype=int64" in output
    
    def test_hierarchy_output(self):
        """Test HIERARCHY step output."""
        self.data.hierarchy = HierarchyData(
            hierarchy={
                "": ModuleInfo(class_name="BertModel", traced_tag="/BertModel"),
                "embeddings": ModuleInfo(class_name="BertEmbeddings", traced_tag="/BertModel/embeddings"),
                "encoder": ModuleInfo(class_name="BertEncoder", traced_tag="/BertModel/encoder"),
                "encoder.layer.0": ModuleInfo(class_name="BertLayer", traced_tag="/BertModel/encoder/layer.0"),
            },
            execution_steps=42,
        )
        
        self.writer.write(ExportStep.HIERARCHY, self.data)
        output = self.get_plain_output()
        
        assert "STEP 3/6: HIERARCHY BUILDING" in output
        assert "Traced 4 modules in hierarchy" in output
        assert "Total execution steps: 42" in output
        assert "Module Hierarchy:" in output
        assert "BertModel" in output
        assert "BertEmbeddings: embeddings" in output
    
    def test_onnx_export_output(self):
        """Test ONNX_EXPORT step output."""
        self.data.onnx_export = ONNXExportData(
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=None,
            onnx_size_mb=17.6,
        )
        
        self.writer.write(ExportStep.ONNX_EXPORT, self.data)
        output = self.get_plain_output()
        
        assert "STEP 4/6: ONNX EXPORT" in output
        assert "Opset version: 17" in output
        assert "Constant folding: True" in output
        assert "['input_ids', 'attention_mask']" in output
        assert "Output names: Not detected" in output
        assert "Model size: 17.60MB" in output
    
    def test_node_tagging_output(self):
        """Test NODE_TAGGING step output."""
        self.data.hierarchy = HierarchyData(
            hierarchy={
                "": ModuleInfo(class_name="BertModel", traced_tag="/BertModel"),
            },
            execution_steps=1,
        )
        
        self.data.node_tagging = NodeTaggingData(
            total_nodes=100,
            tagged_nodes={
                "node1": "/BertModel/embeddings",
                "node2": "/BertModel/encoder",
                "node3": "/BertModel/encoder",
            },
            tagging_stats={
                "direct_matches": 80,
                "parent_matches": 15,
                "root_fallbacks": 5,
            },
            coverage=100.0,
        )
        
        self.writer.write(ExportStep.NODE_TAGGING, self.data)
        output = self.get_plain_output()
        
        assert "STEP 5/6: ONNX NODE TAGGING" in output
        assert "Coverage: 100.0%" in output
        assert "Tagged nodes: 3/100" in output
        assert "Direct matches: 80 (80.0%)" in output
        assert "Parent matches: 15 (15.0%)" in output
        assert "Root fallbacks: 5 (5.0%)" in output
        assert "Empty tags: 0" in output
    
    def test_tag_injection_enabled(self):
        """Test TAG_INJECTION step when enabled."""
        self.data.tag_injection = TagInjectionData(
            tags_injected=True,
            tags_stripped=False,
        )
        
        self.writer.write(ExportStep.TAG_INJECTION, self.data)
        output = self.get_plain_output()
        
        assert "STEP 6/6: TAG INJECTION" in output
        assert "Injecting hierarchy tags into ONNX model..." in output
        assert "Tags successfully embedded" in output
        assert "Model saved to: /tmp/test.onnx" in output
    
    def test_tag_injection_disabled(self):
        """Test TAG_INJECTION step when disabled (clean-onnx mode)."""
        self.data.tag_injection = TagInjectionData(
            tags_injected=False,
            tags_stripped=True,
        )
        
        self.writer.write(ExportStep.TAG_INJECTION, self.data)
        output = self.get_plain_output()
        
        assert "STEP 6/6: TAG INJECTION" in output
        assert "Hierarchy tag injection skipped (--clean-onnx mode)" in output
    
    def test_verbose_false(self):
        """Test that non-verbose writer produces no output."""
        writer = ConsoleWriter(verbose=False)
        self.data.model_prep = ModelPrepData(
            model_class="BertModel",
            total_modules=123,
            total_parameters=4400000,
        )
        
        result = writer.write(ExportStep.MODEL_PREP, self.data)
        
        # Should return 0 (no bytes written)
        assert result == 0
    
    def test_hierarchy_truncation(self):
        """Test that large hierarchies are truncated."""
        # Create a flat hierarchy with many direct children
        # This should definitely exceed 30 lines when rendered
        hierarchy = {
            "": ModuleInfo(class_name="Model", traced_tag="/Model")  # Root
        }
        
        for i in range(40):
            hierarchy[f"layer{i}"] = ModuleInfo(
                class_name=f"Layer{i}",
                traced_tag=f"/Model/layer{i}"
            )
        
        self.data.hierarchy = HierarchyData(
            hierarchy=hierarchy,
            execution_steps=40,
        )
        
        self.writer.write(ExportStep.HIERARCHY, self.data)
        output = self.get_plain_output()
        
        # Should contain hierarchy message
        assert "Module Hierarchy:" in output
        
        # Tree should be truncated since we have 41 modules
        assert "showing first 30 lines" in output
        assert "(truncated for console)" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])