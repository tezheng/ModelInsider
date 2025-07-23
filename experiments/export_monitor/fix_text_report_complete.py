#!/usr/bin/env python3
"""
Fix the text report to include ALL console output without styling or truncation.

The text report should be a complete, plain-text version of everything
shown in the console, suitable for viewing in any text editor.
"""

def show_current_vs_expected_text_report():
    """Show what's currently in text report vs what should be there."""
    print("Current Text Report (Minimal):")
    print("=" * 60)
    
    current_report = """================================================================================
HTP EXPORT FULL REPORT
================================================================================
Generated: 2025-07-19T09:56:53Z

MODEL INFORMATION
----------------------------------------
Model Name: prajjwal1/bert-tiny
Model Class: BertModel
Total Modules: 48
Total Parameters: 4,385,536
Export Strategy: HTP
Output Path: model.onnx
Embed Hierarchy: True

INPUT GENERATION
----------------------------------------
Model Type: bert
Task: feature-extraction
Method: auto

Generated Inputs:
  input_ids: shape=[2, 16], dtype=torch.int64
  attention_mask: shape=[2, 16], dtype=torch.int64
  token_type_ids: shape=[2, 16], dtype=torch.int64
"""
    
    print(current_report)
    
    print("\n\nExpected Text Report (Complete Console Output):")
    print("=" * 60)
    
    expected_report = """================================================================================
HTP EXPORT FULL REPORT
================================================================================
Generated: 2025-07-19T09:56:53Z

🔄 Loading model and exporting: prajjwal1/bert-tiny
🧠 Using HTP (Hierarchical Trace-and-Project) strategy
Auto-loading model from: prajjwal1/bert-tiny
Successfully loaded BertModel
Starting HTP export for BertModel

================================================================================
📋 STEP 1/8: MODEL PREPARATION
================================================================================
✅ Model loaded: BertModel (48 modules, 4.4M parameters)
🎯 Export target: model.onnx
⚙️ Strategy: HTP (Hierarchy-Preserving)
✅ Hierarchy attributes will be embedded in ONNX
✅ Model set to evaluation mode

================================================================================
🔧 STEP 2/8: INPUT GENERATION & VALIDATION
================================================================================
🤖 Auto-generating inputs for: prajjwal1/bert-tiny
   • Model type: bert
   • Auto-detected task: feature-extraction
✅ Created onnx export config for bert with task feature-extraction
🔧 Generated 3 input tensors:
   • input_ids: [2, 16] (torch.int64)
   • attention_mask: [2, 16] (torch.int64)
   • token_type_ids: [2, 16] (torch.int64)

================================================================================
🏗️ STEP 3/8: HIERARCHY BUILDING
================================================================================
✅ Hierarchy building completed with TracingHierarchyBuilder
📈 Traced 18 modules
🔄 Execution steps: 36

🌳 Module Hierarchy:
------------------------------------------------------------
BertModel
├── BertEmbeddings: embeddings
├── BertEncoder: encoder
│   ├── BertLayer: encoder.layer.0
│   │   ├── BertAttention: encoder.layer.0.attention
│   │   │   ├── BertSdpaSelfAttention: encoder.layer.0.attention.self
│   │   │   └── BertSelfOutput: encoder.layer.0.attention.output
│   │   ├── BertIntermediate: encoder.layer.0.intermediate
│   │   │   └── GELUActivation: encoder.layer.0.intermediate.intermediate_act_fn
│   │   └── BertOutput: encoder.layer.0.output
│   └── BertLayer: encoder.layer.1
│       ├── BertAttention: encoder.layer.1.attention
│       │   ├── BertSdpaSelfAttention: encoder.layer.1.attention.self
│       │   └── BertSelfOutput: encoder.layer.1.attention.output
│       ├── BertIntermediate: encoder.layer.1.intermediate
│       │   └── GELUActivation: encoder.layer.1.intermediate.intermediate_act_fn
│       └── BertOutput: encoder.layer.1.output
└── BertPooler: pooler

================================================================================
📦 STEP 4/8: ONNX EXPORT
================================================================================
🎯 Target file: model.onnx
⚙️ Export config:
   • opset_version: 17
   • do_constant_folding: True
   • verbose: False
   • input_names: ['input_ids', 'attention_mask', 'token_type_ids']
✅ ONNX export completed successfully

================================================================================
🏷️ STEP 5/8: NODE TAGGER CREATION
================================================================================
✅ Node tagger created successfully
🏷️ Model root tag: /BertModel
🔧 Operation fallback: disabled

================================================================================
🔗 STEP 6/8: ONNX NODE TAGGING
================================================================================
✅ Node tagging completed successfully
📈 Coverage: 100.0%
📊 Tagged nodes: 136/136
   • Direct matches: 83 (61.0%)
   • Parent matches: 34 (25.0%)
   • Root fallbacks: 19 (14.0%)
✅ Empty tags: 0

📊 Top 20 Nodes by Hierarchy:
------------------------------
 1. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention: 35 nodes
 2. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention: 35 nodes
 3. /BertModel: 19 nodes
 4. /BertModel/BertEmbeddings: 8 nodes
 5. /BertModel/BertEncoder/BertLayer.0/BertIntermediate/GELUActivation: 8 nodes
 6. /BertModel/BertEncoder/BertLayer.1/BertIntermediate/GELUActivation: 8 nodes
 7. /BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfOutput: 4 nodes
 8. /BertModel/BertEncoder/BertLayer.0/BertOutput: 4 nodes
 9. /BertModel/BertEncoder/BertLayer.1/BertAttention/BertSelfOutput: 4 nodes
10. /BertModel/BertEncoder/BertLayer.1/BertOutput: 4 nodes
11. /BertModel/BertPooler: 3 nodes
12. /BertModel/BertEncoder/BertLayer.0/BertIntermediate: 2 nodes
13. /BertModel/BertEncoder/BertLayer.1/BertIntermediate: 2 nodes

🌳 Complete HF Hierarchy with ONNX Nodes:
------------------------------------------------------------
BertModel (19 ONNX nodes)
├── BertEmbeddings: embeddings (8 nodes)
│   └── Operations:
│       ├── Unsqueeze (3x)
│       ├── Shape
│       ├── Gather
│       ├── Add (2x)
│       └── LayerNormalization
├── BertEncoder: encoder
│   ├── BertLayer: encoder.layer.0
│   │   ├── BertAttention: encoder.layer.0.attention
│   │   │   ├── BertSdpaSelfAttention: encoder.layer.0.attention.self (35 nodes)
│   │   │   │   └── Operations:
│   │   │   │       ├── MatMul (6x)
│   │   │   │       ├── Add (4x)
│   │   │   │       ├── Reshape (9x)
│   │   │   │       ├── Transpose (4x)
│   │   │   │       ├── Div
│   │   │   │       ├── Mul (3x)
│   │   │   │       ├── Sub (2x)
│   │   │   │       ├── Softmax
│   │   │   │       └── ScaledDotProductAttention
│   │   │   └── BertSelfOutput: encoder.layer.0.attention.output (4 nodes)
│   │   │       └── Operations:
│   │   │           ├── MatMul
│   │   │           ├── Add (2x)
│   │   │           └── LayerNormalization
... [FULL TREE CONTINUES - NO TRUNCATION]

================================================================================
💾 STEP 7/8: SAVE ONNX MODEL
================================================================================
✅ Model saved to: model.onnx
⚙️ Hierarchy attributes: Embedded in ONNX

================================================================================
🎉 STEP 8/8: EXPORT COMPLETE
================================================================================
🎉 HTP Export completed successfully in 2.35s!
📊 Export Statistics:
   • Export time: 2.35s
   • Hierarchy modules: 18
   • ONNX nodes: 136
   • Tagged nodes: 136
   • Coverage: 100.0%
📁 Output files:
   • ONNX model: model.onnx (17.5 MB)
   • Metadata: model_htp_metadata.json
   • Report: model_htp_export_report.txt
   • Console log: model_console.log
"""
    
    print(expected_report[:2000] + "\n... [Full report continues with ALL console output]")
    
    return current_report, expected_report


def create_proper_report_writer():
    """Show how the report writer should capture all console output."""
    print("\n\nProper HTPReportWriter Implementation:")
    print("=" * 60)
    
    code = '''
class HTPReportWriter(StepAwareWriter):
    """Full text report writer that captures ALL console output."""
    
    def __init__(self, output_path: str, console_buffer: io.StringIO = None):
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_htp_export_report.txt"
        self.buffer = io.StringIO()
        self.console_buffer = console_buffer  # Reference to console output buffer
        self._write_header()
    
    def _write_header(self):
        """Write report header."""
        self.buffer.write("=" * 80 + "\\n")
        self.buffer.write("HTP EXPORT FULL REPORT\\n")
        self.buffer.write("=" * 80 + "\\n")
        self.buffer.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\\n\\n")
    
    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re
        # Pattern to match ANSI escape sequences
        ansi_escape = re.compile(r'\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def flush(self):
        """Write the complete console output to report file."""
        # If we have access to console buffer, use it
        if self.console_buffer:
            console_output = self.console_buffer.getvalue()
            # Strip ANSI codes for plain text report
            plain_output = self._strip_ansi_codes(console_output)
            self.buffer.write(plain_output)
        
        # Write to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
    
    # Alternative: Capture output in each step method
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Capture model preparation output."""
        self.buffer.write("\\n" + "=" * 80 + "\\n")
        self.buffer.write("📋 STEP 1/8: MODEL PREPARATION\\n")
        self.buffer.write("=" * 80 + "\\n")
        self.buffer.write(f"✅ Model loaded: {data.model_class} ")
        self.buffer.write(f"({data.total_modules} modules, {data.total_parameters/1e6:.1f}M parameters)\\n")
        self.buffer.write(f"🎯 Export target: {data.output_path}\\n")
        self.buffer.write("⚙️ Strategy: HTP (Hierarchy-Preserving)\\n")
        
        if data.embed_hierarchy_attributes:
            self.buffer.write("✅ Hierarchy attributes will be embedded in ONNX\\n")
        else:
            self.buffer.write("⚠️ Hierarchy attributes will NOT be embedded (clean ONNX)\\n")
        
        self.buffer.write("✅ Model set to evaluation mode\\n")
        return 1
    
    @step(HTPExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Write COMPLETE hierarchy tree without truncation."""
        self.buffer.write("\\n🌳 Module Hierarchy:\\n")
        self.buffer.write("-" * 60 + "\\n")
        
        # Build the FULL tree - NO TRUNCATION
        tree_lines = self._build_complete_tree(data.hierarchy)
        for line in tree_lines:
            self.buffer.write(line + "\\n")
        
        return 1
    
    def _build_complete_tree(self, hierarchy: dict) -> list[str]:
        """Build complete tree representation without any truncation."""
        lines = []
        
        # Find root
        root_info = hierarchy.get("", {})
        root_name = root_info.get("class_name", "Model")
        lines.append(root_name)
        
        # Build tree for all modules
        def add_module(path: str, prefix: str = "", is_last: bool = True):
            if not path:
                return
            
            info = hierarchy.get(path, {})
            class_name = info.get("class_name", "Unknown")
            
            # Determine tree characters
            if prefix:
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{class_name}: {path}")
            
            # Find all direct children
            children = []
            for other_path in hierarchy:
                if other_path.startswith(path + ".") and other_path.count(".") == path.count(".") + 1:
                    children.append(other_path)
            
            # Add all children
            for i, child in enumerate(sorted(children)):
                is_last_child = (i == len(children) - 1)
                child_prefix = prefix + ("    " if is_last else "│   ")
                add_module(child, child_prefix, is_last_child)
        
        # Add all root modules
        root_modules = [p for p in hierarchy if p and "." not in p]
        for i, module in enumerate(sorted(root_modules)):
            is_last = (i == len(root_modules) - 1)
            add_module(module, "", is_last)
        
        return lines
'''
    
    print(code)
    return code


def show_key_differences():
    """Show the key differences between current and expected."""
    print("\n\nKey Differences:")
    print("=" * 60)
    
    differences = {
        "Console Output Capture": {
            "Current": "Not capturing console output",
            "Expected": "Capture ALL console output in text report"
        },
        "Text Styling": {
            "Current": "May contain ANSI codes",
            "Expected": "Strip all ANSI codes for plain text"
        },
        "Truncation": {
            "Current": "Trees and lists are truncated",
            "Expected": "NO truncation - show everything"
        },
        "Content Coverage": {
            "Current": "Minimal summary information",
            "Expected": "Complete mirror of console output"
        },
        "Hierarchy Display": {
            "Current": "List format or truncated tree",
            "Expected": "Full tree with all modules and operations"
        }
    }
    
    for aspect, diff in differences.items():
        print(f"\n{aspect}:")
        print(f"  Current: {diff['Current']}")
        print(f"  Expected: {diff['Expected']}")
    
    return differences


def main():
    """Analyze text report issues and show fixes."""
    print("Text Report Analysis - Full Console Output Capture")
    print("=" * 80)
    
    # Show current vs expected
    current, expected = show_current_vs_expected_text_report()
    
    # Show proper implementation
    implementation = create_proper_report_writer()
    
    # Show key differences
    differences = show_key_differences()
    
    print("\n\nSummary:")
    print("=" * 60)
    print("The text report should be a COMPLETE, PLAIN TEXT copy of console output:")
    print("1. Capture ALL console output (no truncation)")
    print("2. Strip ANSI color codes for plain text")
    print("3. Include full hierarchy trees")
    print("4. Include all statistics and details")
    print("5. Be readable in any text editor")


if __name__ == "__main__":
    main()