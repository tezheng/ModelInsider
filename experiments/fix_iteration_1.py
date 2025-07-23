#!/usr/bin/env python3
"""
Fix Iteration 1 - Comprehensive fixes for HTP Export Monitor

Issues to fix:
1. Text color - only color numbers before decimal point
2. Remove SAVE and COMPLETE as numbered steps (make them 7 steps total)
3. Generate full hierarchy for report without truncation
4. Add Relu to the list of nodes that show full paths
"""

# First, let's look at the exact color pattern in baseline
baseline_examples = """
From baseline console_output_with_colors.txt:

Line 10: ✅ Model loaded: BertModel [1m([0m[1;36m48[0m modules, [1;36m4.[0m4M parameters[1m)[0m
         - "48" is fully cyan: [1;36m48[0m
         - "4." is cyan but "4M" is not: [1;36m4.[0m4M
         
Line 151: 🎉 HTP Export completed successfully in [1;36m4.[0m83s!
         - "4." is cyan but "83s" is not: [1;36m4.[0m83s
         
Line 153:    • Export time: [1;36m4.[0m83s
         - Same pattern
"""

# TextStyler fixes needed
text_styler_fixes = """
@staticmethod
def bold_cyan_decimal(number: float, suffix: str = "") -> str:
    '''Format number with only integer part colored, decimal part plain.'''
    if '.' in str(number):
        parts = str(number).split('.')
        return f"\033[1;36m{parts[0]}.\033[0m{parts[1]}{suffix}"
    else:
        return f"\033[1;36m{number}\033[0m{suffix}"
"""

# Step counting fix - only 7 actual export steps
step_fixes = """
# In HTPExportStep enum - remove SAVE and COMPLETE
class HTPExportStep(Enum):
    '''Export process steps.'''
    MODEL_PREP = "model_preparation"
    INPUT_GEN = "input_generation" 
    HIERARCHY = "hierarchy_building"
    ONNX_EXPORT = "onnx_export"
    TAGGER_CREATION = "tagger_creation"
    NODE_TAGGING = "node_tagging"
    TAG_INJECTION = "tag_injection"  # This is step 7, the final step
    # Remove: SAVE = "model_save"
    # Remove: COMPLETE = "export_complete"

# Update _total_steps in ConsoleWriter
self._total_steps = 7  # Not 8

# Change write_save to write_tag_injection
@step(HTPExportStep.TAG_INJECTION)
def write_tag_injection(self, export_step: HTPExportStep, data: HTPExportData) -> int:
    '''Step 7: Tag injection.'''
    # ... implementation

# write_complete becomes write_metadata_and_summary (not a numbered step)
def write_metadata_and_summary(self, data: HTPExportData) -> int:
    '''Write metadata generation message and final summary (not numbered steps).'''
    # First, metadata generation section
    self._print("")
    self._print_separator()
    self._print("📄 METADATA GENERATION")  # No step number!
    self._print_separator()
    self._print("✅ Metadata file created successfully")
    if data.output_path:
        metadata_path = str(Path(data.output_path).with_suffix('')) + "_htp_metadata.json"
        self._print(f"📄 Metadata file: {metadata_path}")
    
    # Then final summary
    self._print("")
    self._print_header("📋 FINAL EXPORT SUMMARY")
    # ... rest of summary
"""

# Report truncation fix
report_fixes = """
# In HTPReportWriter, add a method to generate full hierarchy
def _generate_full_hierarchy(self, data: HTPExportData) -> str:
    '''Generate complete hierarchy tree without truncation for report.'''
    if not data.hierarchy or not data.tagged_nodes:
        return ""
    
    lines = []
    # ... same logic as console but WITHOUT truncation
    # Return '\n'.join(lines)
    
# In flush method, don't just copy console output
def flush(self):
    '''Write the complete report with full hierarchy.'''
    # Generate sections programmatically without truncation
    # Not just copying console output
"""

# Relu node fix
relu_fix = """
# In line 785, add "Relu" to the list
if any(x in node_name for x in ["LayerNorm", "Gather", "Gemm", "Tanh", "Div", 
                                 "Shape", "Slice", "Softmax", "MatMul", "Add", "Relu"]):
"""

print("Fix plan ready. Will implement in actual code.")