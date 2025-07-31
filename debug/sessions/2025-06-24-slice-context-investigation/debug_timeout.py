#!/usr/bin/env python3
"""
Debug script to isolate the timeout issue in HTP export.
"""

import signal
import sys

from transformers import AutoModel, AutoTokenizer

from modelexport.hierarchy_exporter import HierarchyExporter


def timeout_handler(signum, frame):
    print("TIMEOUT: Script execution taking too long!")
    print("Current frame info:")
    print(f"  File: {frame.f_code.co_filename}")
    print(f"  Function: {frame.f_code.co_name}")
    print(f"  Line: {frame.f_lineno}")
    sys.exit(1)

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    print("Loading BERT model...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    print("Creating inputs...")
    inputs = tokenizer("Hello world", return_tensors="pt")
    example_inputs = (inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"])
    
    print("Creating exporter...")
    exporter = HierarchyExporter(strategy='htp')
    
    print("Starting export (this is where timeout likely occurs)...")
    result = exporter.export(model, example_inputs, 'temp/debug_bert_test.onnx')
    
    print("Export completed successfully!")
    signal.alarm(0)  # Cancel timeout
    
except Exception as e:
    print(f"Error during export: {e}")
    import traceback
    traceback.print_exc()
    signal.alarm(0)