#!/bin/bash

# Manual command to run in terminal:
export QNN_SDK_ROOT="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"
export PYTHONPATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

python3 "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-linux-clang/qairt-converter" \
    --input_network "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf" \
    --output_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_linux.dlc" \
    --input_layout "input_ids,NONTRIVIAL" \
    --input_layout "attention_mask,NONTRIVIAL" \
    --preserve_io "datatype,input_ids,attention_mask" \
    --float_fallback \
    --float_bitwidth 16 \
    --enable_cpu_fallback
