#!/bin/bash
# Real GGUF to QNN conversion script
# Generated: perform_real_conversion.py

export QNN_SDK_ROOT="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"
export PYTHONPATH="${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}"
export PATH="${QNN_SDK_ROOT}/bin/x86_64-windows-msvc:${PATH}"

echo "Converting GGUF to QNN DLC..."
python "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-windows-msvc/qairt-converter" \
    --input_network "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf" \
    --output_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_real.dlc" \
    --float_fallback \
    --float_bitwidth 16 \
    --enable_cpu_fallback

if [ $? -eq 0 ]; then
    echo "✅ Conversion successful!"
    echo "DLC created at: /home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_real.dlc"
    
    # Generate context binary
    echo "Generating context binary..."
    "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe" \
        --dlc_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_real.dlc" \
        --backend "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/x86_64-windows-msvc/libQnnHtp.dll" \
        --binary_file "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_real.bin" \
        --output_dir "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp"
else
    echo "❌ Conversion failed"
fi
