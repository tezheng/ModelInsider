#!/bin/bash

# Convert DeepSeek GGUF model to QNN Context Binary
# This script demonstrates native GGUF support in QNN SDK

echo "============================================================"
echo "🚀 DeepSeek GGUF to QNN Converter"
echo "============================================================"

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GGUF_MODEL="$SCRIPT_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
OUTPUT_DIR="$SCRIPT_DIR/output"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if model exists
if [ ! -f "$GGUF_MODEL" ]; then
    echo "❌ Error: GGUF model not found at: $GGUF_MODEL"
    echo "Please ensure the model is in the models/ directory"
    exit 1
fi

echo "✅ Found GGUF model: $(basename $GGUF_MODEL)"
echo "📊 Model size: $(du -h $GGUF_MODEL | cut -f1)"
echo ""

# Check for QNN SDK
if [ ! -d "$QNN_SDK_ROOT" ]; then
    echo "⚠️ QNN SDK not found at: $QNN_SDK_ROOT"
    echo "Running simulation mode..."
    python "$SCRIPT_DIR/run_conversion.py"
    exit $?
fi

echo "✅ QNN SDK found at: $QNN_SDK_ROOT"
echo ""

# Find qairt-converter
QAIRT_CONVERTER=""
if [ -f "$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter.exe" ]; then
    QAIRT_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter.exe"
elif [ -f "$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter" ]; then
    QAIRT_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qairt-converter"
elif [ -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter" ]; then
    QAIRT_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter"
fi

if [ -z "$QAIRT_CONVERTER" ]; then
    echo "❌ Error: qairt-converter not found in QNN SDK"
    echo "Running simulation mode..."
    python "$SCRIPT_DIR/run_conversion.py"
    exit $?
fi

echo "🔧 Using converter: $QAIRT_CONVERTER"
echo ""

# Output paths
DLC_OUTPUT="$OUTPUT_DIR/deepseek_qwen_1.5b.dlc"
CTX_OUTPUT="$OUTPUT_DIR/deepseek_qwen_1.5b.bin"

# Step 1: Convert GGUF to DLC using native support
echo "============================================================"
echo "🔄 Step 1: Native GGUF to DLC Conversion"
echo "============================================================"

python "$QAIRT_CONVERTER" \
    --input_network "$GGUF_MODEL" \
    --output_path "$DLC_OUTPUT" \
    --input_layout "input_ids,NONTRIVIAL" \
    --input_layout "attention_mask,NONTRIVIAL" \
    --preserve_io datatype,input_ids,attention_mask \
    --float_fallback \
    --float_bitwidth 16 \
    --enable_cpu_fallback

if [ $? -eq 0 ]; then
    echo "✅ DLC generation successful: $DLC_OUTPUT"
else
    echo "❌ DLC generation failed"
    exit 1
fi

# Step 2: Generate context binary (optional)
echo ""
echo "============================================================"
echo "🔄 Step 2: Context Binary Generation"
echo "============================================================"

# Find context generator
CTX_GENERATOR=""
if [ -f "$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe" ]; then
    CTX_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe"
elif [ -f "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" ]; then
    CTX_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
fi

# Find HTP backend library
BACKEND_LIB=""
if [ -f "$QNN_SDK_ROOT/lib/x86_64-windows-msvc/libQnnHtp.dll" ]; then
    BACKEND_LIB="$QNN_SDK_ROOT/lib/x86_64-windows-msvc/libQnnHtp.dll"
elif [ -f "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" ]; then
    BACKEND_LIB="$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"
fi

if [ -n "$CTX_GENERATOR" ] && [ -n "$BACKEND_LIB" ]; then
    "$CTX_GENERATOR" \
        --dlc_path "$DLC_OUTPUT" \
        --backend "$BACKEND_LIB" \
        --binary_file "$CTX_OUTPUT" \
        --output_dir "$OUTPUT_DIR" \
        --target_arch sm8650
    
    if [ $? -eq 0 ]; then
        echo "✅ Context binary generated: $CTX_OUTPUT"
    else
        echo "⚠️ Context binary generation failed (optional)"
    fi
else
    echo "⚠️ Context binary generator not found (optional)"
fi

# Summary
echo ""
echo "============================================================"
echo "📋 Conversion Summary"
echo "============================================================"
echo "✅ Input: $(basename $GGUF_MODEL)"
echo "✅ DLC: $(basename $DLC_OUTPUT)"
if [ -f "$CTX_OUTPUT" ]; then
    echo "✅ Context: $(basename $CTX_OUTPUT)"
fi
echo "✅ Output directory: $OUTPUT_DIR"
echo ""
echo "🎉 Conversion complete! Model ready for NPU deployment."
echo "============================================================"