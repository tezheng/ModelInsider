#!/bin/bash
# QNN Converter Wrapper Script
# Auto-generated to handle library path issues

export QNN_SDK_ROOT="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"
export PYTHONPATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python:$PYTHONPATH"
export PATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

# Add platform-specific Python library to path
export PYTHONPATH="/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python/qti/aisw/converters/common/linux-x86_64:$PYTHONPATH"

echo "🚀 Running QNN Converter with proper environment..."
echo "   QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "   Using converter: /mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-linux-clang/qairt-converter"

# Run the actual converter
exec python3 "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-linux-clang/qairt-converter" "$@"
