#!/usr/bin/env python3
import os
import sys

# Set up QNN environment
os.environ['QNN_SDK_ROOT'] = '/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424'
os.environ['PYTHONPATH'] = f'/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python:{os.environ.get("PYTHONPATH", "")}'
os.environ['LD_LIBRARY_PATH'] = f'/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/x86_64-linux-clang:{os.environ.get("LD_LIBRARY_PATH", "")}'

sys.path.insert(0, '/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python/qti/aisw/converters/common/linux-x86_64')
sys.path.insert(0, '/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/lib/python')

try:
    from qti.aisw.converters.common import ir_graph
    print("✅ QNN import successful!")
    return True
except ImportError as e:
    print(f"❌ QNN import failed: {e}")
    return False
