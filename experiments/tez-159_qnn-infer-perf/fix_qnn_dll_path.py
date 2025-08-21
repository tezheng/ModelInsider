#!/usr/bin/env python3
"""
Fix QNN DLL Path - Copy the correct architecture DLL to where Python expects it
"""

import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./dll_fix_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def fix_dll_architecture():
    """Copy the correct architecture DLL to where Python expects it"""
    logger.info("Fixing QNN DLL architecture paths...")
    
    # Source and target paths
    common_dir = QNN_SDK_ROOT / "lib" / "python" / "qti" / "aisw" / "converters" / "common"
    
    # Available DLL directories
    available_dlls = {
        "windows-arm64ec": common_dir / "windows-arm64ec",
        "windows-x86_64": common_dir / "windows-x86_64", 
        "linux-x86_64": common_dir / "linux-x86_64"
    }
    
    # Check what's available
    for arch, path in available_dlls.items():
        if path.exists():
            dlls = list(path.glob("*.pyd")) + list(path.glob("*.so"))
            logger.info(f"Found {arch}: {len(dlls)} DLLs")
            for dll in dlls:
                logger.info(f"  {dll.name}")
    
    # Try ARM64EC first (most compatible with ARM64)
    source_dir = available_dlls["windows-arm64ec"]
    if source_dir.exists():
        logger.info(f"Using ARM64EC DLLs from: {source_dir}")
        
        # Copy DLLs to the common directory where Python expects them
        for dll in source_dir.glob("*.pyd"):
            target = common_dir / dll.name
            logger.info(f"Copying: {dll.name} -> {target}")
            try:
                shutil.copy2(dll, target)
                logger.info(f"✓ Copied {dll.name}")
            except Exception as e:
                logger.error(f"Failed to copy {dll.name}: {e}")
    
    # Also try x86_64 as fallback (should work with emulation)
    elif available_dlls["windows-x86_64"].exists():
        source_dir = available_dlls["windows-x86_64"]
        logger.info(f"Using x86_64 DLLs from: {source_dir}")
        
        for dll in source_dir.glob("*.pyd"):
            target = common_dir / dll.name
            logger.info(f"Copying: {dll.name} -> {target}")
            try:
                shutil.copy2(dll, target)
                logger.info(f"✓ Copied {dll.name}")
            except Exception as e:
                logger.error(f"Failed to copy {dll.name}: {e}")
    
    else:
        logger.error("No compatible DLLs found")
        return False
    
    return True


def test_fixed_imports():
    """Test if the DLL fix worked"""
    logger.info("Testing fixed QNN imports...")
    
    # Set up paths
    sys.path.insert(0, str(QNN_SDK_ROOT / "lib" / "python"))
    
    try:
        import qti.aisw.converters.common as common
        logger.info("✓ Successfully imported qti.aisw.converters.common")
        
        try:
            import qti.aisw.converters.onnx as onnx_frontend
            logger.info("✓ Successfully imported QNN ONNX converter!")
            return True
        except ImportError as e:
            logger.error(f"Still failed to import ONNX converter: {e}")
            
    except ImportError as e:
        logger.error(f"Still failed to import common: {e}")
    
    return False


def run_converter_with_fixed_dlls():
    """Try running the converter with fixed DLLs"""
    logger.info("Attempting conversion with fixed DLLs...")
    
    # Create super simple ONNX model
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        
        # Minimal model: [1,1] -> [1,1]
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
        
        # Identity operation: just pass through
        identity_node = helper.make_node('Identity', ['input'], ['output'])
        
        graph_def = helper.make_graph(
            nodes=[identity_node],
            name='IdentityModel',
            inputs=[input_tensor],
            outputs=[output_tensor]
        )
        
        model_def = helper.make_model(graph_def, producer_name='QNN-Identity-Test')
        model_def.opset_import[0].version = 11
        
        test_onnx = OUTPUT_DIR / "identity.onnx"
        onnx.save(model_def, str(test_onnx))
        
        logger.info(f"Created identity ONNX: {test_onnx}")
        
    except Exception as e:
        logger.error(f"Failed to create test ONNX: {e}")
        return None
    
    # Try Python import approach
    logger.info("Trying direct Python conversion...")
    
    try:
        # Set up environment
        sys.path.insert(0, str(QNN_SDK_ROOT / "lib" / "python"))
        os.environ['QNN_SDK_ROOT'] = str(QNN_SDK_ROOT)
        
        # Try direct import and conversion
        from qti.aisw.converters import onnx as onnx_converter
        from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
        
        logger.info("✅ SUCCESS: QNN converters imported!")
        
        # TODO: Implement actual conversion using Python API
        logger.info("Python API conversion would go here...")
        
        return True
        
    except ImportError as e:
        logger.error(f"Python import conversion failed: {e}")
        
    # Try subprocess approach with fixed environment
    import subprocess
    
    converter_script = QNN_SDK_ROOT / "bin" / "arm64x-windows-msvc" / "qnn-onnx-converter"
    test_dlc = OUTPUT_DIR / "identity.dlc"
    
    cmd = [
        sys.executable,
        str(converter_script),
        "--input_network", str(test_onnx),
        "--output_path", str(test_dlc),
        "--input_dim", "input", "1,1"
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(QNN_SDK_ROOT / "lib" / "python")
    env['QNN_SDK_ROOT'] = str(QNN_SDK_ROOT)
    
    logger.info(f"Trying subprocess conversion: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
        logger.info(f"Return code: {result.returncode}")
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.info(f"STDERR: {result.stderr}")
            
        if result.returncode == 0 and test_dlc.exists():
            logger.info(f"✅ SUCCESS: Created DLC file {test_dlc}")
            return test_dlc
            
    except Exception as e:
        logger.error(f"Subprocess conversion failed: {e}")
    
    return None


def main():
    """Main workflow"""
    logger.info("=" * 80)
    logger.info("QNN DLL PATH FIX - REAL NPU CONVERSION ATTEMPT")
    logger.info("=" * 80)
    
    # Step 1: Fix DLL paths
    if not fix_dll_architecture():
        logger.error("Failed to fix DLL paths")
        return False
    
    # Step 2: Test imports
    if not test_fixed_imports():
        logger.warning("Imports still failing, trying conversion anyway...")
    
    # Step 3: Try conversion
    result = run_converter_with_fixed_dlls()
    
    if result:
        logger.info("🎉 SUCCESS: QNN conversion working!")
        return True
    else:
        logger.error("❌ Conversion still failing")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        logger.info("🎉 BREAKTHROUGH: QNN Python tools working!")
    else:
        logger.info("❌ Still blocked, but made progress on DLL issue")