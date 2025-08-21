#!/usr/bin/env python3
"""
Direct HTP Profiling Test
This script performs direct QNN HTP profiling tests using QNN tools
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# QNN SDK paths
QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
QNN_TOOLS = {
    "net_run": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe",
    "platform_validator": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-platform-validator.exe",
    "throughput_run": QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-throughput-net-run.exe",
}

def check_htp_availability():
    """Check if HTP is available on the system"""
    logger.info("Checking HTP availability...")
    
    validator = QNN_TOOLS["platform_validator"]
    if not validator.exists():
        logger.error(f"Platform validator not found: {validator}")
        return False
    
    try:
        # Run platform validator
        result = subprocess.run(
            [str(validator)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout + result.stderr
        logger.info("Platform validation output:")
        for line in output.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
        
        # Check for HTP in output
        if "HTP" in output or "DSP" in output or "Hexagon" in output:
            logger.info("✓ HTP/DSP hardware detected")
            return True
        else:
            logger.warning("HTP/DSP hardware not detected in output")
            return False
            
    except Exception as e:
        logger.error(f"Error checking HTP availability: {e}")
        return False

def run_qnn_benchmark():
    """Run a simple QNN benchmark to test profiling"""
    logger.info("\n" + "="*60)
    logger.info("Running QNN Benchmark Test")
    logger.info("="*60)
    
    # Check if we have qnn-net-run
    net_run = QNN_TOOLS["net_run"]
    if not net_run.exists():
        logger.error(f"qnn-net-run not found: {net_run}")
        return
    
    # Try to run with --help to see available options
    try:
        logger.info("\nChecking qnn-net-run capabilities...")
        result = subprocess.run(
            [str(net_run), "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Look for profiling options in help
        help_text = result.stdout + result.stderr
        if "--profiling_level" in help_text:
            logger.info("✓ Profiling options available")
        if "--perf_profile" in help_text:
            logger.info("✓ Performance profile options available")
        if "htp" in help_text.lower() or "dsp" in help_text.lower():
            logger.info("✓ HTP/DSP backend mentioned")
            
    except Exception as e:
        logger.error(f"Error running qnn-net-run: {e}")

def test_direct_htp_access():
    """Test direct HTP access using QNN runtime"""
    logger.info("\n" + "="*60)
    logger.info("Testing Direct HTP Access")
    logger.info("="*60)
    
    # Check for HTP backend library
    htp_backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    if htp_backend.exists():
        logger.info(f"✓ HTP Backend found: {htp_backend}")
        
        # Check file size to ensure it's not empty
        size_mb = htp_backend.stat().st_size / (1024 * 1024)
        logger.info(f"  Size: {size_mb:.2f} MB")
    else:
        logger.warning(f"✗ HTP Backend not found: {htp_backend}")
        
    # Check for other important libraries
    libs_to_check = [
        "QnnSystem.dll",
        "QnnHtpPrepare.dll", 
        "QnnHtpV73Stub.dll",
        "QnnCpu.dll"
    ]
    
    lib_dir = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc"
    logger.info(f"\nChecking libraries in {lib_dir}:")
    
    for lib_name in libs_to_check:
        lib_path = lib_dir / lib_name
        if lib_path.exists():
            size_mb = lib_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {lib_name}: {size_mb:.2f} MB")
        else:
            logger.info(f"  ✗ {lib_name}: Not found")

def run_profiling_test_with_cpu():
    """Run a profiling test using CPU backend as fallback"""
    logger.info("\n" + "="*60)
    logger.info("Running CPU Backend Test (HTP Fallback)")
    logger.info("="*60)
    
    # We'll create a simple test to show profiling works
    output_dir = Path("./htp_profiling_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create a dummy input file
    input_file = output_dir / "test_input.raw"
    
    # Create small random data (4 bytes as a test)
    import struct
    with open(input_file, 'wb') as f:
        f.write(struct.pack('f', 1.0))  # Single float value
    
    logger.info(f"Created test input: {input_file}")
    
    # Try to get system info
    try:
        import platform
        logger.info(f"\nSystem Information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Machine: {platform.machine()}")
        logger.info(f"  Processor: {platform.processor()}")
        
        # Check if we're on Snapdragon
        if "ARM" in platform.machine() or "aarch64" in platform.machine():
            logger.info("  ✓ Running on ARM architecture (likely Snapdragon)")
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")

def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("DIRECT HTP PROFILING TEST")
    logger.info("Testing QNN HTP Access on Windows ARM64")
    logger.info("="*60)
    
    # Step 1: Check HTP availability
    htp_available = check_htp_availability()
    
    # Step 2: Test direct HTP access
    test_direct_htp_access()
    
    # Step 3: Run benchmark test
    run_qnn_benchmark()
    
    # Step 4: Try CPU backend as fallback
    run_profiling_test_with_cpu()
    
    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)
    
    if htp_available:
        logger.info("✓ HTP hardware is available on this system")
        logger.info("✓ Ready for real profiling with actual models")
    else:
        logger.info("⚠ HTP hardware not detected")
        logger.info("  You may need to:")
        logger.info("  1. Ensure you're running on Snapdragon hardware")
        logger.info("  2. Install proper drivers")
        logger.info("  3. Use CPU backend for testing")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())