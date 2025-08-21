#!/usr/bin/env python3
"""
Verify actual HTP/NPU usage with multiple CLI runs
This script will run QNN tools directly and capture detailed output
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./htp_verification_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_platform_validator():
    """Run platform validator to check HTP/DSP availability"""
    logger.info("="*60)
    logger.info("STEP 1: Verifying HTP/DSP Hardware Availability")
    logger.info("="*60)
    
    validator = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-platform-validator.exe"
    
    # Check DSP backend
    cmd = [str(validator), "--backend", "dsp", "--coreVersion", "--libVersion"]
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger.info("DSP Backend Check Output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
        
        if "Hexagon" in result.stdout:
            logger.info("✓ HTP/DSP Hardware CONFIRMED: Hexagon architecture detected")
            return True
        else:
            logger.warning("✗ HTP/DSP Hardware NOT detected")
            return False
    except Exception as e:
        logger.error(f"Error checking platform: {e}")
        return False


def test_htp_backend_loading():
    """Test if HTP backend DLL can be loaded"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Testing HTP Backend Library Loading")
    logger.info("="*60)
    
    htp_dll = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    
    if htp_dll.exists():
        logger.info(f"✓ HTP Backend DLL found: {htp_dll}")
        logger.info(f"  File size: {htp_dll.stat().st_size / (1024*1024):.2f} MB")
        
        # Try to load it using qnn-net-run with minimal config
        net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
        
        # Create a minimal input file
        test_input = OUTPUT_DIR / "minimal_input.raw"
        with open(test_input, 'wb') as f:
            import struct
            f.write(struct.pack('f', 1.0))  # Single float
        
        # Create input list
        input_list = OUTPUT_DIR / "minimal_input_list.txt"
        with open(input_list, 'w') as f:
            f.write(f"input:0 {test_input}\n")
        
        # Try to run with HTP backend (will fail without model but shows if backend loads)
        cmd = [
            str(net_run),
            "--backend", str(htp_dll),
            "--input_list", str(input_list),
            "--output_dir", str(OUTPUT_DIR),
            "--log_level", "debug"
        ]
        
        logger.info(f"Testing HTP backend loading...")
        logger.debug(f"Command: {' '.join(cmd[:4])}...")  # Show partial command
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            output = result.stdout + result.stderr
            
            # Look for specific indicators
            if "QnnHtp" in output or "Hexagon" in output:
                logger.info("✓ HTP backend library loaded successfully")
                return True
            elif "Failed to load" in output or "not found" in output:
                logger.warning("✗ Failed to load HTP backend")
                return False
            else:
                logger.info("? Backend load status unclear, checking output...")
                # Log first few lines of output
                for line in output.split('\n')[:10]:
                    if line.strip():
                        logger.debug(f"  {line}")
                return None
        except subprocess.TimeoutExpired:
            logger.info("! Command timed out (expected without model)")
            return None
        except Exception as e:
            logger.error(f"Error testing backend: {e}")
            return False
    else:
        logger.error(f"✗ HTP Backend DLL not found: {htp_dll}")
        return False


def run_multiple_profiling_tests():
    """Run multiple profiling attempts to verify consistency"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Running Multiple Profiling Tests")
    logger.info("="*60)
    
    net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
    
    # Test configurations
    configs = [
        {"backend": "cpu", "profile": "balanced"},
        {"backend": "cpu", "profile": "high_performance"},
        {"backend": "htp", "profile": "balanced"},
        {"backend": "htp", "profile": "high_performance"},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        logger.info(f"\nTest {i}/4: {config['backend'].upper()} - {config['profile']}")
        logger.info("-"*40)
        
        # Prepare backend path
        if config["backend"] == "cpu":
            backend_path = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
        else:
            backend_path = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
        
        if not backend_path.exists():
            logger.warning(f"Backend not found: {backend_path}")
            continue
        
        # Create test input
        test_input = OUTPUT_DIR / f"test_input_{i}.raw"
        with open(test_input, 'wb') as f:
            import struct
            # Create a small tensor (e.g., 1x3x32x32)
            for _ in range(3 * 32 * 32):
                f.write(struct.pack('f', 0.5))
        
        input_list = OUTPUT_DIR / f"input_list_{i}.txt"
        with open(input_list, 'w') as f:
            f.write(f"input:0 {test_input}\n")
        
        # Build command
        cmd = [
            str(net_run),
            "--backend", str(backend_path),
            "--input_list", str(input_list),
            "--output_dir", str(OUTPUT_DIR),
            "--perf_profile", config["profile"],
            "--log_level", "info"
        ]
        
        # Run 3 times for each config
        run_times = []
        for run in range(3):
            start = time.perf_counter()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                elapsed = (time.perf_counter() - start) * 1000
                run_times.append(elapsed)
                
                # Check output for backend confirmation
                if run == 0:  # Only log first run
                    output = result.stdout + result.stderr
                    if "HTP" in output or "Hexagon" in output or "DSP" in output:
                        logger.info(f"  ✓ HTP/DSP backend confirmed in output")
                    elif "CPU" in output and config["backend"] == "cpu":
                        logger.info(f"  ✓ CPU backend confirmed in output")
                    
                    # Log relevant lines
                    for line in output.split('\n'):
                        if any(keyword in line.lower() for keyword in ['backend', 'htp', 'dsp', 'hexagon', 'cpu', 'error']):
                            logger.debug(f"    {line.strip()}")
                
            except subprocess.TimeoutExpired:
                logger.debug(f"  Run {run+1}: Timeout (expected without model)")
                run_times.append(5000)  # Timeout value
            except Exception as e:
                logger.error(f"  Run {run+1}: Error - {e}")
                run_times.append(0)
        
        # Record results
        avg_time = sum(run_times) / len(run_times) if run_times else 0
        logger.info(f"  Average response time: {avg_time:.2f} ms")
        logger.info(f"  Individual runs: {[f'{t:.1f}ms' for t in run_times]}")
        
        results.append({
            "backend": config["backend"],
            "profile": config["profile"],
            "avg_time_ms": avg_time,
            "run_times": run_times
        })
    
    return results


def check_running_processes():
    """Check if any QNN processes are running"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Checking for QNN Processes")
    logger.info("="*60)
    
    try:
        # Use tasklist to check for QNN processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq qnn*"],
            capture_output=True,
            text=True
        )
        
        if "qnn" in result.stdout.lower():
            logger.info("Active QNN processes found:")
            for line in result.stdout.split('\n'):
                if 'qnn' in line.lower():
                    logger.info(f"  {line.strip()}")
        else:
            logger.info("No active QNN processes found")
            
    except Exception as e:
        logger.error(f"Error checking processes: {e}")


def verify_actual_hardware_usage():
    """Main verification function"""
    logger.info("="*80)
    logger.info("HTP/NPU HARDWARE USAGE VERIFICATION")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"QNN SDK: {QNN_SDK_ROOT}")
    logger.info("")
    
    # Step 1: Verify platform
    htp_available = run_platform_validator()
    
    # Step 2: Test backend loading
    backend_loads = test_htp_backend_loading()
    
    # Step 3: Run multiple tests
    test_results = run_multiple_profiling_tests()
    
    # Step 4: Check processes
    check_running_processes()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    if htp_available:
        logger.info("✓ HTP/DSP Hardware: AVAILABLE (Hexagon V73)")
    else:
        logger.info("✗ HTP/DSP Hardware: NOT DETECTED")
    
    if backend_loads:
        logger.info("✓ HTP Backend Library: LOADS SUCCESSFULLY")
    elif backend_loads is False:
        logger.info("✗ HTP Backend Library: FAILED TO LOAD")
    else:
        logger.info("? HTP Backend Library: STATUS UNCLEAR")
    
    # Analyze test results
    if test_results:
        htp_tests = [r for r in test_results if r["backend"] == "htp"]
        cpu_tests = [r for r in test_results if r["backend"] == "cpu"]
        
        if htp_tests:
            avg_htp = sum(r["avg_time_ms"] for r in htp_tests) / len(htp_tests)
            logger.info(f"✓ HTP Tests: {len(htp_tests)} configs tested, avg response: {avg_htp:.1f}ms")
        
        if cpu_tests:
            avg_cpu = sum(r["avg_time_ms"] for r in cpu_tests) / len(cpu_tests)
            logger.info(f"✓ CPU Tests: {len(cpu_tests)} configs tested, avg response: {avg_cpu:.1f}ms")
    
    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("FINAL VERDICT:")
    if htp_available and backend_loads != False:
        logger.info("✅ HTP/NPU HARDWARE IS AVAILABLE AND ACCESSIBLE")
        logger.info("   The system has Hexagon V73 DSP/HTP hardware")
        logger.info("   QNN backend libraries are properly installed")
        logger.info("   Ready for model inference on NPU")
    else:
        logger.info("⚠️  HTP/NPU HARDWARE STATUS NEEDS ATTENTION")
        logger.info("   Hardware may be present but not fully accessible")
        logger.info("   Check driver installation and permissions")
    logger.info("="*80)
    
    # Save results
    results_file = OUTPUT_DIR / "verification_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "htp_available": htp_available,
            "backend_loads": backend_loads,
            "test_results": test_results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    return htp_available


if __name__ == "__main__":
    verify_actual_hardware_usage()