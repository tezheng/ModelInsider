#!/usr/bin/env python3
"""
Create Dummy DLC Model - Reverse engineer and create a minimal valid DLC file
This bypasses the converter by directly creating the binary format
"""

import os
import struct
import json
import subprocess
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QNN_SDK_ROOT = Path("C:/Qualcomm/AIStack/qairt/2.34.0.250424")
OUTPUT_DIR = Path("./dummy_dlc_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_existing_model_formats():
    """First, let's check if there are any example DLC files or specs"""
    logger.info("Searching for DLC format information...")
    
    # Check for any DLC-related files in SDK
    dlc_related = []
    for ext in ['*.dlc', '*.DLC', '*dlc*', '*.proto', '*.schema']:
        files = list(QNN_SDK_ROOT.glob(f"**/{ext}"))
        if files:
            for f in files[:5]:
                dlc_related.append(f)
                logger.info(f"  Found: {f.name} ({f.stat().st_size} bytes)")
    
    # Check if there's a protobuf definition
    proto_files = list(QNN_SDK_ROOT.glob("**/*.proto"))
    if proto_files:
        logger.info(f"Found {len(proto_files)} proto files - DLC might use protobuf")
    
    return dlc_related


def create_minimal_dlc_v1():
    """Create a minimal DLC based on common deep learning container formats"""
    logger.info("Creating minimal DLC v1 (basic structure)...")
    
    dlc_data = bytearray()
    
    # DLC Header - Based on analysis of similar formats
    # Most DL containers start with magic bytes
    dlc_data.extend(b'DLC\x00')  # Magic: "DLC" + null
    dlc_data.extend(struct.pack('<I', 1))  # Version: 1
    dlc_data.extend(struct.pack('<I', 0x100))  # Header size: 256 bytes
    
    # Model metadata
    dlc_data.extend(struct.pack('<I', 1))  # Model version
    dlc_data.extend(struct.pack('<I', 1))  # Number of graphs
    dlc_data.extend(struct.pack('<I', 1))  # Number of layers
    dlc_data.extend(struct.pack('<I', 2))  # Number of tensors (input + output)
    
    # Flags and properties
    dlc_data.extend(struct.pack('<I', 0))  # Flags
    dlc_data.extend(struct.pack('<I', 0))  # Reserved
    
    # Input/Output info
    dlc_data.extend(struct.pack('<I', 1))  # Number of inputs
    dlc_data.extend(struct.pack('<I', 1))  # Number of outputs
    dlc_data.extend(struct.pack('<I', 4))  # Input size (1 float = 4 bytes)
    dlc_data.extend(struct.pack('<I', 4))  # Output size (1 float = 4 bytes)
    
    # Pad header to declared size
    while len(dlc_data) < 0x100:
        dlc_data.append(0)
    
    # Graph data section
    # Graph header
    dlc_data.extend(b'GRPH')  # Graph marker
    dlc_data.extend(struct.pack('<I', 1))  # Graph ID
    dlc_data.extend(struct.pack('<I', 1))  # Number of nodes
    
    # Node data (Identity operation)
    dlc_data.extend(b'NODE')  # Node marker
    dlc_data.extend(struct.pack('<I', 0))  # Node ID
    dlc_data.extend(struct.pack('<I', 0))  # Operation type (0 = Identity/Copy)
    dlc_data.extend(struct.pack('<I', 1))  # Number of inputs
    dlc_data.extend(struct.pack('<I', 1))  # Number of outputs
    dlc_data.extend(struct.pack('<I', 0))  # Input tensor ID
    dlc_data.extend(struct.pack('<I', 1))  # Output tensor ID
    
    # Tensor definitions
    dlc_data.extend(b'TENS')  # Tensor section marker
    dlc_data.extend(struct.pack('<I', 2))  # Number of tensors
    
    # Input tensor
    dlc_data.extend(struct.pack('<I', 0))  # Tensor ID
    dlc_data.extend(struct.pack('<I', 1))  # Rank (1D)
    dlc_data.extend(struct.pack('<I', 1))  # Shape[0] = 1
    dlc_data.extend(struct.pack('<I', 0))  # Data type (0 = FLOAT32)
    
    # Output tensor
    dlc_data.extend(struct.pack('<I', 1))  # Tensor ID
    dlc_data.extend(struct.pack('<I', 1))  # Rank (1D)
    dlc_data.extend(struct.pack('<I', 1))  # Shape[0] = 1
    dlc_data.extend(struct.pack('<I', 0))  # Data type (0 = FLOAT32)
    
    # End marker
    dlc_data.extend(b'END\x00')
    
    # Save DLC
    dlc_path = OUTPUT_DIR / "dummy_v1.dlc"
    with open(dlc_path, 'wb') as f:
        f.write(dlc_data)
    
    logger.info(f"Created DLC v1: {dlc_path} ({len(dlc_data)} bytes)")
    return dlc_path


def create_minimal_dlc_v2():
    """Create DLC v2 with more sophisticated structure"""
    logger.info("Creating minimal DLC v2 (protobuf-like structure)...")
    
    dlc_data = bytearray()
    
    # Alternative format - based on SNPE/QNN patterns
    # Magic signature
    dlc_data.extend(b'\x89DLC')  # Similar to PNG signature pattern
    dlc_data.extend(b'\r\n\x1a\n')  # Line ending detection
    
    # File format version
    dlc_data.extend(struct.pack('<H', 3))  # Major version
    dlc_data.extend(struct.pack('<H', 0))  # Minor version
    
    # Model info chunk
    dlc_data.extend(struct.pack('>I', 32))  # Chunk size (big-endian like PNG)
    dlc_data.extend(b'INFO')  # Chunk type
    
    # Model info data
    dlc_data.extend(struct.pack('<I', 1))  # Model ID
    dlc_data.extend(b'Identity\x00\x00\x00\x00\x00\x00\x00\x00')  # Model name (16 bytes)
    dlc_data.extend(struct.pack('<I', 1))  # Input count
    dlc_data.extend(struct.pack('<I', 1))  # Output count
    dlc_data.extend(struct.pack('<I', 0))  # CRC or checksum
    
    # Network chunk
    dlc_data.extend(struct.pack('>I', 64))  # Chunk size
    dlc_data.extend(b'NTWK')  # Network chunk
    
    # Simple network: one Identity layer
    dlc_data.extend(struct.pack('<I', 1))  # Layer count
    
    # Layer definition
    dlc_data.extend(b'LAYR')  # Layer marker
    dlc_data.extend(struct.pack('<I', 20))  # Layer data size
    dlc_data.extend(struct.pack('<I', 0))  # Layer ID
    dlc_data.extend(struct.pack('<I', 100))  # Layer type (100 = Identity)
    dlc_data.extend(b'input\x00\x00\x00')  # Input name (8 bytes)
    dlc_data.extend(b'output\x00\x00')  # Output name (8 bytes)
    
    # Padding to chunk size
    while len(dlc_data) < 8 + 32 + 8 + 64:
        dlc_data.append(0)
    
    # Weights chunk (empty for Identity)
    dlc_data.extend(struct.pack('>I', 8))  # Chunk size
    dlc_data.extend(b'WGHT')  # Weights chunk
    dlc_data.extend(struct.pack('<I', 0))  # No weights
    dlc_data.extend(struct.pack('<I', 0))  # Checksum
    
    # End chunk
    dlc_data.extend(struct.pack('>I', 0))  # Zero size
    dlc_data.extend(b'IEND')  # End marker (like PNG)
    
    # Save DLC
    dlc_path = OUTPUT_DIR / "dummy_v2.dlc"
    with open(dlc_path, 'wb') as f:
        f.write(dlc_data)
    
    logger.info(f"Created DLC v2: {dlc_path} ({len(dlc_data)} bytes)")
    return dlc_path


def create_minimal_dlc_v3():
    """Create DLC v3 - Try to mimic actual QNN format more closely"""
    logger.info("Creating minimal DLC v3 (QNN-specific format)...")
    
    dlc_data = bytearray()
    
    # Based on QNN documentation patterns
    # QNN uses a specific serialization format
    
    # Header
    dlc_data.extend(b'QAIRT')  # Qualcomm AI Runtime
    dlc_data.extend(struct.pack('<H', 2))  # Format version
    dlc_data.extend(struct.pack('<H', 34))  # SDK version hint (2.34)
    
    # Model descriptor
    dlc_data.extend(struct.pack('<I', 1))  # Descriptor version
    dlc_data.extend(struct.pack('<I', 0x200))  # Model offset
    dlc_data.extend(struct.pack('<I', 0x100))  # Model size
    dlc_data.extend(struct.pack('<I', 0))  # Metadata offset
    dlc_data.extend(struct.pack('<I', 0))  # Metadata size
    
    # Graph descriptor
    dlc_data.extend(struct.pack('<I', 1))  # Number of graphs
    dlc_data.extend(struct.pack('<I', 1))  # Default graph index
    
    # Graph header
    dlc_data.extend(struct.pack('<I', 1))  # Graph version
    dlc_data.extend(struct.pack('<I', 1))  # Number of operations
    dlc_data.extend(struct.pack('<I', 2))  # Number of tensors
    dlc_data.extend(struct.pack('<I', 1))  # Number of inputs
    dlc_data.extend(struct.pack('<I', 1))  # Number of outputs
    
    # Pad to model offset
    while len(dlc_data) < 0x200:
        dlc_data.append(0)
    
    # Model data section
    # Operation: Identity/Reshape
    dlc_data.extend(struct.pack('<I', 0x1000))  # Op type (Identity/Reshape)
    dlc_data.extend(struct.pack('<I', 1))  # Input count
    dlc_data.extend(struct.pack('<I', 1))  # Output count
    dlc_data.extend(struct.pack('<I', 0))  # Input tensor index
    dlc_data.extend(struct.pack('<I', 1))  # Output tensor index
    dlc_data.extend(struct.pack('<I', 0))  # Attributes size
    
    # Tensor descriptors
    # Input tensor
    dlc_data.extend(struct.pack('<I', 0))  # Tensor index
    dlc_data.extend(b'input\x00\x00\x00\x00\x00\x00\x00')  # Name (12 bytes)
    dlc_data.extend(struct.pack('<I', 1))  # Rank
    dlc_data.extend(struct.pack('<I', 1))  # Dim[0]
    dlc_data.extend(struct.pack('<I', 10))  # Data type (10 = FLOAT32)
    dlc_data.extend(struct.pack('<I', 4))  # Element size
    
    # Output tensor
    dlc_data.extend(struct.pack('<I', 1))  # Tensor index
    dlc_data.extend(b'output\x00\x00\x00\x00\x00\x00')  # Name (12 bytes)
    dlc_data.extend(struct.pack('<I', 1))  # Rank
    dlc_data.extend(struct.pack('<I', 1))  # Dim[0]
    dlc_data.extend(struct.pack('<I', 10))  # Data type (10 = FLOAT32)
    dlc_data.extend(struct.pack('<I', 4))  # Element size
    
    # Pad to declared model size
    while len(dlc_data) < 0x200 + 0x100:
        dlc_data.append(0)
    
    # Footer
    dlc_data.extend(b'QEND')
    dlc_data.extend(struct.pack('<I', len(dlc_data) + 4))  # Total file size
    
    # Save DLC
    dlc_path = OUTPUT_DIR / "dummy_v3.dlc"
    with open(dlc_path, 'wb') as f:
        f.write(dlc_data)
    
    logger.info(f"Created DLC v3: {dlc_path} ({len(dlc_data)} bytes)")
    return dlc_path


def test_dlc_with_qnn(dlc_path):
    """Test if the DLC file is accepted by QNN"""
    logger.info(f"Testing DLC: {dlc_path}")
    
    # Create minimal input
    input_file = OUTPUT_DIR / "test_input.raw"
    with open(input_file, 'wb') as f:
        f.write(struct.pack('f', 1.0))  # Single float value
    
    # Create input list
    input_list = OUTPUT_DIR / "input_list.txt"
    with open(input_list, 'w') as f:
        # Try different input naming conventions
        f.write(f"input {input_file}\n")
        f.write(f"input:0 {input_file}\n")
        f.write(f"0 {input_file}\n")
    
    # Test with qnn-net-run
    net_run = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-net-run.exe"
    backends = [
        QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll",
        QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    ]
    
    for backend in backends:
        if not backend.exists():
            continue
            
        backend_name = "CPU" if "Cpu" in backend.name else "HTP"
        logger.info(f"  Testing with {backend_name} backend...")
        
        cmd = [
            str(net_run),
            "--model", str(dlc_path),
            "--backend", str(backend),
            "--input_list", str(input_list),
            "--output_dir", str(OUTPUT_DIR),
            "--log_level", "debug"
        ]
        
        try:
            start = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            elapsed = (time.perf_counter() - start) * 1000
            
            if result.returncode == 0:
                logger.info(f"    ✅ SUCCESS! DLC accepted by {backend_name}!")
                logger.info(f"    Execution time: {elapsed:.2f}ms")
                
                # Check for output
                outputs = list(OUTPUT_DIR.glob("Result_*"))
                if outputs:
                    logger.info(f"    Generated {len(outputs)} output files")
                    # Read output
                    with open(outputs[0], 'rb') as f:
                        output_val = struct.unpack('f', f.read(4))[0]
                        logger.info(f"    Output value: {output_val}")
                
                return True
            else:
                logger.warning(f"    ❌ Rejected by {backend_name}")
                if "Invalid model" in result.stderr:
                    logger.debug("    Error: Invalid model format")
                elif "unsupported" in result.stderr.lower():
                    logger.debug("    Error: Unsupported format")
                elif result.stderr:
                    # Parse error for clues about format
                    error_lines = result.stderr.split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            logger.debug(f"    {line.strip()}")
                            
        except subprocess.TimeoutExpired:
            logger.warning(f"    Timeout with {backend_name}")
        except Exception as e:
            logger.error(f"    Error: {e}")
    
    return False


def create_dlc_from_context_binary():
    """Try creating a DLC from a QNN context binary"""
    logger.info("Attempting to create DLC from context binary...")
    
    # First create a context binary
    ctx_gen = QNN_SDK_ROOT / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-generator.exe"
    
    if not ctx_gen.exists():
        logger.warning("Context binary generator not found")
        return None
    
    # We need a model to create context from
    # Let's try using one of our dummy DLCs
    dummy_dlc = OUTPUT_DIR / "dummy_v3.dlc"
    if not dummy_dlc.exists():
        dummy_dlc = create_minimal_dlc_v3()
    
    context_bin = OUTPUT_DIR / "context.bin"
    backend = QNN_SDK_ROOT / "lib" / "aarch64-windows-msvc" / "QnnCpu.dll"
    
    cmd = [
        str(ctx_gen),
        "--model", str(dummy_dlc),
        "--backend", str(backend),
        "--binary_file", str(context_bin),
        "--log_level", "debug"
    ]
    
    logger.info("Generating context binary...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and context_bin.exists():
            logger.info(f"✅ Created context binary: {context_bin}")
            logger.info(f"  Size: {context_bin.stat().st_size} bytes")
            
            # The context binary might be a valid model format
            return context_bin
        else:
            logger.warning("Failed to create context binary")
            if result.stderr:
                logger.debug(f"Error: {result.stderr[:200]}")
                
    except Exception as e:
        logger.error(f"Context generation error: {e}")
    
    return None


def analyze_dlc_with_hexdump(dlc_path):
    """Analyze what QNN expects by looking at error patterns"""
    logger.info(f"Analyzing DLC format requirements from: {dlc_path}")
    
    # Read first 256 bytes
    with open(dlc_path, 'rb') as f:
        data = f.read(256)
    
    logger.info("First 64 bytes (hex):")
    for i in range(0, min(64, len(data)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        logger.info(f"  {i:04x}: {hex_str:<48} {ascii_str}")


def main():
    """Main workflow to create and test dummy DLC models"""
    logger.info("="*80)
    logger.info("CREATING DUMMY DLC MODELS")
    logger.info("="*80)
    
    # Step 1: Analyze existing formats
    existing = analyze_existing_model_formats()
    
    # Step 2: Create different DLC versions
    dlc_files = []
    
    dlc_v1 = create_minimal_dlc_v1()
    dlc_files.append(dlc_v1)
    
    dlc_v2 = create_minimal_dlc_v2()
    dlc_files.append(dlc_v2)
    
    dlc_v3 = create_minimal_dlc_v3()
    dlc_files.append(dlc_v3)
    
    # Step 3: Test each DLC
    logger.info("\n" + "="*80)
    logger.info("TESTING DLC FILES")
    logger.info("="*80)
    
    successful_dlc = None
    
    for dlc in dlc_files:
        logger.info(f"\nTesting: {dlc.name}")
        analyze_dlc_with_hexdump(dlc)
        
        if test_dlc_with_qnn(dlc):
            logger.info(f"🎉 SUCCESS: {dlc.name} is a valid DLC!")
            successful_dlc = dlc
            break
        else:
            logger.info(f"❌ {dlc.name} not accepted")
    
    # Step 4: Try context binary approach
    if not successful_dlc:
        logger.info("\nTrying context binary approach...")
        ctx_bin = create_dlc_from_context_binary()
        if ctx_bin:
            if test_dlc_with_qnn(ctx_bin):
                successful_dlc = ctx_bin
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    if successful_dlc:
        logger.info(f"✅ SUCCESS: Created working DLC model!")
        logger.info(f"  File: {successful_dlc}")
        logger.info(f"  Size: {successful_dlc.stat().st_size} bytes")
        logger.info("\n🎉 We can now run REAL NPU INFERENCE with this dummy model!")
        return True
    else:
        logger.info("❌ Could not create valid DLC format")
        logger.info("\nThe DLC format is proprietary and requires proper conversion tools.")
        logger.info("Options:")
        logger.info("1. Fix Python dependencies (install VC++ ARM64 runtime)")
        logger.info("2. Use pre-converted DLC models from Qualcomm")
        logger.info("3. Convert on another machine with working tools")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        logger.info("\n🎊 BREAKTHROUGH: Dummy DLC model created!")
        logger.info("We can now test full NPU inference metrics!")
    else:
        logger.info("\n📝 DLC format is complex and proprietary")
        logger.info("Need official conversion tools or pre-built models")