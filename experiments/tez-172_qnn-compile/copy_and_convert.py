#!/usr/bin/env python3
"""
Copy GGUF model to Windows and attempt conversion with local paths.
This avoids UNC path issues and uses Windows native paths.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def copy_model_to_windows():
    """Copy the GGUF model to Windows temp directory to avoid UNC path issues"""
    
    script_dir = Path(__file__).parent
    gguf_source = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    
    if not gguf_source.exists():
        print(f"ERROR: GGUF model not found: {gguf_source}")
        return None
    
    # Copy to C:\temp for local access
    windows_temp = Path("/mnt/c/temp")
    windows_temp.mkdir(exist_ok=True)
    
    gguf_dest = windows_temp / "deepseek_model.gguf"
    
    print(f"Copying model to Windows temp directory...")
    print(f"  From: {gguf_source}")
    print(f"  To: {gguf_dest}")
    
    shutil.copy2(gguf_source, gguf_dest)
    
    size_mb = gguf_dest.stat().st_size / (1024**2)
    print(f"  Copied successfully: {size_mb:.1f} MB")
    
    return gguf_dest

def create_windows_batch_conversion():
    """Create a Windows batch script for local conversion"""
    
    batch_content = r'''@echo off
echo ======================================================================
echo QNN Native Conversion (Local Paths)
echo ======================================================================

set QNN_SDK_ROOT=C:\Qualcomm\AIStack\qairt\2.34.0.250424
set PYTHONPATH=%QNN_SDK_ROOT%\lib\python
set PATH=%QNN_SDK_ROOT%\bin\x86_64-windows-msvc;%PATH%

set INPUT_MODEL=C:\temp\deepseek_model.gguf
set OUTPUT_DLC=C:\temp\deepseek_local.dlc
set OUTPUT_BIN=C:\temp\deepseek_local.bin

echo Input: %INPUT_MODEL%
echo Output: %OUTPUT_DLC%
echo.

echo Checking if input exists...
if not exist "%INPUT_MODEL%" (
    echo ERROR: Input model not found
    pause
    exit /b 1
)

echo Step 1: GGUF to DLC Conversion
echo ----------------------------------------------------------------------
python "%QNN_SDK_ROOT%\bin\x86_64-windows-msvc\qairt-converter" ^
    --input_network "%INPUT_MODEL%" ^
    --output_path "%OUTPUT_DLC%" ^
    --input_layout input_ids,NONTRIVIAL ^
    --input_layout attention_mask,NONTRIVIAL ^
    --preserve_io datatype,input_ids,attention_mask ^
    --float_fallback ^
    --float_bitwidth 16 ^
    --enable_cpu_fallback

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] DLC generated: %OUTPUT_DLC%
    
    echo.
    echo Step 2: Generating Context Binary
    echo ----------------------------------------------------------------------
    "%QNN_SDK_ROOT%\bin\x86_64-windows-msvc\qnn-context-binary-generator.exe" ^
        --dlc_path "%OUTPUT_DLC%" ^
        --backend "%QNN_SDK_ROOT%\lib\x86_64-windows-msvc\libQnnHtp.dll" ^
        --binary_file "%OUTPUT_BIN%" ^
        --output_dir "C:\temp" ^
        --target_arch sm8650
    
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] Context binary generated: %OUTPUT_BIN%
    ) else (
        echo [WARNING] Context binary generation failed
    )
    
    echo.
    echo Copying results back to WSL...
    copy "%OUTPUT_DLC%" "\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_real.dlc" 2>nul
    copy "%OUTPUT_BIN%" "\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp\deepseek_real.bin" 2>nul
    
) else (
    echo [ERROR] DLC generation failed
)

echo.
echo ======================================================================
echo Conversion Complete!
echo ======================================================================
pause
'''
    
    batch_script = Path("/mnt/c/temp/convert_local.bat")
    with open(batch_script, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    return batch_script

def run_local_conversion():
    """Run conversion with local Windows paths"""
    
    print("=" * 60)
    print("Local Path QNN Conversion")
    print("=" * 60)
    print("Copying model locally to avoid UNC path issues")
    print()
    
    # Step 1: Copy model
    gguf_dest = copy_model_to_windows()
    if not gguf_dest:
        return 1
    
    # Step 2: Create batch script
    print("\nCreating local conversion batch script...")
    batch_script = create_windows_batch_conversion()
    print(f"Batch script: {batch_script}")
    
    # Step 3: Run batch script
    print("\nRunning local Windows conversion...")
    
    try:
        result = subprocess.run(
            [str(batch_script)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd="C:\\temp",
            encoding='utf-8',
            errors='replace'
        )
        
        print("Windows Batch Output:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check for results in temp directory
        temp_dir = Path(__file__).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        dlc_file = temp_dir / "deepseek_real.dlc"
        bin_file = temp_dir / "deepseek_real.bin"
        
        # Also check C:\temp directly
        c_dlc = Path("/mnt/c/temp/deepseek_local.dlc")
        c_bin = Path("/mnt/c/temp/deepseek_local.bin")
        
        print("\n" + "=" * 60)
        print("LOCAL CONVERSION RESULTS:")
        print("=" * 60)
        
        if c_dlc.exists():
            size_mb = c_dlc.stat().st_size / (1024**2)
            print(f"SUCCESS: DLC created - {c_dlc} ({size_mb:.1f} MB)")
            
            # Copy to our temp directory if not already there
            if not dlc_file.exists():
                shutil.copy2(c_dlc, dlc_file)
                print(f"Copied to: {dlc_file}")
        else:
            print("FAILED: No DLC file created")
        
        if c_bin.exists():
            size_mb = c_bin.stat().st_size / (1024**2)
            print(f"SUCCESS: Binary created - {c_bin} ({size_mb:.1f} MB)")
            
            if not bin_file.exists():
                shutil.copy2(c_bin, bin_file)
                print(f"Copied to: {bin_file}")
        else:
            print("INFO: No binary file")
        
        success = c_dlc.exists()
        if success:
            print("\nSUCCESS: Real QNN conversion completed!")
            print("Files ready for Snapdragon NPU deployment")
            
            # Cleanup Windows temp
            try:
                os.remove(gguf_dest)
                print("Cleaned up temporary model copy")
            except:
                pass
                
            return 0
        else:
            print("\nFAILED: No output files generated")
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_local_conversion())