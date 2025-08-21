#!/usr/bin/env python3
"""
Hybrid approach: Use Windows QNN tools from WSL for model conversion.
Since the target is Windows ARM64 anyway, this makes sense.
"""

import os
import sys
import subprocess
from pathlib import Path

def convert_using_windows_tools():
    """Use Windows QNN tools from WSL"""
    
    print("=" * 70)
    print("🔀 Hybrid Windows/Linux QNN Conversion")
    print("=" * 70)
    print("Using Windows QNN tools from WSL environment")
    print("This works because the target deployment is Windows ARM64")
    print()
    
    # Paths
    script_dir = Path(__file__).parent
    gguf_path = script_dir / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf"
    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # QNN SDK Windows tools
    qnn_sdk_root = Path("/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424")
    
    # Use Windows ARM64 tools since that's our target platform
    converter = qnn_sdk_root / "bin" / "aarch64-windows-msvc" / "qairt-converter"
    if not converter.exists():
        converter = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qairt-converter" 
    
    ctx_generator = qnn_sdk_root / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-generator.exe"
    if not ctx_generator.exists():
        ctx_generator = qnn_sdk_root / "bin" / "x86_64-windows-msvc" / "qnn-context-binary-generator.exe"
    
    backend_lib = qnn_sdk_root / "lib" / "aarch64-windows-msvc" / "QnnHtp.dll"
    if not backend_lib.exists():
        backend_lib = qnn_sdk_root / "lib" / "x86_64-windows-msvc" / "libQnnHtp.dll"
    
    print(f"📁 QNN SDK: {qnn_sdk_root}")
    print(f"🔧 Converter: {converter} ({'✓' if converter.exists() else '✗'})")
    print(f"🔧 Context Gen: {ctx_generator} ({'✓' if ctx_generator.exists() else '✗'})")
    print(f"🔧 HTP Backend: {backend_lib} ({'✓' if backend_lib.exists() else '✗'})")
    print()
    
    if not gguf_path.exists():
        print(f"❌ GGUF model not found: {gguf_path}")
        return 1
    
    print(f"📦 Input: {gguf_path} ({gguf_path.stat().st_size / (1024**3):.2f} GB)")
    
    # Output files
    dlc_path = temp_dir / "deepseek_hybrid.dlc"
    bin_path = temp_dir / "deepseek_hybrid.bin"
    
    # Environment for Windows tools
    env = os.environ.copy()
    env['QNN_SDK_ROOT'] = str(qnn_sdk_root)
    env['PYTHONPATH'] = f"{qnn_sdk_root}/lib/python;{env.get('PYTHONPATH', '')}"
    
    print()
    print("🔄 Step 1: GGUF to DLC Conversion")
    print("-" * 70)
    
    # Convert using Windows tools via Python
    # Note: We'll run this through cmd.exe to use Windows Python
    cmd = [
        "cmd.exe", "/c",
        f"set QNN_SDK_ROOT={qnn_sdk_root} && "
        f"set PYTHONPATH={qnn_sdk_root}\\lib\\python;%PYTHONPATH% && "
        f"python {converter} "
        f'--input_network "{gguf_path}" '
        f'--output_path "{dlc_path}" '
        f'--input_layout "input_ids,NONTRIVIAL" '
        f'--input_layout "attention_mask,NONTRIVIAL" '
        f'--preserve_io "datatype,input_ids,attention_mask" '
        f'--float_fallback '
        f'--float_bitwidth 16 '
        f'--enable_cpu_fallback'
    ]
    
    print("Command structure:")
    print("  Using Windows cmd.exe to run Windows Python with QNN tools")
    print("  This bypasses Linux architecture issues")
    print()
    
    try:
        print("🚀 Running Windows-based conversion...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(script_dir)
        )
        
        if result.returncode == 0:
            print("✅ DLC generation successful!")
            if dlc_path.exists():
                size_mb = dlc_path.stat().st_size / (1024**2)
                print(f"   📄 DLC: {dlc_path.name}")
                print(f"   📊 Size: {size_mb:.1f} MB")
            
        else:
            print(f"❌ DLC generation failed (code: {result.returncode})")
            print("Error output:")
            print(result.stderr[:1000] if result.stderr else "No error output")
            
            # Try alternative approach
            print()
            print("🔄 Alternative: Direct Windows PowerShell approach")
            print("-" * 70)
            
            # Create PowerShell script for Windows execution
            ps_script = temp_dir / "convert_direct.ps1"
            
            ps_content = f'''
# PowerShell conversion script
$env:QNN_SDK_ROOT = "{qnn_sdk_root.as_posix().replace("/mnt/c", "C:")}"
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\\lib\\python;$env:PYTHONPATH"

Write-Host "Converting GGUF to QNN DLC..." -ForegroundColor Green

python "$env:QNN_SDK_ROOT\\bin\\x86_64-windows-msvc\\qairt-converter" `
    --input_network "{gguf_path.as_posix().replace("/mnt/c", "C:")}" `
    --output_path "{dlc_path.as_posix().replace("/mnt/c", "C:")}" `
    --input_layout "input_ids,NONTRIVIAL" `
    --input_layout "attention_mask,NONTRIVIAL" `
    --preserve_io "datatype,input_ids,attention_mask" `
    --float_fallback `
    --float_bitwidth 16 `
    --enable_cpu_fallback

if ($LASTEXITCODE -eq 0) {{
    Write-Host "✅ DLC conversion successful!" -ForegroundColor Green
    
    Write-Host "Generating context binary..." -ForegroundColor Green
    & "$env:QNN_SDK_ROOT\\bin\\x86_64-windows-msvc\\qnn-context-binary-generator.exe" `
        --dlc_path "{dlc_path.as_posix().replace("/mnt/c", "C:")}" `
        --backend "$env:QNN_SDK_ROOT\\lib\\x86_64-windows-msvc\\libQnnHtp.dll" `
        --binary_file "{bin_path.as_posix().replace("/mnt/c", "C:")}" `
        --output_dir "{temp_dir.as_posix().replace("/mnt/c", "C:")}" `
        --target_arch sm8650
        
    if ($LASTEXITCODE -eq 0) {{
        Write-Host "✅ Context binary generated!" -ForegroundColor Green
    }}
}} else {{
    Write-Host "❌ DLC conversion failed" -ForegroundColor Red
}}
'''
            
            with open(ps_script, 'w') as f:
                f.write(ps_content)
                
            print(f"✅ Created PowerShell script: {ps_script}")
            print("To run on Windows:")
            print(f"  powershell -ExecutionPolicy Bypass {ps_script.as_posix().replace('/mnt/c', 'C:')}")
            
            return 1
    
    except subprocess.TimeoutExpired:
        print("⚠️ Conversion timed out after 10 minutes")
        return 1
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 1
    
    # Step 2: Context binary generation (if DLC successful)
    if dlc_path.exists():
        print()
        print("🔄 Step 2: Context Binary Generation")
        print("-" * 70)
        
        cmd_ctx = [
            "cmd.exe", "/c",
            f"set QNN_SDK_ROOT={qnn_sdk_root} && "
            f'"{ctx_generator}" '
            f'--dlc_path "{dlc_path}" '
            f'--backend "{backend_lib}" '
            f'--binary_file "{bin_path}" '
            f'--output_dir "{temp_dir}" '
            f'--target_arch sm8650'
        ]
        
        try:
            result = subprocess.run(cmd_ctx, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Context binary generation successful!")
                if bin_path.exists():
                    size_mb = bin_path.stat().st_size / (1024**2)
                    print(f"   📄 Binary: {bin_path.name}")
                    print(f"   📊 Size: {size_mb:.1f} MB")
            else:
                print(f"⚠️ Context binary generation failed (optional)")
                print("Error:", result.stderr[:500] if result.stderr else "No error")
                
        except Exception as e:
            print(f"⚠️ Context binary error: {e}")
    
    # Summary
    print()
    print("=" * 70)
    print("📋 Hybrid Conversion Summary")
    print("=" * 70)
    
    dlc_success = dlc_path.exists()
    bin_success = bin_path.exists()
    
    if dlc_success:
        dlc_size = dlc_path.stat().st_size / (1024**2)
        print(f"✅ DLC: {dlc_path.name} ({dlc_size:.1f} MB)")
    else:
        print("❌ DLC: Not generated")
    
    if bin_success:
        bin_size = bin_path.stat().st_size / (1024**2)
        print(f"✅ Binary: {bin_path.name} ({bin_size:.1f} MB)")
    else:
        print("⚠️ Binary: Not generated")
    
    print()
    if dlc_success:
        print("🎉 SUCCESS! Model converted using Windows QNN tools via WSL")
        print("   This approach works because:")
        print("   • Target deployment is Windows ARM64 anyway") 
        print("   • Windows tools are fully compatible with the QNN runtime")
        print("   • Avoids Linux architecture compatibility issues")
        return 0
    else:
        print("⚠️ Conversion incomplete - see PowerShell script for Windows execution")
        return 1

if __name__ == "__main__":
    sys.exit(convert_using_windows_tools())