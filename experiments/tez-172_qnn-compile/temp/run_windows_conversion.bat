@echo off
REM Direct Windows conversion script - run this in Windows Command Prompt
REM Auto-generated from WSL analysis

echo ======================================================================
echo 🚀 DeepSeek GGUF to QNN Conversion (Windows Native)
echo ======================================================================

set QNN_SDK_ROOT=C:\Qualcomm\AIStack\qairt\2.34.0.250424
set PYTHONPATH=%QNN_SDK_ROOT%\lib\python;%PYTHONPATH%
set PATH=%QNN_SDK_ROOT%\bin\x86_64-windows-msvc;%PATH%

REM Convert WSL paths to Windows paths
set GGUF_MODEL=\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf
set OUTPUT_DIR=\\wsl.localhost\Ubuntu\home\zhengte\modelexport_tez47\experiments\tez-172_qnn-compile\temp
set OUTPUT_DLC=%OUTPUT_DIR%\deepseek_qwen_real.dlc
set OUTPUT_BIN=%OUTPUT_DIR%\deepseek_qwen_real.bin

echo Input: %GGUF_MODEL%
echo Output: %OUTPUT_DLC%
echo.

echo Step 1: GGUF to DLC Conversion
echo ----------------------------------------------------------------------
python %QNN_SDK_ROOT%\bin\x86_64-windows-msvc\qairt-converter ^
    --input_network "%GGUF_MODEL%" ^
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
    %QNN_SDK_ROOT%\bin\x86_64-windows-msvc\qnn-context-binary-generator.exe ^
        --dlc_path "%OUTPUT_DLC%" ^
        --backend "%QNN_SDK_ROOT%\lib\x86_64-windows-msvc\libQnnHtp.dll" ^
        --binary_file "%OUTPUT_BIN%" ^
        --output_dir "%OUTPUT_DIR%" ^
        --target_arch sm8650
    
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] Context binary generated: %OUTPUT_BIN%
    ) else (
        echo [WARNING] Context binary generation failed
    )
) else (
    echo [ERROR] DLC generation failed
)

echo.
echo ======================================================================
echo Conversion Complete!
echo ======================================================================
echo Check the temp folder in WSL for output files
pause