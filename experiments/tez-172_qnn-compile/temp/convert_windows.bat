@echo off
REM Windows batch script for actual GGUF to QNN conversion
REM This must be run on Windows with QNN SDK installed

echo ======================================================================
echo Converting DeepSeek GGUF to QNN Binary
echo ======================================================================

set QNN_SDK_ROOT=C:\Qualcomm\AIStack\qairt\2.34.0.250424
set PYTHONPATH=%QNN_SDK_ROOT%\lib\python;%PYTHONPATH%
set PATH=%QNN_SDK_ROOT%\bin\x86_64-windows-msvc;%PATH%

set GGUF_MODEL=%~dp0\..\models\DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf
set OUTPUT_DLC=%~dp0\deepseek_qwen.dlc
set OUTPUT_BIN=%~dp0\deepseek_qwen.bin

echo.
echo Step 1: Converting GGUF to DLC
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
        --output_dir "%~dp0" ^
        --target_arch sm8650
    
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] Context binary generated: %OUTPUT_BIN%
    ) else (
        echo [WARNING] Context binary generation failed
    )
) else (
    echo [ERROR] DLC generation failed
    exit /b 1
)

echo.
echo ======================================================================
echo Conversion Complete!
echo ======================================================================
echo DLC: %OUTPUT_DLC%
echo Binary: %OUTPUT_BIN%
echo.
pause