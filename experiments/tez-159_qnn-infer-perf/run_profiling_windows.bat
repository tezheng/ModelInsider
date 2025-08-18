@echo off
REM Batch script to run QNN profiling on Windows ARM64

echo ========================================
echo QNN Profiling Setup for Windows ARM64
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check/Set QNN SDK Root
if "%QNN_SDK_ROOT%"=="" (
    echo QNN_SDK_ROOT not set. Searching common locations...
    
    if exist "C:\Qualcomm\AIStack\qairt\2.34.0.250424" (
        set QNN_SDK_ROOT=C:\Qualcomm\AIStack\qairt\2.34.0.250424
        echo Found QNN SDK at: C:\Qualcomm\AIStack\qairt\2.34.0.250424
    ) else if exist "C:\Qualcomm\AIStack\QAIRT\2.34.0.250424" (
        set QNN_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.34.0.250424
        echo Found QNN SDK at: C:\Qualcomm\AIStack\QAIRT\2.34.0.250424
    ) else (
        echo ERROR: QNN SDK not found. Please install and set QNN_SDK_ROOT
        echo Download from: https://developer.qualcomm.com/software/qualcomm-ai-stack
        pause
        exit /b 1
    )
)

echo Using QNN SDK: %QNN_SDK_ROOT%
echo.

REM Add QNN Python packages to PYTHONPATH
set PYTHONPATH=%QNN_SDK_ROOT%\lib\python;%PYTHONPATH%

REM Install required Python packages
echo Installing required Python packages...
pip install numpy >nul 2>&1

REM Optional: Install PyTorch and ONNX for model creation
echo.
echo Optional: Install PyTorch and ONNX for test model creation?
echo (You can skip this if you have your own model)
choice /c YN /m "Install PyTorch and ONNX"
if %errorlevel%==1 (
    echo Installing PyTorch and ONNX...
    pip install torch torchvision onnx
)

echo.
echo ========================================
echo Running QNN Profiling
echo ========================================
echo.

REM Run the profiling script
python run_real_profiling_windows.py

echo.
echo ========================================
echo Profiling Complete
echo ========================================
echo.

REM Keep window open
pause