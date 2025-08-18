# Running QNN Profiling on Windows ARM64 with Qualcomm NPU

This guide helps you run actual QNN profiling on your Windows ARM64 device with Qualcomm NPU.

## Prerequisites

1. **Windows ARM64 device** with Qualcomm Snapdragon processor (8cx, 8cx Gen2, 8cx Gen3, X Elite, etc.)
2. **QNN SDK installed** (Qualcomm AI Stack)
3. **Python 3.8+** installed
4. **Visual Studio 2019/2022** C++ runtime (usually pre-installed)

## Quick Start

### Step 1: Test Your Setup

Run this first to check if everything is configured correctly:

```cmd
python quick_profile_test.py
```

This will:
- Find your QNN SDK installation
- Verify required binaries are present
- Check Python SDK availability
- Save configuration for later use

### Step 2: Run Profiling

#### Option A: With Batch Script (Easiest)
```cmd
run_profiling_windows.bat
```

#### Option B: Direct Python
```cmd
python run_real_profiling_windows.py
```

### Step 3: View Results

1. Look for output in `profiling_output_real/` or `profiling_output_python/`
2. Open Chrome and navigate to `chrome://tracing`
3. Click "Load" and select `chrome_trace.json` from the output directory

## File Descriptions

- `quick_profile_test.py` - Tests your QNN setup and saves configuration
- `run_real_profiling_windows.py` - Main profiling script with actual NPU execution
- `run_profiling_windows.bat` - Windows batch script for easy execution
- `qnn_profiling_poc.py` - Original POC (simulation only, no real hardware needed)

## WSL vs Native Windows

### Native Windows (Recommended)
✅ **Pros:**
- Direct NPU/HTP hardware access
- Best performance
- Full SDK feature support
- Native profiling tools

❌ **Cons:**
- Must use Windows Python
- Windows-specific paths

### WSL (Limited Support)
✅ **Pros:**
- Linux development environment
- Can use Linux QNN SDK

❌ **Cons:**
- No direct NPU access (must use network mode)
- Performance overhead
- Complex setup required

## Setting Up QNN SDK

### 1. Download QNN SDK

Download from [Qualcomm Developer Network](https://developer.qualcomm.com/software/qualcomm-ai-stack)

### 2. Install Location

Typical installation paths:
- `C:\Qualcomm\AIStack\qairt\2.34.0.250424\`
- `C:\Program Files\Qualcomm\AIStack\qairt\`

### 3. Set Environment Variable

```cmd
setx QNN_SDK_ROOT "C:\Qualcomm\AIStack\qairt\2.34.0.250424"
```

### 4. Verify Installation

```cmd
echo %QNN_SDK_ROOT%
dir "%QNN_SDK_ROOT%\bin\aarch64-windows-msvc\"
```

## Using Your Own Model

### Option 1: ONNX Model

If you have an ONNX model:

```python
# The script will automatically convert ONNX to DLC
model_path = Path("your_model.onnx")
```

### Option 2: Pre-compiled DLC

If you already have a DLC file:

```python
model_path = Path("your_model.dlc")
```

### Option 3: Convert Manually

```cmd
"%QNN_SDK_ROOT%\bin\aarch64-windows-msvc\qnn-onnx-converter.exe" ^
    --input_network model.onnx ^
    --output_path model.dlc
```

## Profiling Levels

- **basic** - High-level timing metrics
- **detailed** - Node-level execution times (recommended)
- **backend** - Backend-specific metrics
- **client** - Client-side metrics
- **linting** - Deep analysis (native SDK only)

## Performance Profiles

- **low_balanced** - Power efficiency
- **balanced** - Default balance
- **high_performance** - Maximum performance (recommended for profiling)
- **sustained_high_performance** - Sustained workloads
- **extreme_performance** - Burst performance

## Example Output

After successful profiling, you'll see:

```
=== PROFILING RESULTS ===
inference_time_ms: 12.45
htp_execution_time_us: 11234.00
vtcm_acquisition_time_us: 145.00
hvx_utilization_percent: 72.50
hmx_utilization_percent: 65.30
chrome_trace: profiling_output_real\chrome_trace.json

📊 View detailed timeline:
1. Open Chrome browser
2. Navigate to: chrome://tracing
3. Click 'Load' and select: profiling_output_real\chrome_trace.json
```

## Troubleshooting

### "QNN SDK not found"
- Install QNN SDK from Qualcomm Developer Network
- Set `QNN_SDK_ROOT` environment variable
- Restart command prompt after setting environment variable

### "HTP backend not found"
- Your device might not support HTP
- Try CPU backend as fallback
- Check SDK installation is complete

### "Access denied" errors
- Run command prompt as Administrator
- Check file permissions
- Disable antivirus temporarily for testing

### "Module qairt not found"
- Add QNN Python path: `set PYTHONPATH=%QNN_SDK_ROOT%\lib\python;%PYTHONPATH%`
- Install required packages: `pip install numpy`

## Support

For QNN-specific issues:
- [Qualcomm Developer Forums](https://developer.qualcomm.com/forums)
- [QNN Documentation](https://developer.qualcomm.com/software/qualcomm-ai-stack/documentation)

For modelexport integration:
- Create an issue in the modelexport repository
- Reference Linear task TEZ-159