# PowerShell script for QNN conversion
$env:QNN_SDK_ROOT = "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424"
$env:PYTHONPATH = "${env:QNN_SDK_ROOT}\lib\python;${env:PYTHONPATH}"
$env:PATH = "${env:QNN_SDK_ROOT}\bin\x86_64-windows-msvc;${env:PATH}"

Write-Host "Converting GGUF to QNN DLC..." -ForegroundColor Green

python "/mnt/c/Qualcomm/AIStack/qairt/2.34.0.250424/bin/x86_64-windows-msvc/qairt-converter" `
    --input_network "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf" `
    --output_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_qwen_real.dlc" `
    --float_fallback `
    --float_bitwidth 16 `
    --enable_cpu_fallback

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Conversion successful!" -ForegroundColor Green
} else {
    Write-Host "❌ Conversion failed" -ForegroundColor Red
}
