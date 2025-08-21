
# PowerShell conversion script
$env:QNN_SDK_ROOT = "C:/Qualcomm/AIStack/qairt/2.34.0.250424"
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python;$env:PYTHONPATH"

Write-Host "Converting GGUF to QNN DLC..." -ForegroundColor Green

python "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc\qairt-converter" `
    --input_network "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf" `
    --output_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_hybrid.dlc" `
    --input_layout "input_ids,NONTRIVIAL" `
    --input_layout "attention_mask,NONTRIVIAL" `
    --preserve_io "datatype,input_ids,attention_mask" `
    --float_fallback `
    --float_bitwidth 16 `
    --enable_cpu_fallback

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ DLC conversion successful!" -ForegroundColor Green
    
    Write-Host "Generating context binary..." -ForegroundColor Green
    & "$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc\qnn-context-binary-generator.exe" `
        --dlc_path "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_hybrid.dlc" `
        --backend "$env:QNN_SDK_ROOT\lib\x86_64-windows-msvc\libQnnHtp.dll" `
        --binary_file "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp/deepseek_hybrid.bin" `
        --output_dir "/home/zhengte/modelexport_tez47/experiments/tez-172_qnn-compile/temp" `
        --target_arch sm8650
        
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Context binary generated!" -ForegroundColor Green
    }
} else {
    Write-Host "❌ DLC conversion failed" -ForegroundColor Red
}
