#!/usr/bin/env python3
"""
Performance Benchmarking Script

Measures conversion times, memory usage, and file sizes for different model sizes.
"""

import time
import psutil
import os
import subprocess
import re
from pathlib import Path


def benchmark_model_conversion(model_name, description):
    print(f'\n🧪 Benchmarking: {model_name} ({description})')
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    temp_dir = Path(f'temp/benchmark_{model_name.replace("/", "_").replace("-", "_")}')
    temp_dir.mkdir(exist_ok=True)
    
    # Export ONNX with HTP
    print('📦 Exporting ONNX...')
    export_start = time.time()
    
    result = subprocess.run([
        'uv', 'run', 'modelexport', 'export',
        '--model', model_name,
        '--output', str(temp_dir / 'model.onnx'),
        '--strategy', 'htp'
    ], capture_output=True, text=True)
    
    export_time = time.time() - export_start
    
    if result.returncode != 0:
        print(f'❌ Export failed: {result.stderr}')
        return None
    
    # Get ONNX size
    onnx_size = (temp_dir / 'model.onnx').stat().st_size / 1024 / 1024  # MB
    
    # Convert to GraphML
    print('📊 Converting to GraphML...')
    graphml_start = time.time()
    
    result = subprocess.run([
        'uv', 'run', 'modelexport', 'graphml',
        str(temp_dir / 'model.onnx'),
        '--htp-metadata', str(temp_dir / 'model_htp_metadata.json'),
        '-o', str(temp_dir / 'model.graphml')
    ], capture_output=True, text=True)
    
    graphml_time = time.time() - graphml_start
    
    if result.returncode != 0:
        print(f'❌ GraphML conversion failed: {result.stderr}')
        return None
    
    # Get final memory and file sizes
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    graphml_size = (temp_dir / 'model.graphml').stat().st_size / 1024 / 1024  # MB
    
    # Parse GraphML conversion output for metrics
    output_lines = result.stdout.split('\n')
    nodes = edges = compound_nodes = 0
    for line in output_lines:
        if 'Nodes:' in line:
            match = re.search(r'Nodes:\s*(\d+)', line)
            if match:
                nodes = int(match.group(1))
        elif 'Edges:' in line:
            match = re.search(r'Edges:\s*(\d+)', line) 
            if match:
                edges = int(match.group(1))
        elif 'Compound nodes:' in line:
            match = re.search(r'Compound nodes:\s*(\d+)', line)
            if match:
                compound_nodes = int(match.group(1))
    
    # Validate the GraphML
    print('✅ Validating GraphML...')
    validation_start = time.time()
    
    validation_result = subprocess.run([
        'uv', 'run', 'python', 'scripts/validate_graphml.py',
        str(temp_dir / 'model.graphml'),
        '--onnx-file', str(temp_dir / 'model.onnx'),
        '--quiet'
    ], capture_output=True, text=True)
    
    validation_time = time.time() - validation_start
    validation_passed = validation_result.returncode == 0
    
    metrics = {
        'model': model_name,
        'description': description,
        'export_time': export_time,
        'graphml_time': graphml_time,
        'validation_time': validation_time,
        'total_time': export_time + graphml_time,
        'memory_used_mb': memory_used,
        'onnx_size_mb': onnx_size,
        'graphml_size_mb': graphml_size,
        'nodes': nodes,
        'edges': edges,
        'compound_nodes': compound_nodes,
        'validation_passed': validation_passed
    }
    
    print(f'📊 Results:')
    print(f'   Export time: {export_time:.2f}s')
    print(f'   GraphML time: {graphml_time:.2f}s')
    print(f'   Validation time: {validation_time:.3f}s')
    print(f'   Total time: {metrics["total_time"]:.2f}s')
    print(f'   Memory used: {memory_used:.1f} MB')
    print(f'   ONNX size: {onnx_size:.1f} MB')
    print(f'   GraphML size: {graphml_size:.1f} MB')
    print(f'   Nodes: {nodes}, Edges: {edges}, Compound: {compound_nodes}')
    print(f'   Validation: {"✅ PASS" if validation_passed else "❌ FAIL"}')
    
    return metrics


def main():
    print('🚀 Starting Performance Benchmarking')
    
    # Test different model sizes
    models = [
        ('prajjwal1/bert-tiny', 'Small BERT (4M params)'),
    ]
    
    results = []
    for model_name, description in models:
        try:
            result = benchmark_model_conversion(model_name, description)
            if result:
                results.append(result)
        except Exception as e:
            print(f'❌ Benchmark failed for {model_name}: {e}')
    
    if results:
        print('\n📈 Performance Summary:')
        print('Model                    | Export | GraphML | Total  | Memory | Nodes | Validation')
        print('-------------------------|--------|---------|--------|--------|-------|----------')
        for r in results:
            model_short = r['model'].split('/')[-1][:20]
            print(f'{model_short:<24} | {r["export_time"]:5.1f}s | {r["graphml_time"]:6.2f}s | {r["total_time"]:5.1f}s | {r["memory_used_mb"]:5.1f}M | {r["nodes"]:5d} | {"✅" if r["validation_passed"] else "❌"}')
        
        # Performance classification
        print('\n🎯 Performance Classification:')
        for r in results:
            total_time = r['total_time']
            if total_time < 1:
                category = "🚀 Excellent (<1s)"
            elif total_time < 5:
                category = "✅ Good (<5s)"
            elif total_time < 30:
                category = "⚠️ Acceptable (<30s)"
            else:
                category = "❌ Slow (>30s)"
            
            print(f'   {r["model"]}: {category}')


if __name__ == '__main__':
    main()