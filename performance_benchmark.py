#!/usr/bin/env python3
"""Performance benchmark for GraphML generation."""

import os
import time
from pathlib import Path

import psutil

from modelexport.strategies.htp.htp_exporter import HTPExporter


def benchmark_model_export(model_name: str, description: str) -> dict:
    """Benchmark a model export with performance metrics."""
    
    output_dir = f"temp/benchmark_{model_name.replace('/', '_')}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Memory before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\n🔍 Benchmarking: {description}")
    print(f"Model: {model_name}")
    print(f"Memory before: {memory_before:.1f} MB")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Export with HTP
        exporter = HTPExporter(
            verbose=False,
            enable_reporting=False,
            embed_hierarchy_attributes=True,
            torch_module=["Linear", "LayerNorm", "Embedding", "Dropout", "Tanh"]
        )
        
        stats = exporter.export(
            model_name_or_path=model_name,
            output_path=f"{output_dir}model.onnx"
        )
        
        # The metadata file is automatically created with _htp_metadata.json suffix
        metadata_path = f"{output_dir}model_htp_metadata.json"
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Generate GraphML
        graphml_start = time.time()
        from modelexport.graphml.hierarchical_converter import (
            EnhancedHierarchicalConverter,
        )
        
        converter = EnhancedHierarchicalConverter(htp_metadata_path=metadata_path)
        graphml_content = converter.convert(f"{output_dir}model.onnx")
        
        with open(f"{output_dir}model.graphml", 'w') as f:
            f.write(graphml_content)
        
        graphml_time = time.time() - graphml_start
        
        # Count compound nodes
        import xml.etree.ElementTree as ET
        ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
        
        root = ET.fromstring(graphml_content)
        all_nodes = root.findall(".//g:node", ns)
        compound_count = sum(1 for node in all_nodes if node.findall("./g:graph", ns))
        
        results = {
            'model': model_name,
            'description': description,
            'success': True,
            'total_time': total_time,
            'graphml_time': graphml_time,
            'memory_used': memory_used,
            'hierarchy_modules': stats['hierarchy_modules'],
            'onnx_nodes': stats['onnx_nodes'],
            'tagged_nodes': stats['tagged_nodes'],
            'coverage': stats['coverage_percentage'],
            'compound_nodes': compound_count
        }
        
        print(f"✅ Success! Time: {total_time:.2f}s, GraphML: {graphml_time:.3f}s")
        print(f"   Memory used: {memory_used:.1f} MB")
        print(f"   Hierarchy modules: {stats['hierarchy_modules']}")
        print(f"   Compound nodes: {compound_count}")
        print(f"   Coverage: {stats['coverage_percentage']:.1f}%")
        
        return results
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Failed: {str(e)} (Time: {end_time - start_time:.2f}s)")
        
        return {
            'model': model_name,
            'description': description,
            'success': False,
            'error': str(e),
            'time': end_time - start_time
        }

def main():
    """Run performance benchmarks."""
    
    print("🚀 GraphML Performance Benchmark Suite")
    print("=" * 50)
    
    models = [
        ("prajjwal1/bert-tiny", "BERT-tiny (4.4M params, 2 layers)"),
        ("distilbert-base-uncased", "DistilBERT (66M params, 6 layers)"),
    ]
    
    results = []
    
    for model_name, description in models:
        try:
            result = benchmark_model_export(model_name, description)
            results.append(result)
        except KeyboardInterrupt:
            print("\n⏹️ Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"❌ Benchmark failed for {model_name}: {e}")
            results.append({
                'model': model_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        print(f"Successful benchmarks: {len(successful)}/{len(results)}")
        print("\nPerformance Metrics:")
        for result in successful:
            print(f"\n{result['description']}:")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  GraphML generation: {result['graphml_time']:.3f}s")
            print(f"  Memory usage: {result['memory_used']:.1f} MB")
            print(f"  Compound nodes: {result['compound_nodes']}")
            print(f"  ONNX coverage: {result['coverage']:.1f}%")
    
    failed = [r for r in results if not r.get('success', False)]
    if failed:
        print(f"\nFailed benchmarks: {len(failed)}")
        for result in failed:
            print(f"  {result['model']}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()