#!/usr/bin/env python3
"""
Iteration 8: Performance benchmarking and optimization for FX exporter.

This script provides comprehensive performance testing to identify bottlenecks
and optimization opportunities in the FX-based hierarchy preservation approach.
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import time
import statistics
from typing import Dict, List, Any

# Add modelexport to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.fx_hierarchy_exporter import FXHierarchyExporter

def benchmark_export_performance():
    """Benchmark export performance across different model sizes and types."""
    print("=" * 60)
    print("BENCHMARK: Export Performance Analysis")
    print("=" * 60)
    
    benchmark_models = [
        {
            'name': 'Small_CNN',
            'model_fn': lambda: create_cnn_model(channels=[3, 16, 32], image_size=32),
            'inputs_fn': lambda: torch.randn(1, 3, 32, 32),
            'category': 'vision'
        },
        {
            'name': 'Medium_CNN',
            'model_fn': lambda: create_cnn_model(channels=[3, 32, 64, 128], image_size=64),
            'inputs_fn': lambda: torch.randn(1, 3, 64, 64),
            'category': 'vision'
        },
        {
            'name': 'Small_MLP',
            'model_fn': lambda: create_mlp_model([784, 256, 128, 10]),
            'inputs_fn': lambda: torch.randn(1, 784),
            'category': 'feedforward'
        },
        {
            'name': 'Large_MLP',
            'model_fn': lambda: create_mlp_model([2048, 1024, 512, 256, 128, 64, 10]),
            'inputs_fn': lambda: torch.randn(1, 2048),
            'category': 'feedforward'
        },
        {
            'name': 'Simple_Attention',
            'model_fn': lambda: create_attention_model(embed_dim=128, num_heads=8, seq_len=50),
            'inputs_fn': lambda: torch.randn(1, 50, 128),
            'category': 'attention'
        },
        {
            'name': 'Complex_Attention',
            'model_fn': lambda: create_attention_model(embed_dim=512, num_heads=16, seq_len=200),
            'inputs_fn': lambda: torch.randn(1, 200, 512),
            'category': 'attention'
        }
    ]
    
    results = {}
    
    for model_spec in benchmark_models:
        print(f"\nBenchmarking {model_spec['name']}...")
        
        try:
            model = model_spec['model_fn']()
            inputs = model_spec['inputs_fn']()
            
            # Get model statistics
            param_count = sum(p.numel() for p in model.parameters())
            module_count = len(list(model.named_modules()))
            
            # Run multiple iterations for statistical significance
            times = []
            hierarchy_nodes = []
            
            for i in range(3):  # 3 iterations
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                    exporter = FXHierarchyExporter(auto_fallback=False)  # Pure FX for benchmarking
                    
                    start_time = time.time()
                    result = exporter.export(model, inputs, tmp.name)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    hierarchy_nodes.append(result['hierarchy_nodes'])
                    
                    # Cleanup
                    for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:
                        if cleanup_file and os.path.exists(cleanup_file):
                            os.unlink(cleanup_file)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            avg_nodes = statistics.mean(hierarchy_nodes)
            
            results[model_spec['name']] = {
                'success': True,
                'category': model_spec['category'],
                'param_count': param_count,
                'module_count': module_count,
                'avg_export_time': avg_time,
                'std_export_time': std_time,
                'avg_hierarchy_nodes': avg_nodes,
                'time_per_param': avg_time / param_count * 1000000,  # microseconds per parameter
                'time_per_module': avg_time / module_count * 1000,   # milliseconds per module
                'iterations': len(times)
            }
            
            print(f"   ✅ {avg_time:.3f}s ± {std_time:.3f}s ({avg_nodes:.0f} nodes)")
            print(f"      {param_count:,} params, {module_count} modules")
            print(f"      {results[model_spec['name']]['time_per_param']:.2f} μs/param")
            
        except Exception as e:
            results[model_spec['name']] = {
                'success': False,
                'category': model_spec['category'],
                'error': str(e)
            }
            print(f"   ❌ Failed: {e}")
    
    return results

def benchmark_phase_breakdown():\n    \"\"\"Analyze performance breakdown by export phases.\"\"\"\n    print(\"=\" * 60)\n    print(\"BENCHMARK: Phase-by-Phase Performance Breakdown\")\n    print(\"=\" * 60)\n    \n    # Create a representative model\n    model = create_cnn_model(channels=[3, 32, 64], image_size=64)\n    inputs = torch.randn(1, 3, 64, 64)\n    \n    print(\"Analyzing performance phases...\")\n    \n    # Custom exporter with timing hooks\n    class TimingFXExporter(FXHierarchyExporter):\n        def __init__(self):\n            super().__init__(auto_fallback=False)\n            self.phase_times = {}\n        \n        def export(self, model, example_inputs, output_path, **kwargs):\n            total_start = time.time()\n            \n            # Phase 0: Architecture analysis\n            phase_start = time.time()\n            self._model_root = model\n            model.eval()\n            self.phase_times['setup'] = time.time() - phase_start\n            \n            # Phase 1: FX analysis\n            phase_start = time.time()\n            fx_result = self._analyze_fx_hierarchy(model, example_inputs)\n            self._fx_result = fx_result\n            self.phase_times['fx_analysis'] = time.time() - phase_start\n            \n            # Phase 2: ONNX export\n            phase_start = time.time()\n            torch.onnx.export(model, example_inputs, output_path, **kwargs)\n            self.phase_times['onnx_export'] = time.time() - phase_start\n            \n            # Phase 3: Hierarchy injection\n            phase_start = time.time()\n            enhanced_onnx_path = self._inject_hierarchy_metadata(output_path, fx_result)\n            self.phase_times['hierarchy_injection'] = time.time() - phase_start\n            \n            # Phase 4: Analysis files\n            phase_start = time.time()\n            analysis_results = self._generate_analysis_files(output_path, fx_result)\n            self.phase_times['analysis_generation'] = time.time() - phase_start\n            \n            self.phase_times['total'] = time.time() - total_start\n            \n            return {\n                'onnx_path': enhanced_onnx_path,\n                'sidecar_path': analysis_results['sidecar_path'],\n                'module_info_path': analysis_results['module_info_path'],\n                'hierarchy_nodes': len(fx_result.node_hierarchy),\n                'export_time': self.phase_times['total'],\n                'strategy': 'fx_graph'\n            }\n    \n    try:\n        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:\n            timing_exporter = TimingFXExporter()\n            result = timing_exporter.export(model, inputs, tmp.name)\n            \n            phase_times = timing_exporter.phase_times\n            total_time = phase_times['total']\n            \n            print(f\"\\n📊 Phase Performance Breakdown:\")\n            print(f\"   Setup:               {phase_times['setup']:.3f}s ({phase_times['setup']/total_time*100:.1f}%)\")\n            print(f\"   FX Analysis:         {phase_times['fx_analysis']:.3f}s ({phase_times['fx_analysis']/total_time*100:.1f}%)\")\n            print(f\"   ONNX Export:         {phase_times['onnx_export']:.3f}s ({phase_times['onnx_export']/total_time*100:.1f}%)\")\n            print(f\"   Hierarchy Injection: {phase_times['hierarchy_injection']:.3f}s ({phase_times['hierarchy_injection']/total_time*100:.1f}%)\")\n            print(f\"   Analysis Generation: {phase_times['analysis_generation']:.3f}s ({phase_times['analysis_generation']/total_time*100:.1f}%)\")\n            print(f\"   Total:               {total_time:.3f}s\")\n            \n            # Identify bottlenecks\n            max_phase = max(phase_times.items(), key=lambda x: x[1] if x[0] != 'total' else 0)\n            print(f\"\\n🎯 Primary bottleneck: {max_phase[0]} ({max_phase[1]:.3f}s)\")\n            \n            # Cleanup\n            for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:\n                if cleanup_file and os.path.exists(cleanup_file):\n                    os.unlink(cleanup_file)\n            \n            return phase_times\n    \n    except Exception as e:\n        print(f\"❌ Phase breakdown failed: {e}\")\n        return {}\n\ndef benchmark_vs_htp_comparison():\n    \"\"\"Compare FX vs HTP performance on compatible models.\"\"\"\n    print(\"=\" * 60)\n    print(\"BENCHMARK: FX vs HTP Performance Comparison\")\n    print(\"=\" * 60)\n    \n    # Test models that both approaches can handle\n    test_models = [\n        {\n            'name': 'Medium_CNN',\n            'model_fn': lambda: create_cnn_model(channels=[3, 32, 64], image_size=64),\n            'inputs_fn': lambda: torch.randn(1, 3, 64, 64)\n        },\n        {\n            'name': 'Large_MLP',\n            'model_fn': lambda: create_mlp_model([1024, 512, 256, 128, 10]),\n            'inputs_fn': lambda: torch.randn(1, 1024)\n        }\n    ]\n    \n    comparison_results = {}\n    \n    for model_spec in test_models:\n        print(f\"\\nComparing {model_spec['name']}...\")\n        \n        model = model_spec['model_fn']()\n        inputs = model_spec['inputs_fn']()\n        \n        # Test FX approach\n        fx_times = []\n        fx_nodes = 0\n        \n        print(\"   Testing FX approach...\")\n        for i in range(3):\n            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:\n                try:\n                    exporter = FXHierarchyExporter(auto_fallback=False)\n                    start_time = time.time()\n                    result = exporter.export(model, inputs, tmp.name)\n                    fx_times.append(time.time() - start_time)\n                    fx_nodes = result['hierarchy_nodes']\n                    \n                    # Cleanup\n                    for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:\n                        if cleanup_file and os.path.exists(cleanup_file):\n                            os.unlink(cleanup_file)\n                            \n                except Exception as e:\n                    print(f\"      FX failed: {e}\")\n                    fx_times = []\n                    break\n        \n        # Test HTP approach\n        htp_times = []\n        htp_nodes = 0\n        \n        print(\"   Testing HTP approach...\")\n        for i in range(3):\n            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:\n                try:\n                    from modelexport.hierarchy_exporter import HierarchyExporter\n                    exporter = HierarchyExporter(strategy='htp')\n                    start_time = time.time()\n                    result = exporter.export(model, inputs, tmp.name)\n                    htp_times.append(time.time() - start_time)\n                    htp_nodes = result.get('tagged_operations', 0)\n                    \n                    # Cleanup\n                    cleanup_file = tmp.name.replace('.onnx', '_hierarchy.json')\n                    if os.path.exists(cleanup_file):\n                        os.unlink(cleanup_file)\n                    if os.path.exists(tmp.name):\n                        os.unlink(tmp.name)\n                        \n                except Exception as e:\n                    print(f\"      HTP failed: {e}\")\n                    htp_times = []\n                    break\n        \n        # Compare results\n        if fx_times and htp_times:\n            fx_avg = statistics.mean(fx_times)\n            htp_avg = statistics.mean(htp_times)\n            speedup = htp_avg / fx_avg\n            \n            comparison_results[model_spec['name']] = {\n                'fx_time': fx_avg,\n                'htp_time': htp_avg,\n                'fx_nodes': fx_nodes,\n                'htp_nodes': htp_nodes,\n                'speedup': speedup\n            }\n            \n            print(f\"   📊 Results:\")\n            print(f\"      FX:  {fx_avg:.3f}s ({fx_nodes} nodes)\")\n            print(f\"      HTP: {htp_avg:.3f}s ({htp_nodes} nodes)\")\n            print(f\"      Speedup: {speedup:.2f}x {'(FX faster)' if speedup > 1 else '(HTP faster)'}\")\n        else:\n            print(f\"   ❌ Comparison failed\")\n    \n    return comparison_results\n\n# Helper functions to create test models\ndef create_cnn_model(channels: List[int], image_size: int) -> nn.Module:\n    \"\"\"Create a CNN model with specified channels and image size.\"\"\"\n    layers = []\n    \n    for i in range(len(channels) - 1):\n        layers.extend([\n            nn.Conv2d(channels[i], channels[i+1], 3, padding=1),\n            nn.ReLU(),\n            nn.MaxPool2d(2)\n        ])\n    \n    # Calculate final feature size\n    final_size = image_size // (2 ** (len(channels) - 1))\n    final_features = channels[-1] * final_size * final_size\n    \n    layers.extend([\n        nn.AdaptiveAvgPool2d((1, 1)),\n        nn.Flatten(),\n        nn.Linear(channels[-1], 10)\n    ])\n    \n    return nn.Sequential(*layers)\n\ndef create_mlp_model(layer_sizes: List[int]) -> nn.Module:\n    \"\"\"Create an MLP model with specified layer sizes.\"\"\"\n    layers = []\n    \n    for i in range(len(layer_sizes) - 1):\n        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))\n        if i < len(layer_sizes) - 2:  # No activation after last layer\n            layers.extend([nn.ReLU(), nn.Dropout(0.1)])\n    \n    return nn.Sequential(*layers)\n\ndef create_attention_model(embed_dim: int, num_heads: int, seq_len: int) -> nn.Module:\n    \"\"\"Create an attention-based model.\"\"\"\n    class AttentionModel(nn.Module):\n        def __init__(self, embed_dim, num_heads):\n            super().__init__()\n            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)\n            self.norm1 = nn.LayerNorm(embed_dim)\n            self.norm2 = nn.LayerNorm(embed_dim)\n            self.ff = nn.Sequential(\n                nn.Linear(embed_dim, embed_dim * 4),\n                nn.ReLU(),\n                nn.Linear(embed_dim * 4, embed_dim)\n            )\n            self.classifier = nn.Linear(embed_dim, 10)\n        \n        def forward(self, x):\n            # Self-attention\n            attn_out, _ = self.attention(x, x, x)\n            x = self.norm1(x + attn_out)\n            \n            # Feed-forward\n            ff_out = self.ff(x)\n            x = self.norm2(x + ff_out)\n            \n            # Classification\n            x = x.mean(dim=1)  # Global average pooling\n            return self.classifier(x)\n    \n    return AttentionModel(embed_dim, num_heads)\n\ndef main():\n    \"\"\"Run comprehensive performance benchmarks.\"\"\"\n    print(\"🚀 FX Exporter Performance Benchmarking Suite\")\n    print(f\"PyTorch version: {torch.__version__)\")\n    \n    all_results = {}\n    \n    # Benchmark 1: Export performance\n    print(f\"\\n📋 Running Export Performance Benchmarks\")\n    try:\n        export_results = benchmark_export_performance()\n        all_results['export_performance'] = export_results\n    except Exception as e:\n        print(f\"❌ Export benchmarks crashed: {e}\")\n        all_results['export_performance'] = {'error': str(e)}\n    \n    # Benchmark 2: Phase breakdown\n    print(f\"\\n📋 Running Phase Breakdown Analysis\")\n    try:\n        phase_results = benchmark_phase_breakdown()\n        all_results['phase_breakdown'] = phase_results\n    except Exception as e:\n        print(f\"❌ Phase breakdown crashed: {e}\")\n        all_results['phase_breakdown'] = {'error': str(e)}\n    \n    # Benchmark 3: FX vs HTP comparison\n    print(f\"\\n📋 Running FX vs HTP Comparison\")\n    try:\n        comparison_results = benchmark_vs_htp_comparison()\n        all_results['fx_vs_htp'] = comparison_results\n    except Exception as e:\n        print(f\"❌ FX vs HTP comparison crashed: {e}\")\n        all_results['fx_vs_htp'] = {'error': str(e)}\n    \n    # Performance analysis and recommendations\n    print(\"\\n\" + \"=\" * 60)\n    print(\"PERFORMANCE ANALYSIS & OPTIMIZATION RECOMMENDATIONS\")\n    print(\"=\" * 60)\n    \n    # Export performance analysis\n    if 'export_performance' in all_results and 'error' not in all_results['export_performance']:\n        export_data = all_results['export_performance']\n        successful_models = {k: v for k, v in export_data.items() if v.get('success', False)}\n        \n        if successful_models:\n            # Performance by category\n            categories = {}\n            for model_name, data in successful_models.items():\n                category = data['category']\n                if category not in categories:\n                    categories[category] = []\n                categories[category].append(data['avg_export_time'])\n            \n            print(\"⚡ Performance by Architecture:\")\n            for category, times in categories.items():\n                avg_time = statistics.mean(times)\n                print(f\"   {category.capitalize()}: {avg_time:.3f}s average\")\n            \n            # Identify fastest and slowest\n            fastest = min(successful_models.items(), key=lambda x: x[1]['avg_export_time'])\n            slowest = max(successful_models.items(), key=lambda x: x[1]['avg_export_time'])\n            \n            print(f\"\\n🏃 Fastest: {fastest[0]} ({fastest[1]['avg_export_time']:.3f}s)\")\n            print(f\"🐌 Slowest: {slowest[0]} ({slowest[1]['avg_export_time']:.3f}s)\")\n    \n    # Phase breakdown insights\n    if 'phase_breakdown' in all_results and 'error' not in all_results['phase_breakdown']:\n        phase_data = all_results['phase_breakdown']\n        if phase_data:\n            bottleneck_phases = sorted([(k, v) for k, v in phase_data.items() if k != 'total'], \n                                     key=lambda x: x[1], reverse=True)[:2]\n            \n            print(f\"\\n🎯 Top Optimization Targets:\")\n            for phase, time_spent in bottleneck_phases:\n                print(f\"   {phase}: {time_spent:.3f}s ({time_spent/phase_data['total']*100:.1f}% of total)\")\n    \n    # FX vs HTP comparison insights\n    if 'fx_vs_htp' in all_results and 'error' not in all_results['fx_vs_htp']:\n        comp_data = all_results['fx_vs_htp']\n        if comp_data:\n            avg_speedup = statistics.mean([data['speedup'] for data in comp_data.values()])\n            print(f\"\\n⚖️  FX vs HTP Performance:\")\n            print(f\"   Average speedup: {avg_speedup:.2f}x {'(FX faster)' if avg_speedup > 1 else '(HTP faster)'}\")\n    \n    # Overall recommendations\n    print(f\"\\n💡 Optimization Recommendations:\")\n    print(f\"   1. Focus optimization on bottleneck phases identified above\")\n    print(f\"   2. Cache FX graphs for repeated exports of same models\")\n    print(f\"   3. Implement parallel processing for analysis file generation\")\n    print(f\"   4. Consider streaming ONNX processing for large models\")\n    \n    print(f\"\\n🎉 Iteration 8 (Performance Benchmarking) completed!\")\n    print(f\"   ➡️  Comprehensive performance analysis available for optimization\")\n    \n    return 0\n\nif __name__ == '__main__':\n    exit(main())