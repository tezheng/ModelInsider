#!/usr/bin/env python3
"""
HTP Optimization Benchmarking Script for Iteration 17

Tests the optimized HTP methods against baseline to measure improvements.
"""

import time
import torch
from pathlib import Path
from transformers import AutoModel
import json

from modelexport.strategies.htp import HTPHierarchyExporter
from modelexport.strategies.htp.optimizations import apply_htp_optimizations, HuggingFaceSpecificOptimizations


class HTPOptimizationBenchmark:
    """Benchmark HTP optimizations against baseline performance."""
    
    def __init__(self, output_dir: str = "temp/iteration_17"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def benchmark_model(self, model_name: str, num_runs: int = 3):
        """Benchmark a model with both original and optimized HTP."""
        
        print(f"\\n🏁 Benchmarking {model_name} (${num_runs} runs each)...")
        
        # Load model once
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Prepare inputs
        if 'resnet' in model_name:
            inputs = torch.randn(1, 3, 224, 224)
        else:
            inputs = torch.randn(1, 3, 1024, 1024)
        
        # Test original HTP
        original_times = []
        for run in range(num_runs):
            print(f"  📊 Original HTP - Run {run + 1}/{num_runs}")
            
            exporter = HTPHierarchyExporter()
            output_path = self.output_dir / f"{model_name.replace('/', '_')}_original_run{run}.onnx"
            
            start_time = time.time()
            result = exporter.export(model, inputs, str(output_path))
            end_time = time.time()
            
            original_times.append(end_time - start_time)
            print(f"    ⏱️  {end_time - start_time:.3f}s")
        
        # Test optimized HTP
        optimized_times = []
        for run in range(num_runs):
            print(f"  ⚡ Optimized HTP - Run {run + 1}/{num_runs}")
            
            exporter = HTPHierarchyExporter()
            
            # Apply optimizations
            architecture_info = HuggingFaceSpecificOptimizations.detect_transformer_architecture(model)
            hf_optimizations = HuggingFaceSpecificOptimizations.apply_transformer_optimizations(
                exporter, model, architecture_info
            )
            exporter = apply_htp_optimizations(exporter)
            
            output_path = self.output_dir / f"{model_name.replace('/', '_')}_optimized_run{run}.onnx"
            
            start_time = time.time()
            result = exporter.export(model, inputs, str(output_path))
            end_time = time.time()
            
            optimized_times.append(end_time - start_time)
            print(f"    ⏱️  {end_time - start_time:.3f}s")
        
        # Calculate statistics
        original_avg = sum(original_times) / len(original_times)
        optimized_avg = sum(optimized_times) / len(optimized_times)
        
        improvement = ((original_avg - optimized_avg) / original_avg) * 100
        
        benchmark_result = {
            'model': model_name,
            'original_times': original_times,
            'optimized_times': optimized_times,
            'original_avg': original_avg,
            'optimized_avg': optimized_avg,
            'improvement_percentage': improvement,
            'improvement_seconds': original_avg - optimized_avg,
            'architecture_info': architecture_info,
            'hf_optimizations_applied': hf_optimizations
        }
        
        self.results[model_name] = benchmark_result
        
        print(f"\\n📈 Results for {model_name}:")
        print(f"   Original average: {original_avg:.3f}s")
        print(f"   Optimized average: {optimized_avg:.3f}s")
        print(f"   Improvement: {improvement:+.1f}% ({original_avg - optimized_avg:+.3f}s)")
        
        return benchmark_result
    
    def compare_with_baseline(self):
        """Compare results with Iteration 16 baseline metrics."""
        
        # Baseline from Iteration 16
        baseline_metrics = {
            'microsoft/resnet-50': {'htp': 4.08, 'usage_based': 3.62},
            'facebook/sam-vit-base': {'htp': 40.61, 'usage_based': 33.64}
        }
        
        comparison = {
            'baseline_comparison': {},
            'overall_improvement': 0,
            'models_improved': 0,
            'fastest_strategy': None
        }
        
        total_improvement = 0
        models_tested = 0
        
        for model_name, result in self.results.items():
            if model_name in baseline_metrics:
                baseline_htp = baseline_metrics[model_name]['htp']
                baseline_usage = baseline_metrics[model_name]['usage_based']
                current_optimized = result['optimized_avg']
                
                vs_baseline_htp = ((baseline_htp - current_optimized) / baseline_htp) * 100
                vs_baseline_usage = ((baseline_usage - current_optimized) / baseline_usage) * 100
                
                comparison['baseline_comparison'][model_name] = {
                    'baseline_htp': baseline_htp,
                    'baseline_usage_based': baseline_usage,
                    'current_optimized_htp': current_optimized,
                    'improvement_vs_baseline_htp': vs_baseline_htp,
                    'improvement_vs_usage_based': vs_baseline_usage,
                    'faster_than_usage_based': current_optimized < baseline_usage
                }
                
                total_improvement += vs_baseline_htp
                models_tested += 1
        
        if models_tested > 0:
            comparison['overall_improvement'] = total_improvement / models_tested
            comparison['models_improved'] = sum(
                1 for comp in comparison['baseline_comparison'].values()
                if comp['improvement_vs_baseline_htp'] > 0
            )
        
        return comparison
    
    def save_benchmark_results(self):
        """Save comprehensive benchmark results."""
        
        comparison = self.compare_with_baseline()
        
        report = {
            'iteration': 17,
            'benchmark_type': 'htp_optimization_performance',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmark_results': self.results,
            'baseline_comparison': comparison,
            'summary': {
                'models_tested': len(self.results),
                'average_improvement': sum(r['improvement_percentage'] for r in self.results.values()) / len(self.results),
                'best_improvement': max(r['improvement_percentage'] for r in self.results.values()),
                'worst_improvement': min(r['improvement_percentage'] for r in self.results.values())
            }
        }
        
        # Save detailed report
        report_file = self.output_dir / "htp_optimization_benchmark.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n📊 Benchmark report saved to: {report_file}")
        return report
    
    def print_final_summary(self):
        """Print comprehensive benchmark summary."""
        
        comparison = self.compare_with_baseline()
        
        print(f"\\n{'='*70}")
        print("🏆 HTP OPTIMIZATION BENCHMARK SUMMARY")
        print(f"{'='*70}")
        
        print(f"\\n📊 Individual Model Results:")
        for model_name, result in self.results.items():
            print(f"\\n  {model_name}:")
            print(f"    Original HTP: {result['original_avg']:.3f}s")
            print(f"    Optimized HTP: {result['optimized_avg']:.3f}s")
            print(f"    Improvement: {result['improvement_percentage']:+.1f}% ({result['improvement_seconds']:+.3f}s)")
            
            if model_name in comparison['baseline_comparison']:
                baseline_comp = comparison['baseline_comparison'][model_name]
                print(f"    vs Iteration 16 HTP: {baseline_comp['improvement_vs_baseline_htp']:+.1f}%")
                print(f"    vs Iteration 16 Usage: {baseline_comp['improvement_vs_usage_based']:+.1f}%")
                if baseline_comp['faster_than_usage_based']:
                    print(f"    🏆 FASTER THAN USAGE-BASED!")
        
        print(f"\\n🎯 Overall Performance:")
        if self.results:
            avg_improvement = sum(r['improvement_percentage'] for r in self.results.values()) / len(self.results)
            print(f"    Average optimization improvement: {avg_improvement:+.1f}%")
            print(f"    vs Baseline HTP: {comparison.get('overall_improvement', 0):+.1f}%")
            print(f"    Models improved: {comparison.get('models_improved', 0)}/{len(self.results)}")
        
        print(f"\\n🚀 Optimization Impact:")
        print(f"    Strategy ranking: Optimized HTP → Original HTP")
        
        # Check if we beat usage-based
        beats_usage_based = any(
            comp.get('faster_than_usage_based', False) 
            for comp in comparison.get('baseline_comparison', {}).values()
        )
        
        if beats_usage_based:
            print(f"    🏆 ACHIEVEMENT: HTP now faster than Usage-Based strategy!")
        else:
            print(f"    🎯 TARGET: Continue optimizing to match Usage-Based performance")


def main():
    """Run HTP optimization benchmarks."""
    
    benchmark = HTPOptimizationBenchmark()
    
    print("🚀 Starting HTP Optimization Benchmarking...")
    print("This will test original vs optimized HTP performance")
    
    # Test with ResNet-50 (faster model)
    benchmark.benchmark_model("microsoft/resnet-50", num_runs=2)
    
    # Save and summarize results
    benchmark.save_benchmark_results()
    benchmark.print_final_summary()
    
    print(f"\\n✅ Benchmark completed! Results saved to: {benchmark.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())