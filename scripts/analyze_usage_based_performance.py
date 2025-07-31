#!/usr/bin/env python3
"""
Usage-Based Strategy Performance Analysis for Iteration 18

Analyzes performance bottlenecks in the Usage-Based strategy to identify
optimization opportunities.
"""

import cProfile
import json
import pstats
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

from modelexport.strategies.usage_based import UsageBasedExporter


class UsageBasedPerformanceAnalyzer:
    """Analyze Usage-Based strategy performance with detailed profiling."""
    
    def __init__(self, output_dir: str = "temp/iteration_18"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def profile_method(self, method_name: str, func, *args, **kwargs):
        """Profile a specific method and return timing results."""
        
        # Setup profiler
        profiler = cProfile.Profile()
        
        # Profile the method
        start_time = time.time()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        profiler.disable()
        end_time = time.time()
        
        # Analyze profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Save detailed profiling
        profile_file = self.output_dir / f"{method_name}_profile.prof"
        stats.dump_stats(str(profile_file))
        
        # Extract top function calls
        top_functions = []
        for func_info, (cc, _nc, _tt, ct, _callers) in stats.stats.items():
            if ct > 0.001:  # Only functions taking more than 1ms
                filename, line, func_name = func_info
                top_functions.append({
                    'function': f"{filename}:{line}({func_name})",
                    'calls': cc,
                    'total_time': ct,
                    'per_call_time': ct/cc if cc > 0 else 0
                })
        
        # Sort by total time
        top_functions.sort(key=lambda x: x['total_time'], reverse=True)
        
        return {
            'method': method_name,
            'success': success,
            'error': error,
            'total_time': end_time - start_time,
            'top_functions': top_functions[:10],
            'profile_file': str(profile_file)
        }
    
    def analyze_usage_based_phases(self, model_name: str = "microsoft/resnet-50"):
        """Analyze each phase of Usage-Based export to identify bottlenecks."""
        
        print(f"🔍 Analyzing Usage-Based performance phases for {model_name}...")
        
        # Load model and inputs
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        if 'resnet' in model_name:
            inputs = torch.randn(1, 3, 224, 224)
        else:
            inputs = torch.randn(1, 3, 1024, 1024)
        
        # Initialize exporter
        exporter = UsageBasedExporter()
        
        phase_results = {}
        
        # Phase 1: Module usage tracking (hook registration + forward pass)
        print("📌 Phase 1: Module usage tracking")
        phase_results['module_usage_tracking'] = self.profile_method(
            'module_usage_tracking',
            exporter._track_module_usage,
            model, inputs
        )
        
        # Phase 2: ONNX export
        print("📌 Phase 2: ONNX export")
        output_path = self.output_dir / f"{model_name.replace('/', '_')}_usage_test.onnx"
        phase_results['onnx_export'] = self.profile_method(
            'onnx_export',
            torch.onnx.export,
            model, inputs, str(output_path),
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        
        # Phase 3: Create hierarchy mapping
        print("📌 Phase 3: Create hierarchy mapping")
        phase_results['hierarchy_mapping'] = self.profile_method(
            'hierarchy_mapping',
            exporter._create_hierarchy_mapping
        )
        
        # Phase 4: ONNX model loading and validation
        print("📌 Phase 4: ONNX loading")
        from modelexport.core.onnx_utils import ONNXUtils
        phase_results['onnx_load'] = self.profile_method(
            'onnx_load',
            ONNXUtils.load_and_validate,
            str(output_path)
        )
        
        # Load the model for next phase
        onnx_model = ONNXUtils.load_and_validate(str(output_path))
        
        # Phase 5: Inject hierarchy metadata
        print("📌 Phase 5: Inject hierarchy metadata")
        hierarchy_mapping = exporter._create_hierarchy_mapping()
        phase_results['metadata_injection'] = self.profile_method(
            'metadata_injection',
            ONNXUtils.inject_hierarchy_metadata,
            onnx_model, hierarchy_mapping, "usage_based"
        )
        
        # Phase 6: Save ONNX model
        print("📌 Phase 6: Save ONNX model")
        import onnx
        phase_results['onnx_save'] = self.profile_method(
            'onnx_save',
            onnx.save,
            onnx_model, str(output_path)
        )
        
        return phase_results
    
    def analyze_hook_overhead(self, model_name: str = "microsoft/resnet-50"):
        """Analyze overhead of hook registration and removal."""
        
        print(f"\\n🔍 Analyzing hook overhead for {model_name}...")
        
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # Count modules
        module_count = sum(1 for _ in model.named_modules())
        print(f"  Total modules: {module_count}")
        
        # Measure hook registration time
        hooks = []
        start_time = time.time()
        
        for _name, module in model.named_modules():
            def dummy_hook(module, inputs, outputs):
                pass
            
            hook = module.register_forward_hook(dummy_hook)
            hooks.append(hook)
        
        registration_time = time.time() - start_time
        
        # Measure hook removal time
        start_time = time.time()
        for hook in hooks:
            hook.remove()
        removal_time = time.time() - start_time
        
        return {
            'module_count': module_count,
            'hook_registration_time': registration_time,
            'hook_removal_time': removal_time,
            'per_module_registration': registration_time / module_count,
            'per_module_removal': removal_time / module_count
        }
    
    def analyze_bottlenecks(self, phase_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze phase results to identify bottlenecks."""
        
        analysis = {
            'total_time': sum(r['total_time'] for r in phase_results.values() if r['success']),
            'phase_breakdown': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Calculate time breakdown
        for phase, result in phase_results.items():
            if result['success']:
                analysis['phase_breakdown'][phase] = {
                    'time': result['total_time'],
                    'percentage': 0  # Will calculate after total
                }
        
        # Calculate percentages
        total_time = analysis['total_time']
        for phase_data in analysis['phase_breakdown'].values():
            phase_data['percentage'] = (phase_data['time'] / total_time) * 100
        
        # Identify bottlenecks (phases taking > 15% of time)
        for phase, data in analysis['phase_breakdown'].items():
            if data['percentage'] > 15:
                analysis['bottlenecks'].append({
                    'phase': phase,
                    'time': data['time'],
                    'percentage': data['percentage']
                })
        
        # Sort bottlenecks by time
        analysis['bottlenecks'].sort(key=lambda x: x['time'], reverse=True)
        
        # Generate optimization opportunities
        for phase, result in phase_results.items():
            if result['success'] and result['top_functions']:
                # Look at top time-consuming functions
                for func in result['top_functions'][:3]:
                    if func['total_time'] > 0.1:  # > 100ms
                        analysis['optimization_opportunities'].append({
                            'phase': phase,
                            'function': func['function'],
                            'time': func['total_time'],
                            'calls': func['calls']
                        })
        
        return analysis
    
    def generate_optimization_report(self, model_name: str = "microsoft/resnet-50"):
        """Generate comprehensive Usage-Based optimization report."""
        
        print(f"\\n🚀 Starting Usage-Based performance analysis for {model_name}...")
        
        # Run phase analysis
        phase_results = self.analyze_usage_based_phases(model_name)
        
        # Analyze hook overhead
        hook_analysis = self.analyze_hook_overhead(model_name)
        
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks(phase_results)
        
        # Compare with baselines
        baseline_usage_time = 3.62  # From Iteration 16 for ResNet-50
        current_time = bottleneck_analysis['total_time']
        
        # Generate comprehensive report
        report = {
            'iteration': 18,
            'analysis_type': 'usage_based_performance_bottleneck',
            'model_tested': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase_results': phase_results,
            'hook_analysis': hook_analysis,
            'bottleneck_analysis': bottleneck_analysis,
            'baseline_comparison': {
                'baseline_time': baseline_usage_time,
                'current_time': current_time,
                'difference': current_time - baseline_usage_time,
                'percentage_change': ((current_time - baseline_usage_time) / baseline_usage_time) * 100
            },
            'key_findings': [],
            'optimization_priorities': []
        }
        
        # Key findings
        report['key_findings'].extend([
            f"Total export time: {current_time:.3f}s",
            f"Hook overhead: {hook_analysis['hook_registration_time'] + hook_analysis['hook_removal_time']:.3f}s ({hook_analysis['module_count']} modules)",
            f"Primary bottleneck: {bottleneck_analysis['bottlenecks'][0]['phase'] if bottleneck_analysis['bottlenecks'] else 'None'}"
        ])
        
        # Optimization priorities
        if bottleneck_analysis['optimization_opportunities']:
            for opp in bottleneck_analysis['optimization_opportunities'][:3]:
                report['optimization_priorities'].append(
                    f"{opp['phase']}: Optimize {opp['function'].split('/')[-1]} ({opp['time']:.3f}s, {opp['calls']} calls)"
                )
        
        # Save report
        report_file = self.output_dir / f"usage_based_performance_analysis_{model_name.replace('/', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_analysis_summary(report)
        
        return report
    
    def print_analysis_summary(self, report: dict[str, Any]):
        """Print analysis summary."""
        
        print(f"\\n{'='*60}")
        print("📊 USAGE-BASED PERFORMANCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\\n🎯 Overall Performance:")
        print(f"   Total export time: {report['bottleneck_analysis']['total_time']:.3f}s")
        print(f"   vs Baseline: {report['baseline_comparison']['percentage_change']:+.1f}%")
        
        print(f"\\n🔍 Phase Breakdown:")
        for phase, data in report['bottleneck_analysis']['phase_breakdown'].items():
            print(f"   {phase}: {data['time']:.3f}s ({data['percentage']:.1f}%)")
        
        print(f"\\n🪝 Hook Analysis:")
        hook_data = report['hook_analysis']
        print(f"   Modules tracked: {hook_data['module_count']}")
        print(f"   Registration time: {hook_data['hook_registration_time']:.3f}s")
        print(f"   Removal time: {hook_data['hook_removal_time']:.3f}s")
        print(f"   Per-module overhead: {(hook_data['per_module_registration'] + hook_data['per_module_removal'])*1000:.1f}ms")
        
        print(f"\\n🚨 Top Bottlenecks:")
        for bottleneck in report['bottleneck_analysis']['bottlenecks'][:3]:
            print(f"   • {bottleneck['phase']}: {bottleneck['time']:.3f}s ({bottleneck['percentage']:.1f}%)")
        
        print(f"\\n🎯 Optimization Priorities:")
        for priority in report['optimization_priorities']:
            print(f"   • {priority}")


def main():
    """Run Usage-Based performance analysis."""
    analyzer = UsageBasedPerformanceAnalyzer()
    
    # Test with ResNet-50
    report = analyzer.generate_optimization_report("microsoft/resnet-50")
    
    print(f"\\n✅ Analysis completed. Detailed results saved to: {analyzer.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())