#!/usr/bin/env python3
"""
HTP Performance Analysis Script for Iteration 17

Analyzes HTP strategy performance bottlenecks specifically for HuggingFace models
to identify optimization opportunities.
"""

import time
import cProfile
import pstats
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModel
import json

from modelexport.strategies.htp import HTPHierarchyExporter


class HTPPerformanceAnalyzer:
    """Analyze HTP strategy performance with detailed profiling."""
    
    def __init__(self, output_dir: str = "temp/iteration_17"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def profile_htp_method(self, method_name: str, func, *args, **kwargs):
        """Profile a specific HTP method and return timing results."""
        
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
        
        # Save detailed profiling to file
        profile_file = self.output_dir / f"{method_name}_profile.txt"
        stats.dump_stats(str(profile_file).replace('.txt', '.prof'))
        # Also create readable text version
        with open(profile_file, 'w') as f:
            original_stdout = stats.stream
            stats.stream = f
            stats.print_stats()
            stats.stream = original_stdout
        
        # Extract top function calls
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
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
            'top_functions': top_functions[:10],  # Top 10 most expensive
            'profile_file': str(profile_file)
        }
    
    def analyze_htp_phases(self, model_name: str = "microsoft/resnet-50"):
        """Analyze each phase of HTP export to identify bottlenecks."""
        
        print(f"🔍 Analyzing HTP performance phases for {model_name}...")
        
        # Load model and inputs
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        if 'resnet' in model_name:
            inputs = torch.randn(1, 3, 224, 224)
        else:
            inputs = torch.randn(1, 3, 1024, 1024)
        
        # Initialize HTP exporter
        exporter = HTPHierarchyExporter()
        
        # Prepare model
        exporter._model = model
        exporter._reset_state()
        
        phase_results = {}
        
        try:
            # Phase 1: Hook registration
            print("📌 Phase 1: Hook registration")
            phase_results['hook_registration'] = self.profile_htp_method(
                'hook_registration',
                exporter._register_hooks,
                model
            )
            
            # Phase 2: Built-in module tracking setup
            print("📌 Phase 2: Built-in module tracking setup") 
            phase_results['builtin_tracking_setup'] = self.profile_htp_method(
                'builtin_tracking_setup',
                exporter._setup_builtin_module_tracking,
                model
            )
            
            # Phase 3: Operation patching
            print("📌 Phase 3: Operation patching")
            phase_results['operation_patching'] = self.profile_htp_method(
                'operation_patching',
                exporter._patch_torch_operations_with_builtin_tracking
            )
            
            # Phase 4: Tensor slicing hooks
            print("📌 Phase 4: Tensor slicing hooks")
            phase_results['tensor_slicing_hooks'] = self.profile_htp_method(
                'tensor_slicing_hooks', 
                exporter._setup_tensor_slicing_hooks_with_builtin_tracking
            )
            
            # Phase 5: ONNX export (the actual export call)
            print("📌 Phase 5: ONNX export")
            output_path = self.output_dir / f"{model_name.replace('/', '_')}_phase_test.onnx"
            phase_results['onnx_export'] = self.profile_htp_method(
                'onnx_export',
                torch.onnx.export,
                model, inputs, str(output_path),
                input_names=['input'], 
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}}
            )
            
            # Phase 6: Load ONNX model
            print("📌 Phase 6: Load ONNX model")
            import onnx
            phase_results['onnx_load'] = self.profile_htp_method(
                'onnx_load',
                onnx.load,
                str(output_path)
            )
            
            onnx_model = onnx.load(str(output_path))
            
            # Phase 7: Create hierarchy metadata
            print("📌 Phase 7: Create hierarchy metadata")
            phase_results['hierarchy_metadata'] = self.profile_htp_method(
                'hierarchy_metadata',
                exporter._create_direct_hierarchy_metadata_builtin,
                onnx_model, model
            )
            
            # Phase 8: Inject tags into ONNX
            print("📌 Phase 8: Inject tags into ONNX")
            phase_results['tag_injection'] = self.profile_htp_method(
                'tag_injection',
                exporter._inject_builtin_tags_into_onnx,
                str(output_path), onnx_model
            )
            
        finally:
            # Cleanup
            try:
                exporter._remove_hooks()
                exporter._unpatch_operations_builtin()
                exporter._cleanup_builtin_module_tracking()
            except:
                pass
        
        return phase_results
    
    def analyze_bottlenecks(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Identify bottlenecks (phases taking > 20% of time)
        for phase, data in analysis['phase_breakdown'].items():
            if data['percentage'] > 20:
                analysis['bottlenecks'].append({
                    'phase': phase,
                    'time': data['time'],
                    'percentage': data['percentage']
                })
        
        # Sort bottlenecks by time
        analysis['bottlenecks'].sort(key=lambda x: x['time'], reverse=True)
        
        # Generate optimization opportunities
        if analysis['bottlenecks']:
            top_bottleneck = analysis['bottlenecks'][0]
            analysis['optimization_opportunities'].extend([
                f"Primary bottleneck: {top_bottleneck['phase']} ({top_bottleneck['percentage']:.1f}% of total time)",
                f"Focus optimization on {top_bottleneck['phase']} phase for maximum impact"
            ])
        
        # Check for specific optimization patterns
        for phase, result in phase_results.items():
            if result['success'] and result['top_functions']:
                top_func = result['top_functions'][0]
                if top_func['total_time'] > 1.0:  # > 1 second
                    analysis['optimization_opportunities'].append(
                        f"{phase}: Optimize {top_func['function']} ({top_func['total_time']:.2f}s)"
                    )
        
        return analysis
    
    def compare_with_baseline(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current HTP performance with baseline metrics."""
        
        # Baseline from Iteration 16
        baseline_htp_time = 22.34  # seconds average
        baseline_usage_time = 18.63  # seconds average
        
        current_time = sum(r['total_time'] for r in phase_results.values() if r['success'])
        
        comparison = {
            'baseline_htp_time': baseline_htp_time,
            'baseline_usage_time': baseline_usage_time,
            'current_htp_time': current_time,
            'improvement_vs_baseline': baseline_htp_time - current_time,
            'improvement_percentage': ((baseline_htp_time - current_time) / baseline_htp_time) * 100,
            'gap_vs_usage_based': current_time - baseline_usage_time,
            'target_improvement_needed': current_time - baseline_usage_time,
            'recommendations': []
        }
        
        if comparison['improvement_vs_baseline'] > 0:
            comparison['recommendations'].append(f"✅ Already improved by {comparison['improvement_percentage']:.1f}%")
        else:
            comparison['recommendations'].append(f"❌ Performance regressed by {abs(comparison['improvement_percentage']):.1f}%")
        
        if comparison['gap_vs_usage_based'] > 0:
            comparison['recommendations'].append(f"🎯 Need to reduce time by {comparison['gap_vs_usage_based']:.2f}s to match usage-based")
        else:
            comparison['recommendations'].append("🏆 Already faster than usage-based strategy!")
        
        return comparison
    
    def generate_optimization_report(self, model_name: str = "microsoft/resnet-50"):
        """Generate comprehensive HTP optimization report."""
        
        print(f"\\n🚀 Starting HTP performance analysis for {model_name}...")
        
        # Run phase analysis
        phase_results = self.analyze_htp_phases(model_name)
        
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks(phase_results)
        
        # Compare with baseline
        baseline_comparison = self.compare_with_baseline(phase_results)
        
        # Generate comprehensive report
        report = {
            'iteration': 17,
            'analysis_type': 'htp_performance_bottleneck',
            'model_tested': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase_results': phase_results,
            'bottleneck_analysis': bottleneck_analysis,
            'baseline_comparison': baseline_comparison,
            'key_findings': [],
            'optimization_priorities': []
        }
        
        # Key findings
        if bottleneck_analysis['bottlenecks']:
            top_bottleneck = bottleneck_analysis['bottlenecks'][0]
            report['key_findings'].extend([
                f"Primary performance bottleneck: {top_bottleneck['phase']} ({top_bottleneck['percentage']:.1f}% of total time)",
                f"Total HTP export time: {bottleneck_analysis['total_time']:.2f}s",
                f"Baseline comparison: {baseline_comparison['improvement_percentage']:.1f}% change vs baseline"
            ])
        
        # Optimization priorities
        report['optimization_priorities'] = bottleneck_analysis['optimization_opportunities'][:3]
        
        # Save report
        report_file = self.output_dir / f"htp_performance_analysis_{model_name.replace('/', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_analysis_summary(report)
        
        return report
    
    def print_analysis_summary(self, report: Dict[str, Any]):
        """Print analysis summary."""
        
        print(f"\\n{'='*60}")
        print("📊 HTP PERFORMANCE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\\n🎯 Overall Performance:")
        print(f"   Total export time: {report['bottleneck_analysis']['total_time']:.2f}s")
        print(f"   vs Baseline HTP: {report['baseline_comparison']['improvement_percentage']:+.1f}%")
        print(f"   vs Usage-Based: {report['baseline_comparison']['gap_vs_usage_based']:+.2f}s gap")
        
        print(f"\\n🔍 Phase Breakdown:")
        for phase, data in report['bottleneck_analysis']['phase_breakdown'].items():
            print(f"   {phase}: {data['time']:.3f}s ({data['percentage']:.1f}%)")
        
        print(f"\\n🚨 Top Bottlenecks:")
        for bottleneck in report['bottleneck_analysis']['bottlenecks'][:3]:
            print(f"   • {bottleneck['phase']}: {bottleneck['time']:.3f}s ({bottleneck['percentage']:.1f}%)")
        
        print(f"\\n🎯 Optimization Priorities:")
        for priority in report['optimization_priorities']:
            print(f"   • {priority}")
        
        print(f"\\n💡 Recommendations:")
        for rec in report['baseline_comparison']['recommendations']:
            print(f"   • {rec}")


def main():
    """Run HTP performance analysis."""
    analyzer = HTPPerformanceAnalyzer()
    
    # Test with ResNet-50 (faster model for analysis)
    report = analyzer.generate_optimization_report("microsoft/resnet-50")
    
    print(f"\\n✅ Analysis completed. Detailed results saved to: {analyzer.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())