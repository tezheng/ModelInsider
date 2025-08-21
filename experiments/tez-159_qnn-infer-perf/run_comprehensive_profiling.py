#!/usr/bin/env python3
"""
Non-interactive runner for comprehensive QNN profiling
Automatically profiles all configurations and generates complete metrics report
"""

import sys
from pathlib import Path
import numpy as np

# Import the comprehensive profiler
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_qnn_profiling import ComprehensiveQNNProfiler

def main():
    """Run comprehensive profiling automatically"""
    
    print("\n" + "="*80)
    print("AUTOMATED COMPREHENSIVE QNN HTP PROFILING")
    print("Running ALL Configuration Tests")
    print("="*80)
    
    # Initialize profiler
    profiler = ComprehensiveQNNProfiler(
        output_dir=Path("./comprehensive_profiling_results")
    )
    
    # Create test input data
    input_shape = (1, 3, 224, 224)  # Standard ImageNet shape
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Model path (will use test configuration if not found)
    model_path = Path("test_model.dlc")
    
    print(f"\nInput Shape: {input_shape}")
    print(f"Model: {model_path}")
    print(f"Output Directory: {profiler.output_dir}")
    
    # Profile single best configuration first
    print("\n" + "-"*60)
    print("Phase 1: Quick Single Configuration Test")
    print("-"*60)
    
    best_metrics = profiler.profile_single_configuration(
        model_path=model_path,
        input_data=input_data,
        profiling_level="detailed",
        perf_profile="high_performance",
        warmup_runs=3,
        profile_runs=10
    )
    
    print("\nQuick Test Results:")
    print(f"  Inference Time: {best_metrics.total_inference_time_ms:.2f} ms")
    print(f"  Throughput: {best_metrics.throughput_fps:.1f} FPS")
    print(f"  HVX Utilization: {best_metrics.hvx_utilization_percent:.1f}%")
    print(f"  DDR Bandwidth: {best_metrics.ddr_bandwidth_mbps:.0f} MB/s")
    
    # Profile ALL configurations
    print("\n" + "-"*60)
    print("Phase 2: Comprehensive All-Configuration Profiling")
    print("-"*60)
    print("Testing combinations of:")
    print(f"  - Profiling Levels: {profiler.PROFILING_LEVELS}")
    print(f"  - Performance Profiles: {profiler.PERF_PROFILES}")
    print(f"  - Total Configurations: {len(profiler.PROFILING_LEVELS) * len(profiler.PERF_PROFILES)}")
    
    all_metrics = profiler.profile_all_configurations(
        model_path=model_path,
        input_data=input_data,
        warmup_runs=2,
        profile_runs=5
    )
    
    # Generate comprehensive report
    print("\n" + "-"*60)
    print("Phase 3: Analysis and Reporting")
    print("-"*60)
    
    report = profiler.generate_comprehensive_report()
    
    # Print summary results
    if report and "summary" in report:
        summary = report["summary"]
        
        print("\nPROFILING COMPLETE!")
        print("="*80)
        
        if summary.get("best_configuration"):
            best = summary["best_configuration"]
            print("\nBEST CONFIGURATION:")
            print(f"  Profile: {best['perf_profile']}")
            print(f"  Level: {best['profiling_level']}")
            print(f"  Inference Time: {best['inference_time_ms']:.2f} ms")
            print(f"  Throughput: {best['throughput_fps']:.1f} FPS")
        
        if summary.get("worst_configuration"):
            worst = summary["worst_configuration"]
            print("\nWORST CONFIGURATION:")
            print(f"  Profile: {worst['perf_profile']}")
            print(f"  Level: {worst['profiling_level']}")
            print(f"  Inference Time: {worst['inference_time_ms']:.2f} ms")
            print(f"  Throughput: {worst['throughput_fps']:.1f} FPS")
        
        if "performance_comparison" in report:
            comp = report["performance_comparison"]
            if "inference_time_range_ms" in comp:
                time_range = comp["inference_time_range_ms"]
                print(f"\nPERFORMANCE RANGE:")
                print(f"  Best Time: {time_range['min']:.2f} ms")
                print(f"  Worst Time: {time_range['max']:.2f} ms")
                print(f"  Improvement: {time_range['improvement_percent']:.1f}%")
        
        if "recommendations" in report and report["recommendations"]:
            print("\nOPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
    
    # Print all tested configurations summary
    print("\n" + "-"*60)
    print("ALL CONFIGURATIONS TESTED:")
    print("-"*60)
    
    for config_name, metrics in all_metrics.items():
        print(f"\n{config_name}:")
        print(f"  Inference: {metrics.total_inference_time_ms:.2f} ms")
        print(f"  HVX Util: {metrics.hvx_utilization_percent:.1f}%")
        print(f"  HMX Util: {metrics.hmx_utilization_percent:.1f}%")
        print(f"  DDR BW: {metrics.ddr_bandwidth_mbps:.0f} MB/s")
        print(f"  VTCM: {metrics.vtcm_peak_usage_kb:.0f} KB")
    
    print("\n" + "="*80)
    print("RESULTS SAVED:")
    print(f"  Directory: {profiler.output_dir}")
    print(f"  Report: {profiler.output_dir}/comprehensive_performance_report.json")
    print(f"  Chrome Traces: {profiler.output_dir}/chrome_trace_*.json")
    print("\nTo visualize traces:")
    print("  1. Open Chrome browser")
    print("  2. Navigate to: chrome://tracing")
    print(f"  3. Load trace files from: {profiler.output_dir}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())