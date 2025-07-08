#!/usr/bin/env python3
"""
HTP Integration Verification Script

This script verifies that the OptimizedTracingHierarchyBuilder integration
with HTP strategy maintains quality while improving performance.
"""

import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def run_cli_command(args: list) -> Tuple[bool, str, float]:
    """Run CLI command and return success, output, and execution time."""
    start_time = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "modelexport"] + args,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()
        
        return (
            result.returncode == 0,
            result.stdout + result.stderr,
            end_time - start_time
        )
    except subprocess.TimeoutExpired:
        return False, "Command timed out", time.time() - start_time
    except Exception as e:
        return False, f"Command failed: {e}", time.time() - start_time

def verify_onnx_file(onnx_path: str) -> Dict[str, Any]:
    """Verify ONNX file is valid and extract metrics."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        return {
            'valid': True,
            'node_count': len(model.graph.node),
            'input_count': len(model.graph.input),
            'output_count': len(model.graph.output),
            'file_size_mb': os.path.getsize(onnx_path) / 1024 / 1024
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def load_hierarchy_metadata(onnx_path: str) -> Dict[str, Any]:
    """Load hierarchy metadata from sidecar file."""
    metadata_path = onnx_path.replace('.onnx', '_hierarchy.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def run_baseline_test() -> Dict[str, Any]:
    """Run baseline HTP test before optimization."""
    print("🔄 Running baseline HTP test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "baseline_htp.onnx")
        
        # Export with current HTP
        success, output, export_time = run_cli_command([
            "export", "prajjwal1/bert-tiny", output_path,
            "--strategy", "htp",
            "--input-text", "Hello world"
        ])
        
        if not success:
            return {'success': False, 'error': f"Export failed: {output}"}
        
        # Verify output
        onnx_metrics = verify_onnx_file(output_path)
        hierarchy_data = load_hierarchy_metadata(output_path)
        
        return {
            'success': True,
            'export_time': export_time,
            'onnx_metrics': onnx_metrics,
            'hierarchy_data': hierarchy_data,
            'output': output
        }

def run_optimized_test() -> Dict[str, Any]:
    """Run optimized HTP test after integration."""
    print("🔄 Running optimized HTP test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "optimized_htp.onnx")
        
        # Export with optimized HTP
        success, output, export_time = run_cli_command([
            "export", "prajjwal1/bert-tiny", output_path,
            "--strategy", "htp",
            "--input-text", "Hello world"
        ])
        
        if not success:
            return {'success': False, 'error': f"Export failed: {output}"}
        
        # Verify output
        onnx_metrics = verify_onnx_file(output_path)
        hierarchy_data = load_hierarchy_metadata(output_path)
        
        return {
            'success': True,
            'export_time': export_time,
            'onnx_metrics': onnx_metrics,
            'hierarchy_data': hierarchy_data,
            'output': output
        }

def compare_results(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline vs optimized results."""
    
    comparison = {
        'performance': {},
        'quality': {},
        'compatibility': {}
    }
    
    # Performance comparison
    if baseline['success'] and optimized['success']:
        baseline_time = baseline['export_time']
        optimized_time = optimized['export_time']
        
        comparison['performance'] = {
            'baseline_time': baseline_time,
            'optimized_time': optimized_time,
            'speedup': baseline_time / optimized_time,
            'time_savings_pct': ((baseline_time - optimized_time) / baseline_time) * 100
        }
        
        # Quality comparison
        baseline_onnx = baseline['onnx_metrics']
        optimized_onnx = optimized['onnx_metrics']
        
        comparison['quality'] = {
            'onnx_valid': baseline_onnx.get('valid', False) and optimized_onnx.get('valid', False),
            'node_count_change': optimized_onnx.get('node_count', 0) - baseline_onnx.get('node_count', 0),
            'file_size_change_mb': optimized_onnx.get('file_size_mb', 0) - baseline_onnx.get('file_size_mb', 0)
        }
        
        # Hierarchy comparison
        baseline_hierarchy = baseline.get('hierarchy_data', {})
        optimized_hierarchy = optimized.get('hierarchy_data', {})
        
        baseline_modules = len(baseline_hierarchy.get('node_tags', {}))
        optimized_modules = len(optimized_hierarchy.get('node_tags', {}))
        
        comparison['compatibility'] = {
            'baseline_tagged_nodes': baseline_modules,
            'optimized_tagged_nodes': optimized_modules,
            'module_reduction': baseline_modules - optimized_modules,
            'module_reduction_pct': ((baseline_modules - optimized_modules) / max(baseline_modules, 1)) * 100
        }
    
    return comparison

def print_results(baseline: Dict[str, Any], optimized: Dict[str, Any], comparison: Dict[str, Any]):
    """Print verification results."""
    
    print("\n" + "="*60)
    print("🎯 HTP INTEGRATION VERIFICATION RESULTS")
    print("="*60)
    
    # Success status
    print(f"\n📊 Test Results:")
    print(f"   Baseline HTP: {'✅ SUCCESS' if baseline['success'] else '❌ FAILED'}")
    print(f"   Optimized HTP: {'✅ SUCCESS' if optimized['success'] else '❌ FAILED'}")
    
    if not (baseline['success'] and optimized['success']):
        print("\n❌ Cannot compare results - one or both tests failed")
        if not baseline['success']:
            print(f"   Baseline error: {baseline.get('error', 'Unknown')}")
        if not optimized['success']:
            print(f"   Optimized error: {optimized.get('error', 'Unknown')}")
        return
    
    # Performance results
    perf = comparison['performance']
    print(f"\n⚡ Performance Results:")
    print(f"   Baseline export time: {perf['baseline_time']:.2f}s")
    print(f"   Optimized export time: {perf['optimized_time']:.2f}s")
    print(f"   Speedup: {perf['speedup']:.2f}x")
    print(f"   Time savings: {perf['time_savings_pct']:.1f}%")
    
    # Quality results
    quality = comparison['quality']
    print(f"\n🎯 Quality Results:")
    print(f"   ONNX validation: {'✅ PASSED' if quality['onnx_valid'] else '❌ FAILED'}")
    print(f"   Node count change: {quality['node_count_change']:+d}")
    print(f"   File size change: {quality['file_size_change_mb']:+.2f}MB")
    
    # Compatibility results
    compat = comparison['compatibility']
    print(f"\n🔧 Hierarchy Results:")
    print(f"   Baseline tagged nodes: {compat['baseline_tagged_nodes']}")
    print(f"   Optimized tagged nodes: {compat['optimized_tagged_nodes']}")
    print(f"   Module reduction: {compat['module_reduction']} ({compat['module_reduction_pct']:.1f}%)")
    
    # Overall assessment
    print(f"\n🏆 Overall Assessment:")
    
    # Check success criteria
    success_criteria = []
    
    if perf['speedup'] >= 1.0:
        success_criteria.append("✅ Performance maintained or improved")
    else:
        success_criteria.append("❌ Performance regression detected")
    
    if quality['onnx_valid']:
        success_criteria.append("✅ ONNX output quality maintained")
    else:
        success_criteria.append("❌ ONNX validation failed")
    
    if compat['module_reduction'] > 0:
        success_criteria.append("✅ Module processing optimized")
    else:
        success_criteria.append("⚠️  No module reduction achieved")
    
    for criterion in success_criteria:
        print(f"   {criterion}")
    
    # Final verdict
    passed_criteria = sum(1 for c in success_criteria if c.startswith("✅"))
    total_criteria = len(success_criteria)
    
    if passed_criteria == total_criteria:
        print(f"\n🎉 INTEGRATION SUCCESSFUL: All {total_criteria} criteria passed!")
    elif passed_criteria >= total_criteria * 0.67:
        print(f"\n⚠️  INTEGRATION PARTIAL: {passed_criteria}/{total_criteria} criteria passed")
    else:
        print(f"\n❌ INTEGRATION FAILED: Only {passed_criteria}/{total_criteria} criteria passed")

def main():
    """Main verification function."""
    print("🚀 Starting HTP Integration Verification")
    print("="*50)
    
    # Note: This script assumes we'll run it twice - before and after integration
    # For now, we'll run the current version and prepare for comparison
    
    try:
        # Run baseline test (current implementation)
        baseline_result = run_baseline_test()
        
        # For now, run the same test as "optimized" since we haven't integrated yet
        # After integration, this will use the optimized version
        optimized_result = run_optimized_test()
        
        # Compare results
        comparison = compare_results(baseline_result, optimized_result)
        
        # Print results
        print_results(baseline_result, optimized_result, comparison)
        
        # Save results for future comparison
        results = {
            'baseline': baseline_result,
            'optimized': optimized_result,
            'comparison': comparison,
            'timestamp': time.time()
        }
        
        results_file = Path(__file__).parent / "htp_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)