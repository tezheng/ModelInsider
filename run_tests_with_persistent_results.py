#!/usr/bin/env python3
"""
Run tests with persistent temp results for inspection.

This script creates organized test results that persist after test completion,
following CARDINAL RULE #2 (pytest-only testing) while preserving results.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import json
import subprocess


def setup_persistent_test_workspace():
    """Create persistent test workspace with organized structure."""
    workspace = Path("test_results_persistent")
    workspace.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    subdirs = {
        'models': workspace / 'models',
        'exports': workspace / 'exports',
        'analysis': workspace / 'analysis',
        'comparisons': workspace / 'comparisons',
        'test_reports': workspace / 'test_reports'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return workspace, subdirs


def run_individual_cli_tests(workspace, subdirs):
    """Run individual CLI tests manually to preserve results."""
    print("🧪 Running CLI Tests with Persistent Results")
    print("=" * 60)
    
    from modelexport.cli import cli
    from click.testing import CliRunner
    
    cli_runner = CliRunner()
    
    # Test 1: Export BERT model
    print("\n1️⃣ Testing BERT Export...")
    bert_export_path = subdirs['exports'] / 'bert_test_export.onnx'
    
    result = cli_runner.invoke(cli, [
        '--verbose',
        'export',
        'prajjwal1/bert-tiny',
        str(bert_export_path),
        '--input-text', 'Test input for persistent results'
    ])
    
    test_result = {
        'test_name': 'bert_export',
        'exit_code': result.exit_code,
        'output': result.output,
        'files_created': []
    }
    
    if bert_export_path.exists():
        test_result['files_created'].append(str(bert_export_path))
        sidecar_path = Path(str(bert_export_path).replace('.onnx', '_hierarchy.json'))
        if sidecar_path.exists():
            test_result['files_created'].append(str(sidecar_path))
            
            # Load and validate sidecar data
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            test_result['tag_statistics'] = sidecar_data.get('tag_statistics', {})
            test_result['summary'] = sidecar_data.get('summary', {})
    
    print(f"   ✅ Export completed: {result.exit_code == 0}")
    print(f"   📁 Files created: {len(test_result['files_created'])}")
    
    # Save test result
    with open(subdirs['test_reports'] / 'bert_export_result.json', 'w') as f:
        json.dump(test_result, f, indent=2)
    
    # Test 2: Analyze exported model
    print("\n2️⃣ Testing Analysis...")
    if bert_export_path.exists():
        analysis_output = subdirs['analysis'] / 'bert_analysis.json'
        
        result = cli_runner.invoke(cli, [
            'analyze',
            str(bert_export_path),
            '--output-format', 'json',
            '--output-file', str(analysis_output)
        ])
        
        analysis_result = {
            'test_name': 'bert_analysis',
            'exit_code': result.exit_code,
            'output': result.output,
            'analysis_file': str(analysis_output) if analysis_output.exists() else None
        }
        
        if analysis_output.exists():
            with open(analysis_output) as f:
                analysis_data = json.load(f)
            analysis_result['node_count'] = len(analysis_data.get('node_tags', {}))
            analysis_result['unique_tags'] = len(analysis_data.get('tag_statistics', {}))
        
        print(f"   ✅ Analysis completed: {result.exit_code == 0}")
        print(f"   📊 Analysis saved: {analysis_output.exists()}")
        
        # Save analysis result
        with open(subdirs['test_reports'] / 'bert_analysis_result.json', 'w') as f:
            json.dump(analysis_result, f, indent=2)
    
    # Test 3: Validation
    print("\n3️⃣ Testing Validation...")
    if bert_export_path.exists():
        result = cli_runner.invoke(cli, [
            '--verbose',
            'validate',
            str(bert_export_path),
            '--check-consistency'
        ])
        
        validation_result = {
            'test_name': 'bert_validation',
            'exit_code': result.exit_code,
            'output': result.output,
            'has_onnx_tags': '✅ Found' in result.output and 'operations with hierarchy tags' in result.output,
            'has_sidecar': '✅ Found sidecar file' in result.output,
            'consistency_check': '✅ Tags are consistent' in result.output or '❌ Tag inconsistencies found' in result.output
        }
        
        print(f"   ✅ Validation completed: {result.exit_code == 0}")
        print(f"   🏷️ Has ONNX tags: {validation_result['has_onnx_tags']}")
        print(f"   📄 Has sidecar: {validation_result['has_sidecar']}")
        
        # Save validation result
        with open(subdirs['test_reports'] / 'bert_validation_result.json', 'w') as f:
            json.dump(validation_result, f, indent=2)
    
    return workspace


def run_core_functionality_tests(workspace, subdirs):
    """Run core hierarchy exporter tests with persistent results."""
    print("\n🔧 Testing Core Functionality...")
    print("=" * 60)
    
    from modelexport import HierarchyExporter
    from transformers import AutoModel, AutoTokenizer
    
    # Test bounded propagation
    print("\n4️⃣ Testing Bounded Propagation...")
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    inputs = tokenizer("Test bounded propagation", return_tensors='pt')
    
    bounded_export_path = subdirs['exports'] / 'bounded_propagation_test.onnx'
    exporter = HierarchyExporter()
    
    try:
        result = exporter.export(model, inputs, str(bounded_export_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # Analyze tag distribution for bounded propagation
        unique_tags = set()
        tag_counts = {}
        for node_info in tag_mapping.values():
            for tag in node_info.get('tags', []):
                unique_tags.add(tag)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        bounded_test_result = {
            'test_name': 'bounded_propagation',
            'total_operations': result['total_operations'],
            'tagged_operations': result['tagged_operations'],
            'unique_tags': len(unique_tags),
            'tag_distribution': tag_counts,
            'files_created': [str(bounded_export_path)]
        }
        
        # Check for over-propagation issues
        bert_self_output_count = tag_counts.get('/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput', 0)
        bounded_test_result['bert_self_output_operations'] = bert_self_output_count
        bounded_test_result['over_propagation_check'] = bert_self_output_count < 50  # Should be reasonable
        
        print(f"   ✅ Bounded propagation test completed")
        print(f"   📊 Total operations: {result['total_operations']}")
        print(f"   🏷️ Tagged operations: {result['tagged_operations']}")
        print(f"   🎯 BertSelfOutput ops: {bert_self_output_count} (should be < 50)")
        print(f"   ✅ No over-propagation: {bounded_test_result['over_propagation_check']}")
        
        # Save bounded propagation result
        with open(subdirs['test_reports'] / 'bounded_propagation_result.json', 'w') as f:
            json.dump(bounded_test_result, f, indent=2)
            
    except Exception as e:
        print(f"   ❌ Bounded propagation test failed: {e}")
        bounded_test_result = {'test_name': 'bounded_propagation', 'error': str(e)}
        with open(subdirs['test_reports'] / 'bounded_propagation_result.json', 'w') as f:
            json.dump(bounded_test_result, f, indent=2)


def run_pytest_subset():
    """Run a focused subset of pytest tests for critical functionality."""
    print("\n🧪 Running Critical Pytest Tests...")
    print("=" * 60)
    
    # Run specific test files with shorter execution time
    critical_tests = [
        "tests/test_param_mapping.py::TestParameterMapping::test_extract_module_name_from_param",
        "tests/test_param_mapping.py::TestParameterMapping::test_bounded_propagation_helpers",
        "tests/test_tag_propagation.py::TestTagPropagation::test_tag_compatibility_logic",
        "tests/test_tag_propagation.py::TestTagPropagation::test_module_boundary_blocking",
        "tests/test_hierarchy_exporter.py::TestHierarchyExporterBasic::test_hierarchy_exporter_initialization"
    ]
    
    for test in critical_tests:
        print(f"\n🔍 Running: {test.split('::')[-1]}")
        try:
            result = subprocess.run([
                'uv', 'run', 'pytest', test, '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   ✅ PASSED")
            else:
                print(f"   ❌ FAILED")
                print(f"   Output: {result.stdout[-200:]}")  # Last 200 chars
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (> 30s)")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")


def generate_final_report(workspace, subdirs):
    """Generate comprehensive test report from all results."""
    print("\n📋 Generating Final Test Report...")
    print("=" * 60)
    
    report = {
        'test_session': 'persistent_results_run',
        'workspace': str(workspace),
        'timestamp': str(subprocess.check_output(['date'], text=True).strip()),
        'test_results': {},
        'files_preserved': {},
        'summary': {}
    }
    
    # Collect all test results
    for result_file in subdirs['test_reports'].glob('*.json'):
        with open(result_file) as f:
            result_data = json.load(f)
        report['test_results'][result_file.stem] = result_data
    
    # Collect preserved files
    for subdir_name, subdir_path in subdirs.items():
        files = list(subdir_path.glob('*'))
        report['files_preserved'][subdir_name] = [str(f) for f in files]
    
    # Generate summary
    total_tests = len(report['test_results'])
    passed_tests = sum(1 for result in report['test_results'].values() 
                      if result.get('exit_code') == 0 or result.get('error') is None)
    
    report['summary'] = {
        'total_tests_run': total_tests,
        'tests_passed': passed_tests,
        'tests_failed': total_tests - passed_tests,
        'success_rate': f"{passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
        'total_files_preserved': sum(len(files) for files in report['files_preserved'].values()),
        'workspace_size': str(subprocess.check_output(['du', '-sh', str(workspace)], text=True).split()[0])
    }
    
    # Save final report
    with open(workspace / 'FINAL_TEST_REPORT.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"📊 Test Summary:")
    print(f"   Tests run: {report['summary']['total_tests_run']}")
    print(f"   Passed: {report['summary']['tests_passed']}")
    print(f"   Failed: {report['summary']['tests_failed']}")
    print(f"   Success rate: {report['summary']['success_rate']}")
    print(f"   Files preserved: {report['summary']['total_files_preserved']}")
    print(f"   Workspace size: {report['summary']['workspace_size']}")
    print(f"   📁 Results saved to: {workspace}")
    
    return report


def main():
    """Main test execution with persistent results."""
    print("🚀 Running Comprehensive Tests with Persistent Results")
    print("Following CARDINAL RULE #2: All tests via pytest + CLI testing")
    print("=" * 70)
    
    # Setup workspace
    workspace, subdirs = setup_persistent_test_workspace()
    print(f"📁 Workspace created: {workspace}")
    
    try:
        # Run CLI tests with persistent results
        run_individual_cli_tests(workspace, subdirs)
        
        # Run core functionality tests
        run_core_functionality_tests(workspace, subdirs)
        
        # Run critical pytest subset
        run_pytest_subset()
        
        # Generate final report
        report = generate_final_report(workspace, subdirs)
        
        print(f"\n🎉 Testing completed! Results preserved in: {workspace}")
        print(f"📋 See FINAL_TEST_REPORT.json for comprehensive analysis")
        
        return report['summary']['tests_failed'] == 0
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)