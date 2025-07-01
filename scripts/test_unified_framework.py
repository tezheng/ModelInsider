#!/usr/bin/env python3
"""
Unified Framework Testing for Iteration 19

Tests the complete unified optimization framework with automatic strategy selection.
"""

import time
import torch
from pathlib import Path
from transformers import AutoModel
import json
from typing import Dict, Any

from modelexport.unified_export import UnifiedExporter, export_model
from modelexport.core.strategy_selector import ExportStrategy, StrategySelector
from modelexport.core.unified_optimizer import UnifiedOptimizer


class UnifiedFrameworkTester:
    """Test the unified framework end-to-end."""
    
    def __init__(self, output_dir: str = "temp/iteration_19"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def test_automatic_strategy_selection(self):
        """Test automatic strategy selection for different model types."""
        
        print("\\n🧠 Testing Automatic Strategy Selection...")
        
        # Test models with different characteristics
        test_cases = [
            {
                "name": "HuggingFace ResNet",
                "model_name": "microsoft/resnet-50",
                "input_shape": (1, 3, 224, 224),
                "expected_strategy": "usage_based"
            },
            {
                "name": "Simple CNN",
                "model": self._create_simple_cnn(),
                "input_shape": (1, 3, 32, 32),
                "expected_strategy": "usage_based"  # Still fastest overall
            }
        ]
        
        selection_results = {}
        
        for case in test_cases:
            print(f"  📋 Testing {case['name']}...")
            
            # Load or create model
            if "model_name" in case:
                model = AutoModel.from_pretrained(case["model_name"])
            else:
                model = case["model"]
            
            model.eval()
            inputs = torch.randn(*case["input_shape"])
            
            # Test strategy selection
            selector = StrategySelector()
            recommendation = selector.recommend_strategy(model, prioritize_speed=True)
            
            selection_results[case["name"]] = {
                "recommended_strategy": recommendation.primary_strategy.value,
                "expected_strategy": case["expected_strategy"],
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "warnings": recommendation.warnings,
                "correct_selection": recommendation.primary_strategy.value == case["expected_strategy"]
            }
            
            print(f"    Recommended: {recommendation.primary_strategy.value}")
            print(f"    Expected: {case['expected_strategy']}")
            print(f"    Confidence: {recommendation.confidence:.2f}")
            print(f"    ✅ Correct" if selection_results[case["name"]]["correct_selection"] else "❌ Incorrect")
        
        return selection_results
    
    def test_unified_export_interface(self):
        """Test the unified export interface with different configurations."""
        
        print("\\n🔧 Testing Unified Export Interface...")
        
        # Create test model
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        test_configs = [
            {
                "name": "Auto Strategy",
                "config": {"strategy": "auto", "optimize": True},
                "expected_success": True
            },
            {
                "name": "Forced Usage-Based",
                "config": {"strategy": "usage_based", "optimize": True},
                "expected_success": True
            },
            {
                "name": "Forced HTP",
                "config": {"strategy": "htp", "optimize": True},
                "expected_success": True
            },
            {
                "name": "No Optimizations",
                "config": {"strategy": "usage_based", "optimize": False},
                "expected_success": True
            }
        ]
        
        interface_results = {}
        
        for config in test_configs:
            print(f"  🧪 Testing {config['name']}...")
            
            try:
                output_path = self.output_dir / f"unified_{config['name'].lower().replace(' ', '_')}.onnx"
                
                # Test using the unified export function
                start_time = time.time()
                report = export_model(
                    model, inputs, output_path,
                    **config["config"]
                )
                export_time = time.time() - start_time
                
                interface_results[config["name"]] = {
                    "success": report["summary"]["success"],
                    "export_time": export_time,
                    "strategy_used": report["summary"]["final_strategy"],
                    "optimizations_applied": len(report.get("optimizations_applied", [])),
                    "file_size": report["summary"]["file_size"],
                    "warnings": len(report.get("warnings", [])),
                    "expected_success": config["expected_success"]
                }
                
                print(f"    ✅ Success: {export_time:.3f}s using {report['summary']['final_strategy']}")
                print(f"    📦 File size: {report['summary']['file_size'] / 1024 / 1024:.2f} MB")
                print(f"    ⚡ Optimizations: {len(report.get('optimizations_applied', []))}")
                
            except Exception as e:
                interface_results[config["name"]] = {
                    "success": False,
                    "error": str(e),
                    "expected_success": config["expected_success"]
                }
                print(f"    ❌ Failed: {e}")
        
        return interface_results
    
    def test_optimization_framework(self):
        """Test the unified optimization framework."""
        
        print("\\n⚡ Testing Optimization Framework...")
        
        # Create test model
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        optimization_results = {}
        
        # Test each strategy with and without optimizations
        strategies = ["usage_based", "htp"]
        
        for strategy in strategies:
            print(f"  🔬 Testing {strategy} optimization...")
            
            # Test without optimizations
            exporter_unopt = UnifiedExporter(
                strategy=strategy,
                enable_optimizations=False,
                enable_monitoring=True
            )
            
            output_path_unopt = self.output_dir / f"unopt_{strategy}.onnx"
            start_time = time.time()
            report_unopt = exporter_unopt.export(model, inputs, output_path_unopt)
            time_unopt = time.time() - start_time
            
            # Test with optimizations
            exporter_opt = UnifiedExporter(
                strategy=strategy,
                enable_optimizations=True,
                enable_monitoring=True
            )
            
            output_path_opt = self.output_dir / f"opt_{strategy}.onnx"
            start_time = time.time()
            report_opt = exporter_opt.export(model, inputs, output_path_opt)
            time_opt = time.time() - start_time
            
            # Calculate improvement
            improvement = ((time_unopt - time_opt) / time_unopt) * 100
            
            optimization_results[strategy] = {
                "unoptimized_time": time_unopt,
                "optimized_time": time_opt,
                "improvement_percentage": improvement,
                "optimizations_applied": len(report_opt.get("optimizations_applied", [])),
                "both_successful": report_unopt["summary"]["success"] and report_opt["summary"]["success"]
            }
            
            print(f"    Unoptimized: {time_unopt:.3f}s")
            print(f"    Optimized: {time_opt:.3f}s")
            print(f"    Improvement: {improvement:+.1f}%")
            print(f"    Optimizations: {len(report_opt.get('optimizations_applied', []))}")
        
        return optimization_results
    
    def test_benchmarking_suite(self):
        """Test the built-in benchmarking functionality."""
        
        print("\\n📊 Testing Benchmarking Suite...")
        
        # Create test model
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        # Run benchmark
        exporter = UnifiedExporter(enable_optimizations=True)
        
        try:
            benchmark_results = exporter.benchmark_strategies(
                model, inputs, strategies=["usage_based", "htp"]
            )
            
            print(f"  🏆 Strategy Ranking:")
            for i, strategy in enumerate(benchmark_results["ranking"], 1):
                time_taken = benchmark_results["strategies"][strategy]["export_time"]
                print(f"    {i}. {strategy}: {time_taken:.3f}s")
            
            return {
                "success": True,
                "strategies_tested": len(benchmark_results["strategies"]),
                "fastest_strategy": benchmark_results["ranking"][0] if benchmark_results["ranking"] else None,
                "results": benchmark_results
            }
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_fallback_mechanism(self):
        """Test strategy fallback for incompatible models."""
        
        print("\\n🔄 Testing Fallback Mechanism...")
        
        # Create test model
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        # Force FX strategy (should fail and fallback)
        try:
            exporter = UnifiedExporter(
                strategy="fx_graph",
                enable_optimizations=True,
                enable_monitoring=True
            )
            
            output_path = self.output_dir / "fallback_test.onnx"
            report = exporter.export(model, inputs, output_path)
            
            return {
                "fallback_triggered": report["summary"]["final_strategy"] != "fx_graph",
                "final_strategy": report["summary"]["final_strategy"],
                "success": report["summary"]["success"],
                "strategies_tried": report["summary"].get("strategies_tried", []),
                "warnings": report.get("warnings", [])
            }
            
        except Exception as e:
            return {
                "fallback_triggered": False,
                "success": False,
                "error": str(e)
            }
    
    def _create_simple_cnn(self):
        """Create a simple CNN for testing."""
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.fc = torch.nn.Linear(32 * 8 * 8, 10)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(-1, 32 * 8 * 8)
                x = self.fc(x)
                return x
        
        model = SimpleCNN()
        model.eval()
        return model
    
    def run_comprehensive_test(self):
        """Run all tests and generate comprehensive report."""
        
        print("\\n🚀 Starting Unified Framework Comprehensive Testing...")
        
        # Run all tests
        test_results = {
            "iteration": 19,
            "test_type": "unified_framework_comprehensive",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "tests": {}
        }
        
        print("\\n" + "="*60)
        print("🧪 UNIFIED FRAMEWORK TESTING")
        print("="*60)
        
        # Test 1: Strategy Selection
        test_results["tests"]["strategy_selection"] = self.test_automatic_strategy_selection()
        
        # Test 2: Unified Interface
        test_results["tests"]["unified_interface"] = self.test_unified_export_interface()
        
        # Test 3: Optimization Framework
        test_results["tests"]["optimization_framework"] = self.test_optimization_framework()
        
        # Test 4: Benchmarking Suite
        test_results["tests"]["benchmarking_suite"] = self.test_benchmarking_suite()
        
        # Test 5: Fallback Mechanism
        test_results["tests"]["fallback_mechanism"] = self.test_fallback_mechanism()
        
        # Generate summary
        test_results["summary"] = self._generate_test_summary(test_results["tests"])
        
        # Save results
        report_file = self.output_dir / "unified_framework_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Print summary
        self._print_test_summary(test_results)
        
        return test_results
    
    def _generate_test_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        
        summary = {
            "total_tests": len(tests),
            "tests_passed": 0,
            "tests_failed": 0,
            "key_findings": [],
            "recommendations": []
        }
        
        # Analyze strategy selection
        if "strategy_selection" in tests:
            selection_tests = tests["strategy_selection"]
            correct_selections = sum(1 for test in selection_tests.values() if test["correct_selection"])
            summary["tests_passed"] += 1 if correct_selections == len(selection_tests) else 0
            summary["key_findings"].append(f"Strategy selection: {correct_selections}/{len(selection_tests)} correct")
        
        # Analyze unified interface
        if "unified_interface" in tests:
            interface_tests = tests["unified_interface"]
            successful_interfaces = sum(1 for test in interface_tests.values() if test["success"])
            summary["tests_passed"] += 1 if successful_interfaces == len(interface_tests) else 0
            summary["key_findings"].append(f"Unified interface: {successful_interfaces}/{len(interface_tests)} successful")
        
        # Analyze optimizations
        if "optimization_framework" in tests:
            opt_tests = tests["optimization_framework"]
            working_optimizations = sum(1 for test in opt_tests.values() if test["both_successful"])
            summary["tests_passed"] += 1 if working_optimizations == len(opt_tests) else 0
            summary["key_findings"].append(f"Optimizations: {working_optimizations}/{len(opt_tests)} strategies optimized")
        
        # Analyze benchmarking
        if "benchmarking_suite" in tests:
            benchmark_test = tests["benchmarking_suite"]
            summary["tests_passed"] += 1 if benchmark_test["success"] else 0
            if benchmark_test["success"]:
                summary["key_findings"].append(f"Fastest strategy: {benchmark_test['fastest_strategy']}")
        
        # Analyze fallback
        if "fallback_mechanism" in tests:
            fallback_test = tests["fallback_mechanism"]
            summary["tests_passed"] += 1 if fallback_test["success"] else 0
            if fallback_test["fallback_triggered"]:
                summary["key_findings"].append("Fallback mechanism working correctly")
        
        summary["tests_failed"] = summary["total_tests"] - summary["tests_passed"]
        
        # Generate recommendations
        if summary["tests_passed"] == summary["total_tests"]:
            summary["recommendations"].append("✅ Unified framework ready for production")
        else:
            summary["recommendations"].append("⚠️ Some tests failed - review before production")
        
        return summary
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """Print comprehensive test summary."""
        
        summary = test_results["summary"]
        
        print(f"\\n{'='*60}")
        print("📋 UNIFIED FRAMEWORK TEST SUMMARY")
        print(f"{'='*60}")
        
        print(f"\\n🎯 Overall Results:")
        print(f"   Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
        print(f"   Success Rate: {(summary['tests_passed']/summary['total_tests'])*100:.1f}%")
        
        print(f"\\n🔍 Key Findings:")
        for finding in summary["key_findings"]:
            print(f"   • {finding}")
        
        print(f"\\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")
        
        print(f"\\n📁 Detailed report saved to:")
        print(f"   {self.output_dir}/unified_framework_test_report.json")


def main():
    """Run unified framework testing."""
    
    tester = UnifiedFrameworkTester()
    results = tester.run_comprehensive_test()
    
    print(f"\\n✅ Unified Framework Testing completed!")
    return 0


if __name__ == '__main__':
    exit(main())