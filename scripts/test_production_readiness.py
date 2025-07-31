#!/usr/bin/env python3
"""
Production Readiness Testing for Iteration 19

Tests the complete modelexport package for production deployment.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModel

# Test the main package interface
import modelexport


class ProductionReadinessTester:
    """Test production readiness of the modelexport package."""
    
    def __init__(self, output_dir: str = "temp/iteration_19"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def test_simple_api_usage(self):
        """Test the simplest possible API usage."""
        
        print("\\n📱 Testing Simple API Usage...")
        
        # Create test model
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        try:
            # Test the simplest export call
            output_path = self.output_dir / "simple_api_test.onnx"
            
            start_time = time.time()
            report = modelexport.export_model(
                model, inputs, output_path
            )
            export_time = time.time() - start_time
            
            # Verify results
            success = (
                report["summary"]["success"] and
                Path(output_path).exists() and
                Path(output_path).stat().st_size > 1000000  # >1MB
            )
            
            result = {
                "success": success,
                "export_time": export_time,
                "strategy_used": report["summary"]["final_strategy"],
                "file_size_mb": Path(output_path).stat().st_size / 1024 / 1024 if Path(output_path).exists() else 0,
                "optimizations": len(report.get("optimizations_applied", [])),
                "api_call": "modelexport.export_model(model, inputs, 'output.onnx')"
            }
            
            print(f"    ✅ Success: {export_time:.3f}s using {report['summary']['final_strategy']}")
            print(f"    📦 File: {result['file_size_mb']:.1f}MB with {result['optimizations']} optimizations")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_package_imports(self):
        """Test that all public APIs can be imported."""
        
        print("\\n📦 Testing Package Imports...")
        
        import_tests = [
            # Main interface
            ("export_model", "modelexport.export_model"),
            ("UnifiedExporter", "modelexport.UnifiedExporter"),
            
            # Strategy selection
            ("ExportStrategy", "modelexport.ExportStrategy"),
            ("StrategySelector", "modelexport.StrategySelector"),
            
            # Individual strategies
            ("UsageBasedExporter", "modelexport.UsageBasedExporter"),
            ("HTPHierarchyExporter", "modelexport.HTPHierarchyExporter"),
            ("FXHierarchyExporter", "modelexport.FXHierarchyExporter"),
            
            # Utilities
            ("UnifiedOptimizer", "modelexport.UnifiedOptimizer"),
            ("create_optimized_exporter", "modelexport.create_optimized_exporter"),
        ]
        
        import_results = {}
        
        for name, import_path in import_tests:
            try:
                # Test import
                parts = import_path.split('.')
                obj = modelexport
                for part in parts[1:]:
                    obj = getattr(obj, part)
                
                import_results[name] = {
                    "success": True,
                    "type": str(type(obj)),
                    "callable": callable(obj)
                }
                print(f"    ✅ {name}: {type(obj).__name__}")
                
            except Exception as e:
                import_results[name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ {name}: {e}")
        
        # Summary
        successful_imports = sum(1 for r in import_results.values() if r["success"])
        total_imports = len(import_results)
        
        print(f"\\n    📊 Import Success: {successful_imports}/{total_imports}")
        
        return import_results
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        
        print("\\n🚨 Testing Error Handling...")
        
        error_tests = {}
        
        # Test 1: Invalid model
        try:
            result = modelexport.export_model(
                "not_a_model",  # Invalid model
                torch.randn(1, 3, 224, 224),
                "invalid.onnx"
            )
            error_tests["invalid_model"] = {"expected_error": True, "got_error": False}
        except Exception as e:
            error_tests["invalid_model"] = {"expected_error": True, "got_error": True, "error": str(e)}
            print(f"    ✅ Invalid model correctly rejected: {type(e).__name__}")
        
        # Test 2: Invalid output path
        try:
            model = AutoModel.from_pretrained("microsoft/resnet-50")
            result = modelexport.export_model(
                model,
                torch.randn(1, 3, 224, 224),
                "/invalid/path/that/does/not/exist/model.onnx"
            )
            error_tests["invalid_path"] = {"expected_error": True, "got_error": False}
        except Exception as e:
            error_tests["invalid_path"] = {"expected_error": True, "got_error": True, "error": str(e)}
            print(f"    ✅ Invalid path correctly handled: {type(e).__name__}")
        
        # Test 3: Invalid strategy
        try:
            model = AutoModel.from_pretrained("microsoft/resnet-50")
            exporter = modelexport.UnifiedExporter(strategy="invalid_strategy")
            result = exporter.export(
                model,
                torch.randn(1, 3, 224, 224),
                self.output_dir / "test.onnx"
            )
            error_tests["invalid_strategy"] = {"expected_error": True, "got_error": False}
        except Exception as e:
            error_tests["invalid_strategy"] = {"expected_error": True, "got_error": True, "error": str(e)}
            print(f"    ✅ Invalid strategy correctly rejected: {type(e).__name__}")
        
        return error_tests
    
    def test_performance_expectations(self):
        """Test that performance meets expectations."""
        
        print("\\n⚡ Testing Performance Expectations...")
        
        model = AutoModel.from_pretrained("microsoft/resnet-50")
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        # Expected performance based on benchmarks
        expected_times = {
            "usage_based": 3.0,  # Should be around 2.5s
            "htp": 6.0,          # Should be around 5-6s
        }
        
        performance_results = {}
        
        for strategy, expected_time in expected_times.items():
            print(f"    🏃 Testing {strategy} performance...")
            
            try:
                output_path = self.output_dir / f"perf_{strategy}.onnx"
                
                start_time = time.time()
                report = modelexport.export_model(
                    model, inputs, output_path,
                    strategy=strategy,
                    optimize=True
                )
                actual_time = time.time() - start_time
                
                meets_expectation = actual_time <= expected_time * 1.5  # 50% tolerance
                
                performance_results[strategy] = {
                    "success": report["summary"]["success"],
                    "actual_time": actual_time,
                    "expected_time": expected_time,
                    "meets_expectation": meets_expectation,
                    "performance_ratio": actual_time / expected_time
                }
                
                status = "✅" if meets_expectation else "⚠️"
                print(f"      {status} {actual_time:.3f}s (expected <{expected_time:.1f}s)")
                
            except Exception as e:
                performance_results[strategy] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"      ❌ Failed: {e}")
        
        return performance_results
    
    def test_documentation_examples(self):
        """Test that documentation examples work correctly."""
        
        print("\\n📚 Testing Documentation Examples...")
        
        doc_tests = {}
        
        # Test 1: Quick start example from __init__.py
        try:
            model = AutoModel.from_pretrained("microsoft/resnet-50")
            model.eval()
            
            # Example from documentation
            report = modelexport.export_model(
                model,
                torch.randn(1, 3, 224, 224),
                self.output_dir / "doc_example_1.onnx"
            )
            print(f"Exported using {report['summary']['final_strategy']} strategy")
            
            doc_tests["quick_start"] = {
                "success": report["summary"]["success"],
                "strategy": report["summary"]["final_strategy"]
            }
            print(f"    ✅ Quick start example works")
            
        except Exception as e:
            doc_tests["quick_start"] = {"success": False, "error": str(e)}
            print(f"    ❌ Quick start example failed: {e}")
        
        # Test 2: Advanced usage example
        try:
            model = AutoModel.from_pretrained("microsoft/resnet-50")
            model.eval()
            
            # Advanced usage
            exporter = modelexport.UnifiedExporter(
                strategy="auto",
                enable_optimizations=True,
                verbose=False
            )
            
            report = exporter.export(
                model,
                torch.randn(1, 3, 224, 224),
                self.output_dir / "doc_example_2.onnx"
            )
            
            doc_tests["advanced_usage"] = {
                "success": report["summary"]["success"],
                "strategy": report["summary"]["final_strategy"]
            }
            print(f"    ✅ Advanced usage example works")
            
        except Exception as e:
            doc_tests["advanced_usage"] = {"success": False, "error": str(e)}
            print(f"    ❌ Advanced usage example failed: {e}")
        
        return doc_tests
    
    def run_production_readiness_test(self):
        """Run comprehensive production readiness test."""
        
        print("\\n🚀 Starting Production Readiness Testing...")
        
        test_results = {
            "iteration": 19,
            "test_type": "production_readiness",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "package_version": modelexport.__version__,
            "tests": {}
        }
        
        print("\\n" + "="*60)
        print("🏭 PRODUCTION READINESS TESTING")
        print("="*60)
        
        # Run all tests
        test_results["tests"]["simple_api"] = self.test_simple_api_usage()
        test_results["tests"]["package_imports"] = self.test_package_imports()
        test_results["tests"]["error_handling"] = self.test_error_handling()
        test_results["tests"]["performance"] = self.test_performance_expectations()
        test_results["tests"]["documentation"] = self.test_documentation_examples()
        
        # Generate summary
        test_results["summary"] = self._generate_production_summary(test_results["tests"])
        
        # Save results
        report_file = self.output_dir / "production_readiness_report.json"
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Print summary
        self._print_production_summary(test_results)
        
        return test_results
    
    def _generate_production_summary(self, tests):
        """Generate production readiness summary."""
        
        summary = {
            "overall_readiness": "unknown",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "score": 0,
            "max_score": 0
        }
        
        # Analyze simple API
        if tests["simple_api"]["success"]:
            summary["score"] += 20
            summary["recommendations"].append("✅ Simple API working correctly")
        else:
            summary["critical_issues"].append("❌ Simple API not working")
        summary["max_score"] += 20
        
        # Analyze imports
        import_results = tests["package_imports"]
        successful_imports = sum(1 for r in import_results.values() if r["success"])
        total_imports = len(import_results)
        
        if successful_imports == total_imports:
            summary["score"] += 20
            summary["recommendations"].append("✅ All package imports working")
        else:
            summary["warnings"].append(f"⚠️ {total_imports - successful_imports} import failures")
            summary["score"] += int(20 * successful_imports / total_imports)
        summary["max_score"] += 20
        
        # Analyze error handling
        error_results = tests["error_handling"]
        proper_errors = sum(1 for r in error_results.values() if r.get("expected_error") and r.get("got_error"))
        total_error_tests = len(error_results)
        
        if proper_errors == total_error_tests:
            summary["score"] += 15
            summary["recommendations"].append("✅ Error handling robust")
        else:
            summary["warnings"].append("⚠️ Some error cases not handled properly")
            summary["score"] += int(15 * proper_errors / total_error_tests)
        summary["max_score"] += 15
        
        # Analyze performance
        perf_results = tests["performance"]
        good_performance = sum(1 for r in perf_results.values() if r.get("meets_expectation", False))
        total_perf_tests = len(perf_results)
        
        if good_performance == total_perf_tests:
            summary["score"] += 25
            summary["recommendations"].append("✅ Performance meets expectations")
        else:
            summary["warnings"].append("⚠️ Some performance expectations not met")
            summary["score"] += int(25 * good_performance / total_perf_tests)
        summary["max_score"] += 25
        
        # Analyze documentation
        doc_results = tests["documentation"]
        working_examples = sum(1 for r in doc_results.values() if r.get("success", False))
        total_doc_tests = len(doc_results)
        
        if working_examples == total_doc_tests:
            summary["score"] += 20
            summary["recommendations"].append("✅ Documentation examples working")
        else:
            summary["warnings"].append("⚠️ Some documentation examples broken")
            summary["score"] += int(20 * working_examples / total_doc_tests)
        summary["max_score"] += 20
        
        # Determine overall readiness
        score_percentage = (summary["score"] / summary["max_score"]) * 100
        
        if score_percentage >= 95:
            summary["overall_readiness"] = "PRODUCTION_READY"
        elif score_percentage >= 85:
            summary["overall_readiness"] = "READY_WITH_MINOR_ISSUES"
        elif score_percentage >= 70:
            summary["overall_readiness"] = "NEEDS_IMPROVEMENTS"
        else:
            summary["overall_readiness"] = "NOT_READY"
        
        return summary
    
    def _print_production_summary(self, test_results):
        """Print production readiness summary."""
        
        summary = test_results["summary"]
        
        print(f"\\n{'='*60}")
        print("🏭 PRODUCTION READINESS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\\n🎯 Overall Assessment:")
        print(f"   Status: {summary['overall_readiness']}")
        print(f"   Score: {summary['score']}/{summary['max_score']} ({(summary['score']/summary['max_score'])*100:.1f}%)")
        print(f"   Package Version: {test_results['package_version']}")
        
        if summary["critical_issues"]:
            print(f"\\n🚨 Critical Issues:")
            for issue in summary["critical_issues"]:
                print(f"   {issue}")
        
        if summary["warnings"]:
            print(f"\\n⚠️ Warnings:")
            for warning in summary["warnings"]:
                print(f"   {warning}")
        
        print(f"\\n✅ Status:")
        for rec in summary["recommendations"]:
            print(f"   {rec}")
        
        # Deployment recommendation
        if summary["overall_readiness"] == "PRODUCTION_READY":
            print(f"\\n🚀 RECOMMENDATION: Ready for production deployment!")
        elif summary["overall_readiness"] == "READY_WITH_MINOR_ISSUES":
            print(f"\\n🔧 RECOMMENDATION: Ready for production with minor fixes")
        else:
            print(f"\\n⚠️ RECOMMENDATION: Requires fixes before production")


def main():
    """Run production readiness testing."""
    
    tester = ProductionReadinessTester()
    results = tester.run_production_readiness_test()
    
    print(f"\\n✅ Production Readiness Testing completed!")
    print(f"📁 Report saved to: {tester.output_dir}/production_readiness_report.json")
    
    return 0


if __name__ == '__main__':
    exit(main())