#!/usr/bin/env python3
"""
HuggingFace Model Baseline Testing Script for Iteration 16

Tests specific HuggingFace models with all strategies to establish
baseline performance metrics and compatibility patterns.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel

from modelexport.strategies.fx import FXHierarchyExporter
from modelexport.strategies.htp import HTPHierarchyExporter
from modelexport.strategies.usage_based import UsageBasedExporter


class HuggingFaceModelTester:
    """Test HuggingFace models with all available strategies."""
    
    def __init__(self, output_dir: str = "temp/iteration_16"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all strategies
        self.strategies = {
            'fx': FXHierarchyExporter(),
            'htp': HTPHierarchyExporter(),
            'usage_based': UsageBasedExporter()
        }
        
        # Test models from iteration requirements
        self.test_models = {
            'microsoft/resnet-50': {
                'type': 'vision',
                'input_shape': (3, 224, 224),
                'expected_fx_compatible': False,  # HF ResNet has control flow
                'description': 'HuggingFace ResNet-50 with control flow'
            },
            'facebook/sam-vit-base': {
                'type': 'vision', 
                'input_shape': (3, 1024, 1024),
                'expected_fx_compatible': False,  # Complex vision transformer
                'description': 'Segment Anything Model (SAM) vision transformer'
            }
        }
        
        self.results = {}
    
    def prepare_model_inputs(self, model_name: str, model: torch.nn.Module) -> torch.Tensor:
        """Prepare appropriate inputs for the model."""
        config = self.test_models[model_name]
        
        if config['type'] == 'vision':
            # Create dummy image tensor
            batch_size = 1
            channels, height, width = config['input_shape']
            return torch.randn(batch_size, channels, height, width)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    def test_model_with_strategy(self, model_name: str, strategy_name: str) -> dict[str, Any]:
        """Test a single model with a single strategy."""
        print(f"\\n🧪 Testing {model_name} with {strategy_name} strategy...")
        
        result = {
            'model_name': model_name,
            'strategy': strategy_name,
            'success': False,
            'error': None,
            'export_time': 0.0,
            'export_stats': {},
            'compatibility': {},
            'file_size': 0,
            'notes': []
        }
        
        try:
            # Load model
            print(f"📥 Loading {model_name}...")
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            # Prepare inputs
            inputs = self.prepare_model_inputs(model_name, model)
            
            # Prepare output path
            safe_model_name = model_name.replace('/', '_')
            output_path = self.output_dir / f"{safe_model_name}_{strategy_name}.onnx"
            
            # Test export
            strategy = self.strategies[strategy_name]
            start_time = time.time()
            
            export_result = strategy.export(model, inputs, str(output_path))
            
            export_time = time.time() - start_time
            
            # Collect results
            result.update({
                'success': True,
                'export_time': export_time,
                'export_stats': export_result.get('stats', {}),
                'compatibility': export_result.get('compatibility', {}),
                'file_size': output_path.stat().st_size if output_path.exists() else 0
            })
            
            # Strategy-specific analysis
            if strategy_name == 'fx':
                fx_stats = export_result.get('fx_graph_stats', {})
                result['notes'].extend([
                    f"FX compatibility: {export_result.get('fx_compatible', 'unknown')}",
                    f"Coverage ratio: {fx_stats.get('coverage_ratio', 0):.3f}",
                    f"Fallback used: {export_result.get('fallback_used', False)}"
                ])
            elif strategy_name == 'htp':
                result['notes'].extend([
                    f"Operations traced: {export_result.get('traced_operations', 0)}",
                    f"Hook registrations: {export_result.get('hook_count', 0)}"
                ])
            elif strategy_name == 'usage_based':
                result['notes'].extend([
                    f"Module usage tracked: {export_result.get('module_usage_count', 0)}"
                ])
            
            print(f"✅ {strategy_name}: Export successful ({export_time:.2f}s)")
            
        except Exception as e:
            result.update({
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            print(f"❌ {strategy_name}: Export failed - {e}")
        
        return result
    
    def test_all_models(self) -> dict[str, list[dict[str, Any]]]:
        """Test all models with all strategies."""
        print(f"🚀 Starting HuggingFace model baseline testing...")
        print(f"📂 Output directory: {self.output_dir}")
        
        for model_name in self.test_models:
            print(f"\\n{'='*60}")
            print(f"🔬 Testing model: {model_name}")
            print(f"📝 Description: {self.test_models[model_name]['description']}")
            
            model_results = []
            
            for strategy_name in self.strategies:
                result = self.test_model_with_strategy(model_name, strategy_name)
                model_results.append(result)
            
            self.results[model_name] = model_results
        
        return self.results
    
    def analyze_results(self) -> dict[str, Any]:
        """Analyze test results and create summary."""
        analysis = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'strategy_performance': {},
            'model_compatibility': {},
            'performance_metrics': {},
            'key_findings': []
        }
        
        for model_name, model_results in self.results.items():
            analysis['total_tests'] += len(model_results)
            
            for result in model_results:
                strategy = result['strategy']
                
                if strategy not in analysis['strategy_performance']:
                    analysis['strategy_performance'][strategy] = {
                        'success_count': 0,
                        'failure_count': 0,
                        'avg_export_time': 0,
                        'total_export_time': 0
                    }
                
                if result['success']:
                    analysis['successful_tests'] += 1
                    analysis['strategy_performance'][strategy]['success_count'] += 1
                    analysis['strategy_performance'][strategy]['total_export_time'] += result['export_time']
                else:
                    analysis['failed_tests'] += 1
                    analysis['strategy_performance'][strategy]['failure_count'] += 1
        
        # Calculate averages
        for strategy_data in analysis['strategy_performance'].values():
            if strategy_data['success_count'] > 0:
                strategy_data['avg_export_time'] = (
                    strategy_data['total_export_time'] / strategy_data['success_count']
                )
        
        # Model compatibility analysis
        for model_name, model_results in self.results.items():
            compatibility = {}
            for result in model_results:
                strategy = result['strategy']
                compatibility[strategy] = {
                    'compatible': result['success'],
                    'notes': result['notes']
                }
            analysis['model_compatibility'][model_name] = compatibility
        
        # Key findings
        analysis['key_findings'].extend([
            f"Overall success rate: {analysis['successful_tests']}/{analysis['total_tests']} ({100*analysis['successful_tests']/analysis['total_tests']:.1f}%)",
            f"Best performing strategy: {max(analysis['strategy_performance'].keys(), key=lambda s: analysis['strategy_performance'][s]['success_count'])}",
            f"HF ResNet FX compatibility: {'✅' if any(r['success'] and r['strategy'] == 'fx' for r in self.results.get('microsoft/resnet-50', [])) else '❌'}",
            f"SAM FX compatibility: {'✅' if any(r['success'] and r['strategy'] == 'fx' for r in self.results.get('facebook/sam-vit-base', [])) else '❌'}"
        ])
        
        return analysis
    
    def save_results(self, filename: str = "hf_baseline_results.json"):
        """Save detailed results to JSON file."""
        output_file = self.output_dir / filename
        
        full_report = {
            'iteration': 16,
            'test_type': 'huggingface_baseline',
            'models_tested': list(self.test_models.keys()),
            'strategies_tested': list(self.strategies.keys()),
            'detailed_results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\\n📊 Detailed results saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a summary of the test results."""
        analysis = self.analyze_results()
        
        print(f"\\n{'='*60}")
        print("📊 ITERATION 16 BASELINE TESTING SUMMARY")
        print(f"{'='*60}")
        
        print(f"\\n🎯 Overall Results:")
        print(f"   Total tests: {analysis['total_tests']}")
        print(f"   Successful: {analysis['successful_tests']}")
        print(f"   Failed: {analysis['failed_tests']}")
        print(f"   Success rate: {100*analysis['successful_tests']/analysis['total_tests']:.1f}%")
        
        print(f"\\n🔧 Strategy Performance:")
        for strategy, perf in analysis['strategy_performance'].items():
            success_rate = 100 * perf['success_count'] / (perf['success_count'] + perf['failure_count'])
            print(f"   {strategy.upper()}: {perf['success_count']}/{perf['success_count'] + perf['failure_count']} success ({success_rate:.1f}%)")
            if perf['success_count'] > 0:
                print(f"      Avg export time: {perf['avg_export_time']:.2f}s")
        
        print(f"\\n🏆 Key Findings:")
        for finding in analysis['key_findings']:
            print(f"   • {finding}")


def main():
    """Run the HuggingFace model baseline testing."""
    tester = HuggingFaceModelTester()
    
    try:
        # Run all tests
        results = tester.test_all_models()
        
        # Print summary
        tester.print_summary()
        
        # Save results
        tester.save_results()
        
        print(f"\\n✅ Iteration 16 baseline testing completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Testing failed: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())