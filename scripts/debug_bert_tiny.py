#!/usr/bin/env python3
"""
BERT-tiny Debugging Script
=========================

Comprehensive debugging script for bert-tiny model export and analysis.
Uses model export config and generates proper input data.

Features:
- Loads bert-tiny model and config
- Generates random input data based on config specs
- Tests all export strategies (enhanced_semantic, htp, usage_based, fx_graph)
- Validates ONNX output
- Provides detailed analysis and statistics
- Debug mode for troubleshooting
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import onnx
import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.core import tag_utils

# Import our exporters
from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter
from modelexport.strategies.fx.fx_hierarchy_exporter import FXHierarchyExporter
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter
from modelexport.strategies.usage_based.usage_based_exporter import UsageBasedExporter


class BertTinyDebugger:
    """Comprehensive BERT-tiny debugging tool."""
    
    def __init__(self, config_path: str = "bert_tiny_optimized_config.json", verbose: bool = True):
        """
        Initialize debugger.
        
        Args:
            config_path: Path to the export configuration file
            verbose: Enable verbose output
        """
        self.model_name = "prajjwal1/bert-tiny"
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.output_dir = Path("temp/bert_tiny_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.dummy_inputs = None
        
        # Results storage
        self.results = {}
        
    def _load_config(self) -> dict[str, Any]:
        """Load export configuration."""
        if not self.config_path.exists():
            print(f"❌ Config file not found: {self.config_path}")
            print("Available configs in project root:")
            for config_file in Path(".").glob("*config*.json"):
                print(f"   {config_file}")
            sys.exit(1)
            
        with open(self.config_path) as f:
            config = json.load(f)
            
        if self.verbose:
            print(f"📄 Loaded config from: {self.config_path}")
            print(f"   Opset version: {config.get('opset_version', 'default')}")
            print(f"   Input specs: {list(config.get('input_specs', {}).keys())}")
            
        return config
    
    def _load_model(self) -> None:
        """Load BERT-tiny model and tokenizer."""
        if self.verbose:
            print(f"🤖 Loading {self.model_name}...")
            
        try:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.eval()
            
            if self.verbose:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"   Model loaded: {type(self.model).__name__}")
                print(f"   Total parameters: {total_params:,}")
                print(f"   Model device: {next(self.model.parameters()).device}")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def _generate_input_data(self) -> dict[str, torch.Tensor]:
        """Generate random input data based on config specs."""
        if self.verbose:
            print("🎲 Generating input data...")
            
        inputs = {}
        input_specs = self.config.get('input_specs', {})
        
        for input_name, spec in input_specs.items():
            dtype_str = spec.get('dtype', 'float')
            dtype = torch.long if dtype_str == 'long' else torch.float32
            
            # Use dynamic axes for shape if available
            dynamic_axes = self.config.get('dynamic_axes', {})
            if input_name in dynamic_axes:
                # Default shape for dynamic inputs: batch_size=1, sequence_length=16
                shape = [1, 16]
            else:
                shape = spec.get('shape', [1, 16])
            
            # Generate values within specified range
            if 'range' in spec:
                min_val, max_val = spec['range']
                if dtype == torch.long:
                    # For integer types, generate random integers in range
                    inputs[input_name] = torch.randint(
                        min_val, max_val + 1, shape, dtype=dtype
                    )
                else:
                    # For float types, generate random floats in range
                    inputs[input_name] = (
                        torch.rand(shape, dtype=dtype) * (max_val - min_val) + min_val
                    )
            else:
                # Default generation
                if dtype == torch.long:
                    inputs[input_name] = torch.randint(0, 1000, shape, dtype=dtype)
                else:
                    inputs[input_name] = torch.randn(shape, dtype=dtype)
        
        # Fallback to simple generation if no specs
        if not inputs:
            inputs = {
                'input_ids': torch.randint(0, 30522, (1, 16), dtype=torch.long),
                'attention_mask': torch.ones((1, 16), dtype=torch.long)
            }
        
        if self.verbose:
            print("   Generated inputs:")
            for name, tensor in inputs.items():
                print(f"     {name}: {tensor.shape} {tensor.dtype} "
                      f"[{tensor.min():.0f}..{tensor.max():.0f}]")
        
        self.dummy_inputs = inputs
        return inputs
    
    def _test_model_forward(self) -> bool:
        """Test model forward pass with generated inputs."""
        if self.verbose:
            print("🔍 Testing model forward pass...")
            
        try:
            with torch.no_grad():
                # Convert dict to args for model call
                if isinstance(self.dummy_inputs, dict):
                    outputs = self.model(**self.dummy_inputs)
                else:
                    outputs = self.model(*self.dummy_inputs)
                
            if self.verbose:
                if hasattr(outputs, 'last_hidden_state'):
                    print(f"   ✅ Forward pass successful")
                    print(f"   Output shape: {outputs.last_hidden_state.shape}")
                else:
                    print(f"   ✅ Forward pass successful")
                    print(f"   Output type: {type(outputs)}")
                    
            return True
            
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            return False
    
    def _export_with_strategy(self, strategy: str) -> dict[str, Any]:
        """Export model using specified strategy."""
        if self.verbose:
            print(f"\n🚀 Testing {strategy} export strategy...")
            
        start_time = time.time()
        output_path = self.output_dir / f"model_{strategy}.onnx"
        
        try:
            if strategy == 'enhanced_semantic':
                exporter = EnhancedSemanticExporter(verbose=self.verbose)
                
                # Convert dict inputs to tuple for enhanced semantic
                if isinstance(self.dummy_inputs, dict):
                    args = tuple(self.dummy_inputs.values())
                else:
                    args = self.dummy_inputs
                
                result = exporter.export(
                    model=self.model,
                    args=args,
                    output_path=str(output_path),
                    **{k: v for k, v in self.config.items() 
                       if k not in ['input_specs', 'optimization']}
                )
                
            elif strategy == 'htp':
                exporter = HierarchyExporter(strategy='htp')
                result = exporter.export(
                    model=self.model,
                    example_inputs=self.dummy_inputs,
                    output_path=str(output_path),
                    **{k: v for k, v in self.config.items() 
                       if k not in ['input_specs', 'optimization']}
                )
                
            elif strategy == 'usage_based':
                exporter = UsageBasedExporter()
                result = exporter.export(
                    model=self.model,
                    example_inputs=self.dummy_inputs,
                    output_path=str(output_path),
                    **{k: v for k, v in self.config.items() 
                       if k not in ['input_specs', 'optimization']}
                )
                
            elif strategy == 'fx_graph':
                exporter = FXHierarchyExporter()
                result = exporter.export(
                    model=self.model,
                    example_inputs=self.dummy_inputs,
                    output_path=str(output_path),
                    **{k: v for k, v in self.config.items() 
                       if k not in ['input_specs', 'optimization']}
                )
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            export_time = time.time() - start_time
            
            # Validate ONNX model
            if output_path.exists():
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                
                result.update({
                    'success': True,
                    'export_time': export_time,
                    'onnx_path': str(output_path),
                    'onnx_size_mb': output_path.stat().st_size / (1024 * 1024),
                    'total_onnx_nodes': len(onnx_model.graph.node)
                })
                
                if self.verbose:
                    print(f"   ✅ Export successful")
                    print(f"   Export time: {export_time:.2f}s")
                    print(f"   ONNX file: {output_path.name}")
                    print(f"   File size: {result['onnx_size_mb']:.2f} MB")
                    print(f"   ONNX nodes: {result['total_onnx_nodes']}")
                    
                    # Strategy-specific stats
                    if strategy == 'enhanced_semantic':
                        print(f"   HF mappings: {result.get('hf_module_mappings', 0)}")
                        print(f"   Coverage: {self._calculate_coverage(result):.1f}%")
                    elif strategy in ['htp', 'usage_based']:
                        tagged_ops = result.get('tagged_operations', 0)
                        total_ops = result.get('total_operations', 1)
                        print(f"   Tagged ops: {tagged_ops}/{total_ops} "
                              f"({tagged_ops/total_ops*100:.1f}%)")
                    elif strategy == 'fx_graph':
                        print(f"   FX nodes: {result.get('hierarchy_nodes', 0)}")
                        print(f"   Unique modules: {result.get('unique_modules', 0)}")
                
            else:
                result = {
                    'success': False,
                    'error': 'ONNX file not created',
                    'export_time': export_time
                }
                print(f"   ❌ ONNX file not created")
                
        except Exception as e:
            export_time = time.time() - start_time
            result = {
                'success': False,
                'error': str(e),
                'export_time': export_time
            }
            print(f"   ❌ Export failed: {e}")
            
        return result
    
    def _calculate_coverage(self, result: dict[str, Any]) -> float:
        """Calculate coverage percentage for enhanced semantic results."""
        total = result.get('total_onnx_nodes', 0)
        if total == 0:
            return 0.0
        
        mapped = result.get('hf_module_mappings', 0)
        inferred = result.get('operation_inferences', 0) 
        fallback = result.get('pattern_fallbacks', 0)
        
        return (mapped + inferred + fallback) / total * 100
    
    def _analyze_tags(self, strategy: str) -> dict[str, Any]:
        """Analyze hierarchy tags in exported model."""
        if self.verbose:
            print(f"📊 Analyzing {strategy} tags...")
            
        onnx_path = self.output_dir / f"model_{strategy}.onnx"
        
        if not onnx_path.exists():
            return {'error': 'ONNX file not found'}
        
        try:
            # Get tag statistics
            stats = tag_utils.get_tag_statistics(str(onnx_path))
            
            analysis = {
                'total_unique_tags': len(stats),
                'total_tagged_operations': sum(stats.values()),
                'tag_distribution': stats
            }
            
            if self.verbose and stats:
                print(f"   Unique tags: {len(stats)}")
                print(f"   Tagged operations: {sum(stats.values())}")
                print("   Top tags:")
                sorted_tags = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                for tag, count in sorted_tags[:5]:
                    print(f"     {tag}: {count}")
                if len(sorted_tags) > 5:
                    print(f"     ... and {len(sorted_tags) - 5} more")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_strategies(self) -> None:
        """Compare results across all strategies."""
        if self.verbose:
            print("\n📈 Strategy Comparison:")
            print("=" * 70)
            
        # Prepare comparison data
        strategies = ['enhanced_semantic', 'htp', 'usage_based', 'fx_graph']
        comparison_data = []
        
        for strategy in strategies:
            if strategy in self.results:
                result = self.results[strategy]
                if result.get('success', False):
                    comparison_data.append({
                        'strategy': strategy,
                        'export_time': result.get('export_time', 0),
                        'file_size': result.get('onnx_size_mb', 0),
                        'total_nodes': result.get('total_onnx_nodes', 0),
                        'coverage': self._calculate_coverage(result)
                    })
        
        if comparison_data:
            print(f"{'Strategy':<18} {'Time(s)':<8} {'Size(MB)':<10} {'Nodes':<8} {'Coverage':<10}")
            print("-" * 70)
            
            for data in comparison_data:
                print(f"{data['strategy']:<18} "
                      f"{data['export_time']:<8.2f} "
                      f"{data['file_size']:<10.2f} "
                      f"{data['total_nodes']:<8} "
                      f"{data['coverage']:<10.1f}%")
    
    def _generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        report_path = self.output_dir / "debug_report.json"
        
        summary = {
            'model_name': self.model_name,
            'config_path': str(self.config_path),
            'debug_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_class': type(self.model).__name__
            },
            'input_specs': self.config.get('input_specs', {}),
            'generated_inputs': {
                name: {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'min': float(tensor.min()),
                    'max': float(tensor.max())
                }
                for name, tensor in self.dummy_inputs.items()
            },
            'export_results': self.results,
            'output_directory': str(self.output_dir)
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        if self.verbose:
            print(f"\n📋 Summary report saved: {report_path}")
    
    def run_full_debug(self, strategies: list | None = None) -> bool:
        """Run complete debugging workflow."""
        if strategies is None:
            strategies = ['enhanced_semantic', 'htp', 'usage_based', 'fx_graph']
            
        print(f"🔧 BERT-tiny Debugging Script")
        print(f"=" * 50)
        
        # Step 1: Load model
        self._load_model()
        
        # Step 2: Generate input data
        self._generate_input_data()
        
        # Step 3: Test forward pass
        if not self._test_model_forward():
            print("❌ Model forward pass failed, aborting...")
            return False
        
        # Step 4: Test all export strategies
        success_count = 0
        for strategy in strategies:
            result = self._export_with_strategy(strategy)
            self.results[strategy] = result
            
            if result.get('success', False):
                success_count += 1
                
                # Analyze tags for successful exports
                tag_analysis = self._analyze_tags(strategy)
                self.results[strategy]['tag_analysis'] = tag_analysis
        
        # Step 5: Compare strategies
        if success_count > 1:
            self._compare_strategies()
        
        # Step 6: Generate summary report
        self._generate_summary_report()
        
        print(f"\n✅ Debugging complete!")
        print(f"   Successful exports: {success_count}/{len(strategies)}")
        print(f"   Output directory: {self.output_dir}")
        
        return success_count > 0


def main():
    """Main entry point for debugging script."""
    parser = argparse.ArgumentParser(
        description="BERT-tiny debugging script with config-based export"
    )
    parser.add_argument(
        '--config', 
        default='bert_tiny_optimized_config.json',
        help='Path to export configuration file'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['enhanced_semantic', 'htp', 'usage_based', 'fx_graph'],
        default=['enhanced_semantic', 'htp', 'usage_based'],
        help='Export strategies to test'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--output-dir',
        default='temp/bert_tiny_debug',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = BertTinyDebugger(
        config_path=args.config,
        verbose=not args.quiet
    )
    
    # Override output directory if specified
    if args.output_dir != 'temp/bert_tiny_debug':
        debugger.output_dir = Path(args.output_dir)
        debugger.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run debugging
    success = debugger.run_full_debug(args.strategies)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()