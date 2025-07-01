#!/usr/bin/env python3
"""
Ground Truth Analysis for BERT-tiny
===================================

This script creates the definitive ground truth of what hierarchy-preserving 
ONNX export should produce for prajjwal1/bert-tiny, based on:

1. Requirements analysis (REQUIREMENTS.md)
2. PyTorch's actual module hierarchy
3. Expected tag format: /BertModel/BertEncoder/BertLayer.0/BertAttention

This serves as the reference implementation to verify what we're actually chasing.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class BertTinyGroundTruth:
    """Create ground truth analysis for BERT-tiny hierarchy tagging."""
    
    def __init__(self):
        self.model_name = "prajjwal1/bert-tiny"
        self.model = None
        self.tokenizer = None
        self.ground_truth = {}
        
    def load_model(self):
        """Load the BERT-tiny model and tokenizer."""
        print(f"📦 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        print("✅ Model loaded successfully")
        
    def analyze_model_hierarchy(self) -> Dict[str, Any]:
        """Analyze the complete module hierarchy of BERT-tiny."""
        
        print("\n🔍 Analyzing model hierarchy...")
        
        hierarchy_analysis = {
            'total_modules': 0,
            'module_tree': {},
            'by_type': defaultdict(list),
            'tag_mapping': {},  # module_name -> expected_tag
            'torch_nn_modules': [],
            'hf_modules': [],
            'filtered_modules': []  # What should be filtered out
        }
        
        # Analyze all modules
        for name, module in self.model.named_modules():
            module_info = {
                'name': name,
                'class': type(module).__name__,
                'module_path': type(module).__module__,
                'level': name.count('.') if name else 0,
                'children': []
            }
            
            # Classify module type
            module_path = type(module).__module__
            if module_path.startswith('torch.nn'):
                module_info['category'] = 'torch.nn'
                hierarchy_analysis['torch_nn_modules'].append(module_info)
            elif 'transformers' in module_path:
                module_info['category'] = 'huggingface'
                hierarchy_analysis['hf_modules'].append(module_info)
            else:
                module_info['category'] = 'other'
            
            # Determine if this module should be filtered
            should_filter = self._should_filter_module(module_info)
            if should_filter:
                hierarchy_analysis['filtered_modules'].append(module_info)
            
            # Generate expected hierarchy tag
            expected_tag = self._generate_expected_tag(name, module_info, should_filter)
            module_info['expected_tag'] = expected_tag
            
            hierarchy_analysis['by_type'][type(module).__name__].append(module_info)
            hierarchy_analysis['tag_mapping'][name] = expected_tag
            hierarchy_analysis['total_modules'] += 1
        
        return hierarchy_analysis
    
    def _should_filter_module(self, module_info: Dict[str, Any]) -> bool:
        """Determine if a module should be filtered according to MUST-002."""
        
        # MUST-002: Filter most torch.nn modules except whitelist
        if module_info['category'] == 'torch.nn':
            # Default exceptions (from REQUIREMENTS.md)
            whitelist = {'LayerNorm', 'Embedding', 'BatchNorm1d', 'BatchNorm2d', 'GroupNorm', 'InstanceNorm'}
            return module_info['class'] not in whitelist
        
        # Don't filter HuggingFace or other modules
        return False
    
    def _generate_expected_tag(self, module_name: str, module_info: Dict[str, Any], should_filter: bool) -> str:
        """Generate the expected hierarchy tag for a module according to R12."""
        
        if should_filter:
            return ""  # Filtered modules get empty tags
        
        if not module_name:  # Root module
            return "/" + module_info['class']
        
        # Convert PyTorch module path to hierarchy tag
        # Example: "bert.encoder.layer.0.attention.self" -> "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention"
        
        parts = module_name.split('.')
        tag_parts = ["/"]
        
        # Special handling for BERT structure
        for i, part in enumerate(parts):
            if part == 'bert':
                tag_parts.append('BertModel')
            elif part == 'encoder':
                tag_parts.append('BertEncoder')
            elif part == 'layer':
                # Next part should be a number
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    tag_parts.append(f'BertLayer.{parts[i + 1]}')
                    # Skip the number part in next iteration
                    parts[i + 1] = None
            elif part == 'attention':
                tag_parts.append('BertAttention')
            elif part == 'self':
                tag_parts.append('BertSelfAttention')
            elif part == 'output':
                tag_parts.append('BertOutput')
            elif part == 'intermediate':
                tag_parts.append('BertIntermediate')
            elif part == 'embeddings':
                tag_parts.append('BertEmbeddings')
            elif part == 'pooler':
                tag_parts.append('BertPooler')
            elif part and part != 'None':  # Skip None placeholders
                # Convert to PascalCase if needed
                tag_parts.append(part.title() if part.islower() else part)
        
        # Join with forward slashes
        tag = '/'.join(tag_parts)
        
        # Clean up any double slashes
        while '//' in tag:
            tag = tag.replace('//', '/')
        
        return tag
    
    def analyze_expected_onnx_operations(self) -> Dict[str, Any]:
        """Analyze what ONNX operations we expect and their tagging."""
        
        print("\n⚙️ Analyzing expected ONNX operations...")
        
        # Create dummy inputs for analysis
        text = "Hello world"
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Expected ONNX operation types for BERT-tiny
        expected_ops = {
            'MatMul': 'Matrix multiplication operations (attention, dense layers)',
            'Add': 'Addition operations (residual connections, bias)',
            'Div': 'Division operations (attention scaling)',
            'Softmax': 'Attention probability computation',
            'Mul': 'Multiplication operations (attention masking)',
            'Sub': 'Subtraction operations (attention masking)',
            'Slice': 'Tensor slicing operations (positional embeddings)',
            'Shape': 'Shape operations (dynamic reshaping)',
            'Gather': 'Embedding lookups',
            'Unsqueeze': 'Dimension expansion',
            'Concat': 'Tensor concatenation',
            'Transpose': 'Matrix transposition',
            'Reshape': 'Tensor reshaping',
            'Cast': 'Type casting operations',
            'Equal': 'Equality comparisons (padding masks)',
            'Where': 'Conditional operations',
            'ReduceSum': 'Reduction operations',
            'Sqrt': 'Square root (layer norm)',
            'Pow': 'Power operations (layer norm)',
            'Erf': 'Error function (GELU activation)'
        }
        
        # Expected tagging patterns
        tagging_expectations = {
            'critical_ops': ['MatMul', 'Add', 'Softmax', 'Mul'],  # Must be tagged
            'support_ops': ['Shape', 'Unsqueeze', 'Cast'],  # Context-dependent tagging OK
            'preprocessing_ops': ['Slice', 'Gather'],  # Empty tags acceptable for some
        }
        
        return {
            'expected_operation_types': expected_ops,
            'tagging_expectations': tagging_expectations,
            'input_shape': input_ids.shape,
            'attention_mask_shape': attention_mask.shape
        }
    
    def create_verification_baseline(self) -> Dict[str, Any]:
        """Create the definitive baseline for what BERT-tiny export should produce."""
        
        print("\n📋 Creating verification baseline...")
        
        # Key requirements from REQUIREMENTS.md
        requirements_checklist = {
            'R7_topology_preservation': {
                'description': 'Export must preserve IDENTICAL graph topology to baseline',
                'verification': 'Compare with standard torch.onnx.export()',
                'status': 'CRITICAL'
            },
            'R10_operation_attribution': {
                'description': 'Map every ONNX operation to source HF module class',
                'verification': 'Check hf_hierarchy_tag attributes on ONNX nodes',
                'status': 'CRITICAL'
            },
            'R12_instance_specific_paths': {
                'description': 'Preserve instance numbers: BertLayer.0 vs BertLayer.1',
                'verification': 'Look for .0, .1 in hierarchy tags',
                'status': 'CRITICAL'
            },
            'MUST_001_no_hardcoded_logic': {
                'description': 'No hardcoded model architectures, node names, etc.',
                'verification': 'Implementation uses universal PyTorch principles',
                'status': 'CARDINAL_RULE'
            },
            'MUST_002_torch_nn_filtering': {
                'description': 'Filter most torch.nn except whitelist',
                'verification': 'Most torch.nn modules should have empty tags',
                'status': 'CARDINAL_RULE'
            }
        }
        
        # Expected tag format examples
        expected_tag_examples = [
            '/BertModel',  # Root
            '/BertModel/BertEmbeddings',  # Embeddings
            '/BertModel/BertEncoder',  # Encoder
            '/BertModel/BertEncoder/BertLayer.0',  # First layer
            '/BertModel/BertEncoder/BertLayer.0/BertAttention',  # Attention module
            '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention',  # Self attention
            '/BertModel/BertEncoder/BertLayer.1',  # Second layer
            '/BertModel/BertPooler'  # Pooler
        ]
        
        # Performance expectations
        performance_expectations = {
            'export_time': '<10 seconds',
            'topology_preservation': '100%',
            'tagged_operations': '>80%',  # Most operations should be tagged
            'empty_tags': '<20%',  # Some support operations may have empty tags
            'contamination_reduction': '>50%'  # Based on achievements
        }
        
        return {
            'model_name': self.model_name,
            'requirements_checklist': requirements_checklist,
            'expected_tag_examples': expected_tag_examples,
            'performance_expectations': performance_expectations,
            'verification_strategy': {
                'step_1': 'Export with working HTP implementation',
                'step_2': 'Analyze ONNX model tags',
                'step_3': 'Verify against this ground truth',
                'step_4': 'Check requirements compliance'
            }
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive ground truth report."""
        
        hierarchy = self.analyze_model_hierarchy()
        operations = self.analyze_expected_onnx_operations()
        baseline = self.create_verification_baseline()
        
        report = []
        report.append("=" * 80)
        report.append("🎯 BERT-TINY GROUND TRUTH ANALYSIS")
        report.append("=" * 80)
        report.append(f"\nModel: {self.model_name}")
        report.append(f"Total modules: {hierarchy['total_modules']}")
        
        # Module distribution
        report.append("\n📊 MODULE DISTRIBUTION:")
        report.append(f"   HuggingFace modules: {len(hierarchy['hf_modules'])}")
        report.append(f"   PyTorch nn modules: {len(hierarchy['torch_nn_modules'])}")
        report.append(f"   Modules to filter (MUST-002): {len(hierarchy['filtered_modules'])}")
        
        # Expected hierarchy structure
        report.append("\n🏗️ EXPECTED HIERARCHY STRUCTURE:")
        
        # Show key modules and their expected tags
        key_modules = [name for name in hierarchy['tag_mapping'].keys() 
                      if hierarchy['tag_mapping'][name] and 
                      not hierarchy['tag_mapping'][name].startswith('')]
        
        for module_name in sorted(key_modules)[:15]:  # Show first 15
            expected_tag = hierarchy['tag_mapping'][module_name]
            if expected_tag:
                report.append(f"   {module_name:30s} → {expected_tag}")
        
        if len(key_modules) > 15:
            report.append(f"   ... and {len(key_modules) - 15} more modules")
        
        # Requirements verification
        report.append("\n✅ REQUIREMENTS VERIFICATION:")
        for req_id, req_info in baseline['requirements_checklist'].items():
            status = req_info['status']
            desc = req_info['description']
            report.append(f"   {req_id}: {status}")
            report.append(f"      {desc}")
        
        # Expected ONNX operations
        report.append("\n⚙️ EXPECTED ONNX OPERATIONS:")
        critical_ops = operations['tagging_expectations']['critical_ops']
        report.append(f"   Critical operations (must be tagged): {', '.join(critical_ops)}")
        
        support_ops = operations['tagging_expectations']['support_ops']
        report.append(f"   Support operations (context-dependent): {', '.join(support_ops)}")
        
        # Expected tag format
        report.append("\n🏷️ EXPECTED TAG FORMAT (R12):")
        for example in baseline['expected_tag_examples'][:8]:
            report.append(f"   {example}")
        
        # Performance expectations
        report.append("\n🚀 PERFORMANCE EXPECTATIONS:")
        for metric, expectation in baseline['performance_expectations'].items():
            report.append(f"   {metric}: {expectation}")
        
        # Key insights
        report.append("\n💡 KEY INSIGHTS:")
        report.append("   ✅ BERT-tiny has 2 transformer layers (BertLayer.0, BertLayer.1)")
        report.append("   ✅ Each layer has attention + feed-forward components")
        report.append("   ✅ Most torch.nn modules should be filtered per MUST-002")
        report.append("   ✅ Instance numbers must be preserved in tags (R12)")
        report.append("   ✅ Full hierarchy paths required (not just instance names)")
        
        # Verification strategy
        report.append("\n🔍 VERIFICATION STRATEGY:")
        for step, action in baseline['verification_strategy'].items():
            report.append(f"   {step}: {action}")
        
        return "\n".join(report)
    
    def save_ground_truth(self, output_path: str):
        """Save the complete ground truth analysis to JSON."""
        
        hierarchy = self.analyze_model_hierarchy()
        operations = self.analyze_expected_onnx_operations()
        baseline = self.create_verification_baseline()
        
        ground_truth_data = {
            'metadata': {
                'model_name': self.model_name,
                'created_by': 'ground_truth_bert_tiny.py',
                'purpose': 'Definitive reference for BERT-tiny hierarchy export verification'
            },
            'hierarchy_analysis': hierarchy,
            'onnx_operations': operations,
            'verification_baseline': baseline,
            'summary': {
                'total_modules': hierarchy['total_modules'],
                'hf_modules': len(hierarchy['hf_modules']),
                'filtered_modules': len(hierarchy['filtered_modules']),
                'expected_tags': len([tag for tag in hierarchy['tag_mapping'].values() if tag])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        
        print(f"💾 Ground truth saved to: {output_path}")


def main():
    """Generate the definitive ground truth for BERT-tiny."""
    
    print("🎯 GENERATING BERT-TINY GROUND TRUTH")
    print("=" * 50)
    
    # Create ground truth analyzer
    analyzer = BertTinyGroundTruth()
    
    # Load model
    analyzer.load_model()
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save detailed ground truth
    output_path = "temp/bert_tiny_ground_truth.json"
    Path("temp").mkdir(exist_ok=True)
    analyzer.save_ground_truth(output_path)
    
    print("\n" + "=" * 80)
    print("✅ GROUND TRUTH GENERATION COMPLETE")
    print("=" * 80)
    print("\nThis ground truth serves as the definitive reference for:")
    print("1. What BERT-tiny hierarchy export should produce")
    print("2. Requirements verification checklist")
    print("3. Expected tag format and performance")
    print("4. Verification strategy")
    print("\nUse this to verify any HTP implementation against the requirements!")


if __name__ == "__main__":
    main()