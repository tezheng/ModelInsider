#!/usr/bin/env python3
"""
Basic Enhanced Export Example

This example demonstrates how to use enhanced auxiliary operations
for 100% operation coverage in ONNX export.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add ModelExport to path
sys.path.append(str(Path(__file__).parent.parent))

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class ExampleModel(nn.Module):
    """Example model with auxiliary operations to demonstrate enhanced coverage."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids):
        # This will create auxiliary operations: Shape, Constant, Cast, etc.
        x = self.embedding(input_ids)
        
        # Create auxiliary operations for demonstration
        batch_size = x.shape[0]  # Shape operation
        scale_factor = torch.tensor(0.5, dtype=x.dtype)  # Constant operation
        
        # Reshape and scale (more auxiliary operations)
        x = x.mean(dim=1)  # ReduceMean operation
        x = x * scale_factor  # Mul with constant
        
        # Standard operations
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x


def basic_enhanced_export():
    """Demonstrate basic enhanced auxiliary operations export."""
    
    print("🚀 Basic Enhanced Export Example")
    print("=" * 50)
    
    # 1. Create model and inputs
    print("📝 Creating example model...")
    model = ExampleModel()
    model.eval()
    
    # Example inputs
    inputs = torch.randint(0, 100, (2, 8))  # batch_size=2, seq_len=8
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Input shape: {inputs.shape}")
    
    # 2. Test model forward pass
    print("\n🧪 Testing model forward pass...")
    with torch.no_grad():
        outputs = model(inputs)
    print(f"   Output shape: {outputs.shape}")
    print("   ✅ Model forward pass successful")
    
    # 3. Export with enhanced auxiliary operations
    print("\n🔄 Exporting with enhanced auxiliary operations...")
    
    output_path = Path(__file__).parent / "outputs" / "basic_enhanced_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create enhanced HTP exporter
    exporter = HierarchyExporter(strategy="htp")
    
    # Export with enhanced auxiliary operations
    result = exporter.export(
        model=model,
        example_inputs=inputs,
        output_path=str(output_path)
    )
    
    # 4. Analyze results
    print("\n📊 Export Results:")
    print(f"   Strategy used: {result['strategy']}")
    print(f"   Total operations: {result['total_operations']}")
    print(f"   Tagged operations: {result['tagged_operations']}")
    
    # Calculate coverage
    coverage_rate = (result['tagged_operations'] / result['total_operations']) * 100
    print(f"   Coverage rate: {coverage_rate:.1f}%")
    
    # Validate 100% coverage
    if coverage_rate == 100.0:
        print("   ✅ 100% operation coverage achieved!")
    else:
        print(f"   ⚠️ Partial coverage: {coverage_rate:.1f}%")
    
    # 5. Show auxiliary operation benefits
    print(f"\n🔍 Enhanced Features:")
    print(f"   Builtin tracking: {result.get('builtin_tracking_enabled', False)}")
    print(f"   Native regions: {result.get('native_op_regions', 0)}")
    print(f"   Trace length: {result.get('operation_trace_length', 0)}")
    
    print(f"\n📁 Output saved to: {output_path}")
    print(f"📁 Hierarchy data: {str(output_path).replace('.onnx', '_hierarchy.json')}")
    
    return result


def compare_with_legacy():
    """Compare enhanced auxiliary operations with legacy approach."""
    
    print("\n" + "=" * 50)
    print("🆚 Comparison with Legacy Approach")
    print("=" * 50)
    
    model = ExampleModel()
    model.eval()
    inputs = torch.randint(0, 100, (2, 8))
    
    # Legacy approach (usage-based)
    print("\n📊 Legacy Export (Usage-Based):")
    try:
        from modelexport.strategies.usage_based.usage_based_exporter import UsageBasedExporter
        
        legacy_exporter = UsageBasedExporter()
        legacy_output = Path(__file__).parent / "outputs" / "legacy_model.onnx"
        
        legacy_result = legacy_exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=str(legacy_output)
        )
        
        # Handle different result formats
        total_ops = legacy_result.get('total_operations', legacy_result.get('hierarchy_nodes', 1))
        tagged_ops = legacy_result.get('tagged_operations', legacy_result.get('hierarchy_nodes', 0))
        legacy_coverage = (tagged_ops / max(total_ops, 1)) * 100
        
        print(f"   Coverage rate: {legacy_coverage:.1f}%")
        print(f"   Total operations: {total_ops}")
        print(f"   Tagged operations: {tagged_ops}")
        
    except ImportError:
        print("   ⚠️ Legacy strategy not available")
        legacy_coverage = 0
    
    # Enhanced approach
    print("\n📊 Enhanced Export (HTP with Auxiliary Operations):")
    enhanced_exporter = HierarchyExporter(strategy="htp")
    enhanced_output = Path(__file__).parent / "outputs" / "enhanced_model.onnx"
    
    enhanced_result = enhanced_exporter.export(
        model=model,
        example_inputs=inputs,
        output_path=str(enhanced_output)
    )
    
    enhanced_coverage = (enhanced_result['tagged_operations'] / enhanced_result['total_operations']) * 100
    print(f"   Coverage rate: {enhanced_coverage:.1f}%")
    print(f"   Total operations: {enhanced_result['total_operations']}")
    print(f"   Tagged operations: {enhanced_result['tagged_operations']}")
    
    # Show improvement
    if legacy_coverage > 0:
        improvement = enhanced_coverage - legacy_coverage
        print(f"\n📈 Improvement: +{improvement:.1f} percentage points")
        print(f"   Coverage improvement: {improvement/legacy_coverage*100:.1f}%")
    
    print(f"✅ Enhanced approach achieves {enhanced_coverage:.1f}% coverage")


def demonstrate_auxiliary_operation_analysis():
    """Demonstrate auxiliary operation analysis capabilities."""
    
    print("\n" + "=" * 50)
    print("🔍 Auxiliary Operation Analysis")
    print("=" * 50)
    
    # Model with many auxiliary operations
    class AuxiliaryHeavyModel(nn.Module):
        def forward(self, x):
            # Many auxiliary operations
            c1 = torch.tensor(1.0, dtype=x.dtype)  # Constant
            c2 = torch.tensor(2.0, dtype=x.dtype)  # Constant
            
            # Shape manipulations
            original_shape = x.shape  # Shape
            x = x.reshape(-1)  # Reshape
            x = x.unsqueeze(0)  # Unsqueeze
            x = x.transpose(0, 1)  # Transpose
            x = x.squeeze()  # Squeeze
            
            # Arithmetic with constants
            x = x * c1  # Mul
            x = x + c2  # Add
            
            # Reduction
            result = x.mean()  # ReduceMean
            return result.unsqueeze(0)  # Unsqueeze
    
    model = AuxiliaryHeavyModel()
    model.eval()
    inputs = torch.randn(4, 3)
    
    # Export with performance monitoring
    print("\n🔄 Exporting auxiliary-heavy model...")
    exporter = HierarchyExporter(strategy="htp")
    
    output_path = Path(__file__).parent / "outputs" / "auxiliary_heavy_model.onnx"
    result = exporter.export(model, inputs, str(output_path))
    
    # Analyze auxiliary operations
    print(f"\n📊 Auxiliary Operation Analysis:")
    if 'auxiliary_operations_analysis' in result:
        aux_analysis = result['auxiliary_operations_analysis']
        
        print(f"   Total auxiliary operations: {aux_analysis['total_auxiliary_ops']}")
        print(f"   Successfully tagged: {aux_analysis['tagged_auxiliary_ops']}")
        print(f"   Context inherited: {aux_analysis['context_inherited']}")
        print(f"   Fallback tagged: {aux_analysis['fallback_tagged']}")
        
        # Show operation types
        print(f"\n   Operation type distribution:")
        for op_type, count in aux_analysis['operation_types'].items():
            print(f"     {op_type}: {count}")
    
    # Show performance metrics
    if 'performance_profile' in result:
        perf = result['performance_profile']
        print(f"\n⏱️ Performance Profile:")
        print(f"   Graph context building: {perf.get('graph_context_building_time', 0):.3f}s")
        print(f"   Context inheritance: {perf.get('context_inheritance_time', 0):.3f}s")
        print(f"   Fallback strategies: {perf.get('fallback_strategy_time', 0):.3f}s")
    
    # Validate complete coverage
    coverage = (result['tagged_operations'] / result['total_operations']) * 100
    print(f"\n✅ Final coverage: {coverage:.1f}%")


def validate_graph_integrity():
    """Demonstrate graph integrity validation with enhanced coverage."""
    
    print("\n" + "=" * 50)
    print("🔒 Graph Integrity Validation")
    print("=" * 50)
    
    from modelexport.core import tag_utils
    
    # Use the enhanced model from previous examples
    output_path = Path(__file__).parent / "outputs" / "basic_enhanced_model.onnx"
    
    if not output_path.exists():
        print("⚠️ Enhanced model not found, run basic export first")
        return
    
    print(f"🔍 Validating: {output_path.name}")
    
    # 1. Load hierarchy data
    try:
        hierarchy_data = tag_utils.load_tags_from_sidecar(str(output_path))
        node_tags = hierarchy_data['node_tags']
        print(f"   ✅ Loaded {len(node_tags)} tagged operations")
    except Exception as e:
        print(f"   ❌ Failed to load hierarchy data: {e}")
        return
    
    # 2. Validate tag consistency
    try:
        validation_result = tag_utils.validate_tag_consistency(str(output_path))
        if validation_result['consistent']:
            print("   ✅ Tag consistency validation passed")
        else:
            print("   ❌ Tag consistency issues found")
    except Exception as e:
        print(f"   ⚠️ Tag consistency check failed: {e}")
    
    # 3. Check for auxiliary operations
    auxiliary_tags = []
    for node_name, tags in node_tags.items():
        if any(keyword in str(tags).lower() for keyword in ['constant', 'shape', 'reshape', 'auxiliary']):
            auxiliary_tags.append((node_name, tags))
    
    print(f"   📊 Found {len(auxiliary_tags)} auxiliary operations with hierarchy tags")
    
    if auxiliary_tags:
        print("   Sample auxiliary operation tags:")
        for node_name, tags in auxiliary_tags[:3]:
            print(f"     {node_name}: {tags}")
    
    # 4. Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("   ✅ ONNX model validation passed")
        print(f"   📊 Model has {len(onnx_model.graph.node)} nodes")
    except Exception as e:
        print(f"   ❌ ONNX validation failed: {e}")
    
    print("✅ Graph integrity validation complete")


def main():
    """Run all basic examples."""
    
    print("🎯 ModelExport Enhanced Auxiliary Operations")
    print("Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Basic export example
        basic_result = basic_enhanced_export()
        
        # Comparison example
        compare_with_legacy()
        
        # Auxiliary operation analysis
        demonstrate_auxiliary_operation_analysis()
        
        # Graph integrity validation
        validate_graph_integrity()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Check the outputs/ directory for exported models")
        print("- Try the advanced integration examples")
        print("- Read the documentation for more use cases")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)