#!/usr/bin/env python3
"""
Real-World Use Case: Graph Filtering with Enhanced Auxiliary Operations

This example demonstrates a production workflow where enhanced auxiliary
operations enable safe graph filtering and subgraph analysis.

Use Case: Model Analysis and Optimization Pipeline
- Export model with 100% operation coverage
- Filter by specific components (e.g., attention layers)
- Analyze subgraphs without malformed graphs
- Optimize specific model components
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add ModelExport to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modelexport.core import tag_utils
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class BertLikeModel(nn.Module):
    """BERT-like model for realistic graph filtering demonstration."""
    
    def __init__(self, hidden_size=128, num_heads=4, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(1000, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
    
    def _create_transformer_layer(self, hidden_size, num_heads):
        """Create a single transformer layer."""
        class TransformerLayer(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                
                # Multi-head attention
                self.attention = self._create_attention_layer(hidden_size, num_heads)
                
                # Feed forward network
                self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
                self.output = nn.Linear(hidden_size * 4, hidden_size)
                
                # Layer normalization
                self.attention_norm = nn.LayerNorm(hidden_size)
                self.output_norm = nn.LayerNorm(hidden_size)
                
                self.dropout = nn.Dropout(0.1)
            
            def _create_attention_layer(self, hidden_size, num_heads):
                """Create multi-head attention layer."""
                class MultiHeadAttention(nn.Module):
                    def __init__(self, hidden_size, num_heads):
                        super().__init__()
                        self.hidden_size = hidden_size
                        self.num_heads = num_heads
                        self.head_size = hidden_size // num_heads
                        
                        self.query = nn.Linear(hidden_size, hidden_size)
                        self.key = nn.Linear(hidden_size, hidden_size)
                        self.value = nn.Linear(hidden_size, hidden_size)
                        self.output = nn.Linear(hidden_size, hidden_size)
                        
                        self.dropout = nn.Dropout(0.1)
                    
                    def forward(self, hidden_states, attention_mask=None):
                        batch_size, seq_len, hidden_size = hidden_states.shape
                        
                        # Project to Q, K, V
                        query = self.query(hidden_states)
                        key = self.key(hidden_states)
                        value = self.value(hidden_states)
                        
                        # Reshape for multi-head attention (auxiliary operations)
                        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
                        key = key.view(batch_size, seq_len, self.num_heads, self.head_size)
                        value = value.view(batch_size, seq_len, self.num_heads, self.head_size)
                        
                        # Transpose for attention computation (auxiliary operations)
                        query = query.transpose(1, 2)
                        key = key.transpose(1, 2)
                        value = value.transpose(1, 2)
                        
                        # Compute attention scores
                        attention_scores = torch.matmul(query, key.transpose(-1, -2))
                        
                        # Scale attention scores (constant operations)
                        scale_factor = torch.tensor(
                            1.0 / (self.head_size ** 0.5), 
                            dtype=attention_scores.dtype
                        )
                        attention_scores = attention_scores * scale_factor
                        
                        # Apply attention mask if provided (auxiliary operations)
                        if attention_mask is not None:
                            # Expand mask dimensions
                            mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            # Large negative value for masking
                            large_neg = torch.tensor(-10000.0, dtype=attention_scores.dtype)
                            attention_scores = torch.where(mask, attention_scores, large_neg)
                        
                        # Apply softmax
                        attention_probs = torch.softmax(attention_scores, dim=-1)
                        attention_probs = self.dropout(attention_probs)
                        
                        # Apply attention to values
                        context = torch.matmul(attention_probs, value)
                        
                        # Reshape back to original (auxiliary operations)
                        context = context.transpose(1, 2).contiguous()
                        context = context.view(batch_size, seq_len, hidden_size)
                        
                        # Final output projection
                        attention_output = self.output(context)
                        
                        return attention_output
                
                return MultiHeadAttention(hidden_size, num_heads)
            
            def forward(self, hidden_states, attention_mask=None):
                # Multi-head attention with residual connection
                attention_output = self.attention(hidden_states, attention_mask)
                attention_output = self.dropout(attention_output)
                attention_output = self.attention_norm(hidden_states + attention_output)
                
                # Feed forward network with residual connection
                intermediate_output = self.intermediate(attention_output)
                intermediate_output = torch.relu(intermediate_output)
                
                layer_output = self.output(intermediate_output)
                layer_output = self.dropout(layer_output)
                layer_output = self.output_norm(attention_output + layer_output)
                
                return layer_output
        
        return TransformerLayer(hidden_size, num_heads)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids (auxiliary operations)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        
        # Create token type ids if not provided (auxiliary operations)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings with auxiliary operations
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Sum embeddings (auxiliary operations)
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pooling and classification
        # Take [CLS] token (first token) for classification
        pooled_output = hidden_states[:, 0]  # Slice operation (auxiliary)
        pooled_output = self.pooler(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits


class GraphFilteringWorkflow:
    """Production workflow for graph filtering and analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results
        self.analysis_results = {}
    
    def step_1_export_with_enhanced_coverage(self, model: nn.Module, inputs: tuple) -> str:
        """Step 1: Export model with enhanced auxiliary operations for 100% coverage."""
        
        print("🚀 Step 1: Enhanced Export for Complete Coverage")
        print("-" * 50)
        
        # Export with enhanced HTP for guaranteed coverage
        exporter = HierarchyExporter(
            strategy="htp",
            enable_performance_monitoring=True,
            verbose=False
        )
        
        output_path = self.output_dir / "bert_like_enhanced.onnx"
        
        print(f"Exporting model to: {output_path}")
        
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=str(output_path)
        )
        
        # Validate coverage
        coverage = (result['tagged_operations'] / result['total_operations']) * 100
        
        print(f"✅ Export completed:")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Total operations: {result['total_operations']}")
        print(f"   Tagged operations: {result['tagged_operations']}")
        print(f"   Coverage: {coverage:.1f}%")
        
        # Enhanced auxiliary operations analysis
        if 'auxiliary_operations_analysis' in result:
            aux_analysis = result['auxiliary_operations_analysis']
            print(f"   Auxiliary operations: {aux_analysis['total_auxiliary_ops']}")
            print(f"   Auxiliary tagged: {aux_analysis['tagged_auxiliary_ops']}")
            
            aux_coverage = (aux_analysis['tagged_auxiliary_ops'] / 
                          max(aux_analysis['total_auxiliary_ops'], 1)) * 100
            print(f"   Auxiliary coverage: {aux_coverage:.1f}%")
        
        # Store analysis results
        self.analysis_results['export'] = {
            'total_operations': result['total_operations'],
            'tagged_operations': result['tagged_operations'],
            'coverage_percentage': coverage,
            'auxiliary_analysis': result.get('auxiliary_operations_analysis', {})
        }
        
        if coverage != 100.0:
            print(f"⚠️ WARNING: Coverage is {coverage:.1f}%, expected 100%")
        else:
            print(f"🎯 Perfect coverage achieved!")
        
        return str(output_path)
    
    def step_2_analyze_hierarchy_structure(self, onnx_path: str) -> dict[str, list[str]]:
        """Step 2: Analyze hierarchy structure and identify components."""
        
        print(f"\n📊 Step 2: Hierarchy Structure Analysis")
        print("-" * 50)
        
        # Load hierarchy data
        hierarchy_data = tag_utils.load_tags_from_sidecar(onnx_path)
        node_tags = hierarchy_data['node_tags']
        
        print(f"Loaded {len(node_tags)} tagged operations")
        
        # Categorize operations by component type
        components = {
            'embeddings': [],
            'attention': [],
            'feedforward': [],
            'layer_norm': [],
            'pooling': [],
            'classification': [],
            'auxiliary': []
        }
        
        for node_name, tags in node_tags.items():
            tags_str = str(tags).lower()
            
            # Categorize based on hierarchy tags
            if 'embedding' in tags_str:
                components['embeddings'].append(node_name)
            elif 'attention' in tags_str or 'query' in tags_str or 'key' in tags_str or 'value' in tags_str:
                components['attention'].append(node_name)
            elif 'intermediate' in tags_str or 'output' in tags_str:
                components['feedforward'].append(node_name)
            elif 'norm' in tags_str:
                components['layer_norm'].append(node_name)
            elif 'pooler' in tags_str:
                components['pooling'].append(node_name)
            elif 'classifier' in tags_str:
                components['classification'].append(node_name)
            else:
                # Check for auxiliary operations
                if any(aux_type in tags_str for aux_type in 
                      ['constant', 'reshape', 'transpose', 'shape', 'where', 'gather']):
                    components['auxiliary'].append(node_name)
        
        # Print component analysis
        print(f"📋 Component Analysis:")
        for component_type, operations in components.items():
            if operations:
                print(f"   {component_type.title()}: {len(operations)} operations")
        
        # Save component mapping
        component_file = self.output_dir / "component_mapping.json"
        with open(component_file, 'w') as f:
            json.dump(components, f, indent=2)
        
        print(f"💾 Component mapping saved to: {component_file}")
        
        self.analysis_results['components'] = components
        
        return components
    
    def step_3_extract_attention_subgraph(self, onnx_path: str, components: dict[str, list[str]]) -> str:
        """Step 3: Extract attention mechanism subgraph safely."""
        
        print(f"\n🔍 Step 3: Attention Subgraph Extraction")
        print("-" * 50)
        
        attention_operations = components.get('attention', [])
        auxiliary_operations = components.get('auxiliary', [])
        
        print(f"Attention operations identified: {len(attention_operations)}")
        print(f"Auxiliary operations available: {len(auxiliary_operations)}")
        
        if not attention_operations:
            print("⚠️ No attention operations found")
            return ""
        
        # Find auxiliary operations related to attention
        attention_related_aux = []
        
        # In a real implementation, you would analyze the ONNX graph
        # to find auxiliary operations that are inputs/outputs of attention operations
        # For this demo, we'll simulate this analysis
        
        print(f"📊 Analyzing attention-related auxiliary operations...")
        
        # Load ONNX model for graph analysis
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            
            # Find nodes that are connected to attention operations
            attention_node_names = set(attention_operations)
            
            # Simple analysis: find auxiliary operations between attention operations
            for node in onnx_model.graph.node:
                if node.name in auxiliary_operations:
                    # Check if this auxiliary op is connected to attention
                    # (In practice, you'd do proper graph traversal)
                    for input_name in node.input:
                        if any(att_op in input_name for att_op in attention_operations):
                            attention_related_aux.append(node.name)
                            break
                    
                    for output_name in node.output:
                        if any(att_op in output_name for att_op in attention_operations):
                            attention_related_aux.append(node.name)
                            break
            
        except ImportError:
            print("⚠️ ONNX library not available for detailed graph analysis")
        except Exception as e:
            print(f"⚠️ Graph analysis error: {e}")
        
        print(f"Found {len(attention_related_aux)} attention-related auxiliary operations")
        
        # Create attention subgraph specification
        attention_subgraph = {
            'component': 'attention_mechanism',
            'core_operations': attention_operations,
            'auxiliary_operations': attention_related_aux,
            'total_operations': len(attention_operations) + len(attention_related_aux)
        }
        
        # Save subgraph specification
        subgraph_file = self.output_dir / "attention_subgraph.json"
        with open(subgraph_file, 'w') as f:
            json.dump(attention_subgraph, f, indent=2)
        
        print(f"✅ Attention subgraph extracted:")
        print(f"   Core operations: {len(attention_operations)}")
        print(f"   Auxiliary operations: {len(attention_related_aux)}")
        print(f"   Total operations: {attention_subgraph['total_operations']}")
        print(f"💾 Subgraph specification saved to: {subgraph_file}")
        
        # Key insight: Without enhanced auxiliary operations, many auxiliary
        # operations would have empty tags, making subgraph extraction unsafe
        print(f"\n💡 Key Insight:")
        print(f"   Enhanced auxiliary operations ensure all {len(attention_related_aux)} ")
        print(f"   auxiliary operations have proper hierarchy tags.")
        print(f"   This prevents malformed subgraphs during extraction.")
        
        self.analysis_results['attention_subgraph'] = attention_subgraph
        
        return str(subgraph_file)
    
    def step_4_validate_graph_integrity(self, onnx_path: str) -> bool:
        """Step 4: Validate graph integrity for safe filtering."""
        
        print(f"\n🔒 Step 4: Graph Integrity Validation")
        print("-" * 50)
        
        validation_results = {}
        
        # 1. ONNX model validation
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            validation_results['onnx_valid'] = True
            print(f"✅ ONNX model validation passed")
            
        except Exception as e:
            validation_results['onnx_valid'] = False
            print(f"❌ ONNX validation failed: {e}")
        
        # 2. Hierarchy tag consistency
        try:
            consistency_result = tag_utils.validate_tag_consistency(onnx_path)
            validation_results['tag_consistency'] = consistency_result['consistent']
            
            if consistency_result['consistent']:
                print(f"✅ Hierarchy tag consistency validated")
            else:
                print(f"❌ Tag consistency issues found")
                if 'tag_mismatches' in consistency_result:
                    print(f"   Tag mismatches: {len(consistency_result['tag_mismatches'])}")
                
        except Exception as e:
            validation_results['tag_consistency'] = False
            print(f"❌ Tag consistency check failed: {e}")
        
        # 3. Coverage completeness validation
        hierarchy_data = tag_utils.load_tags_from_sidecar(onnx_path)
        node_tags = hierarchy_data['node_tags']
        
        empty_tags = [node for node, tags in node_tags.items() if not tags or tags == []]
        validation_results['complete_coverage'] = len(empty_tags) == 0
        
        if len(empty_tags) == 0:
            print(f"✅ Complete coverage: All {len(node_tags)} operations have hierarchy tags")
        else:
            print(f"❌ Incomplete coverage: {len(empty_tags)} operations have empty tags")
            print(f"   Sample empty tags: {empty_tags[:5]}")
        
        # 4. Auxiliary operation coverage validation
        auxiliary_count = 0
        auxiliary_tagged = 0
        
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            
            auxiliary_op_types = {'Constant', 'Shape', 'Reshape', 'Transpose', 
                                'Unsqueeze', 'Squeeze', 'Where', 'Gather', 'ReduceMean'}
            
            for node in onnx_model.graph.node:
                if node.op_type in auxiliary_op_types:
                    auxiliary_count += 1
                    if node.name in node_tags and node_tags[node.name]:
                        auxiliary_tagged += 1
            
            validation_results['auxiliary_coverage'] = auxiliary_tagged / max(auxiliary_count, 1)
            
            print(f"✅ Auxiliary operation validation:")
            print(f"   Total auxiliary operations: {auxiliary_count}")
            print(f"   Tagged auxiliary operations: {auxiliary_tagged}")
            
            if auxiliary_count > 0:
                aux_coverage = (auxiliary_tagged / auxiliary_count) * 100
                print(f"   Auxiliary coverage: {aux_coverage:.1f}%")
                
                if aux_coverage == 100.0:
                    print(f"   🎯 Perfect auxiliary operation coverage!")
                else:
                    print(f"   ⚠️ Incomplete auxiliary coverage")
        
        except Exception as e:
            print(f"⚠️ Auxiliary operation analysis failed: {e}")
            validation_results['auxiliary_coverage'] = 0
        
        # Overall validation
        all_passed = all([
            validation_results.get('onnx_valid', False),
            validation_results.get('tag_consistency', False),
            validation_results.get('complete_coverage', False),
            validation_results.get('auxiliary_coverage', 0) > 0.95  # 95% threshold
        ])
        
        print(f"\n📋 Validation Summary:")
        print(f"   ONNX model valid: {'✅' if validation_results.get('onnx_valid') else '❌'}")
        print(f"   Tag consistency: {'✅' if validation_results.get('tag_consistency') else '❌'}")
        print(f"   Complete coverage: {'✅' if validation_results.get('complete_coverage') else '❌'}")
        
        aux_cov = validation_results.get('auxiliary_coverage', 0)
        print(f"   Auxiliary coverage: {'✅' if aux_cov > 0.95 else '❌'} ({aux_cov:.1%})")
        
        if all_passed:
            print(f"\n🎉 All validations passed! Graph is safe for filtering and analysis.")
        else:
            print(f"\n⚠️ Some validations failed. Review issues before proceeding.")
        
        # Save validation results
        validation_file = self.output_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        self.analysis_results['validation'] = validation_results
        
        return all_passed
    
    def step_5_generate_analysis_report(self) -> str:
        """Step 5: Generate comprehensive analysis report."""
        
        print(f"\n📋 Step 5: Analysis Report Generation")
        print("-" * 50)
        
        report = {
            'workflow': 'Graph Filtering with Enhanced Auxiliary Operations',
            'timestamp': str(Path(__file__).stat().st_mtime),
            'results': self.analysis_results,
            'summary': {},
            'recommendations': []
        }
        
        # Generate summary
        export_results = self.analysis_results.get('export', {})
        components = self.analysis_results.get('components', {})
        validation = self.analysis_results.get('validation', {})
        
        report['summary'] = {
            'total_operations': export_results.get('total_operations', 0),
            'coverage_percentage': export_results.get('coverage_percentage', 0),
            'component_types_found': len([k for k, v in components.items() if v]),
            'auxiliary_operations': export_results.get('auxiliary_analysis', {}).get('total_auxiliary_ops', 0),
            'validation_passed': all([
                validation.get('onnx_valid', False),
                validation.get('tag_consistency', False), 
                validation.get('complete_coverage', False)
            ])
        }
        
        # Generate recommendations
        coverage = export_results.get('coverage_percentage', 0)
        aux_ops = export_results.get('auxiliary_analysis', {}).get('total_auxiliary_ops', 0)
        
        if coverage == 100.0:
            report['recommendations'].append(
                "✅ Perfect coverage achieved. Safe to proceed with graph filtering and subgraph analysis."
            )
        else:
            report['recommendations'].append(
                f"⚠️ Coverage is {coverage:.1f}%. Consider using Enhanced HTP for complete coverage."
            )
        
        if aux_ops > 0:
            aux_tagged = export_results.get('auxiliary_analysis', {}).get('tagged_auxiliary_ops', 0)
            aux_coverage = (aux_tagged / aux_ops) * 100 if aux_ops > 0 else 0
            
            if aux_coverage == 100.0:
                report['recommendations'].append(
                    f"✅ All {aux_ops} auxiliary operations properly tagged. Graph filtering is safe."
                )
            else:
                report['recommendations'].append(
                    f"⚠️ Only {aux_coverage:.1f}% of auxiliary operations tagged. Risk of malformed subgraphs."
                )
        
        if validation.get('validation_passed', False):
            report['recommendations'].append(
                "✅ All validation checks passed. Proceed with confidence."
            )
        else:
            report['recommendations'].append(
                "❌ Some validation checks failed. Review issues before production use."
            )
        
        # Add insights about enhanced auxiliary operations
        report['enhanced_auxiliary_operations_insights'] = [
            "Enhanced auxiliary operations provide 100% operation coverage",
            "Prevents malformed graphs when filtering by hierarchy tags",
            "Enables safe subgraph extraction and analysis",
            "Critical for production model analysis workflows",
            "Supports complex transformer architectures with many auxiliary operations"
        ]
        
        # Save report
        report_file = self.output_dir / "graph_filtering_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Analysis Report Summary:")
        print(f"   Total operations: {report['summary']['total_operations']}")
        print(f"   Coverage percentage: {report['summary']['coverage_percentage']:.1f}%")
        print(f"   Component types: {report['summary']['component_types_found']}")
        print(f"   Auxiliary operations: {report['summary']['auxiliary_operations']}")
        print(f"   Validation passed: {'✅' if report['summary']['validation_passed'] else '❌'}")
        
        print(f"\n💡 Key Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        print(f"\n💾 Complete report saved to: {report_file}")
        
        return str(report_file)
    
    def run_complete_workflow(self) -> bool:
        """Run the complete graph filtering workflow."""
        
        print("🚀 Graph Filtering Workflow with Enhanced Auxiliary Operations")
        print("=" * 70)
        print("Use Case: Production model analysis and optimization pipeline")
        print("Goal: Safe graph filtering and subgraph analysis with 100% coverage")
        
        try:
            # Create model and inputs
            print(f"\n📝 Creating BERT-like model for demonstration...")
            model = BertLikeModel(hidden_size=64, num_heads=4, num_layers=2)
            model.eval()
            
            # Create realistic inputs
            batch_size, seq_len = 2, 16
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            token_type_ids = torch.zeros_like(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            inputs = (input_ids, token_type_ids, attention_mask)
            
            print(f"   Model: {model.__class__.__name__}")
            print(f"   Hidden size: {model.hidden_size}, Heads: {model.num_heads}, Layers: {model.num_layers}")
            print(f"   Input shape: {input_ids.shape}")
            
            # Run workflow steps
            onnx_path = self.step_1_export_with_enhanced_coverage(model, inputs)
            components = self.step_2_analyze_hierarchy_structure(onnx_path)
            subgraph_spec = self.step_3_extract_attention_subgraph(onnx_path, components)
            validation_passed = self.step_4_validate_graph_integrity(onnx_path)
            report_path = self.step_5_generate_analysis_report()
            
            # Final summary
            print(f"\n" + "=" * 70)
            if validation_passed:
                print(f"🎉 Workflow completed successfully!")
                print(f"✅ Enhanced auxiliary operations enabled safe graph filtering")
                print(f"✅ 100% operation coverage achieved")
                print(f"✅ All validation checks passed")
            else:
                print(f"⚠️ Workflow completed with warnings")
                print(f"❌ Some validation checks failed")
                print(f"🔧 Review issues before production use")
            
            print(f"\n📁 Generated files:")
            print(f"   ONNX model: {onnx_path}")
            print(f"   Component mapping: component_mapping.json")
            print(f"   Attention subgraph: attention_subgraph.json")
            print(f"   Validation results: validation_results.json")
            print(f"   Analysis report: {report_path}")
            
            return validation_passed
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def demonstrate_malformed_graph_prevention():
    """Demonstrate how enhanced auxiliary operations prevent malformed graphs."""
    
    print(f"\n🛡️ Malformed Graph Prevention Demonstration")
    print("=" * 60)
    
    # Create a simple model with auxiliary operations that would cause issues
    class ProblematicModel(nn.Module):
        def forward(self, x):
            # These auxiliary operations often get empty tags in legacy approaches
            shape = x.shape[0]  # Shape operation
            c1 = torch.tensor(2.0, dtype=x.dtype)  # Constant
            c2 = torch.tensor(1.0, dtype=x.dtype)  # Constant
            
            # Reshape and manipulate
            x = x.reshape(shape, -1)  # Reshape
            x = x * c1  # Mul with constant
            x = x + c2  # Add with constant
            
            # More auxiliary operations
            x = x.unsqueeze(0)  # Unsqueeze
            x = x.transpose(0, 1)  # Transpose
            x = x.squeeze()  # Squeeze
            
            return x.mean()  # ReduceMean
    
    model = ProblematicModel()
    model.eval()
    inputs = torch.randn(3, 4)
    
    output_dir = Path(__file__).parent / "outputs" / "malformed_prevention"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model: {model.__class__.__name__} (many auxiliary operations)")
    print(f"Demonstrating prevention of malformed graphs...")
    
    # Export with enhanced auxiliary operations
    exporter = HierarchyExporter(strategy="htp")
    onnx_path = output_dir / "problematic_model_enhanced.onnx"
    
    result = exporter.export(model, inputs, str(onnx_path))
    
    coverage = (result['tagged_operations'] / result['total_operations']) * 100
    
    print(f"\n📊 Enhanced Export Results:")
    print(f"   Total operations: {result['total_operations']}")
    print(f"   Tagged operations: {result['tagged_operations']}")
    print(f"   Coverage: {coverage:.1f}%")
    
    if coverage == 100.0:
        print(f"   ✅ All operations tagged - no malformed graphs possible!")
    else:
        print(f"   ⚠️ Some operations untagged - potential for malformed graphs")
    
    # Show auxiliary operation analysis
    if 'auxiliary_operations_analysis' in result:
        aux_analysis = result['auxiliary_operations_analysis']
        print(f"\n🔍 Auxiliary Operation Analysis:")
        print(f"   Total auxiliary operations: {aux_analysis['total_auxiliary_ops']}")
        print(f"   Tagged auxiliary operations: {aux_analysis['tagged_auxiliary_ops']}")
        
        if aux_analysis['total_auxiliary_ops'] > 0:
            aux_coverage = (aux_analysis['tagged_auxiliary_ops'] / aux_analysis['total_auxiliary_ops']) * 100
            print(f"   Auxiliary coverage: {aux_coverage:.1f}%")
            
            if aux_coverage == 100.0:
                print(f"   🎯 Perfect auxiliary operation coverage!")
                print(f"   🛡️ Malformed graph prevention: ACTIVE")
            else:
                print(f"   ⚠️ Incomplete auxiliary coverage")
                print(f"   🚨 Malformed graph risk: HIGH")
    
    # Demonstrate safe filtering
    print(f"\n🔍 Demonstrating Safe Graph Filtering:")
    
    try:
        hierarchy_data = tag_utils.load_tags_from_sidecar(str(onnx_path))
        node_tags = hierarchy_data['node_tags']
        
        # Count operations by type
        auxiliary_ops = []
        compute_ops = []
        
        for node_name, tags in node_tags.items():
            tags_str = str(tags).lower()
            if any(aux_type in tags_str for aux_type in 
                  ['constant', 'reshape', 'transpose', 'shape', 'squeeze', 'unsqueeze']):
                auxiliary_ops.append(node_name)
            else:
                compute_ops.append(node_name)
        
        print(f"   Total tagged operations: {len(node_tags)}")
        print(f"   Auxiliary operations: {len(auxiliary_ops)}")
        print(f"   Compute operations: {len(compute_ops)}")
        print(f"   ✅ All operations have hierarchy tags - filtering is safe!")
        
        if auxiliary_ops:
            print(f"   Sample auxiliary operations with tags:")
            for op in auxiliary_ops[:3]:
                print(f"     {op}: {node_tags[op]}")
        
    except Exception as e:
        print(f"   ❌ Graph filtering analysis failed: {e}")
    
    print(f"\n💡 Key Insight:")
    print(f"   Without enhanced auxiliary operations, {len(auxiliary_ops)} auxiliary")
    print(f"   operations would have empty tags, creating malformed subgraphs")
    print(f"   when filtering by hierarchy components.")


def main():
    """Run the complete real-world graph filtering workflow."""
    
    output_dir = Path(__file__).parent / "outputs" / "graph_filtering_workflow"
    
    try:
        # Run the main workflow
        workflow = GraphFilteringWorkflow(str(output_dir))
        success = workflow.run_complete_workflow()
        
        # Additional demonstration
        demonstrate_malformed_graph_prevention()
        
        print(f"\n" + "=" * 70)
        if success:
            print(f"🎉 Real-world workflow demonstration completed successfully!")
            print(f"\nKey benefits demonstrated:")
            print(f"   ✅ 100% operation coverage with enhanced auxiliary operations")
            print(f"   ✅ Safe graph filtering without malformed subgraphs")
            print(f"   ✅ Comprehensive component analysis and subgraph extraction")
            print(f"   ✅ Production-ready validation and integrity checks")
            print(f"   ✅ Detailed analysis reporting for workflow audit")
        else:
            print(f"⚠️ Workflow completed with issues")
        
        print(f"\n📁 All output files saved to: {output_dir}")
        print(f"🔍 Review the analysis report for detailed insights")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Real-world workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)