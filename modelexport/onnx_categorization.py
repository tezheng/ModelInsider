"""
ONNX Operator Categorization Module

Provides categorization of ONNX operators for enhanced tagging analysis and reporting.
Based on ONNX specification analysis and operator functionality.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

# ONNX Operator Categories with thresholds and criticality
ONNX_OPERATOR_CATEGORIES = {
    'mathematical_operations': {
        'operators': [
            'Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Gemm', 'Pow', 'Sqrt', 
            'Abs', 'Ceil', 'Floor', 'Round', 'Exp', 'Log', 'Max', 'Min',
            'Sum', 'Mean', 'Neg', 'Reciprocal', 'Sign'
        ],
        'threshold': 95.0,
        'critical': True,
        'description': 'Core mathematical operations'
    },
    
    'activation_functions': {
        'operators': [
            'Relu', 'LeakyRelu', 'Elu', 'Selu', 'Sigmoid', 'Tanh', 'Softmax',
            'LogSoftmax', 'Softplus', 'Softsign', 'HardSigmoid', 'Gelu', 'Erf',
            'ThresholdedRelu', 'PRelu', 'Swish'
        ],
        'threshold': 95.0,
        'critical': True,
        'description': 'Activation and nonlinear functions'
    },
    
    'neural_network_layers': {
        'operators': [
            'Conv', 'ConvTranspose', 'BatchNormalization', 'InstanceNormalization',
            'GroupNormalization', 'LayerNormalization', 'Dropout', 'LSTM', 'GRU',
            'RNN', 'Attention', 'MultiHeadAttention'
        ],
        'threshold': 95.0,
        'critical': True,
        'description': 'Neural network layer operations'
    },
    
    'reduction_operations': {
        'operators': [
            'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd',
            'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp',
            'ReduceSumSquare', 'ArgMax', 'ArgMin'
        ],
        'threshold': 95.0,
        'critical': True,
        'description': 'Reduction and aggregation operations'
    },
    
    'lookup_indexing': {
        'operators': [
            'Gather', 'GatherElements', 'GatherND', 'Scatter', 'ScatterElements',
            'ScatterND', 'OneHot', 'EyeLike'
        ],
        'threshold': 90.0,
        'critical': True,
        'description': 'Lookup, indexing, and scatter operations'
    },
    
    'tensor_manipulation': {
        'operators': [
            'Reshape', 'Transpose', 'Concat', 'Split', 'Slice', 'Squeeze', 'Unsqueeze',
            'Expand', 'Tile', 'Pad', 'Flip', 'Roll', 'Flatten', 'Identity'
        ],
        'threshold': 85.0,
        'critical': False,
        'description': 'Tensor shape and structure manipulation'
    },
    
    'shape_metadata': {
        'operators': [
            'Shape', 'Size', 'Range', 'NonZero'
        ],
        'threshold': 80.0,
        'critical': False,
        'description': 'Shape and metadata extraction'
    },
    
    'parameter_constants': {
        'operators': [
            'Constant', 'ConstantOfShape', 'RandomNormal', 'RandomUniform',
            'RandomNormalLike', 'RandomUniformLike', 'Bernoulli'
        ],
        'threshold': 70.0,
        'critical': False,
        'description': 'Constants and parameter generation'
    },
    
    'comparison_logic': {
        'operators': [
            'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'Not',
            'And', 'Or', 'Xor', 'Where', 'IsInf', 'IsNaN'
        ],
        'threshold': 75.0,
        'critical': False,
        'description': 'Comparison and logical operations'
    },
    
    'type_conversion': {
        'operators': [
            'Cast', 'CastLike'
        ],
        'threshold': 60.0,
        'critical': False,
        'description': 'Data type conversion operations'
    },
    
    'control_flow': {
        'operators': [
            'If', 'Loop', 'Scan'
        ],
        'threshold': 80.0,
        'critical': False,
        'description': 'Control flow operations'
    }
}

def categorize_operation(op_type: str) -> str:
    """
    Return category name for given ONNX operation type.
    
    Args:
        op_type: ONNX operation type (e.g., 'Add', 'Conv', 'Relu')
        
    Returns:
        Category name, or 'uncategorized' if not found
    """
    for category, info in ONNX_OPERATOR_CATEGORIES.items():
        if op_type in info['operators']:
            return category
    return 'uncategorized'

def get_category_info(category: str) -> Dict[str, Any]:
    """Get information about a category."""
    return ONNX_OPERATOR_CATEGORIES.get(category, {
        'threshold': 50.0,
        'critical': False,
        'description': 'Uncategorized operations'
    })

def analyze_tagging_by_category(onnx_model, node_tags: Dict) -> Dict[str, Any]:
    """
    Analyze tagging performance by operation category.
    
    Args:
        onnx_model: Loaded ONNX model
        node_tags: Dictionary of node name -> tag info
        
    Returns:
        Dictionary with category statistics
    """
    # Count operations by category
    category_totals = defaultdict(int)
    category_tagged = defaultdict(int)
    
    for node in onnx_model.graph.node:
        node_name = node.name or f"{node.op_type}_{len([n for n in node_tags.keys() if node.op_type in n])}"
        op_type = node.op_type
        category = categorize_operation(op_type)
        
        category_totals[category] += 1
        
        # Check if this node is tagged
        if node_name in node_tags:
            tags = node_tags[node_name].get('tags', [])
            if tags:
                category_tagged[category] += 1
    
    # Calculate statistics for each category
    category_stats = {}
    total_nodes = 0
    total_tagged = 0
    
    for category in list(ONNX_OPERATOR_CATEGORIES.keys()) + ['uncategorized']:
        total = category_totals[category]
        tagged = category_tagged[category]
        
        if total > 0:
            percentage = (tagged / total) * 100
            category_info = get_category_info(category)
            
            category_stats[category] = {
                'total': total,
                'tagged': tagged,
                'percentage': percentage,
                'threshold': category_info['threshold'],
                'critical': category_info['critical'],
                'description': category_info['description'],
                'needs_attention': percentage < category_info['threshold']
            }
            
            total_nodes += total
            total_tagged += tagged
    
    # Overall statistics
    overall_percentage = (total_tagged / total_nodes * 100) if total_nodes > 0 else 0
    
    return {
        'category_stats': category_stats,
        'overall': {
            'total_nodes': total_nodes,
            'total_tagged': total_tagged,
            'percentage': overall_percentage
        },
        'categories_needing_attention': [
            cat for cat, stats in category_stats.items() 
            if stats['needs_attention']
        ]
    }

def format_category_summary(category_analysis: Dict, format_type: str = 'cli') -> str:
    """
    Format category statistics for display.
    
    Args:
        category_analysis: Result from analyze_tagging_by_category
        format_type: 'cli' or 'json'
        
    Returns:
        Formatted string for display
    """
    if format_type == 'cli':
        return _format_cli_summary(category_analysis)
    elif format_type == 'json':
        return _format_json_summary(category_analysis)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def _format_cli_summary(analysis: Dict) -> str:
    """Format category summary for CLI display."""
    lines = ["📊 Tagging Performance by Category:"]
    
    category_stats = analysis['category_stats']
    
    # Sort categories by criticality and percentage
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: (not x[1]['critical'], -x[1]['percentage'])
    )
    
    for category, stats in sorted_categories:
        # Format category name
        display_name = category.replace('_', ' ').title()
        
        # Create status indicator
        if stats['needs_attention']:
            if stats['critical']:
                indicator = " 🚨"  # Critical issue
            else:
                indicator = " ⚠️"   # Warning
        else:
            indicator = ""
        
        # Format line
        line = f"   {display_name:<25} {stats['percentage']:>5.1f}% tagged ({stats['tagged']}/{stats['total']} nodes){indicator}"
        lines.append(line)
    
    # Add attention note if needed
    attention_categories = analysis['categories_needing_attention']
    if attention_categories:
        critical_attention = [
            cat for cat in attention_categories 
            if category_stats[cat]['critical']
        ]
        
        if critical_attention:
            lines.append("")
            lines.append("   🚨 Critical categories below threshold - model correctness may be affected")
        else:
            lines.append("")
            lines.append("   ⚠️  Some categories below recommended threshold")
    
    return "\n".join(lines)

def _format_json_summary(analysis: Dict) -> Dict:
    """Format category summary for JSON output."""
    return {
        'category_statistics': {
            cat: {
                'total': stats['total'],
                'tagged': stats['tagged'], 
                'percentage': round(stats['percentage'], 1),
                'threshold': stats['threshold'],
                'critical': stats['critical'],
                'needs_attention': stats['needs_attention']
            }
            for cat, stats in analysis['category_stats'].items()
        },
        'overall_statistics': analysis['overall'],
        'categories_needing_attention': analysis['categories_needing_attention']
    }