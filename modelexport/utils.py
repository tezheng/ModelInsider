"""Utility functions for modelexport."""

import torch
from typing import List, Optional


def should_tag_module(module: torch.nn.Module, exceptions: Optional[List[str]] = None) -> bool:
    """
    Determine if a module should be tagged in the hierarchy based on semantic importance.
    
    Filters out torch.nn infrastructure modules while preserving semantically important ones.
    
    Args:
        module: PyTorch module to evaluate
        exceptions: List of torch.nn class names to include despite being infrastructure.
                   If None, uses default exceptions: ["LayerNorm"]
        
    Returns:
        True if module should be tagged in hierarchy, False otherwise
        
    Examples:
        >>> # HuggingFace modules - always included
        >>> should_tag_module(bert_embeddings)  # BertEmbeddings -> True
        >>> should_tag_module(bert_attention)   # BertAttention -> True
        
        >>> # torch.nn with semantic importance - included by default
        >>> should_tag_module(layer_norm)       # LayerNorm -> True
        
        >>> # torch.nn infrastructure - excluded
        >>> should_tag_module(embedding)        # Embedding -> False
        >>> should_tag_module(dropout)          # Dropout -> False  
        >>> should_tag_module(relu)            # ReLU -> False
        >>> should_tag_module(linear)          # Linear -> False
        
        >>> # Custom exceptions
        >>> should_tag_module(embedding, exceptions=["LayerNorm", "Embedding"])  # Embedding -> True
    """
    # Default exception for semantically important torch.nn modules
    if exceptions is None:
        exceptions = ["LayerNorm"]
    
    module_class_name = module.__class__.__name__
    module_full_path = f"{module.__class__.__module__}.{module.__class__.__name__}"
    
    # Check if it's torch.nn infrastructure
    is_torch_infrastructure = (
        module.__class__.__module__.startswith('torch.nn') or 
        module.__class__.__module__.startswith('torch._C')
    )
    
    if is_torch_infrastructure:
        # Include only if it's in the exceptions list
        return module_class_name in exceptions
    else:
        # Include all non-torch.nn modules (HuggingFace, user-defined, etc.)
        return True


def build_hierarchy_path(model_root: torch.nn.Module, module_path: str, all_modules: dict) -> str:
    """
    Build hierarchical class path for a module instance.
    
    Args:
        model_root: Root model (e.g., BertModel)
        module_path: Module instance path (e.g., "encoder.layer.0.attention.self")
        all_modules: Dict mapping module paths to modules from named_modules()
        
    Returns:
        Hierarchical path (e.g., "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention")
        
    Examples:
        >>> build_hierarchy_path(bert_model, "embeddings", all_modules)
        "/BertModel/BertEmbeddings"
        
        >>> build_hierarchy_path(bert_model, "encoder.layer.0.attention.self", all_modules)
        "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"
        
        >>> build_hierarchy_path(bert_model, "embeddings.LayerNorm", all_modules)
        "/BertModel/BertEmbeddings/LayerNorm"
    """
    # Start with root model class name
    root_class = model_root.__class__.__name__
    hierarchy_parts = [root_class]
    
    if not module_path:  # Root module
        return "/" + "/".join(hierarchy_parts)
    
    # Split the module path into parts
    path_parts = module_path.split('.')
    current_path = ""
    
    for i, part in enumerate(path_parts):
        if current_path:
            current_path += "." + part
        else:
            current_path = part
            
        # Get the module at this path
        if current_path in all_modules:
            module = all_modules[current_path]
            class_name = module.__class__.__name__
            
            # Handle numeric indices in the path (e.g., "layer.0" -> "BertLayer.0")
            if i > 0 and path_parts[i-1] in ['layer'] and part.isdigit():
                # Previous part was "layer" and current part is a number
                # Combine with the class name: "BertLayer.0"
                hierarchy_parts[-1] = f"{class_name}.{part}"
            else:
                hierarchy_parts.append(class_name)
    
    return "/" + "/".join(hierarchy_parts)