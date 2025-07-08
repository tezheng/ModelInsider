from transformers import AutoModel

# Configuration - set your model here
MODEL_NAME = "prajjwal1/bert-tiny"  # Change this to test different models

# Load the model
print(f"Loading {MODEL_NAME}...")
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

def is_hf_class(module):
    """Check if a module is a HuggingFace class"""
    module_path = module.__class__.__module__
    return module_path.startswith('transformers')

def build_hf_hierarchy_mapping(model):
    """Recursively build HF module hierarchy mapping"""
    hierarchy_mapping = {}
    
    def recursive_build(module, current_tag, module_name, parent_children_names=None):
        """Recursively build hierarchy for a module"""
        
        # If this is an HF class, update the tag
        if is_hf_class(module):
            class_name = module.__class__.__name__
            
            # Add index if this is a repeated class among siblings
            if parent_children_names and module_name:
                module_basename = module_name.split('.')[-1]
                # Count how many siblings have the same class name
                same_class_siblings = []
                for sibling_name in parent_children_names:
                    if sibling_name == module_basename:
                        same_class_siblings.append(sibling_name)
                
                # If there are multiple siblings with same class, add index
                if len(same_class_siblings) > 1 or module_basename.isdigit():
                    # Extract index from module name (e.g., "0" from "layer.0")
                    if module_basename.isdigit():
                        index = module_basename
                        current_tag = f"{current_tag}/{class_name}.{index}"
                    else:
                        current_tag = f"{current_tag}/{class_name}"
                else:
                    current_tag = f"{current_tag}/{class_name}"
            else:
                current_tag = f"{current_tag}/{class_name}"
        
        # Map this module to its hierarchy tag
        if module_name:  # Skip root module
            hierarchy_mapping[module_name] = current_tag
        
        # Get children names for indexing
        children_names = [name for name, _ in module.named_children()]
        
        # Recursively process children
        for child_name, child_module in module.named_children():
            child_full_name = f"{module_name}.{child_name}" if module_name else child_name
            recursive_build(child_module, current_tag, child_full_name, children_names)
    
    # Start with root model - use simple class name without duplication
    root_class = model.__class__.__name__
    initial_tag = f"/{root_class}" if is_hf_class(model) else ""
    
    # Skip the root model itself and start with its children to avoid duplication
    for child_name, child_module in model.named_children():
        recursive_build(child_module, initial_tag, child_name, [name for name, _ in model.named_children()])
    
    return hierarchy_mapping

# Build the mapping
print("Building HF hierarchy mapping...")
hierarchy_mapping = build_hf_hierarchy_mapping(model)

print(f"\nFound {len(hierarchy_mapping)} module mappings:")
print("=" * 70)

for module_name, hierarchy_tag in sorted(hierarchy_mapping.items()):
    print(f"{module_name:40} -> {hierarchy_tag}")


# Analyze HF vs PyTorch classes
hf_modules = []
pytorch_modules = []

for name, module in model.named_modules():
    if name == '':  # Skip root
        continue
    
    if is_hf_class(module):
        hf_modules.append((name, module.__class__.__name__, module.__class__.__module__))
    else:
        pytorch_modules.append((name, module.__class__.__name__, module.__class__.__module__))

print(f"HuggingFace modules ({len(hf_modules)}):")
print("=" * 70)
for name, class_name, module_path in hf_modules:
    print(f"{name:30} -> {class_name:20}")

print(f"\nPyTorch modules ({len(pytorch_modules)}):")
print("=" * 70)
for name, class_name, module_path in pytorch_modules[:10]:  # Show first 10
    print(f"{name:30} -> {class_name:20}")

if len(pytorch_modules) > 10:
    print(f"... and {len(pytorch_modules) - 10} more PyTorch modules")
