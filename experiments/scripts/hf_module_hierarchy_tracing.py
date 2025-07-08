from transformers import AutoModel, AutoTokenizer
import torch

# Configuration - set your model here
MODEL_NAME = "prajjwal1/bert-tiny"  # Change this to test different models

# Load the model
print(f"Loading {MODEL_NAME}...")
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

class TracingHierarchyBuilder:
    """Tracing-based HF hierarchy builder using forward hooks."""
    
    def __init__(self):
        self.tag_stack = []
        self.execution_trace = []
        self.operation_context = {}
        self.hooks = []
        
    def is_hf_class(self, module):
        """Check if a module is a HuggingFace class"""
        module_path = module.__class__.__module__
        return module_path.startswith('transformers')
    
    def should_create_hierarchy_level(self, module):
        """Determine if module should create a new hierarchy level"""
        if self.is_hf_class(module):
            return True
        # Include some important torch.nn modules
        important_torch_nn = ['LayerNorm', 'Embedding']
        return module.__class__.__name__ in important_torch_nn
    
    def extract_module_info(self, module_name: str, module):
        """Extract module information for hierarchy building"""
        name_parts = module_name.split(".")
        
        # Check if this is an indexed module (e.g., layer.0)
        is_indexed_module = False
        module_index = None
        
        if len(name_parts) >= 2:
            last_part = name_parts[-1]
            second_last_part = name_parts[-2]
            
            if (last_part.isdigit() and 
                second_last_part in ['layer', 'layers', 'block', 'blocks', 'h']):
                is_indexed_module = True
                module_index = last_part
        
        return {
            'class_name': module.__class__.__name__,
            'module_index': module_index,
            'full_name': module_name,
            'is_indexed': is_indexed_module,
            'name_parts': name_parts,
        }
    
    def create_pre_hook(self, module_info):
        """Create pre-forward hook to push tag onto stack"""
        def pre_hook(module, inputs):
            # Get parent context from stack
            parent_tag = self.tag_stack[-1] if self.tag_stack else ""
            
            # Build current class name with index if needed
            if module_info['is_indexed']:
                current_class_name = f"{module_info['class_name']}.{module_info['module_index']}"
            else:
                current_class_name = module_info['class_name']
            
            # Build hierarchical tag
            hierarchical_tag = f"{parent_tag}/{current_class_name}"
            self.tag_stack.append(hierarchical_tag)
            
            # Record execution trace
            trace_entry = {
                'module_name': module_info['full_name'],
                'tag': hierarchical_tag,
                'action': 'enter',
                'stack_depth': len(self.tag_stack),
                'execution_order': len(self.execution_trace)
            }
            self.execution_trace.append(trace_entry)
            
            # Record in operation context
            self.operation_context[module_info['full_name']] = {
                "tag": hierarchical_tag,
                "module_class": module_info['class_name'],
                "creates_hierarchy": True,
                "stack_depth": len(self.tag_stack),
                "execution_order": len(self.execution_trace) - 1,
                "module_info": module_info
            }
            
        return pre_hook
    
    def create_post_hook(self, module_info):
        """Create post-forward hook to pop tag from stack"""
        def post_hook(module, inputs, outputs):
            # Record exit
            trace_entry = {
                'module_name': module_info['full_name'],
                'tag': self.tag_stack[-1] if self.tag_stack else "",
                'action': 'exit',
                'stack_depth': len(self.tag_stack),
                'execution_order': len(self.execution_trace)
            }
            self.execution_trace.append(trace_entry)
            
            # Pop the tag when module execution completes
            if self.tag_stack:
                self.tag_stack.pop()
                
        return post_hook
    
    def register_hooks(self, model):
        """Register forward hooks for tracing"""
        # Initialize stack with root module tag
        root_tag = f"/{model.__class__.__name__}"
        self.tag_stack = [root_tag]
        
        # Register hooks on all modules
        for name, module in model.named_modules():
            if name:  # Skip root module
                module_info = self.extract_module_info(name, module)
                
                # Only hook modules that should create hierarchy levels
                if self.should_create_hierarchy_level(module):
                    # Register pre-hook
                    pre_hook = module.register_forward_pre_hook(
                        self.create_pre_hook(module_info)
                    )
                    self.hooks.append(pre_hook)
                    
                    # Register post-hook
                    post_hook = module.register_forward_hook(
                        self.create_post_hook(module_info)
                    )
                    self.hooks.append(post_hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def trace_model_execution(self, model, example_inputs):
        """Trace model execution to build hierarchy mapping"""
        self.register_hooks(model)
        
        try:
            # Run model forward pass to trigger hooks
            model.eval()
            with torch.no_grad():
                _ = model(*example_inputs)
        finally:
            self.remove_hooks()
    
    def get_hierarchy_mapping(self):
        """Get the traced hierarchy mapping"""
        hierarchy_mapping = {}
        
        for module_name, context in self.operation_context.items():
            hierarchy_mapping[module_name] = context['tag']
        
        return hierarchy_mapping
    
    def get_execution_summary(self):
        """Get summary of execution trace"""
        return {
            'total_modules_traced': len(self.operation_context),
            'execution_steps': len(self.execution_trace),
            'max_stack_depth': max([t['stack_depth'] for t in self.execution_trace] + [0]),
            'hierarchy_mapping': self.get_hierarchy_mapping()
        }
    

# Prepare example inputs for tracing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = "Hello world, this is a test sentence."
inputs = tokenizer(text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
example_inputs = (inputs["input_ids"], inputs["attention_mask"])

print(f"Example inputs prepared:")
print(f"Input IDs shape: {example_inputs[0].shape}")
print(f"Attention mask shape: {example_inputs[1].shape}")

# Create tracing hierarchy builder
print("\nCreating tracing hierarchy builder...")
tracer = TracingHierarchyBuilder()

# Trace model execution
print("\nTracing model execution...")
tracer.trace_model_execution(model, example_inputs)

# Get results
traced_mapping = tracer.get_hierarchy_mapping()
execution_summary = tracer.get_execution_summary()

print(f"\nTracing completed!")
print(f"Execution summary: {execution_summary}")


# Display traced hierarchy mapping
print("Traced Hierarchy Mapping:")
print("=" * 70)

for module_name, hierarchy_tag in sorted(traced_mapping.items()):
    print(f"{module_name:40} -> {hierarchy_tag}")

print(f"\nFound {len(traced_mapping)} modules with traced hierarchy tags")
