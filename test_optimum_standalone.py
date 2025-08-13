#!/usr/bin/env python3
"""
Standalone test to verify if TasksManager is working correctly.
This bypasses our codebase entirely.
"""

def test_tasks_manager():
    print("Testing TasksManager directly...")
    
    # Import everything the CLI might import - including ONNX configs
    try:
        from optimum.exporters.tasks import TasksManager
        print("✅ Imported TasksManager")
        
        # Force import of ONNX configs - this might register them!
        try:
            from optimum.exporters.onnx import main_export  # This might trigger registrations
            print("✅ Imported ONNX main_export")
        except ImportError:
            pass
            
        try:
            import optimum.exporters.onnx  # Force import the entire module
            print("✅ Imported optimum.exporters.onnx module")
        except ImportError:
            pass
        
        # Check if the TasksManager has the right registrations
        print("Checking TasksManager registrations:")
        
        # Check general model types
        if hasattr(TasksManager, '_SUPPORTED_MODEL_TYPE'):
            supported_models = getattr(TasksManager, '_SUPPORTED_MODEL_TYPE', {})
            print(f"  General model types: {len(supported_models)} models")
            print(f"  BERT in general registry: {'bert' in supported_models}")
        
        # Check ONNX-specific registrations
        if hasattr(TasksManager, '_SUPPORTED_BACKENDS'):
            backends = getattr(TasksManager, '_SUPPORTED_BACKENDS', {})
            print(f"  Supported backends: {list(backends.keys())}")
            if 'onnx' in backends:
                onnx_models = backends['onnx']
                print(f"  ONNX models: {len(onnx_models)} registered")
                print(f"  BERT in ONNX registry: {'bert' in onnx_models}")
                if len(onnx_models) == 0:
                    print("  ❌ ONNX registry is EMPTY!")
                else:
                    print(f"  First few ONNX models: {list(onnx_models.keys())[:5]}")
        
        # Try to manually check what get_supported_backends returns
        try:
            backends_check = TasksManager.get_supported_backends()
            print(f"  get_supported_backends(): {backends_check}")
        except Exception as e:
            print(f"  get_supported_backends() error: {e}")
        
        # Check for any attributes that might contain ONNX registrations
        print(f"  TasksManager attributes: {[attr for attr in dir(TasksManager) if not attr.startswith('__')]}")
        
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return
    
    # Test BERT model type with different parameter combinations
    try:
        # Try the way our code calls it
        print("Testing our approach:")
        supported = TasksManager.get_supported_tasks_for_model_type(
            'bert', 
            exporter='onnx', 
            library_name='transformers'
        )
        print(f"BERT supported tasks: {supported}")
        
        # Try without explicit parameter names
        print("Testing positional approach:")
        supported2 = TasksManager.get_supported_tasks_for_model_type('bert', 'onnx', 'transformers')
        print(f"BERT supported tasks: {supported2}")
        
        # Try with different order
        print("Testing library_name first:")
        supported3 = TasksManager.get_supported_tasks_for_model_type(
            model_type='bert',
            library_name='transformers', 
            exporter='onnx'
        )
        print(f"BERT supported tasks: {supported3}")
        
        if supported:
            print("✅ BERT is supported!")
            task = next(iter(supported.keys()))
            print(f"First available task: {task}")
            
            # Try to get config constructor
            try:
                config_constructor = TasksManager.get_exporter_config_constructor(
                    'onnx', 'bert', task, 'transformers'
                )
                print(f"✅ Config constructor: {config_constructor}")
                
                # Try to create actual config
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
                export_config = config_constructor(config)
                print(f"✅ Export config created: {type(export_config)}")
                
                # Try to generate dummy inputs
                dummy_inputs = export_config.generate_dummy_inputs()
                print(f"✅ Dummy inputs generated: {list(dummy_inputs.keys())}")
                
            except Exception as e:
                print(f"❌ Error creating config: {e}")
                
        else:
            print("❌ BERT not supported - empty dict returned")
            
    except Exception as e:
        print(f"❌ Error testing TasksManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tasks_manager()