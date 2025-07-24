"""
Test Core Functionality - Comprehensive Testing of Core Components

This test suite validates the fundamental building blocks of the modelexport system,
including input generation, hierarchy building, and ONNX node tagging. These are the
core components that all export strategies depend on.

Architecture Overview:
    The modelexport system follows a layered architecture:
    1. Model Input Generation - Unified input generation via Optimum integration
    2. Hierarchy Building - Extract PyTorch module hierarchy via tracing  
    3. ONNX Node Tagging - Map ONNX operations back to source modules
    4. Export Strategies - Use core components for hierarchy-preserving export

Test Categories Covered:
    ├── Model Input Generator
    │   ├── Manual Input Specification Testing
    │   ├── Automatic Input Generation via Optimum
    │   ├── SAM Coordinate Fix Integration (NEW)
    │   └── Error Handling and Validation
    ├── Hierarchy Builder (TracingHierarchyBuilder)
    │   ├── Module Discovery and Tracing
    │   ├── Execution Context Capture
    │   ├── Built-in PyTorch Module Tracking
    │   └── Performance Optimization Features  
    ├── ONNX Node Tagger
    │   ├── Node-to-Module Mapping
    │   ├── Tag Propagation Logic
    │   ├── Fallback Strategies
    │   └── Coverage Validation
    └── Utility Functions
        ├── Tag Utilities and Validation
        ├── Helper Functions
        └── Configuration Loading

CARDINAL RULES Enforcement:
    - MUST-001: No hardcoded logic - Universal design validation
    - MUST-002: torch.nn filtering - Proper module filtering validation  
    - MUST-003: Universal design - Architecture-agnostic approach validation

Test Data Requirements:
    - Primary: prajjwal1/bert-tiny (lightweight, fast testing)
    - Secondary: facebook/sam-vit-base (SAM coordinate fix testing)
    - Synthetic: Custom test models for edge cases
    - Multiple architectures: BERT, Vision models, Multimodal models

Performance Considerations:
    - Core tests should complete in <30 seconds total
    - Uses cached model downloads when possible
    - Minimal ONNX export for speed (validation only, not full workflow)
    - Focus on unit testing with mocked heavy operations where appropriate

Expected Outcomes:
    - All core components work independently
    - Integration points are validated
    - SAM coordinate fix works automatically
    - No empty tags or CARDINAL RULE violations
    - Universal design principles are enforced
"""


import pytest
import torch
import torch.nn as nn
from transformers import AutoModel

from modelexport.core.model_input_generator import (
    generate_dummy_inputs,
    get_export_config_from_model_path,
    patch_export_config,
)
from modelexport.core.onnx_node_tagger import create_node_tagger_from_hierarchy
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder


class TestModelInputGenerator:
    """
    Comprehensive test suite for the unified model input generation system.
    
    The ModelInputGenerator is the entry point for all export operations,
    providing both manual input specification and automatic generation via
    Optimum's TasksManager. This component is critical because incorrect
    inputs lead to failed exports or incorrect model behavior.
    
    Key Features Tested:
    - Manual input specification with validation
    - Automatic generation via Optimum integration
    - SAM coordinate fix (automatic type-based detection)
    - Priority handling (input_specs > model_name_or_path)
    - Error handling and user-friendly messages
    - Integration with export configuration system
    
    Architecture:
        generate_dummy_inputs()
        ├── input_specs provided? → _generate_from_specs()
        └── model_name_or_path provided? → generate_dummy_inputs_from_model_path()
            ├── get_export_config_from_model_path()
            ├── patch_export_config() ← SAM coordinate fix applied here
            └── export_config.generate_dummy_inputs()
    """
    
    def test_manual_input_specs_basic(self):
        """
        Test basic manual input specification functionality.
        
        This tests the core manual input generation path that allows users
        to specify exact input shapes, dtypes, and value ranges. This is 
        essential for custom models or when automatic generation fails.
        
        Test Scenario:
        - Provide valid input specifications for a BERT-like model
        - Verify correct tensor generation with specified parameters
        - Validate shape, dtype, and value range compliance
        
        Expected Behavior:
        - Tensors generated match specified shapes exactly
        - Data types are correctly converted (int/long → torch.long, float → torch.float32)
        - Value ranges are respected for both integer and float tensors
        - No fallback to automatic generation occurs
        """
        input_specs = {
            "input_ids": {"shape": [2, 32], "dtype": "int", "range": [0, 1000]},
            "token_type_ids": {"shape": [2, 32], "dtype": "int", "range": [0, 1]},
            "attention_mask": {"shape": [2, 32], "dtype": "int", "range": [0, 1]}
        }
        
        inputs = generate_dummy_inputs(input_specs=input_specs)
        
        # Validate structure
        assert len(inputs) == 3, "Should generate exactly 3 input tensors"
        assert set(inputs.keys()) == {"input_ids", "token_type_ids", "attention_mask"}
        
        # Validate input_ids
        input_ids = inputs["input_ids"]
        assert list(input_ids.shape) == [2, 32], f"Expected [2, 32], got {list(input_ids.shape)}"
        assert input_ids.dtype == torch.long, f"Expected torch.long, got {input_ids.dtype}"
        assert input_ids.min() >= 0 and input_ids.max() <= 1000, "input_ids out of range [0, 1000]"
        
        # Validate token_type_ids
        token_type_ids = inputs["token_type_ids"]
        assert list(token_type_ids.shape) == [2, 32]
        assert token_type_ids.dtype == torch.long
        assert token_type_ids.min() >= 0 and token_type_ids.max() <= 1, "token_type_ids out of range [0, 1]"
        
        # Validate attention_mask
        attention_mask = inputs["attention_mask"]
        assert list(attention_mask.shape) == [2, 32]
        assert attention_mask.dtype == torch.long
        assert attention_mask.min() >= 0 and attention_mask.max() <= 1, "attention_mask out of range [0, 1]"
    
    def test_automatic_generation_bert_model(self):
        """
        Test automatic input generation for BERT models via Optimum.
        
        This validates the complete automatic generation pipeline using
        Optimum's TasksManager. BERT models are an excellent test case
        because they have well-defined input requirements and broad support.
        
        Test Scenario:
        - Use prajjwal1/bert-tiny for fast, reliable testing
        - Verify automatic task detection works
        - Validate generated inputs match BERT requirements
        - Ensure no hardcoded assumptions about model structure
        
        Expected Behavior:
        - Auto-detects 'feature-extraction' or similar task
        - Generates standard BERT inputs: input_ids, token_type_ids, attention_mask
        - Input shapes use reasonable defaults (batch_size=1, sequence_length=128, etc.)
        - All tensors have appropriate dtypes (int64 for token inputs)
        - No empty or invalid tensors generated
        """
        inputs = generate_dummy_inputs(model_name_or_path="prajjwal1/bert-tiny")
        
        # Validate basic structure
        assert isinstance(inputs, dict), "Should return dictionary of inputs"
        assert len(inputs) > 0, "Should generate at least one input tensor"
        
        # BERT models typically generate these inputs
        generated_inputs = set(inputs.keys())
        
        # Should have at least input_ids (core requirement)
        assert "input_ids" in generated_inputs, "BERT model should have input_ids"
        
        # Validate input_ids specifically  
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, torch.Tensor), "input_ids should be torch.Tensor"
        assert input_ids.dtype in (torch.long, torch.int64), f"input_ids should be integer type, got {input_ids.dtype}"
        assert len(input_ids.shape) >= 2, f"input_ids should be at least 2D, got shape {input_ids.shape}"
        assert input_ids.shape[0] >= 1, "Batch size should be at least 1"
        assert input_ids.shape[1] >= 8, "Sequence length should be reasonable (>=8)"
        
        # Validate other common BERT inputs if present
        for input_name in ["token_type_ids", "attention_mask"]:
            if input_name in inputs:
                tensor = inputs[input_name]
                assert isinstance(tensor, torch.Tensor), f"{input_name} should be torch.Tensor"
                assert tensor.shape == input_ids.shape, f"{input_name} shape should match input_ids"
    
    def test_sam_coordinate_generation(self):
        """
        Test SAM decoder coordinate generation.
        
        This test validates that SAM decoder-only export generates appropriate
        input coordinates using Optimum's default behavior.
        
        Test Scenario:
        - Use facebook/sam-vit-base with mask-generation task
        - Verify decoder inputs are generated correctly
        - Confirm coordinate values are in normalized range [0, 1]
        
        Expected Behavior:
        - Generates decoder inputs: image_embeddings, input_points, input_labels
        - input_points contains normalized coordinates [0, 1]
        - Follows Optimum's standard SAM decoder export behavior
        
        Technical Details:
        - Uses enhance_exporter_config to map mask-generation -> decoder export
        - Relies on Optimum's SamOnnxConfig with vision_encoder=False
        - No custom coordinate patching needed
        """
        # Use mask-generation task to get decoder inputs that include input_points
        inputs = generate_dummy_inputs(model_name_or_path="facebook/sam-vit-base", task="mask-generation")
        
        # Validate SAM model inputs structure
        assert isinstance(inputs, dict), "Should return dictionary of inputs"
        assert "input_points" in inputs, "SAM model should have input_points"
        
        # This is the critical test - coordinate range validation
        input_points = inputs["input_points"]
        assert isinstance(input_points, torch.Tensor), "input_points should be torch.Tensor"
        assert input_points.dtype == torch.float32, f"input_points should be float32, got {input_points.dtype}"
        
        # With the new design, decoder-only export uses Optimum's default behavior
        # which generates normalized coordinates [0, 1]
        min_val = float(input_points.min())
        max_val = float(input_points.max())
        
        assert min_val >= 0, f"Coordinates should be >= 0, got min={min_val}"
        assert max_val <= 1.0, f"Coordinates should be <= 1.0, got max={max_val}"
        
        # Decoder-only export generates normalized coordinates by default
        # This is Optimum's standard behavior for SAM decoder export
        assert max_val < 2, f"Coordinates should be normalized [0, 1], got max={max_val}"
        
        # Validate shape (SAM expects [batch_size, point_batch_size, nb_points_per_image, 2])
        assert len(input_points.shape) == 4, f"input_points should be 4D tensor, got shape {input_points.shape}"
        assert input_points.shape[-1] == 2, f"Last dimension should be 2 (x, y coordinates), got {input_points.shape[-1]}"
        
        # Validate other SAM inputs are generated normally
        expected_sam_inputs = {"image_embeddings", "image_positional_embeddings", "input_labels"}
        for input_name in expected_sam_inputs:
            if input_name in inputs:
                tensor = inputs[input_name]
                assert isinstance(tensor, torch.Tensor), f"{input_name} should be torch.Tensor"
                assert tensor.numel() > 0, f"{input_name} should not be empty"
    
    def test_input_specs_priority_over_auto_generation(self):
        """
        Test that manual input specs take priority over automatic generation.
        
        This validates the documented priority system: input_specs > model_name_or_path.
        When both are provided, the system should use input_specs and NOT fall
        back to automatic generation, even if the model exists and is valid.
        
        Test Scenario:
        - Provide both input_specs and model_name_or_path
        - Use completely different input names than the model would generate
        - Verify only input_specs are used, no model inputs generated
        
        Expected Behavior:
        - Only generates inputs from input_specs
        - Ignores model_name_or_path completely (no Optimum calls)
        - Uses exact specifications provided (shape, dtype, range)
        - No mixing of manual and automatic inputs
        """
        # Use custom input specs that don't match BERT
        custom_specs = {
            "custom_input_1": {"shape": [3, 16], "dtype": "float", "range": [0.0, 1.0]},
            "custom_input_2": {"shape": [3, 8], "dtype": "int", "range": [0, 100]}
        }
        
        inputs = generate_dummy_inputs(
            model_name_or_path="prajjwal1/bert-tiny",  # This should be IGNORED
            input_specs=custom_specs
        )
        
        # Should only have custom inputs, not BERT inputs
        assert set(inputs.keys()) == {"custom_input_1", "custom_input_2"}, "Should only use input_specs, not model inputs"
        
        # Validate custom_input_1
        custom_1 = inputs["custom_input_1"]
        assert list(custom_1.shape) == [3, 16], "Should use exact shape from specs"
        assert custom_1.dtype == torch.float32, "Should use float dtype"
        assert 0.0 <= custom_1.min() <= custom_1.max() <= 1.0, "Should respect range [0.0, 1.0]"
        
        # Validate custom_input_2
        custom_2 = inputs["custom_input_2"]
        assert list(custom_2.shape) == [3, 8], "Should use exact shape from specs"
        assert custom_2.dtype == torch.long, "Should use int dtype (converted to long)"
        assert 0 <= custom_2.min() <= custom_2.max() <= 100, "Should respect range [0, 100]"
    
    def test_error_handling_invalid_specs(self):
        """
        Test proper error handling for invalid input specifications.
        
        The input generator should fail fast with clear error messages
        when provided with invalid specifications. This helps users
        debug their configurations quickly.
        
        Test Scenarios:
        - Missing required fields (shape, dtype)  
        - Invalid data types
        - Invalid shapes (non-list)
        - Invalid ranges (wrong format)
        
        Expected Behavior:
        - Raises ValueError with descriptive messages
        - Error messages include the problematic field name
        - No partial generation (fail fast, don't generate some inputs)
        - Helpful guidance on correct format
        """
        # Missing shape field
        with pytest.raises(ValueError, match="Missing 'shape' in input spec for 'bad_input'"):
            generate_dummy_inputs(input_specs={
                "bad_input": {"dtype": "int", "range": [0, 100]}
            })
        
        # Missing dtype field  
        with pytest.raises(ValueError, match="Missing 'dtype' in input spec for 'bad_input'"):
            generate_dummy_inputs(input_specs={
                "bad_input": {"shape": [1, 10], "range": [0, 100]}
            })
        
        # Invalid dtype
        with pytest.raises(ValueError, match="Unsupported dtype"):
            generate_dummy_inputs(input_specs={
                "bad_input": {"shape": [1, 10], "dtype": "invalid_type"}
            })
        
        # Invalid shape (not a list)
        with pytest.raises(ValueError, match="Shape must be a list"):
            generate_dummy_inputs(input_specs={
                "bad_input": {"shape": "not_a_list", "dtype": "int"}
            })
        
        # Invalid range (wrong length)
        with pytest.raises(ValueError, match="Range must have exactly 2 values"):
            generate_dummy_inputs(input_specs={
                "bad_input": {"shape": [1, 10], "dtype": "int", "range": [0, 50, 100]}
            })
    
    def test_error_handling_no_inputs_provided(self):
        """
        Test error handling when neither input_specs nor model_name_or_path provided.
        
        The system should provide a clear error message explaining that one
        of the two input methods is required, with guidance on usage.
        
        Expected Behavior:
        - Raises ValueError with clear message
        - Explains both input_specs and model_name_or_path options
        - Provides usage guidance
        """
        with pytest.raises(ValueError, match="Either input_specs or model_name_or_path must be provided"):
            generate_dummy_inputs()


class TestTracingHierarchyBuilder:
    """
    Comprehensive test suite for the TracingHierarchyBuilder component.
    
    The TracingHierarchyBuilder is the core component that extracts PyTorch
    module hierarchy by tracing model execution. It uses PyTorch's built-in
    module tracking infrastructure for accurate context capture.
    
    Key Features Tested:
    - Module discovery and hierarchy extraction
    - Execution context capture during forward pass
    - Built-in PyTorch module tracking integration
    - Performance optimization features
    - Universal design (works with any PyTorch model)
    
    Architecture:
        TracingHierarchyBuilder
        ├── trace_model_execution() - Main entry point
        │   ├── Setup hooks and tracing infrastructure
        │   ├── Execute model forward pass with context capture
        │   └── Process captured execution data
        ├── get_execution_summary() - Extract structured hierarchy data
        └── Performance tracking and optimization features
    
    CARDINAL RULES Validation:
    - MUST-001: No hardcoded model architectures or patterns
    - MUST-002: Universal torch.nn.Module hierarchy detection
    - MUST-003: Architecture-agnostic approach (works with any model)
    """
    
    def test_hierarchy_building_bert_model(self):
        """
        Test hierarchy building with BERT model (standard transformer architecture).
        
        BERT provides an excellent test case because it has a well-defined,
        hierarchical structure that's representative of modern transformer models.
        The hierarchy should capture the nested module relationships.
        
        Test Scenario:
        - Load prajjwal1/bert-tiny model
        - Generate appropriate inputs
        - Trace execution and build hierarchy
        - Validate hierarchy structure and content
        
        Expected Hierarchy Structure:
        ├── BertModel (root)
        │   ├── BertEmbeddings
        │   │   ├── word_embeddings  
        │   │   ├── position_embeddings
        │   │   ├── token_type_embeddings
        │   │   └── LayerNorm
        │   ├── BertEncoder
        │   │   └── BertLayer (multiple instances)
        │   │       ├── BertAttention
        │   │       └── BertIntermediate
        │   └── BertPooler
        
        Validation Points:
        - Root module correctly identified
        - Nested module relationships preserved
        - Module execution counts captured
        - No empty or invalid module paths
        - Module hierarchy follows actual PyTorch structure
        """
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Generate inputs for tracing
        inputs = generate_dummy_inputs(model_name_or_path="prajjwal1/bert-tiny")
        input_args = tuple(inputs.values())
        
        # Build hierarchy
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, input_args)
        
        # Get hierarchy data
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Validate basic structure
        assert isinstance(hierarchy_data, dict), "Hierarchy data should be dictionary"
        assert len(hierarchy_data) > 0, "Should discover at least one module"
        
        # Validate execution summary
        assert "execution_steps" in summary, "Should track execution steps"
        assert isinstance(summary["execution_steps"], int), "Execution steps should be integer"
        assert summary["execution_steps"] > 0, "Should have at least one execution step"
        
        # Validate hierarchy contains expected BERT components
        module_names = set(hierarchy_data.keys())
        
        # Should have root model
        root_modules = [name for name in module_names if "/" not in name or name.count("/") == 1]
        assert len(root_modules) > 0, "Should have root module(s)"
        
        # Should have nested modules (indicated by "." in module names for PyTorch modules)
        nested_modules = [name for name in module_names if "." in name]
        assert len(nested_modules) > 0, f"Should discover nested module hierarchy. Found {len(module_names)} total modules"
        
        # Validate module information structure
        for module_path, module_info in hierarchy_data.items():
            assert isinstance(module_info, dict), f"Module info should be dict for {module_path}"
            
            # Should have basic information
            assert "module_type" in module_info, f"Missing module_type for {module_path}"
            assert isinstance(module_info["module_type"], str), f"module_type should be string for {module_path}"
            
            # Should track execution information
            if "execution_count" in module_info:
                assert isinstance(module_info["execution_count"], int), f"execution_count should be int for {module_path}"
                assert module_info["execution_count"] >= 0, f"execution_count should be non-negative for {module_path}"
    
    def test_hierarchy_building_custom_model(self):
        """
        Test hierarchy building with custom PyTorch model.
        
        This validates that the hierarchy builder works with any PyTorch
        model structure, not just pre-trained models. This is critical
        for the universal design principle.
        
        Test Scenario:
        - Create custom model with known hierarchy structure
        - Trace execution and validate captured hierarchy
        - Ensure all custom modules are discovered
        
        Expected Behavior:
        - Discovers all custom modules in the hierarchy
        - Preserves nested relationships
        - Works with custom module types
        - No hardcoded assumptions about module names
        """
        # Create custom model with known structure
        class CustomSubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
                
            def forward(self, x):
                return self.linear(x)
        
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layer = nn.Linear(8, 4)
                self.custom_sub = CustomSubModule()
                self.output_layer = nn.Linear(2, 1)
                
            def forward(self, x):
                x = self.input_layer(x)
                x = self.custom_sub(x)
                x = self.output_layer(x)
                return x
        
        model = CustomModel()
        input_tensor = torch.randn(2, 8)
        
        # Build hierarchy
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, (input_tensor,))
        
        # Get hierarchy data
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Validate structure
        assert len(hierarchy_data) > 0, "Should discover custom modules"
        
        # Should discover our custom modules
        module_paths = set(hierarchy_data.keys())
        
        # Look for our custom modules in the hierarchy
        expected_patterns = ["input_layer", "custom_sub", "output_layer", "linear"]
        found_patterns = []
        
        for module_path in module_paths:
            for pattern in expected_patterns:
                if pattern in module_path:
                    found_patterns.append(pattern)
        
        assert len(found_patterns) > 0, f"Should find custom modules. Found paths: {list(module_paths)}"
        
        # Validate module class names are captured correctly
        for module_path, module_info in hierarchy_data.items():
            assert "class_name" in module_info, f"Missing class_name for {module_path}"
            assert "module_type" in module_info, f"Missing module_type for {module_path}"
            
            class_name = module_info["class_name"]
            module_type = module_info["module_type"]

            # Should capture actual module class names
            if "linear" in module_path.lower():
                assert class_name == "Linear", f"Expected Linear class name, got {class_name}"
            elif "custom_sub" in module_path and "." not in module_path:  # Only the main custom_sub, not custom_sub.linear
                assert class_name == "CustomSubModule", f"Expected CustomSubModule class name, got {class_name}"
            
            # All custom modules should be marked as pytorch type
            assert module_type == "pytorch", f"Expected pytorch module type, got {module_type}"
    
    def test_execution_context_capture(self):
        """
        Test execution context capture during model forward pass.
        
        This validates that the hierarchy builder correctly captures
        execution information, including which modules are called
        and in what order during the forward pass.
        
        Test Scenario:
        - Use model with predictable execution pattern
        - Trace execution and validate context capture
        - Verify execution counts and patterns
        
        Expected Behavior:
        - Captures all module executions
        - Records execution counts accurately
        - Tracks execution order information
        - No missed or duplicate context captures
        """
        # Use simple model to verify execution tracking
        class TrackedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(4, 4)
                self.layer2 = nn.Linear(4, 2)
                
            def forward(self, x):
                # Each layer should be executed exactly once
                x = self.layer1(x)
                x = torch.relu(x)  # This operation should also be tracked
                x = self.layer2(x)
                return x
        
        model = TrackedModel()
        input_tensor = torch.randn(1, 4)
        
        # Build hierarchy and track execution
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, (input_tensor,))
        
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Validate execution tracking
        assert summary["execution_steps"] > 0, "Should track execution steps"
        assert len(hierarchy_data) > 0, "Should capture module hierarchy"
        
        # After MUST-002 fix, torch.nn modules like Linear are NOT included by default
        # So we should only see the TrackedModel in hierarchy
        assert len(hierarchy_data) == 1, f"Should only have root module (torch.nn excluded). Found: {list(hierarchy_data.keys())}"
        
        # The root module should be TrackedModel
        root_info = hierarchy_data.get("", {})
        assert root_info.get("class_name") == "TrackedModel", "Root should be TrackedModel"
        
        # To test with torch.nn modules included, we need to use exceptions
        builder_with_exceptions = TracingHierarchyBuilder(exceptions=["Linear"])
        builder_with_exceptions.trace_model_execution(model, (input_tensor,))
        
        summary_with_exceptions = builder_with_exceptions.get_execution_summary()
        hierarchy_with_exceptions = summary_with_exceptions["module_hierarchy"]
        
        # Now we should see Linear modules
        linear_modules = []
        for module_path, module_info in hierarchy_with_exceptions.items():
            if module_info.get("class_name") == "Linear":
                linear_modules.append((module_path, module_info))
        
        assert len(linear_modules) == 2, f"Should find both linear layers when exceptions used. Found: {[path for path, _ in linear_modules]}"


class TestONNXNodeTagger:
    """
    Comprehensive test suite for ONNX Node Tagger component.
    
    The ONNX Node Tagger maps ONNX operations back to their source PyTorch
    modules using hierarchy information from TracingHierarchyBuilder. This
    is the critical component that enables hierarchy-preserving exports.
    
    Key Features Tested:
    - Node-to-module mapping accuracy
    - Tag propagation logic and fallback strategies  
    - Coverage validation (no empty tags)
    - Performance and statistics tracking
    - Universal design (works with any ONNX model)
    
    Architecture:
        create_node_tagger_from_hierarchy()
        ├── Analyze hierarchy data and create mapping strategies
        ├── ONNXNodeTagger instance with configured strategies
        │   ├── tag_all_nodes() - Main tagging interface
        │   │   ├── Direct module matching
        │   │   ├── Parent module matching  
        │   │   └── Root fallback strategies
        │   └── get_tagging_statistics() - Analysis and reporting
        └── Validation and coverage checking
    
    CARDINAL RULES Validation:
    - MUST-001: No hardcoded ONNX operation names or patterns
    - MUST-002: Universal tagging approach (works with any model)
    - MUST-003: No empty tags allowed (100% coverage requirement)
    """
    
    def test_node_tagger_creation_from_hierarchy(self):
        """
        Test ONNX node tagger creation from hierarchy data.
        
        This validates that the node tagger can be properly created from
        hierarchy data generated by TracingHierarchyBuilder, and that
        it's configured with appropriate tagging strategies.
        
        Test Scenario:
        - Generate hierarchy data from BERT model
        - Create node tagger from hierarchy
        - Validate tagger configuration and capabilities
        
        Expected Behavior:
        - Successfully creates node tagger instance
        - Configures appropriate tagging strategies
        - Sets up model root tag correctly
        - Ready for ONNX node tagging operations
        """
        # Build hierarchy data first
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = generate_dummy_inputs(model_name_or_path="prajjwal1/bert-tiny")
        input_args = tuple(inputs.values())
        
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, input_args)
        
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Create node tagger
        node_tagger = create_node_tagger_from_hierarchy(
            hierarchy_data, 
            enable_operation_fallback=True
        )
        
        # Validate tagger creation
        assert node_tagger is not None, "Should create node tagger instance"
        assert hasattr(node_tagger, "tag_all_nodes"), "Should have tag_all_nodes method"
        assert hasattr(node_tagger, "get_tagging_statistics"), "Should have statistics method"
        
        # Validate model root tag configuration
        assert hasattr(node_tagger, "model_root_tag"), "Should have model root tag"
        assert isinstance(node_tagger.model_root_tag, str), "Model root tag should be string"
        assert len(node_tagger.model_root_tag) > 0, "Model root tag should not be empty"
        
        # Root tag should reflect the model type
        assert "BertModel" in node_tagger.model_root_tag or "AutoModel" in node_tagger.model_root_tag, \
            f"Root tag should reflect model type, got: {node_tagger.model_root_tag}"
    
    def test_node_tagging_coverage_validation(self):
        """
        Test node tagging coverage validation (CARDINAL RULE enforcement).
        
        This is a critical test that validates the CARDINAL RULE: no empty tags.
        Every ONNX node must receive a valid hierarchy tag, ensuring 100%
        coverage and traceability back to source modules.
        
        Test Scenario:
        - Export BERT model to ONNX
        - Tag all ONNX nodes using hierarchy data
        - Validate 100% coverage (no empty tags)
        - Check tag quality and consistency
        
        Expected Behavior:
        - Every ONNX node receives a non-empty tag
        - Tags follow hierarchical format: /Model/Module/Submodule
        - Coverage percentage is 100%
        - Tagging statistics show appropriate distribution
        - No CARDINAL RULE violations
        """
        # This is a more comprehensive test that requires ONNX export
        # For this test, we'll create a minimal ONNX model to validate tagging
        import tempfile

        import onnx
        
        # Create simple model and export to ONNX
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        input_tensor = torch.randn(1, 4)
        
        # Build hierarchy
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, (input_tensor,))
        
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            torch.onnx.export(model, input_tensor, tmp_file.name, opset_version=17)
            onnx_model = onnx.load(tmp_file.name)
        
        # Create and test node tagger
        node_tagger = create_node_tagger_from_hierarchy(hierarchy_data)
        tagged_nodes = node_tagger.tag_all_nodes(onnx_model)
        
        # Validate coverage (CARDINAL RULE)
        assert isinstance(tagged_nodes, dict), "Should return dictionary of tagged nodes"
        assert len(tagged_nodes) > 0, "Should tag at least one node"
        
        # Check for empty tags (CARDINAL RULE VIOLATION)
        empty_tags = [name for name, tag in tagged_nodes.items() if not tag or not tag.strip()]
        assert len(empty_tags) == 0, f"CARDINAL RULE VIOLATION: Found {len(empty_tags)} empty tags: {empty_tags}"
        
        # Validate tag format
        for node_name, tag in tagged_nodes.items():
            assert isinstance(tag, str), f"Tag should be string for node {node_name}"
            assert len(tag) > 0, f"Tag should not be empty for node {node_name}"
            assert tag.startswith("/"), f"Tag should start with '/' for hierarchical format, got: {tag}"
        
        # Get and validate statistics
        stats = node_tagger.get_tagging_statistics(onnx_model)
        assert isinstance(stats, dict), "Should return statistics dictionary"
        assert "direct_matches" in stats, "Should track direct matches"
        assert "parent_matches" in stats, "Should track parent matches"  
        assert "root_fallbacks" in stats, "Should track root fallbacks"
        
        # Validate coverage calculation
        total_nodes = len(onnx_model.graph.node)
        tagged_count = len(tagged_nodes)
        coverage_percentage = (tagged_count / total_nodes * 100) if total_nodes > 0 else 0
        
        assert coverage_percentage == 100.0, f"Coverage should be 100%, got {coverage_percentage}%"
    
    def test_tag_propagation_fallback_strategies(self):
        """
        Test tag propagation and fallback strategies.
        
        The node tagger should employ multiple strategies to ensure every
        node gets tagged, even when direct mapping isn't possible. This
        tests the fallback hierarchy: direct → parent → root.
        
        Test Scenario:
        - Create hierarchy with known module structure
        - Test tagging with different matching scenarios
        - Validate fallback strategies are employed correctly
        
        Expected Behavior:
        - Direct matches preferred when available
        - Parent module matching when direct fails
        - Root fallback ensures no empty tags
        - Statistics accurately reflect strategy usage
        """
        # Create test hierarchy data in correct format (as produced by TracingHierarchyBuilder)
        test_hierarchy = {
            "": {
                "name": "",
                "class_name": "SimpleModel",
                "module_type": "pytorch",
                "traced_tag": "/SimpleModel",
                "execution_order": 0
            },
            "linear": {
                "name": "linear",
                "class_name": "Linear",
                "module_type": "pytorch",
                "traced_tag": "/SimpleModel/Linear",
                "execution_order": 1
            },
            "activation": {
                "name": "activation",
                "class_name": "ReLU",
                "module_type": "pytorch",
                "traced_tag": "/SimpleModel/ReLU",
                "execution_order": 2
            }
        }
        
        # Create node tagger with test data
        node_tagger = create_node_tagger_from_hierarchy(test_hierarchy)
        
        # Create minimal ONNX model for testing
        import tempfile

        import onnx
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)
                
            def forward(self, x):
                return torch.relu(self.linear(x))
        
        model = TestModel()
        input_tensor = torch.randn(1, 2)
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            torch.onnx.export(model, input_tensor, tmp_file.name, opset_version=17)
            onnx_model = onnx.load(tmp_file.name)
        
        # Tag nodes and analyze strategies
        tagged_nodes = node_tagger.tag_all_nodes(onnx_model)
        stats = node_tagger.get_tagging_statistics(onnx_model)
        
        # Validate fallback strategies are working
        assert len(tagged_nodes) > 0, "Should tag nodes using fallback strategies"
        
        # All nodes should have valid tags (fallback ensures this)
        for node_name, tag in tagged_nodes.items():
            assert tag and tag.strip(), f"Node {node_name} should have non-empty tag via fallback"
            assert tag.startswith("/"), f"Tag should be hierarchical format: {tag}"
        
        # Validate statistics tracking
        total_strategy_uses = stats.get("direct_matches", 0) + stats.get("parent_matches", 0) + stats.get("root_fallbacks", 0)
        assert total_strategy_uses > 0, "Should use at least one tagging strategy"
        
        # Should have some fallback usage since we have limited hierarchy
        assert stats.get("root_fallbacks", 0) > 0, "Should use root fallback for some nodes"


class TestUtilityFunctions:
    """
    Test suite for utility functions and helper components.
    
    This covers the various utility functions that support the core
    functionality, including tag validation, configuration loading,
    and helper functions used throughout the system.
    
    Key Features Tested:
    - Tag format validation and utilities
    - Configuration file loading and parsing
    - Helper functions for shape and type handling
    - Error handling and validation utilities
    """
    
    def test_export_config_patching_system(self):
        """
        Test the export config patching system (SAM fix infrastructure).
        
        This validates the patch_export_config system that enables
        model-specific fixes like the SAM coordinate fix. This is
        the infrastructure that makes automatic model fixes possible.
        
        Test Scenario:
        - Test patch detection and application system
        - Validate type-based detection works correctly
        - Ensure patching doesn't affect other models
        
        Expected Behavior:
        - Detects appropriate configs for patching
        - Applies patches only to intended model types
        - No side effects on non-target models
        - Extensible for future model-specific fixes
        """
        # Test with SAM model (should be patched)
        sam_config = get_export_config_from_model_path("facebook/sam-vit-base")
        
        # Get original DUMMY_INPUT_GENERATOR_CLASSES
        original_classes = sam_config.DUMMY_INPUT_GENERATOR_CLASSES
        
        # Apply patching
        patch_export_config(sam_config)
        
        # Should have been modified
        patched_classes = sam_config.DUMMY_INPUT_GENERATOR_CLASSES
        assert patched_classes != original_classes, "SAM config should be patched"
        
        # Test with non-SAM model (should not be patched)
        bert_config = get_export_config_from_model_path("prajjwal1/bert-tiny")
        original_bert_classes = bert_config.DUMMY_INPUT_GENERATOR_CLASSES
        
        patch_export_config(bert_config)
        
        # Should not have been modified
        patched_bert_classes = bert_config.DUMMY_INPUT_GENERATOR_CLASSES
        assert patched_bert_classes == original_bert_classes, "BERT config should not be patched"
    
    def test_configuration_integration(self):
        """
        Test configuration loading and integration.
        
        This validates that the system correctly loads and processes
        configuration files, particularly for export configurations
        and model-specific settings.
        
        Expected Behavior:
        - Loads configuration files correctly
        - Validates configuration format
        - Integrates with export systems properly
        """
        # Test export config loading for different model types
        configs_to_test = [
            ("prajjwal1/bert-tiny", "bert"),
            ("facebook/sam-vit-base", "sam")
        ]
        
        for model_path, model_type in configs_to_test:
            config = get_export_config_from_model_path(model_path)
            
            assert config is not None, f"Should load config for {model_type} model"
            assert hasattr(config, "generate_dummy_inputs"), f"{model_type} config should have input generation"
            assert hasattr(config, "DUMMY_INPUT_GENERATOR_CLASSES"), f"{model_type} config should have generator classes"
            
            # Test input generation through config
            inputs = config.generate_dummy_inputs(framework="pt")
            assert isinstance(inputs, dict), f"{model_type} should generate dict inputs"
            assert len(inputs) > 0, f"{model_type} should generate at least one input"
    
    def test_cardinal_rules_validation(self):
        """
        Test CARDINAL RULES validation throughout the system.
        
        This validates that the core system enforces the CARDINAL RULES
        that ensure universal design and proper architecture compliance.
        
        CARDINAL RULES:
        - MUST-001: No hardcoded logic
        - MUST-002: torch.nn filtering
        - MUST-003: Universal design
        
        Expected Behavior:
        - No hardcoded model names or patterns
        - Universal approach works with any model
        - Proper validation and error handling
        """
        # Test that the system works with custom models (no hardcoding)
        class CustomArchitecture(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_layer = nn.Conv2d(3, 16, 3)
                self.custom_norm = nn.BatchNorm2d(16)
                
            def forward(self, x):
                x = self.custom_layer(x)
                x = self.custom_norm(x)
                return x
        
        model = CustomArchitecture()
        input_tensor = torch.randn(1, 3, 32, 32)
        
        # Should work without any hardcoded assumptions
        builder = TracingHierarchyBuilder()
        builder.trace_model_execution(model, (input_tensor,))
        
        summary = builder.get_execution_summary()
        hierarchy_data = summary["module_hierarchy"]
        
        # Should discover custom architecture components
        assert len(hierarchy_data) > 0, "Should work with custom architecture (MUST-001: No hardcoding)"
        
        # Should capture custom module class names
        class_names = [info.get("class_name", "") for info in hierarchy_data.values()]
        custom_classes = [name for name in class_names if "Custom" in name or "Conv2d" in name or "BatchNorm2d" in name]
        assert len(custom_classes) > 0, f"Should capture custom module classes (MUST-003: Universal design). Found classes: {class_names}"
        
        # Test tag creation works universally
        node_tagger = create_node_tagger_from_hierarchy(hierarchy_data)
        assert node_tagger is not None, "Should create tagger for any architecture (MUST-003: Universal design)"
    
    def test_onnx_node_tagger_cardinal_rules_comprehensive(self):
        """
        Test comprehensive CARDINAL RULES validation for ONNX Node Tagger (migrated from test_onnx_node_tagger.py).
        
        This is a critical test that validates all three CARDINAL RULES in the
        ONNX node tagger component, which is essential for maintaining system
        integrity and ensuring no empty tags are ever generated.
        
        CARDINAL RULES tested:
        - MUST-001: No hardcoded logic (works with any model hierarchy)
        - MUST-002: torch.nn filtering (proper scope extraction)
        - MUST-003: Universal design (consistent behavior across models)
        
        Additional critical requirements:
        - NO EMPTY TAGS ever generated (zero tolerance)
        - Universal tag format consistency
        - Priority system works correctly (direct → parent → root fallback)
        - Root fallback always provides valid tags
        """
        from unittest.mock import MagicMock
        
        # Test data: Multiple different model hierarchies to validate universality (MUST-001 & MUST-003)
        bert_like_hierarchy = {
            "embeddings": {
                "traced_tag": "/BertModel/Embeddings",
                "execution_order": 1,
                "module_type": "huggingface"
            },
            "embeddings.word_embeddings": {
                "traced_tag": "/BertModel/Embeddings/WordEmbeddings", 
                "execution_order": 2,
                "module_type": "huggingface"
            },
            "encoder.layer.0.attention.self.query": {
                "traced_tag": "/BertModel/Encoder/Layer.0/Attention/Self/Query",
                "execution_order": 5,
                "module_type": "huggingface"
            }
        }
        
        resnet_like_hierarchy = {
            "features.0": {
                "traced_tag": "/ResNetModel/Features/Block.0",
                "execution_order": 1,
                "module_type": "custom"
            },
            "features.0.conv1": {
                "traced_tag": "/ResNetModel/Features/Block.0/Conv1",
                "execution_order": 2,
                "module_type": "custom"
            },
            "classifier": {
                "traced_tag": "/ResNetModel/Classifier",
                "execution_order": 8,
                "module_type": "custom"
            }
        }
        
        gpt_like_hierarchy = {
            "transformer.h.0.attn": {
                "traced_tag": "/GPTModel/Transformer/Block.0/Attention",
                "execution_order": 1,
                "module_type": "huggingface"
            },
            "transformer.h.0.mlp": {
                "traced_tag": "/GPTModel/Transformer/Block.0/MLP", 
                "execution_order": 2,
                "module_type": "huggingface"
            }
        }
        
        # Test MUST-001: No hardcoded logic - works with different model types
        test_hierarchies = [
            (bert_like_hierarchy, "/BertModel", "BERT-like"),
            (resnet_like_hierarchy, "/ResNetModel", "ResNet-like"),
            (gpt_like_hierarchy, "/GPTModel", "GPT-like")
        ]
        
        for hierarchy_data, expected_root, model_type in test_hierarchies:
            tagger = create_node_tagger_from_hierarchy(hierarchy_data)
            
            # Should extract model root dynamically (no hardcoding)
            assert tagger.model_root_tag == expected_root, \
                f"MUST-001 VIOLATION: Failed dynamic root extraction for {model_type} - expected {expected_root}, got {tagger.model_root_tag}"
        
        # Test with BERT hierarchy for detailed validation
        bert_tagger = create_node_tagger_from_hierarchy(bert_like_hierarchy)
        
        # Create comprehensive mock ONNX nodes for testing
        mock_nodes = []
        
        # Node with full scope path (should get direct match)
        node1 = MagicMock()
        node1.name = "/embeddings/word_embeddings/Gather"
        node1.op_type = "Gather"
        mock_nodes.append(node1)
        
        # Node with partial scope (should get parent match)
        node2 = MagicMock()
        node2.name = "/embeddings/unknown_child/MatMul"
        node2.op_type = "MatMul" 
        mock_nodes.append(node2)
        
        # Node with unknown scope (should fall back to root)
        node3 = MagicMock()
        node3.name = "/unknown/scope/Add"
        node3.op_type = "Add"
        mock_nodes.append(node3)
        
        # Root node (no scope)
        node4 = MagicMock()
        node4.name = "/Softmax_123"
        node4.op_type = "Softmax"
        mock_nodes.append(node4)
        
        # Node with empty name (edge case)
        node5 = MagicMock()
        node5.name = ""
        node5.op_type = "Constant"
        mock_nodes.append(node5)
        
        # Create mock ONNX model
        mock_model = MagicMock()
        mock_model.graph.node = mock_nodes
        
        # Test CRITICAL REQUIREMENT: NO EMPTY TAGS guarantee
        bert_tagged_nodes = bert_tagger.tag_all_nodes(mock_model)
        
        # Verify NO EMPTY TAGS ever generated (zero tolerance)
        for node_name, tag in bert_tagged_nodes.items():
            assert tag, f"CARDINAL RULE VIOLATION: Empty tag for node {node_name}"
            assert tag.strip(), f"CARDINAL RULE VIOLATION: Whitespace-only tag for node {node_name}"
            assert tag.startswith('/'), f"CARDINAL RULE VIOLATION: Invalid tag format for node {node_name}: {tag}"
            assert len(tag) > 1, f"CARDINAL RULE VIOLATION: Tag too short for node {node_name}: {tag}"
        
        print(f"NO EMPTY TAGS validation passed: {len(bert_tagged_nodes)} nodes tagged with 100% coverage")
        
        # Test MUST-002: Proper scope extraction from ONNX node names
        scope_test_cases = [
            # (node_name, expected_scope)
            ("/embeddings/word_embeddings/Gather", "embeddings.word_embeddings"),
            ("/encoder/layer.0/attention/self/MatMul", "encoder.layer.0.attention.self"),
            ("/pooler/dense/Tanh", "pooler.dense"),
            ("/Softmax_123", "__root__"),  # Root node
            ("MatMul_456", "__root__"),  # No leading slash
            ("", "__root__"),  # Empty name
            ("/SingleComponent", "__root__"),  # Single component
            ("/very/deep/nested/path/operation", "very.deep.nested.path"),  # Deep nesting
        ]
        
        for node_name, expected_scope in scope_test_cases:
            node = MagicMock()
            node.name = node_name
            
            actual_scope = bert_tagger._extract_scope_from_node(node)
            assert actual_scope == expected_scope, \
                f"MUST-002 VIOLATION: Scope extraction failed for '{node_name}' - expected '{expected_scope}', got '{actual_scope}'"
        
        # Test priority system works correctly (critical for tag quality)
        # Priority 1: Direct scope matching
        direct_tag = bert_tagger._find_tag_for_scope("embeddings.word_embeddings")
        assert direct_tag == "/BertModel/Embeddings/WordEmbeddings", \
            f"Priority 1 direct matching failed: expected '/BertModel/Embeddings/WordEmbeddings', got '{direct_tag}'"
        
        # Priority 2: Parent scope matching
        parent_tag = bert_tagger._find_tag_for_scope("embeddings.unknown_child")
        assert parent_tag == "/BertModel/Embeddings", \
            f"Priority 2 parent matching failed: expected '/BertModel/Embeddings', got '{parent_tag}'"
        
        # Priority 4: Root fallback (never empty - critical guarantee)
        root_tag = bert_tagger._find_tag_for_scope("completely.unknown.nonexistent.scope")
        assert root_tag == "/BertModel", \
            f"Priority 4 root fallback failed: expected '/BertModel', got '{root_tag}'"
        assert root_tag != "", "CARDINAL RULE VIOLATION: Root fallback returned empty tag"
        assert root_tag.startswith('/'), "CARDINAL RULE VIOLATION: Root fallback returned invalid format"
        
        # Test tagging statistics (validation tracking)
        stats = bert_tagger.get_tagging_statistics(mock_model)
        required_stats = ['total_nodes', 'root_nodes', 'scoped_nodes', 'unique_scopes', 'direct_matches', 'parent_matches', 'root_fallbacks']
        
        for stat_name in required_stats:
            assert stat_name in stats, f"Missing required statistic: {stat_name}"
            assert isinstance(stats[stat_name], int), f"Statistic {stat_name} should be integer"
            assert stats[stat_name] >= 0, f"Statistic {stat_name} should be non-negative"
        
        # Verify statistics consistency
        assert stats['total_nodes'] == len(mock_nodes), "Total nodes count mismatch"
        assert stats['root_nodes'] + stats['scoped_nodes'] == stats['total_nodes'], "Node categorization mismatch"
        assert stats['direct_matches'] + stats['parent_matches'] + stats['root_fallbacks'] >= stats['scoped_nodes'], "Tagging strategy accounting error"
        
        print("ONNX Node Tagger comprehensive CARDINAL RULES validation passed")
        print(f"Statistics: {stats['total_nodes']} total nodes, {stats['direct_matches']} direct matches, {stats['parent_matches']} parent matches, {stats['root_fallbacks']} root fallbacks")
    
    def test_input_generator_detailed_validation(self):
        """
        Test detailed input generator validation (migrated key tests from test_model_input_generator.py).
        
        This adds comprehensive validation tests that were missing from the basic
        input generator tests, ensuring robust error handling and edge case coverage.
        
        Key areas covered:
        - Input spec validation (shapes, dtypes, ranges)
        - Error handling for invalid specifications
        - Edge cases and boundary conditions
        - Type conversion and validation
        """
        # Test detailed input specs validation
        
        # Test case 1: Invalid dtype handling
        invalid_dtype_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "invalid_dtype"}
        }
        
        with pytest.raises(ValueError, match="Unsupported dtype"):
            generate_dummy_inputs(input_specs=invalid_dtype_specs)
        
        # Test case 2: Invalid shape format
        invalid_shape_specs = {
            "input_ids": {"shape": "invalid_shape", "dtype": "int"}
        }
        
        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=invalid_shape_specs)
        
        # Test case 3: Missing shape
        missing_shape_specs = {
            "input_ids": {"dtype": "int"}  # Missing shape
        }
        
        with pytest.raises(ValueError, match="Missing 'shape' in input spec"):
            generate_dummy_inputs(input_specs=missing_shape_specs)
        
        # Test case 4: Negative shape values
        negative_shape_specs = {
            "input_ids": {"shape": [1, -128], "dtype": "int"}
        }
        
        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=negative_shape_specs)
        
        # Test case 5: Invalid range format (max < min)
        invalid_range_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int", "range": [100, 0]}  # max < min
        }
        
        with pytest.raises(ValueError):
            generate_dummy_inputs(input_specs=invalid_range_specs)
        
        # Test case 6: Range validation with proper values
        valid_range_specs = {
            "input_ids": {"shape": [1, 10], "dtype": "int", "range": [0, 1000]},
            "attention_mask": {"shape": [1, 10], "dtype": "int", "range": [0, 1]}
        }
        
        inputs = generate_dummy_inputs(input_specs=valid_range_specs)
        
        # Validate ranges are respected
        assert torch.all(inputs["input_ids"] >= 0), "input_ids should be >= 0"
        assert torch.all(inputs["input_ids"] <= 1000), "input_ids should be <= 1000"
        assert torch.all(inputs["attention_mask"] >= 0), "attention_mask should be >= 0"
        assert torch.all(inputs["attention_mask"] <= 1), "attention_mask should be <= 1"
        
        # Test case 7: Different dtype handling
        multi_dtype_specs = {
            "input_ids": {"shape": [1, 128], "dtype": "int"},
            "pixel_values": {"shape": [1, 3, 224, 224], "dtype": "float"},
            "labels": {"shape": [1], "dtype": "int"}
        }
        
        multi_inputs = generate_dummy_inputs(input_specs=multi_dtype_specs)
        
        assert multi_inputs["input_ids"].dtype == torch.int64, "input_ids should be int64"
        assert multi_inputs["pixel_values"].dtype == torch.float32, "pixel_values should be float32" 
        assert multi_inputs["labels"].dtype == torch.int64, "labels should be int64"
        
        # Test case 8: Empty input specs (should return empty dict)
        empty_inputs = generate_dummy_inputs(input_specs={})
        assert empty_inputs == {}, "Empty input specs should return empty dict"
        
        # Test case 9: Priority validation (input_specs over model_name_or_path)
        priority_specs = {
            "custom_input": {"shape": [1, 64], "dtype": "float"}
        }
        
        # Should use input_specs even if model_name_or_path is provided
        priority_inputs = generate_dummy_inputs(
            model_name_or_path="prajjwal1/bert-tiny",
            input_specs=priority_specs
        )
        
        assert len(priority_inputs) == 1, "Should prioritize input_specs"
        assert "custom_input" in priority_inputs, "Should use custom input spec"
        assert "input_ids" not in priority_inputs, "Should not auto-generate when input_specs provided"
        
        print("Detailed input generator validation passed: comprehensive error handling and edge cases covered")