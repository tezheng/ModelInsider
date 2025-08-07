Simple ONNX Inference with Optimum (No Quantization)

  1. Using Pipelines (Simplest)

  from optimum.pipelines import pipeline

  # Just use pipeline with accelerator="ort"
  classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", accelerator="ort")
  result = classifier("I love this movie!")

  2. Direct ONNX Model Loading

  from optimum.onnxruntime import ORTModelForSequenceClassification
  from transformers import AutoTokenizer

  # Load pre-converted ONNX models from Hub
  model = ORTModelForSequenceClassification.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")
  tokenizer = AutoTokenizer.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")

  # Use it exactly like a regular transformers model
  inputs = tokenizer("I love this!", return_tensors="pt")
  outputs = model(**inputs)

  3. Auto-Convert PyTorch to ONNX

  from optimum.onnxruntime import ORTModelForSequenceClassification

  # Convert any HuggingFace model to ONNX on-the-fly
  model = ORTModelForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      export=True  # Auto-converts PyTorch → ONNX
  )

  4. Using Your Exported ONNX Models

  Since you're working on modelexport that creates ONNX files:
  from optimum.onnxruntime import ORTModelForSequenceClassification
  from transformers import AutoTokenizer

  # Load your exported ONNX model
  model = ORTModelForSequenceClassification.from_pretrained(
      "path/to/your/exported/model",
      file_name="model.onnx"  # Your exported ONNX file
  )
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  # Use with pipeline
  pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

  5. GPU Acceleration (No Quantization)

  # CUDA
  model = ORTModelForSequenceClassification.from_pretrained(
      "model-name",
      provider="CUDAExecutionProvider"
  )

  # TensorRT (still float32, just optimized)
  model = ORTModelForSequenceClassification.from_pretrained(
      "model-name",
      provider="TensorrtExecutionProvider"
  )

  Available Model Types:

  - ORTModelForSequenceClassification - Text classification
  - ORTModelForQuestionAnswering - Q&A tasks
  - ORTModelForTokenClassification - NER
  - ORTModelForCausalLM - Text generation (GPT-like)
  - ORTModelForSeq2SeqLM - Translation (T5, BART)
  - ORTModelForImageClassification - Vision models
  - ORTDiffusionPipeline - Stable Diffusion

  Complete Example:

  from optimum.onnxruntime import ORTModelForQuestionAnswering
  from transformers import AutoTokenizer, pipeline

  # Method 1: Load from your exported ONNX
  model = ORTModelForQuestionAnswering.from_pretrained("./temp/bert-tiny/")
  tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

  # Method 2: Auto-export from HuggingFace
  model = ORTModelForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", export=True)

  # Use with pipeline (most HuggingFace-like)
  qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
  result = qa_pipeline(
      question="What is my name?",
      context="My name is Claude and I help with code."
  )
  print(result)


Understanding config.json for ONNX Models in HuggingFace Optimum

  Why config.json is Required

  The config.json file is essential for ONNX models in the HuggingFace ecosystem because:

  1. Architecture Information: ONNX files only contain the computational graph - they don't know what "task" the model performs
  2. Model Metadata: Contains vocab size, hidden dimensions, number of layers, etc.
  3. Task Mapping: Tells Optimum which output to use for which task (e.g., which logits for classification)
  4. Tokenizer Integration: Provides necessary info for tokenizer compatibility
  5. Framework Compatibility: Ensures the ONNX model can be used with HuggingFace pipelines

  Must-Have Fields in config.json

  Core Required Fields:

  {
    "architectures": ["BertForQuestionAnswering"],  // Model architecture
    "model_type": "bert",                           // Base model type
    "hidden_size": 128,                             // Hidden dimension size
    "num_attention_heads": 2,                       // Attention heads
    "num_hidden_layers": 2,                         // Number of layers
    "vocab_size": 30522,                            // Vocabulary size
    "max_position_embeddings": 512,                 // Max sequence length
    "type_vocab_size": 2,                           // Token type vocab size
    "hidden_act": "gelu"                            // Activation function
  }

  Task-Specific Fields:

  For Question Answering:
  {
    "architectures": ["BertForQuestionAnswering"],
    // No additional fields needed - QA uses start/end logits
  }

  For Sequence Classification:
  {
    "architectures": ["BertForSequenceClassification"],
    "id2label": {           // Label mapping
      "0": "NEGATIVE",
      "1": "POSITIVE"
    },
    "label2id": {
      "NEGATIVE": 0,
      "POSITIVE": 1
    },
    "num_labels": 2         // Number of classes
  }

  For Token Classification (NER):
  {
    "architectures": ["BertForTokenClassification"],
    "id2label": {
      "0": "O",
      "1": "B-PER",
      "2": "I-PER",
      // ... more labels
    },
    "num_labels": 9
  }

  For Causal LM (GPT-style):
  {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "task_specific_params": {
      "text-generation": {
        "do_sample": true,
        "max_length": 50
      }
    }
  }

  How to Generate config.json

  Method 1: From Original HuggingFace Model (Recommended)

  from transformers import AutoConfig

  # Load original config
  config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")

  # Modify if needed (e.g., for custom tasks)
  config.num_labels = 3  # For 3-class classification

  # Save it
  config.save_pretrained("./path/to/onnx/model/")

  Method 2: Create Manually for Custom Models

  from transformers import BertConfig

  config = BertConfig(
      vocab_size=30522,
      hidden_size=128,
      num_hidden_layers=2,
      num_attention_heads=2,
      intermediate_size=512,
      hidden_act="gelu",
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
      type_vocab_size=2,
      architectures=["BertForSequenceClassification"],
      num_labels=2
  )

  config.save_pretrained("./path/to/onnx/model/")

  Method 3: During ONNX Export

  from optimum.onnxruntime import ORTModelForSequenceClassification
  from transformers import AutoConfig, AutoTokenizer

  # Export with config
  model = ORTModelForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      export=True
  )

  # Save everything together
  model.save_pretrained("./my_onnx_model/")
  # This automatically saves: model.onnx, config.json

  Must-Have Sources for config.json

  1. Original Model Card: Always check the model's HuggingFace page
  2. Model Architecture: Must match the ONNX model's actual architecture
  3. Tokenizer Config: Should be compatible with the model's vocabulary

  Complete Example with Your ModelExport

  # In your modelexport workflow
  import json
  from pathlib import Path
  from transformers import AutoConfig

  def export_with_config(model_name, output_dir):
      # Your existing ONNX export code
      # ... exports to output_dir/model.onnx

      # Add config.json
      config = AutoConfig.from_pretrained(model_name)
      config.save_pretrained(output_dir)

      # Also save tokenizer for convenience
      from transformers import AutoTokenizer
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      tokenizer.save_pretrained(output_dir)

      # Now the directory has everything needed:
      # output_dir/
      #   ├── model.onnx
      #   ├── config.json
      #   ├── tokenizer.json
      #   ├── tokenizer_config.json
      #   └── vocab.txt

  Verifying Your Setup

  from optimum.onnxruntime import ORTModelForQuestionAnswering

  # This will work if config.json is present and correct
  try:
      model = ORTModelForQuestionAnswering.from_pretrained("./temp/bert-tiny/")
      print("✓ Config is valid!")
  except Exception as e:
      print(f"✗ Config issue: {e}")

  Common Issues and Solutions

  1. Wrong Architecture: Ensure architectures field matches your ONNX model's task
  2. Missing num_labels: Required for classification tasks
  3. Incompatible Dimensions: Config dimensions must match ONNX model dimensions
  4. Missing model_type: Required for model initialization

  The config.json bridges the gap between the raw ONNX computational graph and the high-level HuggingFace API, making your ONNX model "understand" its purpose and integrate
  seamlessly with the ecosystem.


 # Generating Optimum-compatible model configs:

  Proposal 1: Lightweight Integration - Add Config Export to HTP Exporter

  Approach: Minimal changes to existing HTP exporter to save config.json alongside ONNX export.

  # In htp_exporter.py, add after model loading (around line 179):
  if model_name_or_path:
      from transformers import AutoConfig
      config = AutoConfig.from_pretrained(model_name_or_path)

      # Save config.json alongside ONNX model
      config_path = Path(output_path).with_suffix('.json').parent / 'config.json'
      config.save_pretrained(Path(output_path).parent)

  # Also save tokenizer if needed for full Optimum compatibility
  if model_name_or_path:
      from transformers import AutoTokenizer
      try:
          tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
          tokenizer.save_pretrained(Path(output_path).parent)
      except:
          pass  # Some models don't have tokenizers

  Pros:
  - Minimal code changes
  - Leverages existing AutoConfig loading
  - Immediate Optimum compatibility
  - Works with all HuggingFace models

  Cons:
  - Only works when model_name_or_path is provided
  - No config generation for custom models

  Proposal 2: Config Generation from Model Analysis

  Approach: Generate config.json by analyzing the exported model structure.

  class OptimumConfigGenerator:
      """Generate Optimum-compatible config from model analysis."""

      def generate_config_from_model(self, model: nn.Module, model_class: str,
                                    hierarchy_data: dict) -> dict:
          """Generate config.json from model structure."""
          config = {
              "model_type": self._infer_model_type(model_class),
              "architectures": [model_class],
              "torch_dtype": "float32",
              "_name_or_path": "exported_model",
          }

          # Analyze model structure
          if hasattr(model, 'config'):
              # Use existing config as base
              config.update(model.config.to_dict())
          else:
              # Infer from hierarchy
              config.update(self._infer_config_from_hierarchy(hierarchy_data))

          # Add task-specific fields
          config.update(self._infer_task_specific_fields(model, hierarchy_data))

          return config

      def _infer_model_type(self, model_class: str) -> str:
          """Infer model type from class name."""
          # Map common patterns
          if 'Bert' in model_class:
              return 'bert'
          elif 'GPT' in model_class:
              return 'gpt2'
          elif 'T5' in model_class:
              return 't5'
          # ... more mappings
          return 'custom'

  Pros:
  - Works with any model (including custom)
  - No dependency on model_name_or_path
  - Can infer configuration from structure

  Cons:
  - Complex inference logic needed
  - May miss model-specific config fields
  - Requires maintenance for new architectures

  Proposal 3: Deep Optimum Integration

  Approach: Use Optimum's export functionality directly with HTP enhancements.

  from optimum.exporters.onnx import main_export
  from optimum.onnxruntime import ORTModelForQuestionAnswering

  class HTPOptimumExporter(HTPExporter):
      """HTP Exporter with full Optimum integration."""

      def export(self, model=None, output_path="", model_name_or_path=None, **kwargs):
          if model_name_or_path:
              # Use Optimum's export with HTP post-processing
              from optimum.exporters.onnx import export
              from optimum.exporters.onnx.model_configs import AutoOnnxConfig

              # Get Optimum config
              onnx_config = AutoOnnxConfig.from_pretrained(model_name_or_path)

              # Export with Optimum (creates config.json automatically)
              export(
                  model_name_or_path=model_name_or_path,
                  output=Path(output_path).parent,
                  task=self._detect_task(model_name_or_path),
              )

              # Load exported ONNX and apply HTP tagging
              onnx_model = onnx.load(output_path)
              self._apply_htp_enhancements(onnx_model, model)
              onnx.save(onnx_model, output_path)
          else:
              # Fallback to standard HTP export
              super().export(model, output_path, **kwargs)

  Pros:
  - Full Optimum compatibility guaranteed
  - Leverages Optimum's robust export pipeline
  - Automatic config/tokenizer handling
  - Best practices built-in

  Cons:
  - Requires optimum as dependency
  - May conflict with custom export logic
  - Less control over export process

  Proposal 4: Hybrid Approach with Config Enhancement

  Approach: Combine HTP export with intelligent config generation.

  class HTPConfigBuilder:
      """Build Optimum-compatible configs with HTP metadata."""

      def __init__(self, htp_metadata: dict):
          self.metadata = htp_metadata

      def build_config(self, model=None, model_name_or_path=None) -> dict:
          """Build comprehensive config."""
          config = {}

          # 1. Try to load from HuggingFace
          if model_name_or_path:
              try:
                  from transformers import AutoConfig
                  config = AutoConfig.from_pretrained(model_name_or_path).to_dict()
              except:
                  pass

          # 2. Enhance with model analysis
          if model and hasattr(model, 'config'):
              config.update(model.config.to_dict())

          # 3. Add HTP-specific metadata
          config['htp_metadata'] = {
              'hierarchy_modules': len(self.metadata.get('modules', {})),
              'module_types': self.metadata.get('statistics', {}).get('module_types', []),
              'export_strategy': 'htp',
          }

          # 4. Ensure required Optimum fields
          config.setdefault('model_type', self._infer_model_type(config, model))
          config.setdefault('architectures', [model.__class__.__name__] if model else [])

          return config

  Integration in HTP Exporter:
  # In htp_exporter.py, after export completes:
  if output_path:
      # Generate and save config
      config_builder = HTPConfigBuilder(self._export_stats)
      config = config_builder.build_config(model, model_name_or_path)

      config_path = Path(output_path).parent / 'config.json'
      with open(config_path, 'w') as f:
          json.dump(config, f, indent=2)

  Pros:
  - Best of both worlds
  - Graceful fallbacks
  - Enriched with HTP metadata
  - Maintains backward compatibility

  Cons:
  - More complex implementation
  - Potential config conflicts

  Recommendation

  I recommend Proposal 4 (Hybrid Approach) because it:

  1. Maintains compatibility with your existing HTP workflow
  2. Provides immediate value by enabling Optimum usage
  3. Gracefully handles both HuggingFace and custom models
  4. Enriches configs with HTP-specific metadata
  5. Allows incremental adoption - start simple, enhance over time

  The implementation would involve:
  1. Adding a HTPConfigBuilder class
  2. Modifying HTPExporter.export() to save config.json
  3. Optional: Save tokenizer files for full compatibility
  4. Optional: Add --no-config flag to disable if needed