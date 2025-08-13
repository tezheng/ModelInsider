"""
Test constants for modelexport test suite.

This module centralizes magic numbers, strings, and configuration values
used across test files to improve maintainability and reduce duplication.
"""

from __future__ import annotations

# Model dimensions and sizes
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 128
BERT_VOCAB_SIZE = 30522
BERT_HIDDEN_SIZE = 768
BERT_ATTENTION_HEADS = 12
BERT_LAYERS = 12

# Image processing constants
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_CHANNELS = 3
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Audio processing constants
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_AUDIO_LENGTH = 16000  # 1 second at 16kHz
AUDIO_FEATURE_SIZE = 1
AUDIO_DOWNSAMPLE_RATIO = 320  # Wav2Vec2 downsampling

# Video processing constants
DEFAULT_VIDEO_FRAMES = 16
VIDEO_PATCH_SIZE = 16
VIDEO_HIDDEN_PATCHES = 1568  # VideoMAE patches

# CLIP specific constants
CLIP_TEXT_LENGTH = 77
CLIP_EMBEDDING_SIZE = 512
CLIP_VOCAB_SIZE = 49408

# ONNX constants
ONNX_OPSET_VERSION = 17
ONNX_PRODUCER_NAME = "test_producer"

# Model type constants
MODEL_BERT = "bert"
MODEL_VIT = "vit"
MODEL_WAV2VEC2 = "wav2vec2"
MODEL_VIDEOMAE = "videomae"
MODEL_CLIP = "clip"

# Task constants
TASK_FEATURE_EXTRACTION = "feature-extraction"
TASK_IMAGE_CLASSIFICATION = "image-classification"
TASK_ASR = "automatic-speech-recognition"
TASK_VIDEO_CLASSIFICATION = "video-classification"
TASK_ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"

# Processor type constants
PROCESSOR_TOKENIZER = "tokenizer"
PROCESSOR_IMAGE = "image_processor"
PROCESSOR_FEATURE_EXTRACTOR = "feature_extractor"
PROCESSOR_VIDEO = "video_processor"
PROCESSOR_MULTIMODAL = "multimodal"

# Common tensor names
TENSOR_INPUT_IDS = "input_ids"
TENSOR_ATTENTION_MASK = "attention_mask"
TENSOR_TOKEN_TYPE_IDS = "token_type_ids"
TENSOR_PIXEL_VALUES = "pixel_values"
TENSOR_INPUT_VALUES = "input_values"
TENSOR_INPUT_FEATURES = "input_features"
TENSOR_LAST_HIDDEN_STATE = "last_hidden_state"
TENSOR_POOLER_OUTPUT = "pooler_output"
TENSOR_TEXT_EMBEDS = "text_embeds"
TENSOR_IMAGE_EMBEDS = "image_embeds"
TENSOR_LOGITS_PER_TEXT = "logits_per_text"
TENSOR_LOGITS_PER_IMAGE = "logits_per_image"

# Common node names for GraphML tests
NODE_BERT_EMBEDDINGS = "bert/embeddings/Gather"
NODE_BERT_ENCODER = "bert/encoder/layer_0/MatMul"
NODE_BERT_ADD = "bert/encoder/Add"

# Test performance thresholds
PERFORMANCE_PROCESSOR_CREATION_MAX = 0.1  # seconds
PERFORMANCE_TEXT_PROCESSING_MAX = 0.01  # seconds per text
PERFORMANCE_IMAGE_PROCESSING_MAX = 0.05  # seconds per image
PERFORMANCE_MEMORY_GROWTH_MAX = 100 * 1024 * 1024  # 100MB

# Test data sizes
PERFORMANCE_TEXT_SAMPLES = 50
PERFORMANCE_IMAGE_SAMPLES = 20
CONCURRENT_WORKERS = 3
CONCURRENT_ITEMS_PER_WORKER = 10
MEMORY_TEST_ITERATIONS = 1000
MEMORY_CHECK_INTERVAL = 100

# File extensions
ONNX_EXTENSION = ".onnx"
JSON_EXTENSION = ".json"
GRAPHML_EXTENSION = ".graphml"
TXT_EXTENSION = ".txt"

# Common token IDs (BERT-style)
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
UNK_TOKEN_ID = 100
MASK_TOKEN_ID = 103

# Common tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

# Test file naming patterns
FIXTURE_SUFFIX = "_fixture"
TEST_SUFFIX = "_test"
PERF_SUFFIX = "_perf"
MOCK_PREFIX = "mock_"

# Performance test limits
BATCH_PROCESSING_OVERHEAD = 1.1  # 10% overhead allowed
