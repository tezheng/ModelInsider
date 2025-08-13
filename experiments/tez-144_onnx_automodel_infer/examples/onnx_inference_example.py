#!/usr/bin/env python3
"""
ONNX Inference Example

Simple demonstration of using ONNX models with ONNXAutoProcessor.
Shows the correct factory pattern with ONNXAutoProcessor.from_model().
"""

import sys
from pathlib import Path

import click
import numpy as np

# Add src to path
sys.path.append('../src')

from enhanced_pipeline import pipeline
from onnx_auto_processor import ONNXAutoProcessor
from optimum.onnxruntime import ORTModelForFeatureExtraction


@click.command()
@click.argument(
    'model_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default='../models/bert-tiny-optimum',
    required=False
)
def main(model_dir: Path):
    """
    Simple ONNX inference example.
    
    MODEL_DIR: Path to ONNX model directory (default: ../models/bert-tiny-optimum)
    """
    print(f"Loading model from: {model_dir}")
    
    # Load ONNX model
    model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
    
    # Create processor using ONNXAutoProcessor.from_model() (primary method)
    processor = ONNXAutoProcessor.from_model(
        onnx_model_path=model_dir / "model.onnx",
        hf_model_path=model_dir
    )
    
    # Create pipeline
    pipe = pipeline(
        "feature-extraction",
        model=model,
        data_processor=processor
    )
    
    # Single text inference
    text = "The ONNX inference pipeline is incredibly fast!"
    result = pipe(text)
    print(f"Single text output shape: {np.array(result).shape}")
    
    # Batch processing
    batch = [
        "First sentence for testing.",
        "Second sentence goes here.",
        "Third sentence in the batch."
    ]
    results = pipe(batch)
    print(f"Batch output shape: {np.array(results).shape}")


if __name__ == "__main__":
    main()