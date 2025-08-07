"""
Text classification with ONNX models using Optimum.

This example demonstrates sentiment analysis and other text classification
tasks using exported ONNX models.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.auto_model_loader import AutoModelForONNX
from src.inference_utils import load_preprocessor, create_inference_pipeline


def sentiment_analysis_example():
    """Run sentiment analysis on sample texts."""
    
    print("=" * 60)
    print("Sentiment Analysis with ONNX")
    print("=" * 60)
    
    # Model path (you should replace with your exported model)
    model_path = Path("temp/sentiment-model-exported")
    
    # For demo, we'll show how it would work
    if not model_path.exists():
        print(f"\n⚠️  Model not found at {model_path}")
        print("\nTo run this example, export a sentiment model:")
        print("  modelexport export --model nlptown/bert-base-multilingual-uncased-sentiment \\")
        print("    --output temp/sentiment-model-exported/model.onnx")
        
        # Demo with mock data
        print("\n📝 Demo mode - showing expected output format:")
        demo_results()
        return
    
    # Load model with explicit task
    print("\n1️⃣ Loading ONNX model for text classification...")
    model = AutoModelForONNX.from_pretrained(
        model_path,
        task="text-classification"
    )
    tokenizer = load_preprocessor(model_path)
    
    # Test texts
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. Very disappointed.",
        "It's okay, nothing special.",
        "Fantastic experience, highly recommend!",
        "Waste of money, don't buy this.",
    ]
    
    print("\n2️⃣ Analyzing sentiments...")
    print("-" * 50)
    
    for text in texts:
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Inference
        outputs = model(**inputs)
        
        # Process results
        import torch
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Map to sentiment (adjust based on your model)
        sentiment_map = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive"
        }
        sentiment = sentiment_map.get(predicted_class, f"Class {predicted_class}")
        
        print(f"📝 Text: '{text[:50]}...'")
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.2%})")
        print()


def pipeline_classification_example():
    """Use HuggingFace pipeline for easier inference."""
    
    print("=" * 60)
    print("Pipeline-based Text Classification")
    print("=" * 60)
    
    model_path = Path("temp/bert-tiny-exported")
    
    if not model_path.exists():
        print(f"\n⚠️  Model not found. Showing demo mode...")
        demo_pipeline_results()
        return
    
    # Create pipeline (much simpler!)
    print("\n🔧 Creating inference pipeline...")
    pipe = create_inference_pipeline(
        model_path,
        task="text-classification",
        device=-1  # CPU
    )
    
    # Batch processing
    texts = [
        "The movie was absolutely fantastic!",
        "I'm not sure how I feel about this.",
        "This is the worst experience ever.",
        "Pretty good, would recommend.",
        "Absolutely terrible, avoid at all costs!",
    ]
    
    print("\n📊 Batch classification...")
    results = pipe(texts)
    
    print("\n📈 Results:")
    print("-" * 50)
    for text, result in zip(texts, results):
        print(f"Text: '{text}'")
        print(f"  Label: {result['label']}, Score: {result['score']:.3f}")
        print()


def multi_label_classification():
    """Demonstrate multi-label classification."""
    
    print("=" * 60)
    print("Multi-label Classification")
    print("=" * 60)
    
    print("\n📝 Multi-label classification example:")
    print("   (Classifying text into multiple categories)")
    
    # Example categories
    categories = ["Technology", "Sports", "Politics", "Entertainment", "Science"]
    
    texts = [
        "The new smartphone features AI-powered cameras and 5G connectivity.",
        "The team won the championship after a thrilling overtime victory.",
        "Scientists discover a new exoplanet that could support life.",
        "The latest Marvel movie breaks box office records worldwide.",
        "New climate policy aims to reduce emissions by 50% by 2030.",
    ]
    
    print("\n📊 Classification results (simulated):")
    print("-" * 50)
    
    import random
    random.seed(42)
    
    for text in texts:
        print(f"Text: '{text[:60]}...'")
        print("Categories:")
        
        # Simulate multi-label predictions
        scores = [random.random() for _ in categories]
        threshold = 0.5
        
        detected = []
        for cat, score in zip(categories, scores):
            if score > threshold:
                detected.append(cat)
                print(f"  ✓ {cat}: {score:.2%}")
        
        if not detected:
            max_idx = scores.index(max(scores))
            print(f"  ✓ {categories[max_idx]}: {scores[max_idx]:.2%} (highest)")
        
        print()


def demo_results():
    """Show demo results when model is not available."""
    
    print("\n📊 Expected output format:")
    print("-" * 50)
    
    demo_texts = [
        ("I absolutely love this product!", "Very Positive", 0.95),
        ("This is terrible.", "Very Negative", 0.88),
        ("It's okay, nothing special.", "Neutral", 0.72),
    ]
    
    for text, sentiment, confidence in demo_texts:
        print(f"📝 Text: '{text}'")
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.2%})")
        print()


def demo_pipeline_results():
    """Show demo pipeline results."""
    
    print("\n📊 Expected pipeline output:")
    print("-" * 50)
    
    demo_results = [
        {"text": "The movie was fantastic!", "label": "POSITIVE", "score": 0.987},
        {"text": "I'm not sure about this.", "label": "NEUTRAL", "score": 0.645},
        {"text": "Terrible experience.", "label": "NEGATIVE", "score": 0.923},
    ]
    
    for result in demo_results:
        print(f"Text: '{result['text']}'")
        print(f"  Label: {result['label']}, Score: {result['score']:.3f}")
        print()


def compare_models():
    """Compare different classification models."""
    
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    models_to_compare = [
        ("bert-tiny", "temp/bert-tiny-exported"),
        ("distilbert", "temp/distilbert-exported"),
        ("roberta", "temp/roberta-exported"),
    ]
    
    test_text = "This product exceeded my expectations in every way!"
    
    print(f"\n📝 Test text: '{test_text}'")
    print("\n📊 Comparing models (simulated):")
    print("-" * 50)
    
    import time
    import random
    
    for model_name, model_path in models_to_compare:
        print(f"\n{model_name}:")
        
        # Simulate inference time
        inference_time = random.uniform(5, 20)
        confidence = random.uniform(0.7, 0.99)
        
        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Prediction: POSITIVE ({confidence:.2%})")
        
        # Simulate model size
        model_size = random.uniform(10, 500)
        print(f"  Model size: {model_size:.1f} MB")


def main():
    """Run all text classification examples."""
    
    print("\n🚀 Text Classification with ONNX Models\n")
    
    # Sentiment analysis
    sentiment_analysis_example()
    
    # Pipeline example
    print("\n")
    pipeline_classification_example()
    
    # Multi-label
    print("\n")
    multi_label_classification()
    
    # Model comparison
    print("\n")
    compare_models()
    
    print("\n" + "=" * 60)
    print("✅ All text classification examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()