#!/usr/bin/env python3
"""
Test HTPExporter with various model architectures.
"""

import subprocess
from pathlib import Path

# Test models from different architectures
TEST_MODELS = [
    ("prajjwal1/bert-tiny", "BERT"),
    ("microsoft/resnet-18", "ResNet"),  
    ("facebook/sam-vit-base", "SAM"),
    ("google/vit-base-patch16-224", "ViT"),
    ("openai/clip-vit-base-patch32", "CLIP"),
]

def test_model(model_name: str, arch_name: str):
    """Test exporting a single model."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing {arch_name}: {model_name}")
    print(f"{'='*80}")
    
    output_dir = Path(f"temp/model_tests/{arch_name.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"
    
    cmd = [
        "uv", "run", "modelexport", "export",
        "--model", model_name,
        "--output", str(output_path),
        "--verbose",
        "--with-report"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {arch_name} export successful!")
            
            # Check if hierarchy is displayed correctly by looking for tree output
            if "Module Hierarchy:" in result.stdout:
                # Extract hierarchy section
                lines = result.stdout.split('\n')
                in_hierarchy = False
                hierarchy_lines = []
                
                for line in lines:
                    if "Module Hierarchy:" in line:
                        in_hierarchy = True
                    elif in_hierarchy and line.strip() and line.startswith("==="):
                        break
                    elif in_hierarchy:
                        hierarchy_lines.append(line)
                
                print("\n📊 Hierarchy Structure:")
                for line in hierarchy_lines[:20]:  # Show first 20 lines
                    print(line)
                if len(hierarchy_lines) > 20:
                    print(f"... and {len(hierarchy_lines) - 20} more lines")
            else:
                print("⚠️ No hierarchy found in output!")
            
            # Check output files
            files = list(output_dir.glob("*"))
            print(f"\n📁 Generated files: {len(files)}")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   • {f.name} ({size_mb:.1f} MB)")
                
        else:
            print(f"❌ {arch_name} export failed!")
            print(f"Error: {result.stderr}")
            if "hierarchy" in result.stdout.lower() or "tree" in result.stdout.lower():
                print("\n⚠️ Hierarchy-related output:")
                for line in result.stdout.split('\n'):
                    if any(word in line.lower() for word in ["hierarchy", "tree", "module"]):
                        print(f"   {line}")
                        
    except subprocess.TimeoutExpired:
        print(f"❌ {arch_name} export timed out!")
    except Exception as e:
        print(f"❌ {arch_name} export error: {e}")


def main():
    """Run tests on all models."""
    print("🔬 Testing HTPExporter with Various Model Architectures")
    print("=" * 80)
    
    failed_models = []
    hierarchy_issues = []
    
    for model_name, arch_name in TEST_MODELS:
        try:
            test_model(model_name, arch_name)
        except Exception as e:
            print(f"❌ Error testing {arch_name}: {e}")
            failed_models.append((model_name, arch_name))
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 Test Summary")
    print(f"{'='*80}")
    print(f"Total models tested: {len(TEST_MODELS)}")
    print(f"Failed exports: {len(failed_models)}")
    
    if failed_models:
        print("\n❌ Failed models:")
        for model_name, arch_name in failed_models:
            print(f"   • {arch_name}: {model_name}")
    
    if hierarchy_issues:
        print("\n⚠️ Models with hierarchy issues:")
        for model_name, arch_name in hierarchy_issues:
            print(f"   • {arch_name}: {model_name}")


if __name__ == "__main__":
    main()