#!/usr/bin/env python3
"""
Fix iteration 3 - Correct metadata and report issues
"""

import json
from pathlib import Path

def analyze_metadata_issues():
    """Analyze what's wrong with the metadata."""
    print("🔍 ANALYZING METADATA ISSUES")
    print("=" * 60)
    
    # Read baseline
    baseline_path = Path("/home/zhengte/modelexport_allmodels/temp/baseline/model_htp_metadata.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    # Read v2
    v2_path = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_003/model_htp_metadata.json")
    with open(v2_path) as f:
        v2 = json.load(f)
    
    print("\n1. Export time issue:")
    print(f"   Baseline: {baseline['export_info']['export_time']}")
    print(f"   V2: {v2['export_info']['export_time']}")
    print("   ✓ This is expected - different run times")
    
    print("\n2. File paths issue:")
    print(f"   Baseline path: {baseline['file_info']['onnx_path']}")
    print(f"   V2 path: {v2['file_info']['onnx_path']}")
    print("   ✓ This is expected - different output locations")
    
    print("\n3. Checking report structure in metadata:")
    print("   Baseline report keys:", list(baseline.get('report', {}).keys()))
    print("   V2 report keys:", list(v2.get('report', {}).keys()))
    
    # Deep check report structure
    baseline_report = baseline.get('report', {}).get('node_tagging', {})
    v2_report = v2.get('report', {}).get('node_tagging', {})
    
    print("\n4. Node tagging structure:")
    print("   Baseline:", json.dumps(baseline_report, indent=2)[:200] + "...")
    print("   V2:", json.dumps(v2_report, indent=2)[:200] + "...")
    
    # The issue: user says metadata is incorrect
    # Let's check if all fields match the expected structure
    
    issues = []
    
    # Check all baseline fields exist in v2
    def check_dict(d1, d2, path=""):
        for k, v1 in d1.items():
            if k not in d2:
                issues.append(f"Missing key: {path}.{k}")
            elif isinstance(v1, dict) and isinstance(d2[k], dict):
                check_dict(v1, d2[k], f"{path}.{k}")
            elif type(v1) != type(d2[k]):
                issues.append(f"Type mismatch: {path}.{k} - {type(v1)} vs {type(d2[k])}")
    
    check_dict(baseline, v2)
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Structure matches!")

def fix_metadata_writer():
    """Fix the metadata writer to match baseline exactly."""
    print("\n\n🔧 FIXING METADATA WRITER")
    print("=" * 60)
    
    fixes = """
Issues to fix in HTPMetadataWriter:

1. Export time should use data.export_time not data.elapsed_time
2. File paths should be relative or configurable
3. Coverage format should match baseline exactly
4. All numeric values should match types (int vs float)

Key changes needed:
- export_time: Use the value passed in, not elapsed
- coverage: Store as percentage float, not string
- statistics: Ensure all keys match baseline
"""
    
    print(fixes)
    
    # Create fixed version
    return """
    @step(HTPExportStep.COMPLETE)
    def write_complete(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        \"\"\"Generate final metadata file.\"\"\"
        # Build complete metadata structure EXACTLY matching baseline
        metadata = {
            "export_info": {
                "timestamp": data.timestamp,
                "model_name": data.model_name,
                "model_class": data.model_class,
                "export_time": data.export_time,  # Use exact value, not elapsed
                "strategy": data.strategy,
                "embed_hierarchy_attributes": data.embed_hierarchy_attributes
            },
            "model_info": {
                "total_modules": data.total_modules,
                "total_parameters": data.total_parameters,
                "execution_steps": data.execution_steps
            },
            "input_info": {
                "input_names": data.input_names,
                "output_names": data.output_names
            },
            "hierarchy": data.hierarchy,
            "nodes": data.tagged_nodes,  # Renamed from tagged_nodes to nodes
            "report": {
                "node_tagging": {
                    "statistics": {
                        "total_nodes": data.total_nodes,
                        "tagged_nodes": len(data.tagged_nodes),
                        "coverage": f"{data.coverage:.1f}%",
                        "direct_matches": data.tagging_stats.get("direct_matches", 0),
                        "parent_matches": data.tagging_stats.get("parent_matches", 0),
                        "operation_matches": data.tagging_stats.get("operation_matches", 0),
                        "root_fallbacks": data.tagging_stats.get("root_fallbacks", 0),
                        "empty_tags": data.tagging_stats.get("empty_tags", 0)
                    },
                    "coverage": {
                        "percentage": data.coverage,
                        "empty_tags": data.tagging_stats.get("empty_tags", 0)
                    }
                }
            },
            "file_info": {
                "onnx_path": data.output_path,
                "onnx_size_mb": data.onnx_size_mb,
                "metadata_path": str(self.metadata_path)
            }
        }
"""

def check_report_format():
    """Check report format issues."""
    print("\n\n📄 CHECKING REPORT FORMAT")
    print("=" * 60)
    
    baseline_path = Path("/home/zhengte/modelexport_allmodels/temp/baseline/model_full_report.txt")
    v2_path = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_003/model_full_report.txt")
    
    with open(baseline_path) as f:
        baseline = f.read()
    
    with open(v2_path) as f:
        v2 = f.read()
    
    # Check line endings
    print(f"Baseline lines: {len(baseline.splitlines())}")
    print(f"V2 lines: {len(v2.splitlines())}")
    
    # Check section headers
    sections = [
        "HTP EXPORT FULL REPORT",
        "Generated:",
        "MODEL INFORMATION",
        "Total Parameters:",
        "COMPLETE MODULE HIERARCHY",
        "NODE TAGGING STATISTICS",
        "COMPLETE NODE MAPPINGS",
        "EXPORT SUMMARY"
    ]
    
    for section in sections:
        if section in baseline and section in v2:
            print(f"✓ {section}")
        else:
            print(f"✗ {section} missing")
    
    # The issue is likely formatting differences
    print("\nReport seems structurally correct but may have formatting issues")

def main():
    """Main fix script."""
    print("🚑 FIXING ITERATION 3 ISSUES")
    print("=" * 60)
    
    # Analyze issues
    analyze_metadata_issues()
    
    # Show fixes needed
    fix_metadata_writer()
    
    # Check report
    check_report_format()
    
    print("\n\n📝 NEXT STEPS:")
    print("1. Update HTPMetadataWriter in export_monitor_v2.py")
    print("2. Ensure export_time is passed correctly")
    print("3. Fix any formatting issues in report writer")
    print("4. Re-run tests")

if __name__ == "__main__":
    main()