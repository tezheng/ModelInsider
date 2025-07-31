#!/usr/bin/env python3
"""
Prepare for iteration 3 - Test metadata and report outputs
"""

import json
from pathlib import Path


def analyze_iteration_2_results():
    """Analyze the results from iteration 2."""
    print("📊 ITERATION 2 RESULTS ANALYSIS")
    print("=" * 60)
    
    # Check console output similarity
    iteration_2_dir = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_002")
    
    if iteration_2_dir.exists():
        console_files = list(iteration_2_dir.glob("console*.txt"))
        print(f"\nConsole output files found: {len(console_files)}")
        for f in console_files:
            print(f"  - {f.name}")
    
    # Read iteration 2 summary
    summary_files = list(Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor").glob("iteration_002*.md"))
    print(f"\nSummary files found: {len(summary_files)}")
    for f in summary_files:
        print(f"  - {f.name}")
    
    print("\n✅ Iteration 2 Achievements:")
    print("  1. Created HTPExportConfig class")
    print("  2. Removed all hardcoded values") 
    print("  3. Integrated Rich console")
    print("  4. Achieved 70% console output similarity")
    
    print("\n📋 Remaining Tasks for Iteration 3:")
    print("  1. Test metadata JSON output generation")
    print("  2. Compare metadata with baseline")
    print("  3. Test report TXT output generation")
    print("  4. Compare report with baseline")
    print("  5. Ensure all outputs match expected format")

def plan_iteration_3():
    """Plan the tasks for iteration 3."""
    print("\n\n📝 ITERATION 3 PLAN")
    print("=" * 60)
    
    tasks = [
        {
            "id": 1,
            "task": "Capture baseline metadata JSON",
            "description": "Run original HTP exporter and save metadata JSON"
        },
        {
            "id": 2,
            "task": "Capture baseline report TXT",
            "description": "Run original HTP exporter and save report TXT"
        },
        {
            "id": 3,
            "task": "Test metadata generation with v2",
            "description": "Generate metadata using refactored export monitor"
        },
        {
            "id": 4,
            "task": "Compare metadata outputs",
            "description": "Compare v2 metadata with baseline, check structure and values"
        },
        {
            "id": 5,
            "task": "Test report generation with v2",
            "description": "Generate report using refactored export monitor"
        },
        {
            "id": 6,
            "task": "Compare report outputs",
            "description": "Compare v2 report with baseline, check formatting"
        },
        {
            "id": 7,
            "task": "Fix any discrepancies",
            "description": "Update v2 to match baseline metadata/report format"
        },
        {
            "id": 8,
            "task": "Document iteration 3 results",
            "description": "Create comprehensive summary of changes and results"
        }
    ]
    
    print("\nPlanned Tasks:")
    for task in tasks:
        print(f"\n{task['id']}. {task['task']}")
        print(f"   {task['description']}")
    
    print("\n\n🎯 Expected Outcomes:")
    print("  - Metadata JSON structure matches baseline")
    print("  - Report TXT format matches baseline")
    print("  - All three outputs (console, metadata, report) working correctly")
    print("  - Ready to move to iteration 4 with further improvements")

def check_baseline_files():
    """Check what baseline files we have available."""
    print("\n\n📁 BASELINE FILES CHECK")
    print("=" * 60)
    
    baseline_dir = Path("/home/zhengte/modelexport_allmodels/temp/baseline")
    
    if baseline_dir.exists():
        files = list(baseline_dir.glob("*"))
        print(f"\nFiles in baseline directory: {len(files)}")
        for f in sorted(files):
            size = f.stat().st_size if f.is_file() else 0
            print(f"  - {f.name} ({size:,} bytes)")
            
            # Check metadata file
            if f.name.endswith("_htp_metadata.json"):
                print("    ✓ Metadata JSON found")
                with open(f) as jf:
                    data = json.load(jf)
                    print(f"    Keys: {list(data.keys())}")
            
            # Check report file  
            elif f.name.endswith("_htp_export_report.txt") or f.name.endswith("_full_report.txt"):
                print("    ✓ Report TXT found")
                with open(f) as rf:
                    lines = rf.readlines()
                    print(f"    Lines: {len(lines)}")
    else:
        print("⚠️ Baseline directory not found!")
        print("  Need to generate baseline files first")

def main():
    """Main preparation script."""
    print("🚀 PREPARING FOR ITERATION 3")
    print("=" * 60)
    
    # Analyze iteration 2 results
    analyze_iteration_2_results()
    
    # Plan iteration 3
    plan_iteration_3()
    
    # Check baseline files
    check_baseline_files()
    
    print("\n\n✅ Ready to start iteration 3!")
    print("Next step: Generate baseline metadata and report if not available")

if __name__ == "__main__":
    main()