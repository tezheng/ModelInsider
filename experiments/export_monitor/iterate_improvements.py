#!/usr/bin/env python3
"""
Iterative improvement script for HTP Export Monitor.
This script runs the export monitor, compares with baseline, and tracks improvements.
"""

import os
import sys
import json
import time
import subprocess
import difflib
from pathlib import Path
from datetime import datetime
from rich.console import Console
from io import StringIO

class ExportMonitorIterator:
    """Manages iterative improvements to the export monitor."""
    
    def __init__(self):
        self.iteration = 0
        self.base_dir = Path("experiments/export_monitor")
        self.baseline_dir = Path("temp/baseline")
        self.iterations_dir = self.base_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)
        
        # Load baseline
        self.baseline_console = self._load_baseline_console()
        self.baseline_metadata = self._load_baseline_metadata()
        self.baseline_report = self._load_baseline_report()
        
    def _load_baseline_console(self) -> list[str]:
        """Load baseline console output."""
        with open(self.baseline_dir / "console_output_raw.txt") as f:
            return f.readlines()
    
    def _load_baseline_metadata(self) -> dict:
        """Load baseline metadata."""
        with open(self.baseline_dir / "metadata_pretty.json") as f:
            return json.load(f)
    
    def _load_baseline_report(self) -> list[str]:
        """Load baseline report."""
        with open(self.baseline_dir / "model_htp_export_report.txt") as f:
            return f.readlines()
    
    def run_iteration(self) -> dict:
        """Run one iteration of testing and improvement."""
        self.iteration += 1
        print(f"\n{'='*80}")
        print(f"🔄 ITERATION {self.iteration}")
        print(f"{'='*80}")
        
        iteration_dir = self.iterations_dir / f"iteration_{self.iteration:03d}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Run export with current implementation
        console_output = self._run_export(iteration_dir)
        
        # Load generated files
        metadata = self._load_generated_metadata(iteration_dir)
        report = self._load_generated_report(iteration_dir)
        
        # Compare outputs
        results = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "console_diff": self._compare_console(console_output, iteration_dir),
            "metadata_diff": self._compare_metadata(metadata, iteration_dir),
            "report_diff": self._compare_report(report, iteration_dir),
            "issues": [],
            "improvements": []
        }
        
        # Analyze issues
        self._analyze_issues(results)
        
        # Save iteration results
        with open(iteration_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _run_export(self, iteration_dir: Path) -> list[str]:
        """Run the export and capture console output."""
        print("🚀 Running export...")
        
        # Use Rich to capture with formatting
        console = Console(record=True, width=120, force_terminal=True)
        
        # Run the command
        cmd = [
            "uv", "run", "modelexport", "export",
            "--model", "prajjwal1/bert-tiny",
            "--output", str(iteration_dir / "model.onnx"),
            "--strategy", "htp",
            "--verbose",
            "--with-report"
        ]
        
        # Capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        lines = []
        for line in process.stdout:
            lines.append(line)
            console.print(line.rstrip(), style="")
        
        process.wait()
        
        # Save outputs
        with open(iteration_dir / "console_output.txt", "w") as f:
            f.writelines(lines)
        
        # Save Rich formatted output
        console_html = console.export_html()
        with open(iteration_dir / "console_output.html", "w") as f:
            f.write(console_html)
        
        return lines
    
    def _load_generated_metadata(self, iteration_dir: Path) -> dict | None:
        """Load generated metadata."""
        metadata_path = iteration_dir / "model_htp_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None
    
    def _load_generated_report(self, iteration_dir: Path) -> list[str] | None:
        """Load generated report."""
        # Try different report names
        for report_name in ["model_htp_export_report.txt", "model_full_report.txt"]:
            report_path = iteration_dir / report_name
            if report_path.exists():
                with open(report_path) as f:
                    return f.readlines()
        return None
    
    def _compare_console(self, current: list[str], iteration_dir: Path) -> dict:
        """Compare console outputs."""
        diff = list(difflib.unified_diff(
            self.baseline_console,
            current,
            fromfile="baseline",
            tofile=f"iteration_{self.iteration}",
            lineterm=""
        ))
        
        # Save diff
        with open(iteration_dir / "console_diff.txt", "w") as f:
            f.write("\n".join(diff))
        
        # Analyze differences
        missing_sections = []
        extra_sections = []
        
        # Check for key sections from baseline
        baseline_text = "".join(self.baseline_console)
        current_text = "".join(current)
        
        key_sections = [
            "STEP 1/8: MODEL PREPARATION",
            "STEP 2/8: INPUT GENERATION",
            "STEP 3/8: HIERARCHY BUILDING",
            "STEP 4/8: ONNX EXPORT",
            "STEP 5/8: NODE TAGGER CREATION",
            "STEP 6/8: ONNX NODE TAGGING",
            "STEP 7/8: TAG INJECTION",
            "STEP 8/8: METADATA GENERATION",
            "Module Hierarchy:",
            "Complete HF Hierarchy with ONNX Nodes:"
        ]
        
        for section in key_sections:
            if section in baseline_text and section not in current_text:
                missing_sections.append(section)
        
        return {
            "lines_baseline": len(self.baseline_console),
            "lines_current": len(current),
            "diff_lines": len(diff),
            "missing_sections": missing_sections,
            "match_percentage": self._calculate_similarity(baseline_text, current_text)
        }
    
    def _compare_metadata(self, current: dict | None, iteration_dir: Path) -> dict:
        """Compare metadata structures."""
        if not current:
            return {"error": "No metadata generated"}
        
        # Deep comparison
        differences = self._deep_diff(self.baseline_metadata, current)
        
        # Save pretty diff
        with open(iteration_dir / "metadata_diff.json", "w") as f:
            json.dump(differences, f, indent=2)
        
        return {
            "keys_baseline": list(set(self._flatten_keys(self.baseline_metadata))),
            "keys_current": list(set(self._flatten_keys(current))),
            "differences": differences
        }
    
    def _compare_report(self, current: list[str] | None, iteration_dir: Path) -> dict:
        """Compare report outputs."""
        if not current:
            return {"error": "No report generated"}
        
        diff = list(difflib.unified_diff(
            self.baseline_report,
            current,
            fromfile="baseline",
            tofile=f"iteration_{self.iteration}",
            lineterm=""
        ))
        
        with open(iteration_dir / "report_diff.txt", "w") as f:
            f.write("\n".join(diff))
        
        return {
            "lines_baseline": len(self.baseline_report),
            "lines_current": len(current),
            "diff_lines": len(diff)
        }
    
    def _analyze_issues(self, results: dict) -> None:
        """Analyze and categorize issues."""
        issues = []
        
        # Console issues
        console_diff = results["console_diff"]
        if console_diff["lines_current"] < 50:  # Baseline has ~165 lines
            issues.append("Console output is too short - verbose mode may not be working")
        
        if console_diff["missing_sections"]:
            issues.append(f"Missing console sections: {', '.join(console_diff['missing_sections'])}")
        
        # Metadata issues
        metadata_diff = results["metadata_diff"]
        if isinstance(metadata_diff, dict) and "error" in metadata_diff:
            issues.append(metadata_diff["error"])
        
        # Report issues
        report_diff = results["report_diff"]
        if isinstance(report_diff, dict) and "error" in report_diff:
            issues.append(report_diff["error"])
        
        results["issues"] = issues
    
    def _print_summary(self, results: dict) -> None:
        """Print iteration summary."""
        print("\n📊 ITERATION SUMMARY")
        print("-" * 40)
        
        console_diff = results["console_diff"]
        print(f"Console Output:")
        print(f"  Baseline: {console_diff['lines_baseline']} lines")
        print(f"  Current: {console_diff['lines_current']} lines")
        print(f"  Similarity: {console_diff['match_percentage']:.1f}%")
        
        if console_diff["missing_sections"]:
            print(f"  Missing sections: {len(console_diff['missing_sections'])}")
        
        print(f"\nIssues Found: {len(results['issues'])}")
        for i, issue in enumerate(results['issues'], 1):
            print(f"  {i}. {issue}")
        
        if results['improvements']:
            print(f"\nImprovements: {len(results['improvements'])}")
            for i, improvement in enumerate(results['improvements'], 1):
                print(f"  {i}. {improvement}")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity percentage."""
        seq = difflib.SequenceMatcher(None, text1, text2)
        return seq.ratio() * 100
    
    def _flatten_keys(self, d: dict, parent_key: str = '') -> list[str]:
        """Flatten nested dict keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_keys(v, new_key))
            else:
                items.append(new_key)
        return items
    
    def _deep_diff(self, d1: dict, d2: dict, path: str = '') -> dict:
        """Deep diff of two dicts."""
        diff = {}
        
        # Check keys
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        
        if keys1 != keys2:
            diff['missing_keys'] = list(keys1 - keys2)
            diff['extra_keys'] = list(keys2 - keys1)
        
        # Check values for common keys
        for key in keys1 & keys2:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                subdiff = self._deep_diff(d1[key], d2[key], f"{path}.{key}")
                if subdiff:
                    diff[key] = subdiff
            elif d1[key] != d2[key]:
                diff[key] = {"baseline": d1[key], "current": d2[key]}
        
        return diff

def main():
    """Run iterative improvements."""
    iterator = ExportMonitorIterator()
    
    # Run first iteration
    results = iterator.run_iteration()
    
    # Check if we need to continue
    if results["issues"]:
        print("\n⚠️ Issues detected! Please fix the export monitor and run again.")
        print("\nNext steps:")
        print("1. Fix the identified issues in export_monitor.py")
        print("2. Run this script again to test iteration 2")
        print("3. Continue until all issues are resolved")
    else:
        print("\n✅ All outputs match baseline! Export monitor is working correctly.")

if __name__ == "__main__":
    main()