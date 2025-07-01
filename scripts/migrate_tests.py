#!/usr/bin/env python3
"""
Test Migration Script for Iteration 15

Categorizes and migrates existing tests to the new test structure.
"""

import os
import shutil
from pathlib import Path

# Test categorization based on file analysis
TEST_MIGRATIONS = {
    # Tests to archive (obsolete due to architecture changes)
    'archive': [
        'test_hierarchy_exporter.py',  # Old HierarchyExporter class
        'test_operation_config.py',  # Old operation config (superseded by core version)
        'test_param_mapping.py',  # Outdated parameter mapping
        'test_tag_propagation.py',  # Likely outdated due to strategy changes
    ],
    
    # Tests to move to integration/
    'integration': [
        'test_cli.py',  # CLI integration (may need merging)
        'test_cli_integration.py',  # Already exists in integration/
        'test_baseline_comparison.py',  # Cross-strategy comparison
        'test_strategy_comparison.py',  # Cross-strategy comparison
        'test_complex_models.py',  # Integration testing
    ],
    
    # Tests to move to unit/test_core/
    'unit_core': [
        # These would need analysis to see if they're core functionality
    ],
    
    # Tests to move to unit/test_strategies/
    'unit_strategies': [
        'test_htp_operation_tracing.py',  # HTP strategy specific
        'test_hf_module_tagging.py',  # Could be strategy-specific
    ],
    
    # Tests to move to fixtures/ or keep in root
    'keep_root': [
        'simple_hierarchy_test.py',  # Demo/example
        'operation_mapping_demo.py',  # Demo/example
        'analyze_vit_structure.py',  # Analysis script
    ],
    
    # Tests that need manual review
    'review': [
        'test_edge_cases.py',
        'test_performance_stress.py',
        'test_smoke_tests.py',
        'test_slice_tagging_validation.py',
        'test_topology_preservation.py',
        'test_tagged_onnx_validation.py',
    ]
}

def migrate_tests():
    """Execute the test migration."""
    tests_dir = Path('/mnt/d/BYOM/modelexport/tests')
    archive_dir = tests_dir / 'archive'
    
    # Create archive directory
    archive_dir.mkdir(exist_ok=True)
    
    print("Starting test migration for Iteration 15...")
    
    # Archive obsolete tests
    for test_file in TEST_MIGRATIONS['archive']:
        src = tests_dir / test_file
        if src.exists():
            dst = archive_dir / test_file
            print(f"Archiving: {test_file}")
            shutil.move(str(src), str(dst))
    
    # Move integration tests (avoid duplicates)
    integration_dir = tests_dir / 'integration'
    for test_file in TEST_MIGRATIONS['integration']:
        src = tests_dir / test_file
        dst = integration_dir / test_file
        if src.exists() and not dst.exists():
            print(f"Moving to integration/: {test_file}")
            shutil.move(str(src), str(dst))
        elif src.exists() and dst.exists():
            print(f"Duplicate found - archiving old: {test_file}")
            shutil.move(str(src), str(archive_dir / f"old_{test_file}"))
    
    # Move strategy-specific tests
    for test_file in TEST_MIGRATIONS['unit_strategies']:
        src = tests_dir / test_file
        if src.exists():
            # Determine which strategy folder based on content analysis
            if 'htp' in test_file.lower():
                dst_dir = tests_dir / 'unit' / 'test_strategies' / 'htp'
            else:
                dst_dir = tests_dir / 'unit' / 'test_strategies' / 'usage_based'  # Default
            
            dst = dst_dir / test_file
            print(f"Moving to {dst_dir.relative_to(tests_dir)}/: {test_file}")
            shutil.move(str(src), str(dst))
    
    print("\\nMigration completed. Review remaining tests manually:")
    for test_file in TEST_MIGRATIONS['review']:
        src = tests_dir / test_file
        if src.exists():
            print(f"  - {test_file}")
    
    print(f"\\nArchived tests are in: {archive_dir}")

if __name__ == '__main__':
    migrate_tests()