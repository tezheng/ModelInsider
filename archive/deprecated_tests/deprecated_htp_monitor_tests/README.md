# Deprecated HTP Export Monitor Tests

These tests were archived on 2025-07-24 because they were testing an obsolete version of the HTP Export Monitor.

## Issues with these tests:

1. **Wrong import path**: Tests import from `modelexport.strategies.htp.export_monitor` but the actual implementation uses `modelexport.strategies.htp_new.export_monitor`

2. **Non-existent API**: Tests expect methods and attributes that don't exist:
   - `console_buffer` attribute
   - `log_step()` method
   - The actual API uses `update()` method

3. **Outdated expectations**: Tests check for specific output formatting that has changed

## Archived files:

- `test_htp_console_output.py` - Testing console output with non-existent API
- `test_htp_export_monitor_comprehensive.py` - Comprehensive tests for old API
- `test_htp_export_monitor_no_duplicates.py` - Testing duplicate prevention in old API
- `test_htp_export_monitor_updated.py` - Tests for "updated" version that's still outdated
- `test_module_hierarchy_format.py` - Testing specific output formatting that has changed

## Recommendation:

If HTP Export Monitor testing is needed in the future, new tests should be written against the actual `modelexport.strategies.htp_new.export_monitor` implementation with the correct API.

## Note:

Two test files that import from the old path were NOT archived because they still pass:
- `test_hierarchy_truncation.py` - Tests truncation logic (3/3 pass)
- `test_node_tagging_hierarchy_detailed.py` - Tests node tagging details (4/4 pass)

These should eventually be updated to import from the correct path but are kept active since they still provide value.