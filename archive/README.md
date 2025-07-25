# Archive Folder Structure

This folder contains archived code, tests, and documentation that are no longer actively used but kept for historical reference.

## Directory Structure

### `/deprecated_tests/`
Contains test files that are no longer valid due to API changes or deprecated functionality.
- `deprecated_htp_monitor_tests/` - Tests for old HTP Export Monitor implementation

### `/old_tests/`
Contains older test implementations that have been replaced with newer versions.

### `/old_implementations/`
Contains previous implementations of code modules:
- `htp_old/` - Old HTP (Hierarchy-preserving Tags Protocol) implementation
- `htp_old_backup/` - Backup of old HTP implementation

### `/old_schemas/`
Contains previous versions of JSON schemas:
- `htp_metadata_schema_backup.json` - Backup of HTP metadata schema
- `htp_metadata_schema_old.json` - Previous version of HTP metadata schema
- `htp_metadata_schema_fixed.json` - Fixed version that was superseded

### `/strategies/`
Contains old strategy implementations that have been replaced.

### `/usage_based/`
Contains usage-based test files and experiments.

### `/legacy/`
Contains legacy code and implementations.

## Other Files
- Various test files for edge cases, performance benchmarks, regression tests, etc.
- Old CLI implementations and experimental code

## Note
Files in this archive should not be imported or used in active development. They are kept for:
1. Historical reference
2. Understanding evolution of the codebase
3. Potential recovery of useful patterns or logic