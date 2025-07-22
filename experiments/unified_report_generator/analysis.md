# Unified Report Generator Analysis

## Current State Analysis

### 1. Data Sources
- `self._export_report`: Contains step-by-step execution details
- `self._export_stats`: Contains overall statistics
- `self._hierarchy_data`: Module hierarchy information
- `self._tagged_nodes`: ONNX node to tag mappings
- `self._tagging_stats`: Detailed tagging statistics

### 2. Output Formats

#### Console Output (verbose mode)
- 8-step format with headers
- Real-time progress updates
- Tree visualizations (truncated)
- Statistics and summaries

#### Metadata JSON
- export_context
- model info
- modules (hierarchy data)
- nodes (tagged nodes)
- tracing info
- report section with steps
- statistics

#### Text Report
- Complete module hierarchy
- Complete node mappings
- Full console output

### 3. Issues to Address
1. **Inconsistent data access**: Same data accessed differently for each format
2. **Missing data**: Node tagging statistics not in metadata steps
3. **Code duplication**: Similar formatting logic repeated
4. **Order dependency**: Report file created after metadata
5. **Truncation inconsistency**: Console truncates, report doesn't

## Design Goals

### Unified Report Generator Should:
1. Single source of truth for all report data
2. Consistent data structure across all formats
3. Shared formatting and rendering logic
4. Configurable output formats
5. No data loss between formats
6. Extensible for future formats

### Key Principles:
- **Data First**: Collect all data in a unified structure
- **Format Later**: Apply formatting based on output type
- **Consistency**: Same data appears consistently across formats
- **Completeness**: No data loss or missing fields