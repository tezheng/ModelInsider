# Timestamp Field Analysis in HTP Metadata Schema

## Current Timestamp Fields

### 1. export_context.timestamp
```json
{
  "description": "Export timestamp in ISO format",
  "title": "Timestamp", 
  "type": "string"
}
```
- Has description ✓
- Has title ✓
- Is required ✓
- No pattern validation ✗

### 2. Step Timestamps (in report.steps)
Multiple locations with inconsistent definitions:

#### NodeTaggingStep.timestamp
```json
{
  "type": "string"
}
```
- No description ✗
- No title ✗
- No pattern validation ✗
- Optional (not in required) ✗

#### ExportSteps timestamps
- model_preparation.timestamp
- hierarchy_building.timestamp
- node_tagging.timestamp

All have minimal definition:
```json
{
  "type": "string"
}
```

## Inconsistencies Found

1. **Missing ISO Format Validation**: None of the timestamp fields enforce ISO 8601 format
2. **Inconsistent Descriptions**: Only export_context.timestamp has a description
3. **Missing Pattern Validation**: No regex pattern to ensure correct format
4. **Optional vs Required**: Some timestamps are required, others optional
5. **No Reusable Definition**: Each timestamp is defined separately

## Recommended Solution

### 1. Create Reusable Timestamp Definition
```json
"ISOTimestamp": {
  "type": "string",
  "description": "ISO 8601 formatted timestamp (YYYY-MM-DDTHH:MM:SSZ)",
  "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d{3})?Z$",
  "examples": ["2025-07-22T12:00:00Z", "2025-07-22T12:00:00.123Z"]
}
```

### 2. Apply Consistently
Use `$ref` to reference the reusable definition:
```json
"timestamp": {
  "$ref": "#/$defs/ISOTimestamp"
}
```

### 3. Consider Optional Timestamps
For step timestamps that might not always be present:
```json
"timestamp": {
  "oneOf": [
    {"$ref": "#/$defs/ISOTimestamp"},
    {"type": "null"}
  ],
  "default": null
}
```

## Impact Analysis

### Files to Update
1. `htp_metadata_schema.json` - Add ISOTimestamp definition and update all timestamp fields
2. `metadata_builder.py` - Ensure timestamp generation uses correct format
3. `base_writer.py` - Check timestamp property implementation

### Testing Required
1. Validate existing metadata files still pass
2. Test invalid timestamp formats are rejected
3. Test optional timestamp handling
4. Performance impact of pattern matching

## Implementation Priority
1. High - This affects data quality and consistency
2. Should be done before implementing runtime validation
3. Backward compatibility consideration needed