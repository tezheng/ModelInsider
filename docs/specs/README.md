# Technical Specifications

This directory contains technical format specifications and standards for the ModelExport project.

## What is a Technical Specification?

A technical specification defines the precise format, structure, and requirements for data formats, APIs, or protocols. Unlike ADRs (which document decisions), specifications define standards and formats.

## Current Specifications

| Specification | Title | Version | Status | Last Updated |
|---------------|-------|---------|--------|--------------|
| [graphml-format-specification.md](graphml-format-specification.md) | GraphML Format Specification | v1.1 | ✅ Implemented | 2025-07-31 |

## Specification Categories

### Format Specifications
- **GraphML v1.1**: Complete ONNX ↔ GraphML bidirectional conversion format
- **Future**: Additional format specifications as needed

### API Specifications  
- **Future**: REST API specifications
- **Future**: Python API specifications

### Protocol Specifications
- **Future**: Communication protocol specifications

## Creating a New Specification

1. Create a new `.md` file with descriptive name
2. Follow the specification template format:
   - Overview and purpose
   - Format version and compatibility
   - Detailed specification sections
   - Examples and validation rules
   - Implementation notes
3. Add entry to the index above
4. Submit for review via pull request

## Specification vs ADR

**Use Specifications for:**
- Data format definitions (XML, JSON, binary)
- API interface definitions
- Protocol specifications
- Standard compliance requirements

**Use ADRs for:**
- Architectural decisions and trade-offs
- Technology choices and rationale
- Design pattern selections
- Implementation approach decisions

## Version Management

- **Major version** (x.0.0): Breaking changes to format
- **Minor version** (x.y.0): Backward-compatible additions
- **Patch version** (x.y.z): Bug fixes and clarifications

## Implementation Status

- ✅ **Implemented**: Specification is fully implemented and tested
- 🚧 **In Progress**: Specification is being implemented  
- 📋 **Planned**: Specification is planned for future implementation
- ❌ **Deprecated**: Specification is no longer recommended