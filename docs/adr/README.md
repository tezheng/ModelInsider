# Architecture Decision Records (ADRs)

This directory contains all Architecture Decision Records for the ModelExport project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

All ADRs follow the template in `adr-template.md`. Key sections include:
- Context and Problem Statement
- Decision Drivers
- Considered Options
- Decision Outcome
- Consequences (positive, negative, neutral)
- Implementation Notes

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted | 2025-06-26 |
| [ADR-002](ADR-002-auxiliary-operations-tagging.md) | Auxiliary Operations Tagging | Accepted | 2025-06-28 |
| [ADR-003](ADR-003-standard-vs-function-export.md) | Standard vs Function Export | Accepted | 2025-07-02 |
| [ADR-004](ADR-004-onnx-node-tagging-priorities.md) | ONNX Node Tagging Priorities | Accepted | 2025-07-03 |
| [ADR-005](ADR-005-tree-visualization-library.md) | Tree Visualization Library | Accepted | 2025-07-07 |
| [ADR-006](ADR-006-timestamp-handling-best-practices.md) | Timestamp Handling Best Practices | Accepted | 2025-07-12 |
| [ADR-007](ADR-007-root-module-hook-strategy.md) | Root Module Hook Strategy | Accepted | - |
| [ADR-008](ADR-008-onnx-to-graphml-package-selection.md) | ONNX to GraphML Package Selection | Proposed | 2025-07-28 |
| [ADR-009](ADR-009-graphml-converter-architecture.md) | GraphML Converter Architecture | Proposed | 2025-07-28 |
| [ADR-010](ADR-010-onnx-graphml-format-specification.md) | ONNX GraphML Format Specification | Accepted | 2025-07-28 |

## Creating a New ADR

1. Copy `adr-template.md` to a new file named `ADR-XXX-short-title.md`
2. Fill in all sections of the template
3. Add an entry to the index above
4. Submit for review via pull request

## ADR Status Values

- **Proposed**: Initial state, decision under discussion
- **Accepted**: Decision has been made and is being implemented
- **Deprecated**: Decision is no longer recommended
- **Superseded**: Replaced by another ADR (reference the new ADR)

## Review Process

1. Author creates ADR as "Proposed"
2. Team reviews and discusses
3. Decision makers approve
4. Status updated to "Accepted"
5. Implementation proceeds based on ADR