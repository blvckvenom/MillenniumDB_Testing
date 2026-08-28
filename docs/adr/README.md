# Architectural Decision Records (ADRs)

> Timeline of design decisions for the GNN Training System thesis project.
> Each ADR captures the context, analysis, and rationale behind a significant decision.

## Decision Log

| # | Date | Title | Status |
|---|------|-------|--------|
| [001](001_tensor_store_redesign.md) | 2026-03-09 | GNN Tensor Store Redesign: Key-Value → FeatureMatrix | ACCEPTED (Draft) |
| [002](002_topology_snapshot_and_gpu_projection.md) | 2026-03-24 | TopologySnapshot, GPU-Accelerated Projection, LibTorch Removal | ACCEPTED |

## How to Use

- **New decisions**: Create `NNN_short_name.md` following the ADR template
- **Status values**: `PROPOSED` → `ACCEPTED` → `IMPLEMENTED` | `SUPERSEDED` | `REJECTED`
- **Referencing**: Link ADRs from phase documents and code comments where relevant

## Template

```markdown
# ADR-NNN: [Title]

> **Date:** YYYY-MM-DD
> **Status:** PROPOSED | ACCEPTED | IMPLEMENTED | SUPERSEDED | REJECTED

## Context
[What is the issue/need?]

## Decision
[What was decided and why?]

## Consequences
[Positive, negative, and neutral impacts]
```
