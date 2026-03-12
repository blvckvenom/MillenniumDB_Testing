# Guía de Navegación

**Última actualización**: 8 de Marzo, 2026

---

## Inicio Rápido

| Si buscas... | Ve a... |
|--------------|---------|
| Documentación de GQL | [MillenniumDB.wiki/GQL.md](MillenniumDB.wiki/GQL.md) |
| Conceptos de grafos de propiedad | [MillenniumDB.wiki/GQL-Property-graphs.md](MillenniumDB.wiki/GQL-Property-graphs.md) |
| Patrones de consulta | [MillenniumDB.wiki/GQL-Patterns.md](MillenniumDB.wiki/GQL-Patterns.md) |
| Estándar ISO GQL | [external_references/ISO_IEC_39075_extracted/INDEX.md](external_references/ISO_IEC_39075_extracted/INDEX.md) |
| Manual Neo4j GDS | [external_references/NEO4J_USER_MANUAL_DOC/](external_references/NEO4J_USER_MANUAL_DOC/) |
| Proyecciones nativas | [native_projection_review/](native_projection_review/) |
| Validación de implementación | [validation_report/](validation_report/) |
| Roadmap GNN (plan maestro) | [../Partial_Idea/README.md](../Partial_Idea/README.md) |
| Planes de ejecución | [../docs/plans/INDEX.md](plans/INDEX.md) |
| Módulo GNN (82 archivos) | [architecture/GNN_MODULE.md](architecture/GNN_MODULE.md) |
| Trabajo diferido | [plans/DEFERRED_WORK.md](plans/DEFERRED_WORK.md) |

---

## Contenido por Carpeta

### MillenniumDB.wiki/

Wiki oficial de MillenniumDB con documentación completa:

| Archivo | Descripción |
|---------|-------------|
| `Home.md` | Página principal de la wiki |
| `Setup.md` | Guía de instalación y configuración |
| `Database-models.md` | Modelos de base de datos soportados |
| `GQL.md` | Introducción a GQL |
| `GQL-Concepts.md` | Conceptos fundamentales de GQL |
| `GQL-Property-graphs.md` | Grafos de propiedad |
| `GQL-Patterns.md` | Patrones de consulta |
| `GQL-Paths.md` | Consultas de caminos |
| `GQL-Statements.md` | Sentencias GQL |
| `GQL-Types.md` | Sistema de tipos |
| `GQL-Supported-features.md` | Features soportados |
| `GQL-Unsupported-features.md` | Features no soportados |
| `MQL.md` | MillenniumDB Query Language |
| `Quad-Model.md` | Modelo Quad |
| `SPARQL-Implementation-Status.md` | Estado de SPARQL |
| `Working-with-tensors.md` | Operaciones con tensores |

---

### external_references/

Referencias externas para desarrollo:

#### ISO_IEC_39075_extracted/

Estándar GQL (ISO/IEC 39075:2024) extraído:

- `INDEX.md` - Índice navegable de secciones
- `ISO IEC 39075-2024.pdf` - Documento PDF completo (628 páginas)
- `sections/` - 32 secciones en formato markdown

**Secciones clave:**
- `section_001` - Scope y conformance
- `section_004-006` - Graph types y data model
- `section_010-015` - Query expressions
- `section_020-025` - Path patterns

#### NEO4J_USER_MANUAL_DOC/

Manual de Neo4j Graph Data Science:

- `neo4j_graph_data_science_manual_.md` - Manual completo

**Contenido:**
- Graph projections (native y cypher)
- Algoritmos de centralidad
- Community detection
- Similarity algorithms
- Machine learning

---

### native_projection_review/

Análisis de proyecciones nativas GQL en MillenniumDB:

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Visión general del análisis |
| `CURRENT_PLAN.md` | Plan de implementación actual |
| `01_implementation_status.md` | Estado de implementación (~70% Neo4j, ~61% ISO) |
| `02_neo4j_comparison.md` | Comparación detallada con Neo4j GDS |
| `03_iso_compliance.md` | Cumplimiento con ISO 39075 |
| `04_feature_gap_analysis.md` | Análisis de brechas y prioridades |

---

### validation_report/

Reportes de validación del código:

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Descripción del proceso de validación |
| `codebase_map.md` | Mapa del código fuente |
| `discrepancies.md` | Discrepancias encontradas |
| `documentation_inventory.md` | Inventario de documentación |
| `gql_grammar_analysis.md` | Análisis de gramática GQL |
| `recommendations.md` | Recomendaciones de mejora |
| `validation_matrix.md` | Matriz de validación |

---

### plans/

Registros de trabajo completado y planificación:

| Archivo | Descripción |
|---------|-------------|
| `INDEX.md` | Índice maestro que conecta Partial_Idea, .planning y docs/plans |
| `DEFERRED_WORK.md` | Items diferidos de fases 10-13 con priorización |
| `2026-03-06-merge-upstream-dev.md` | Merge de upstream/dev |
| `2026-03-07-fix-gnn-fork-errors.md` | Campaña de 17 bug fixes |
| `2026-03-08-audit-findings-complete.md` | Auditoría: 45 hallazgos |
| `2026-03-08-fix-audit-findings.md` | Ejecución de 35 commits de fixes |

---

### architecture/

Documentos de arquitectura del sistema:

| Archivo | Descripción |
|---------|-------------|
| `GNN_MODULE.md` | Inventario de los 82 archivos del módulo GNN |
| `TENSOR_SYSTEMS_CRITIQUE.md` | Análisis de tensores MDB vs GNN |
| `diagrams/comparison_matrix.md` | Matriz comparativa de capacidades |

---

### Partial_Idea/ (fuera de docs/)

Roadmap activo del pipeline GNN — **el plan maestro**:

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Visión general: 7 fases, 3 completas, 4 pendientes |
| `GQL_PROCEDURE_INTEGRATION.md` | 9 procedimientos implementados + 3 pendientes |
| `cross_cutting/EMBEDDING_MANAGEMENT.md` | HNSW y gestión de embeddings |
| `phase_00-06/` | Documentación detallada de cada fase |

---

### .planning/phases/ (fuera de docs/)

Planes de ejecución con tareas, código y summaries:

| Fase | Nombre | Archivos |
|------|--------|----------|
| 10 | Tensor Integration Fix | `10-01-PLAN.md`, `10-01-SUMMARY.md` |
| 11 | GQL Import Tensors | `11-01-PLAN.md`, `11-01-SUMMARY.md` |
| 12 | HNSW + GNN Integration | `12-01-PLAN.md`, `12-01-SUMMARY.md` |
| 13 | Projection Optimization | `13-01-PLAN.md`, `13-01-SUMMARY.md` |

---

## Flujos de Trabajo Comunes

### Investigar una feature de GQL

1. Revisar la wiki: `MillenniumDB.wiki/GQL-*.md`
2. Consultar el estándar ISO: `external_references/ISO_IEC_39075_extracted/`
3. Ver cómo lo hace Neo4j: `external_references/NEO4J_USER_MANUAL_DOC/`

### Evaluar estado de proyecciones

1. Ver estado actual: `native_projection_review/01_implementation_status.md`
2. Comparar con Neo4j: `native_projection_review/02_neo4j_comparison.md`
3. Verificar cumplimiento ISO: `native_projection_review/03_iso_compliance.md`
4. Priorizar trabajo: `native_projection_review/04_feature_gap_analysis.md`

### Validar implementación

1. Consultar matriz: `validation_report/validation_matrix.md`
2. Ver discrepancias: `validation_report/discrepancies.md`
3. Revisar recomendaciones: `validation_report/recommendations.md`

---

## Atajos Útiles

```bash
# Ver estructura de carpetas
tree -L 2 docs/

# Buscar en el estándar ISO
grep -r "graph projection" docs/external_references/ISO_IEC_39075_extracted/

# Buscar en la wiki
grep -r "MATCH" docs/MillenniumDB.wiki/

# Ver estado de proyecciones
cat docs/native_projection_review/01_implementation_status.md
```

---

## Notas

- La carpeta `api/` está excluida (documentación Doxygen generada)
- Los archivos `.pdf` están incluidos para referencia offline
- La wiki `MillenniumDB.wiki/` se sincroniza con el proyecto principal
