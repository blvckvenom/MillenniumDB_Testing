# Guía de Navegación

**Última actualización**: 25 de Diciembre, 2025

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
