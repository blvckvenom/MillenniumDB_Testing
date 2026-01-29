# MillenniumDB - Documentación Personal

**Última actualización**: 25 de Diciembre, 2025

---

## Descripción

Repositorio de documentación personal para el desarrollo de MillenniumDB, con énfasis en:
- **GQL Native Projections** - Sistema de proyecciones de grafos
- **Estándar ISO GQL** - Referencia del estándar ISO/IEC 39075:2024
- **Neo4j GDS** - Comparación con Graph Data Science de Neo4j

---

## Estructura del Repositorio

```
docs/
├── README.md                    ← Este archivo
├── GUIA_NAVEGACION.md           ← Guía rápida de navegación
│
├── MillenniumDB.wiki/           ← Wiki oficial de MillenniumDB
│   ├── Home.md                  ← Página principal
│   ├── GQL.md                   ← Documentación GQL
│   ├── GQL-Concepts.md          ← Conceptos de GQL
│   ├── GQL-Patterns.md          ← Patrones de consulta
│   ├── GQL-Paths.md             ← Consultas de caminos
│   ├── GQL-Statements.md        ← Sentencias GQL
│   ├── GQL-Types.md             ← Sistema de tipos
│   ├── MQL.md                   ← MillenniumDB Query Language
│   ├── Quad-Model.md            ← Modelo Quad
│   ├── SPARQL-Implementation-Status.md
│   ├── Setup.md                 ← Guía de instalación
│   └── Working-with-tensors.md  ← Operaciones con tensores
│
├── external_references/         ← Referencias externas
│   ├── ISO_IEC_39075_extracted/ ← Estándar ISO GQL
│   │   ├── INDEX.md             ← Índice de secciones
│   │   ├── ISO IEC 39075-2024.pdf
│   │   └── sections/            ← Secciones extraídas (32 archivos)
│   │
│   └── NEO4J_USER_MANUAL_DOC/   ← Manual Neo4j GDS
│       └── neo4j_graph_data_science_manual_.md
│
├── native_projection_review/    ← Análisis de proyecciones GQL
│   ├── README.md                ← Descripción del análisis
│   ├── CURRENT_PLAN.md          ← Plan de implementación actual
│   ├── 01_implementation_status.md  ← Estado de implementación
│   ├── 02_neo4j_comparison.md   ← Comparación con Neo4j
│   ├── 03_iso_compliance.md     ← Cumplimiento ISO
│   └── 04_feature_gap_analysis.md   ← Análisis de brechas
│
└── validation_report/           ← Reportes de validación
    ├── README.md                ← Descripción del reporte
    ├── codebase_map.md          ← Mapa del código fuente
    ├── discrepancies.md         ← Discrepancias encontradas
    ├── documentation_inventory.md
    ├── gql_grammar_analysis.md  ← Análisis de gramática GQL
    ├── recommendations.md       ← Recomendaciones
    └── validation_matrix.md     ← Matriz de validación
```

---

## Guías de Inicio Rápido

### Para entender GQL en MillenniumDB

1. **[MillenniumDB.wiki/GQL.md](MillenniumDB.wiki/GQL.md)** - Introducción a GQL
2. **[MillenniumDB.wiki/GQL-Concepts.md](MillenniumDB.wiki/GQL-Concepts.md)** - Conceptos fundamentales
3. **[MillenniumDB.wiki/GQL-Patterns.md](MillenniumDB.wiki/GQL-Patterns.md)** - Patrones de consulta

### Para implementar proyecciones nativas

1. **[native_projection_review/README.md](native_projection_review/README.md)** - Visión general
2. **[native_projection_review/01_implementation_status.md](native_projection_review/01_implementation_status.md)** - Estado actual
3. **[native_projection_review/02_neo4j_comparison.md](native_projection_review/02_neo4j_comparison.md)** - Comparación con Neo4j

### Para consultar el estándar ISO GQL

1. **[external_references/ISO_IEC_39075_extracted/INDEX.md](external_references/ISO_IEC_39075_extracted/INDEX.md)** - Índice del estándar
2. Buscar sección específica en `sections/`

---

## Contenido Destacado

### Wiki Oficial

La carpeta `MillenniumDB.wiki/` contiene la documentación oficial del proyecto, incluyendo:
- Guías de instalación y configuración
- Documentación de los tres modelos (RDF, Quad, GQL)
- Referencia de lenguajes de consulta
- Trabajo con tensores

### Estándar ISO GQL (ISO/IEC 39075:2024)

Documentación completa del estándar GQL extraída y organizada por secciones para fácil referencia:
- 628 páginas del estándar oficial
- 32 secciones markdown extraídas
- Índice navegable

### Manual Neo4j GDS

Referencia completa del sistema Graph Data Science de Neo4j para comparación de funcionalidades:
- Algoritmos de grafos
- Proyecciones nativas
- Machine Learning en grafos

---

## Notas

- La carpeta `/api/` está excluida del repositorio (contiene documentación Doxygen generada)
- Este repositorio es para documentación personal de desarrollo
- La wiki oficial del proyecto está en `MillenniumDB.wiki/`

---

## Enlaces Relacionados

- **Repositorio principal**: [MillenniumDB](https://github.com/MillenniumDB/MillenniumDB)
- **Repositorio de trabajo**: [MillenniumDB-Trabajo-Personal-todo-](https://github.com/blvckvenom/MillenniumDB-Trabajo-Personal-todo-)
