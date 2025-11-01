# 📚 MillenniumDB - Documentación del Sistema de Proyecciones y GNN

**Última actualización**: 1 de Noviembre, 2025
**Versión**: 2.1 (Reorganizada + Investigación Exhaustiva)

---

## 📖 Índice General

Este directorio contiene toda la documentación relacionada con:
- **Sistema PROJECT/USE GRAPH** (Proyecciones persistentes en disco)
- **Arquitectura GNN Nativa** (Entrenamiento de Graph Neural Networks dentro del DBMS)
- **Historial de implementación** (Phases 1-2 completadas)
- **Debugging y verificación** (Problemas resueltos y lecciones aprendidas)

---

## 🗂️ Estructura de Directorios

```
docs/
├── README.md (este archivo) ...................... Índice maestro de navegación
│
├── 01_PROJECT_System/ ............................ Sistema de Proyecciones (COMPLETO ✅)
│   ├── INDICE_DOCUMENTACION_PROJECT.md .......... Guía de navegación del sistema PROJECT
│   ├── PROJECT_USE_RESUMEN.md ................... Resumen ejecutivo (10 págs, inicio rápido)
│   ├── GUIA_PROJECT_Y_USE_GRAPH.md .............. Guía completa (50+ págs, referencia)
│   └── PROJECT_CODIGO_INTERNO.md ................ Código interno detallado (para devs)
│
├── 02_GNN_Architecture/ .......................... Arquitectura de Machine Learning (ROADMAP 🚧)
│   ├── ANALISIS_COMPARATIVO_NEO4J_GDS.md ........ MillenniumDB vs Neo4j GDS
│   ├── ARQUITECTURA_GNN_NATIVA.md ............... Diseño completo del motor GNN nativo
│   └── ROADMAP_PREREQUISITOS_GNN_NATIVA.md ...... Pre-requisitos para Fase 0 (estado actual)
│
├── 03_Implementation_Phases/ ..................... Historial de Fases Completadas (ARCHIVE)
│   ├── PHASE1_LABEL_SUPPORT_COMPLETE.md ......... Implementación de INCLUDE LABELS ✅
│   └── PHASE2_PROPERTY_SUPPORT_COMPLETE.md ...... Implementación de INCLUDE PROPERTIES ✅
│
└── 04_Debugging_History/ ......................... Problemas Resueltos (ARCHIVE)
    ├── BUG_INVESTIGATION_SUMMARY.md ............. Investigación de bug de propiedades NULL
    └── PROJECTION_PROPERTIES_WORKING.md ......... Verificación final (properties funcionan)
```

---

## 🚀 Guía de Inicio Rápido

### Para Usuarios Nuevos del Sistema PROJECT

**¿Primera vez usando PROJECT/USE?** → Lee esto en orden:

1. **[`01_PROJECT_System/PROJECT_USE_RESUMEN.md`](01_PROJECT_System/PROJECT_USE_RESUMEN.md)** (30 min)
   - ¿Qué hacen PROJECT y USE?
   - Ejemplo completo paso a paso
   - Comparación visual: grafo principal vs proyección
   - Comandos de referencia rápida

2. **Probar ejemplos del resumen** (15 min)
   ```bash
   # Crear proyección
   MATCH (n)-[e]->(m) RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)

   # Consultar proyección
   USE "test" MATCH (n) WHERE n.age > 30 RETURN n.name

   # Volver al grafo principal
   USE CURRENT_GRAPH
   ```

3. **[`01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md`](01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md)** - Secciones específicas según necesidad
   - Arquitectura interna
   - Opciones avanzadas
   - Limitaciones y errores comunes

---

### Para Desarrolladores que van a Modificar el Código

**¿Vas a tocar el código de proyecciones?** → Lee esto en orden:

1. **[`01_PROJECT_System/PROJECT_USE_RESUMEN.md`](01_PROJECT_System/PROJECT_USE_RESUMEN.md)** (orientación general, 30 min)

2. **[`01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md`](01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md)** - Sección "Arquitectura Interna" (45 min)

3. **[`01_PROJECT_System/PROJECT_CODIGO_INTERNO.md`](01_PROJECT_System/PROJECT_CODIGO_INTERNO.md)** (código detallado, 60 min)
   - Flujo completo con código C++
   - 6 fases de PROJECT (parsing → ejecución → escritura)
   - Enrutamiento dinámico en GQLModel
   - Tabla con archivos y líneas exactas

4. **Leer código fuente** con las referencias de líneas de PROJECT_CODIGO_INTERNO.md

---

### Para Implementar Machine Learning / GNN

**¿Quieres implementar GNN training o algoritmos analíticos?** → Lee esto en orden:

1. **[`02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md`](02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md)** (PRIMERO, 60 min)
   - **Estado actual del proyecto** (qué existe vs qué falta)
   - **4 pre-requisitos críticos** que BLOQUEAN la implementación de GNN:
     1. MUTATE PROPERTY (3-4 semanas)
     2. Sistema de Procedimientos GQL (2-3 semanas)
     3. TensorStore maduro (3-4 semanas)
     4. Integración LibTorch (2-3 semanas)
   - Especificaciones detalladas de implementación con código de ejemplo
   - Timeline: 12 semanas (1 dev) o 8 semanas (2 devs paralelos)

2. **[`02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md`](02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md)** (visión a largo plazo, 90 min)
   - Arquitectura completa del motor GNN nativo
   - 5 fases de implementación (Fase 0 hasta Fase 4)
   - Comparación con Neo4j GDS
   - Ejemplos de uso final

3. **[`02_GNN_Architecture/ANALISIS_COMPARATIVO_NEO4J_GDS.md`](02_GNN_Architecture/ANALISIS_COMPARATIVO_NEO4J_GDS.md)** (contexto, 60 min)
   - Análisis exhaustivo de Neo4j GDS
   - Brechas identificadas vs MillenniumDB
   - Ventajas competitivas de MillenniumDB

---

### Para Debugging de Problemas

**¿Algo no funciona con proyecciones?** → Lee esto:

1. **[`04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md`](04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md)** (proceso de investigación)
   - Problema típico: "propiedades retornan NULL"
   - Root cause: patrón MATCH incorrecto (no bug del sistema)
   - Metodología de debugging paso a paso

2. **[`04_Debugging_History/PROJECTION_PROPERTIES_WORKING.md`](04_Debugging_History/PROJECTION_PROPERTIES_WORKING.md)** (verificación)
   - Tests de verificación completa
   - Comandos para validar que properties funcionan

3. **[`01_PROJECT_System/PROJECT_CODIGO_INTERNO.md`](01_PROJECT_System/PROJECT_CODIGO_INTERNO.md)** - Fase específica del problema
   - Si el problema es con labels → Fase 3 (procesamiento de labels)
   - Si el problema es con properties → Fase 5 (extracción de propiedades)
   - Si el problema es con USE → Sección de enrutamiento dinámico

---

## 🔬 Investigaciones Activas

### Investigación Exhaustiva del Sistema PROJECT (1 de Noviembre, 2025)

**Ubicación**: [`../investigacion_project_2025_11_01/`](../investigacion_project_2025_11_01/)

**Análisis completo** del estado actual del sistema PROJECT, identificación de bugs, funcionalidades faltantes, y roadmap detallado de mejoras:

**Contenido**:
- 📊 **[INFORME_ESTADO_PROJECT_EXHAUSTIVO.md](../investigacion_project_2025_11_01/INFORME_ESTADO_PROJECT_EXHAUSTIVO.md)** (1,987 líneas)
  - Resumen ejecutivo con métricas clave
  - Metodología de investigación exhaustiva
  - Arquitectura actual (6 fases de ejecución)
  - 7 funcionalidades implementadas (análisis detallado)
  - 4 bugs identificados (1 crítico, 1 medio, 2 menores)
  - 15 funcionalidades faltantes priorizadas
  - Análisis comparativo vs Neo4j GDS
  - Roadmap de mejoras (6-12 meses)

- 🧪 **[test_project_exhaustive.sh](../investigacion_project_2025_11_01/test_project_exhaustive.sh)** - Script con 60+ casos de prueba
- 📝 **[README.md](../investigacion_project_2025_11_01/README.md)** - Índice y resumen ejecutivo

**Hallazgos Clave**:
- ✅ Funcionalidades Core: 7/7 (100%)
- ❌ Funcionalidades Avanzadas: 0/15 (0%)
- ⚠️ Neo4j GDS Paridad: 41%
- 🔴 Bug crítico: `USE CURRENT_GRAPH` no funciona (fix: 2-3h)
- 🟡 Bug medio: Labels no se validan (fix: 3-4h)
- ⚠️ Faltan 3 features críticas que bloquean GNN:
  1. UNDIRECTED orientation (4-6 semanas)
  2. Relationship aggregation (4-6 semanas)
  3. Management commands (4-5 semanas)

**Tiempo de Lectura**:
- Resumen ejecutivo: 10 minutos
- Informe completo: 60-90 minutos

**Audiencia**: Desarrolladores, product managers, investigadores

---

## 📊 Archivos de Código Fuente Principales

### Core de Proyecciones (PROJECT/USE)

| Archivo | Descripción | Líneas | Estado |
|---------|-------------|--------|--------|
| `src/query/executor/binding_iter/aggregation/gql/agg_project.h` | Función de agregación PROJECT | 428 | ✅ Completo |
| `src/graph_models/gql/projection/projection_storage.{h,cc}` | Escritura a índices B+Tree | 201 + 369 | ✅ Completo |
| `src/graph_models/gql/projection/projection_manager.{h,cc}` | Gestión de catálogo | ~150 | ✅ Completo |
| `src/graph_models/gql/projection/projection_query_context.{h,cc}` | Contexto para USE | 79 | ✅ Completo |
| `src/graph_models/gql/gql_model.{h,cc}` | Enrutamiento dinámico de índices | 77 + 200 | ✅ Completo |

### Parsing y Ejecución

| Archivo | Descripción | Líneas | Estado |
|---------|-------------|--------|--------|
| `src/query/parser/grammar/gql/query_visitor.cc` | Parser de PROJECT y USE | ~2000 | ✅ Completo |
| `src/query/query_context.{h,cc}` | Estado de proyección activa | ~500 | ✅ Completo |

### Tests y Verificación

| Archivo | Descripción | Estado |
|---------|-------------|--------|
| `tests/projection/unit_tests/projection_storage_test.cc` | Test unitario de ProjectionStorage | ✅ Completo |
| `tests/projection/unit_tests/projection_features_test.cc` | Test de backward compatibility | ✅ Completo |
| `tests/projection/scripts/test_label_support.sh` | Test end-to-end de labels | ✅ Completo |
| `tests/projection/scripts/test_quick.sh` | Test rápido de todas las features | ✅ Completo |

---

## 🎯 Estado de Implementación (Timeline)

### ✅ Fase 0: Infraestructura Básica (COMPLETA - Oct 2025)
- [x] Proyecciones con solo estructura (nodes, edges)
- [x] Sistema de catálogo (ProjectionManager)
- [x] USE GRAPH para context switching
- [x] Enrutamiento dinámico en GQLModel

### ✅ Fase 1: Soporte de Labels (COMPLETA - Oct 17, 2025)
- [x] Sintaxis `INCLUDE LABELS`
- [x] Índices duales: node_label, label_node, edge_label, label_edge
- [x] Queries con filtros por label: `MATCH (n:User)`
- [x] Tests de verificación completos

### ✅ Fase 2: Soporte de Properties (COMPLETA - Oct 17, 2025)
- [x] Sintaxis `INCLUDE PROPERTIES`
- [x] Índices duales: node_key_value, key_value_node, edge_key_value, key_value_edge
- [x] Queries con acceso a propiedades: `MATCH (n) RETURN n.age`
- [x] Queries con filtros: `WHERE n.age > 30`
- [x] Tests de verificación completos

### 🚧 Fase 3: Pre-requisitos GNN (EN PLANIFICACIÓN - ETA: Q1 2026)
- [ ] MUTATE PROPERTY (3-4 semanas)
- [ ] Sistema de Procedimientos GQL (2-3 semanas)
- [ ] TensorStore maduro (3-4 semanas)
- [ ] Integración LibTorch (2-3 semanas)

**Total**: 12 semanas (1 dev) o 8 semanas (2 devs)

### 🔮 Fase 4-8: Arquitectura GNN Nativa (DISEÑADA - ETA: Q2-Q4 2026)
Ver [`02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md`](02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md) para detalles completos.

---

## 🔑 Conceptos Clave

### ¿Qué es una Proyección?

Una **proyección** es un **subgrafo materializado persistente en disco** que:
- Contiene solo los nodos/aristas que coinciden con un patrón MATCH
- Opcionalmente incluye labels (INCLUDE LABELS)
- Opcionalmente incluye properties (INCLUDE PROPERTIES)
- Se almacena en archivos B+Tree separados
- Es **estática** (no se actualiza automáticamente cuando el grafo principal cambia)

### ¿Para qué sirven las Proyecciones?

1. **Performance**: Consultar un subgrafo pequeño es más rápido que el grafo completo
   - Proyección de 1,000 nodos vs grafo de 1,000,000: ~100x más rápido

2. **Aislamiento**: Trabajar en un subgrafo sin afectar el grafo principal

3. **Versionado**: Crear múltiples vistas del grafo (ej: "users_2024", "users_2025")

4. **Machine Learning** (futuro): Base para GNN training y algoritmos analíticos

### Índices B+Tree Creados

#### Siempre (estructura básica):
- `node_id.{dir,leaf}` - Lista de nodos
- `from_to_edge.{dir,leaf}` - Aristas direccionales (from → to)
- `to_from_edge.{dir,leaf}` - Aristas inversas (to → from)

#### Si `INCLUDE LABELS`:
- `node_label.{dir,leaf}` - {node_id, label_id}
- `label_node.{dir,leaf}` - {label_id, node_id}
- `edge_label.{dir,leaf}` - {edge_id, label_id}
- `label_edge.{dir,leaf}` - {label_id, edge_id}

#### Si `INCLUDE PROPERTIES`:
- `node_key_value.{dir,leaf}` - {node_id, key_id, value_id}
- `key_value_node.{dir,leaf}` - {key_id, value_id, node_id}
- `edge_key_value.{dir,leaf}` - {edge_id, key_id, value_id}
- `key_value_edge.{dir,leaf}` - {key_id, value_id, edge_id}

---

## 📝 Comandos de Referencia Rápida

### Crear Proyecciones

```gql
-- Básica (solo estructura)
MATCH (n)-[e]->(m) RETURN PROJECT("nombre")

-- Con etiquetas
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)

-- Con propiedades
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE PROPERTIES)

-- Completa (todo)
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)
```

### Usar Proyecciones

```gql
-- Cambiar a proyección
USE "nombre"

-- Consultar
MATCH (n) WHERE n.age > 30 RETURN n.name, n.age

-- Volver al grafo principal
USE CURRENT_GRAPH
```

### Verificar Proyecciones

```bash
# Ver archivos en disco
ls -lh data/dbs/gql/<database>/projections/<nombre>/

# Ver tamaño
du -sh data/dbs/gql/<database>/projections/<nombre>/

# Contar elementos
USE "nombre" MATCH (n) RETURN count(n)
USE "nombre" MATCH ()-[e]->() RETURN count(e)
```

---

## 🧪 Tests de Verificación Rápida

### Test 1: Funcionalidad Básica

```bash
# 1. Importar datos de ejemplo
./build/Release/bin/mdb import data/example/gql/posts/posts.gql test_db

# 2. Iniciar servidor
./build/Release/bin/mdb server test_db &

# 3. Crear proyección
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]->(m) RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)'

# 4. Verificar estructura
curl -X POST http://localhost:1234/gql --data \
  'USE "test" MATCH (n) RETURN count(n)'

# 5. Verificar labels
curl -X POST http://localhost:1234/gql --data \
  'USE "test" MATCH (n:User) RETURN count(n)'

# 6. Verificar properties
curl -X POST http://localhost:1234/gql --data \
  'USE "test" MATCH (n) WHERE n.age > 30 RETURN n.name, n.age LIMIT 5'
```

### Test 2: Script Automatizado

```bash
# Usar el script de test rápido
cd tests/projection/scripts
./test_quick.sh
```

---

## ❓ Preguntas Frecuentes

### ¿Cuándo se actualiza una proyección?

Las proyecciones son **estáticas**. NO se actualizan automáticamente. Debes recrearlas manualmente:

```bash
# Borrar proyección vieja
rm -rf data/dbs/gql/<db>/projections/<nombre>/

# Crear nueva versión
MATCH (patrón) RETURN PROJECT("nombre" ...)
```

### ¿Puedo tener múltiples proyecciones?

Sí, tantas como quieras:

```gql
PROJECT("usuarios")
PROJECT("posts")
PROJECT("comentarios")
PROJECT("usuarios_activos")
```

Cambias entre ellas con `USE "nombre"`.

### ¿Cuánto espacio ocupan?

Depende del tamaño del subgrafo y las opciones:

- **Solo estructura**: ~10-20% del tamaño del grafo principal
- **Con labels**: ~20-40%
- **Con properties**: ~50-80%
- **Completa (todo)**: ~70-100%

### ¿Es más rápido consultar una proyección?

SÍ, si la proyección es significativamente más pequeña:

- Proyección de 1,000 nodos vs grafo de 1,000,000: **~100x más rápido**
- Proyección de 100,000 nodos vs grafo de 1,000,000: **~5-10x más rápido**
- Proyección de 900,000 nodos vs grafo de 1,000,000: **~1.1x más rápido** (no vale la pena)

---

## 🐛 Debugging

### Problema: "Propiedades retornan NULL"

**Causa común**: Patrón MATCH incorrecto que no coincide con ningún edge.

```gql
-- ❌ INCORRECTO: Si no hay edges User->User, proyección vacía
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)

-- ✅ CORRECTO: Usa patrón que coincida con tus datos
MATCH (u)-[e]->(v) RETURN PROJECT("test" INCLUDE PROPERTIES)
```

**Solución completa**: Ver [`04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md`](04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md)

### Problema: "Cannot use node labels with projection"

**Causa**: Proyección creada sin `INCLUDE LABELS`.

**Solución**:

```gql
-- Recrear proyección con labels
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)
```

### Problema: "Cannot access node properties with projection"

**Causa**: Proyección creada sin `INCLUDE PROPERTIES`.

**Solución**:

```gql
-- Recrear proyección con properties
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE PROPERTIES)
```

---

## 📞 Contacto y Contribuciones

Para reportar bugs o sugerir mejoras:
- **GitHub Issues**: https://github.com/MillenniumDB/MillenniumDB/issues
- **Documentación del proyecto**: `MillenniumDB.wiki/`

---

## 📜 Historial de Cambios

### v2.1 (Noviembre 1, 2025) - Investigación Exhaustiva + Consolidación
- ✅ Investigación exhaustiva del sistema PROJECT (1,987 líneas)
- ✅ Consolidados tests en `tests/projection/` (estructura estándar)
- ✅ Movida guía de navegación a `docs/GUIA_NAVEGACION.md`
- ✅ Archivados documentos históricos (Oct 20, 2025)
- ✅ Root limpio con solo 3 archivos .md esenciales

### v2.0 (Octubre 20, 2025) - Reorganización Completa
- ✅ Reorganizada toda la documentación en 4 categorías lógicas
- ✅ Creado índice maestro de navegación
- ✅ Organizados tests de proyecciones
- ✅ Agregadas guías de inicio rápido para 4 perfiles de usuarios
- ✅ Documentación exhaustiva del estado actual y roadmap GNN

### v1.0 (Octubre 17, 2025) - Fase 2 Completada
- ✅ Implementación completa de INCLUDE PROPERTIES
- ✅ Implementación completa de INCLUDE LABELS
- ✅ Sistema PROJECT/USE 100% funcional
- ✅ Tests de verificación completos
- ✅ Documentación inicial creada

---

**Estado de la documentación**: ✅ Completa y actualizada
**Próxima actualización prevista**: Al completar pre-requisitos GNN (Q1 2026)
