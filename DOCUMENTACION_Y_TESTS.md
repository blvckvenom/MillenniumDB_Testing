# 📚 Guía de Navegación: Documentación y Tests

**Última actualización**: 20 de Octubre, 2025

---

## 🎯 Inicio Rápido

### ¿Qué buscas?

| Si quieres... | Ve aquí |
|--------------|---------|
| **Aprender a usar PROJECT/USE** | [`docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md`](docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md) |
| **Entender el código interno** | [`docs/01_PROJECT_System/PROJECT_CODIGO_INTERNO.md`](docs/01_PROJECT_System/PROJECT_CODIGO_INTERNO.md) |
| **Implementar GNN/ML** | [`docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md`](docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md) |
| **Ejecutar tests** | [`tests_projection/README.md`](tests_projection/README.md) |
| **Debug de problemas** | [`docs/04_Debugging_History/`](docs/04_Debugging_History/) |
| **Índice completo** | [`docs/README.md`](docs/README.md) ⭐ |

---

## 📂 Estructura del Proyecto

```
MillenniumDB/
│
├── docs/ ................................. Documentación completa reorganizada
│   ├── README.md ........................ Índice maestro (EMPIEZA AQUÍ)
│   ├── 01_PROJECT_System/ ............... Sistema de Proyecciones (COMPLETO)
│   ├── 02_GNN_Architecture/ ............. Arquitectura GNN (ROADMAP)
│   ├── 03_Implementation_Phases/ ........ Historial de fases
│   └── 04_Debugging_History/ ............ Problemas resueltos
│
├── tests_projection/ ..................... Tests de proyecciones
│   ├── README.md ........................ Guía de tests
│   ├── unit_tests/ ...................... Tests C++
│   ├── integration_tests/ ............... Tests Python (futuro)
│   └── scripts/ ......................... Scripts bash
│
├── src/ .................................. Código fuente
│   ├── query/executor/binding_iter/aggregation/gql/
│   │   └── agg_project.h ................ Implementación de PROJECT
│   ├── graph_models/gql/projection/
│   │   ├── projection_storage.{h,cc} .... Almacenamiento en disco
│   │   ├── projection_manager.{h,cc} .... Catálogo de proyecciones
│   │   └── projection_query_context.h ... Contexto de USE GRAPH
│   └── graph_models/gql/
│       └── gql_model.{h,cc} ............. Enrutamiento dinámico
│
├── tests/ ................................ Tests principales del proyecto
│   ├── gql/ ............................. Tests GQL
│   ├── sparql/ .......................... Tests SPARQL
│   └── mql/ ............................. Tests MQL (Quad Model)
│
├── scripts/ .............................. Scripts del proyecto
│   ├── run-tests ........................ Ejecutar todos los tests
│   └── query ............................ Ejecutar query HTTP
│
├── data/example/ ......................... Datos de ejemplo
│   └── gql/posts/ ....................... Dataset posts (usado en tests)
│
└── DOCUMENTACION_Y_TESTS.md .............. Este archivo
```

---

## 📖 Documentación Principal

### **👉 [`docs/README.md`](docs/README.md) ⭐**

**El índice maestro de toda la documentación.** Incluye:

- 📚 Guía completa de navegación
- 🚀 Inicio rápido para 4 perfiles de usuarios
- 📊 Estado de implementación (timeline)
- 🔑 Conceptos clave
- 📝 Comandos de referencia rápida
- ❓ FAQ
- 🐛 Guía de debugging

**Tiempo de lectura**: 10 minutos (índice), 30-120 minutos (documentos enlazados)

---

## 🧪 Tests

### **👉 [`tests_projection/README.md`](tests_projection/README.md)**

**Guía completa de tests de proyecciones.** Incluye:

- 🗂️ Estructura de tests
- 🧪 Tests unitarios (C++)
- 🔄 Tests de integración (Bash)
- 🚀 Cómo ejecutar tests
- ➕ Cómo agregar nuevos tests
- 🐛 Debugging de tests fallidos

**Tiempo de lectura**: 5-10 minutos

---

## 🎓 Rutas de Aprendizaje

### Para Usuarios Nuevos

1. **Lee** [`docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md`](docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md) (30 min)
2. **Ejecuta** ejemplos del resumen (15 min)
3. **Ejecuta** [`tests_projection/scripts/test_quick.sh`](tests_projection/scripts/test_quick.sh) (5 min)

**Total**: ~50 minutos para entender y usar el sistema.

---

### Para Desarrolladores

1. **Lee** [`docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md`](docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md) (30 min)
2. **Lee** [`docs/01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md`](docs/01_PROJECT_System/GUIA_PROJECT_Y_USE_GRAPH.md) - Sección "Arquitectura Interna" (45 min)
3. **Lee** [`docs/01_PROJECT_System/PROJECT_CODIGO_INTERNO.md`](docs/01_PROJECT_System/PROJECT_CODIGO_INTERNO.md) (60 min)
4. **Lee** código fuente con referencias de líneas

**Total**: ~3 horas para entender la implementación completa.

---

### Para Implementar Machine Learning

1. **Lee** [`docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md`](docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md) (60 min)
   - ⚠️ **IMPORTANTE**: Muestra qué falta ANTES de poder implementar GNN
   - 4 pre-requisitos críticos con especificaciones completas
   - Timeline: 12 semanas

2. **Lee** [`docs/02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md`](docs/02_GNN_Architecture/ARQUITECTURA_GNN_NATIVA.md) (90 min)
   - Visión completa del motor GNN nativo

3. **Lee** [`docs/02_GNN_Architecture/ANALISIS_COMPARATIVO_NEO4J_GDS.md`](docs/02_GNN_Architecture/ANALISIS_COMPARATIVO_NEO4J_GDS.md) (60 min)
   - Comparación con Neo4j GDS

**Total**: ~4 horas para entender el roadmap completo de ML.

---

## 📊 Estado Actual del Proyecto

### ✅ Completado (Octubre 2025)

| Feature | Documentación | Tests | Estado |
|---------|---------------|-------|--------|
| PROJECT básico | ✅ | ✅ | ✅ COMPLETO |
| INCLUDE LABELS | ✅ | ✅ | ✅ COMPLETO |
| INCLUDE PROPERTIES | ✅ | ✅ | ✅ COMPLETO |
| USE GRAPH | ✅ | ✅ | ✅ COMPLETO |
| Enrutamiento dinámico | ✅ | ⚠️ | ✅ COMPLETO |

**Sistema PROJECT/USE**: 100% funcional y documentado.

---

### 🚧 En Roadmap (Q1-Q4 2026)

| Feature | Documentación | Esfuerzo | ETA |
|---------|---------------|----------|-----|
| MUTATE PROPERTY | ✅ Spec completa | 3-4 semanas | Q1 2026 |
| Sistema de Procedimientos | ✅ Spec completa | 2-3 semanas | Q1 2026 |
| TensorStore maduro | ✅ Spec completa | 3-4 semanas | Q1 2026 |
| Integración LibTorch | ✅ Spec completa | 2-3 semanas | Q1 2026 |
| Motor GNN nativo (Fase 0-4) | ✅ Arquitectura completa | 24+ semanas | Q2-Q4 2026 |

**Pre-requisitos GNN**: 100% documentados, 0% implementados.

---

## 🔗 Enlaces Rápidos

### Documentos Más Importantes

1. **[`docs/README.md`](docs/README.md)** - Índice maestro
2. **[`docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md`](docs/01_PROJECT_System/PROJECT_USE_RESUMEN.md)** - Inicio rápido
3. **[`docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md`](docs/02_GNN_Architecture/ROADMAP_PREREQUISITOS_GNN_NATIVA.md)** - Siguiente fase
4. **[`tests_projection/README.md`](tests_projection/README.md)** - Ejecutar tests

### Código Fuente Clave

1. **`src/query/executor/binding_iter/aggregation/gql/agg_project.h`** (428 líneas)
   - Implementación completa de PROJECT

2. **`src/graph_models/gql/projection/projection_storage.cc`** (369 líneas)
   - Escritura a índices B+Tree

3. **`src/graph_models/gql/gql_model.cc`** (200 líneas)
   - Enrutamiento dinámico (get_node_key_value, get_node_label, etc.)

### Tests Clave

1. **`tests_projection/unit_tests/projection_features_test.cc`**
   - Test de backward compatibility

2. **`tests_projection/scripts/test_quick.sh`**
   - Test rápido de todas las features

---

## 📝 Comandos de Referencia Ultra-Rápida

```bash
# CREAR PROYECCIONES
MATCH (n)-[e]->(m) RETURN PROJECT("nombre")                                    # Básica
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)                     # Con labels
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE PROPERTIES)                 # Con properties
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)  # Completa

# USAR PROYECCIONES
USE "nombre"                                   # Cambiar a proyección
MATCH (n) WHERE n.age > 30 RETURN n.name      # Consultar
USE CURRENT_GRAPH                              # Volver al grafo principal

# VERIFICAR
ls -lh data/dbs/gql/<db>/projections/<nombre>/              # Ver archivos
du -sh data/dbs/gql/<db>/projections/<nombre>/              # Ver tamaño
USE "nombre" MATCH (n) RETURN count(n)                       # Contar nodos
USE "nombre" MATCH ()-[e]->() RETURN count(e)                # Contar aristas

# TESTS
./scripts/run-tests                                          # Todos los tests
./build/Release/bin/projection_features_test                 # Test unitario
./tests_projection/scripts/test_quick.sh                     # Test integración
```

---

## 🐛 Debugging Rápido

### Problema: "Propiedades retornan NULL"

**Causa**: Patrón MATCH incorrecto.

```gql
-- ❌ INCORRECTO (si no hay edges User→User)
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test" INCLUDE PROPERTIES)

-- ✅ CORRECTO
MATCH (u)-[e]->(v) RETURN PROJECT("test" INCLUDE PROPERTIES)
```

**Ver**: [`docs/04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md`](docs/04_Debugging_History/BUG_INVESTIGATION_SUMMARY.md)

---

### Problema: "Cannot use node labels"

**Causa**: Proyección sin `INCLUDE LABELS`.

**Solución**:
```gql
-- Recrear proyección
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)
```

---

## 📞 Contacto

- **GitHub Issues**: https://github.com/MillenniumDB/MillenniumDB/issues
- **Documentación Wiki**: `MillenniumDB.wiki/`

---

## 📜 Versión

**v2.0** (Octubre 20, 2025)

Cambios principales:
- ✅ Reorganizada toda la documentación en `docs/`
- ✅ Reorganizados tests en `tests_projection/`
- ✅ Creados índices maestros con guías de navegación
- ✅ Agregados 4 perfiles de usuario (nuevo, dev, ML, debug)
- ✅ Documentación 100% exhaustiva y actualizada

**Próxima actualización**: Al completar pre-requisitos GNN (Q1 2026)

---

**¿Perdido? → Empieza por [`docs/README.md`](docs/README.md) ⭐**
