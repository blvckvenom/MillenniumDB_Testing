# Índice de Documentación - Sistema de Proyecciones MillenniumDB

## Documentos Disponibles

### 1. **GUIA_PROJECT_Y_USE_GRAPH.md** 📚
**Descripción**: Guía completa y detallada (50+ páginas)
**Contenido**:
- Introducción conceptual
- Sintaxis completa de PROJECT y USE
- Proceso interno paso a paso
- Estructura de archivos creados
- Arquitectura con diagramas de flujo
- Ejemplos completos de uso
- Limitaciones y errores comunes
- Comandos de referencia

**Cuándo leer**: Cuando necesitas entender TODO el sistema en profundidad

---

### 2. **PROJECT_USE_RESUMEN.md** ⚡
**Descripción**: Resumen ejecutivo (10 páginas)
**Contenido**:
- Qué hacen PROJECT y USE (explicación simple)
- Ejemplo completo paso a paso
- Comparación visual: grafo principal vs proyección
- Ejemplos de uso real (redes sociales, e-commerce, etc.)
- Opciones de PROJECT y cuándo usar cada una
- Flujo de trabajo típico
- Errores comunes con soluciones
- Comandos de referencia rápida

**Cuándo leer**: Cuando necesitas empezar rápido o refrescar conceptos

---

### 3. **PROJECT_CODIGO_INTERNO.md** 💻
**Descripción**: Código interno detallado con ejemplos C++
**Contenido**:
- Flujo completo con código real
- Fase 1: Parsing (query_visitor.cc)
- Fase 2: Ejecución del MATCH
- Fase 3: Inicialización (agg_project.h)
- Fase 4: Procesamiento de bindings (5 pasos detallados)
- Fase 5: Escritura a disco (projection_storage.cc)
- Fase 6: Finalización
- Uso con USE y enrutamiento dinámico
- Tabla con archivos y líneas exactas

**Cuándo leer**: Cuando necesitas modificar el código o debuggear

---

### 4. **PROJECTION_PROPERTIES_WORKING.md** ✅
**Descripción**: Documentación de verificación (Phase 2 completa)
**Contenido**:
- Resumen del problema investigado
- Root cause del bug (era un patrón MATCH incorrecto)
- Verificación completa con tests
- Ejemplos de queries funcionales
- Detalles de implementación
- Performance notes

**Cuándo leer**: Para verificar que el soporte de propiedades funciona correctamente

---

### 5. **BUG_INVESTIGATION_SUMMARY.md** 🔍
**Descripción**: Documentación del proceso de investigación
**Contenido**:
- Descripción del problema inicial
- Pasos de investigación tomados
- Root cause encontrado
- Solución aplicada
- Verificación de la solución

**Cuándo leer**: Para entender el proceso de debugging que llevó a la solución

---

## Guía de Lectura Recomendada

### Para Usuarios Nuevos del Sistema
```
1. PROJECT_USE_RESUMEN.md (30 minutos)
   ↓
2. Probar ejemplos del resumen
   ↓
3. GUIA_PROJECT_Y_USE_GRAPH.md - Secciones específicas según necesidad
```

### Para Desarrolladores que van a Modificar el Código
```
1. PROJECT_USE_RESUMEN.md (orientación general)
   ↓
2. GUIA_PROJECT_Y_USE_GRAPH.md - Sección "Arquitectura Interna"
   ↓
3. PROJECT_CODIGO_INTERNO.md (código detallado)
   ↓
4. Leer código fuente con las referencias de líneas
```

### Para Debugging de Problemas
```
1. BUG_INVESTIGATION_SUMMARY.md (proceso de investigación)
   ↓
2. PROJECTION_PROPERTIES_WORKING.md (verificación)
   ↓
3. PROJECT_CODIGO_INTERNO.md - Fase específica del problema
```

---

## Archivos de Código Fuente Principales

### Core de Proyecciones

| Archivo | Descripción |
|---------|-------------|
| `src/query/executor/binding_iter/aggregation/gql/agg_project.h` | Función de agregación PROJECT |
| `src/graph_models/gql/projection/projection_storage.{h,cc}` | Escritura a índices B+Tree |
| `src/graph_models/gql/projection/projection_manager.{h,cc}` | Gestión de catálogo |
| `src/graph_models/gql/projection/projection_query_context.{h,cc}` | Contexto para USE |

### Parsing y Ejecución

| Archivo | Descripción |
|---------|-------------|
| `src/query/parser/grammar/gql/query_visitor.cc` | Parser de PROJECT y USE |
| `src/query/query_context.{h,cc}` | Estado de proyección activa |
| `src/graph_models/gql/gql_model.{h,cc}` | Enrutamiento dinámico de índices |

---

## Comandos Rápidos

### Crear Proyecciones

```bash
# Básica (solo estructura)
MATCH (n)-[e]->(m) RETURN PROJECT("nombre")

# Con etiquetas
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)

# Con propiedades
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE PROPERTIES)

# Completa
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)
```

### Usar Proyecciones

```bash
# Cambiar a proyección
USE "nombre"

# Consultar
MATCH (n) WHERE n.age > 30 RETURN n.name

# Volver al grafo principal
USE CURRENT_GRAPH
```

### Verificar Proyecciones

```bash
# Ver archivos en disco
ls -lh data/dbs/gql/<db>/projections/<nombre>/

# Ver tamaño
du -sh data/dbs/gql/<db>/projections/<nombre>/

# Contar elementos
USE "nombre" MATCH (n) RETURN count(n)
USE "nombre" MATCH ()-[e]->() RETURN count(e)
```

---

## Estructura de Directorios de Proyecciones

```
data/dbs/gql/<database>/projections/<nombre>/
├── catalog.dat                    # Metadatos
│
# Índices Básicos (SIEMPRE):
├── node_id.{dir,leaf}             # Nodos
├── from_to_edge.{dir,leaf}        # Aristas (from→to)
├── to_from_edge.{dir,leaf}        # Aristas inversas (to→from)
│
# Si INCLUDE LABELS:
├── node_label.{dir,leaf}          # {node, label}
├── label_node.{dir,leaf}          # {label, node}
├── edge_label.{dir,leaf}          # {edge, label}
├── label_edge.{dir,leaf}          # {label, edge}
│
# Si INCLUDE PROPERTIES:
├── node_key_value.{dir,leaf}      # {node, key, value}
├── key_value_node.{dir,leaf}      # {key, value, node}
├── edge_key_value.{dir,leaf}      # {edge, key, value}
└── key_value_edge.{dir,leaf}      # {key, value, edge}
```

---

## Flujo Visual Simplificado

### PROJECT

```
MATCH (patrón) → [ResultSet] → AggProject → [Archivos B+Tree en disco]
                                    ↓
                              ProjectionManager
                              (actualiza catálogo)
```

### USE

```
USE "nombre" → QueryContext.load_projection() → [Abre índices B+Tree]
                                                         ↓
MATCH ... → Necesita índice → GQLModel.get_*() → ¿Proyección activa?
                                                    ├─ SÍ → índice proyección
                                                    └─ NO → índice grafo principal
```

---

## Tests de Verificación

### Test 1: Funcionalidad Básica

```bash
# Crear proyección
curl -X POST http://localhost:1234/gql --data \
'MATCH (u)-[e]->(v) RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)'

# Verificar estructura
USE "test" MATCH (n) RETURN count(n)
USE "test" MATCH ()-[e]->() RETURN count(e)

# Verificar etiquetas
USE "test" MATCH (n:User) RETURN count(n)

# Verificar propiedades
USE "test" MATCH (n) WHERE n.age > 30 RETURN n.name, n.age LIMIT 5
```

### Test 2: Comparación Main Graph vs Proyección

```bash
# Main graph
echo "Main graph:"
curl -X POST http://localhost:1234/gql --data 'MATCH (n) RETURN count(n)'
curl -X POST http://localhost:1234/gql --data 'MATCH (n:User) WHERE n.age > 30 RETURN count(n)'

# Proyección
echo "Projection:"
curl -X POST http://localhost:1234/gql --data 'USE "test" MATCH (n) RETURN count(n)'
curl -X POST http://localhost:1234/gql --data 'USE "test" MATCH (n:User) WHERE n.age > 30 RETURN count(n)'

# Los números deberían coincidir si la proyección incluye todo el grafo
```

---

## Recursos Adicionales

### Archivos de Tests/Scripts

- `test_projection_e2e.sh` - Test end-to-end
- `test_projection_persistence.sh` - Test de persistencia
- `verify_persistence.sh` - Verificación de datos persistidos

### Archivos de Ejemplo

- `data/example/gql/posts/posts.gql` - Datos de ejemplo
- `tests/projection/` - Tests de integración

### Herramientas de Debugging

```bash
# Ver logs del servidor
tail -f /tmp/debug_server.log

# Ver logs de creación de proyección
grep "AggProject" /tmp/debug_server.log

# Ver logs de property extraction
grep "PROPERTY EXTRACTION" /tmp/debug_server.log

# Ver logs de enrutamiento
grep "GQLModel" /tmp/debug_server.log
```

---

## Preguntas Frecuentes

### ¿Cuándo se actualiza una proyección?

Las proyecciones son **estáticas**. Una vez creadas, NO se actualizan automáticamente cuando el grafo principal cambia. Debes recrearlas manualmente:

```bash
# Borrar proyección vieja (manual)
rm -rf data/dbs/gql/<db>/projections/<nombre>/

# Crear nueva versión
MATCH (patrón) RETURN PROJECT("nombre" ...)
```

### ¿Puedo tener múltiples proyecciones?

Sí, puedes tener tantas como quieras:

```bash
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

SÍ, si la proyección es significativamente más pequeña que el grafo principal:

- Proyección de 1,000 nodos vs grafo de 1,000,000: **~100x más rápido**
- Proyección de 100,000 nodos vs grafo de 1,000,000: **~5-10x más rápido**
- Proyección de 900,000 nodos vs grafo de 1,000,000: **~1.1x más rápido** (no vale la pena)

---

## Contacto y Contribuciones

Para reportar bugs o sugerir mejoras:
- GitHub Issues: https://github.com/MillenniumDB/MillenniumDB/issues
- Email del equipo: [ver README]

---

**Última actualización**: Octubre 17, 2025
**Versión de documentación**: 1.0
**Estado de implementación**: Phase 1 (Labels) ✅ | Phase 2 (Properties) ✅
