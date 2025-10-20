# Guía Completa: PROJECT y USE GRAPH en MillenniumDB

**Fecha**: 17 de Octubre, 2025
**Autor**: Documentación del sistema de proyecciones GQL

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Función PROJECT](#función-project)
3. [Cláusula USE GRAPH](#cláusula-use-graph)
4. [Arquitectura Interna](#arquitectura-interna)
5. [Ejemplos Completos](#ejemplos-completos)
6. [Limitaciones y Errores Comunes](#limitaciones-y-errores-comunes)

---

## Introducción

MillenniumDB implementa dos funcionalidades complementarias para trabajar con subgrafos:

- **`PROJECT()`**: Crea una proyección (subgrafo filtrado) desde el grafo principal
- **`USE "nombre"`**: Cambia el contexto de consulta a una proyección existente

Estas funcionalidades permiten:
- Crear vistas materializadas de porciones específicas del grafo
- Consultar subgrafos de manera eficiente sin afectar el grafo principal
- Incluir opcionalmente etiquetas y propiedades en las proyecciones

---

## Función PROJECT

### Sintaxis

```gql
MATCH <pattern>
RETURN PROJECT("<nombre>" [INCLUDE LABELS] [INCLUDE PROPERTIES])
```

### Parámetros

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|-------------|-------------|
| `nombre` | String | Sí | Nombre único para la proyección |
| `INCLUDE LABELS` | Flag | No | Incluye etiquetas de nodos y aristas |
| `INCLUDE PROPERTIES` | No | No | Incluye propiedades de nodos y aristas |

**IMPORTANTE**: Los parámetros opcionales NO se separan con comas.

### ¿Qué hace PROJECT?

La función `PROJECT()` crea un **subgrafo materializado** que contiene:

1. **Nodos y aristas** que coinciden con el patrón MATCH
2. **Opcionalmente**:
   - Etiquetas (si se especifica `INCLUDE LABELS`)
   - Propiedades (si se especifica `INCLUDE PROPERTIES`)

### Proceso Interno

Cuando ejecutas `PROJECT()`, el sistema realiza estos pasos:

#### 1. Inicialización (AggProject::initialize_if_needed)

```cpp
// Archivo: src/query/executor/binding_iter/aggregation/gql/agg_project.h
// Líneas: 49-121

void initialize_if_needed() {
    // 1. Evalúa el nombre de la proyección
    ObjectId name_oid = expr->eval(*binding);
    projection_name = Conversions::unpack_string(name_oid);

    // 2. Crea el directorio de la proyección
    auto& proj_manager = ProjectionManager::get_instance();
    std::string proj_dir = proj_manager.create_projection(projection_name);

    // 3. Configura las características (features)
    ProjectionStorage::Features features;
    features.include_node_labels = options.include_labels;
    features.include_edge_labels = options.include_labels;
    features.include_node_properties = options.include_properties;
    features.include_edge_properties = options.include_properties;

    // 4. Inicializa el almacenamiento de la proyección
    projection_storage = std::make_unique<ProjectionStorage>(
        proj_dir, db_folder, projection_name, features
    );
    projection_storage->init();
}
```

**Resultado**: Se crea la estructura de directorios y archivos de índices B+Tree.

#### 2. Procesamiento de Resultados (AggProject::process)

```cpp
// Archivo: src/query/executor/binding_iter/aggregation/gql/agg_project.h
// Líneas: 123-379

void process() override {
    // Primer paso: Recolectar nodos y aristas del MATCH
    // Segundo paso: Agregar nodos a la proyección
    // Tercer paso: Agregar aristas a la proyección
    // Cuarto paso: Extraer etiquetas (si INCLUDE LABELS)
    // Quinto paso: Extraer propiedades (si INCLUDE PROPERTIES)
}
```

Veamos cada paso en detalle:

##### Paso 1: Recolección de Nodos y Aristas

```cpp
// Líneas 143-227
std::map<ObjectId, std::unordered_map<std::string, ObjectId>> node_properties;
std::map<ObjectId, std::unordered_map<std::string, ObjectId>> edge_properties;
std::map<ObjectId, bool> edges_seen;

for (size_t i = 0; i < binding->size; i++) {
    VarId var_id(i);
    ObjectId oid = (*binding)[var_id];
    auto var_name = get_query_ctx().get_var_name(var_id);
    auto type = GQL_OID::get_type(oid);

    if (type == GQL_OID::Type::NODE) {
        node_properties[oid] = std::unordered_map<std::string, ObjectId>();
    }
    else if (type == GQL_OID::Type::DIRECTED_EDGE ||
             type == GQL_OID::Type::UNDIRECTED_EDGE) {
        edges_seen[oid] = (type == GQL_OID::Type::DIRECTED_EDGE);
    }
}
```

**¿Qué hace?**
- Itera sobre todas las variables en el resultado del MATCH
- Identifica nodos (TYPE::NODE) y aristas (TYPE::DIRECTED_EDGE / UNDIRECTED_EDGE)
- Infiere los extremos de las aristas basándose en la secuencia de nodos

##### Paso 2: Agregar Nodos

```cpp
// Líneas 229-238
for (const auto& [node_id, props] : node_properties) {
    ProjectedNode node;
    node.node_id = node_id;
    node.properties = props;
    projection_storage->add_node(node);
}
```

**¿Qué hace?**
- Inserta cada nodo en el índice `node_id` de la proyección
- Escribe en: `data/dbs/gql/<db>/projections/<nombre>/node_id.{dir,leaf}`

##### Paso 3: Agregar Aristas

```cpp
// Líneas 240-268
for (const auto& [edge_id, is_directed] : edges_seen) {
    ProjectedEdge edge;
    edge.edge_id = edge_id;
    edge.is_directed = is_directed;
    edge.from_node = endpoints_it->second.first;
    edge.to_node = endpoints_it->second.second;

    projection_storage->add_edge(edge);
}
```

**¿Qué hace?**
- Inserta cada arista en los índices:
  - `from_to_edge`: {from, to, edge}
  - `to_from_edge`: {to, from, edge} (para consultas inversas)

##### Paso 4: Extraer Etiquetas (si INCLUDE LABELS)

```cpp
// Líneas 270-319
if (options.include_labels) {
    // Etiquetas de nodos
    for (const auto& [node_id, props] : node_properties) {
        BptIter<2> it = gql_model.node_label->get_range(
            &interruption,
            { node_id.id, 0 },
            { node_id.id, UINT64_MAX }
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId label_id((*record)[1]);
            projection_storage->add_node_label(node_id, label_id);
            record = it.next();
        }
    }

    // Similar para etiquetas de aristas...
}
```

**¿Qué hace?**
- Escanea el índice `node_label` del grafo principal para cada nodo
- Copia todas las etiquetas encontradas a los índices de la proyección:
  - `node_label`: {node, label}
  - `label_node`: {label, node}
  - `edge_label`: {edge, label}
  - `label_edge`: {label, edge}

##### Paso 5: Extraer Propiedades (si INCLUDE PROPERTIES)

```cpp
// Líneas 321-373
if (options.include_properties) {
    // Propiedades de nodos
    for (const auto& [node_id, props] : node_properties) {
        BptIter<3> it = gql_model.node_key_value->get_range(
            &interruption,
            { node_id.id, 0, 0 },
            { node_id.id, UINT64_MAX, UINT64_MAX }
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId key_id((*record)[1]);
            ObjectId value_id((*record)[2]);
            projection_storage->add_node_property(node_id, key_id, value_id);
            record = it.next();
        }
    }

    // Similar para propiedades de aristas...
}
```

**¿Qué hace?**
- Escanea el índice `node_key_value` del grafo principal para cada nodo
- Copia todas las propiedades encontradas a los índices **bidireccionales**:
  - `node_key_value`: {node, key, value} (para acceso por nodo)
  - `key_value_node`: {key, value, node} (para búsquedas por propiedad)
  - `edge_key_value`: {edge, key, value}
  - `key_value_edge`: {key, value, edge}

#### 3. Finalización (AggProject::get)

```cpp
// Líneas 382-409
ObjectId get() override {
    // 1. Flush de escrituras pendientes
    projection_storage->flush();

    // 2. Actualizar cache del ProjectionManager
    auto& proj_manager = ProjectionManager::get_instance();
    proj_manager.scan_projections();

    // 3. Retornar el nombre de la proyección como ObjectId
    return projection_name_oid;
}
```

**¿Qué hace?**
- Asegura que todos los datos estén escritos en disco
- Actualiza el catálogo para que la proyección sea inmediatamente visible
- Retorna el nombre de la proyección como resultado de la consulta

### Estructura de Archivos Creados

Cuando creas una proyección, se genera esta estructura:

```
data/dbs/gql/<database>/projections/<nombre>/
├── catalog.dat                  # Metadatos de la proyección
├── node_id.dir                  # Índice de nodos (directorio B+Tree)
├── node_id.leaf                 # Índice de nodos (hojas B+Tree)
├── from_to_edge.dir             # Índice (from, to, edge)
├── from_to_edge.leaf
├── to_from_edge.dir             # Índice inverso (to, from, edge)
├── to_from_edge.leaf
│
# Si INCLUDE LABELS:
├── node_label.dir               # Índice (node, label)
├── node_label.leaf
├── label_node.dir               # Índice (label, node)
├── label_node.leaf
├── edge_label.dir               # Índice (edge, label)
├── edge_label.leaf
├── label_edge.dir               # Índice (label, edge)
├── label_edge.leaf
│
# Si INCLUDE PROPERTIES:
├── node_key_value.dir           # Índice (node, key, value)
├── node_key_value.leaf
├── key_value_node.dir           # Índice (key, value, node)
├── key_value_node.leaf
├── edge_key_value.dir           # Índice (edge, key, value)
├── edge_key_value.leaf
└── key_value_edge.dir           # Índice (key, value, edge)
    key_value_edge.leaf
```

---

## Cláusula USE GRAPH

### Sintaxis

```gql
USE "<nombre_proyección>"
MATCH <pattern>
RETURN <resultados>
```

O para volver al grafo principal:

```gql
USE CURRENT_GRAPH
MATCH <pattern>
RETURN <resultados>
```

### ¿Qué hace USE?

La cláusula `USE` **cambia el contexto de ejecución** de todas las consultas subsiguientes para que usen los índices de la proyección en lugar del grafo principal.

### Proceso Interno

#### 1. Parsing (QueryVisitor::visitUseGraphClause)

```cpp
// Archivo: src/query/parser/grammar/gql/query_visitor.cc
// Líneas: 2354-2361

std::any QueryVisitor::visitUseGraphClause(GQLParser::UseGraphClauseContext* ctx) {
    if (ctx->graphExpression()) {
        visitGraphExpression(ctx->graphExpression());
    }
    return 0;
}
```

#### 2. Evaluación de Expresión (QueryVisitor::visitGraphExpression)

```cpp
// Líneas: 2224-2290

std::any QueryVisitor::visitGraphExpression(GQLParser::GraphExpressionContext* ctx) {
    // Caso 1: Nombre de proyección simple
    if (ctx->objectNameOrBindingVariable()) {
        std::string projection_name = ctx->objectNameOrBindingVariable()->getText();

        // Verificar que existe
        auto& proj_manager = ProjectionManager::get_instance();
        if (!proj_manager.projection_exists(projection_name)) {
            auto projections = proj_manager.list_projections();
            std::string available;
            for (auto& p : projections) {
                available += p.name + ", ";
            }
            throw QuerySemanticException(
                "Projection '" + projection_name + "' does not exist. "
                "Available projections: [" + available + "]"
            );
        }

        // Cargar proyección en QueryContext
        get_query_ctx().load_projection(projection_name);
        return 0;
    }

    // Caso 2: CURRENT_GRAPH (volver al grafo principal)
    if (ctx->currentGraph()) {
        get_query_ctx().clear_active_projection();
        return 0;
    }

    // Otros casos...
}
```

#### 3. Carga de Contexto (QueryContext::load_projection)

```cpp
// Archivo: src/query/query_context.cc
// Líneas: 56-61

void QueryContext::load_projection(const std::string& proj_name) {
    active_projection = proj_name;
    projection_ctx = std::make_unique<GQL::ProjectionQueryContext>(proj_name);
}
```

**¿Qué hace?**
- Establece `active_projection` = nombre de la proyección
- Crea un `ProjectionQueryContext` que abre todos los índices B+Tree de la proyección
- Este contexto permanece activo hasta que se ejecute `USE CURRENT_GRAPH` o termine la sesión

#### 4. Enrutamiento Dinámico de Índices (GQLModel::get_*)

Cuando una consulta necesita acceder a un índice, se usa **enrutamiento dinámico**:

```cpp
// Archivo: src/graph_models/gql/gql_model.cc
// Líneas: 53-67

BPlusTree<3>& GQLModel::get_from_to_edge() {
    auto& ctx = get_query_ctx();

    // ¿Estamos usando una proyección?
    if (ctx.is_using_projection()) {
        // Verificar que la proyección tenga el índice
        if (!ctx.projection_ctx || !ctx.projection_ctx->from_to_edge_index) {
            throw std::runtime_error(
                "Projection context not loaded for '" +
                ctx.active_projection + "'"
            );
        }
        // Usar índice de la proyección
        return *ctx.projection_ctx->from_to_edge_index;
    }

    // Usar índice del grafo principal
    return *from_to_edge;
}
```

**¿Qué hace?**
- Verifica si hay una proyección activa (`ctx.is_using_projection()`)
- Si hay proyección: retorna el índice B+Tree de la proyección
- Si no hay proyección: retorna el índice B+Tree del grafo principal

Este patrón se repite para **todos los índices**:
- `get_from_to_edge()` - Aristas por origen y destino
- `get_to_from_edge()` - Aristas inversas
- `get_node_label()` - Etiquetas de nodos
- `get_edge_label()` - Etiquetas de aristas
- `get_node_key_value()` - Propiedades de nodos
- `get_edge_key_value()` - Propiedades de aristas
- Y sus índices auxiliares...

#### 5. Detección de Índices Faltantes

Si intentas usar una funcionalidad no incluida en la proyección, obtienes un error descriptivo:

```cpp
// Ejemplo: Acceder a propiedades en proyección sin INCLUDE PROPERTIES
BPlusTree<3>& GQLModel::get_node_key_value() {
    auto& ctx = get_query_ctx();
    if (ctx.is_using_projection()) {
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_key_value_index) {
            throw std::runtime_error(
                "Cannot access node properties with projection '" +
                ctx.active_projection + "'.\n\n"
                "Reason: This projection does not include node property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"" + ctx.active_projection +
                "\", INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }
        return *ctx.projection_ctx->node_key_value_index;
    }
    return *node_key_value;
}
```

---

## Arquitectura Interna

### Diagrama de Flujo: PROJECT

```
Usuario ejecuta:
MATCH (n)-[e]->(m) RETURN PROJECT("mi_proj" INCLUDE LABELS INCLUDE PROPERTIES)
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 1. PARSER (query_visitor.cc)                              │
│    - Reconoce función PROJECT()                            │
│    - Extrae nombre y opciones (INCLUDE LABELS, etc.)       │
│    - Crea nodo AggProject en el árbol de ejecución         │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 2. EJECUCIÓN - MATCH (linear_pattern_path.cc, etc.)       │
│    - Ejecuta el patrón MATCH                               │
│    - Genera bindings (filas de resultados)                 │
│    - Cada binding contiene: nodos, aristas, propiedades    │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 3. AGREGACIÓN (agg_project.h::process)                    │
│    Para cada binding del MATCH:                            │
│    ┌──────────────────────────────────────────────┐       │
│    │ a) Recolectar nodos y aristas                 │       │
│    │ b) Agregar nodos a proyección                 │       │
│    │ c) Agregar aristas a proyección               │       │
│    │ d) Si INCLUDE LABELS:                         │       │
│    │    - Escanear node_label del grafo principal  │       │
│    │    - Copiar a node_label de proyección        │       │
│    │    - Similar para edge_label                  │       │
│    │ e) Si INCLUDE PROPERTIES:                     │       │
│    │    - Escanear node_key_value del grafo        │       │
│    │    - Copiar a node_key_value de proyección    │       │
│    │    - Similar para edge_key_value              │       │
│    └──────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 4. ALMACENAMIENTO (projection_storage.cc)                 │
│    - Escribe a archivos B+Tree en disco                    │
│    - Índices bidireccionales para propiedades              │
│    - Flush al finalizar                                    │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 5. REGISTRO (projection_manager.cc)                       │
│    - Actualiza catálogo de proyecciones                    │
│    - Proyección ahora disponible para USE                  │
└────────────────────────────────────────────────────────────┘
         |
         v
    RESULTADO: Proyección creada y lista para usar
```

### Diagrama de Flujo: USE

```
Usuario ejecuta:
USE "mi_proj" MATCH (n) WHERE n.age > 30 RETURN n.name
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 1. PARSER (query_visitor.cc::visitUseGraphClause)         │
│    - Reconoce USE cláusula                                 │
│    - Extrae nombre de proyección                           │
│    - Verifica que existe (ProjectionManager)               │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 2. CARGA DE CONTEXTO (query_context.cc::load_projection)  │
│    - Crea ProjectionQueryContext                           │
│    - Abre TODOS los índices B+Tree de la proyección        │
│    - Guarda en QueryContext::projection_ctx                │
│    - Establece QueryContext::active_projection = "mi_proj" │
└────────────────────────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────────────────────────┐
│ 3. EJECUCIÓN DEL MATCH                                     │
│    Cada vez que se necesita un índice:                     │
│    ┌──────────────────────────────────────────────┐       │
│    │ Llamada a GQLModel::get_node_key_value()     │       │
│    │      |                                        │       │
│    │      v                                        │       │
│    │ ¿ctx.is_using_projection()?                  │       │
│    │      |                                        │       │
│    │      ├─ NO  ──> Retornar índice grafo        │       │
│    │      │            principal (node_key_value)  │       │
│    │      │                                        │       │
│    │      └─ SÍ  ──> Retornar índice proyección   │       │
│    │                  (projection_ctx->            │       │
│    │                   node_key_value_index)       │       │
│    └──────────────────────────────────────────────┘       │
│                                                             │
│    - IndexScan usa índice de proyección                    │
│    - Filtros (WHERE) operan sobre datos de proyección      │
│    - RETURN accede a propiedades de proyección             │
└────────────────────────────────────────────────────────────┘
         |
         v
    RESULTADO: Datos consultados SOLO desde la proyección
```

### Clases Clave

| Clase | Archivo | Responsabilidad |
|-------|---------|-----------------|
| `AggProject` | `src/query/executor/binding_iter/aggregation/gql/agg_project.h` | Función de agregación que crea proyecciones |
| `ProjectionStorage` | `src/graph_models/gql/projection/projection_storage.{h,cc}` | Escribe datos a índices B+Tree de proyección |
| `ProjectionManager` | `src/graph_models/gql/projection/projection_manager.{h,cc}` | Gestiona catálogo de proyecciones |
| `ProjectionQueryContext` | `src/graph_models/gql/projection/projection_query_context.{h,cc}` | Mantiene índices abiertos durante consultas USE |
| `QueryContext` | `src/query/query_context.{h,cc}` | Mantiene estado de proyección activa |
| `GQLModel` | `src/graph_models/gql/gql_model.{h,cc}` | Enrutamiento dinámico de índices |

---

## Ejemplos Completos

### Ejemplo 1: Proyección Simple (Solo Estructura)

```bash
# 1. Crear proyección con solo nodos y aristas
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]->(m) RETURN PROJECT("estructura")'

# Resultado: Proyección "estructura" creada
# Contiene: Nodos, aristas, pero SIN etiquetas ni propiedades
```

**Estructura de archivos creados:**
```
data/dbs/gql/mi_db/projections/estructura/
├── catalog.dat
├── node_id.dir
├── node_id.leaf
├── from_to_edge.dir
├── from_to_edge.leaf
├── to_from_edge.dir
└── to_from_edge.leaf
```

**Uso:**
```bash
# Consultar cuántos nodos tiene
curl -X POST http://localhost:1234/gql --data \
  'USE "estructura" MATCH (n) RETURN count(n)'

# Consultar cuántas aristas
curl -X POST http://localhost:1234/gql --data \
  'USE "estructura" MATCH ()-[e]->() RETURN count(e)'

# ESTO FALLARÁ (no incluye etiquetas):
curl -X POST http://localhost:1234/gql --data \
  'USE "estructura" MATCH (n:User) RETURN count(n)'
# Error: Cannot use node labels with projection 'estructura'
```

### Ejemplo 2: Proyección con Etiquetas

```bash
# 1. Crear proyección con etiquetas
curl -X POST http://localhost:1234/gql --data \
  'MATCH (u)-[e]->(v) RETURN PROJECT("con_labels" INCLUDE LABELS)'
```

**Estructura de archivos creados:**
```
data/dbs/gql/mi_db/projections/con_labels/
├── catalog.dat
├── node_id.{dir,leaf}
├── from_to_edge.{dir,leaf}
├── to_from_edge.{dir,leaf}
├── node_label.{dir,leaf}      # NUEVO
├── label_node.{dir,leaf}      # NUEVO
├── edge_label.{dir,leaf}      # NUEVO
└── label_edge.{dir,leaf}      # NUEVO
```

**Uso:**
```bash
# Ahora SÍ podemos filtrar por etiquetas
curl -X POST http://localhost:1234/gql --data \
  'USE "con_labels" MATCH (n:User) RETURN count(n)'
# Funciona! ✅

curl -X POST http://localhost:1234/gql --data \
  'USE "con_labels" MATCH (n) RETURN labels(n) LIMIT 5'
# Retorna: [User, Admin], [User, Guest], etc. ✅

# PERO esto fallará (no incluye propiedades):
curl -X POST http://localhost:1234/gql --data \
  'USE "con_labels" MATCH (n) RETURN n.name LIMIT 5'
# Error: Cannot access node properties with projection 'con_labels'
```

### Ejemplo 3: Proyección Completa (Etiquetas + Propiedades)

```bash
# 1. Crear proyección con todo
curl -X POST http://localhost:1234/gql --data \
  'MATCH (u)-[e]->(v) RETURN PROJECT("completa" INCLUDE LABELS INCLUDE PROPERTIES)'
```

**Estructura de archivos creados:**
```
data/dbs/gql/mi_db/projections/completa/
├── catalog.dat
├── node_id.{dir,leaf}
├── from_to_edge.{dir,leaf}
├── to_from_edge.{dir,leaf}
├── node_label.{dir,leaf}
├── label_node.{dir,leaf}
├── edge_label.{dir,leaf}
├── label_edge.{dir,leaf}
├── node_key_value.{dir,leaf}     # NUEVO
├── key_value_node.{dir,leaf}     # NUEVO
├── edge_key_value.{dir,leaf}     # NUEVO
└── key_value_edge.{dir,leaf}     # NUEVO
```

**Uso:**
```bash
# Ahora TODO funciona:

# Contar por etiquetas
curl -X POST http://localhost:1234/gql --data \
  'USE "completa" MATCH (n:User) RETURN count(n)'
# ✅

# Acceder a propiedades
curl -X POST http://localhost:1234/gql --data \
  'USE "completa" MATCH (n) RETURN n.name, n.age LIMIT 5'
# ✅ Retorna: "Megan Chang", 59, etc.

# Filtrar por propiedades
curl -X POST http://localhost:1234/gql --data \
  'USE "completa" MATCH (n) WHERE n.age > 50 RETURN n.name, n.age'
# ✅

# Propiedades de aristas
curl -X POST http://localhost:1234/gql --data \
  'USE "completa" MATCH ()-[e]->() RETURN properties(e) LIMIT 3'
# ✅ Retorna: {via:"web"}, {via:"mobile"}, etc.
```

### Ejemplo 4: Proyección Filtrada (Subgrafo Específico)

```bash
# Crear proyección SOLO de usuarios adultos
curl -X POST http://localhost:1234/gql --data \
  'MATCH (u:User)-[e]->(v:User)
   WHERE u.age >= 18 AND v.age >= 18
   RETURN PROJECT("adultos" INCLUDE LABELS INCLUDE PROPERTIES)'

# Ahora "adultos" contiene SOLO las aristas entre usuarios >= 18 años

# Verificar:
curl -X POST http://localhost:1234/gql --data \
  'USE "adultos"
   MATCH (n)
   RETURN min(n.age), max(n.age), avg(n.age), count(n)'
# Resultado: min >= 18, porque filtramos en el MATCH original
```

### Ejemplo 5: Múltiples Proyecciones y Cambio de Contexto

```bash
# Crear varias proyecciones
curl -X POST http://localhost:1234/gql --data \
  'MATCH (u:User)-[e]->(v)
   RETURN PROJECT("usuarios_out" INCLUDE LABELS INCLUDE PROPERTIES)'

curl -X POST http://localhost:1234/gql --data \
  'MATCH (u)-[e]->(v:User)
   RETURN PROJECT("usuarios_in" INCLUDE LABELS INCLUDE PROPERTIES)'

curl -X POST http://localhost:1234/gql --data \
  'MATCH (a:Admin)-[e]->(v)
   RETURN PROJECT("admins" INCLUDE LABELS INCLUDE PROPERTIES)'

# Cambiar entre proyecciones:

# Consultar proyección 1
curl -X POST http://localhost:1234/gql --data \
  'USE "usuarios_out" MATCH (n) RETURN count(n)'

# Consultar proyección 2
curl -X POST http://localhost:1234/gql --data \
  'USE "usuarios_in" MATCH (n) RETURN count(n)'

# Consultar proyección 3
curl -X POST http://localhost:1234/gql --data \
  'USE "admins" MATCH (n) RETURN count(n)'

# Volver al grafo principal
curl -X POST http://localhost:1234/gql --data \
  'USE CURRENT_GRAPH MATCH (n) RETURN count(n)'
```

### Ejemplo 6: Comparación Grafo Principal vs Proyección

```bash
# Contar TODOS los nodos en grafo principal
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n) RETURN count(n)'
# Resultado: 100 nodos

# Crear proyección SOLO de nodos en aristas
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]-() RETURN PROJECT("conectados" INCLUDE PROPERTIES)'

# Contar nodos en proyección
curl -X POST http://localhost:1234/gql --data \
  'USE "conectados" MATCH (n) RETURN count(n)'
# Resultado: 91 nodos (solo los que tienen aristas)

# Comparación detallada:
echo "== GRAFO PRINCIPAL =="
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n) WHERE n.name IS NOT NULL RETURN count(n)'
# Resultado: 50 nodos con propiedad 'name'

echo "== PROYECCIÓN =="
curl -X POST http://localhost:1234/gql --data \
  'USE "conectados" MATCH (n) WHERE n.name IS NOT NULL RETURN count(n)'
# Resultado: 29 nodos (solo los que están en aristas Y tienen 'name')
```

### Ejemplo 7: Propiedades de Aristas

```bash
# Crear proyección con propiedades de aristas
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]->(m) RETURN PROJECT("edges_props" INCLUDE PROPERTIES)'

# Consultar propiedades de aristas
curl -X POST http://localhost:1234/gql --data \
  'USE "edges_props"
   MATCH ()-[e]->()
   RETURN e.via, count(e)
   GROUP BY e.via'
# Resultado:
# e.via, count(e)
# "web", 25
# "mobile", 30
# "desktop", 20

# Filtrar por propiedad de arista
curl -X POST http://localhost:1234/gql --data \
  'USE "edges_props"
   MATCH (n)-[e {via: "mobile"}]->(m)
   RETURN count(e)'
# Resultado: 30
```

---

## Limitaciones y Errores Comunes

### Error 1: Sintaxis Incorrecta (Comas)

❌ **INCORRECTO:**
```gql
MATCH (n)-[e]->(m)
RETURN PROJECT("test", INCLUDE LABELS, INCLUDE PROPERTIES)
```

**Error:** `mismatched input ','`

✅ **CORRECTO:**
```gql
MATCH (n)-[e]->(m)
RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)
```

**Razón:** Los parámetros opcionales NO se separan con comas.

### Error 2: Patrón MATCH Retorna 0 Resultados

❌ **PROBLEMA:**
```gql
# Si no hay aristas User->User en tu base de datos:
MATCH (u:User)-[e]->(v:User)
RETURN PROJECT("vacio" INCLUDE PROPERTIES)
```

**Resultado:** Proyección creada, pero VACÍA (0 nodos, 0 aristas, 0 propiedades)

✅ **SOLUCIÓN:**
```gql
# Primero verificar que el patrón retorna resultados:
MATCH (u:User)-[e]->(v:User) RETURN count(e)
# Si retorna 0, usar patrón más general:

MATCH (u)-[e]->(v) RETURN PROJECT("correcto" INCLUDE PROPERTIES)
```

### Error 3: Acceder a Funcionalidad No Incluida

❌ **PROBLEMA:**
```gql
# Crear proyección sin etiquetas
MATCH (n)-[e]->(m) RETURN PROJECT("sin_labels")

# Intentar usar etiquetas
USE "sin_labels" MATCH (n:User) RETURN count(n)
```

**Error:**
```
Cannot use node labels with projection 'sin_labels'.

Reason: This projection does not include node label information.

Solutions:
  1. Query the main graph instead:
     Remove the USE clause from your query

  2. Recreate projection with labels:
     MATCH ... RETURN PROJECT("sin_labels" INCLUDE LABELS)

  3. Switch to main graph temporarily:
     USE CURRENT_GRAPH MATCH ... RETURN ...
```

✅ **SOLUCIÓN:**
```gql
# Recrear con labels
MATCH (n)-[e]->(m) RETURN PROJECT("con_labels" INCLUDE LABELS)
USE "con_labels" MATCH (n:User) RETURN count(n)  # Ahora funciona
```

### Error 4: Proyección No Existe

❌ **PROBLEMA:**
```gql
USE "no_existe" MATCH (n) RETURN count(n)
```

**Error:**
```
Projection 'no_existe' does not exist.
Available projections: [estructura, con_labels, completa, adultos]
```

✅ **SOLUCIÓN:**
```bash
# Listar proyecciones disponibles primero
curl -X POST http://localhost:1234/gql --data 'SHOW PROJECTIONS'

# O usar una existente
USE "completa" MATCH (n) RETURN count(n)
```

### Error 5: Índices No Soportados en Proyecciones

❌ **PROBLEMA:**
```gql
USE "mi_proj" MATCH (n)-[e]-(n) RETURN count(e)  # Self-loops
```

**Error:**
```
Self-loop queries are not supported in projections.
Projection 'mi_proj' does not have specialized self-loop indexes.
Query the main graph with CURRENT_GRAPH for self-loop patterns.
```

✅ **SOLUCIÓN:**
```gql
# Volver al grafo principal para consultas no soportadas
USE CURRENT_GRAPH MATCH (n)-[e]-(n) RETURN count(e)
```

### Limitaciones Actuales

| Funcionalidad | Grafo Principal | Proyección | Notas |
|---------------|-----------------|------------|-------|
| Nodos y aristas básicos | ✅ | ✅ | Siempre incluidos |
| Etiquetas de nodos/aristas | ✅ | ⚠️ | Solo si INCLUDE LABELS |
| Propiedades de nodos/aristas | ✅ | ⚠️ | Solo si INCLUDE PROPERTIES |
| Self-loops (n)-[e]-(n) | ✅ | ❌ | No soportado |
| Índice edge_from_to | ✅ | ❌ | No soportado |
| Actualizar proyección | ❌ | ❌ | Debe recrearse |
| Borrar proyección | ⚠️ | ⚠️ | Manual (borrar directorio) |

---

## Resumen de Comandos Útiles

### Crear Proyecciones

```bash
# Básica
MATCH (n)-[e]->(m) RETURN PROJECT("nombre")

# Con etiquetas
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS)

# Con propiedades
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE PROPERTIES)

# Completa
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)

# Filtrada
MATCH (u:User)-[e]->(v:User) WHERE u.age > 18
RETURN PROJECT("adultos" INCLUDE LABELS INCLUDE PROPERTIES)
```

### Usar Proyecciones

```bash
# Cambiar a proyección
USE "nombre" MATCH (n) RETURN count(n)

# Volver al grafo principal
USE CURRENT_GRAPH MATCH (n) RETURN count(n)

# Consultas en proyección
USE "nombre"
MATCH (n:User) WHERE n.age > 30
RETURN n.name, n.age
ORDER BY n.age DESC
LIMIT 10
```

### Verificar Proyecciones

```bash
# Ver archivos creados
ls -lh data/dbs/gql/<db>/projections/<nombre>/

# Tamaño de índices
du -sh data/dbs/gql/<db>/projections/<nombre>/*

# Verificar contenido
USE "<nombre>" MATCH (n) RETURN count(n)
USE "<nombre>" MATCH ()-[e]->() RETURN count(e)
```

---

## Conclusión

El sistema de proyecciones en MillenniumDB proporciona:

1. **PROJECT()**: Crea subgrafos materializados con opciones configurables
2. **USE**: Cambia el contexto de consulta para usar proyecciones
3. **Enrutamiento dinámico**: El sistema automáticamente usa los índices correctos
4. **Errores descriptivos**: Mensajes claros cuando falta funcionalidad

Este sistema permite consultas eficientes en subgrafos específicos sin duplicar toda la base de datos.
