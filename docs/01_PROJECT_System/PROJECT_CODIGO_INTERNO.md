# Funcionamiento Interno de PROJECT y USE GRAPH - Código Detallado

## Flujo Completo con Código Real

### Ejemplo de Query

```gql
MATCH (u:User)-[e]->(v:User)
WHERE u.age >= 18
RETURN PROJECT("adultos" INCLUDE LABELS INCLUDE PROPERTIES)
```

---

## FASE 1: Parsing y Creación del Árbol de Ejecución

### Archivo: `src/query/parser/grammar/gql/query_visitor.cc`

```cpp
// Cuando el parser encuentra PROJECT(...) en el RETURN
std::any QueryVisitor::visitGqlProjectFunction(GQLParser::GqlProjectFunctionContext* ctx) {
    LOG_VISITOR
    std::cerr << "PROJECT function - visiting projectionName" << std::endl;

    // 1. Obtener el nombre de la proyección (ej: "adultos")
    auto projection_name_term = visit(ctx->projectionName()).as<std::unique_ptr<Expr>>();

    // 2. Procesar las opciones (INCLUDE LABELS, INCLUDE PROPERTIES)
    ProjectionOptions options;  // Por defecto: labels=false, properties=false

    if (ctx->projectionOptions()) {
        visitChildren(ctx->projectionOptions());
        // visitProjectionIncludeClause se ejecuta para cada INCLUDE
    }

    std::cerr << "PROJECT function - options: labels=" << options.include_labels
              << ", properties=" << options.include_properties << std::endl;

    // 3. Crear el nodo AggProject en el árbol de ejecución
    VarId project_var = get_query_ctx().get_internal_var();
    auto agg = std::make_unique<GQL::AggProject>(
        project_var,
        std::move(projection_name_term),
        options  // { include_labels=true, include_properties=true }
    );

    // Esto se convertirá en un operador Aggregation en el plan de ejecución
    return agg;
}

// Cuando encuentra "INCLUDE LABELS" o "INCLUDE PROPERTIES"
std::any QueryVisitor::visitProjectionIncludeClause(
    GQLParser::ProjectionIncludeClauseContext* ctx
) {
    if (ctx->LABELS()) {
        std::cerr << "Projection option: INCLUDE LABELS" << std::endl;
        current_projection_options.include_labels = true;
    }
    if (ctx->PROPERTIES()) {
        std::cerr << "Projection option: INCLUDE PROPERTIES" << std::endl;
        current_projection_options.include_properties = true;
    }
    return 0;
}
```

**Resultado de esta fase:**
```
Plan de ejecución:
  Aggregation(agg: PROJECT("adultos"))
    PathToBinding()
      LinearPatternPath(vars: ?u ?e ?v)
        IndexNestedLoopJoin()
          IndexScan(ranges: ?u ?v ?e)
          IndexScan(ranges: [?u] User)
          IndexScan(ranges: [?v] User)
```

---

## FASE 2: Ejecución del MATCH

El MATCH ejecuta y produce bindings (filas de resultados):

```
Binding 1: ?u=0xd400000000000000 (User id=0), ?e=0xe00000000000001b (edge), ?v=0xd400000000000002 (User id=2)
Binding 2: ?u=0xd400000000000000 (User id=0), ?e=0xe00000000000001c (edge), ?v=0xd400000000000004 (User id=4)
Binding 3: ?u=0xd400000000000002 (User id=2), ?e=0xe00000000000001d (edge), ?v=0xd400000000000006 (User id=6)
...
Total: 45 bindings (45 aristas entre usuarios >= 18)
```

---

## FASE 3: Inicialización de AggProject

### Archivo: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`

```cpp
void AggProject::initialize_if_needed() {
    if (initialized) {
        return;  // Ya inicializado
    }

    // 1. EVALUAR EL NOMBRE DE LA PROYECCIÓN
    // expr contiene ExprTerm("adultos"), se evalúa para obtener ObjectId
    ObjectId name_oid = expr->eval(*binding);
    // name_oid = 0x4100...  (STRING_SIMPLE tipo, contiene "adultos")

    // 2. EXTRAER LA CADENA
    auto type = GQL_OID::get_type(name_oid);
    if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
        type == GQL_OID::Type::STRING_SIMPLE_EXTERN ||
        type == GQL_OID::Type::STRING_SIMPLE_TMP) {
        projection_name = Conversions::unpack_string(name_oid);
        // projection_name = "adultos"
    }

    std::cerr << "AggProject: Creating projection '" << projection_name << "'" << std::endl;

    // 3. CREAR DIRECTORIO DE PROYECCIÓN
    auto& proj_manager = ProjectionManager::get_instance();
    std::string proj_dir = proj_manager.create_projection(projection_name);
    // proj_dir = "/data/dbs/gql/mi_db/projections/adultos"

    // create_projection hace:
    //   mkdir -p /data/dbs/gql/mi_db/projections/adultos
    //   return el path absoluto

    // 4. CONFIGURAR FEATURES
    ProjectionStorage::Features features;
    features.include_node_labels = options.include_labels;      // = true
    features.include_edge_labels = options.include_labels;      // = true
    features.include_node_properties = options.include_properties;  // = true
    features.include_edge_properties = options.include_properties;  // = true

    std::cerr << "AggProject: Features - labels: " << features.include_node_labels
              << ", properties: " << features.include_node_properties << std::endl;

    // 5. INICIALIZAR ALMACENAMIENTO
    projection_storage = std::make_unique<ProjectionStorage>(
        proj_dir,                    // "/data/.../adultos"
        proj_manager.get_db_folder(), // "/data/dbs/gql/mi_db"
        projection_name,             // "adultos"
        features                     // Configuración de índices
    );

    projection_storage->init();
    // init() crea todos los archivos B+Tree:
    //   - node_id.{dir,leaf}
    //   - from_to_edge.{dir,leaf}
    //   - to_from_edge.{dir,leaf}
    //   - node_label.{dir,leaf}  (si include_node_labels)
    //   - label_node.{dir,leaf}  (si include_node_labels)
    //   - node_key_value.{dir,leaf}  (si include_node_properties)
    //   - key_value_node.{dir,leaf}  (si include_node_properties)
    //   ... etc

    initialized = true;
}
```

**Estado después de inicialización:**
```
Archivos creados en disco:
/data/dbs/gql/mi_db/projections/adultos/
├── catalog.dat (0 bytes - solo header)
├── node_id.dir (4096 bytes - vacío)
├── node_id.leaf (4096 bytes - vacío)
├── from_to_edge.dir (4096 bytes - vacío)
├── from_to_edge.leaf (4096 bytes - vacío)
├── ... (todos los índices vacíos)

projection_storage apunta a estos archivos abiertos en modo escritura.
```

---

## FASE 4: Procesamiento de Cada Binding

### Archivo: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`

```cpp
void AggProject::process() override {
    initialize_if_needed();  // Ya ejecutado, no hace nada

    std::cerr << "[AggProject] process() called, binding size: " << binding->size << std::endl;
    // binding->size = número de variables en el MATCH (ej: 3 para ?u, ?e, ?v)

    // ESTRUCTURAS PARA RECOLECTAR DATOS
    std::map<ObjectId, std::unordered_map<std::string, ObjectId>> node_properties;
    std::map<ObjectId, std::unordered_map<std::string, ObjectId>> edge_properties;
    std::map<ObjectId, bool> edges_seen;  // edge_id -> is_directed
    std::map<ObjectId, std::pair<ObjectId, ObjectId>> edge_endpoints;  // edge -> (from, to)
    std::vector<ObjectId> node_sequence;  // Orden de nodos vistos

    // ============================================================
    // PASO 1: RECOLECTAR NODOS Y ARISTAS DEL BINDING ACTUAL
    // ============================================================
    for (size_t i = 0; i < binding->size; i++) {
        VarId var_id(i);
        ObjectId oid = (*binding)[var_id];  // Obtener valor de la variable

        if (!oid.is_valid()) {
            continue;  // Saltar NULLs
        }

        auto var_name = get_query_ctx().get_var_name(var_id);
        // var_name = "u" para i=0, "e" para i=1, "v" para i=2

        auto type = GQL_OID::get_type(oid);
        // type = NODE para ?u y ?v, DIRECTED_EDGE para ?e

        std::cerr << "[DEBUG] Var " << i << " (" << var_name << "): OID=0x"
                  << std::hex << oid.id << std::dec
                  << " type=" << static_cast<int>(type) << std::endl;

        // DETECTAR NODOS
        if (type == GQL_OID::Type::NODE) {
            node_sequence.push_back(oid);  // Para inferir endpoints de aristas

            // Inicializar mapa de propiedades (vacío por ahora)
            if (node_properties.find(oid) == node_properties.end()) {
                node_properties[oid] = std::unordered_map<std::string, ObjectId>();
            }
        }
        // DETECTAR ARISTAS
        else if (type == GQL_OID::Type::DIRECTED_EDGE ||
                 type == GQL_OID::Type::UNDIRECTED_EDGE) {

            bool is_directed = (type == GQL_OID::Type::DIRECTED_EDGE);
            edges_seen[oid] = is_directed;

            if (edge_properties.find(oid) == edge_properties.end()) {
                edge_properties[oid] = std::unordered_map<std::string, ObjectId>();
            }

            // INFERIR ENDPOINTS
            // Heurística: último nodo visto = from, siguiente nodo = to
            ObjectId from_node = ObjectId(ObjectId::NULL_ID);
            ObjectId to_node = ObjectId(ObjectId::NULL_ID);

            if (!node_sequence.empty()) {
                from_node = node_sequence.back();  // Último nodo visto

                // Buscar siguiente nodo
                for (size_t j = i + 1; j < binding->size; j++) {
                    VarId next_var_id(j);
                    ObjectId next_oid = (*binding)[next_var_id];
                    if (next_oid.is_valid()) {
                        auto next_type = GQL_OID::get_type(next_oid);
                        if (next_type == GQL_OID::Type::NODE) {
                            to_node = next_oid;
                            break;
                        }
                    }
                }
            }

            edge_endpoints[oid] = std::make_pair(from_node, to_node);
        }
    }

    std::cerr << "[AggProject] Collected: " << node_properties.size() << " nodes, "
              << edges_seen.size() << " edges" << std::endl;

    // ============================================================
    // PASO 2: AGREGAR NODOS A LA PROYECCIÓN
    // ============================================================
    for (const auto& [node_id, props] : node_properties) {
        ProjectedNode node;
        node.node_id = node_id;
        node.properties = props;  // Vacío por ahora, se llenará en paso 5

        projection_storage->add_node(node);
        // Esto escribe al índice node_id.{dir,leaf}

        std::cerr << "  Added node: 0x" << std::hex << node_id.id << std::dec << std::endl;
    }

    // ============================================================
    // PASO 3: AGREGAR ARISTAS A LA PROYECCIÓN
    // ============================================================
    std::cerr << "[AggProject] Adding " << edges_seen.size() << " edges" << std::endl;

    for (const auto& [edge_id, is_directed] : edges_seen) {
        ProjectedEdge edge;
        edge.edge_id = edge_id;
        edge.is_directed = is_directed;

        // Añadir propiedades si hay (vacío por ahora)
        auto props_it = edge_properties.find(edge_id);
        if (props_it != edge_properties.end()) {
            edge.properties = props_it->second;
        }

        // Usar endpoints inferidos
        auto endpoints_it = edge_endpoints.find(edge_id);
        if (endpoints_it != edge_endpoints.end()) {
            edge.from_node = endpoints_it->second.first;
            edge.to_node = endpoints_it->second.second;
        }

        std::cerr << "[AggProject] Calling add_edge for edge 0x" << std::hex << edge_id.id
                  << std::dec << " from=0x" << std::hex << edge.from_node.id
                  << " to=0x" << edge.to_node.id << std::dec << std::endl;

        projection_storage->add_edge(edge);
        // Esto escribe a:
        //   - from_to_edge.{dir,leaf}: {from, to, edge}
        //   - to_from_edge.{dir,leaf}: {to, from, edge}
    }

    std::cerr << "[AggProject] Finished adding nodes and edges" << std::endl;

    // ============================================================
    // PASO 4: EXTRAER ETIQUETAS DEL GRAFO PRINCIPAL
    // ============================================================
    if (options.include_labels) {
        std::cerr << "[AggProject] LABEL EXTRACTION STARTED" << std::endl;

        // ETIQUETAS DE NODOS
        for (const auto& [node_id, props] : node_properties) {
            bool interruption = false;

            // Escanear índice node_label del grafo principal
            // node_label es BPlusTree<2> con formato: {node_id, label_id}
            BptIter<2> it = gql_model.node_label->get_range(
                &interruption,
                { node_id.id, 0 },          // Rango desde (node_id, 0)
                { node_id.id, UINT64_MAX }  // Hasta (node_id, MAX)
            );
            // Esto retorna TODAS las etiquetas del nodo

            auto record = it.next();
            while (record != nullptr) {
                ObjectId label_id((*record)[1]);
                // (*record)[0] = node_id, (*record)[1] = label_id

                // Agregar a proyección
                projection_storage->add_node_label(node_id, label_id);
                // Escribe a:
                //   - node_label.{dir,leaf}: {node, label}
                //   - label_node.{dir,leaf}: {label, node}

                std::cerr << "    Added node label: node=0x" << std::hex << node_id.id
                          << " label=0x" << label_id.id << std::dec << std::endl;

                record = it.next();
            }
        }

        // ETIQUETAS DE ARISTAS (similar)
        for (const auto& [edge_id, is_directed] : edges_seen) {
            bool interruption = false;
            BptIter<2> it = gql_model.edge_label->get_range(
                &interruption,
                { edge_id.id, 0 },
                { edge_id.id, UINT64_MAX }
            );

            auto record = it.next();
            while (record != nullptr) {
                ObjectId label_id((*record)[1]);
                projection_storage->add_edge_label(edge_id, label_id);
                // Escribe a edge_label y label_edge

                std::cerr << "    Added edge label: edge=0x" << std::hex << edge_id.id
                          << " label=0x" << label_id.id << std::dec << std::endl;

                record = it.next();
            }
        }

        std::cerr << "[AggProject] LABEL EXTRACTION COMPLETE" << std::endl;
    }

    // ============================================================
    // PASO 5: EXTRAER PROPIEDADES DEL GRAFO PRINCIPAL
    // ============================================================
    if (options.include_properties) {
        std::cerr << "[AggProject] PROPERTY EXTRACTION STARTED" << std::endl;
        std::cerr << "[AggProject] Number of nodes to extract properties for: "
                  << node_properties.size() << std::endl;

        // PROPIEDADES DE NODOS
        for (const auto& [node_id, props] : node_properties) {
            std::cerr << "[AggProject] Extracting properties for node 0x"
                      << std::hex << node_id.id << std::dec << std::endl;

            bool interruption = false;

            // Escanear índice node_key_value del grafo principal
            // node_key_value es BPlusTree<3> con formato: {node_id, key_id, value_id}
            BptIter<3> it = gql_model.node_key_value->get_range(
                &interruption,
                { node_id.id, 0, 0 },                      // Desde (node, 0, 0)
                { node_id.id, UINT64_MAX, UINT64_MAX }     // Hasta (node, MAX, MAX)
            );
            // Esto retorna TODAS las propiedades del nodo

            auto record = it.next();
            int prop_count = 0;

            while (record != nullptr) {
                ObjectId key_id((*record)[1]);    // Nombre de propiedad (ej: "name")
                ObjectId value_id((*record)[2]);  // Valor (ej: "John")

                // Agregar a proyección (escritura bidireccional)
                projection_storage->add_node_property(node_id, key_id, value_id);
                // Escribe a:
                //   - node_key_value: {node, key, value}
                //   - key_value_node: {key, value, node}

                prop_count++;
                std::cerr << "[AggProject]     Added node property #" << prop_count
                          << ": node=0x" << std::hex << node_id.id
                          << " key=0x" << key_id.id
                          << " value=0x" << value_id.id << std::dec << std::endl;

                record = it.next();
            }

            std::cerr << "[AggProject]   Total properties for node 0x"
                      << std::hex << node_id.id << std::dec << ": " << prop_count << std::endl;
        }

        // PROPIEDADES DE ARISTAS (similar)
        std::cerr << "[AggProject] Number of edges to extract properties for: "
                  << edges_seen.size() << std::endl;

        for (const auto& [edge_id, is_directed] : edges_seen) {
            bool interruption = false;
            BptIter<3> it = gql_model.edge_key_value->get_range(
                &interruption,
                { edge_id.id, 0, 0 },
                { edge_id.id, UINT64_MAX, UINT64_MAX }
            );

            auto record = it.next();
            int edge_prop_count = 0;

            while (record != nullptr) {
                ObjectId key_id((*record)[1]);
                ObjectId value_id((*record)[2]);
                projection_storage->add_edge_property(edge_id, key_id, value_id);

                edge_prop_count++;
                std::cerr << "[AggProject]     Added edge property #" << edge_prop_count
                          << ": edge=0x" << std::hex << edge_id.id
                          << " key=0x" << key_id.id
                          << " value=0x" << value_id.id << std::dec << std::endl;

                record = it.next();
            }

            std::cerr << "[AggProject]   Total properties for edge 0x"
                      << std::hex << edge_id.id << std::dec << ": "
                      << edge_prop_count << std::endl;
        }

        std::cerr << "[AggProject] PROPERTY EXTRACTION COMPLETE" << std::endl;
    }

    std::cerr << "[AggProject] process() complete - nodes: " << node_properties.size()
              << ", edges: " << edges_seen.size() << std::endl;
}
```

**Este método `process()` se llama UNA VEZ POR CADA BINDING del MATCH.**

Si el MATCH retorna 45 bindings (45 aristas), `process()` se ejecuta 45 veces.

---

## FASE 5: Escritura a Disco (ProjectionStorage)

### Archivo: `src/graph_models/gql/projection/projection_storage.cc`

```cpp
void ProjectionStorage::add_node_property(ObjectId node_id,
                                          ObjectId key_id,
                                          ObjectId value_id) {
    // Verificar que el índice existe
    if (!node_key_value_index) {
        return;  // Índice no habilitado (INCLUDE PROPERTIES no se usó)
    }

    // ESCRITURA 1: Índice primario {node, key, value}
    // Útil para: "Dame todas las propiedades del nodo X"
    Record<3> node_prop_record;
    node_prop_record[0] = node_id.id;    // 0xd400000000000000
    node_prop_record[1] = key_id.id;     // 0xf000000000000000 (ej: "name")
    node_prop_record[2] = value_id.id;   // 0x4100000000000000 (ej: "John")

    node_key_value_index->insert(node_prop_record);
    // Escribe a node_key_value.{dir,leaf} en disco

    // ESCRITURA 2: Índice auxiliar {key, value, node}
    // Útil para: "Dame todos los nodos con name='John'"
    if (key_value_node_index) {
        Record<3> key_value_node_record;
        key_value_node_record[0] = key_id.id;
        key_value_node_record[1] = value_id.id;
        key_value_node_record[2] = node_id.id;

        key_value_node_index->insert(key_value_node_record);
        // Escribe a key_value_node.{dir,leaf} en disco
    }
}

void ProjectionStorage::add_edge(const ProjectedEdge& edge) {
    // ESCRITURA 1: Índice {from, to, edge}
    // Útil para: "Dame todas las aristas de A hacia B"
    Record<3> from_to_record;
    from_to_record[0] = edge.from_node.id;
    from_to_record[1] = edge.to_node.id;
    from_to_record[2] = edge.edge_id.id;

    from_to_edge_index->insert(from_to_record);

    // ESCRITURA 2: Índice inverso {to, from, edge}
    // Útil para: "Dame todas las aristas que llegan a B"
    Record<3> to_from_record;
    to_from_record[0] = edge.to_node.id;
    to_from_record[1] = edge.from_node.id;
    to_from_record[2] = edge.edge_id.id;

    to_from_edge_index->insert(to_from_record);
}
```

**Cada inserción escribe a buffers en memoria que eventualmente se flushean a disco.**

---

## FASE 6: Finalización

### Archivo: `src/query/executor/binding_iter/aggregation/gql/agg_project.h`

```cpp
ObjectId AggProject::get() override {
    initialize_if_needed();  // Por si acaso

    if (!projection_storage) {
        throw std::runtime_error("ProjectionStorage not initialized");
    }

    // 1. FLUSH: Escribir todos los buffers pendientes a disco
    projection_storage->flush();
    // Ahora TODOS los archivos .dir y .leaf tienen los datos completos

    std::cerr << "[AggProject] Flushed projection storage to disk" << std::endl;

    // 2. ACTUALIZAR CATÁLOGO
    // Hacer que la proyección sea visible inmediatamente
    auto& proj_manager = ProjectionManager::get_instance();
    proj_manager.scan_projections();
    // Esto actualiza el cache interno:
    //   projections["adultos"] = ProjectionInfo{ name, path, features }

    std::cerr << "[AggProject] Refreshed projection manager cache" << std::endl;

    // 3. RETORNAR NOMBRE COMO RESULTADO
    return projection_name_oid;  // ObjectId que representa "adultos"
    // Esto se mostrará al usuario como: "adultos"
}
```

**Estado final en disco:**
```
/data/dbs/gql/mi_db/projections/adultos/
├── catalog.dat (256 bytes - metadatos)
├── node_id.dir (4096 bytes - directorio B+Tree con 30 nodos)
├── node_id.leaf (8192 bytes - hojas B+Tree con 30 nodos)
├── from_to_edge.dir (4096 bytes - 45 aristas)
├── from_to_edge.leaf (12288 bytes - 45 aristas)
├── to_from_edge.dir (4096 bytes - 45 aristas inversos)
├── to_from_edge.leaf (12288 bytes)
├── node_label.dir (4096 bytes - ~60 etiquetas)
├── node_label.leaf (8192 bytes)
├── node_key_value.dir (8192 bytes - ~120 propiedades)
├── node_key_value.leaf (32768 bytes - 4 props/nodo × 30 nodos)
└── ... (otros índices)

Total: ~200 KB
```

---

## USO DE LA PROYECCIÓN CON USE

### Query:
```gql
USE "adultos"
MATCH (u) WHERE u.age > 50
RETURN u.name, u.age
```

### FASE 1: Parsing y Carga

```cpp
// Archivo: src/query/parser/grammar/gql/query_visitor.cc

std::any QueryVisitor::visitGraphExpression(
    GQLParser::GraphExpressionContext* ctx
) {
    if (ctx->objectNameOrBindingVariable()) {
        std::string projection_name = ctx->objectNameOrBindingVariable()->getText();
        // projection_name = "adultos"

        // 1. VERIFICAR QUE EXISTE
        auto& proj_manager = ProjectionManager::get_instance();
        if (!proj_manager.projection_exists(projection_name)) {
            // Listar proyecciones disponibles
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

        // 2. CARGAR PROYECCIÓN EN CONTEXTO
        get_query_ctx().load_projection(projection_name);
        // Esto ejecuta QueryContext::load_projection()

        return 0;
    }
}

// Archivo: src/query/query_context.cc

void QueryContext::load_projection(const std::string& proj_name) {
    std::cerr << "[QueryContext] load_projection() called for '" << proj_name << "'" << std::endl;

    // Guardar nombre activo
    active_projection = proj_name;  // "adultos"

    // Crear contexto de proyección (abre TODOS los índices)
    projection_ctx = std::make_unique<GQL::ProjectionQueryContext>(proj_name);
    // ProjectionQueryContext constructor:
    //   1. Abre /data/dbs/gql/mi_db/projections/adultos/catalog.dat
    //   2. Lee features: include_labels=true, include_properties=true
    //   3. Abre node_id.{dir,leaf} → projection_ctx->node_id_index
    //   4. Abre from_to_edge.{dir,leaf} → projection_ctx->from_to_edge_index
    //   5. Abre node_label.{dir,leaf} → projection_ctx->node_label_index (si habilitado)
    //   6. Abre node_key_value.{dir,leaf} → projection_ctx->node_key_value_index (si habilitado)
    //   ... etc

    std::cerr << "[QueryContext] Projection context created, valid="
              << projection_ctx->is_valid() << std::endl;
    // is_valid() verifica que todos los índices necesarios se abrieron correctamente
}
```

**Estado después de load_projection:**
```cpp
QueryContext {
    active_projection = "adultos"
    projection_ctx = ProjectionQueryContext {
        projection_name = "adultos"
        db_folder = "/data/dbs/gql/mi_db"
        projection_dir = "/data/dbs/gql/mi_db/projections/adultos"

        // Índices abiertos (punteros a BPlusTree):
        node_id_index = BPlusTree<1>("/data/.../adultos/node_id")
        from_to_edge_index = BPlusTree<3>("/data/.../adultos/from_to_edge")
        to_from_edge_index = BPlusTree<3>("/data/.../adultos/to_from_edge")
        node_label_index = BPlusTree<2>("/data/.../adultos/node_label")
        node_key_value_index = BPlusTree<3>("/data/.../adultos/node_key_value")
        // ... etc
    }
}
```

### FASE 2: Ejecución del MATCH con Enrutamiento Dinámico

```cpp
// Durante la ejecución de: MATCH (u) WHERE u.age > 50

// 1. IndexScan necesita escanear todos los nodos
//    Llama a: gql_model.get_node_id()  (NO EXISTE, se usa ObjectEnum)
//    Pero para propiedades llama a: gql_model.get_node_key_value()

// Archivo: src/graph_models/gql/gql_model.cc

BPlusTree<3>& GQLModel::get_node_key_value() {
    auto& ctx = get_query_ctx();

    // ¿ESTAMOS USANDO UNA PROYECCIÓN?
    if (ctx.is_using_projection()) {
        // ctx.active_projection = "adultos" → true

        // VERIFICAR QUE LA PROYECCIÓN TENGA PROPIEDADES
        if (!ctx.projection_ctx || !ctx.projection_ctx->node_key_value_index) {
            throw std::runtime_error(
                "Cannot access node properties with projection 'adultos'.\n\n"
                "Reason: This projection does not include node property information.\n\n"
                "Solutions:\n"
                "  1. Query the main graph instead:\n"
                "     Remove the USE clause from your query\n\n"
                "  2. Recreate projection with properties:\n"
                "     MATCH ... RETURN PROJECT(\"adultos\" INCLUDE PROPERTIES)\n\n"
                "  3. Switch to main graph temporarily:\n"
                "     USE CURRENT_GRAPH MATCH ... RETURN ..."
            );
        }

        // ¡USAR ÍNDICE DE LA PROYECCIÓN!
        std::cerr << "[GQLModel] Using projection node_key_value index" << std::endl;
        return *ctx.projection_ctx->node_key_value_index;
        // Retorna BPlusTree que apunta a:
        //   /data/.../adultos/node_key_value.{dir,leaf}
    }

    // Si no hay proyección activa, usar índice del grafo principal
    std::cerr << "[GQLModel] Using main graph node_key_value index" << std::endl;
    return *node_key_value;
    // Retorna BPlusTree que apunta a:
    //   /data/dbs/gql/mi_db/node_key_value.{dir,leaf}
}
```

**Resultado de enrutamiento:**
```
Query: WHERE u.age > 50

1. Obtener BPlusTree para propiedades
   → gql_model.get_node_key_value()
   → Detecta ctx.is_using_projection() = true
   → Retorna ctx.projection_ctx->node_key_value_index
   → Apunta a /data/.../adultos/node_key_value.{dir,leaf}

2. Para cada nodo en la proyección:
   → Escanear node_key_value buscando {node_id, "age", value}
   → Comparar value > 50
   → Si TRUE, incluir nodo en resultados

3. Para RETURN u.name:
   → Escanear node_key_value buscando {node_id, "name", value}
   → Retornar value

TODO esto opera SOLO sobre archivos de la proyección.
El grafo principal NUNCA se accede.
```

---

## Resumen del Flujo Completo

```
1. PARSING
   └─> Reconocer PROJECT("nombre" INCLUDE ...) → AggProject

2. EJECUCIÓN MATCH
   └─> Generar bindings (filas de resultados)

3. AGREGACIÓN (process() llamado por cada binding)
   ├─> Inicialización (una vez)
   │   ├─> Crear directorio
   │   ├─> Crear archivos B+Tree vacíos
   │   └─> Abrir índices para escritura
   │
   └─> Para cada binding:
       ├─> Recolectar nodos y aristas
       ├─> Escribir nodos a node_id
       ├─> Escribir aristas a from_to_edge y to_from_edge
       ├─> Si INCLUDE LABELS:
       │   ├─> Escanear node_label del grafo principal
       │   └─> Copiar a node_label de proyección
       └─> Si INCLUDE PROPERTIES:
           ├─> Escanear node_key_value del grafo principal
           └─> Copiar a node_key_value de proyección (bidireccional)

4. FINALIZACIÓN (get())
   ├─> flush() → Escribir buffers a disco
   ├─> scan_projections() → Actualizar catálogo
   └─> Retornar nombre de proyección

5. USO CON USE (consultas posteriores)
   ├─> load_projection() → Abrir TODOS los índices de proyección
   ├─> Durante ejecución:
   │   └─> gql_model.get_*() → Enrutamiento dinámico
   │       ├─> Si ctx.is_using_projection() → usar proyección
   │       └─> Si no → usar grafo principal
   └─> Resultados vienen 100% de archivos de proyección
```

---

## Archivos de Código Clave con Líneas Exactas

| Archivo | Líneas | Función |
|---------|--------|---------|
| `agg_project.h` | 49-121 | `initialize_if_needed()` - Crear directorio e índices |
| `agg_project.h` | 143-227 | `process()` - Paso 1: Recolectar nodos/aristas |
| `agg_project.h` | 229-238 | `process()` - Paso 2: Agregar nodos |
| `agg_project.h` | 240-268 | `process()` - Paso 3: Agregar aristas |
| `agg_project.h` | 270-319 | `process()` - Paso 4: Extraer etiquetas |
| `agg_project.h` | 321-373 | `process()` - Paso 5: Extraer propiedades |
| `agg_project.h` | 382-409 | `get()` - Flush y finalización |
| `projection_storage.cc` | 291-313 | `add_node_property()` - Escritura bidireccional |
| `projection_storage.cc` | 315-337 | `add_edge_property()` - Escritura bidireccional |
| `query_visitor.cc` | 2224-2290 | `visitGraphExpression()` - Parsing de USE |
| `query_context.cc` | 56-61 | `load_projection()` - Abrir índices |
| `gql_model.cc` | 53-67 | `get_from_to_edge()` - Enrutamiento dinámico |
| `gql_model.cc` | 164-183 | `get_node_key_value()` - Enrutamiento propiedades |
| `gql_model.cc` | 206-225 | `get_key_value_node()` - Índice auxiliar |

---

Espero que esta guía detallada te ayude a entender exactamente cómo funcionan PROJECT y USE GRAPH en el código! 🚀
