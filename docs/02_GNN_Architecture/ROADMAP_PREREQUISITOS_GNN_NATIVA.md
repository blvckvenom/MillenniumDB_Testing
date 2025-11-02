# Roadmap: Pre-requisitos para Arquitectura GNN Nativa en MillenniumDB
## Del Estado Actual a la Fase 0 de ARQUITECTURA_GNN_NATIVA.md

**Fecha**: 19 de Octubre, 2025
**Versión**: 1.0
**Objetivo**: Implementar los componentes mínimos necesarios para poder iniciar la Fase 0 del entrenamiento GNN nativo

---

## 📋 Tabla de Contenidos

1. [Análisis del Estado Actual](#1-análisis-del-estado-actual)
2. [Gap Analysis: ¿Qué Falta?](#2-gap-analysis-qué-falta)
3. [Pre-requisito 1: MUTATE PROPERTY](#3-pre-requisito-1-mutate-property)
4. [Pre-requisito 2: Sistema de Procedimientos GQL](#4-pre-requisito-2-sistema-de-procedimientos-gql)
5. [Pre-requisito 3: Soporte Maduro de TENSOR/VECTOR](#5-pre-requisito-3-soporte-maduro-de-tensorvector)
6. [Pre-requisito 4: Integración LibTorch](#6-pre-requisito-4-integración-libtorch)
7. [Roadmap de Ejecución](#7-roadmap-de-ejecución)

---

## 1. Análisis del Estado Actual

### 1.1 ✅ Componentes que YA EXISTEN y FUNCIONAN

| Componente | Estado | Ubicación | Notas |
|------------|--------|-----------|-------|
| **PROJECT básico** | ✅ COMPLETO | `src/query/executor/binding_iter/aggregation/gql/agg_project.h` | Crea proyecciones con labels y properties |
| **USE GRAPH** | ✅ COMPLETO | `src/graph_models/gql/projection/projection_query_context.h` | Context switching entre proyecciones |
| **Labels en proyecciones** | ✅ COMPLETO | `src/graph_models/gql/projection/projection_storage.cc` | node_label, edge_label B+Trees |
| **Properties en proyecciones** | ✅ COMPLETO | `src/graph_models/gql/projection/projection_storage.cc` | node_key_value, edge_key_value B+Trees |
| **Servidor HTTP** | ✅ EXISTE | `src/network/server/` | Puerto 1234, endpoints /gql |
| **Streaming básico** | ✅ EXISTE | `src/network/server/session/streaming/` | WebSocket y TCP streaming |
| **Parser GQL con CALL** | ✅ EXISTE | `src/query/parser/grammar/gql/GQLParser.g4` | Sintaxis CALL ya está en gramática |
| **TENSOR type (básico)** | ⚠️ PARCIAL | `src/graph_models/object_id.h:112-120` | TENSOR_FLOAT y TENSOR_DOUBLE existen pero necesitan maduración |

**Verificación práctica**:
```bash
cd /home/benito/B_MillenniumDB/MillenniumDB

# 1. Compilar
cmake --build build/Release -j4

# 2. Test de PROJECT + USE
build/Release/bin/mdb import data/example/gql/posts/posts.gql test_db
build/Release/bin/mdb server test_db &

curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]->(m) RETURN PROJECT("test_proj" INCLUDE LABELS INCLUDE PROPERTIES)'

curl -X POST http://localhost:1234/gql --data \
  'USE "test_proj" MATCH (n) RETURN n.name LIMIT 5'

# ✅ Si esto funciona, los componentes básicos están OK
```

---

### 1.2 ❌ Componentes que FALTAN (Bloqueantes)

| Componente | Impacto | Prioridad | Esfuerzo Estimado |
|------------|---------|-----------|-------------------|
| **MUTATE PROPERTY sintaxis** | BLOQUEANTE | CRÍTICA | 3-4 semanas |
| **Sistema de procedimientos registrables** | BLOQUEANTE | CRÍTICA | 2-3 semanas |
| **TensorStore maduro** | BLOQUEANTE | CRÍTICA | 3-4 semanas |
| **Integración LibTorch** | BLOQUEANTE | CRÍTICA | 2-3 semanas |
| **Cliente Python oficial** | ALTA | ALTA | 4-6 semanas |
| **API REST extendida** | ALTA | ALTA | 2-3 semanas |

**Total pre-requisitos críticos**: ~12-17 semanas (3-4 meses)

---

## 2. Gap Analysis: ¿Qué Falta?

### 2.1 Comparación: Estado Actual vs Fase 0 de ARQUITECTURA_GNN_NATIVA.md

De acuerdo al documento `ARQUITECTURA_GNN_NATIVA.md`, la **Fase 0** requiere:

#### Fase 0 (del documento ARQUITECTURA_GNN_NATIVA.md):
1. ✅ Instalación de LibTorch → **FALTA**: No integrado en build system
2. ❌ Tipo VECTOR en ObjectId → **PARCIAL**: TENSOR existe pero necesita TensorStore
3. ❌ MUTATE PROPERTY Syntax → **FALTA**: Gramática no implementada
4. ❌ Procedimientos GQL (Framework Base) → **FALTA**: No existe sistema de registro

**Conclusión**: De los 4 componentes de Fase 0, solo 0.5/4 están completos. **Necesitamos 3.5 componentes adicionales**.

---

### 2.2 Arquitectura GNN Nativa - Fases Completas

Del documento `ARQUITECTURA_GNN_NATIVA.md`:

```
Fase 0: Infraestructura Base           → 3-4 meses  (ESTE ROADMAP)
Fase 1: Motor GNN Core                 → 4-6 meses
Fase 2: Procedimientos GQL             → 2-3 meses
Fase 3: Optimizaciones + GPU           → 2-3 meses
Fase 4: Arquitecturas Avanzadas        → 3-4 meses
```

**Este roadmap se enfoca únicamente en completar los pre-requisitos para iniciar Fase 0.**

---

## 3. Pre-requisito 1: MUTATE PROPERTY

### 3.1 ¿Por qué es crítico?

**Caso de uso fundamental**:
```gql
-- Usuario ejecuta algoritmo (ej: PageRank) que genera scores
CALL mdb.pagerank() YIELD nodeId, score

-- Necesita GUARDAR esos scores como propiedades persistentes
MUTATE PROPERTY pagerank = score
```

Sin `MUTATE PROPERTY`, los resultados de algoritmos son volátiles (solo en memoria durante la query).

---

### 3.2 Estado Actual

**Gramática GQL**:
```bash
grep -r "MUTATE" src/query/parser/grammar/gql/
# Resultado: NINGUNA COINCIDENCIA
```

**Conclusión**: `MUTATE` NO existe en la gramática actual.

---

### 3.3 Diseño de Implementación

#### 3.3.1 Extender Gramática GQL

**Archivo**: `src/query/parser/grammar/gql/GQLLexer.g4`

```antlr
// Agregar nuevo keyword
MUTATE : 'MUTATE' ;
```

**Archivo**: `src/query/parser/grammar/gql/GQLParser.g4`

```antlr
// Nueva regla de sintaxis
mutatePropertyClause
    : MUTATE PROPERTY propertyAssignments
    ;

propertyAssignments
    : propertyAssignment (',' propertyAssignment)*
    ;

propertyAssignment
    : identifier '=' expr
    ;

// Integrar en query
returnItem
    : expr (AS identifier)?
    | ASTERISK
    | mutatePropertyClause   // NUEVO
    ;
```

**Sintaxis propuesta**:
```gql
-- Opción 1: Inline con YIELD
CALL mdb.pagerank() YIELD nodeId, score
MUTATE PROPERTY pagerank = score

-- Opción 2: Separado
MATCH (n)
WITH n, n.age * 2 AS double_age
MUTATE PROPERTY n.double_age = double_age

-- Opción 3: Batch update
MATCH (n:User)
WHERE n.age > 30
MUTATE PROPERTY n.is_adult = true, n.category = "senior"
```

---

#### 3.3.2 Implementar Binding Iterator

**Archivo nuevo**: `src/query/executor/binding_iter/mutate_property.h`

```cpp
#pragma once

#include "query/executor/binding_iter.h"
#include "graph_models/gql/gql_model.h"
#include <vector>
#include <string>

namespace GQL {

class MutateProperty : public BindingIter {
public:
    struct PropertyMutation {
        std::string property_name;
        VarId value_var;  // Variable que contiene el valor a mutar
        VarId entity_var;  // Variable que contiene el nodo/arista a mutar
        bool is_node;  // true para nodos, false para aristas
    };

    MutateProperty(
        std::unique_ptr<BindingIter> child,
        std::vector<PropertyMutation> mutations
    );

    void accept_visitor(BindingIterVisitor& visitor) override;
    void _begin(Binding& parent_binding) override;
    bool _next() override;
    void _reset() override;

private:
    std::unique_ptr<BindingIter> child;
    std::vector<PropertyMutation> mutations;

    // Buffer para escritura batch
    struct WriteBuffer {
        std::vector<Record<3>> primary_records;
        std::vector<Record<3>> auxiliary_records;
    };

    std::unordered_map<std::string, WriteBuffer> write_buffers;

    void flush_buffers();
    void write_to_btree(const PropertyMutation& mutation, ObjectId entity_id, ObjectId value);
};

} // namespace GQL
```

**Implementación**: `src/query/executor/binding_iter/mutate_property.cc`

```cpp
#include "mutate_property.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/conversions.h"

namespace GQL {

MutateProperty::MutateProperty(
    std::unique_ptr<BindingIter> child,
    std::vector<PropertyMutation> mutations
) : child(std::move(child)), mutations(std::move(mutations)) {}

void MutateProperty::_begin(Binding& parent_binding) {
    this->parent_binding = &parent_binding;
    child->begin(parent_binding);
}

bool MutateProperty::_next() {
    // Obtener siguiente binding del child
    if (!child->next()) {
        // No hay más bindings, flush buffers y retornar false
        flush_buffers();
        return false;
    }

    // Para cada mutación, extraer valor y entidad, agregar a buffer
    for (const auto& mutation : mutations) {
        ObjectId entity_id = parent_binding->get(mutation.entity_var);
        ObjectId value = parent_binding->get(mutation.value_var);

        write_to_btree(mutation, entity_id, value);
    }

    // Retornar binding original (sin modificar)
    return true;
}

void MutateProperty::write_to_btree(
    const PropertyMutation& mutation,
    ObjectId entity_id,
    ObjectId value
) {
    auto& gql_model = get_query_ctx().get_gql_model();

    // Obtener índices apropiados (proyección o grafo principal)
    BPlusTree<3>* primary_index;
    BPlusTree<3>* aux_index;

    if (mutation.is_node) {
        primary_index = &gql_model.get_node_key_value();
        aux_index = &gql_model.get_key_value_node();
    } else {
        primary_index = &gql_model.get_edge_key_value();
        aux_index = &gql_model.get_key_value_edge();
    }

    // Convertir nombre de propiedad a ObjectId
    ObjectId property_name_oid = Conversions::pack_string(mutation.property_name);

    // Crear registros
    Record<3> primary_record;
    primary_record[0] = entity_id.id;
    primary_record[1] = property_name_oid.id;
    primary_record[2] = value.id;

    Record<3> aux_record;
    aux_record[0] = property_name_oid.id;
    aux_record[1] = value.id;
    aux_record[2] = entity_id.id;

    // Agregar a buffer (batch insert para eficiencia)
    auto& buffer = write_buffers[mutation.property_name];
    buffer.primary_records.push_back(primary_record);
    buffer.auxiliary_records.push_back(aux_record);

    // Flush si buffer es grande (ej: cada 1000 records)
    if (buffer.primary_records.size() >= 1000) {
        // Escribir batch
        for (const auto& rec : buffer.primary_records) {
            primary_index->insert(rec);
        }
        for (const auto& rec : buffer.auxiliary_records) {
            aux_index->insert(rec);
        }

        // Limpiar buffer
        buffer.primary_records.clear();
        buffer.auxiliary_records.clear();
    }
}

void MutateProperty::flush_buffers() {
    auto& gql_model = get_query_ctx().get_gql_model();

    for (const auto& [prop_name, buffer] : write_buffers) {
        // Determinar si es node o edge property
        bool is_node = mutations[0].is_node;  // Asumir todos del mismo tipo

        BPlusTree<3>* primary_index;
        BPlusTree<3>* aux_index;

        if (is_node) {
            primary_index = &gql_model.get_node_key_value();
            aux_index = &gql_model.get_key_value_node();
        } else {
            primary_index = &gql_model.get_edge_key_value();
            aux_index = &gql_model.get_key_value_edge();
        }

        // Escribir todos los records restantes
        for (const auto& rec : buffer.primary_records) {
            primary_index->insert(rec);
        }
        for (const auto& rec : buffer.auxiliary_records) {
            aux_index->insert(rec);
        }
    }

    write_buffers.clear();
}

void MutateProperty::_reset() {
    child->reset();
    write_buffers.clear();
}

} // namespace GQL
```

---

#### 3.3.3 Integrar en Query Visitor

**Archivo**: `src/query/parser/grammar/gql/query_visitor.cc`

```cpp
// Agregar método para visitar mutatePropertyClause
std::any GQLQueryVisitor::visitMutatePropertyClause(
    GQLParser::MutatePropertyClauseContext* ctx
) {
    std::vector<MutateProperty::PropertyMutation> mutations;

    for (auto* assignment : ctx->propertyAssignments()->propertyAssignment()) {
        MutateProperty::PropertyMutation mutation;
        mutation.property_name = assignment->identifier()->getText();

        // Parsear expresión del valor
        auto value_expr = std::any_cast<std::unique_ptr<Expr>>(
            visit(assignment->expr())
        );

        // Extraer variable del valor (simplificado)
        mutation.value_var = extract_var_from_expr(value_expr.get());

        // Determinar entidad (nodo o arista)
        // Esto depende del contexto (MATCH previo)
        mutation.entity_var = current_entity_var;
        mutation.is_node = current_entity_is_node;

        mutations.push_back(mutation);
    }

    // Crear binding iter
    auto mutate_iter = std::make_unique<MutateProperty>(
        std::move(current_plan),
        std::move(mutations)
    );

    return mutate_iter;
}
```

---

### 3.4 Tests de MUTATE PROPERTY

**Test 1: Mutate simple property**:
```bash
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n:User)
WHERE n.age > 30
MUTATE PROPERTY n.is_senior = true
'

# Verificar
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n:User)
WHERE n.is_senior = true
RETURN count(n)
'
```

**Test 2: Mutate con resultado de CALL**:
```bash
# (Requiere implementar mdb.pagerank primero, ver Pre-requisito 2)
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
CALL mdb.pagerank() YIELD nodeId, score
MUTATE PROPERTY pagerank = score
'

# Verificar
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n)
RETURN n.name, n.pagerank
ORDER BY n.pagerank DESC
LIMIT 10
'
```

---

### 3.5 Tareas y Estimación

| Tarea | Archivo(s) | Líneas | Esfuerzo |
|-------|-----------|--------|----------|
| Extender gramática ANTLR | `GQLLexer.g4`, `GQLParser.g4` | ~30 | 2 días |
| Implementar MutateProperty iter | `mutate_property.h/cc` | ~250 | 1 semana |
| Integrar en query visitor | `query_visitor.cc` | ~100 | 3 días |
| Tests unitarios | `tests/gql/mutate/` | ~200 | 3 días |
| Tests de integración | Scripts bash | ~100 | 2 días |

**Total Pre-requisito 1**: ~680 líneas, **3-4 semanas**

---

## 4. Pre-requisito 2: Sistema de Procedimientos GQL

### 4.1 ¿Por qué es crítico?

Para que la arquitectura GNN nativa funcione, necesitamos procedimientos como:
```gql
CALL mdb.gnn.define("my_model", {...})
CALL mdb.gnn.train("my_model")
CALL mdb.gnn.predict("my_model") YIELD nodeId, prediction
```

**Estado actual**:
- ✅ Gramática GQL tiene sintaxis `CALL`
- ❌ NO existe sistema de **registro** de procedimientos
- ❌ NO existe clase base `Procedure`

---

### 4.2 Diseño del Sistema de Procedimientos

#### 4.2.1 Clase Base Abstracta

**Archivo nuevo**: `src/query/executor/binding_iter/procedure.h`

```cpp
#pragma once

#include "query/executor/binding_iter.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GQL {

// Clase base para todos los procedimientos
class Procedure : public BindingIter {
public:
    virtual ~Procedure() = default;

    // Nombre del procedimiento (ej: "mdb.pagerank")
    virtual std::string name() const = 0;

    // Columnas que produce el procedimiento (YIELD)
    virtual std::vector<std::string> output_columns() const = 0;

    // Configurar parámetros del procedimiento
    virtual void configure(const nlohmann::json& params) = 0;

    // Validar que los parámetros son correctos
    virtual void validate() = 0;

    // Ejecutar procedimiento (llamado en _begin)
    virtual void execute() = 0;

    // BindingIter interface
    void _begin(Binding& parent_binding) override {
        this->parent_binding = &parent_binding;
        validate();
        execute();
        result_index = 0;
    }

    bool _next() override {
        if (result_index >= results.size()) {
            return false;
        }

        // Llenar binding con resultado actual
        const auto& result_row = results[result_index++];
        for (size_t i = 0; i < output_vars.size(); i++) {
            parent_binding->add(output_vars[i], result_row[i]);
        }

        return true;
    }

    void _reset() override {
        result_index = 0;
    }

protected:
    // Resultados del procedimiento
    std::vector<std::vector<ObjectId>> results;
    std::vector<VarId> output_vars;
    size_t result_index = 0;

    // Helper para agregar fila de resultado
    void add_result(const std::vector<ObjectId>& row) {
        results.push_back(row);
    }
};

} // namespace GQL
```

---

#### 4.2.2 Registro de Procedimientos

**Archivo nuevo**: `src/query/executor/binding_iter/procedure_registry.h`

```cpp
#pragma once

#include "procedure.h"
#include <memory>
#include <unordered_map>
#include <functional>

namespace GQL {

class ProcedureRegistry {
public:
    using ProcedureFactory = std::function<std::unique_ptr<Procedure>()>;

    // Singleton
    static ProcedureRegistry& instance() {
        static ProcedureRegistry registry;
        return registry;
    }

    // Registrar procedimiento
    void register_procedure(const std::string& name, ProcedureFactory factory) {
        procedures[name] = factory;
    }

    // Crear instancia de procedimiento
    std::unique_ptr<Procedure> create(const std::string& name) {
        auto it = procedures.find(name);
        if (it == procedures.end()) {
            throw std::runtime_error("Procedure not found: " + name);
        }
        return it->second();
    }

    // Listar procedimientos disponibles
    std::vector<std::string> list_procedures() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : procedures) {
            names.push_back(name);
        }
        return names;
    }

private:
    std::unordered_map<std::string, ProcedureFactory> procedures;
};

// Helper macro para auto-registro
#define REGISTER_PROCEDURE(ClassName, ProcName) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                ProcedureRegistry::instance().register_procedure( \
                    ProcName, \
                    []() -> std::unique_ptr<Procedure> { \
                        return std::make_unique<ClassName>(); \
                    } \
                ); \
            } \
        }; \
        static ClassName##Registrar global_##ClassName##_registrar; \
    }

} // namespace GQL
```

---

#### 4.2.3 Ejemplo: Procedimiento Echo (Test)

**Archivo nuevo**: `src/query/executor/binding_iter/procedures/echo.h`

```cpp
#pragma once

#include "query/executor/binding_iter/procedure.h"
#include "query/executor/binding_iter/procedure_registry.h"

namespace GQL {

class EchoProcedure : public Procedure {
public:
    std::string name() const override {
        return "mdb.echo";
    }

    std::vector<std::string> output_columns() const override {
        return {"message"};
    }

    void configure(const nlohmann::json& params) override {
        if (params.contains("message")) {
            message = params["message"].get<std::string>();
        } else {
            message = "Hello from MillenniumDB!";
        }
    }

    void validate() override {
        // Ninguna validación necesaria para echo
    }

    void execute() override {
        // Simplemente retornar el mensaje
        ObjectId message_oid = Conversions::pack_string(message);
        add_result({message_oid});
    }

private:
    std::string message;
};

// Auto-registrar
REGISTER_PROCEDURE(EchoProcedure, "mdb.echo");

} // namespace GQL
```

**Uso**:
```gql
CALL mdb.echo({message: "Hello World"})
YIELD message
RETURN message
```

---

#### 4.2.4 Ejemplo: Procedimiento PageRank (Placeholder)

**Archivo nuevo**: `src/query/executor/binding_iter/procedures/pagerank.h`

```cpp
#pragma once

#include "query/executor/binding_iter/procedure.h"
#include "query/executor/binding_iter/procedure_registry.h"
#include "graph_models/gql/gql_model.h"

namespace GQL {

class PageRankProcedure : public Procedure {
public:
    std::string name() const override {
        return "mdb.pagerank";
    }

    std::vector<std::string> output_columns() const override {
        return {"nodeId", "score"};
    }

    void configure(const nlohmann::json& params) override {
        damping_factor = params.value("dampingFactor", 0.85);
        max_iterations = params.value("maxIterations", 20);
        tolerance = params.value("tolerance", 0.0001);
    }

    void validate() override {
        if (damping_factor < 0.0 || damping_factor > 1.0) {
            throw std::runtime_error("dampingFactor must be in [0, 1]");
        }
        if (max_iterations <= 0) {
            throw std::runtime_error("maxIterations must be > 0");
        }
    }

    void execute() override {
        auto& gql_model = get_query_ctx().get_gql_model();

        // TODO: Implementar algoritmo PageRank real (Fase 2)
        // Por ahora, placeholder con scores dummy

        // Iterar sobre todos los nodos
        auto& node_index = gql_model.get_node_id_index();
        bool interruption = false;
        auto it = node_index.get_range(
            &interruption,
            Record<1>{0},
            Record<1>{UINT64_MAX}
        );

        auto record = it.next();
        while (record != nullptr) {
            ObjectId node_id((*record)[0]);

            // Score dummy (0.5 para todos)
            double score = 0.5;  // TODO: Calcular PageRank real
            ObjectId score_oid = Conversions::pack_double(score);

            add_result({node_id, score_oid});

            record = it.next();
        }
    }

private:
    double damping_factor;
    int max_iterations;
    double tolerance;
};

// Auto-registrar
REGISTER_PROCEDURE(PageRankProcedure, "mdb.pagerank");

} // namespace GQL
```

**Uso**:
```gql
CALL mdb.pagerank({dampingFactor: 0.85, maxIterations: 30})
YIELD nodeId, score
MUTATE PROPERTY pagerank = score
```

---

### 4.3 Integración con Parser

**Archivo**: `src/query/parser/grammar/gql/query_visitor.cc`

```cpp
std::any GQLQueryVisitor::visitProcedureCall(
    GQLParser::ProcedureCallContext* ctx
) {
    // Extraer nombre del procedimiento
    std::string proc_name = ctx->procedureName()->getText();

    // Parsear parámetros (JSON)
    nlohmann::json params;
    if (ctx->procedureArguments()) {
        params = parse_json_from_gql(ctx->procedureArguments());
    }

    // Crear instancia del procedimiento
    auto proc = ProcedureRegistry::instance().create(proc_name);

    // Configurar parámetros
    proc->configure(params);

    // Extraer variables de YIELD
    std::vector<VarId> output_vars;
    if (ctx->yieldClause()) {
        for (auto* yield_item : ctx->yieldClause()->yieldItem()) {
            VarId var = get_query_ctx().get_or_create_var(
                yield_item->getText()
            );
            output_vars.push_back(var);
        }
    }

    proc->output_vars = output_vars;

    return proc;
}
```

---

### 4.4 Tests de Procedimientos

**Test 1: Echo procedure**:
```bash
curl -X POST http://localhost:1234/gql --data '
CALL mdb.echo({message: "Test"})
YIELD message
RETURN message
'

# Resultado esperado:
# {"head": ["message"], "results": [["Test"]]}
```

**Test 2: PageRank procedure**:
```bash
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
CALL mdb.pagerank({dampingFactor: 0.85})
YIELD nodeId, score
RETURN nodeId, score
LIMIT 10
'

# Resultado esperado:
# {"head": ["nodeId", "score"], "results": [[123, 0.5], [456, 0.5], ...]}
```

**Test 3: PageRank + MUTATE**:
```bash
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
CALL mdb.pagerank()
YIELD nodeId, score
MUTATE PROPERTY pagerank = score
'

# Verificar
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n)
RETURN n.pagerank
LIMIT 5
'
```

---

### 4.5 Tareas y Estimación

| Tarea | Archivo(s) | Líneas | Esfuerzo |
|-------|-----------|--------|----------|
| Clase base Procedure | `procedure.h` | ~100 | 2 días |
| Registro de procedimientos | `procedure_registry.h` | ~80 | 2 días |
| Procedimiento Echo (test) | `procedures/echo.h` | ~50 | 1 día |
| Procedimiento PageRank (placeholder) | `procedures/pagerank.h` | ~120 | 3 días |
| Integrar en parser | `query_visitor.cc` | ~80 | 3 días |
| Tests | `tests/gql/procedures/` | ~150 | 2 días |

**Total Pre-requisito 2**: ~580 líneas, **2-3 semanas**

---

## 5. Pre-requisito 3: Soporte Maduro de TENSOR/VECTOR

### 5.1 Estado Actual

**ObjectId ya tiene tipos TENSOR**:
```cpp
// src/graph_models/object_id.h:112-120
static constexpr uint64_t MASK_TENSOR                  = 0xB0'00000000000000UL;
static constexpr uint64_t MASK_TENSOR_FLOAT            = 0xB0'00000000000000UL;
static constexpr uint64_t MASK_TENSOR_FLOAT_INLINED    = 0xB0'00000000000000UL;
static constexpr uint64_t MASK_TENSOR_FLOAT_EXTERN     = 0xB1'00000000000000UL;
static constexpr uint64_t MASK_TENSOR_FLOAT_TMP        = 0xB2'00000000000000UL;
static constexpr uint64_t MASK_TENSOR_DOUBLE           = 0xB4'00000000000000UL;
```

**Problema**: No existe `TensorStore` para almacenar tensores externos.

---

### 5.2 Diseño de TensorStore

**Archivo nuevo**: `src/storage/tensor_store.h`

```cpp
#pragma once

#include <vector>
#include <cstdint>
#include <fstream>
#include <unordered_map>

namespace Storage {

class TensorStore {
public:
    enum class DataType {
        FLOAT32,
        FLOAT64
    };

    struct TensorEntry {
        DataType dtype;
        std::vector<uint32_t> shape;  // Dimensiones [batch, features, ...]
        std::vector<uint8_t> data;    // Raw bytes
    };

    TensorStore(const std::string& base_path);
    ~TensorStore();

    // Insertar tensor, retorna ID
    uint64_t insert(const TensorEntry& tensor);

    // Recuperar tensor por ID
    TensorEntry get(uint64_t tensor_id) const;

    // Flush a disco
    void flush();

    // Cerrar archivo
    void close();

private:
    std::string file_path;
    std::fstream file;

    // Cache LRU (opcional)
    std::unordered_map<uint64_t, TensorEntry> cache;
    uint64_t next_id = 1;

    // Índice de offsets (para lectura rápida)
    std::unordered_map<uint64_t, uint64_t> offset_index;

    void rebuild_index();
};

} // namespace Storage
```

**Implementación**: `src/storage/tensor_store.cc`

```cpp
#include "tensor_store.h"
#include <stdexcept>

namespace Storage {

TensorStore::TensorStore(const std::string& base_path)
    : file_path(base_path + "/tensor_store.bin")
{
    // Abrir o crear archivo
    file.open(file_path, std::ios::in | std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        // Crear archivo nuevo
        file.open(file_path, std::ios::out | std::ios::binary);
        file.close();
        file.open(file_path, std::ios::in | std::ios::out | std::ios::binary);
    }

    // Construir índice de offsets
    rebuild_index();
}

TensorStore::~TensorStore() {
    close();
}

uint64_t TensorStore::insert(const TensorEntry& tensor) {
    uint64_t tensor_id = next_id++;

    // Guardar en cache
    cache[tensor_id] = tensor;

    // Seek to end
    file.seekp(0, std::ios::end);
    uint64_t offset = file.tellp();

    // Guardar offset en índice
    offset_index[tensor_id] = offset;

    // Formato en disco:
    // [uint64_t: tensor_id]
    // [uint8_t: dtype]
    // [uint32_t: num_dimensions]
    // [uint32_t * num_dimensions: shape]
    // [uint64_t: data_size_bytes]
    // [uint8_t * data_size_bytes: data]

    file.write(reinterpret_cast<const char*>(&tensor_id), sizeof(uint64_t));

    uint8_t dtype = static_cast<uint8_t>(tensor.dtype);
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(uint8_t));

    uint32_t num_dims = tensor.shape.size();
    file.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint32_t));

    for (uint32_t dim : tensor.shape) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    }

    uint64_t data_size = tensor.data.size();
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(tensor.data.data()), data_size);

    return tensor_id;
}

TensorStore::TensorEntry TensorStore::get(uint64_t tensor_id) const {
    // Buscar en cache
    auto cache_it = cache.find(tensor_id);
    if (cache_it != cache.end()) {
        return cache_it->second;
    }

    // Buscar en índice
    auto offset_it = offset_index.find(tensor_id);
    if (offset_it == offset_index.end()) {
        throw std::runtime_error("Tensor ID not found: " + std::to_string(tensor_id));
    }

    // Seek a offset
    file.seekg(offset_it->second, std::ios::beg);

    // Leer tensor
    uint64_t stored_id;
    file.read(reinterpret_cast<char*>(&stored_id), sizeof(uint64_t));

    if (stored_id != tensor_id) {
        throw std::runtime_error("Tensor ID mismatch");
    }

    TensorEntry entry;

    uint8_t dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(uint8_t));
    entry.dtype = static_cast<DataType>(dtype);

    uint32_t num_dims;
    file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
    entry.shape.resize(num_dims);

    for (uint32_t i = 0; i < num_dims; i++) {
        file.read(reinterpret_cast<char*>(&entry.shape[i]), sizeof(uint32_t));
    }

    uint64_t data_size;
    file.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
    entry.data.resize(data_size);
    file.read(reinterpret_cast<char*>(entry.data.data()), data_size);

    return entry;
}

void TensorStore::flush() {
    file.flush();
}

void TensorStore::close() {
    flush();
    file.close();
}

void TensorStore::rebuild_index() {
    file.seekg(0, std::ios::beg);

    while (!file.eof()) {
        uint64_t offset = file.tellg();

        uint64_t tensor_id;
        file.read(reinterpret_cast<char*>(&tensor_id), sizeof(uint64_t));

        if (file.eof()) break;

        offset_index[tensor_id] = offset;

        if (tensor_id >= next_id) {
            next_id = tensor_id + 1;
        }

        // Skip tensor data
        uint8_t dtype;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(uint8_t));

        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
        file.seekg(num_dims * sizeof(uint32_t), std::ios::cur);

        uint64_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
        file.seekg(data_size, std::ios::cur);
    }

    file.clear();  // Clear EOF flag
}

} // namespace Storage
```

---

### 5.3 Integrar TensorStore en GQLModel

**Archivo**: `src/graph_models/gql/gql_model.h`

```cpp
#include "storage/tensor_store.h"

class GQLModel {
public:
    // ... miembros existentes ...

    std::unique_ptr<Storage::TensorStore> tensor_store;

    void init(const std::string& db_folder) {
        // ... init existente ...

        // Inicializar tensor store
        tensor_store = std::make_unique<Storage::TensorStore>(db_folder);
    }
};
```

**Archivo**: `src/graph_models/gql/projection/projection_storage.h`

```cpp
class ProjectionStorage {
public:
    // ... miembros existentes ...

    std::unique_ptr<Storage::TensorStore> tensor_store;

    void init(const std::string& projection_dir) {
        // ... init existente ...

        tensor_store = std::make_unique<Storage::TensorStore>(projection_dir);
    }
};
```

---

### 5.4 Conversiones para ObjectId

**Archivo**: `src/graph_models/conversions.h`

```cpp
// Agregar métodos para tensores
class Conversions {
public:
    // Pack tensor (1D array de floats) en ObjectId
    static ObjectId pack_tensor_float(
        const std::vector<float>& data,
        Storage::TensorStore& store
    ) {
        Storage::TensorStore::TensorEntry entry;
        entry.dtype = Storage::TensorStore::DataType::FLOAT32;
        entry.shape = {static_cast<uint32_t>(data.size())};  // 1D

        // Convertir floats a bytes
        entry.data.resize(data.size() * sizeof(float));
        std::memcpy(entry.data.data(), data.data(), entry.data.size());

        uint64_t tensor_id = store.insert(entry);

        // Crear ObjectId con tipo TENSOR_FLOAT_EXTERN
        ObjectId oid;
        oid.id = ObjectId::MASK_TENSOR_FLOAT_EXTERN | (tensor_id & 0x00FFFFFFFFFFFFFF);

        return oid;
    }

    // Unpack tensor desde ObjectId
    static std::vector<float> unpack_tensor_float(
        ObjectId oid,
        const Storage::TensorStore& store
    ) {
        if ((oid.id & ObjectId::SUB_TYPE_MASK) != ObjectId::MASK_TENSOR_FLOAT) {
            throw std::runtime_error("ObjectId is not a float tensor");
        }

        uint64_t tensor_id = oid.id & 0x00FFFFFFFFFFFFFF;
        auto entry = store.get(tensor_id);

        // Convertir bytes a floats
        std::vector<float> data(entry.data.size() / sizeof(float));
        std::memcpy(data.data(), entry.data.data(), entry.data.size());

        return data;
    }
};
```

---

### 5.5 Tests de TensorStore

**Test C++**:
```cpp
// tests/storage/tensor_store_test.cc
#include <gtest/gtest.h>
#include "storage/tensor_store.h"

TEST(TensorStore, InsertAndRetrieve) {
    Storage::TensorStore store("/tmp/test_tensor_store");

    // Crear tensor 1D [1.0, 2.0, 3.0]
    Storage::TensorStore::TensorEntry entry;
    entry.dtype = Storage::TensorStore::DataType::FLOAT32;
    entry.shape = {3};
    entry.data.resize(3 * sizeof(float));

    float values[] = {1.0f, 2.0f, 3.0f};
    std::memcpy(entry.data.data(), values, sizeof(values));

    uint64_t id = store.insert(entry);

    // Recuperar
    auto retrieved = store.get(id);

    EXPECT_EQ(retrieved.dtype, Storage::TensorStore::DataType::FLOAT32);
    EXPECT_EQ(retrieved.shape.size(), 1);
    EXPECT_EQ(retrieved.shape[0], 3);

    float* retrieved_values = reinterpret_cast<float*>(retrieved.data.data());
    EXPECT_FLOAT_EQ(retrieved_values[0], 1.0f);
    EXPECT_FLOAT_EQ(retrieved_values[1], 2.0f);
    EXPECT_FLOAT_EQ(retrieved_values[2], 3.0f);
}
```

---

### 5.6 Tareas y Estimación

| Tarea | Archivo(s) | Líneas | Esfuerzo |
|-------|-----------|--------|----------|
| Implementar TensorStore | `tensor_store.h/cc` | ~350 | 1 semana |
| Integrar en GQLModel | `gql_model.h`, `projection_storage.h` | ~30 | 1 día |
| Conversiones ObjectId | `conversions.h` | ~80 | 2 días |
| Tests unitarios | `tests/storage/` | ~150 | 3 días |
| Tests de integración | Scripts + GQL | ~100 | 2 días |

**Total Pre-requisito 3**: ~710 líneas, **3-4 semanas**

---

## 6. Pre-requisito 4: Integración LibTorch

### 6.1 ¿Por qué es crítico?

LibTorch es el C++ API de PyTorch. Es **fundamental** para:
- Entrenar modelos GNN dentro de MillenniumDB
- Implementar GCN, GAT, GraphSAGE en C++
- GPU support

Sin LibTorch, **NO podemos hacer entrenamiento nativo**.

---

### 6.2 Instalación de LibTorch

#### 6.2.1 Descargar LibTorch

```bash
cd /home/benito/B_MillenniumDB/MillenniumDB/third_party

# CPU version (para desarrollo)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# GPU version (para producción, requiere CUDA)
# wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
# unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
```

**Resultado**:
```
third_party/libtorch/
├── include/
│   ├── torch/
│   ├── ATen/
│   └── c10/
├── lib/
│   ├── libtorch.so
│   ├── libc10.so
│   └── ...
└── share/
```

---

#### 6.2.2 Integrar en CMake

**Archivo**: `CMakeLists.txt`

```cmake
# Buscar LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)

# Agregar includes
include_directories(${TORCH_INCLUDE_DIRS})

# Link con LibTorch
target_link_libraries(mdb ${TORCH_LIBRARIES})

# Copiar libs de LibTorch a build directory (para runtime)
if(TORCH_FOUND)
    message(STATUS "LibTorch found: ${TORCH_LIBRARIES}")

    # Copy shared libraries
    file(GLOB TORCH_LIBS "${TORCH_INSTALL_PREFIX}/lib/*.so*")
    foreach(LIB ${TORCH_LIBS})
        file(COPY ${LIB} DESTINATION ${CMAKE_BINARY_DIR}/lib)
    endforeach()
endif()
```

**Verificar**:
```bash
cd /home/benito/B_MillenniumDB/MillenniumDB
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release -j4

# Debe mostrar:
# -- LibTorch found: /home/benito/.../third_party/libtorch/lib/libtorch.so;...
```

---

#### 6.2.3 Test de Integración

**Archivo nuevo**: `tests/libtorch/test_integration.cc`

```cpp
#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(LibTorch, BasicTensor) {
    // Crear tensor 2x3
    torch::Tensor tensor = torch::rand({2, 3});

    EXPECT_EQ(tensor.size(0), 2);
    EXPECT_EQ(tensor.size(1), 3);
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
}

TEST(LibTorch, SimpleForwardPass) {
    // Modelo simple: Linear(3, 2)
    torch::nn::Linear model(3, 2);

    // Input: batch de 4 muestras, 3 features
    torch::Tensor input = torch::rand({4, 3});

    // Forward
    torch::Tensor output = model(input);

    EXPECT_EQ(output.size(0), 4);
    EXPECT_EQ(output.size(1), 2);
}

TEST(LibTorch, GPUAvailable) {
    bool cuda_available = torch::cuda::is_available();

    if (cuda_available) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is NOT available (CPU only)" << std::endl;
    }

    // Test pasa con o sin GPU
    EXPECT_TRUE(true);
}
```

**Compilar y ejecutar**:
```bash
cmake --build build/Release --target test_libtorch_integration
./build/Release/tests/test_libtorch_integration

# Salida esperada:
# [==========] Running 3 tests from 1 test suite.
# [ RUN      ] LibTorch.BasicTensor
# [       OK ] LibTorch.BasicTensor (2 ms)
# [ RUN      ] LibTorch.SimpleForwardPass
# [       OK ] LibTorch.SimpleForwardPass (5 ms)
# [ RUN      ] LibTorch.GPUAvailable
# CUDA is NOT available (CPU only)
# [       OK ] LibTorch.GPUAvailable (0 ms)
# [==========] 3 tests from 1 test suite ran. (7 ms total)
```

---

### 6.3 Ejemplo: Simple GCN Layer

**Archivo**: `src/graph_models/gql/ml/gcn_layer.h`

```cpp
#pragma once

#include <torch/torch.h>

namespace GQL {
namespace ML {

// Simple Graph Convolutional Layer (Kipf & Welling 2017)
class GCNLayer : public torch::nn::Module {
public:
    GCNLayer(int input_dim, int output_dim)
        : linear(register_module("linear", torch::nn::Linear(input_dim, output_dim)))
    {}

    // Forward pass
    // x: [num_nodes, input_dim]
    // adj: [num_nodes, num_nodes] (adjacency matrix)
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        // Message passing: adj @ x
        torch::Tensor messages = torch::mm(adj, x);

        // Linear transformation
        torch::Tensor output = linear(messages);

        return output;
    }

private:
    torch::nn::Linear linear;
};

} // namespace ML
} // namespace GQL
```

**Test**:
```cpp
TEST(GCN, ForwardPass) {
    using namespace GQL::ML;

    // Grafo con 4 nodos, 3 features
    int num_nodes = 4;
    int input_dim = 3;
    int output_dim = 2;

    // Features aleatorios
    torch::Tensor x = torch::rand({num_nodes, input_dim});

    // Adjacency matrix (grafo completo para simplificar)
    torch::Tensor adj = torch::ones({num_nodes, num_nodes});
    adj = adj / num_nodes;  // Normalizar

    // Crear capa GCN
    GCNLayer gcn(input_dim, output_dim);

    // Forward
    torch::Tensor output = gcn.forward(x, adj);

    EXPECT_EQ(output.size(0), num_nodes);
    EXPECT_EQ(output.size(1), output_dim);
}
```

---

### 6.4 Tareas y Estimación

| Tarea | Archivo(s) | Líneas | Esfuerzo |
|-------|-----------|--------|----------|
| Descargar e instalar LibTorch | `third_party/libtorch/` | N/A | 1 día |
| Integrar en CMake | `CMakeLists.txt` | ~30 | 2 días |
| Test de integración básico | `tests/libtorch/test_integration.cc` | ~80 | 1 día |
| Ejemplo GCN layer | `src/graph_models/gql/ml/gcn_layer.h` | ~50 | 2 días |
| Tests de GCN layer | `tests/ml/test_gcn.cc` | ~100 | 2 días |
| Documentación | README | ~50 | 1 día |

**Total Pre-requisito 4**: ~310 líneas, **2-3 semanas**

---

## 7. Roadmap de Ejecución

### 7.1 Orden de Implementación

**Dependencias**:
```
Pre-req 1 (MUTATE)  ←---┐
                        ├── Pre-req 2 (Procedures) puede usarlos
Pre-req 3 (TENSOR)  ←---┘

Pre-req 4 (LibTorch) ← Independiente
```

**Orden recomendado**:
1. **Pre-req 4** (LibTorch) - Independiente, puede hacerse en paralelo
2. **Pre-req 3** (TENSOR) - Necesario para almacenar embeddings
3. **Pre-req 1** (MUTATE) - Necesario para persistir resultados
4. **Pre-req 2** (Procedures) - Depende de MUTATE para ser útil

---

### 7.2 Timeline con 1 Developer

```
Semana 1-2:   Pre-req 4 (LibTorch integration)
Semana 3-5:   Pre-req 3 (TensorStore)
Semana 6-9:   Pre-req 1 (MUTATE PROPERTY)
Semana 10-12: Pre-req 2 (Procedures system)
```

**Total**: **12 semanas (3 meses)**

---

### 7.3 Timeline con 2 Developers (Paralelo)

```
Dev 1:
Semana 1-2:   Pre-req 4 (LibTorch)
Semana 3-5:   Pre-req 3 (TensorStore)
Semana 6-8:   Ayudar en tests de Procedures

Dev 2:
Semana 1-4:   Pre-req 1 (MUTATE PROPERTY)
Semana 5-8:   Pre-req 2 (Procedures system)
```

**Total**: **8 semanas (2 meses)**

---

### 7.4 Milestones y Verificación

#### Milestone 1: LibTorch OK (Semana 2)
```bash
# Test
./build/Release/tests/test_libtorch_integration

# Debe mostrar:
# [==========] 3 tests from 1 test suite ran. (X ms total)
# [  PASSED  ] 3 tests.
```

#### Milestone 2: TensorStore OK (Semana 5)
```bash
# Test almacenamiento de tensores
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n)
RETURN n.id
LIMIT 1
' | grep "id"

# Internamente, verificar que tensor_store.bin existe
ls -lh data/dbs/gql/test_db/tensor_store.bin
```

#### Milestone 3: MUTATE OK (Semana 9)
```bash
# Test MUTATE
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n:User)
MUTATE PROPERTY n.test_property = 123
'

# Verificar
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n:User)
WHERE n.test_property = 123
RETURN count(n)
' | grep "123"  # Debe haber nodos con la propiedad
```

#### Milestone 4: Procedures OK (Semana 12)
```bash
# Test procedimiento echo
curl -X POST http://localhost:1234/gql --data '
CALL mdb.echo({message: "Hello"})
YIELD message
RETURN message
' | grep "Hello"

# Test procedimiento PageRank + MUTATE
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
CALL mdb.pagerank()
YIELD nodeId, score
MUTATE PROPERTY pagerank = score
'

# Verificar persistencia
curl -X POST http://localhost:1234/gql --data '
USE "test_proj"
MATCH (n)
WHERE n.pagerank IS NOT NULL
RETURN count(n)
' | grep -v "0"  # Debe haber nodos con pagerank
```

---

### 7.5 Entregable Final

Al completar estos 4 pre-requisitos, MillenniumDB estará listo para:

✅ **Iniciar Fase 0 de ARQUITECTURA_GNN_NATIVA.md**

**Capacidades habilitadas**:
1. ✅ Procedimientos GQL registrables (ej: `CALL mdb.gnn.define`)
2. ✅ Persistencia de propiedades con `MUTATE PROPERTY`
3. ✅ Almacenamiento de tensores/vectores (embeddings)
4. ✅ LibTorch integrado (para GNN en C++)

**Siguiente paso después**: Implementar la Fase 0 completa del documento `ARQUITECTURA_GNN_NATIVA.md`:
- Model Catalog
- GNN Model Wrappers (GCN, GAT, GraphSAGE)
- Training Pipeline
- Graph Data Loader

---

## 8. Resumen Ejecutivo

### 8.1 Estado Actual vs Objetivo

**Estado Actual**:
- ✅ PROJECT y USE funcionan
- ✅ Streaming básico existe
- ✅ TENSOR type existe (parcial)
- ❌ NO existe MUTATE PROPERTY
- ❌ NO existe sistema de procedimientos
- ❌ NO existe TensorStore maduro
- ❌ LibTorch NO integrado

**Objetivo (Post-Pre-requisitos)**:
- ✅ MUTATE PROPERTY funcional
- ✅ Sistema de procedimientos con registro
- ✅ TensorStore maduro
- ✅ LibTorch integrado y tested
- ✅ Listo para Fase 0 de GNN nativa

---

### 8.2 Esfuerzo Total

| Pre-requisito | Líneas C++ | Esfuerzo | Prioridad |
|---------------|-----------|----------|-----------|
| 1. MUTATE PROPERTY | ~680 | 3-4 semanas | CRÍTICA |
| 2. Procedures System | ~580 | 2-3 semanas | CRÍTICA |
| 3. TensorStore | ~710 | 3-4 semanas | CRÍTICA |
| 4. LibTorch | ~310 | 2-3 semanas | CRÍTICA |
| **TOTAL** | **~2,280** | **10-14 semanas** | **BLOQUEANTE** |

**Con 1 dev**: 12 semanas (3 meses)
**Con 2 devs (paralelo)**: 8 semanas (2 meses)

---

### 8.3 Próximos Pasos

1. ✅ **Aprobar este roadmap**
2. ⏳ **Asignar desarrolladores** (1-2 devs)
3. ⏳ **Iniciar Pre-req 4** (LibTorch, independiente)
4. ⏳ **Iniciar Pre-req 3** (TensorStore)
5. ⏳ **Continuar con Pre-req 1 y 2** (MUTATE + Procedures)
6. ⏳ **Testing exhaustivo** de cada pre-requisito
7. ✅ **Milestone final**: Listo para Fase 0 de GNN nativa

---

**Fin del Roadmap de Pre-requisitos**

---

**Autores**: Claude + Benito
**Fecha**: 19 de Octubre, 2025
**Versión**: 1.0
**Palabras**: ~8,000
**Líneas de código (estimadas)**: ~2,280 C++
