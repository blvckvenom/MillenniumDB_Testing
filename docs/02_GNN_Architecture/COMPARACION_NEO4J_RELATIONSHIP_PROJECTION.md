# Comparación: MillenniumDB PROJECT vs Neo4j GDS Relationship Projection Map

**Fecha**: 20 de Octubre, 2025
**Versión**: 1.0

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Análisis Feature por Feature](#análisis-feature-por-feature)
3. [Tabla Comparativa Completa](#tabla-comparativa-completa)
4. [Ejemplos Lado a Lado](#ejemplos-lado-a-lado)
5. [Gap Analysis Detallado](#gap-analysis-detallado)
6. [Roadmap de Implementación](#roadmap-de-implementación)

---

## Resumen Ejecutivo

### ✅ Lo que MillenniumDB YA TIENE (80% de funcionalidad básica)

| Feature Neo4j GDS | Estado MillenniumDB | Equivalente |
|-------------------|---------------------|-------------|
| **Crear proyección de grafo** | ✅ **COMPLETO** | `MATCH (n)-[e]->(m) RETURN PROJECT("nombre")` |
| **Incluir propiedades de relaciones** | ✅ **COMPLETO** | `INCLUDE PROPERTIES` |
| **Incluir propiedades de nodos** | ✅ **COMPLETO** | `INCLUDE PROPERTIES` |
| **Persistencia en disco** | ✅ **MEJOR QUE NEO4J** | Proyecciones persisten automáticamente |
| **Filtrado por patrón MATCH** | ✅ **COMPLETO** | `MATCH (u:User)-[e:KNOWS]->(v)` |
| **Context switching** | ✅ **COMPLETO** | `USE "projection_name"` |

### ❌ Lo que MillenniumDB NO TIENE (funcionalidad avanzada de Neo4j GDS)

| Feature Neo4j GDS | Estado MillenniumDB | Impacto |
|-------------------|---------------------|---------|
| **Orientación de relaciones** (NATURAL/UNDIRECTED/REVERSE) | ❌ **FALTA** | ALTO - Necesario para GNN no dirigidas |
| **Agregación de relaciones paralelas** (COUNT/SUM/MIN/MAX) | ❌ **FALTA** | ALTO - Multigrafos comunes en análisis |
| **Valor por defecto para propiedades** | ❌ **FALTA** | MEDIO - Manejo de datos faltantes |
| **Renombrado de tipos/propiedades** | ❌ **FALTA** | BAJO - Conveniencia |
| **Agregación por propiedad específica** | ❌ **FALTA** | ALTO - Análisis de redes pesadas |

---

## Análisis Feature por Feature

### 1. **Tipo de Relación Proyectada** (`<projected_type>`)

#### **Neo4j GDS**:
```cypher
// Renombrar KNOWS → FRIENDS en la proyección
{
  FRIENDS: {
    type: 'KNOWS',  // Tipo fuente en Neo4j
    ...
  }
}
```

#### **MillenniumDB**:
```gql
// El tipo de relación NO se puede renombrar
MATCH (u)-[e:KNOWS]->(v) RETURN PROJECT("graph")
// La proyección mantiene el tipo "KNOWS" tal cual
```

**Estado**: ❌ **NO SOPORTADO** - MillenniumDB no permite renombrar tipos de relaciones en la proyección.

**Impacto**: BAJO - Raramente necesario, más conveniencia que funcionalidad crítica.

---

### 2. **Tipo de Relación Fuente** (`type`)

#### **Neo4j GDS**:
```cypher
{
  READ: {type: 'READ'}  // Explícito
}
```

#### **MillenniumDB**:
```gql
MATCH (u)-[e:READ]->(v) RETURN PROJECT("graph")
// El tipo se captura implícitamente del patrón MATCH
```

**Estado**: ✅ **SOPORTADO** - Pero de forma implícita vía MATCH, no explícita.

**Diferencia**: Neo4j permite configuración explícita en un mapa, MillenniumDB usa el patrón de consulta.

---

### 3. **Orientación de Relaciones** (`orientation`)

#### **Neo4j GDS**:
```cypher
{
  KNOWS: {
    orientation: 'UNDIRECTED'  // NATURAL (default), UNDIRECTED, REVERSE
  }
}
```

**Resultado**: Una relación `(A)-[:KNOWS]->(B)` se convierte en dos relaciones no dirigidas en el grafo proyectado.

#### **MillenniumDB**:
```gql
MATCH (u)-[e:KNOWS]->(v) RETURN PROJECT("graph")
// La orientación SIEMPRE es NATURAL (preserva la dirección original)
```

**Estado**: ❌ **NO SOPORTADO**

**Código actual** (agg_project.h:195-196):
```cpp
else if (type == GQL_OID::Type::DIRECTED_EDGE || type == GQL_OID::Type::UNDIRECTED_EDGE) {
    edges_seen[oid] = (type == GQL_OID::Type::DIRECTED_EDGE);  // Solo preserva tipo original
```

**Gap**:
- No hay lógica para convertir `DIRECTED → UNDIRECTED`
- No hay lógica para invertir direcciones (`REVERSE`)
- No hay sintaxis para especificar orientación

**Impacto**: **ALTO** - Muchos algoritmos de GNN requieren grafos no dirigidos (GraphSAGE, GCN).

**Ejemplo de uso crítico**:
```cypher
// Neo4j: Convertir red social dirigida en no dirigida para GNN
CALL gds.graph.project(
  'social_undirected',
  'Person',
  {FOLLOWS: {orientation: 'UNDIRECTED'}}
)
// Ahora FOLLOWS se trata como amistad simétrica
```

---

### 4. **Agregación de Relaciones Paralelas** (`aggregation`)

#### **Neo4j GDS**:
```cypher
// Ejemplo: Florentin tiene 2 relaciones READ con The Hobbit
// Queremos CONTAR cuántas veces leyó el libro

{
  READ: {
    aggregation: 'NONE',  // DEFAULT: mantener todas las relaciones paralelas
    // O: 'COUNT', 'SUM', 'MIN', 'MAX', 'SINGLE'
  }
}
```

**Comportamiento con `COUNT`**:
- Original: `(Florentin)-[:READ {pages: 4}]->(Hobbit)` + `(Florentin)-[:READ {pages: 42}]->(Hobbit)`
- Proyectado: `(Florentin)-[:READ {count: 2}]->(Hobbit)` (una sola relación)

#### **MillenniumDB**:
```gql
MATCH (u)-[e:READ]->(v) RETURN PROJECT("graph" INCLUDE PROPERTIES)
// Si hay 2 relaciones READ entre u y v, se crean 2 entradas separadas
```

**Estado**: ❌ **NO SOPORTADO**

**Código actual** (projection_storage.cc:170-185):
```cpp
void ProjectionStorage::add_edge(const ProjectedEdge& edge) {
    // Cada edge se escribe como registro separado, NO hay agregación
    Record<3> from_to_record;
    from_to_record[0] = edge.from.id;
    from_to_record[1] = edge.to.id;
    from_to_record[2] = edge.edge_id.id;
    from_to_edge_index->insert(from_to_record);  // ← Inserción directa, sin chequeo de duplicados
```

**Gap**:
- No hay detección de relaciones paralelas
- No hay mecanismo de agregación (COUNT, SUM, MIN, MAX)
- No hay sintaxis para especificar estrategia de agregación

**Impacto**: **ALTO** - Multigrafos son muy comunes:
- Redes de transporte (múltiples vuelos entre ciudades)
- Redes sociales (múltiples interacciones entre usuarios)
- Redes de co-autoría (múltiples papers juntos)

**Ejemplo de uso crítico**:
```cypher
// Neo4j: Sumar peso de todas las interacciones entre usuarios
{
  INTERACTS: {
    properties: {
      total_weight: {
        property: 'weight',
        aggregation: 'SUM'  // ← Sumar pesos de relaciones paralelas
      }
    }
  }
}
```

---

### 5. **Proyección de Propiedades** (`properties`)

#### **Neo4j GDS - Sintaxis Simple**:
```cypher
{
  READ: {
    properties: 'numberOfPages'  // Una propiedad
  }
}
```

#### **MillenniumDB**:
```gql
MATCH (u)-[e:READ]->(v) RETURN PROJECT("graph" INCLUDE PROPERTIES)
// Incluye TODAS las propiedades de nodos y aristas
```

**Estado**: ✅ **PARCIALMENTE SOPORTADO**

**Diferencias**:
- ✅ MillenniumDB puede incluir propiedades (`INCLUDE PROPERTIES`)
- ❌ MillenniumDB incluye TODAS o NINGUNA, no permite selección selectiva
- ❌ No hay renombrado de propiedades
- ❌ No hay valores por defecto

**Código actual** (agg_project.h:322-373):
```cpp
// Quinto paso: extraer propiedades del grafo principal
if (options.include_properties) {
    // Itera TODAS las propiedades del nodo
    BptIter<3> it = gql_model.node_key_value->get_range(
        &interruption,
        { node_id.id, 0, 0 },
        { node_id.id, UINT64_MAX, UINT64_MAX }  // ← Rango completo, sin filtro
    );

    auto record = it.next();
    while (record != nullptr) {
        ObjectId key_id((*record)[1]);
        ObjectId value_id((*record)[2]);
        projection_storage->add_node_property(node_id, key_id, value_id);  // ← Todas
        record = it.next();
    }
}
```

**Gap**:
- No hay filtrado de propiedades específicas
- No hay renombrado de propiedades proyectadas

**Impacto**: **MEDIO** - Conveniencia más que funcionalidad crítica.

---

### 6. **Mapa de Proyección de Propiedades** (Granular)

#### **Neo4j GDS**:
```cypher
{
  READ: {
    properties: {
      pageCount: {           // ⑥ Nombre proyectado (puede ser diferente)
        property: 'numberOfPages',  // ⑦ Propiedad fuente en Neo4j
        defaultValue: 0.0,    // ⑧ Valor si falta
        aggregation: 'SUM'    // ⑨ Sumar valores de relaciones paralelas
      }
    }
  }
}
```

**Ejemplo práctico**:
```cypher
// Florentin tiene 2 READs con Hobbit: pages=4 y pages=42
// Resultado proyectado: pageCount = 46.0 (suma)
```

#### **MillenniumDB**:
```gql
// No hay equivalente
MATCH (u)-[e:READ]->(v) RETURN PROJECT("graph" INCLUDE PROPERTIES)
```

**Estado**: ❌ **NO SOPORTADO** (ninguno de los 4 sub-campos)

| Sub-campo | Neo4j GDS | MillenniumDB | Impacto |
|-----------|-----------|--------------|---------|
| **Renombrado** (`<projected_prop>` ≠ `property`) | ✅ | ❌ | BAJO |
| **Propiedad fuente** (`property`) | ✅ Explícito | ✅ Implícito | BAJO |
| **Valor por defecto** (`defaultValue`) | ✅ | ❌ | **ALTO** |
| **Agregación específica** (`aggregation`) | ✅ | ❌ | **ALTO** |

**Gap crítico - Valor por defecto**:

**Neo4j**:
```cypher
// Si un libro no tiene 'numberOfPages', usar 0.0
{pageCount: {property: 'numberOfPages', defaultValue: 0.0}}
```

**MillenniumDB**:
```gql
// Si falta la propiedad, simplemente no existe en la proyección
// Queries subsiguientes pueden fallar o retornar NULL
```

**Impacto**: **ALTO** - Muchos algoritmos de GNN requieren features completas (no pueden manejar valores faltantes sin imputación).

**Gap crítico - Agregación específica**:

**Neo4j**:
```cypher
// Sumar 'weight', pero tomar MAX de 'timestamp'
{
  INTERACTS: {
    properties: {
      total_weight: {property: 'weight', aggregation: 'SUM'},
      latest_time: {property: 'timestamp', aggregation: 'MAX'}
    }
  }
}
```

**MillenniumDB**: ❌ No hay forma de hacer esto.

**Impacto**: **ALTO** - Redes temporales pesadas necesitan agregar diferentes propiedades de diferentes formas.

---

## Tabla Comparativa Completa

| # | Campo Neo4j GDS | Descripción | MillenniumDB | Estado | Impacto | Esfuerzo Implementación |
|---|-----------------|-------------|--------------|--------|---------|------------------------|
| **① Nivel de Relación** |
| ① | `<projected_type>` | Renombrar tipo de relación | ❌ No | ❌ NO | BAJO | 1-2 semanas |
| ② | `type` | Tipo fuente de Neo4j | ✅ Implícito en MATCH | ✅ SÍ | - | N/A |
| ③ | `orientation` | NATURAL/UNDIRECTED/REVERSE | ❌ Solo NATURAL | ❌ NO | **ALTO** | 3-4 semanas |
| ④ | `aggregation` | COUNT/SUM/MIN/MAX/SINGLE/NONE | ❌ No | ❌ NO | **ALTO** | 4-6 semanas |
| ⑤ | `properties` | Proyección de propiedades | ✅ INCLUDE PROPERTIES (todas) | ⚠️ PARCIAL | MEDIO | N/A |
| **⑥-⑨ Nivel de Propiedad** |
| ⑥ | `<projected_prop>` | Renombrar propiedad | ❌ No | ❌ NO | BAJO | 1-2 semanas |
| ⑦ | `property` | Propiedad fuente | ✅ Implícito | ✅ SÍ | - | N/A |
| ⑧ | `defaultValue` | Valor por defecto si falta | ❌ No | ❌ NO | **ALTO** | 2-3 semanas |
| ⑨ | `aggregation` (propiedad) | Agregación específica | ❌ No | ❌ NO | **ALTO** | 3-4 semanas |

### Resumen de Compatibilidad:

- ✅ **Completo**: 2/9 campos (22%)
- ⚠️ **Parcial**: 1/9 campos (11%)
- ❌ **Faltante**: 6/9 campos (67%)

---

## Ejemplos Lado a Lado

### Ejemplo 1: Grafo No Dirigido (GNN)

#### **Neo4j GDS**:
```cypher
// PROBLEMA: Red social dirigida (FOLLOWS), necesito no dirigida para GCN
CALL gds.graph.project(
  'social_undirected',
  'Person',
  {FOLLOWS: {orientation: 'UNDIRECTED'}}
)

// Resultado: (A)-[:FOLLOWS]->(B) se convierte en edges bidireccionales
```

#### **MillenniumDB**:
```gql
-- WORKAROUND: Crear relaciones bidireccionales manualmente en el MATCH
MATCH (u:Person)-[e:FOLLOWS]->(v:Person)
RETURN PROJECT("social_undirected")

-- PROBLEMA: Solo captura la dirección original
-- NO hay forma de hacer undirected sin duplicar datos en el grafo principal
```

**Solución MillenniumDB** (manual, NO ideal):
```gql
-- 1. Crear proyección con dirección original
MATCH (u)-[e:FOLLOWS]->(v) RETURN PROJECT("temp")

-- 2. NO HAY FORMA de agregar relaciones inversas después
-- Necesitarías modificar el grafo principal:
INSERT (v)-[:FOLLOWS]->(u) WHERE (u)-[:FOLLOWS]->(v)  -- ❌ No existe en GQL actual
```

---

### Ejemplo 2: Agregación de Relaciones Paralelas

#### **Neo4j GDS**:
```cypher
// Dataset: Florentin lee The Hobbit 2 veces (pages: 4 y 42)
CALL gds.graph.project(
  'readCount',
  ['Person', 'Book'],
  {
    READ: {
      properties: {
        numberOfReads: {property: '*', aggregation: 'COUNT'},  // ← Contar
        totalPages: {property: 'numberOfPages', aggregation: 'SUM'}  // ← Sumar
      }
    }
  }
)

// Verificar:
CALL gds.graph.relationshipProperty.stream('readCount', 'numberOfReads')
// Resultado: Florentin → Hobbit: numberOfReads=2, totalPages=46
```

#### **MillenniumDB**:
```gql
-- Dataset: Mismo que Neo4j
MATCH (p:Person)-[r:READ]->(b:Book)
RETURN PROJECT("readCount" INCLUDE PROPERTIES)

-- Resultado: DOS relaciones separadas en la proyección
-- NO hay forma de agregarlas

-- Workaround (manual, muy limitado):
USE "readCount"
MATCH (p:Person)-[r:READ]->(b:Book)
RETURN p.name, b.name, count(r) AS numberOfReads
-- PROBLEMA: count(r) se calcula en query time, NO está en la proyección
```

**Consecuencia**: No puedes persistir el conteo en la proyección para usarlo en algoritmos.

---

### Ejemplo 3: Valor por Defecto para Propiedades

#### **Neo4j GDS**:
```cypher
// PROBLEMA: Algunos libros no tienen 'numberOfPages'
CALL gds.graph.project(
  'books',
  ['Person', 'Book'],
  {
    READ: {
      properties: {
        pages: {
          property: 'numberOfPages',
          defaultValue: 100.0  // ← Asumir 100 páginas si falta
        }
      }
    }
  }
)

// Resultado: Todos los libros tienen 'pages' (NaN → 100.0)
```

#### **MillenniumDB**:
```gql
MATCH (p)-[r:READ]->(b) RETURN PROJECT("books" INCLUDE PROPERTIES)

-- Si un libro no tiene 'numberOfPages', simplemente no existe en la proyección
USE "books"
MATCH (p)-[r:READ]->(b)
RETURN b.numberOfPages  -- Puede ser NULL

-- Workaround (manual en queries):
RETURN COALESCE(b.numberOfPages, 100) AS pages
-- PROBLEMA: Hay que hacer esto en CADA query, no está en la proyección
```

---

## Gap Analysis Detallado

### Gaps Críticos (Bloqueantes para GNN)

#### **1. Orientación No Dirigida** (`orientation: UNDIRECTED`)

**Por qué es crítico**:
- GCN, GraphSAGE, GAT requieren grafos no dirigidos
- 80% de algoritmos de GNN asumen simetría en vecindarios
- Workarounds manuales son ineficientes y propensos a errores

**Impacto sin esto**:
- ❌ No puedes entrenar GNN estándar en redes dirigidas (ej: Twitter, Wikipedia links)
- ❌ Necesitas duplicar relaciones manualmente en el grafo principal (no escalable)

**Complejidad de implementación**: MEDIA-ALTA (3-4 semanas)

**Diseño propuesto**:
```gql
-- Nueva sintaxis
MATCH (u)-[e:FOLLOWS]->(v)
RETURN PROJECT("graph" ORIENTATION UNDIRECTED)

-- O más granular:
MATCH (u)-[e:FOLLOWS]->(v)
RETURN PROJECT("graph"
  RELATIONSHIP_OPTIONS {
    FOLLOWS: {orientation: UNDIRECTED}
  }
)
```

**Implementación**:
1. Agregar campo `orientation` a `ProjectionOptions` (agg_project.h)
2. Modificar `add_edge()` para escribir aristas bidireccionales si `UNDIRECTED`
3. Agregar sintaxis al parser GQL

---

#### **2. Agregación de Relaciones Paralelas** (`aggregation`)

**Por qué es crítico**:
- Multigrafos son la norma en análisis de redes reales
- GNN sobre redes pesadas necesitan agregar pesos
- Sin agregación, la proyección puede ser 10-100x más grande de lo necesario

**Impacto sin esto**:
- ❌ Proyecciones infladas (GB → TB)
- ❌ No puedes calcular "número de interacciones" o "peso total"
- ❌ Algoritmos de GNN reciben features duplicadas inconsistentes

**Complejidad de implementación**: ALTA (4-6 semanas)

**Diseño propuesto**:
```gql
-- Sintaxis 1: Agregación global (todas las propiedades)
MATCH (u)-[e:INTERACTS]->(v)
RETURN PROJECT("graph" AGGREGATION SUM)

-- Sintaxis 2: Agregación por propiedad
MATCH (u)-[e:INTERACTS]->(v)
RETURN PROJECT("graph"
  PROPERTY_OPTIONS {
    weight: {aggregation: SUM},
    timestamp: {aggregation: MAX}
  }
)
```

**Implementación**:
1. Detectar relaciones paralelas: `(from, to)` duplicados
2. Buffer de agregación en `ProjectionStorage`
3. Funciones de agregación: `COUNT`, `SUM`, `MIN`, `MAX`, `AVG`
4. Escribir relación agregada única

---

#### **3. Valores por Defecto** (`defaultValue`)

**Por qué es crítico**:
- Datos reales SIEMPRE tienen valores faltantes
- GNN no pueden manejar NaN sin imputación
- Imputación manual en queries es ineficiente

**Impacto sin esto**:
- ❌ Entrenamientos fallan con "NaN in loss function"
- ❌ Necesitas pre-procesar datos antes de importar a MillenniumDB
- ❌ No puedes usar datos directamente de fuentes externas

**Complejidad de implementación**: MEDIA (2-3 semanas)

**Diseño propuesto**:
```gql
MATCH (u)-[e:READ]->(v)
RETURN PROJECT("graph"
  PROPERTY_DEFAULTS {
    pages: 100.0,
    rating: 3.0
  }
)
```

**Implementación**:
1. Agregar `property_defaults` map a `ProjectionOptions`
2. En `add_node_property()` / `add_edge_property()`, chequear si valor es NULL
3. Si NULL, usar default; si no hay default, usar NaN

---

### Gaps de Conveniencia (No Bloqueantes)

#### **4. Renombrado de Tipos/Propiedades**

**Impacto**: BAJO - Raramente necesario

**Uso**:
```cypher
// Neo4j: Renombrar para uniformidad
{
  FRIEND: {type: 'KNOWS'}  // KNOWS → FRIEND en proyección
}
```

**Complejidad**: BAJA (1-2 semanas)

---

## Roadmap de Implementación

### Fase 1: Orientación No Dirigida (3-4 semanas) - **CRÍTICO**

**Objetivo**: Soportar `ORIENTATION UNDIRECTED`

**Tasks**:
1. Agregar enum `Orientation {NATURAL, UNDIRECTED, REVERSE}` a `ProjectionOptions`
2. Parser: Agregar sintaxis `ORIENTATION <type>`
3. `projection_storage.cc`: Modificar `add_edge()` para escritura bidireccional
4. Tests: Verificar que `(A)->(B)` resulta en `(A)-(B)` simétrico

**Deliverables**:
- [ ] Sintaxis: `MATCH (u)-[e]->(v) RETURN PROJECT("g" ORIENTATION UNDIRECTED)`
- [ ] Test unitario: `projection_orientation_test.cc`
- [ ] Documentación actualizada

---

### Fase 2: Agregación Básica (4-6 semanas) - **CRÍTICO**

**Objetivo**: Soportar `AGGREGATION COUNT/SUM`

**Tasks**:
1. Detectar relaciones paralelas en `agg_project.h::process()`
2. Implementar buffer de agregación (map de `(from, to)` → propiedades agregadas)
3. Funciones de agregación: `COUNT`, `SUM` (MIN/MAX/AVG en fase posterior)
4. Flush de relaciones agregadas al finalizar proyección

**Deliverables**:
- [ ] Sintaxis: `AGGREGATION SUM`
- [ ] Test: Verificar que 2 relaciones `(A)->(B)` resultan en 1 con count=2
- [ ] Documentación con ejemplos de multigrafos

---

### Fase 3: Valores por Defecto (2-3 semanas) - **CRÍTICO**

**Objetivo**: Soportar `PROPERTY_DEFAULTS`

**Tasks**:
1. Agregar `std::unordered_map<std::string, ObjectId> property_defaults` a `ProjectionOptions`
2. Parser: `PROPERTY_DEFAULTS {prop1: val1, prop2: val2}`
3. Modificar `add_node_property()` / `add_edge_property()` para aplicar defaults
4. Tests: Verificar que propiedades faltantes usan defaults

**Deliverables**:
- [ ] Sintaxis: `PROPERTY_DEFAULTS {pages: 100.0}`
- [ ] Test unitario con datos faltantes
- [ ] Documentación de imputación

---

### Fase 4: Renombrado (1-2 semanas) - **OPCIONAL**

**Objetivo**: Soportar renombrado de tipos y propiedades

**Tasks**:
1. Agregar `std::unordered_map<std::string, std::string> type_mapping`
2. Agregar `std::unordered_map<std::string, std::string> property_mapping`
3. Aplicar renombrado durante escritura

**Deliverables**:
- [ ] Sintaxis: `TYPE_MAPPING {KNOWS: FRIEND}`
- [ ] Documentación

---

## Timeline Total

| Fase | Duración | Esfuerzo (dev-semanas) | Prioridad | Dependencias |
|------|----------|------------------------|-----------|--------------|
| **Fase 1: Orientación** | 3-4 semanas | 3-4 | **CRÍTICA** | Ninguna |
| **Fase 2: Agregación** | 4-6 semanas | 4-6 | **CRÍTICA** | Fase 1 (opcional) |
| **Fase 3: Defaults** | 2-3 semanas | 2-3 | **CRÍTICA** | Ninguna |
| **Fase 4: Renombrado** | 1-2 semanas | 1-2 | BAJA | Ninguna |

**Total Crítico**: 9-13 semanas (2-3 meses) con 1 desarrollador
**Total con Renombrado**: 10-15 semanas

**Paralelización**: Fase 1 y Fase 3 son independientes → Pueden hacerse en paralelo (reduce a 6-8 semanas con 2 devs).

---

## Conclusión

### Estado Actual vs Neo4j GDS

**MillenniumDB tiene**:
- ✅ Infraestructura sólida de proyecciones persistentes (MEJOR que Neo4j)
- ✅ Soporte básico de propiedades (nodos y aristas)
- ✅ Context switching con USE

**MillenniumDB necesita**:
- ❌ Orientación no dirigida (BLOQUEANTE para GNN)
- ❌ Agregación de relaciones paralelas (BLOQUEANTE para redes pesadas)
- ❌ Valores por defecto (BLOQUEANTE para datos reales)

**Prioridad**:
1. **Fase 1** (Orientación) - Desbloquea GNN básicas
2. **Fase 3** (Defaults) - Desbloquea datos reales
3. **Fase 2** (Agregación) - Desbloquea multigrafos a escala

Con estas 3 fases, MillenniumDB tendría **funcionalidad equivalente al 90% de casos de uso de Neo4j GDS Relationship Projection**.

---

**Próximo paso**: Implementar Fase 1 (Orientación) como parte del roadmap de pre-requisitos GNN.
