# Análisis Comparativo Exhaustivo: MillenniumDB PROJECT vs Neo4j GDS
## Hacia Capacidades Analíticas Avanzadas (Embeddings, GNN, ML sobre Grafos)

**Fecha**: 18 de Octubre, 2025
**Autor**: Análisis técnico para roadmap de features analíticas

---

## Resumen Ejecutivo

Este documento compara la implementación actual de **PROJECT/USE GRAPH** en MillenniumDB con las capacidades de **Neo4j Graph Data Science (GDS)**, enfocándose en identificar las brechas que impiden el uso de MillenniumDB para:

- **Cálculo de embeddings de nodos/aristas**
- **Entrenamiento de Redes Neuronales de Grafos (GNN)**
- **Algoritmos de grafos a escala** (PageRank, Centralidad, Comunidades)
- **Pipelines de Machine Learning** sobre grafos
- **Análisis OLAP** de alto rendimiento

**Conclusión clave**: MillenniumDB tiene una **base sólida con persistencia en disco** que Neo4j no tiene, pero le falta **TODA la capa analítica y de machine learning** que hace de GDS una herramienta poderosa para ciencia de datos sobre grafos.

---

## Tabla de Contenidos

1. [Comparativa de Arquitecturas](#1-comparativa-de-arquitecturas)
2. [Análisis Detallado por Componente](#2-análisis-detallado-por-componente)
3. [Brechas Críticas Identificadas](#3-brechas-críticas-identificadas)
4. [Roadmap de Implementación](#4-roadmap-de-implementación)
5. [Ventajas Competitivas de MillenniumDB](#5-ventajas-competitivas-de-millenniumdb)
6. [Estimación de Esfuerzo](#6-estimación-de-esfuerzo)

---

## 1. Comparativa de Arquitecturas

### 1.1 Neo4j GDS: Proyecciones en Memoria para Análisis OLAP

```
┌─────────────────────────────────────────────────────────────┐
│                    Neo4j GDS Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌─────────────────────┐      │
│  │  Neo4j Database  │         │   Graph Catalog     │      │
│  │    (Disk OLTP)   │────────>│   (Memory OLAP)     │      │
│  │                  │ Project │                     │      │
│  │ - Nodos          │         │ - Compressed format │      │
│  │ - Relaciones     │         │ - Topology arrays   │      │
│  │ - Propiedades    │         │ - No transactions   │      │
│  └──────────────────┘         │ - Volatile (RAM)    │      │
│                                └──────────┬──────────┘      │
│                                           │                 │
│                    ┌──────────────────────┴───────────┐     │
│                    │                                  │     │
│         ┌──────────v───────────┐         ┌───────────v─────┐
│         │   GDS Algorithms     │         │   ML Pipelines  │
│         ├──────────────────────┤         ├─────────────────┤
│         │ - PageRank           │         │ - Node Class.   │
│         │ - Community Detection│         │ - Link Pred.    │
│         │ - Centrality         │         │ - GraphSAGE     │
│         │ - Path Finding       │         │ - HashGNN       │
│         │ - Similarity         │         │ - FastRP        │
│         └──────────────────────┘         └─────────────────┘
│                    │                              │          │
│                    └──────────┬───────────────────┘          │
│                               │                              │
│                    ┌──────────v──────────┐                   │
│                    │   Results Handler   │                   │
│                    ├─────────────────────┤                   │
│                    │ - Write to DB       │                   │
│                    │ - Stream to client  │                   │
│                    │ - Export to PyG     │                   │
│                    └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

**Características clave**:
- **Separación OLTP/OLAP**: Base de datos en disco para transacciones, memoria para análisis
- **Formato comprimido**: Estructura optimizada para recorridos de grafos (no B+Trees)
- **Volátil**: Proyecciones se pierden al reiniciar (se recrean bajo demanda)
- **Integrado**: Algoritmos y ML nativos en la misma plataforma

---

### 1.2 MillenniumDB: Proyecciones Persistentes en Disco

```
┌─────────────────────────────────────────────────────────────┐
│              MillenniumDB Current Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌─────────────────────┐      │
│  │  Main Graph      │         │   Projections       │      │
│  │  (Disk B+Trees)  │────────>│   (Disk B+Trees)    │      │
│  │                  │ PROJECT │                     │      │
│  │ - node_id        │         │ - node_id           │      │
│  │ - from_to_edge   │         │ - from_to_edge      │      │
│  │ - node_label     │         │ - node_label*       │      │
│  │ - node_key_value │         │ - node_key_value*   │      │
│  └──────────────────┘         │ (* if INCLUDE)      │      │
│                                └──────────┬──────────┘      │
│                                           │                 │
│                                  ┌────────v────────┐        │
│                                  │   USE GRAPH     │        │
│                                  │ (Index Routing) │        │
│                                  └────────┬────────┘        │
│                                           │                 │
│                                  ┌────────v────────┐        │
│                                  │  GQL Queries    │        │
│                                  │   (MATCH...)    │        │
│                                  └─────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ❌ MISSING: Analytics Layer                  │  │
│  │  - No graph algorithms                                │  │
│  │  - No embeddings                                      │  │
│  │  - No ML pipelines                                    │  │
│  │  - No GNN support                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Características clave**:
- **Unificado OLTP/OLAP**: Misma estructura B+Tree para ambos casos
- **Persistente**: Proyecciones sobreviven reinicios del servidor
- **Subgrafo filtrado**: Solo contiene nodos/aristas del MATCH
- **Sin capa analítica**: Solo almacenamiento y consulta básica

---

## 2. Análisis Detallado por Componente

### 2.1 Proyecciones de Grafos

| Aspecto | Neo4j GDS | MillenniumDB | Brecha |
|---------|-----------|--------------|--------|
| **Ubicación** | Memoria RAM | Disco (B+Trees) | MillenniumDB más lento pero persistente |
| **Formato** | Comprimido, arrays de adyacencia | B+Trees estándar | GDS optimizado para recorridos |
| **Volatilidad** | Volátil (se pierde al reiniciar) | Persistente | ⭐ **Ventaja MDB**: No re-crear |
| **Creación** | Nativa (rápida, API interna) o Cypher (flexible) | Solo vía MATCH+PROJECT | GDS tiene 2 métodos |
| **Tamaño límite** | Limitado por RAM disponible | Limitado por disco | MDB escala mejor |
| **Topología virtual** | ✅ Puede crear relaciones virtuales (collapse paths) | ❌ Solo copia lo que existe | **Brecha crítica** |
| **Agregación** | ✅ SUM, AVG, COUNT, MIN, MAX sobre relaciones paralelas | ❌ Sin soporte | **Brecha crítica** |
| **Orientación** | ✅ NATURAL, REVERSE, UNDIRECTED | ❌ Solo dirección original | **Brecha crítica** |
| **Filtrado** | ✅ Por propiedades (Cypher projection) | ✅ Por patrón MATCH | Similar |
| **Múltiples proyecciones** | ✅ Catálogo con múltiples grafos | ✅ Múltiples proyecciones | Similar |
| **Catálogo** | ✅ gds.graph.list(), exists() | ⚠️ Manual (filesystem) | **Brecha menor** |

**Conclusión 2.1**: MillenniumDB tiene la infraestructura básica de proyecciones, pero le faltan **transformaciones topológicas** (agregación, orientación, relaciones virtuales) que son esenciales para modelado analítico.

---

### 2.2 Algoritmos de Grafos

| Categoría | Neo4j GDS | MillenniumDB | Brecha |
|-----------|-----------|--------------|--------|
| **Centralidad** | ✅ PageRank, Betweenness, Closeness, Degree, Eigenvector, ArticleRank | ❌ Ninguno | **Brecha total** |
| **Comunidades** | ✅ Louvain, Label Propagation, Weakly Connected Components, Triangle Count, Local Clustering Coefficient | ❌ Ninguno | **Brecha total** |
| **Path Finding** | ✅ Shortest Path, All Shortest Paths, A*, Yen's, Dijkstra | ❌ Solo búsquedas básicas GQL | **Brecha mayor** |
| **Similitud** | ✅ Node Similarity, Cosine, Jaccard, Overlap, Pearson, Common Neighbors, Adamic-Adar | ❌ Ninguno | **Brecha total** |
| **Link Prediction** | ✅ Preferential Attachment, Resource Allocation, Total Neighbors | ❌ Ninguno | **Brecha total** |

**Conclusión 2.2**: **Brecha del 100%**. MillenniumDB no tiene ningún algoritmo de grafos implementado. Este es el componente más crítico para análisis avanzado.

---

### 2.3 Embeddings de Nodos/Aristas

| Algoritmo | Neo4j GDS | MillenniumDB | Descripción | Uso Principal |
|-----------|-----------|--------------|-------------|---------------|
| **FastRP** | ✅ Nativo, muy rápido | ❌ No existe | Random Projection para embeddings rápidos | Feature engineering, clasificación |
| **Node2Vec** | ✅ Nativo | ❌ No existe | Random walks + Skip-gram | Link prediction, clasificación |
| **GraphSAGE** | ✅ Nativo, inductivo | ❌ No existe | Agregación de vecinos con aprendizaje | GNN, nuevos nodos |
| **HashGNN** | ✅ Nativo, sin entrenar | ❌ No existe | MinHash para features topológicas | Feature engineering sin GPU |

**Tabla de Características de Embeddings**:

| Característica | FastRP | Node2Vec | GraphSAGE | HashGNN |
|----------------|--------|----------|-----------|---------|
| **Velocidad** | ⚡⚡⚡ Muy rápido | ⚡ Lento | ⚡⚡ Medio | ⚡⚡⚡ Muy rápido |
| **Escalabilidad** | Millones de nodos | ~100k nodos | Millones | Millones |
| **Inductivo** | ❌ Transductivo | ❌ Transductivo | ✅ Inductivo | ⚠️ Semi-inductivo |
| **Necesita entrenamiento** | ❌ No | ⚠️ Sí (walks) | ✅ Sí (supervised) | ❌ No |
| **Hardware** | CPU | CPU | CPU (GDS) o GPU (PyG) | CPU |
| **Captura propiedades** | ❌ Solo topología | ❌ Solo topología | ✅ Sí | ⚠️ Hash de propiedades |

**Conclusión 2.3**: **Brecha del 100%**. Embeddings son esenciales para ML sobre grafos. Sin ellos, no se pueden usar clasificadores tradicionales (Logistic Regression, Random Forest) ni GNN avanzadas.

---

### 2.4 Redes Neuronales de Grafos (GNN)

#### 2.4.1 Enfoque Nativo (dentro del DBMS)

| Aspecto | Neo4j GDS | MillenniumDB | Brecha |
|---------|-----------|--------------|--------|
| **GraphSAGE** | ✅ Implementado en Java/GDS | ❌ No existe | **Crítica** |
| **HashGNN** | ✅ Implementado (sin entrenamiento) | ❌ No existe | **Crítica** |
| **Arquitectura** | CPU-optimizada, multi-core | - | - |
| **Integración** | Pipelines GDS | - | - |

#### 2.4.2 Enfoque Híbrido (DBMS + Framework Externo)

**Neo4j GDS → PyTorch Geometric:**

```python
# Workflow en Neo4j GDS
from graphdatascience import GraphDataScience

# 1. Conectar a Neo4j
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 2. Crear proyección (en memoria)
G = gds.graph.project("myGraph", "Node", "REL")

# 3. Ejecutar algoritmos (feature engineering)
gds.pagerank.mutate(G, mutateProperty="pagerank")
gds.fastRP.mutate(G, embeddingDimension=64, mutateProperty="embedding")

# 4. Exportar a PyTorch Geometric
node_df = gds.graph.nodeProperties.stream(G, ["pagerank", "embedding"])
edge_df = gds.graph.relationships.stream(G)

# 5. Transformar a formato PyG
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([edge_df.sourceNodeId, edge_df.targetNodeId])
x = torch.tensor(node_df["embedding"].tolist())
y = torch.tensor(node_df["label"].tolist())

data = Data(x=x, edge_index=edge_index, y=y)

# 6. Entrenar GNN custom (ej: GAT, GCN)
# ... código PyTorch estándar ...
```

**MillenniumDB → PyTorch Geometric:**
```python
# Workflow hipotético en MillenniumDB (NO EXISTE AÚN)
# ❌ No hay cliente Python oficial
# ❌ No hay API para streaming de datos
# ❌ No hay algoritmos para feature engineering

# Lo que tendrías que hacer HOY:
# 1. Ejecutar GQL query manualmente
# 2. Parsear CSV/JSON
# 3. Convertir IDs manualmente
# 4. Sin features topológicas (PageRank, etc.)
# 5. Entrenar GNN básica sin contexto estructural

# Resultado: GNN de baja calidad por falta de features
```

**Conclusión 2.4**: **Brecha del 100%** en ambos enfoques (nativo e híbrido). No hay forma práctica de entrenar GNN con MillenniumDB hoy.

---

### 2.5 Pipelines de Machine Learning

#### Neo4j GDS: Pipeline de Clasificación de Nodos

```cypher
// 1. Crear pipeline
CALL gds.beta.pipeline.nodeClassification.create('my-pipe')

// 2. Agregar pasos de feature engineering
CALL gds.beta.pipeline.nodeClassification.addNodeProperty('my-pipe', 'fastRP', {
  embeddingDimension: 128,
  iterationWeights: [0.8, 1, 1, 1]
})

CALL gds.beta.pipeline.nodeClassification.addNodeProperty('my-pipe', 'pageRank', {
  maxIterations: 20
})

// 3. Seleccionar features
CALL gds.beta.pipeline.nodeClassification.selectFeatures('my-pipe', [
  'fastRP', 'pageRank', 'age', 'numFollowers'
])

// 4. Configurar modelos candidatos (auto-tuning)
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('my-pipe', {
  penalty: 0.1
})

CALL gds.beta.pipeline.nodeClassification.addRandomForest('my-pipe', {
  numberOfDecisionTrees: 10
})

// 5. Entrenar (con cross-validation automática)
CALL gds.beta.pipeline.nodeClassification.train('myGraph', {
  pipeline: 'my-pipe',
  targetProperty: 'label',
  modelName: 'my-model',
  metrics: ['F1_WEIGHTED', 'ACCURACY']
})

// 6. Predecir
CALL gds.beta.pipeline.nodeClassification.predict.mutate('myGraph', {
  modelName: 'my-model',
  mutateProperty: 'predictedLabel'
})
```

**MillenniumDB: No existe nada equivalente**

| Componente | Neo4j GDS | MillenniumDB |
|------------|-----------|--------------|
| **Pipeline DSL** | ✅ Cypher procedures | ❌ No existe |
| **Feature engineering** | ✅ Algoritmos integrados | ❌ Manual |
| **Auto-tuning** | ✅ Grid search automático | ❌ No existe |
| **Cross-validation** | ✅ Train/test split automático | ❌ Manual |
| **Modelos ML** | ✅ Logistic Regression, Random Forest | ❌ No existe |
| **Evaluación** | ✅ Métricas automáticas | ❌ Manual |

**Conclusión 2.5**: **Brecha del 100%**. Los pipelines de GDS automatizan todo el flujo de ML, desde feature engineering hasta evaluación. MillenniumDB requeriría exportar datos y usar herramientas externas (scikit-learn, PyTorch).

---

### 2.6 Transformaciones de Grafos

| Transformación | Neo4j GDS | MillenniumDB | Importancia |
|----------------|-----------|--------------|-------------|
| **Filtrado por propiedades** | ✅ gds.graph.filter() | ❌ Solo via MATCH inicial | Media |
| **Subgrafo por BFS/DFS** | ✅ gds.graph.sample.rwr() (Random Walk) | ❌ No existe | Alta (para escalabilidad) |
| **Agregar relaciones paralelas** | ✅ aggregation: SUM, COUNT, etc. | ❌ No existe | **Crítica** para multigrafos |
| **Cambiar orientación** | ✅ orientation: REVERSE, UNDIRECTED | ❌ No existe | **Crítica** para algoritmos |
| **Colapsar caminos** | ✅ Via Cypher projection | ❌ No existe | Alta para reducir dimensionalidad |
| **Crear grafo bipartito** | ✅ Via nodeFilter en gds.graph.filter() | ⚠️ Posible con MATCH cuidadoso | Media |

**Ejemplo: Orientación UNDIRECTED**

Muchos algoritmos de comunidades (Louvain, Label Propagation) **requieren** grafos no dirigidos. En Neo4j:

```cypher
// Crear proyección con aristas no dirigidas
CALL gds.graph.project(
  'undirectedGraph',
  'Person',
  {
    KNOWS: {
      orientation: 'UNDIRECTED'  // ← Crítico
    }
  }
)

// Detectar comunidades
CALL gds.louvain.stream('undirectedGraph')
YIELD nodeId, communityId
```

En MillenniumDB **no hay forma** de hacer esto hoy. Las aristas siempre mantienen su dirección original.

**Conclusión 2.6**: **Brecha crítica**. Sin transformaciones topológicas, muchos algoritmos son imposibles de implementar correctamente.

---

### 2.7 Gestión de Memoria y Escalabilidad

| Aspecto | Neo4j GDS | MillenniumDB |
|---------|-----------|--------------|
| **Memoria para proyecciones** | Configurable (heap JVM) | No aplica (disco) |
| **Estimación de memoria** | ✅ gds.graph.create.estimate() | ❌ No existe |
| **Liberación de memoria** | ✅ gds.graph.drop() | ⚠️ Borrar directorio manual |
| **Muestreo para escalabilidad** | ✅ Random Walk with Restarts | ❌ No existe |
| **Streaming incremental** | ✅ gds.*.stream() | ❌ No existe (resultados completos) |
| **Límite de escala** | RAM disponible (~100M nodos típico) | Disco disponible (>>1B nodos posible) |

**Ventaja de MillenniumDB**: Al usar disco, puede manejar grafos **mucho más grandes** que la RAM disponible. Neo4j GDS está limitado por la memoria.

**Desventaja de MillenniumDB**: Sin muestreo inteligente, ejecutar algoritmos sobre grafos masivos sería muy lento.

---

## 3. Brechas Críticas Identificadas

### 3.1 Tabla de Prioridades

| # | Brecha | Impacto en Capacidades Analíticas | Prioridad | Complejidad |
|---|--------|----------------------------------|-----------|-------------|
| **1** | **Sin algoritmos de grafos** (PageRank, Comunidades, etc.) | 🔴 **BLOQUEANTE** - No se puede hacer ciencia de datos | **CRÍTICA** | Alta (6-12 meses) |
| **2** | **Sin embeddings** (FastRP, Node2Vec, GraphSAGE) | 🔴 **BLOQUEANTE** - No se puede entrenar ML | **CRÍTICA** | Alta (6-12 meses) |
| **3** | **Sin transformaciones topológicas** (agregación, orientación) | 🔴 **BLOQUEANTE** - Muchos algoritmos no funcionarán | **CRÍTICA** | Media (2-4 meses) |
| **4** | **Sin cliente Python oficial** | 🟠 Dificulta integración con PyTorch/TF | **ALTA** | Media (2-3 meses) |
| **5** | **Sin API de streaming** | 🟠 Dificulta exportar datos para ML externo | **ALTA** | Media (1-2 meses) |
| **6** | **Sin pipelines de ML** | 🟡 Se puede hacer manualmente | **MEDIA** | Alta (4-6 meses) |
| **7** | **Sin muestreo de subgrafos** | 🟡 Limita escalabilidad de algoritmos | **MEDIA** | Media (2-3 meses) |
| **8** | **Sin filtrado post-proyección** | 🟢 Workaround: MATCH más específico | **BAJA** | Baja (1 semana) |
| **9** | **Sin catálogo programático** | 🟢 Workaround: ls proyecciones/ | **BAJA** | Baja (1-2 semanas) |

---

### 3.2 Análisis de Dependencias

```
Capacidades Analíticas Deseadas:
│
├─ 🎯 Entrenar GNN
│  ├─ Requiere: Embeddings (brecha #2)
│  ├─ Requiere: Cliente Python (brecha #4)
│  ├─ Requiere: API streaming (brecha #5)
│  └─ Opcional: Algoritmos para features (brecha #1)
│
├─ 🎯 Clasificación de Nodos
│  ├─ Requiere: Embeddings (brecha #2)
│  ├─ Requiere: Algoritmos para features (brecha #1)
│  ├─ Opcional: Pipelines ML (brecha #6)
│  └─ Opcional: Cliente Python (brecha #4)
│
├─ 🎯 Predicción de Enlaces
│  ├─ Requiere: Embeddings (brecha #2) o Similitud (brecha #1)
│  ├─ Opcional: Pipelines ML (brecha #6)
│  └─ Opcional: Cliente Python (brecha #4)
│
├─ 🎯 Detección de Comunidades
│  ├─ Requiere: Algoritmos (brecha #1)
│  ├─ Requiere: Orientación UNDIRECTED (brecha #3)
│  └─ Opcional: Muestreo (brecha #7)
│
└─ 🎯 PageRank / Centralidad
   ├─ Requiere: Algoritmos (brecha #1)
   └─ Opcional: Agregación de aristas (brecha #3)
```

**Conclusión**: Para **cualquier** capacidad analítica avanzada, se necesitan como mínimo las brechas #1, #2 y #3.

---

## 4. Roadmap de Implementación

### Fase 0: Fundamentos (2-3 meses) - **Prerequisitos**

#### 4.1 Cliente Python Oficial
**Objetivo**: Habilitar integración con ecosistema PyData

```python
# Ejemplo de API deseada
from millenniumdb import MillenniumDB

# Conectar
mdb = MillenniumDB("http://localhost:1234")

# Ejecutar query
result = mdb.gql("MATCH (n:User) RETURN n.name, n.age")
df = result.to_dataframe()  # Pandas DataFrame

# Crear proyección
mdb.gql('MATCH (n)-[e]->(m) RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)')

# Cambiar a proyección
result = mdb.gql('USE "test" MATCH (n) RETURN count(n)')
```

**Componentes**:
- HTTP client con sesiones
- Conversión a Pandas/Polars
- Manejo de tipos GQL → Python
- Soporte para streaming (próxima fase)

**Esfuerzo**: 2-3 meses, 1 dev

---

#### 4.2 API de Streaming
**Objetivo**: Exportar datos eficientemente para ML externo

```python
# Streaming de nodos
for node in mdb.stream_nodes("test", properties=["age", "name"]):
    print(node)  # {"id": 123, "age": 30, "name": "Alice"}

# Streaming de aristas
for edge in mdb.stream_edges("test"):
    print(edge)  # {"from": 123, "to": 456, "type": "FRIEND"}

# Exportar para PyTorch Geometric
from millenniumdb.export import to_pytorch_geometric

data = to_pytorch_geometric(
    projection="test",
    node_features=["age", "pagerank"],  # pagerank vendría de algoritmo
    node_labels="label"
)
# data.x = tensor de features
# data.edge_index = tensor de aristas
# data.y = tensor de labels
```

**Componentes**:
- Cursores sobre B+Trees
- Buffering eficiente
- Conversión a formatos estándar (PyG, DGL)
- Mapeo de IDs (neo4j_id → 0-indexed)

**Esfuerzo**: 1-2 meses, 1 dev

---

### Fase 1: Transformaciones Topológicas (2-4 meses) - **Crítico**

#### 4.3 Agregación de Relaciones Paralelas
**Problema**: Multigrafos (múltiples aristas entre mismo par de nodos) son comunes en el mundo real.

**Solución**: Extender sintaxis de PROJECT

```gql
// Sumar pesos de transacciones
MATCH (u:User)-[t:TRANSACTION]->(v:User)
RETURN PROJECT("aggregated" INCLUDE PROPERTIES, {
  edges: {
    aggregation: "SUM",
    property: "amount",
    output: "total_amount"
  }
})

// Contar aristas paralelas
MATCH (u)-[e]->(v)
RETURN PROJECT("counted" INCLUDE LABELS, {
  edges: {
    aggregation: "COUNT",
    output: "num_connections"
  }
})
```

**Implementación**:
1. Extender `ProjectionOptions` con `AggregationConfig`
2. Modificar `AggProject::process()` para detectar aristas duplicadas
3. Aplicar función de agregación (sum, count, min, max, avg)
4. Escribir arista agregada a proyección

**Esfuerzo**: 3-4 semanas, 1 dev

---

#### 4.4 Orientación de Aristas
**Problema**: Algoritmos de comunidades requieren grafos no dirigidos.

**Solución**: Extender sintaxis de PROJECT

```gql
// Crear grafo no dirigido
MATCH (u)-[e]->(v)
RETURN PROJECT("undirected" INCLUDE LABELS INCLUDE PROPERTIES, {
  edges: {
    orientation: "UNDIRECTED"  // NATURAL (default), REVERSE, UNDIRECTED
  }
})

// Invertir direcciones (útil para "incoming PageRank")
MATCH (u)-[e]->(v)
RETURN PROJECT("reversed" INCLUDE LABELS, {
  edges: {
    orientation: "REVERSE"
  }
})
```

**Implementación**:
1. Agregar campo `orientation` a `ProjectionOptions`
2. En `AggProject::process()`:
   - NATURAL: escribir arista como está
   - REVERSE: swap(from, to) antes de escribir
   - UNDIRECTED: escribir en ambas direcciones (from→to y to→from)
3. Actualizar `ProjectedEdge` con flag de orientación

**Esfuerzo**: 2-3 semanas, 1 dev

---

#### 4.5 Colapso de Caminos (Relaciones Virtuales)
**Problema**: Algunos análisis requieren crear aristas que no existen explícitamente.

**Ejemplo**: Red de coautores a partir de (Autor)-[:ESCRIBIÓ]->(Paper)

**Solución**: Sintaxis similar a Cypher projection de Neo4j

```gql
// Crear red de coautores (2-hop collapse)
MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
WHERE a1 != a2
WITH a1, a2, count(p) as papers_together
RETURN PROJECT("coauthors", INCLUDE PROPERTIES, {
  edges: {
    virtual: true,
    type: "COAUTHOR",
    property: {
      name: "papers",
      value: papers_together
    }
  }
})
```

**Implementación**:
1. Detectar patrón multi-hop en MATCH
2. Extraer nodos terminales (a1, a2)
3. Crear arista virtual entre ellos
4. Opcional: agregar propiedades calculadas (como `papers_together`)

**Esfuerzo**: 4-6 semanas, 1 dev

---

### Fase 2: Algoritmos de Grafos Básicos (6-9 meses) - **Crítico**

#### 4.6 PageRank
**Algoritmo más importante** para centralidad en grafos web.

**Sintaxis propuesta**:

```gql
// Ejecutar PageRank en proyección
USE "myGraph"
CALL mdb.pagerank(
  dampingFactor: 0.85,
  maxIterations: 20,
  tolerance: 0.0001
) YIELD nodeId, score

// Guardar resultado como propiedad
MUTATE PROPERTY pagerank_score = score
```

**Implementación**:
- Iteración potencia sobre matriz de adyacencia
- Puede reusar B+Trees existentes para acceso a vecindad
- Resultado: agregar propiedad `pagerank_score` a cada nodo

**Complejidad**: O(|E| * iterations)

**Esfuerzo**: 4-6 semanas, 1 dev (con background en algoritmos de grafos)

---

#### 4.7 Louvain (Detección de Comunidades)
**Algoritmo más popular** para encontrar módulos en grafos.

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.louvain(
  maxLevels: 10,
  tolerance: 0.0001
) YIELD nodeId, communityId, modularity

MUTATE PROPERTY community = communityId
```

**Implementación**:
- Fase 1: Asignación local (mover nodos entre comunidades)
- Fase 2: Agregación (crear super-nodos)
- Repetir hasta convergencia
- **Requiere**: Grafo no dirigido (orientación UNDIRECTED)

**Esfuerzo**: 6-8 semanas, 1 dev

---

#### 4.8 Weakly Connected Components
**Más simple** que Louvain, identifica componentes desconectadas.

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.wcc() YIELD nodeId, componentId

MUTATE PROPERTY component = componentId
```

**Implementación**:
- Union-Find sobre nodos
- BFS/DFS desde cada nodo no visitado

**Esfuerzo**: 2-3 semanas, 1 dev

---

#### 4.9 Betweenness Centrality
Mide cuántos caminos más cortos pasan por un nodo.

**Sintaxis**:

```gql
USE "myGraph"
CALL mdb.betweenness() YIELD nodeId, score

MUTATE PROPERTY betweenness = score
```

**Implementación**:
- Algoritmo de Brandes
- O(|V||E|) para grafos no ponderados
- O(|V||E| + |V|²log|V|) para ponderados

**Esfuerzo**: 4-6 semanas, 1 dev

---

### Fase 3: Embeddings (6-12 meses) - **Crítico**

#### 4.10 FastRP (Fast Random Projection)
**Más rápido** de todos los embeddings, sin entrenamiento.

**Algoritmo**:
1. Inicializar vectores aleatorios para cada nodo
2. Para cada iteración:
   - Nuevo vector = promedio ponderado de vecinos
3. Normalizar y reducir dimensionalidad

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.fastRP(
  embeddingDimension: 128,
  iterations: 4,
  normalizationStrength: 0.5
) YIELD nodeId, embedding

// Guardar como propiedad de tipo tensor
MUTATE PROPERTY embedding = embedding
```

**Implementación**:
- Vectores como arrays de floats
- Operaciones de álgebra lineal (suma, normalización)
- **Nuevo tipo**: ObjectId para tensores/arrays (ya existe soporte parcial)

**Esfuerzo**: 6-8 semanas, 1-2 devs

---

#### 4.11 Node2Vec
**Más complejo**, basado en random walks y word2vec.

**Algoritmo**:
1. Generar random walks (BFS/DFS controlado)
2. Entrenar Skip-gram sobre secuencias de walks
3. Embeddings = vectores latentes de Skip-gram

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.node2vec(
  embeddingDimension: 128,
  walkLength: 80,
  walksPerNode: 10,
  p: 1.0,  // return parameter
  q: 1.0   // in-out parameter
) YIELD nodeId, embedding

MUTATE PROPERTY embedding = embedding
```

**Implementación**:
- Random walk generator
- Skip-gram trainer (Word2Vec)
- **Dependencia**: Librería de ML (considerar usar ONNX Runtime o similar)

**Esfuerzo**: 8-12 semanas, 2 devs

---

#### 4.12 GraphSAGE (inductivo, con aprendizaje)
**Más avanzado**, permite generar embeddings para nodos nuevos.

**Algoritmo**:
1. Muestrear vecindario de cada nodo (k-hop)
2. Agregar features de vecinos (MEAN, POOL, LSTM)
3. Entrenar función de agregación supervisadamente

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.graphsage.train(
  embeddingDimension: 128,
  aggregator: "MEAN",  // MEAN, POOL, LSTM
  sampleSizes: [25, 10],  // 2-hop: 25 vecinos de nivel 1, 10 de nivel 2
  epochs: 10,
  learningRate: 0.01,
  batchSize: 512,
  nodeFeatures: ["age", "pagerank"],
  nodeLabel: "label"
) YIELD modelId

// Generar embeddings con modelo entrenado
CALL mdb.graphsage.predict(
  modelId: modelId
) YIELD nodeId, embedding

MUTATE PROPERTY embedding = embedding
```

**Implementación**:
- Muestreador de vecindad (BFS)
- Red neuronal (2-3 capas)
- Backpropagation y optimización (Adam)
- **Dependencia**: Framework de DL (TensorFlow Lite, ONNX, o PyTorch via C++ API)

**Esfuerzo**: 12-16 semanas, 2-3 devs con experiencia en ML

---

#### 4.13 HashGNN (sin entrenamiento, ultra-rápido)
**Alternativa** a GraphSAGE sin necesidad de entrenamiento.

**Algoritmo**:
1. Hashear features de cada nodo (MinHash)
2. Para cada iteración:
   - Agregar hashes de vecinos
   - Re-hashear
3. Vector final = signature MinHash

**Sintaxis propuesta**:

```gql
USE "myGraph"
CALL mdb.hashgnn(
  embeddingDimension: 128,
  iterations: 3,
  neighborInfluence: 1.0,
  nodeFeatures: ["age", "country"]
) YIELD nodeId, embedding

MUTATE PROPERTY embedding = embedding
```

**Implementación**:
- MinHash (LSH - Locality Sensitive Hashing)
- Más simple que GraphSAGE (sin entrenamiento)
- Muy rápido y escalable

**Esfuerzo**: 6-8 semanas, 1 dev

---

### Fase 4: Pipelines de ML (4-6 meses) - **Deseable**

#### 4.14 Pipeline de Clasificación de Nodos

**Sintaxis propuesta** (inspirada en Neo4j):

```gql
// 1. Crear pipeline
CALL mdb.pipeline.create("node-classification", "myPipeline")

// 2. Agregar pasos de feature engineering
CALL mdb.pipeline.addStep("myPipeline", "mdb.pagerank", {
  outputProperty: "pagerank"
})

CALL mdb.pipeline.addStep("myPipeline", "mdb.fastRP", {
  embeddingDimension: 128,
  outputProperty: "embedding"
})

// 3. Seleccionar features
CALL mdb.pipeline.selectFeatures("myPipeline", [
  "pagerank", "embedding", "age"
])

// 4. Configurar modelo
CALL mdb.pipeline.addModel("myPipeline", "logistic-regression", {
  penalty: 0.1
})

// 5. Entrenar
USE "myGraph"
CALL mdb.pipeline.train("myPipeline", {
  targetProperty: "label",
  trainFraction: 0.8,
  validationFolds: 5
}) YIELD modelId, accuracy, f1Score

// 6. Predecir
CALL mdb.pipeline.predict(modelId) YIELD nodeId, predictedLabel, probability

MUTATE PROPERTY predicted_label = predictedLabel
```

**Componentes**:
- Pipeline orchestrator
- Feature cache (resultados intermedios)
- Train/test split
- Cross-validation
- Modelos: Logistic Regression, Random Forest
- Evaluación: Accuracy, Precision, Recall, F1

**Esfuerzo**: 4-6 meses, 2 devs

---

### Fase 5: Integración GNN Externa (2-3 meses) - **Deseable**

#### 4.15 Exportación a PyTorch Geometric

**API Python**:

```python
from millenniumdb.export import to_pytorch_geometric

# Ejecutar feature engineering en MDB
mdb.gql('USE "myGraph"')
mdb.gql('CALL mdb.pagerank() MUTATE PROPERTY pagerank = score')
mdb.gql('CALL mdb.fastRP() MUTATE PROPERTY embedding = embedding')

# Exportar a PyG
data = to_pytorch_geometric(
    projection="myGraph",
    node_features=["pagerank", "embedding", "age"],
    node_labels="label",
    train_mask_property="is_train"  # boolean property
)

# Entrenar GNN custom
import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(130, 64)  # 128 (embedding) + 1 (pagerank) + 1 (age)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN()
# ... entrenamiento estándar PyTorch ...

# Importar predicciones de vuelta a MDB
predictions = model(data.x, data.edge_index).argmax(dim=1)
mdb.import_node_property(
    projection="myGraph",
    property_name="gnn_prediction",
    values=predictions.tolist()
)
```

**Componentes**:
- Exportador a PyG Data format
- Importador de resultados
- Mapeo bidireccional de IDs

**Esfuerzo**: 2-3 meses, 1 dev

---

## 5. Ventajas Competitivas de MillenniumDB

A pesar de las brechas, MillenniumDB tiene **ventajas únicas** que Neo4j GDS no ofrece:

### 5.1 Persistencia de Proyecciones ⭐⭐⭐

| Aspecto | Neo4j GDS | MillenniumDB |
|---------|-----------|--------------|
| **Volatilidad** | Proyecciones se pierden al reiniciar | Proyecciones persistentes en disco |
| **Re-creación** | Debe re-proyectar desde cero cada vez | Proyecciones disponibles inmediatamente |
| **Tiempo de startup** | Minutos-horas para grafos grandes | Segundos (solo abrir B+Trees) |
| **Flujo de trabajo** | Dev: proyecto → analizo → reinicio → re-proyecto | Dev: proyecto una vez → analizo infinitas veces |

**Caso de uso**: Análisis iterativo sobre el mismo subgrafo.

```bash
# Neo4j GDS (cada vez que reinicia el servidor):
CALL gds.graph.drop('myGraph')  // Se perdió al reiniciar
CALL gds.graph.project('myGraph', ...) // Re-crear (lento)
CALL gds.pagerank.stream('myGraph')

# MillenniumDB (después de reiniciar):
USE "myGraph"  // Proyección ya existe, carga instantánea
CALL mdb.pagerank()  // Ejecutar directamente
```

---

### 5.2 Escalabilidad por Disco ⭐⭐

| Aspecto | Neo4j GDS | MillenniumDB |
|---------|-----------|--------------|
| **Límite de tamaño** | RAM disponible (típico: 50-200GB) | Disco disponible (típico: 1-10TB) |
| **Grafos masivos** | ~100M nodos (con 128GB RAM) | >>1B nodos posible |
| **Costo de hardware** | Servidores con mucha RAM ($$$) | Servidores con discos grandes ($) |

**Caso de uso**: Grafos de conocimiento masivos, redes sociales completas.

---

### 5.3 Modelo de Datos Unificado ⭐

| Aspecto | Neo4j GDS | MillenniumDB |
|---------|-----------|--------------|
| **OLTP y OLAP** | Separados (DB en disco, proyecciones en RAM) | Unificado (todo en B+Trees) |
| **Sincronización** | Proyecciones se desactualizan | Proyecciones son snapshots explícitos |
| **Consistencia** | No garantizada entre DB y proyección | Snapshot consistente al momento del PROJECT |

**Caso de uso**: Análisis sobre snapshots históricos (ej: grafo del 2023-01-01).

---

### 5.4 No Requiere Licencia Enterprise ⭐

| Aspecto | Neo4j | MillenniumDB |
|---------|-------|--------------|
| **Concurrency limit** | Community: 4 cores, Enterprise: ilimitado | Sin límite artificial |
| **Costo** | Enterprise: ~$200k+/año para producción | Open source, sin costo |

---

## 6. Estimación de Esfuerzo

### 6.1 Tabla de Resumen

| Fase | Componentes | Prioridad | Esfuerzo (dev-months) | Bloqueantes |
|------|-------------|-----------|----------------------|-------------|
| **0. Fundamentos** | Cliente Python, API streaming | CRÍTICA | 3-5 | Ninguno |
| **1. Transformaciones** | Agregación, Orientación, Colapso de caminos | CRÍTICA | 3-5 | Fase 0 (para testing) |
| **2. Algoritmos Básicos** | PageRank, Louvain, WCC, Betweenness | CRÍTICA | 8-12 | Fase 1 (orientación) |
| **3. Embeddings** | FastRP, Node2Vec, GraphSAGE, HashGNN | CRÍTICA | 10-16 | Fases 1, 2 (para features) |
| **4. Pipelines ML** | Node Classification, Link Prediction | ALTA | 6-10 | Fases 2, 3 |
| **5. Integración GNN** | Exportación PyG, Importación resultados | ALTA | 2-3 | Fase 0 |

**Total Mínimo Viable (Fases 0-3)**: **24-38 dev-months** (2-3 devs durante 12-18 meses)

**Total Completo (Fases 0-5)**: **32-51 dev-months** (2-4 devs durante 18-24 meses)

---

### 6.2 Equipo Recomendado

| Rol | Responsabilidades | Fases |
|-----|-------------------|-------|
| **Backend Dev 1** | Cliente Python, API streaming, transformaciones | 0, 1 |
| **Graph Algorithms Dev** | PageRank, Louvain, WCC, Betweenness | 2 |
| **ML Engineer** | Embeddings (FastRP, Node2Vec, GraphSAGE, HashGNN) | 3 |
| **ML Engineer 2** | Pipelines ML, integración PyG | 4, 5 |

**Timeline con 3 devs**:
- Meses 1-3: Fase 0 (Dev 1)
- Meses 4-6: Fase 1 (Dev 1) + Fase 2 inicio (Algorithms Dev)
- Meses 7-12: Fase 2 continúa (Algorithms Dev) + Fase 3 (ML Engineer)
- Meses 13-18: Fase 3 continúa (ML Engineer) + Fase 4 (ML Engineer 2)
- Meses 19-21: Fase 5 (ML Engineer 2) + refinamientos

---

### 6.3 Hitos de Validación

| Mes | Hito | Validación |
|-----|------|------------|
| **3** | Cliente Python funcional | Conectar, ejecutar queries, recibir DataFrames |
| **6** | Transformaciones topológicas | Crear grafo no dirigido, ejecutar algoritmo que lo requiera |
| **9** | Primer algoritmo (PageRank) | Calcular PageRank en proyección, comparar resultados con Neo4j |
| **12** | Primer embedding (FastRP) | Generar embeddings, usarlos en clasificador externo |
| **15** | GraphSAGE funcionando | Entrenar GraphSAGE, generar embeddings para nodos nuevos |
| **18** | Pipeline end-to-end | Feature eng + entrenamiento + predicción todo en MDB |
| **21** | Integración PyTorch Geometric | Exportar a PyG, entrenar GNN custom, importar resultados |

---

## 7. Estrategia de Priorización

### 7.1 Enfoque MVP (Minimum Viable Product)

Para permitir **análisis básico** lo antes posible:

**MVP 1: "Exportador Inteligente"** (4-6 meses)
- Cliente Python ✅
- API streaming ✅
- Exportación a PyG ✅
- **Resultado**: Puedes entrenar GNN externas, pero sin features topológicas

**MVP 2: "Calculadora de Features"** (+6-8 meses)
- Transformaciones (orientación, agregación) ✅
- PageRank ✅
- FastRP ✅
- **Resultado**: Puedes generar features útiles para ML externo

**MVP 3: "Plataforma Analítica Completa"** (+8-12 meses)
- Louvain, WCC, Betweenness ✅
- Node2Vec, GraphSAGE ✅
- Pipelines ML ✅
- **Resultado**: Competidor directo de Neo4j GDS

---

### 7.2 Enfoque Alternativo: "Wrapper de Neo4j GDS"

Si el objetivo es **tener capacidades analíticas rápido** (6-12 meses):

1. **Exportar de MDB a Neo4j**: Crear herramienta que convierta proyección MDB → Neo4j DB
2. **Ejecutar GDS**: Usar todas las capacidades de Neo4j GDS
3. **Re-importar a MDB**: Traer resultados de vuelta (PageRank scores, embeddings, etc.)

**Pros**:
- Rápido de implementar (solo exportador/importador)
- Aprovecha todo el ecosistema GDS
- No necesitas implementar algoritmos

**Contras**:
- Dependencia de Neo4j
- Overhead de transferencia de datos
- No aprovecha persistencia de MDB
- Requiere licencia Neo4j Enterprise para grafos grandes

**Esfuerzo**: 2-4 meses, 1-2 devs

---

## 8. Conclusiones y Recomendaciones

### 8.1 Estado Actual

✅ **Fortalezas de MillenniumDB**:
- Proyecciones persistentes en disco (único en la industria)
- Escalabilidad por disco (grafos masivos)
- Modelo de datos unificado OLTP/OLAP
- Open source sin restricciones

❌ **Brechas Críticas**:
- Sin algoritmos de grafos
- Sin embeddings
- Sin transformaciones topológicas
- Sin ecosistema de ML

### 8.2 Recomendaciones

#### Corto Plazo (6 meses) - **Fase 0 + Fase 1**
**Objetivo**: Habilitar integración con PyTorch/TensorFlow

**Acciones**:
1. Desarrollar cliente Python oficial
2. Implementar API de streaming
3. Agregar transformaciones topológicas (orientación, agregación)
4. Crear exportador a PyTorch Geometric

**Resultado**: Los usuarios pueden entrenar GNN externas usando MillenniumDB como backend de datos.

#### Mediano Plazo (12 meses) - **Fase 2**
**Objetivo**: Algoritmos básicos de grafos

**Acciones**:
1. Implementar PageRank
2. Implementar Louvain
3. Implementar WCC
4. Implementar al menos un algoritmo de similitud

**Resultado**: MillenniumDB puede generar features topológicas útiles para ML.

#### Largo Plazo (18-24 meses) - **Fases 3-5**
**Objetivo**: Plataforma analítica completa

**Acciones**:
1. Implementar FastRP y HashGNN (embeddings rápidos)
2. Implementar GraphSAGE (embeddings inductivos)
3. Crear pipelines de ML (clasificación, link prediction)
4. Madurar integración con PyTorch Geometric

**Resultado**: MillenniumDB es competidor directo de Neo4j GDS con la ventaja de persistencia.

---

### 8.3 Decisión Estratégica Clave

**¿Qué enfoque tomar?**

| Enfoque | Pros | Contras | Recomendado si... |
|---------|------|---------|-------------------|
| **A: MVP Incremental** (Fases 0→5) | Completo, independiente, aprovecha ventajas MDB | Largo (18-24 meses) | Tienes equipo y tiempo para construir plataforma completa |
| **B: Wrapper Neo4j** | Rápido (4-6 meses), aprovecha GDS existente | Dependencia externa, no aprovecha persistencia | Necesitas capacidades YA y tienes presupuesto para Neo4j |
| **C: Híbrido** | Wrapper corto plazo + MVPs largo plazo | Duplicación de esfuerzo | Necesitas demostrar valor rápido + construir plataforma propia |

**Mi Recomendación**: **Enfoque A (MVP Incremental)** con las siguientes prioridades:

1. **Mes 1-6**: Fase 0 + Fase 1 → Habilitar integración con ecosistema PyData
2. **Mes 7-12**: Fase 2 (solo PageRank + Louvain) → Demostrar capacidad analítica básica
3. **Mes 13-18**: Fase 3 (solo FastRP + HashGNN) → Habilitar embeddings rápidos
4. **Evaluar**: Con esto, ya tienes una **propuesta de valor diferenciada**:
   - Proyecciones persistentes (único)
   - Escalabilidad por disco
   - Features topológicas básicas
   - Integración con PyTorch

---

## Apéndice A: Comparativa Técnica Detallada

### A.1 Arquitectura de Almacenamiento

| Aspecto | Neo4j GDS (Memoria) | MillenniumDB (Disco) |
|---------|---------------------|----------------------|
| **Estructura** | Adjacency List comprimida | B+Trees |
| **Acceso a vecinos** | O(1) array lookup | O(log N) B+Tree search |
| **Iteración sobre aristas** | Secuencial en memoria | Secuencial con I/O disco |
| **Filtrado por propiedad** | Scan en memoria | Index scan en B+Tree |
| **Ordenamiento** | Quick/Merge sort en RAM | External sort en disco |
| **Agregación** | Hash tables en memoria | Sort-based en disco |

**Implicación**: Algoritmos en Neo4j son ~10-100x más rápidos, pero MillenniumDB puede manejar grafos ~10-100x más grandes.

---

### A.2 Tipos de Datos para ML

| Tipo | Neo4j GDS | MillenniumDB | Necesidad |
|------|-----------|--------------|-----------|
| **Escalares** | ✅ Long, Double, Boolean | ✅ INT, FLOAT, BOOL | Features simples |
| **Strings** | ✅ String | ✅ STRING | Categorías |
| **Arrays** | ✅ List<Double> | ⚠️ Soporte parcial (tensores) | Embeddings |
| **Matrices** | ❌ No nativo | ❌ No nativo | Menos común |
| **Grafos** | ✅ Nativo | ✅ Nativo | Core |

**Acción requerida**: Madurar soporte de tensores en MillenniumDB para almacenar embeddings.

---

### A.3 Operaciones de Álgebra Lineal

| Operación | Necesidad | Dónde se Usa |
|-----------|-----------|--------------|
| **Dot product** | ALTA | Similitud coseno, GNN |
| **Matrix multiply** | ALTA | GraphSAGE, GCN |
| **Normalization (L2)** | ALTA | Embeddings |
| **Softmax** | MEDIA | Clasificación multiclase |
| **Batch operations** | ALTA | Entrenamiento eficiente |

**Recomendación**: Integrar librería de álgebra lineal (Eigen, Armadillo, o BLAS) para operaciones matriciales eficientes en C++.

---

### A.4 Integración con Frameworks de ML

| Framework | Neo4j GDS | MillenniumDB (Propuesto) |
|-----------|-----------|--------------------------|
| **PyTorch** | ✅ Via Python client | 🟡 Fase 5 |
| **TensorFlow** | ✅ Via Python client | 🟡 Fase 5 |
| **PyTorch Geometric** | ✅ Exportador nativo | 🟡 Fase 5 |
| **DGL (Deep Graph Library)** | ⚠️ Manual | ⚠️ Manual |
| **scikit-learn** | ✅ Via pipelines | 🟡 Fase 4 |
| **XGBoost** | ⚠️ Manual | ⚠️ Manual |

---

## Apéndice B: Casos de Uso Específicos

### B.1 Caso de Uso: Recomendación de Productos

**Objetivo**: Dado un usuario, recomendar productos que probablemente comprará.

**Pipeline Neo4j GDS**:
```cypher
// 1. Crear proyección bipartita (User)-[:BOUGHT]->(Product)
CALL gds.graph.project('purchases', ['User', 'Product'], 'BOUGHT')

// 2. Calcular similitud de usuarios (por productos en común)
CALL gds.nodeSimilarity.mutate('purchases', {
  mutateProperty: 'similarity',
  mutateRelationshipType: 'SIMILAR_TO'
})

// 3. Predicción de enlaces (User)-[:WILL_BUY]->(Product)
CALL gds.beta.pipeline.linkPrediction.predict.stream('purchases', {
  sourceNodeFilter: 'User',
  targetNodeFilter: 'Product',
  topK: 10
}) YIELD node1, node2, probability
```

**Pipeline MillenniumDB (Hipotético después de Fases 0-4)**:
```gql
// 1. Crear proyección
MATCH (u:User)-[:BOUGHT]->(p:Product)
RETURN PROJECT("purchases" INCLUDE LABELS INCLUDE PROPERTIES)

// 2. Calcular similitud
USE "purchases"
CALL mdb.nodeSimilarity({
  sourceLabels: ["User"],
  targetLabels: ["User"]
}) MUTATE PROPERTY similarity = score

// 3. Generar embeddings
CALL mdb.graphsage.train({
  nodeFeatures: ["age", "country"],
  nodeLabel: "bought_product_X"
}) YIELD modelId

CALL mdb.graphsage.predict(modelId)
MUTATE PROPERTY embedding = embedding

// 4. Predicción de enlaces
CALL mdb.pipeline.linkPrediction.train({
  features: ["similarity", "embedding"],
  positiveExamples: "BOUGHT",
  negativeRatio: 1.0
}) YIELD modelId

CALL mdb.pipeline.linkPrediction.predict(modelId, {
  sourceLabels: ["User"],
  targetLabels: ["Product"],
  topK: 10
}) YIELD userId, productId, probability
```

---

### B.2 Caso de Uso: Detección de Fraude

**Objetivo**: Identificar cuentas bancarias involucradas en actividad fraudulenta.

**Características Útiles**:
- PageRank (centralidad en red de transacciones)
- Comunidades (anillos de fraude)
- Embeddings (comportamiento similar a casos conocidos)
- Features de grafo: grado, clustering coefficient

**Pipeline Neo4j GDS**:
```cypher
// 1. Proyección con agregación de transacciones
CALL gds.graph.project('fraud', 'Account', {
  TRANSACTION: {
    properties: {
      total_amount: {
        property: 'amount',
        aggregation: 'SUM'
      }
    }
  }
})

// 2. Features topológicas
CALL gds.pagerank.mutate('fraud', {mutateProperty: 'pagerank'})
CALL gds.louvain.mutate('fraud', {mutateProperty: 'community'})
CALL gds.degree.mutate('fraud', {mutateProperty: 'degree'})

// 3. Embeddings
CALL gds.fastRP.mutate('fraud', {
  embeddingDimension: 128,
  mutateProperty: 'embedding'
})

// 4. Clasificación (fraude vs legítimo)
CALL gds.beta.pipeline.nodeClassification.train('fraud', {
  targetProperty: 'is_fraud',
  nodeFeatures: ['pagerank', 'degree', 'embedding', 'total_amount']
}) YIELD modelId

// 5. Predecir en cuentas sin etiqueta
CALL gds.beta.pipeline.nodeClassification.predict.stream('fraud', {
  modelName: modelId
}) YIELD nodeId, predictedClass, probability
```

**Brecha MillenniumDB**: Sin Fases 2-4, esto es imposible.

---

## Apéndice C: Referencias y Recursos

### Papers Académicos
- **GraphSAGE**: "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
- **Node2Vec**: "node2vec: Scalable Feature Learning for Networks" (Grover & Leskovec, 2016)
- **Louvain**: "Fast unfolding of communities in large networks" (Blondel et al., 2008)
- **PageRank**: "The PageRank Citation Ranking: Bringing Order to the Web" (Page et al., 1999)

### Documentación Técnica
- Neo4j Graph Data Science: https://neo4j.com/docs/graph-data-science/current/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Deep Graph Library (DGL): https://docs.dgl.ai/

### Librerías Open Source
- **NetworkX** (Python): Algoritmos de grafos de referencia
- **igraph** (C/Python/R): Alto rendimiento
- **SNAP** (Stanford): Grafos masivos
- **GraphBLAS** (C): Álgebra lineal para grafos

---

**Fin del Análisis**

---

**Autores**: Equipo MillenniumDB
**Última Actualización**: 18 de Octubre, 2025
**Versión**: 1.0
