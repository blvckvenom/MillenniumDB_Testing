# Proyecciones GQL

> **Última verificación E2E:** 2026-04-20
> **Build:** MillenniumDB v1.0.0, Release, `ENABLE_GNN=ON`
> **Idioma:** español
> **Datasets usados:** `data/example/gql/books`, `posts`, `hetero_test` (enriquecido), y una fixture local `agg_example.gql` para agregación

Guía pedagógica con ejemplos ejecutables. Cada bloque `Resultado real`
reproduce lo devuelto por `mdb server` vía HTTP POST al endpoint `/query`
sobre `Linux 6.17.9-76061709-generic` (Pop!_OS 24.04 LTS).

## Tabla de contenidos

1. [Tutorial de 5 minutos](#tutorial-de-5-minutos)
2. [Requisitos previos](#requisitos-previos)
3. [Crear proyecciones — `CALL graph_project`](#crear-proyecciones-call-graph_project)
4. [Variantes sintácticas (STRING, LIST, MAP, wildcard)](#variantes-sintacticas-string-list-map-wildcard)
5. [Configuración global (4° parámetro)](#configuracion-global-4-parametro)
6. [Orientación de aristas](#orientacion-de-aristas)
7. [Agregación de aristas paralelas](#agregacion-de-aristas-paralelas)
8. [Configuración de propiedades](#configuracion-de-propiedades)
9. [Multi-label y configuración por tipo](#multi-label-y-configuracion-por-tipo)
10. [Extracción de datos GNN](#extraccion-de-datos-gnn)
11. [Campos YIELD](#campos-yield)
12. [Consultar con `USE`](#consultar-con-use)
13. [`PROJECT()` como agregado in-query](#project-como-agregado-in-query)
14. [Notas de comportamiento](#notas-de-comportamiento)
15. [Limitaciones](#limitaciones)
16. [Tuning de memoria](#tuning-de-memoria)
17. [Cómo reproducir este documento](#como-reproducir-este-documento)

---

## Tutorial de 5 minutos

Esta sección te lleva desde cero hasta tu primera proyección consultable.
Usaremos el dataset `data/example/gql/books`, que define autores (`Author`),
editores (`Editor`), libros (`Book`), lectores (`Reader`) y aristas `Knows`,
`Published`, `Collected`, `Follows`.

### Paso 1 — Importar el dataset

```bash
build/Release/bin/mdb import data/example/gql/books data/dbs/gql/books_docs
```

Comprueba que existen 24 autores:

**Query:**
```gql
MATCH (n:Author) RETURN count(*) AS authors;
```

**Resultado real:**
```
authors
24
```

### Paso 2 — Levantar el servidor

```bash
build/Release/bin/mdb server data/dbs/gql/books_docs --port 1237
```

`mdb server` acepta peticiones HTTP POST a `http://localhost:1237/query` con
`Content-Type: application/gql`. A diferencia de `mdb cli` (TUI interactivo),
el servidor sí es ejecutable desde scripts, y es el modo recomendado para
reproducir lo que sigue.

### Paso 3 — Crear tu primera proyección

```gql
CALL graph_project('autores_conocidos_t', 'Author', 'Knows')
YIELD graphName, nodeCount, relationshipCount, projectMillis RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount,projectMillis
"autores_conocidos_t",24,12,15
```

**Interpretación:** la proyección persistente `autores_conocidos_t` tiene los
24 nodos `:Author` y 12 aristas `:Knows`, creada en 15 ms. Sobrevive
reinicios del servidor.

### Paso 4 — Consultar la proyección con `USE`

```gql
USE autores_conocidos_t MATCH (n)-[e]->(m) RETURN count(*) AS edge_count;
USE autores_conocidos_t MATCH (a:Author)   RETURN a LIMIT 3;
```

**Resultado real:**
```
edge_count
12
a
_n0
_n1
_n4
```

**Interpretación:** las proyecciones se consultan como si fueran otro grafo,
anteponiendo `USE <nombre>`. En 4 comandos creaste, verificaste y
consultaste un subgrafo reutilizable para alimentar otros procedimientos
(algoritmos, GNN, exportación a `.npy`, etc.).

---

## Requisitos previos

| Requisito | Justificación |
|-----------|---------------|
| Build Release de MillenniumDB | Las proyecciones son parte del core GQL; usar Debug implica ASan/UBSan y mayor overhead |
| Modelo **GQL** únicamente | El procedimiento `graph_project` no está registrado ni expuesto en RDF ni Quad Model |
| `ENABLE_GNN=ON` en CMake | Requerido para que las claves `includeFeatures`, `labelProperty` y `splitProperty` surtan efecto. Sin este flag el parser acepta las claves pero el builder las ignora |
| `mdb server` para ejecutar desde scripts | `mdb cli` es un TUI interactivo y no es apto para automatización; el servidor expone el endpoint HTTP POST `/query` |

Compilación recomendada:

```bash
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON
cmake --build build/Release -j $(nproc)
build/Release/bin/mdb help
```

---

## Crear proyecciones — `CALL graph_project`

### Signatura

```
CALL graph_project(graphName, nodeProjection, relationshipProjection [, configuration])
YIELD graphName, nodeCount, relationshipCount, projectMillis [, featureDim, numClasses]
```

### Parámetros

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `graphName` | STRING | Sí | Nombre de la proyección a crear. No puede contener `/`, `\`, bytes nulos ni caracteres de control; no puede ser `.`, `..`, vacío ni solo whitespace |
| `nodeProjection` | STRING, LIST o MAP | Sí | Label(s) de nodos a incluir. Admite `'*'` como wildcard para todos los labels |
| `relationshipProjection` | STRING, LIST o MAP | Sí | Tipo(s) de aristas a incluir. Admite `'*'` como wildcard para todos los tipos |
| `configuration` | MAP | No | Opciones globales (ver Capítulo 5) |

El procedimiento devuelve al menos los tres contadores básicos más
`projectMillis`. `graphName` se devuelve entre comillas (STRING) y los
contadores son INTEGER. El ejemplo mínimo se muestra en la siguiente
sección (S1).

---

## Variantes sintácticas (STRING, LIST, MAP, wildcard)

`nodeProjection` y `relationshipProjection` aceptan cuatro formas
equivalentes en poder expresivo pero distintas en conveniencia.

### S1 — STRING (un único label/tipo)

```gql
CALL graph_project('s1_doc', 'Author', 'Knows')
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"s1_doc",24,12
```

**Interpretación:** la forma más directa — un único label de nodo y un
único tipo de arista.

### S2 — LIST (varios labels/tipos)

```gql
CALL graph_project('s2_doc', ['Author', 'Book'], ['Knows', 'Published'])
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"s2_doc",41,31
```

**Interpretación:** `Author ∪ Book` = 24 + 17 = 41 nodos. Las 31 aristas son
el total agregado, ya filtrado por *endpoint filtering* (ambos extremos en
la proyección).

### S3 — Wildcard `'*'` (todos los labels/tipos)

```gql
CALL graph_project('s3_doc', '*', '*')
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"s3_doc",150,175
```

**Interpretación:** `'*'` se expande a todos los labels/tipos del catálogo.
Úsalo sólo para exploración ad-hoc; implica escanear todos los índices.

### S4 — MAP (configuración por label/tipo)

```gql
CALL graph_project('s4_doc',
  {Author: {properties: ['name']}, Book: {properties: ['title']}},
  {Knows: {orientation: 'UNDIRECTED'}, Published: {orientation: 'NATURAL'}})
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"s4_doc",41,31
```

**Interpretación:** la forma MAP permite ajustar propiedades por label y
orientación/agregación por tipo en una sola llamada.

### Cuándo usar cada variante

| Variante | Úsala cuando |
|----------|--------------|
| STRING | Tienes exactamente un label y un tipo |
| LIST | Varios labels/tipos sin configuración diferenciada |
| MAP | Necesitas propiedades distintas por label o orientación/agregación por tipo |
| `'*'` | Exploración ad-hoc, o el grafo entero es realmente lo que quieres |

---

## Configuración global (4° parámetro)

El cuarto parámetro es un MAP opcional que fija *defaults* globales. Los
valores por tipo (forma MAP en `relationshipProjection`) sobrescriben estos.

| Clave | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `nodeProperties` | STRING o LIST | `[]` | Propiedades de nodo a incluir |
| `relationshipProperties` | STRING o LIST | `[]` | Propiedades de arista a incluir |
| `orientation` | STRING | `'NATURAL'` | Modo de orientación de aristas (ver Capítulo 6) |
| `aggregation` | STRING | `'SINGLE'` | Estrategia para aristas paralelas (ver Capítulo 7) |
| `aggregationProperty` | STRING | (auto) | Propiedad objetivo para MIN/MAX/SUM; si se omite usa la primera de `relationshipProperties` |
| `includeFeatures` | STRING | `''` | Nombre de una FeatureMatrix registrada (requiere ENABLE_GNN) |
| `labelProperty` | STRING | `''` | Propiedad entera con el label de clasificación (requiere ENABLE_GNN) |
| `splitProperty` | STRING | `''` | Propiedad con `"train"`/`"val"`/`"validation"`/`"test"` |

Ejemplo combinando varias claves:

```gql
CALL graph_project('combined_demo', 'Author', 'Knows', {
  orientation: 'UNDIRECTED', nodeProperties: ['name', 'age'],
  aggregation: 'COUNT'
}) YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"combined_demo",24,12
```

**Interpretación:** la combinación de cuatro claves se aplica en un solo
procedimiento. `Knows` en `books` no tiene aristas paralelas, por eso
`COUNT` no colapsa nada; el 12 corresponde a las 12 aristas originales.

---

## Orientación de aristas

Los tres modos son `NATURAL`, `REVERSE`, `UNDIRECTED`. Usamos el dataset
`posts` (50 aristas `Follows` dirigidas entre `Reader`).

### O1, O2, O3 — NATURAL / REVERSE / UNDIRECTED

```gql
CALL graph_project('o1_doc', 'Reader', 'Follows') YIELD graphName, relationshipCount RETURN *;
CALL graph_project('o2_doc', 'Reader', 'Follows', {orientation:'REVERSE'})    YIELD graphName, relationshipCount RETURN *;
CALL graph_project('o3_doc', 'Reader', 'Follows', {orientation:'UNDIRECTED'}) YIELD graphName, relationshipCount RETURN *;
```

**Resultado real (los tres):**
```
graphName,relationshipCount
"o1_doc",50
"o2_doc",50
"o3_doc",50
```

**Interpretación honesta:** los tres modos reportan `relationshipCount = 50`.
El campo cuenta aristas lógicas materializadas, no *slots* físicos. La
diferencia aparece al consultar con `USE`: `NATURAL` matchea
`(a)-[e]->(b)`; `REVERSE` matchea `(b)-[e]->(a)`; `UNDIRECTED` matchea en
ambas direcciones usando `~[e]~`.

### Overhead de storage

| Modo | Overhead sobre NATURAL | Observación |
|------|------------------------|-------------|
| `NATURAL` | 1.0× | Línea base: aristas tal cual |
| `REVERSE` | 1.0× | Misma cardinalidad física, con el par `(from, to)` invertido |
| `UNDIRECTED` | ≈ 1.75× (aristas originalmente dirigidas) | Duplica índices `from→to` y `to→from` |
| `UNDIRECTED` | 1.0× (aristas originalmente no dirigidas) | Ya son simétricas; sin overhead adicional |

---

## Agregación de aristas paralelas

### Preparación del escenario

El dataset `posts` no tiene aristas paralelas naturales. Para demostrar los
cinco modos construimos una fixture local `agg_example.gql` con 3 usuarios
(`Ana`, `Bob`, `Clara`), 2 posts (`Guia`, `Tutorial`) y 6 aristas `LIKES`
con pesos 1,3,5 (Ana→Guia), 2,4 (Bob→Guia) y 7 (Clara→Tutorial).

**Contenido literal de `agg_example.gql`** (guárdalo en `/tmp/agg_example.gql`):

```
# Agregación example fixture — parallel LIKES edges
0 :User name:"Ana"
1 :User name:"Bob"
2 :User name:"Clara"
10 :Post title:"Guia"
11 :Post title:"Tutorial"

0->10 :LIKES weight:1
0->10 :LIKES weight:3
0->10 :LIKES weight:5
1->10 :LIKES weight:2
1->10 :LIKES weight:4
2->11 :LIKES weight:7
```

Importa como cualquier otro GQL:

```bash
build/Release/bin/mdb import /tmp/agg_example.gql data/dbs/gql/agg_docs
build/Release/bin/mdb server data/dbs/gql/agg_docs --port 1237 &
```

> **Clave:** `LIKES` conecta `User` con `Post`, así que en
> `graph_project` **ambos labels deben estar en `nodeProjection`**. Si
> pones sólo `'User'`, las 6 aristas quedan filtradas por *endpoint
> filtering*.

### A1 — Aristas paralelas crudas

```gql
MATCH (u:User)-[e:LIKES]->(p:Post) RETURN u.name, p.title, e.weight;
```

**Resultado real:**
```
u.name,p.title,e.weight
"Ana","Guia",1
"Ana","Guia",3
"Ana","Guia",5
"Bob","Guia",2
"Bob","Guia",4
"Clara","Tutorial",7
```

**Interpretación:** Ana tiene 3 likes al mismo post (aristas paralelas),
Bob 2, y Clara 1.

### A2 — SINGLE: rechaza aristas paralelas (default)

```gql
CALL graph_project('a1_v2', ['User','Post'], 'LIKES')
YIELD graphName, relationshipCount RETURN *;
```

**Resultado real (error educativo):**
```
Parallel edges detected but aggregation is SINGLE (the default).
SINGLE refuses to drop data silently — choose how to collapse parallels:
  aggregation: 'COUNT'  -> emit one edge with synthetic `_count`
  aggregation: 'SUM'    -> sum `aggregationProperty` across parallels
  aggregation: 'MIN'    -> keep edge with smallest property value
  aggregation: 'MAX'    -> keep edge with largest property value

Example:
  CALL graph_project('g', nodes, edges, {aggregation: 'COUNT'})

For wildcard ('*') you almost always want COUNT because the
projection will touch every edge type, including ones with parallels.
For per-type control use the MAP form on relationshipProjection.
```

**Interpretación:** `SINGLE` es el default estricto: si detecta aristas
paralelas aborta con un mensaje educativo, nunca descarta datos
silenciosamente.

### A3 — COUNT

```gql
CALL graph_project('a2_v2', ['User','Post'], 'LIKES', {aggregation: 'COUNT'})
YIELD graphName, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,relationshipCount
"a2_v2",3
```

Consulta el `_count` sintético por par:

```gql
USE a2_wp MATCH (u:User)-[e:LIKES]->(p:Post)
RETURN u.name, p.title, e._count;
```

**Resultado real:**
```
u.name,p.title,e._count
"Ana","Guia",3
"Bob","Guia",2
"Clara","Tutorial",1
```

**Interpretación:** COUNT colapsa las 6 aristas en 3 (una por par único) y
expone el conteo como la propiedad sintética `e._count`.

### A4 — SUM (requiere `relationshipProperties`)

```gql
CALL graph_project('a3_v2', ['User','Post'], 'LIKES', {
  aggregation: 'SUM', relationshipProperties: ['weight']
})
YIELD graphName, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,relationshipCount
"a3_v2",3
```

```gql
USE a3_wp MATCH (u:User)-[e:LIKES]->(p:Post)
RETURN u.name, p.title, e.weight;
```

**Resultado real:**
```
u.name,p.title,e.weight
"Ana","Guia",9
"Bob","Guia",6
"Clara","Tutorial",7
```

**Interpretación:** SUM suma la propiedad indicada. Ana: 1+3+5=9, Bob: 2+4=6,
Clara: 7.

### A5, A6 — MIN y MAX con `aggregationProperty`

```gql
-- MIN:
CALL graph_project('a4_v2', ['User','Post'], 'LIKES',
  {aggregation:'MIN', aggregationProperty:'weight', relationshipProperties:['weight']})
  YIELD graphName, relationshipCount RETURN *;
-- MAX (misma estructura con aggregation:'MAX', proyección a5_v2):
```

**Resultado real (ambos):**
```
graphName,relationshipCount
"a4_v2",3
"a5_v2",3
```

Inspeccionando con `USE` cada proyección:

```gql
USE a4_wp MATCH (u:User)-[e:LIKES]->(p:Post) RETURN u.name, p.title, e.weight;
USE a5_wp MATCH (u:User)-[e:LIKES]->(p:Post) RETURN u.name, p.title, e.weight;
```

**Resultado real (MIN):**
```
u.name,p.title,e.weight
"Ana","Guia",1
"Bob","Guia",2
"Clara","Tutorial",7
```

**Resultado real (MAX):**
```
u.name,p.title,e.weight
"Ana","Guia",5
"Bob","Guia",4
"Clara","Tutorial",7
```

**Interpretación:** por cada par `(u,p)`, MIN retiene la arista con menor
`weight` y MAX la de mayor `weight`. El resto se descartan.

---

## Configuración de propiedades

### P1 — Lista simple de propiedades de nodo

```gql
CALL graph_project('p1_doc', {Author: {properties: ['name', 'age']}}, 'Knows')
YIELD graphName, nodeCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount
"p1_doc",24
```

```gql
USE p1_doc MATCH (n) RETURN n.name, n.age LIMIT 3;
```

**Resultado real:**
```
n.name,n.age
"John Doe",82
"Jane Doe",42
"Angela Everett",22
```

**Interpretación:** las propiedades listadas pasan a la proyección y son
queryables vía `n.<property>`.

### P2 — Renombrado de propiedades

```gql
CALL graph_project('p2_doc',
  {Author: {properties: {nombre: {property: 'name'}}}}, 'Knows')
YIELD graphName, nodeCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount
"p2_doc",24
```

```gql
USE p2_doc MATCH (n) RETURN n.nombre LIMIT 3;
```

**Resultado real:**
```
n.nombre
"John Doe"
"Jane Doe"
"Angela Everett"
```

**Interpretación:** el MAP anidado permite leer `name` del grafo base y
exponerlo como `nombre` en la proyección. El rename se resuelve en el
builder.

### P3 — Valores por defecto (`defaultValue`)

```gql
CALL graph_project('p3_doc',
  {Author: {properties: {age: {defaultValue: 0}}}}, 'Knows')
YIELD graphName, nodeCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount
"p3_doc",24
```

**Interpretación:** cuando un nodo no tenga `age` definida, la proyección
usa `0` como valor sustituto. Útil para vectorizar y evitar nulls al
entrenar GNNs.

### Agregación por propiedad (aristas)

Cada propiedad de arista puede declarar su propia `aggregation`, siempre
combinada con una `aggregation` por tipo (de lo contrario el default
`SINGLE` rechaza los paralelos antes de llegar a la fase per-propiedad).
Ejemplo sobre `agg_docs`:

```gql
CALL graph_project('per_prop_v3', ['User','Post'], {
  LIKES: {
    aggregation: 'SUM',
    properties: {total: {property: 'weight', aggregation: 'SUM'}}
  }
}) YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"per_prop_v3",5,3
```

**Interpretación:** 3 pares únicos `(User,Post)` tras colapsar las 6
aristas paralelas. La propiedad `weight` se agrega con SUM y se renombra
a `total` en la proyección resultante, independiente de cómo se
agrupen las demás propiedades.

---

## Multi-label y configuración por tipo

### Contexto en `posts`

El dataset `posts` tiene 50 nodos `User`, de los cuales 5 son también
`Admin` y 12 son también `Guest`.

```gql
MATCH (n:User)  RETURN count(*) AS users;   -- 50
MATCH (n:Admin) RETURN count(*) AS admins;  --  5
```

**Resultado real:**
```
users
50
admins
5
```

### M3 — Lista de labels con deduplicación

```gql
CALL graph_project('m1_doc2', ['User','Admin'], 'Friend')
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"m1_doc2",50,50
```

**Interpretación:** `['User','Admin']` devuelve 50 nodos (no 55) porque los
5 `Admin` tienen *también* el label `User`, y el builder deduplica
automáticamente.

### M4 — MAP con orientación por tipo

```gql
CALL graph_project('m2_doc2', 'User', {
  Friend: {orientation: 'UNDIRECTED'}, Posted: {orientation: 'NATURAL'}
})
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"m2_doc2",50,50
```

**Interpretación:** cada tipo de arista puede declarar su propia
`orientation`, sobrescribiendo el default global. `Posted` se guarda
dirigido, `Friend` como no dirigido — en la misma proyección.

### Tip: `labels(n)` + `GROUP BY` devuelve vacío

En la versión actual, la agregación por `labels(n)` retorna listas vacías.
Para listar combinaciones distintas de labels, usa `DISTINCT`:

```gql
MATCH (n:User) RETURN DISTINCT labels(n) LIMIT 8;
```

(Valores observados: `[User,Admin]` × 5, `[User,Guest]` × 3, etc.)

---

## Extracción de datos GNN

### Las tres claves trabajan como trío

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `includeFeatures` | STRING | Nombre de la FeatureMatrix registrada (importada vía `--with-tensors`) |
| `labelProperty` | STRING | Nombre de la propiedad INTEGER que contiene el label de clasificación |
| `splitProperty` | STRING | Nombre de la propiedad STRING con `"train"`, `"val"`, `"validation"`, `"test"` |

**Clave central:** las tres forman un trío. Sin `includeFeatures` apuntando
a una FeatureMatrix registrada, el builder *no* genera `labels.bin`,
`splits.bin` ni `gnn_meta.bin` — aunque `labelProperty` y `splitProperty`
estén presentes, el builder hace *early return* (la fila no tiene entrada
en `RowMapping`, que deriva de la FeatureMatrix). Esto es intencional:
evita escribir archivos parciales sin el contexto de features.

### G0 — Labels del dataset `hetero_test`

`hetero_test` tiene `Paper`, `Author`, `Venue`. Usamos `Paper` porque
tiene `label` int (ML=0, DB=1, NLP=2) y `split` string.

```gql
MATCH (n) RETURN DISTINCT labels(n) AS lbls;
```

**Resultado real:**
```
lbls
["Paper"]
["Author"]
["Venue"]
```

### G1..G3 — Distribución

```gql
MATCH (p:Paper) RETURN count(*) AS papers;                               -- 20
MATCH (p:Paper) WHERE p.label = 0      RETURN count(*) AS ml_count;      --  7
MATCH (p:Paper) WHERE p.split = "train" RETURN count(*) AS train_count;  -- 12
```

**Resultado real (combinado):**
```
papers
20
ml_count
7
train_count
12
```

### G4 — Proyección base (sin GNN keys)

```gql
CALL graph_project('g1', 'Paper', 'CITES')
YIELD graphName, nodeCount, relationshipCount RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount
"g1",20,24
```

**Interpretación:** `CITES` se escribe en MAYÚSCULAS en `hetero_test`
(distinto de `Knows`/`Follows` en `books`/`posts`). Respeta la casing
del dataset fuente.

### G5 — Con `labelProperty` + `splitProperty` pero sin `includeFeatures`

```gql
CALL graph_project('g_int', 'Paper', 'CITES', {
  labelProperty: 'label', splitProperty: 'split'
})
YIELD graphName, nodeCount, relationshipCount, featureDim, numClasses
RETURN *;
```

**Resultado real:**
```
graphName,nodeCount,relationshipCount,featureDim,numClasses
"g_int",20,24,0,0
```

**Archivos producidos:** los B+tree habituales de la proyección
(`catalog.dat`, `nodes.{dir,leaf}`, `node_label.{dir,leaf}`,
`edge_label.{dir,leaf}`, `edge_direction.{dir,leaf}`,
`edge_from_to.{dir,leaf}`, `edge_n1_n2.{dir,leaf}`,
`from_to_edge.{dir,leaf}`, `label_edge.{dir,leaf}`,
`label_node.{dir,leaf}`, `to_from_edge.{dir,leaf}`). Los tres archivos
binarios del pipeline GNN **NO** se generan:
`labels.bin`, `splits.bin` y `gnn_meta.bin` — requieren `includeFeatures`
+ FeatureMatrix.

**Interpretación:** `numClasses=0` y `featureDim=0` aunque los 20 `Paper`
tienen `label` entero. Para activarlos hay que añadir
`includeFeatures: 'node_features'` (o cualquier FeatureMatrix registrada)
al MAP de configuración.

### Flujo completo con `includeFeatures`

Ver `Working-with-tensors.md` y `Partial_Idea/` para el flujo canónico:

> **Ejemplo aspiracional — no ejecutado en este documento.** Requiere
> importar Papers con `--with-tensors <features.npy>` y registrar una
> FeatureMatrix llamada `node_features`, además de los módulos GNN
> de muestreo y entrenamiento. Se incluye como referencia del flujo
> canónico; los YIELDs reales de ese flujo aparecen en las guías
> específicas (`gnn_offline_sample`, `gnn_train`) del wiki.

```gql
CALL graph_project('arxiv', ':Paper', ':CITES', {
    orientation: 'UNDIRECTED', nodeProperties: ['label', 'split'],
    includeFeatures: 'node_features',
    labelProperty: 'label', splitProperty: 'split'
}) YIELD graphName, nodeCount, featureDim, numClasses RETURN *;

CALL gnn_offline_sample('arxiv', 's1', [15, 10],
    {batchSize: 512, usePredefinedSplits: true})
    YIELD totalBatches RETURN *;

CALL gnn_train('s1', 'node_features',
    {hiddenDim: 256, epochs: 50, lr: 0.01})
    YIELD bestValAccuracy, testAccuracy RETURN *;
```

### Requisitos de tipos

- `labelProperty` acepta sólo valores INTEGER. Strings son silenciosamente
  ignorados. Por eso `hetero_test` usa `label: 0/1/2` (int), no
  `"ML"/"DB"/"NLP"` (str).
- `splitProperty` acepta strings `"train"`, `"val"`, `"validation"`,
  `"test"`. Otros valores se mapean a *unlabeled* (byte 255 en `splits.bin`).

---

## Campos YIELD

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `graphName` | STRING | Nombre de la proyección creada |
| `nodeCount` | INTEGER | Nodos totales en la proyección (después de dedup) |
| `relationshipCount` | INTEGER | Aristas totales en la proyección |
| `projectMillis` | INTEGER | Tiempo de construcción en milisegundos |
| `featureDim` | INTEGER | Dimensión de la FeatureMatrix vinculada (0 si `includeFeatures` no está activo) |
| `numClasses` | INTEGER | Clases únicas detectadas (0 si `labelProperty` no está activo) |

Puedes yieldear todos los campos o un subconjunto; por ejemplo
`YIELD nodeCount, relationshipCount RETURN ...`.

---

## Consultar con `USE`

### Cambiar a una proyección

```gql
USE autores_conocidos_t MATCH (n)-[e]->(m) RETURN count(*) AS edge_count;
```

**Resultado real (U1):**
```
edge_count
12
```

El nombre de la proyección va como identificador sin comillas y sin la
palabra clave `GRAPH`.

### U2 — Consulta con propiedades proyectadas

```gql
USE p1_doc MATCH (a:Author) RETURN a.name LIMIT 5;
```

**Resultado real:**
```
a.name
"John Doe"
"Jane Doe"
"Angela Everett"
"Robert Li"
"Mark Greene"
```

### Patrones de query soportados

| Patrón | Requisito | Ejemplo |
|--------|-----------|---------|
| `(n)` | Siempre funciona | `MATCH (n) RETURN count(*)` |
| `(n:Label)` | El label debe existir en la proyección | `MATCH (n:Author) RETURN n` |
| `(n)-[e]->(m)` | Siempre funciona | `MATCH (n)-[e]->(m) RETURN n, m` |
| `(n)~[e]~(m)` | Siempre funciona (undirected query) | `MATCH (n)~[e]~(m) RETURN n, m` |
| `(n)-[e:Type]->(m)` | El tipo debe existir en la proyección | `MATCH (n)-[e:Knows]->(m) RETURN n, m` |
| `n.property` | La propiedad debe haberse incluido al proyectar | `MATCH (n) RETURN n.age` |

### Volver al grafo principal

```gql
USE CURRENT_GRAPH
MATCH (n) RETURN count(*);
```

Equivalente: `USE HOME_GRAPH`.

---

## `PROJECT()` como agregado in-query

Además de `CALL graph_project`, existe `PROJECT()` como función agregada
dentro de `RETURN`. Esto construye la proyección a partir de los
resultados de un MATCH en lugar de escanear los índices B+Tree completos.

**Query:**
```gql
MATCH (a:Author)-[e:Knows]->(b:Author)
RETURN PROJECT('pj1_doc' INCLUDE LABELS INCLUDE PROPERTIES);
```

**Resultado real:**
```
PROJECT('pj1_doc'INCLUDELABELSINCLUDEPROPERTIES)
"pj1_doc"
```

**Interpretación:** `PROJECT()` devuelve el nombre de la proyección creada.
Las opciones `INCLUDE LABELS` y `INCLUDE PROPERTIES` preservan labels y
valores de propiedades respectivamente; ambas son opcionales.

Úsalo cuando ya tienes un pattern-match que filtra el subgrafo deseado y
no quieres re-expresarlo como argumentos a `graph_project`.

---

## Notas de comportamiento

- **Endpoint filtering.** Sólo se incluyen aristas cuyos dos extremos
  estén en el conjunto proyectado de nodos. Si `LIKES` conecta `User` con
  `Post` y sólo proyectas `'User'`, el resultado tendrá 0 aristas.
- **Deduplicación automática.** Listas con labels/tipos repetidos
  (`['User','User']`) se deduplican internamente; cada label/tipo se
  escanea una sola vez.
- **Warnings para labels inexistentes.** Si un label/tipo no existe en la
  base, se emite un warning pero la proyección se crea (con 0 nodos/aristas
  para ese label).
- **Proyecciones persistentes.** Viven en disco junto al grafo principal
  y sobreviven reinicios del servidor.
- **Error `Projection 'X' already exists`.** Re-ejecutar `CALL graph_project`
  con un nombre ya usado falla explícitamente. Para re-crear una
  proyección, hay que borrar primero su directorio físico bajo
  `<db>/projections/<nombre>/`.

---

## Limitaciones

- **Self-loops.** Queries con el patrón `(n)-[e]->(n)` (misma variable en
  ambos extremos) no están soportadas sobre proyecciones.
- **Label/type aliasing discarded.** El MAP acepta `label` como clave
  para renombrar (`{People: {label: 'Person'}}`) pero el alias se
  descarta: el nombre del label fuente se preserva. Ver
  `docs/research/project_aliasing_pending.md`.
- **Streaming aggregation hardcoded a COUNT.** La ruta streaming (O(1) en
  memoria) sólo está implementada para COUNT. SUM/MIN/MAX cargan el
  hashmap completo en RAM y pueden sobrecargar grafos con muchísimas
  aristas paralelas. Ver
  `docs/research/2026-03-25-streaming-aggregation-gate-fix.md`.
- **Labels enteros en GNN.** `labelProperty` sólo acepta valores INTEGER.
  Strings se ignoran silenciosamente.
- **`element_id()` + GROUP BY crash.** Combinar `element_id()` con
  agregación conocida crash; evítalo en queries sobre proyecciones.
- **Nombres de proyección.** No pueden contener `/`, `\`, bytes nulos ni
  caracteres de control; no pueden ser `.`, `..`, vacío ni sólo
  whitespace.

---

## Tuning de memoria

`graph_project` dimensiona su buffer de sort de forma adaptativa: en cada
construcción de índice B+tree lee `/proc/meminfo` y elige
`buffer = max(256 MB, MemAvailable × 3 / 4)`. Activo por defecto en Linux;
en macOS/Windows o donde `/proc/meminfo` no sea legible, el buffer queda
en el piso de 256 MB.

### Override manual

```bash
MDB_SORT_BUFFER_MB=4096 build/Release/bin/mdb server data/dbs/gql/books_docs
```

El piso de 256 MB sigue aplicándose. Útil para benchmarks reproducibles.

### Contenedores y cgroups

Dentro de Docker/Kubernetes, `/proc/meminfo` suele reportar el RAM del
host, no el límite de cgroup. Pasa `MDB_SORT_BUFFER_MB` explícito a una
fracción segura del límite para evitar OOM-kill.

### Diagnóstico

Cada construcción emite una línea a stdout como
`[sort] buffer=9216 MB (source=adaptive)`. Las fuentes posibles:
`adaptive` (default `MemAvailable × 3 / 4`), `env` (env var parseada),
`env_invalid` (env var no parseable), `explicit` (argumento de tests).

---

## Cómo reproducir este documento

### 1) Build

```bash
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release -D ENABLE_GNN=ON
cmake --build build/Release -j $(nproc)
```

### 2) Datasets

```bash
build/Release/bin/mdb import data/example/gql/books       data/dbs/gql/books_docs
build/Release/bin/mdb import data/example/gql/posts       data/dbs/gql/posts_docs
build/Release/bin/mdb import data/example/gql/hetero_test data/dbs/gql/hetero_docs

# Fixture local para el capítulo de agregación — formato nativo de import
cat > /tmp/agg_example.gql <<'EOF'
# Agregación example fixture — parallel LIKES edges
0 :User name:"Ana"
1 :User name:"Bob"
2 :User name:"Clara"
10 :Post title:"Guia"
11 :Post title:"Tutorial"

0->10 :LIKES weight:1
0->10 :LIKES weight:3
0->10 :LIKES weight:5
1->10 :LIKES weight:2
1->10 :LIKES weight:4
2->11 :LIKES weight:7
EOF
build/Release/bin/mdb import /tmp/agg_example.gql data/dbs/gql/agg_docs
```

### 3) Servir y consultar

```bash
build/Release/bin/mdb server data/dbs/gql/books_docs --port 1237 &

curl -s -X POST http://localhost:1237/query \
  -H "Content-Type: application/gql" \
  --data-binary @- <<'EOF'
CALL graph_project('autores_conocidos_t', 'Author', 'Knows')
YIELD graphName, nodeCount, relationshipCount, projectMillis RETURN *;
EOF
```

Respuesta: `"autores_conocidos_t",24,12,15`.

### 4) Convención de puertos

Todos los capítulos de este documento usan el puerto **1237** y asumen
que hay **un único servidor corriendo a la vez**: antes de cambiar de
dataset, detén el servidor anterior con `pkill -f "mdb server"` (o
`kill %1` si lo lanzaste con `&` en la misma shell) y arranca el nuevo
sobre el dataset correspondiente.

| Capítulo | Dataset | Directorio de DB |
|----------|---------|------------------|
| 1–6, 8, 13 | `books_docs` | `data/dbs/gql/books_docs` |
| 9 (M1..M4) | `posts_docs` | `data/dbs/gql/posts_docs` |
| 7 (A1..A6) | `agg_docs` | `data/dbs/gql/agg_docs` |
| 10 (G0..G5) | `hetero_docs` | `data/dbs/gql/hetero_docs` |

Los `projectMillis` variarán con el hardware; los counts (`nodeCount`,
`relationshipCount`, `featureDim`, `numClasses`) deberían coincidir
bit a bit con los mostrados en cada bloque `Resultado real`.
