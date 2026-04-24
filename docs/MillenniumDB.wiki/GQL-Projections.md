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
17. [Modelo de memoria de la fase de scan — `classic` vs `radix`](#modelo-de-memoria-de-la-fase-de-scan--backend-classic-vs-radix)
18. [Cómo reproducir este documento](#como-reproducir-este-documento)

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
| `indexSet` | STRING | `'ALL'` | Preset que controla qué índices B+Tree se materializan (ver "Index set selection" más abajo) |
| `buildTopologySnapshot` | BOOL | `false` | Si `true`, emite los archivos sidecar `topology_fwd.csr`/`topology_rev.csr` con adyacencia CSR para acelerar sampling GNN (ver "Topology snapshot" más abajo) |

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

## Modelo de memoria de la fase de scan — backend `classic` vs `radix`

A partir de 2026-04-21, `graph_project` expone **dos backends** para la
construcción de índices B+Tree, seleccionables en tiempo de ejecución vía
la variable de entorno `MDB_PROJECTION_SORTER`. El default (`classic`)
preserva el comportamiento existente; el nuevo (`radix`) está pensado
para datasets donde el pico de RSS del `classic` pondría en riesgo al
resto de aplicaciones del sistema (caso observado empíricamente en
`ogbn-papers100M`, 111M nodos, 1.6B aristas).

### Diferencias operacionales

| Aspecto | `MDB_PROJECTION_SORTER=classic` (default) | `MDB_PROJECTION_SORTER=radix` |
|---|---|---|
| Pipeline de sort | `ExternalRecordSort` (K-way merge) | 3 fases: scan+particiona → sort paralelo por partición → concatena en B+Tree |
| Paralelismo principal | `std::execution::par` (TBB) dentro del sort | `tbb::parallel_for` sobre particiones independientes |
| Pico de RSS | Crece con `MemAvailable × 3/4` (adaptive buffer); puede acercarse al RAM libre del host | Acotado por construcción: `num_particiones × 4 MB + num_workers × 512 MB ≈ 2.5 GB` |
| Liberación de heap | `malloc_trim(0)` entre índices | `malloc_trim(0)` entre índices **y** entre particiones |
| Archivos temporales | Scratch en `sort_tmp/` (K-way runs) | Scratch en `sort_tmp/.radix_scratch/`: `thread_T/part_P.bin` durante scan, luego `partition_P.bin`, luego `sorted_part_P.bin`; todos removidos al destruir |
| B+Tree resultante | Producido por `build_index_streaming` | Idéntico: mismo formato de páginas, misma dedup en la escritura |
| Override del buffer | `MDB_SORT_BUFFER_MB` aplica | `MDB_SORT_BUFFER_MB` no aplica (el backend usa particiones; ver abajo) |

### Garantía algorítmica del backend `radix`

El pico de RSS del pipeline radix está **acotado por construcción**, no
por el tamaño del dataset:

```
peak_RSS_radix  ≤  num_particiones × 4 MB (buffers de PartitionFile)
              + num_workers × 512 MB    (buffer de sort por worker)
              + O(batch)                (records en vuelo)
```

Con los defaults (`num_particiones ∈ [8, 128]`, `num_workers = 4`,
`worker_memory_budget = 512 MB`), el pico es a lo sumo ~2.5 GB
independientemente de si el dataset tiene 2 708 o 111 millones de nodos.
Esta propiedad elimina la categoría de fallo observada en `papers100M`
Run 5 (kernel reclaim evictó apps interactivas para ceder RAM al proceso
de `mdb`) **sin depender de aislamiento externo** (cgroups vía
`systemd-run --user --scope -p MemoryMax=...`).

### Cuándo usar cada backend

- **`classic` (default)** — graphs pequeños y medianos (hasta ~10M aristas
  en commodity hardware), donde el sort en memoria es más rápido que el
  overhead de particionar + escribir + leer scratch files.
- **`radix`** — graphs grandes (>50M aristas) en hardware con RAM limitada
  relativa al dataset, donde el pico del `classic` pondría en riesgo al
  resto del sistema. También útil para runs reproducibles en CI/benchmarks
  donde el pico de RAM debe ser determinístico.

### Selección y diagnóstico

```bash
# Usar el default (classic) — sin cambios
build/Release/bin/mdb server data/dbs/gql/papers100M

# Usar radix
MDB_PROJECTION_SORTER=radix build/Release/bin/mdb server data/dbs/gql/papers100M

# Los valores no reconocidos caen silenciosamente a classic
MDB_PROJECTION_SORTER=foobar build/Release/bin/mdb server ...   # == classic
```

La variable se lee **una vez** (cacheada vía `std::call_once`); cambiarla
mid-process requiere reiniciar el servidor.

### Evidencia de equivalencia

- **347/347 tests de integración GQL** pasan bajo ambos backends
  (invocar `MDB_PROJECTION_SORTER=classic ./scripts/run-tests gql` y
  `MDB_PROJECTION_SORTER=radix ./scripts/run-tests gql`).
- **Byte-identical B+Tree output** en `cora_gnn` (20/20 archivos `.leaf` /
  `.dir`, ~795 KB totales). Validado por
  `./scripts/test_projection_radix.sh` (retorna 0 on all-match).
- **10 tests unitarios** específicos en `RadixPartitionSortTests` y
  `SorterDispatch`: deterministic bucketing, routing por radix, sort por
  partición monotonic, concatenación globalmente ordenada, clamp de
  partition count, worker count adaptativo, fallback external sort,
  cleanup en destructor.

### Referencias

- Diseño formal: ADR-004 (`Partial_Idea/decisions/004_radix_partition_sort.md`)
- Spec técnica: `docs/superpowers/specs/2026-04-21-radix-partition-sort-design.md`
- Plan de implementación: `docs/superpowers/plans/2026-04-21-radix-partition-sort-plan.md`
- Implementación: `src/graph_models/gql/projection/{sorter_dispatch,partition_file,parallel_scan_partitioner,radix_partition_sort}.{h,cc}`
- Script golden compare: `scripts/test_projection_radix.sh`

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

---

## Index set selection — `indexSet` (added 2026-04-23)

Por defecto, `graph_project` materializa **10 índices B+Tree** (más hasta
4 índices de propiedad cuando se configuran) para soportar cualquier
patrón de query GQL. En cargas GNN solo se usan 2 de los 10, por lo
que los otros 8 son overhead puro. El parámetro `indexSet` controla
qué preset de índices se construye.

### Presets

| Preset | Índices materializados | Uso típico |
|--------|------------------------|-----------|
| `'ALL'` (default) | `nodes`, `node_label`, `label_node`, `from_to_edge`, `to_from_edge`, `edge_from_to`, `edge_direction`, `edge_n1_n2`, `edge_label`, `label_edge` | Queries GQL arbitrarias, analytics |
| `'GNN_MINIMAL'` | `nodes`, `node_label`, `label_node`, `from_to_edge`, `to_from_edge` | GNN training, k-hop sampling |
| `'READONLY_TRAVERSAL'` | `GNN_MINIMAL` + `edge_label`, `label_edge` | Traversal filtrado por label, sin lookup de edge-id |

Los índices de propiedad (`node_key_value`, `key_value_node`,
`edge_key_value`, `key_value_edge`) NO están controlados por `indexSet`:
se construyen condicionalmente según las propiedades solicitadas en
`nodeProjection`/`relationshipProjection` — independientes del preset.

### Ejemplo

```gql
CALL graph_project('paper_gnn', 'Paper', 'CITES', {
  orientation: 'NATURAL',
  indexSet: 'GNN_MINIMAL',
  includeFeatures: 'node_features',
  labelProperty: 'label',
  splitProperty: 'split'
}) YIELD graphName, nodeCount, relationshipCount RETURN *;
```

### Query patterns por preset

| Query pattern | ALL | READONLY | GNN_MINIMAL |
|--------------|:---:|:--------:|:-----------:|
| `MATCH (a)-[r]->(b)` (traversal directa) | ✓ | ✓ | ✓ |
| `MATCH (a)-[r:TYPE]->(b)` (filtrada por label) | ✓ | ✓ | ✗ |
| `MATCH (a)-[r]->(b) WHERE id(r) = X` (lookup por edge-id) | ✓ | ✗ | ✗ |
| `MATCH (a)-[r]-(b)` (UNDIRECTED explícito) | ✓ | ✗ | ✗ |
| GNN `gnn_offline_sample` + `gnn_train` | ✓ | ✓ | ✓ |
| Node label access (`n.label`) | ✓ | ✓ | ✓ |

Cuando se intenta una query que necesita un índice no materializado,
MillenniumDB eleva una `QueryException` clara indicando qué índice falta
y qué preset lo incluye. Por ejemplo, sobre una proyección `GNN_MINIMAL`:

```
QueryException: Cannot execute query - index 'edge_from_to' is not
materialized for projection 'paper_gnn' (indexSet='GNN_MINIMAL').
To enable this query, rebuild the projection with indexSet='ALL'.
```

### Reducciones empíricas (ogbn-products, 62 M aristas)

| Preset | Disco | Wall clock | Reducción disco | Reducción wall-clock |
|--------|------:|-----------:|:---------------:|:--------------------:|
| ALL | 7.24 GB | 493 s | (baseline) | (baseline) |
| GNN_MINIMAL | 2.95 GB | 213 s | **-61%** | **-57%** |
| READONLY_TRAVERSAL | 4.86 GB | 281 s | -36% | -43% |

Medido bajo `MDB_PROJECTION_SERIAL_SCAN=1 MDB_PROJECTION_SORTER=radix`.
Los números escalan proporcionalmente a grafos más grandes como
papers100M (con tope asintótico al 5/9 de índices eliminados).

### Cuándo elegir cada preset

- **ALL**: default razonable para proyecciones multi-uso o cuando no
  se conoce a priori la carga futura.
- **GNN_MINIMAL**: el 95% de las cargas GNN (sampling, training,
  inferencia batch). Si en el futuro necesitas otro patrón de query,
  reconstruyes la proyección con un preset más amplio.
- **READONLY_TRAVERSAL**: proyecciones de lectura con queries
  label-filtradas (e.g. `MATCH (n:Paper)-[:CITES]->(m)`) que NO requieren
  lookup por edge-id.

### Matriz de tareas GNN soportadas

El preset correcto depende de la tarea GNN que vayas a entrenar. Las dos
familias relevantes para MillenniumDB hoy son **Node Property Prediction
(NPP)** y **Link Property Prediction (LPP)**.

| Tarea GNN | Ejemplos dataset | `indexSet` recomendado | Procedures MDB activas |
|---|---|---|---|
| **Node Property Prediction** (clasificación / regresión sobre nodos) | Cora, ogbn-arxiv, ogbn-products, papers100M | `GNN_MINIMAL` | `gnn_offline_sample`, `gnn_materialize_batches`, `gnn_build_feature_store`, `gnn_train`, `gnn_predict`, `EmbeddingWriter` (Phase 6) |
| **Link Property Prediction** (homogeneous: un único tipo de arista) | ogbl-ppa, ogbl-collab, recommender sobre single-type graph | `GNN_MINIMAL` | pipeline node-centric reutilizable; la iteración de aristas positivas se hace con scan lineal de `from_to_edge` |
| **Link Property Prediction** (heterogeneous: varios `edge_label` a filtrar) | ogbl-biokg, ogbl-ddi, knowledge graphs | `READONLY_TRAVERSAL` | mismo pipeline; `MATCH ()-[r:DRUG_TARGET]->()` + scoring function per-type requiere `edge_label` |

**Por qué GNN_MINIMAL basta para LPP homogeneous:** LPP node-centric
samplea el k-hop neighborhood de cada endpoint de una arista positiva,
luego scorea. Ambos pasos son **node-entry** (parten de un nodo, no de
una arista). La iteración de aristas positivas durante training se
resuelve con un scan lineal de `from_to_edge`, no con edge-by-id lookup
— por eso `edge_from_to` sigue siendo innecesario.

**Por qué LPP heterogeneous sube a READONLY_TRAVERSAL:** si el dataset
tiene varios tipos de arista y quieres filtrar por tipo durante
training (p.ej. entrenar scoring separado por cada `edge_label`),
necesitas `edge_label` + `label_edge` para el filtro eficiente.

### Cambiar de preset

Los presets son fijos al momento de creación; están persistidos en
la metadata de la proyección (catalog v1.4). Para cambiar el preset,
`drop_projection` y recrea con el nuevo valor. Las proyecciones
pre-Spec-#3 (catalog v1.3) se leen como si fueran `ALL` para preservar
compatibilidad total hacia atrás.

---

## Topology snapshot — `buildTopologySnapshot` (added 2026-04-25)

### Motivación

Con el B+Tree existente, `get_out_neighbors(v)` cuesta `O(log N + k)` por
lookup. Al hacer sampling GNN con fanout `[15, 10]` sobre papers100M
(~1.6 B aristas, 111 M nodos) son ~2.2 B lookups por epoch, lo que da
~30 h/epoch en un workstation de 30 GB RAM — inviable para entrenamiento.
El sidecar CSR reduce ese costo a `O(1)` mediante un simple slice sobre
memoria mmap'd.

### Archivos sidecar

Cuando `buildTopologySnapshot: true`, al finalizar `graph_project` se
emiten dos archivos adicionales junto a los `.leaf`/`.dir`:

```
projections/<nombre>/
  ... archivos B+Tree existentes ...
  topology_fwd.csr        ← adyacencia saliente  (from → to)
  topology_rev.csr        ← adyacencia entrante   (to → from)
```

Formato binario de cada archivo: header fijo de 64 bytes
(magic `TOPOCSR1`, versión, id_width, flags, num_nodes, num_edges,
SHA-256 del `.leaf` fuente) + `ROW_PTR[N+1]` + `COL_IDX[M]` +
opcional `EDGE_IDS[M]`. Todos little-endian uint64. Ver la especificación
`docs/superpowers/specs/2026-04-25-topology-snapshot-design.md` §5.1.

### Ejemplo

```gql
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
  orientation: 'NATURAL',
  indexSet: 'GNN_MINIMAL',
  buildTopologySnapshot: true,
  includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relationshipCount, topologySnapshotBytes
RETURN *;
```

`topologySnapshotBytes` es un nuevo campo YIELD con el tamaño total
(bytes) de los sidecars efectivamente escritos. `0` si el flag no estaba
activo o si el `IndexSet` activo no incluía ninguno de los dos índices
de aristas.

### Composabilidad con `indexSet`

El builder verifica el `IndexSet` activo antes de escribir cada dirección:

| IndexSet | `topology_fwd.csr` | `topology_rev.csr` |
|---|:---:|:---:|
| `ALL` | ✅ | ✅ |
| `GNN_MINIMAL` | ✅ | ✅ |
| `READONLY_TRAVERSAL` | ✅ | ✅ |

Si alguna dirección no está disponible, el builder log-ea una línea
indicándolo y omite sólo esa dirección (no falla el build).

### Fallback automático

El lado lector (`TopologyAccessor`) detecta la presencia de los sidecars
al abrir la proyección y valida (magic, versión, tamaño, invariantes de
`ROW_PTR`, SHA-256 del `.leaf` fuente). Cualquier falla
(archivo ausente, stale por mutación externa del `.leaf`, corrupción,
mmap error) degrada silenciosamente al path B+Tree existente — GNN
sampling continúa siendo correcto, solo ~100× más lento. El output de
`sample_neighbors` es bit-identical entre ambos paths bajo RNG fijo.

### Tradeoffs

| Costo | Dataset pequeño (cora) | ogbn-products | papers100M (proyectado) |
|---|---:|---:|---:|
| Overhead de disco | ~23 % | ~34 % | ~7-8 % |
| Overhead wall-clock del build | ~45 % (ruido en graphs pequeños) | ~10 % | ~10 % |
| Open-time latency (SHA-256 de fuente) | <1 ms | ~10-20 s | ~40-70 s |
| Speedup de sampling | 3-5× (ya cache-resident) | 50-500× | 100-1000× |

Números concretos en `docs/research/2026-04-25-topology-snapshot-bench.md`
(Gate B).

### Cuándo activar

- **Activa** si la proyección se va a usar como source para
  `gnn_offline_sample` / `gnn_train` / `EmbeddingWriter` con on-the-fly
  k-hop, especialmente en datasets > ~10 M aristas.
- **No actives** en proyecciones usadas solo para queries GQL
  tradicionales (`MATCH`, etc.) — el sidecar no acelera esas queries.
- **No actives** si tienes pressure de disco crítica y cada GB cuenta —
  el B+Tree solo ya es funcionalmente completo para GNN.

### Construcción post-hoc

Si una proyección ya fue creada sin el flag, no es necesario
reconstruirla — se puede generar el sidecar después:

```gql
CALL gnn_build_topology_snapshot('gnn_proj')
YIELD projectionName, fwdBytes, revBytes, durationMillis
RETURN *;
```

Esto lee los `.leaf` existentes y escribe los `.csr` al lado. Idempotente
(sobreescribe con warning si ya existían).

### Borrado

Los `.csr` se pueden borrar manualmente con `rm topology_fwd.csr` /
`rm topology_rev.csr` en cualquier momento — la proyección queda válida
y consultable; el siguiente open de `TopologyAccessor` simplemente usará
el path B+Tree.

### Backwards compat

Zero impact: proyecciones pre-Spec-#4-B no tienen `.csr`; el lector cae
automáticamente al path B+Tree. Binarios de MillenniumDB pre-Spec-#4-B
ignoran los `.csr` (no son referenciados por el catálogo). Catalog
version **no cambia** — no hay migración.

### Referencias

- Spec: `docs/superpowers/specs/2026-04-25-topology-snapshot-design.md`
- Plan: `docs/superpowers/plans/2026-04-25-topology-snapshot-plan.md`
- ADR: `Partial_Idea/decisions/006_topology_snapshot.md`
- Master plan §8-9: `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md`

---

## Leaf encoding — `leafFormat` config parameter (added 2026-04-24, ADR 007)

Por defecto, las páginas hoja del B+Tree usan el formato v1 (`BITSET`):
una bitmask de bytes redundantes seguida de los bytes restantes de cada
record. Spec #5 introduce un formato v2 alternativo (`DELTA_VARINT`) que
delta-codifica cada record contra el anterior y comprime cada delta con
LEB128 + zigzag, reduciendo el tamaño en disco ~80% sobre datasets GNN
reales sin penalizar el throughput de lectura. Es opt-in por proyección y
por índice — el default preserva el formato pre-Spec-#5 byte-a-byte.

### Valores

| Valor | Significado | Compatibilidad |
|-------|-------------|----------------|
| `'BITSET'` (default) | Formato v1 (redundant-byte-bitset). Idéntico al output pre-2026-04. | Catalog v1.4 y anteriores: implícito. Catalog v1.5+: explícito por índice. |
| `'DELTA_VARINT'` | Formato v2 (16 B header + varint LEB128 + zigzag delta payload). | Solo catalog v1.5+ (introducido 2026-04-24). |

### Ejemplo

```gql
CALL graph_project('paper_gnn', 'Paper', 'CITES', {
  orientation: 'NATURAL',
  leafFormat: 'DELTA_VARINT',
  indexSet: 'GNN_MINIMAL',
  buildTopologySnapshot: true,
  includeFeatures: 'node_features'
}) YIELD graphName, nodeCount, relationshipCount RETURN *;
```

El parámetro se aplica uniformemente a **todos los índices materializados**
de la proyección (B+Tree topológicos y de propiedad). La granularidad
per-índice está soportada en el catalog v1.5 (un byte por índice), pero
la API GQL actual expone solo el modo uniforme; selección heurística per
índice queda diferida a Spec #5-B.

### Reducción de tamaño empírica

| Dataset | BITSET | DELTA_VARINT | Ratio | Reducción |
|---------|-------:|-------------:|:-----:|:---------:|
| cora_gnn (GNN_MINIMAL) | 368 KB | 72 KB | 0.196 | **-80%** |
| ogbn-arxiv (GNN_MINIMAL) | 60.10 MB | 12.91 MB | 0.215 | **-79%** |
| ogbn-products (GNN_MINIMAL) | TBD — ver bench report | TBD | TBD | TBD |

Desglose por índice en ogbn-arxiv:

| Índice | BITSET | DELTA_VARINT | Ratio |
|--------|-------:|-------------:|:-----:|
| `from_to_edge` | 28 MB | 5.5 MB | 0.195 |
| `to_from_edge` | 28 MB | 7.2 MB | 0.256 |
| `label_node`, `node_label`, `nodes` | ~similar | ~similar | ~0.20 |

El ratio asimétrico entre `from_to_edge` (0.195) y `to_from_edge` (0.256)
refleja que la entropía de los deltas depende del orden de sort: el lado
forward queda mejor agrupado por destino contiguo que el reverse.

### Throughput de lectura

Tras el fix de cursor secuencial (T5.13b, commit `a94c06cf` que cambió
`search_index` de O(k²) a O(k)):

| Dataset | BITSET | DELTA_VARINT | Ratio v2/v1 |
|---------|-------:|-------------:|:-----------:|
| cora_gnn | 31.1 ms | 31.6 ms | 1.016× |
| ogbn-arxiv | 437.6 ms | 435.7 ms | 0.996× |

Gate C requería ratio ≤ 1.20×; el resultado real (≈ 1.0×) es
Pareto-dominante: comprime mucho sin costar nada en lectura.

El tiempo de build es indistinguible entre ambos formatos (cora_gnn
≈0.06 s; ogbn-arxiv ≈3.1 s) — la fase dominante es el sort, no la
escritura de leaves.

### Cuándo elegir cada formato

- **BITSET** (default): proyecciones existentes que ya están en disco;
  pipelines que necesitan reproducir output pre-Spec-#5 byte-a-byte
  (golden compares, regression suites externos).
- **DELTA_VARINT**: cualquier proyección nueva sobre datasets ≥10 M
  aristas, especialmente para cargas GNN. Combina con `indexSet:
  'GNN_MINIMAL'` y `buildTopologySnapshot: true` para obtener el stack
  completo de compresión Spec #3 + Spec #4-B + Spec #5.

### Cómo funciona

Cada record (Record\<N\>) se interpreta como N enteros uint64.
**Record 0** se escribe como N varints completos (LEB128 sin signo, 1-10
bytes por entero según magnitud). **Records 1..k-1** se escriben como
N **deltas** contra el record anterior, cada delta pasado por
**zigzag** (mapeo bijectivo signed→unsigned `(n << 1) ^ (n >> 63)`) y
codificado como LEB128. Este encoding aprovecha que los records están
ordenados — la mayoría de los deltas son pequeños (1-2 bytes), comparado
con los 8 bytes nominales de un uint64. El header de 16 bytes contiene
`format_version=2`, `record_width=N`, `value_count=k`, `next_leaf` y
campos reservados; padding hasta 4096 al final de la página.

### Catalog format

El leaf format se persiste como **un uint8 por índice materializado** en
el catalog v1.5 (introducido 2026-04-24). Catalog v1.4 (Spec #3) y
anteriores se leen como `BITSET` para todos sus índices, preservando
compatibilidad backward total con proyecciones pre-Spec-#5.

### Ejemplo byte-a-byte

Dado el set de records sorted (Record\<3\>):

```
r[0] = (1000, 2000, 3000)
r[1] = (1000, 2001, 3005)
r[2] = (1001,  500, 3100)
```

Encoding v2:

- **Header (16 B):** `02 03 00 00 03 00 00 00 00 00 00 00 00 00 00 00`
  (format_version=2, record_width=3, value_count=3, next_leaf=0)
- **Record 0** (full varints, 6 B):
  - `1000` → `E8 07`
  - `2000` → `D0 0F`
  - `3000` → `B8 17`
- **Record 1** (zigzag-deltas vs r[0], 3 B):
  - `0` → `00`
  - `+1` → `02`
  - `+5` → `0A`
- **Record 2** (zigzag-deltas vs r[1], 5 B):
  - `+1` → `02`
  - `-1501` → `B9 17`
  - `+95` → `BE 01`

**Total usado: 16 + 6 + 3 + 5 = 30 bytes** (de los 4096 disponibles en
la página). El resto se zero-fillea hasta 4096. La equivalencia bitset
del mismo set ronda los 35 bytes (8 header + 3 bitset + 24 records
parcialmente comprimidos), un -14% en este micro-ejemplo. Los gains
reales escalan a ~-80% en páginas de ~150 records donde el overhead
amortizado del record 0 se diluye.

### Fallback y validación

Cada página v2 valida en el primer open (T5.4-T5.8): magic byte
`format_version=2`, `record_width` coincide con el `Record<N>`
esperado, `value_count` ≤ `leaf_max_records`, decode varint sin
overflow ni underflow. Cualquier corrupción eleva
`BPTLeafV2DecodeException` con offset preciso. Páginas v1 siguen el
camino existente sin cambios.

### Construcción y backwards compat

Los formatos NO son interconvertibles in-place: la página v2 es
inmutable, por lo que cambiar de formato requiere `drop_projection` +
recrear con el nuevo `leafFormat`. Proyecciones pre-Spec-#5 (catalog
v1.4 y anteriores) siguen siendo legibles sin migración — el catalog
v1.5 reader las trata como `BITSET` de forma transparente.

### Archivos

- `src/storage/index/bplus_tree/bpt_leaf_format.{h,cc}` — enum + helpers.
- `src/storage/index/bplus_tree/varint.{h,cc}` — LEB128 + zigzag codec.
- `src/storage/index/bplus_tree/bplus_tree_leaf_v2.{h,cc}` — v2 leaf
  read/write path.
- `src/storage/index/bplus_tree/bpt_mem_import.h` — `BPTLeafV2Writer`
  bulk-load.
- `src/graph_models/gql/projection/projection_catalog.{h,cc}` —
  catalog v1.5.
- `src/query/procedure/builtin/project_procedure.cc` — parsing del
  config key.

### Tests

- Unit: `bpt_leaf_v2_format_test`, `varint_test`, `bpt_leaf_v2_writer_test`,
  `bpt_leaf_v2_reader_test`, `bpt_iter_dispatch_test`,
  `projection_catalog_v5_test`, `projection_leaffmt_config_test`.
- Fuzz: `bpt_leaf_v2_fuzz_test` — 500 K random + 1 K smoke + 10 K
  boundary roundtrips, 100 % tamper-flip detection.
- Integración: `scripts/test_projection_leaffmt.sh` — golden compare
  6-mode (BITSET/DELTA_VARINT × {classic, radix} × {serial, parallel}).
- Bench: `scripts/bench_leaffmt.sh` — Gate C harness (size + read
  throughput por dataset).

### Referencias

- Spec: `docs/superpowers/specs/2026-04-25-delta-varint-leaf-design.md`
- Plan: `docs/superpowers/plans/2026-04-25-delta-varint-leaf-plan.md`
- ADR: `Partial_Idea/decisions/007_delta_varint_leaf.md`
- Master plan: `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md`

---

## Graph storage — `graphStorage` config parameter (added 2026-04-24, ADR 008)

Spec #8 introduce un switch arquitectónico opt-in que hace que las
páginas hoja de los índices de arista **sean** la CSR (Compressed
Sparse Row) en lugar de almacenarla en un sidecar aparte. El resultado
unifica el B+Tree de Spec #5 con la topología de Spec #4-B en una única
estructura on-disk, eliminando la duplicación de ~34 % que costaba el
sidecar y reduciendo el tamaño de los índices de arista ~81 % adicional
sobre la baseline bitset.

### Valores

| Valor | Significado | Compatibilidad |
|-------|-------------|----------------|
| `'BTREE'` (default) | Cada índice usa un B+Tree por separado; el encoding de leaf viene dado por `leafFormat` (BITSET o DELTA_VARINT). Preserva el comportamiento pre-2026-04-24 byte-a-byte. | Catalog v1.5 y anteriores: implícito. Catalog v1.6+: explícito por proyección. |
| `'CSR_HYBRID'` | Los índices de arista (`FROM_TO_EDGE`, `TO_FROM_EDGE`) emiten páginas v3 con layout CSR interno (tabla de offsets + stream de dst comprimido con DELTA_VARINT). Los demás índices (`nodes`, `node_label`, `label_node`, propiedades) siguen usando el `leafFormat` configurado. | Solo catalog v1.6+ (introducido 2026-04-24). |

El valor se persiste **por proyección** como un uint8 en el catalog
v1.6. El scope intencionalmente excluye índices de nodo y propiedad
porque la ganancia CSR se concentra en aristas ordenadas — los índices
de nodo ya tienen cardinalidad 1:1 con la clave.

### Ejemplo

```gql
CALL graph_project('gnn_proj', 'Paper', 'CITES', {
  orientation: 'NATURAL',
  indexSet: 'GNN_MINIMAL',
  leafFormat: 'DELTA_VARINT',
  graphStorage: 'CSR_HYBRID'
}) YIELD graphName, nodeCount, relationshipCount RETURN *;
```

Este es el stack de compresión completo de la tesis: `GNN_MINIMAL`
recorta los índices innecesarios (Spec #3), `DELTA_VARINT` comprime los
índices de nodo/propiedad (Spec #5), y `CSR_HYBRID` colapsa los índices
de arista a layout CSR v3 (Spec #8).

### Reducción de tamaño empírica

| Dataset | BTREE (baseline BITSET) | CSR_HYBRID | Ratio | Reducción |
|---------|------------------------:|-----------:|:-----:|:---------:|
| cora_gnn (GNN_MINIMAL) | 150 KB | 54 KB | 0.361 | **-63.9%** |
| ogbn-arxiv (GNN_MINIMAL) | 35 MB | 6.5 MB | 0.185 | **-81.5%** |

El ratio mejora al escalar porque el header fijo de 16 B se amortiza
mejor sobre páginas más densas. En cora_gnn el overhead de headers
domina; en ogbn-arxiv cada hub comprime cerca del mínimo teórico
(~0.5 B por arista).

### Throughput de scan

Tras el fix de cursor secuencial (T8.12b, commit `b9ca276f` que cambió
`BPTLeafCSR::search_index` de O(n²) a O(n)):

| Dataset | BTREE | CSR_HYBRID | Ratio v3/baseline |
|---------|------:|-----------:|:-----------------:|
| ogbn-arxiv | reference | 1.046× | ≈ paridad |

El Gate D requería ≤ 1.20×; el resultado real (≈ 1.0×) significa que
el scan es efectivamente equivalente al B+Tree original, con la
reducción de disco del 81.5 % además.

### Interacción con `buildTopologySnapshot`

Cuando `graphStorage: 'CSR_HYBRID'` está activo, **`buildTopologySnapshot`
es silenciosamente ignorado**. La razón es que los sidecar
`topology_fwd.csr` + `topology_rev.csr` duplicaban on-disk el mismo
layout CSR que ahora vive dentro de las hojas del B+Tree. El skip se
aplica en `build_topology_snapshots_()` y no emite warning — es la
semántica documentada del opt-in.

Si se necesita el sidecar (por ejemplo, para tooling externo que aún
no conoce v3), dejar `graphStorage: 'BTREE'` y activar
`buildTopologySnapshot: true` es la configuración correcta; los dos
caminos son mutuamente exclusivos a nivel de proyección.

### Composabilidad con `indexSet` y `leafFormat`

- `indexSet`: el scope de `CSR_HYBRID` son los índices de arista; si
  el preset no materializa `FROM_TO_EDGE` / `TO_FROM_EDGE` (no aplica
  a ningún preset actual), el flag es no-op.
- `leafFormat`: `DELTA_VARINT` aplica a los índices de nodo y
  propiedad incluso bajo `CSR_HYBRID`; el stream de dst interno al
  page v3 siempre usa delta-varint independientemente del
  `leafFormat` del resto de índices.
- `MDB_PROJECTION_SORTER` (classic / radix): ambos backends producen
  output v3 byte-idéntico bajo CSR_HYBRID, validado por T8.10.

### Limitaciones conocidas (tracked for Spec #8-B)

1. **`edge_id` no está persistido en v3.** El layout CSR emite solo
   `(src-implied, dst)`; el edge_id queda implícito por posición. Las
   queries GQL que cuentan aristas con `count(e)` o enlazan `e` a una
   variable devuelven conteos inflados o ids sintéticos. Spec #8-B
   añadirá un stream opcional `EDGE_IDS[]` gated por un flag del
   header. Las queries que solo necesitan topología (el caso GNN)
   funcionan sin esta extensión.
2. **Sampling CSR en ogbn-arxiv bloqueado.** El flujo
   `gnn_offline_sample` sobre una proyección `CSR_HYBRID` a escala
   arxiv dispara un fallo `decode_tuple_` en la posición 3974 de una
   hub-page. `cora_gnn` pasa correctamente. El benchmark T8.12
   excluye arxiv+sampling en CSR_HYBRID para evitar falsos positivos;
   el fix está planificado como Spec #8-B.

### Layout on-disk

Cada página v3 (4096 bytes) tiene la siguiente estructura:

```
+--------------------------------------+  offset 0
| header v3 (16 B)                      |   magic, version=3, record_width=3,
|                                       |   num_src_nodes, num_edges, next_leaf
+--------------------------------------+  offset 16
| offset_table[num_src_nodes + 1]       |   uint32 por entrada, apunta a byte
|                                       |   offset dentro del payload por src
+--------------------------------------+
| src_table[num_src_nodes]              |   full LEB128 src-ids (1-10 B cada)
+--------------------------------------+
| csr_entries                           |   para cada src, lista de dst
|   src_0: dst_0 (full varint),         |   codificada con DELTA_VARINT
|          dst_1..dst_m (zigzag-delta)  |   (Spec #5 codec)
|   src_1: dst_0 (full varint),         |
|          ...                          |
+--------------------------------------+
| zero-pad hasta 4096                   |
+--------------------------------------+
```

El reader valida en el primer open (T8.4): `format_version == 3`,
`record_width == 3`, `num_src_nodes + 1 ≤ offset_table_cap`, cada
offset está dentro del page. Cualquier corrupción eleva
`BPTLeafCSRDecodeException` con offset preciso. Páginas v1 / v2
siguen el camino existente sin cambios — el dispatch es a 3 vías en
`BptIter` basado en el primer byte del header.

### Hubs y split de páginas

Cuando la adjacencia de un src excede el budget de página (4 KB), el
writer parte la lista en páginas consecutivas enlazadas por
`next_leaf`. Esto preserva la invariante single-file del B+Tree sin
introducir una chain de overflow separada. En ogbn-arxiv (max degree
13 161) cada hub ocupa 1-2 páginas; en papers100M (max degree 3.8 M)
un hub ocupará ~940 páginas, caso que motiva el follow-up H1 en
Spec #8-B.

### Archivos

- `src/storage/index/bplus_tree/bpt_leaf_csr_format.{h,cc}` — enum
  value `LeafFormat::CSR_HYBRID` + v3 header struct + helpers.
- `src/storage/index/bplus_tree/bplus_tree_leaf_csr.{h,cc}` —
  `BPTLeafCSR` reader con cursor secuencial (T8.12b).
- `src/storage/index/bplus_tree/bpt_mem_import.h` —
  `BPTLeafCSRWriter` bulk-load, integrado en ambos pipelines de sort.
- `src/graph_models/gql/projection/projection_catalog.{h,cc}` —
  catalog v1.6 con byte `graphStorage` por proyección.
- `src/query/procedure/builtin/project_procedure.cc` — parsing del
  config key `graphStorage`.
- `src/gnn/projection/topology_accessor.cc` — fast-path que detecta
  v3 y bypasea el sidecar path.

### Tests

- Unit: `bpt_leaf_csr_format_test`, `bpt_leaf_csr_reader_test`,
  `bpt_leaf_csr_writer_test`, `bpt_iter_dispatch_test` (extendido a
  3 vías), `projection_catalog_v6_test`,
  `projection_graph_storage_config_test`,
  `graph_storage_integration_test`.
- Fuzz: `bpt_leaf_csr_fuzz_test` — 500 K random + 10 K boundary + 1 K
  tamper-flip, zero mismatches bajo seed `0xC5B8_1234_5678_9ABC`.
- Integración: `scripts/test_projection_csr_hybrid.sh` — 4-mode
  golden compare (BTREE/CSR_HYBRID × {classic, radix}) sobre cora_gnn
  + check de supersedencia del sidecar.
- Bench: `scripts/bench_csr_hybrid.sh` — Gate D harness (size + scan
  throughput + GNN sampling por dataset × modo).

### Construcción y backwards compat

Las páginas v3 son inmutables: cambiar `graphStorage` requiere
`drop_projection` + recrear. Proyecciones pre-Spec-#8 (catalog v1.5 y
anteriores) se leen como `BTREE` implícitamente. El catalog v1.6
añade exactamente un byte por proyección sobre v1.5 — sin impacto
material en storage.

### Referencias

- Spec: `docs/superpowers/specs/2026-04-25-csr-hybrid-design.md`
- Plan: `docs/superpowers/plans/2026-04-25-csr-hybrid-plan.md`
- ADR: `Partial_Idea/decisions/008_csr_hybrid.md`
- Master plan: `docs/superpowers/plans/2026-04-23-projection-compression-stack-plan.md`
