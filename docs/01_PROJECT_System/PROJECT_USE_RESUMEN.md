# Resumen Ejecutivo: PROJECT y USE GRAPH

## ¿Qué Hacen?

### PROJECT - Crea un Subgrafo

```gql
MATCH (patrón) RETURN PROJECT("nombre" [INCLUDE LABELS] [INCLUDE PROPERTIES])
```

**Resultado**: Crea una copia filtrada del grafo con solo los nodos/aristas del MATCH.

### USE - Consulta un Subgrafo

```gql
USE "nombre"
MATCH (patrón) RETURN resultados
```

**Resultado**: Todas las consultas subsiguientes usan el subgrafo en lugar del grafo principal.

---

## Ejemplo Completo

### Paso 1: Datos Iniciales

Supongamos que tu grafo principal tiene:

```
Grafo Principal:
- 100 nodos (50 Users, 30 Posts, 20 Comments)
- 150 aristas
- Etiquetas: User, Admin, Guest, Post, Comment
- Propiedades: name, age, email, title, content, etc.
```

### Paso 2: Crear Proyección

```bash
# Crear subgrafo solo de Users mayores de 18 y sus conexiones
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v:User)
 WHERE u.age >= 18 AND v.age >= 18
 RETURN PROJECT("adultos" INCLUDE LABELS INCLUDE PROPERTIES)'
```

**¿Qué pasa internamente?**

```
1. MATCH ejecuta y encuentra: 30 nodos User >= 18, 45 aristas entre ellos

2. PROJECT crea archivos:
   data/dbs/gql/mi_db/projections/adultos/
   ├── node_id.{dir,leaf}           # 30 nodos
   ├── from_to_edge.{dir,leaf}      # 45 aristas
   ├── to_from_edge.{dir,leaf}      # Índice inverso
   ├── node_label.{dir,leaf}        # Labels (User, Admin)
   ├── label_node.{dir,leaf}        # Índice inverso labels
   ├── node_key_value.{dir,leaf}    # Propiedades (name, age, email)
   └── key_value_node.{dir,leaf}    # Índice inverso propiedades

3. Tamaño típico: ~500 KB - 5 MB (mucho menos que el grafo principal)
```

### Paso 3: Consultar Proyección

```bash
# Cambiar a la proyección
curl -X POST http://localhost:1234/gql --data \
'USE "adultos"
 MATCH (u)
 RETURN u.name, u.age
 ORDER BY u.age DESC
 LIMIT 5'
```

**Resultado:**
```csv
u.name,u.age
"Rachel Sutton",75
"Cheryl Durham",72
"James Bradley",71
"Sabrina Davis",61
"Megan Chang",59
```

**¿Por qué es más rápido?**
- Solo escanea 30 nodos en lugar de 100
- Índices más pequeños = menos I/O de disco
- Datos ya filtrados = menos comparaciones

---

## Comparación Visual

### Consulta en Grafo Principal

```
Query: MATCH (u:User) WHERE u.age >= 18 RETURN u.name, u.age

Pasos:
1. Escanear node_label: 100 nodos → Filtrar por :User → 50 nodos
2. Para cada nodo, acceder node_key_value para .age
3. Filtrar WHERE age >= 18 → 30 nodos
4. Acceder node_key_value para .name de 30 nodos
5. Retornar resultados

Índices accedidos:
- node_label (100 entradas)
- node_key_value (~200 accesos: 50 para .age, 30 para .name)

Tiempo estimado: 10-50 ms
```

### Consulta en Proyección "adultos"

```
Query: USE "adultos" MATCH (u) RETURN u.name, u.age

Pasos:
1. Escanear node_id: 30 nodos (todos válidos, ya filtrados)
2. Acceder node_key_value para .name y .age de 30 nodos
3. Retornar resultados

Índices accedidos:
- node_id (30 entradas)
- node_key_value (~60 accesos: 30 para .name, 30 para .age)

Tiempo estimado: 1-5 ms (5-10x más rápido)
```

---

## Ejemplos de Uso Real

### Caso 1: Red Social - Amigos de Amigos

```bash
# Crear proyección de amigos directos
curl -X POST http://localhost:1234/gql --data \
'MATCH (me:User {id: 123})-[:FRIEND]-(friend)
 RETURN PROJECT("mis_amigos" INCLUDE LABELS INCLUDE PROPERTIES)'

# Consultar hobbies de mis amigos
curl -X POST http://localhost:1234/gql --data \
'USE "mis_amigos"
 MATCH (f)
 RETURN f.name, f.hobbies
 WHERE f.age BETWEEN 25 AND 35'
```

**Beneficio**: En lugar de buscar en 10M de usuarios, solo buscas en tus 500 amigos.

### Caso 2: E-commerce - Productos Relacionados

```bash
# Crear proyección de productos de una categoría
curl -X POST http://localhost:1234/gql --data \
'MATCH (p:Product {category: "Electronics"})-[r]-(related)
 RETURN PROJECT("electronics" INCLUDE LABELS INCLUDE PROPERTIES)'

# Analizar relaciones
curl -X POST http://localhost:1234/gql --data \
'USE "electronics"
 MATCH (p1)-[:BOUGHT_TOGETHER]->(p2)
 RETURN p1.name, p2.name, count(*) as frequency
 ORDER BY frequency DESC
 LIMIT 10'
```

**Beneficio**: Análisis rápido de una categoría sin cargar todo el catálogo.

### Caso 3: Grafo Temporal - Eventos Recientes

```bash
# Crear proyección solo de eventos del último mes
curl -X POST http://localhost:1234/gql --data \
'MATCH (e:Event)-[r]-(entity)
 WHERE e.timestamp > 1704067200  // 1 mes atrás
 RETURN PROJECT("recent" INCLUDE LABELS INCLUDE PROPERTIES)'

# Análisis de eventos recientes
curl -X POST http://localhost:1234/gql --data \
'USE "recent"
 MATCH (e:Event)
 RETURN e.type, count(e) as total
 GROUP BY e.type
 ORDER BY total DESC'
```

**Beneficio**: Consultas sobre datos recientes sin afectar análisis históricos.

---

## Opciones de PROJECT

### Solo Estructura (Mínimo)

```gql
PROJECT("nombre")
```

**Incluye:**
- ✅ Nodos (IDs)
- ✅ Aristas (IDs, dirección)
- ❌ NO etiquetas
- ❌ NO propiedades

**Tamaño:** ~50-200 KB para 1000 nodos

**Usar cuando:** Solo necesitas contar nodos/aristas, analizar conectividad.

### Con Etiquetas

```gql
PROJECT("nombre" INCLUDE LABELS)
```

**Incluye:**
- ✅ Nodos (IDs)
- ✅ Aristas (IDs, dirección)
- ✅ Etiquetas de nodos y aristas
- ❌ NO propiedades

**Tamaño:** ~100-500 KB para 1000 nodos

**Usar cuando:** Necesitas filtrar por tipos (User, Post, etc.) pero no propiedades.

### Con Propiedades

```gql
PROJECT("nombre" INCLUDE PROPERTIES)
```

**Incluye:**
- ✅ Nodos (IDs)
- ✅ Aristas (IDs, dirección)
- ❌ NO etiquetas
- ✅ Propiedades de nodos y aristas

**Tamaño:** ~500 KB - 2 MB para 1000 nodos

**Usar cuando:** Necesitas acceder a datos (name, age) pero no filtrar por tipos.

### Completa (Máximo)

```gql
PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)
```

**Incluye:**
- ✅ TODO

**Tamaño:** ~1-5 MB para 1000 nodos

**Usar cuando:** Necesitas funcionalidad completa en la proyección.

---

## Flujo de Trabajo Típico

### Desarrollo de Feature

```bash
# 1. Durante desarrollo, trabajas en el grafo principal
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e:FRIEND]->(v:User)
 RETURN u.name, v.name
 LIMIT 10'

# 2. Una vez que tu query funciona, creas proyección para producción
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e:FRIEND]->(v:User)
 RETURN PROJECT("friends_network" INCLUDE LABELS INCLUDE PROPERTIES)'

# 3. En producción, todas las queries usan la proyección
curl -X POST http://localhost:1234/gql --data \
'USE "friends_network"
 MATCH (u)-[e]->(v)
 WHERE u.age > 18
 RETURN u.name, count(v) as friend_count
 GROUP BY u.name
 ORDER BY friend_count DESC
 LIMIT 10'
```

### A/B Testing de Queries

```bash
# Crear dos proyecciones con diferentes criterios
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v) WHERE u.country = "USA"
 RETURN PROJECT("usa_users" INCLUDE PROPERTIES)'

curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v) WHERE u.country = "Chile"
 RETURN PROJECT("chile_users" INCLUDE PROPERTIES)'

# Comparar métricas
echo "=== USA ==="
curl -X POST http://localhost:1234/gql --data \
'USE "usa_users" MATCH (u) RETURN avg(u.age), count(u)'

echo "=== Chile ==="
curl -X POST http://localhost:1234/gql --data \
'USE "chile_users" MATCH (u) RETURN avg(u.age), count(u)'
```

---

## Errores Comunes y Soluciones

### ❌ Error 1: Comas en PROJECT

```gql
-- INCORRECTO
RETURN PROJECT("test", INCLUDE LABELS, INCLUDE PROPERTIES)
```

**Error:** `mismatched input ','`

```gql
-- CORRECTO (sin comas)
RETURN PROJECT("test" INCLUDE LABELS INCLUDE PROPERTIES)
```

### ❌ Error 2: Proyección Vacía

```gql
-- Si no hay aristas User->User, esto crea proyección vacía
MATCH (u:User)-[e]->(v:User) RETURN PROJECT("vacio")
```

**Solución:**
```bash
# Primero verificar que hay resultados
curl -X POST http://localhost:1234/gql --data \
'MATCH (u:User)-[e]->(v:User) RETURN count(e)'
# Si retorna 0, cambiar el patrón
```

### ❌ Error 3: Usar Funcionalidad No Incluida

```gql
-- Crear sin labels
MATCH (n)-[e]->(m) RETURN PROJECT("sin_labels")

-- Luego intentar usar labels
USE "sin_labels" MATCH (n:User) RETURN count(n)
```

**Error:** `Cannot use node labels with projection 'sin_labels'`

**Solución:** Recrear con `INCLUDE LABELS`

---

## Cuándo Usar Proyecciones

### ✅ SÍ usar cuando:

- Consultas repetitivas sobre el mismo subgrafo
- Necesitas rendimiento (5-10x más rápido)
- Trabajas con un subconjunto específico (categoría, región, periodo)
- Quieres aislar datos para análisis
- Bases de datos grandes (>1M nodos) y consultas sobre subconjuntos pequeños

### ❌ NO usar cuando:

- Consultas ad-hoc diferentes cada vez
- Subgrafo cambia constantemente
- Necesitas datos en tiempo real del grafo principal
- Espacio en disco limitado
- El "subgrafo" es casi todo el grafo (>80%)

---

## Resumen de Archivos

### Archivos de Código Clave

| Archivo | Líneas | Función |
|---------|--------|---------|
| `agg_project.h` | 49-121 | Inicialización: crear directorio, configurar features |
| `agg_project.h` | 123-379 | process(): extraer nodos, aristas, labels, propiedades |
| `agg_project.h` | 382-409 | get(): flush y retornar resultado |
| `query_visitor.cc` | 2354-2361 | Parsing de USE clause |
| `query_visitor.cc` | 2224-2290 | Evaluación de nombre de proyección |
| `query_context.cc` | 56-61 | Cargar contexto de proyección |
| `gql_model.cc` | 53-309 | Enrutamiento dinámico de índices (12 funciones get_*) |

### Tamaños Típicos

| Nodos | Solo Estructura | +Labels | +Properties | Completa |
|-------|-----------------|---------|-------------|----------|
| 100 | ~10 KB | ~30 KB | ~100 KB | ~150 KB |
| 1,000 | ~100 KB | ~300 KB | ~1 MB | ~1.5 MB |
| 10,000 | ~1 MB | ~3 MB | ~10 MB | ~15 MB |
| 100,000 | ~10 MB | ~30 MB | ~100 MB | ~150 MB |

*Nota: Tamaños aproximados, varían según número de propiedades/etiquetas*

---

## Comandos de Referencia Rápida

```bash
# Crear proyección básica
MATCH (n)-[e]->(m) RETURN PROJECT("nombre")

# Crear con todo
MATCH (n)-[e]->(m) RETURN PROJECT("nombre" INCLUDE LABELS INCLUDE PROPERTIES)

# Usar proyección
USE "nombre" MATCH (n) RETURN count(n)

# Volver al grafo principal
USE CURRENT_GRAPH MATCH (n) RETURN count(n)

# Ver tamaño en disco
du -sh data/dbs/gql/<db>/projections/<nombre>/

# Listar archivos
ls -lh data/dbs/gql/<db>/projections/<nombre>/
```

---

**Para documentación completa, ver:** `GUIA_PROJECT_Y_USE_GRAPH.md`
