#!/bin/bash
# Script de prueba exhaustiva para la función PROJECT
set -e

DB_DIR="/tmp/test_projection_investigation"
BIN="./build/Release/bin/mdb"
SERVER="http://localhost:1235/gql"
LOG="/tmp/project_investigation.log"

echo "=========================================="
echo "INVESTIGACIÓN EXHAUSTIVA DE PROJECT"
echo "=========================================="
echo ""

# Cleanup
rm -rf "$DB_DIR" 2>/dev/null || true
rm -f "$LOG" 2>/dev/null || true

# Setup database
echo "1. Importando base de datos de prueba..."
$BIN import data/example/gql/posts/posts.gql "$DB_DIR" > /dev/null 2>&1
echo "   ✓ Base de datos importada"
echo ""

# Start server
echo "2. Iniciando servidor..."
$BIN server "$DB_DIR" --port 1235 --browser false > "$LOG" 2>&1 &
SERVER_PID=$!
sleep 4
echo "   ✓ Servidor iniciado (PID: $SERVER_PID)"
echo ""

# Helper function para queries
query() {
    local desc="$1"
    local q="$2"
    local expected_success="$3"  # "yes" o "no"

    echo "TEST: $desc"
    echo "  Query: $q"

    result=$(curl -s -X POST "$SERVER" --data "$q" 2>&1)

    if echo "$result" | grep -q "error\|Error\|Exception"; then
        if [ "$expected_success" == "yes" ]; then
            echo "  ✗ FALLO: Se esperaba éxito pero hubo error"
            echo "  Respuesta: $(echo $result | head -c 200)"
        else
            echo "  ✓ Error esperado recibido"
        fi
    else
        if [ "$expected_success" == "no" ]; then
            echo "  ✗ FALLO: Se esperaba error pero fue exitoso"
            echo "  Respuesta: $(echo $result | head -c 200)"
        else
            echo "  ✓ Exitoso"
            echo "  Respuesta: $(echo $result | head -c 200)"
        fi
    fi
    echo ""
}

echo "=========================================="
echo "CATEGORÍA 1: PROYECCIONES BÁSICAS"
echo "=========================================="
echo ""

query "1.1: Crear proyección básica (sin opciones)" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_basic\")" \
    "yes"

query "1.2: Usar proyección básica para contar nodos" \
    "USE \"test_basic\" MATCH (n) RETURN count(n)" \
    "yes"

query "1.3: Usar proyección básica para contar edges" \
    "USE \"test_basic\" MATCH ()-[e]->() RETURN count(e)" \
    "yes"

query "1.4: Volver al grafo principal" \
    "USE CURRENT_GRAPH MATCH (n) RETURN count(n)" \
    "yes"

echo "=========================================="
echo "CATEGORÍA 2: INCLUDE LABELS"
echo "=========================================="
echo ""

query "2.1: Crear proyección con INCLUDE LABELS" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_labels\" INCLUDE LABELS)" \
    "yes"

query "2.2: Query con filtro de label en proyección" \
    "USE \"test_labels\" MATCH (n:User) RETURN count(n)" \
    "yes"

query "2.3: Query con label en edge" \
    "USE \"test_labels\" MATCH ()-[e:Follows]->() RETURN count(e)" \
    "yes"

query "2.4: Intentar usar label en proyección sin INCLUDE LABELS (debe fallar)" \
    "USE \"test_basic\" MATCH (n:User) RETURN count(n)" \
    "no"

echo "=========================================="
echo "CATEGORÍA 3: INCLUDE PROPERTIES"
echo "=========================================="
echo ""

query "3.1: Crear proyección con INCLUDE PROPERTIES" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_props\" INCLUDE PROPERTIES)" \
    "yes"

query "3.2: Acceder a propiedades de nodos" \
    "USE \"test_props\" MATCH (n) RETURN n.name, n.age LIMIT 5" \
    "yes"

query "3.3: Filtrar por propiedades" \
    "USE \"test_props\" MATCH (n) WHERE n.age > 25 RETURN n.name LIMIT 5" \
    "yes"

query "3.4: Acceder a propiedades de edges" \
    "USE \"test_props\" MATCH ()-[e]->(m) RETURN e.weight LIMIT 5" \
    "yes"

query "3.5: Intentar acceder a properties sin INCLUDE PROPERTIES (debe fallar)" \
    "USE \"test_basic\" MATCH (n) RETURN n.name LIMIT 5" \
    "no"

echo "=========================================="
echo "CATEGORÍA 4: INCLUDE LABELS + PROPERTIES"
echo "=========================================="
echo ""

query "4.1: Crear proyección con ambas opciones" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_full\" INCLUDE LABELS INCLUDE PROPERTIES)" \
    "yes"

query "4.2: Query combinando labels y properties" \
    "USE \"test_full\" MATCH (n:User) WHERE n.age > 25 RETURN n.name, n.age LIMIT 5" \
    "yes"

query "4.3: Query compleja con labels en edges y properties" \
    "USE \"test_full\" MATCH (n:User)-[e:Follows]->(m:User) RETURN n.name, m.name LIMIT 5" \
    "yes"

echo "=========================================="
echo "CATEGORÍA 5: PATRONES DE MATCH VARIADOS"
echo "=========================================="
echo ""

query "5.1: Proyección con patrón específico (solo un tipo de edge)" \
    "MATCH (n)-[e:Follows]->(m) RETURN PROJECT(\"test_follows\" INCLUDE LABELS)" \
    "yes"

query "5.2: Verificar que solo tiene edges del tipo especificado" \
    "USE \"test_follows\" MATCH ()-[e]->() RETURN count(e)" \
    "yes"

query "5.3: Proyección con patrón de nodos específicos" \
    "MATCH (n:User)-[e]->(m:User) RETURN PROJECT(\"test_users\" INCLUDE LABELS)" \
    "yes"

echo "=========================================="
echo "CATEGORÍA 6: EDGE CASES Y LÍMITES"
echo "=========================================="
echo ""

query "6.1: Crear proyección vacía (patrón que no matchea nada)" \
    "MATCH (n:NonExistent)-[e]->(m) RETURN PROJECT(\"test_empty\" INCLUDE LABELS)" \
    "yes"

query "6.2: Contar nodos en proyección vacía" \
    "USE \"test_empty\" MATCH (n) RETURN count(n)" \
    "yes"

query "6.3: Proyección con nombre muy largo" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_very_long_name_for_projection_testing_purposes_123456789\" INCLUDE LABELS) LIMIT 1" \
    "yes"

query "6.4: Proyección con caracteres especiales en nombre" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test-projection-with-dashes\" INCLUDE LABELS) LIMIT 1" \
    "yes"

echo "=========================================="
echo "CATEGORÍA 7: PERSISTENCIA"
echo "=========================================="
echo ""

echo "TEST: 7.1: Verificar archivos en disco"
if [ -d "$DB_DIR/projections" ]; then
    echo "  ✓ Directorio de proyecciones existe"
    proj_count=$(ls -1 "$DB_DIR/projections" | wc -l)
    echo "  ✓ Número de proyecciones: $proj_count"

    # Listar proyecciones
    for proj in "$DB_DIR/projections"/*; do
        if [ -d "$proj" ]; then
            proj_name=$(basename "$proj")
            echo "  - $proj_name:"
            file_count=$(ls -1 "$proj" | wc -l)
            size=$(du -sh "$proj" | cut -f1)
            echo "    Archivos: $file_count, Tamaño: $size"
        fi
    done
else
    echo "  ✗ FALLO: Directorio de proyecciones no existe"
fi
echo ""

echo "=========================================="
echo "CATEGORÍA 8: ANÁLISIS DE FEATURES FALTANTES"
echo "=========================================="
echo ""

# Tests para features que NO deberían funcionar (aún no implementadas)

echo "TEST: 8.1: UNDIRECTED relationship projection"
echo "  (Neo4j GDS tiene NATURAL, REVERSE, UNDIRECTED)"
query "8.1a: Intentar especificar orientación de edge (no soportado)" \
    "MATCH (n)-[e]-(m) RETURN PROJECT(\"test_undirected\" INCLUDE LABELS)" \
    "yes"
echo "  NOTA: Debería soportar ORIENTATION: UNDIRECTED en el futuro"
echo ""

echo "TEST: 8.2: Aggregation en proyección"
echo "  (Neo4j GDS permite: aggregation: COUNT, MIN, MAX, SUM)"
query "8.2a: Intentar crear proyección con aggregation COUNT" \
    "MATCH (n)-[e]->(m) WITH n, m, count(e) as rel_count RETURN PROJECT(\"test_agg\")" \
    "no"
echo "  NOTA: Aggregation de relationships no está soportada"
echo ""

echo "TEST: 8.3: Node/Relationship properties filtering"
echo "  (Neo4j GDS permite seleccionar qué properties incluir)"
echo "  NOTA: MillenniumDB incluye TODAS las properties con INCLUDE PROPERTIES"
echo "  Neo4j: nodeProperties: ['age', 'name'], relationshipProperties: ['weight']"
echo ""

echo "TEST: 8.4: Default property values"
echo "  (Neo4j GDS permite: defaultValue para properties faltantes)"
echo "  NOTA: MillenniumDB no soporta valores default para properties NULL"
echo ""

echo "TEST: 8.5: Actualización de proyecciones (MUTATE)"
echo "  (Neo4j GDS tiene MUTATE mode para escribir resultados)"
echo "  NOTA: Proyecciones en MillenniumDB son STATIC (no actualizables)"
echo ""

echo "=========================================="
echo "CATEGORÍA 9: INSPECCIÓN DE PROYECCIONES"
echo "=========================================="
echo ""

echo "TEST: 9.1: Listar todas las proyecciones disponibles"
echo "  NOTA: No hay comando LIST PROJECTIONS implementado"
echo "  Workaround: ls $DB_DIR/projections/"
ls -1 "$DB_DIR/projections/" 2>/dev/null | head -10
echo ""

echo "TEST: 9.2: Obtener metadata de proyección"
echo "  NOTA: No hay comando DESCRIBE PROJECTION implementado"
echo "  Workaround: Usar projection_inspect tool"
if [ -f "./build/Release/bin/projection_inspect" ]; then
    echo "  ✓ projection_inspect disponible"
else
    echo "  ✗ projection_inspect no compilado"
fi
echo ""

echo "=========================================="
echo "CATEGORÍA 10: PERFORMANCE Y OPTIMIZACIÓN"
echo "=========================================="
echo ""

echo "TEST: 10.1: Tiempo de creación de proyección"
start_time=$(date +%s%N)
query "10.1a: Crear proyección grande" \
    "MATCH (n)-[e]->(m) RETURN PROJECT(\"test_perf\" INCLUDE LABELS INCLUDE PROPERTIES)" \
    "yes"
end_time=$(date +%s%N)
elapsed=$(( (end_time - start_time) / 1000000 ))
echo "  Tiempo: ${elapsed}ms"
echo ""

echo "TEST: 10.2: Comparar performance query en main graph vs projection"
start_time=$(date +%s%N)
curl -s -X POST "$SERVER" --data 'USE CURRENT_GRAPH MATCH (n) RETURN count(n)' > /dev/null
end_time=$(date +%s%N)
main_time=$(( (end_time - start_time) / 1000000 ))

start_time=$(date +%s%N)
curl -s -X POST "$SERVER" --data 'USE "test_perf" MATCH (n) RETURN count(n)' > /dev/null
end_time=$(date +%s%N)
proj_time=$(( (end_time - start_time) / 1000000 ))

echo "  Query en main graph: ${main_time}ms"
echo "  Query en projection: ${proj_time}ms"
if [ $proj_time -lt $main_time ]; then
    speedup=$((main_time / (proj_time + 1)))
    echo "  ✓ Projection es ${speedup}x más rápida"
else
    echo "  ⚠ Projection no es más rápida (tamaño similar?)"
fi
echo ""

# Cleanup
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true

echo "=========================================="
echo "INVESTIGACIÓN COMPLETADA"
echo "=========================================="
echo ""
echo "Log del servidor guardado en: $LOG"
echo "Base de datos de prueba en: $DB_DIR"
echo ""
echo "Para inspeccionar proyecciones manualmente:"
echo "  ls -la $DB_DIR/projections/"
echo "  ./build/Release/bin/projection_inspect $DB_DIR <projection_name>"
echo ""
