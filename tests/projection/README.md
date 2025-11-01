# Tests del Sistema de Proyecciones (PROJECT/USE GRAPH)

**Última actualización**: 20 de Octubre, 2025

---

## 📋 Índice

1. [Estructura de Tests](#estructura-de-tests)
2. [Tests Unitarios (C++)](#tests-unitarios-c)
3. [Tests de Integración (Bash)](#tests-de-integración-bash)
4. [Cómo Ejecutar los Tests](#cómo-ejecutar-los-tests)
5. [Agregar Nuevos Tests](#agregar-nuevos-tests)

---

## 🗂️ Estructura de Tests

```
tests_projection/
├── README.md (este archivo) .................. Guía de tests de proyecciones
│
├── unit_tests/ ............................... Tests unitarios en C++
│   ├── projection_storage_test.cc ............ Test de ProjectionStorage básico
│   ├── projection_features_test.cc ........... Test de backward compatibility
│   ├── projection_inspect.cc ................. Herramienta de inspección de proyecciones
│   └── projection_inspect_enhanced.cc ........ Herramienta de inspección mejorada
│
├── integration_tests/ ........................ Tests de integración (vacío por ahora)
│   └── (tests Python futuros aquí)
│
└── scripts/ .................................. Scripts de test bash
    ├── test_label_support.sh ................. Test end-to-end de INCLUDE LABELS
    └── test_quick.sh ......................... Test rápido de todas las features
```

---

## 🧪 Tests Unitarios (C++)

Los tests unitarios están en `unit_tests/` y se compilan junto con el proyecto principal.

### 1. `projection_storage_test.cc`

**Propósito**: Verificar que ProjectionStorage compila, enlaza y funciona correctamente.

**Lo que prueba**:
- Inicialización de ProjectionManager
- Creación de una proyección
- ProjectionCatalog (guardar/cargar metadatos)
- ProjectionStorage (crear índices B+Tree)
- Adición de nodos y aristas
- Finalization y persistencia

**Cómo compilar**:
```bash
cd /home/benito/B_MillenniumDB/MillenniumDB
cmake --build build/Release --target projection_storage_test
```

**Cómo ejecutar**:
```bash
./build/Release/bin/projection_storage_test
```

**Salida esperada**:
```
Testing projection storage layer...
Test 1: ProjectionManager initialization... OK
Test 2: Creating projection... OK (dir: test_db/projections/test_projection)
Test 3: ProjectionCatalog... OK
Test 4: ProjectionStorage... OK
Test 5: Adding nodes... OK (added 10 nodes)
Test 6: Adding edges... OK (added 20 edges)
Test 7: Finalization... OK

All tests passed!
```

---

### 2. `projection_features_test.cc`

**Propósito**: Test de backward compatibility y features opcionales.

**Lo que prueba**:
- Proyección v1.0 (solo topología, sin labels/properties)
- Proyección con INCLUDE LABELS solamente
- Proyección con INCLUDE PROPERTIES solamente
- Proyección completa (INCLUDE LABELS + INCLUDE PROPERTIES)
- Verificación de que índices opcionales NO se crean cuando no se requieren

**Cómo compilar**:
```bash
cmake --build build/Release --target projection_features_test
```

**Cómo ejecutar**:
```bash
./build/Release/bin/projection_features_test
```

**Salida esperada**:
```
=== Phase A3: Backward Compatibility & Optional Features Test ===

Test 1: Creating v1.0-style projection (topology only)... OK
Test 2: Verify no label/property indexes created... OK
Test 3: Creating projection with INCLUDE LABELS... OK
Test 4: Verify label indexes exist, property indexes don't... OK
Test 5: Creating projection with INCLUDE PROPERTIES... OK
Test 6: Verify property indexes exist, label indexes don't... OK
Test 7: Creating projection with both INCLUDE LABELS + INCLUDE PROPERTIES... OK
Test 8: Verify all optional indexes exist... OK

All Phase A3 tests passed!
```

---

### 3. `projection_inspect.cc`

**Propósito**: Herramienta de inspección para examinar proyecciones en disco.

**Uso**:
```bash
./build/Release/bin/projection_inspect <db_folder> <projection_name>
```

**Ejemplo**:
```bash
# Crear proyección primero
./build/Release/bin/mdb import data/example/gql/posts/posts.gql test_db
./build/Release/bin/mdb server test_db &
curl -X POST http://localhost:1234/gql --data \
  'MATCH (n)-[e]->(m) RETURN PROJECT("my_proj" INCLUDE LABELS INCLUDE PROPERTIES)'

# Inspeccionar proyección
./build/Release/bin/projection_inspect test_db my_proj
```

**Salida esperada**:
```
=== Projection Inspection Tool ===
Projection: my_proj
Location: test_db/projections/my_proj/

--- Catalog Information ---
Projection name: my_proj
Node count: 50
Edge count: 75
Creation timestamp: 1729397654

--- File Structure ---
✅ catalog.dat (512 bytes)
✅ node_id.dir (4096 bytes)
✅ node_id.leaf (8192 bytes)
✅ from_to_edge.dir (4096 bytes)
✅ from_to_edge.leaf (12288 bytes)
✅ to_from_edge.dir (4096 bytes)
✅ to_from_edge.leaf (12288 bytes)
✅ node_label.dir (4096 bytes)
✅ node_label.leaf (8192 bytes)
✅ label_node.dir (4096 bytes)
✅ label_node.leaf (8192 bytes)
✅ node_key_value.dir (4096 bytes)
✅ node_key_value.leaf (16384 bytes)
✅ key_value_node.dir (4096 bytes)
✅ key_value_node.leaf (16384 bytes)

Total size: 127.5 KB
```

---

### 4. `projection_inspect_enhanced.cc`

**Propósito**: Versión mejorada de projection_inspect con más detalles.

**Funcionalidad adicional**:
- Muestra primeros 10 registros de cada índice
- Decodifica ObjectIds para mostrar valores legibles
- Verifica integridad de referencias
- Estadísticas detalladas

**Uso**:
```bash
./build/Release/bin/projection_inspect_enhanced <db_folder> <projection_name>
```

---

## 🔄 Tests de Integración (Bash)

Los scripts de integración están en `scripts/` y prueban el sistema completo end-to-end.

### 1. `test_label_support.sh`

**Propósito**: Test completo de la funcionalidad INCLUDE LABELS.

**Lo que prueba**:
1. Importar database de ejemplo
2. Iniciar servidor
3. Crear proyección sin labels (backward compatibility)
4. Crear proyección con labels
5. Verificar que queries con labels funcionan en proyección con labels
6. Verificar que queries con labels fallan con mensaje descriptivo en proyección sin labels
7. Contar nodos/aristas con y sin filtros de label

**Cómo ejecutar**:
```bash
cd /home/benito/B_MillenniumDB/MillenniumDB/tests_projection/scripts
./test_label_support.sh
```

**Duración**: ~30 segundos

**Salida esperada**:
```
======================================
Phase 1: Label Support Testing
======================================

1. Cleaning up previous test database...
2. Importing test database...
3. Starting server...
   Server PID: 12345

4. Test: Create projection WITHOUT labels (backward compatibility)
   Query: MATCH (p:Paper)-[e:Cites]->(q:Paper) RETURN PROJECT("test_no_labels")
   [OK] Projection created

5. Test: Create projection WITH labels
   Query: MATCH (p:Paper)-[e:Cites]->(q:Paper) RETURN PROJECT("test_with_labels", INCLUDE LABELS)
   [OK] Projection created

6. Test: Query projection without labels (should fail with helpful error)
   [OK] Error message shown

7. Test: Query projection with labels - count nodes with Paper label
   Query: USE "test_with_labels" MATCH (p:Paper) RETURN count(p)
   Result: 2708

8. Test: Query projection with labels - count edges with Cites label
   Query: USE "test_with_labels" MATCH ()-[e:Cites]->() RETURN count(e)
   Result: 5429

======================================
Phase 1 Testing Complete!
======================================
```

---

### 2. `test_quick.sh`

**Propósito**: Test rápido de todas las features (labels + properties).

**Lo que prueba**:
1. Crear 4 proyecciones diferentes:
   - Básica (solo estructura)
   - Con INCLUDE LABELS
   - Con INCLUDE PROPERTIES
   - Con INCLUDE LABELS + INCLUDE PROPERTIES
2. Queries de labels (MATCH (u:User))
3. Queries de properties (RETURN u.name, u.age)
4. Comparación entre grafo principal y proyección

**Cómo ejecutar**:
```bash
cd /home/benito/B_MillenniumDB/MillenniumDB/tests_projection/scripts
./test_quick.sh
```

**Duración**: ~15 segundos

**Prerequisito**: Servidor ya iniciado en puerto 1234.

**Salida esperada**:
```
╔══════════════════════════════════════════╗
║  TESTS RÁPIDOS DE PROYECCIONES           ║
╚══════════════════════════════════════════╝

1. CREAR PROYECCIONES
━━━━━━━━━━━━━━━━━━━━━━
>>> Proyección básica
{"status":"ok","message":"Projection 'test1' created"}

>>> Proyección con LABELS
{"status":"ok","message":"Projection 'test2' created"}

>>> Proyección con PROPERTIES
{"status":"ok","message":"Projection 'test3' created"}

>>> Proyección con TODO
{"status":"ok","message":"Projection 'test4' created"}

2. TESTS DE LABELS
━━━━━━━━━━━━━━━━━━━━━━
>>> Contar Users (con labels)
{"count":50}

3. TESTS DE PROPERTIES
━━━━━━━━━━━━━━━━━━━━━━
>>> Ver properties (con properties)
name,age
"Alice Johnson",34
"Bob Smith",28
"Charlie Brown",45
...

╔══════════════════════════════════════════╗
║  TESTS COMPLETADOS                       ║
╚══════════════════════════════════════════╝
```

---

## 🚀 Cómo Ejecutar los Tests

### Ejecutar TODOS los Tests del Proyecto

```bash
cd /home/benito/B_MillenniumDB/MillenniumDB
./scripts/run-tests
```

Este script ejecuta:
- Tests unitarios (ctest)
- Tests de integración SPARQL
- Tests de integración MQL
- Tests de integración GQL

**Nota**: Los tests de proyecciones están incluidos en los tests GQL.

---

### Ejecutar Solo Tests Unitarios de Proyecciones

```bash
cd /home/benito/B_MillenniumDB/MillenniumDB

# Compilar tests
cmake --build build/Release --target projection_storage_test
cmake --build build/Release --target projection_features_test

# Ejecutar
./build/Release/bin/projection_storage_test
./build/Release/bin/projection_features_test
```

---

### Ejecutar Solo Tests de Integración de Proyecciones

```bash
# 1. Iniciar servidor
./build/Release/bin/mdb import data/example/gql/posts/posts.gql test_db
./build/Release/bin/mdb server test_db --port 1234 --browser false &

# 2. Ejecutar tests
cd tests_projection/scripts
./test_quick.sh
./test_label_support.sh

# 3. Detener servidor
pkill -f "mdb server"
```

---

## ➕ Agregar Nuevos Tests

### Test Unitario (C++)

1. **Crear archivo** en `tests_projection/unit_tests/`:
   ```bash
   touch tests_projection/unit_tests/my_new_test.cc
   ```

2. **Agregar al CMakeLists.txt**:
   ```cmake
   # En src/tests/CMakeLists.txt o CMakeLists.txt principal
   add_executable(my_new_test
       ${CMAKE_SOURCE_DIR}/tests_projection/unit_tests/my_new_test.cc
   )
   target_link_libraries(my_new_test
       graph_models
       storage
       query
   )
   ```

3. **Compilar y ejecutar**:
   ```bash
   cmake --build build/Release --target my_new_test
   ./build/Release/bin/my_new_test
   ```

---

### Test de Integración (Bash)

1. **Crear script** en `tests_projection/scripts/`:
   ```bash
   touch tests_projection/scripts/test_my_feature.sh
   chmod +x tests_projection/scripts/test_my_feature.sh
   ```

2. **Estructura recomendada**:
   ```bash
   #!/bin/bash
   set -e

   echo "==================================="
   echo "Test: My Feature"
   echo "==================================="

   DB_DIR="test_my_feature_db"
   BIN="./build/Release/bin/mdb"
   SERVER="http://localhost:1234/gql"

   # Cleanup
   rm -rf "$DB_DIR" 2>/dev/null || true

   # Setup
   $BIN import data/example/gql/posts/posts.gql "$DB_DIR"
   $BIN server "$DB_DIR" --port 1234 --browser false > /tmp/test.log 2>&1 &
   SERVER_PID=$!
   sleep 3

   # Test 1
   echo "Test 1: ..."
   curl -s -X POST "$SERVER" --data 'YOUR QUERY HERE'

   # Cleanup
   kill $SERVER_PID
   rm -rf "$DB_DIR"

   echo "Test completed!"
   ```

3. **Ejecutar**:
   ```bash
   cd tests_projection/scripts
   ./test_my_feature.sh
   ```

---

## 📊 Cobertura de Tests Actual

### Features Cubiertas ✅

| Feature | Test Unitario | Test Integración | Estado |
|---------|--------------|------------------|--------|
| PROJECT básico (estructura) | ✅ projection_storage_test.cc | ✅ test_quick.sh | ✅ |
| INCLUDE LABELS | ✅ projection_features_test.cc | ✅ test_label_support.sh | ✅ |
| INCLUDE PROPERTIES | ✅ projection_features_test.cc | ✅ test_quick.sh | ✅ |
| USE GRAPH | ⚠️ Implícito | ✅ test_quick.sh | ✅ |
| USE CURRENT_GRAPH | ❌ | ⚠️ Parcial | ⚠️ |
| ProjectionCatalog | ✅ projection_storage_test.cc | ❌ | ✅ |
| ProjectionManager | ✅ projection_storage_test.cc | ❌ | ✅ |
| Backward compatibility | ✅ projection_features_test.cc | ❌ | ✅ |

---

### Features Pendientes de Testing ⚠️

| Feature | Prioridad | Esfuerzo | Notas |
|---------|-----------|----------|-------|
| Proyecciones con patrones complejos | Media | 1-2 días | Test con OPTIONAL, UNION, etc. |
| Proyecciones con múltiples labels | Alta | 1 día | MATCH (n:User:Admin) |
| Proyecciones con properties NULL | Media | 0.5 días | Valores NULL explícitos |
| Concurrencia (múltiples USE) | Baja | 2-3 días | Threads simultáneos |
| Persistencia tras reinicio | Alta | 1 día | Crear, reiniciar servidor, USE |
| Error handling exhaustivo | Media | 2-3 días | Todos los edge cases |

---

## 🐛 Debugging de Tests Fallidos

### Test Unitario Falla

1. **Compilar en modo Debug**:
   ```bash
   cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
   cmake --build build/Debug --target <test_name>
   ```

2. **Ejecutar con GDB**:
   ```bash
   gdb ./build/Debug/bin/<test_name>
   (gdb) run
   (gdb) backtrace  # Si crashea
   ```

3. **Ver logs detallados**:
   - Agregar `std::cerr << "DEBUG: ..." << std::endl;` en puntos críticos
   - Recompilar y ejecutar

---

### Test de Integración Falla

1. **Ver logs del servidor**:
   ```bash
   tail -f /tmp/test.log  # O donde apunte el script
   ```

2. **Ejecutar queries manualmente**:
   ```bash
   curl -X POST http://localhost:1234/gql --data 'QUERY AQUÍ'
   ```

3. **Inspeccionar proyección en disco**:
   ```bash
   ./build/Release/bin/projection_inspect test_db projection_name
   ```

4. **Verificar que el servidor esté corriendo**:
   ```bash
   ps aux | grep mdb
   curl http://localhost:1234/ping
   ```

---

## 📞 Contacto

Para reportar bugs en tests o sugerir nuevos tests:
- **GitHub Issues**: https://github.com/MillenniumDB/MillenniumDB/issues
- **Tag**: `[tests]` o `[projection-tests]`

---

## 📜 Historial

### v2.0 (Octubre 20, 2025)
- ✅ Reorganizado tests en `tests_projection/`
- ✅ Separados tests unitarios vs integración
- ✅ Creado README.md completo
- ✅ Copiados tests desde `src/tests/`

### v1.0 (Octubre 17, 2025)
- ✅ Tests iniciales de ProjectionStorage
- ✅ Tests de features opcionales (labels, properties)
- ✅ Scripts de integración bash

---

**Estado de tests**: ✅ Cobertura básica completa, mejoras pendientes
**Próxima actualización**: Al agregar tests de concurrencia y error handling
