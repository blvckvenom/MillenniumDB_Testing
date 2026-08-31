#!/bin/bash
# Test rápido de proyecciones - Sintaxis correcta

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
SERVER="http://localhost:1234/gql"

echo "╔══════════════════════════════════════════╗"
echo "║  TESTS RÁPIDOS DE PROYECCIONES           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Función helper
q() {
    echo ">>> $1"
    curl -s -X POST "$SERVER" --data-binary "$2"
    echo ""
    echo ""
}

# ===========================================
# SINTAXIS CORRECTA (sin comas):
# PROJECT("name" INCLUDE LABELS INCLUDE PROPERTIES)
# ===========================================

echo "1. CREAR PROYECCIONES"
echo "━━━━━━━━━━━━━━━━━━━━━━"
q "Proyección básica" \
  'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test1")'

q "Proyección con LABELS" \
  'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test2" INCLUDE LABELS)'

q "Proyección con PROPERTIES" \
  'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test3" INCLUDE PROPERTIES)'

q "Proyección con TODO" \
  'MATCH (u:User)-[e]->(v:User) RETURN PROJECT("test4" INCLUDE LABELS INCLUDE PROPERTIES)'

echo ""
echo "2. TESTS DE LABELS"
echo "━━━━━━━━━━━━━━━━━━━━━━"
q "Contar Users (con labels)" \
  'USE "test2" MATCH (u:User) RETURN count(u)'

q "Contar Admins (con labels)" \
  'USE "test2" MATCH (u:Admin) RETURN count(u)'

q "Contar Guests (con labels)" \
  'USE "test2" MATCH (u:Guest) RETURN count(u)'

echo ""
echo "3. TESTS DE PROPERTIES"
echo "━━━━━━━━━━━━━━━━━━━━━━"
q "Ver properties (con properties)" \
  'USE "test3" MATCH (u) RETURN u.name, u.age LIMIT 5'

q "Ver properties (con todo)" \
  'USE "test4" MATCH (u:User) RETURN u.name, u.age, u.email LIMIT 5'

echo ""
echo "4. COMPARACIÓN CON GRAFO PRINCIPAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
q "Grafo principal - contar Users" \
  'MATCH (u:User) RETURN count(u)'

q "Proyección - contar Users" \
  'USE "test2" MATCH (u:User) RETURN count(u)'

echo "╔══════════════════════════════════════════╗"
echo "║  TESTS COMPLETADOS                       ║"
echo "╚══════════════════════════════════════════╝"
