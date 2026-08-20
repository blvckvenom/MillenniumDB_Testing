#!/usr/bin/env bash
# hnsw_rdf_qm_regression.sh -- ejercita el camino HNSW de RDF y de Quad Model de
# punta a punta y deja la salida en un archivo, para poder diffear el binario de
# antes contra el de despues de tocar HNSWIndex::index_single.
#
# Uso: hnsw_rdf_qm_regression.sh <etiqueta>
# Deja el resultado en /home/bfuentes/.claude/jobs/0729db73/tmp/hnsw_reg_<etiqueta>.txt
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
TAG="${1:?falta la etiqueta}"
OUTDIR=/home/bfuentes/.claude/jobs/0729db73/tmp
OUT="$OUTDIR/hnsw_reg_${TAG}.txt"
PORT_RDF=7891
PORT_QM=7892
WORK=$(mktemp -d "$OUTDIR/hnswreg.XXXXXX")
SRV=""

cleanup() {
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
  sleep 1
  rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

: > "$OUT"
say() { echo "[reg:$TAG] $*"; }
emit() { echo "$@" >> "$OUT"; }

wait_up() {
  local port=$1 body=$2 ctype=$3 ROUTE=${4:-}
  for _ in $(seq 1 90); do
    curl -s --max-time 2 -X POST "http://localhost:$port/$ROUTE" \
      -H "Content-Type: $ctype" --data-binary "$body" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  return 1
}

############################### RDF ###################################
say "importando RDF cora"
"$MDB" import "$REPO/data/example/rdf/cora/cora.ttl" "$WORK/rdf" >/dev/null 2>&1 \
  || { emit "RDF import: FALLO"; exit 1; }

"$MDB" server "$WORK/rdf" -p $PORT_RDF -t 600 --browser false >"$WORK/rdf.log" 2>&1 &
SRV=$!
wait_up $PORT_RDF 'SELECT * WHERE { ?s ?p ?o } LIMIT 1' 'application/sparql-query' sparql \
  || { emit "RDF server: NO LEVANTO"; exit 1; }

sq() { curl -s --max-time 300 -X POST "http://localhost:$PORT_RDF/sparql" \
        -H 'Content-Type: application/sparql-query' -H 'Accept: text/csv' --data-binary "$1"; }
su() { curl -s --max-time 600 -X POST "http://localhost:$PORT_RDF/update" \
        -H 'Content-Type: application/sparql-update' --data-binary "$1"; }

say "creando indice HNSW en RDF"
CREATE_OUT=$(su 'PREFIX ex: <http://example.com/>
CREATE HNSW INDEX "cora_idx" WITH {
  "predicate"     = ex:embedding,
  "dimension"     = 1433,
  "maxCandidates" = 128,
  "maxEdges"      = 16,
  "metric"        = "cosineDistance"
}')
emit "== RDF create =="
emit "$CREATE_OUT"

say "consultando HNSW_TOP_K en RDF"
emit "== RDF HNSW_TOP_K (ex:101811, k=10) =="
sq 'PREFIX ex: <http://example.com/>
SELECT ?similar ?distance WHERE {
  BIND(ex:101811 AS ?fixed)
  ?fixed ex:embedding ?emb .
  HNSW_TOP_K("cora_idx", ?emb, 10, 200) AS (?similar, ?distance)
}' | sort >> "$OUT"

emit "== RDF conteo de tensores (control) =="
sq 'PREFIX ex: <http://example.com/>
SELECT (COUNT(*) AS ?n) WHERE { ?s ex:embedding ?o }' >> "$OUT"

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
sleep 1

########################### Quad Model ################################
say "importando Quad Model cora"
"$MDB" import "$REPO/data/example/qm/cora/cora.qm" "$WORK/qm" >/dev/null 2>&1 \
  || { emit "QM import: FALLO"; exit 1; }

"$MDB" server "$WORK/qm" -p $PORT_QM -t 600 --browser false >"$WORK/qm.log" 2>&1 &
SRV=$!
wait_up $PORT_QM 'MATCH (?n) RETURN ?n LIMIT 1' 'application/mql' \
  || { emit "QM server: NO LEVANTO"; exit 1; }

mq() { curl -s --max-time 600 -X POST "http://localhost:$PORT_QM/" \
        -H 'Content-Type: application/mql' -H 'Accept: text/csv' --data-binary "$1"; }

say "creando indice HNSW en Quad Model"
emit "== QM create =="
mq 'CREATE HNSW INDEX "cora_idx" WITH {
  "property"      = "embedding",
  "dimension"     = 1433,
  "maxCandidates" = 128,
  "maxEdges"      = 16,
  "metric"        = "cosineDistance"
}' >> "$OUT"

say "consultando HNSW_TOP_K en Quad Model"
emit "== QM HNSW_TOP_K (art101811, k=10) =="
mq 'MATCH (?a) WHERE ?a == art101811
CALL HNSW_TOP_K("cora_idx", ?a.embedding, 10, 200)
     YIELD ?object AS ?similar, ?distance
RETURN ?similar, ?distance' | sort >> "$OUT"

emit "== QM conteo de tensores (control) =="
mq 'MATCH (?a) WHERE ?a.embedding IS NOT NULL RETURN COUNT(*) AS ?n' >> "$OUT"

say "listo -> $OUT"
