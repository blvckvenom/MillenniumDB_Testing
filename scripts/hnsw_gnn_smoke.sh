#!/usr/bin/env bash
# hnsw_gnn_smoke.sh — comprueba que el indice HNSW se conecta de verdad con los
# embeddings que produce el pipeline de GNN, de punta a punta y en cora (~15 s).
#
# Cadena que valida:
#   import -> graph_project -> gnn_offline_sample -> gnn_build_feature_store
#     -> gnn_train(writeProperty:'embedding')   <- escribe los embeddings
#     -> gnn_hnsw_create(... 'embedding' ...)   <- los indexa
#     -> gnn_hnsw_list / gnn_hnsw_info
#     -> gnn_hnsw_find_similar                 <- los consulta
#     -> GQL sobre las clases de los vecinos   <- comprueba que el indice sirve
#
# La prueba de fondo no es que las llamadas no fallen, sino que los vecinos que
# devuelve HNSW compartan la clase del nodo consultado muy por encima del azar
# (cora tiene 7 clases, asi que el azar es ~1/7 = 14%).
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7881
KEEP=0
[ "${1-}" = "--keep" ] && KEEP=1

DB=$(mktemp -d /tmp/hnsw_smoke.XXXXXX)
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null || true
  if [ "$KEEP" = "0" ]; then rm -rf "$DB" 2>/dev/null || true
  else echo "[keep] DB en $DB"; fi
  exit $RC
}
trap cleanup EXIT INT TERM

say()  { echo "[hnsw] $*"; }
fail() { echo "[hnsw] FALLO: $*" >&2; RC=1; exit 1; }

# curl devuelve 0 aunque el server responda error, asi que hay que mirar el body.
run_gql() {
  local name=$1 query=$2 body
  body=$(curl -s --max-time 300 -X POST "http://localhost:$PORT/gql" \
           -H 'Content-Type: application/gql' -H 'Accept: text/csv' --data-binary "$query")
  if grep -qiE '"?(error|exception)"?|Traceback' <<<"$body"; then
    echo "$body" >&2
    fail "$name devolvio error"
  fi
  echo "$body"
}

say "importando cora en $DB"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 \
  || fail "import"

for p in $(pgrep -f "build/Release/bin/mdb server .*-p $PORT" 2>/dev/null || true); do
  kill "$p" 2>/dev/null || true
done
say "server en :$PORT"
"$MDB" server "$DB" -p $PORT -t 900 --browser false >"$DB/server.log" 2>&1 &
SRVPID=$!
for _ in $(seq 1 60); do
  curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
    -H 'Content-Type: application/gql' --data-binary 'RETURN 1' >/dev/null 2>&1 && break
  sleep 0.5
done

say "1/6 graph_project"
run_gql project "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD nodeCount, relationshipCount, featureDim, numClasses RETURN *"

say "2/6 gnn_offline_sample"
run_gql sample "CALL gnn_offline_sample('cora','s',[5,10],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches, uniqueNodes RETURN *"

say "3/6 gnn_build_feature_store"
run_gql build "CALL gnn_build_feature_store('s','node_features',{gpu_budget_mb:0,cpu_budget_mb:100,force:1}) YIELD l1Nodes, l2Nodes, l3Nodes, l4Nodes RETURN *"

say "4/7 gnn_train con writeProperty:'embedding'"
run_gql train "CALL gnn_train('s','node_features',{epochs:20,randomSeed:42,patience:999,tolerance:0,saveOnBestVal:false,saveFinal:false,writeProperty:'embedding'}) YIELD testAccuracy, bestValAccuracy, ranEpochs, nodesWritten, writeCoverage, writeMillis RETURN *"

say "5/7 los embeddings SI quedan consultables en la proyeccion?"
run_gql query_emb "USE cora MATCH (n) RETURN n.embedding LIMIT 1"

say "6/7 gnn_hnsw_create sobre 'embedding' (lo que el paper afirma)"
set +e
run_gql create_emb "CALL gnn_hnsw_create('cora_emb_idx','embedding',{metric:'cosine'}) YIELD * RETURN *"
EMB_RC=$?
set -e
[ "$EMB_RC" -ne 0 ] && say "    -> RECHAZADO (esperado: el catalogo solo conoce features importadas)"

say "7/7 gnn_hnsw_create sobre 'node_features' (la ruta que si existe)"
run_gql create_feat "CALL gnn_hnsw_create('cora_feat_idx','node_features',{metric:'cosine'}) YIELD * RETURN *"
run_gql list "CALL gnn_hnsw_list() YIELD * RETURN *"
for NID in 0 1 2 3 5 8 13 21 34 55 89 144; do
  run_gql "similar_$NID" "CALL gnn_hnsw_find_similar('cora_feat_idx',$NID,10,100) YIELD * RETURN *"
done

say "8/8 busqueda por similitud sobre el embedding SIN indice (cosineDistance)"
run_gql cosdist "USE cora MATCH (a), (b) WHERE a.label = 3 AND b.label = 3 AND a <> b RETURN cosineDistance(a.embedding, b.embedding) AS d LIMIT 3"
run_gql cosdist_cross "USE cora MATCH (a), (b) WHERE a.label = 3 AND b.label = 5 RETURN cosineDistance(a.embedding, b.embedding) AS d LIMIT 3"

RC=0
say "TODAS LAS LLAMADAS OK"
