#!/usr/bin/env bash
# hnsw_over_embeddings.sh — indexa con HNSW los embeddings de 256 dims que produce
# el pipeline de GNN, y compara sus vecinos contra los del espacio de entrada.
#
# No toca una linea de codigo. El truco: `register_gnn_feature` esta cableado a
# "node_features" en el importador, asi que en vez de registrar una feature nueva,
# importamos una SEGUNDA base donde los embeddings OCUPAN el lugar de las features.
# HNSW entonces los indexa sin cambio alguno, y podemos medir sus vecinos.
#
# Cuidado con el orden de filas: HNSW reconstruye el nodo como MASK_NODE|fila, o sea
# que asume fila == ordinal del nodo. embeddings.npy viene en orden de BATCH, asi que
# hay que dispersarlo por seed_ids antes de importar o los vecinos serian de otros nodos.
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
PORT=7884
RC=1
DB1=""; DB2=""; SRVPID=""

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null || true
  [ -n "$DB1" ] && rm -rf "$DB1"
  [ -n "$DB2" ] && rm -rf "$DB2"
  exit $RC
}
trap cleanup EXIT INT TERM
say() { echo "[hnsw-emb] $*"; }
fail() { echo "[hnsw-emb] FALLO: $*" >&2; exit 1; }

# 1) pipeline de GNN sobre cora, conservando la base para leer los .npy exportados
say "1/5 corriendo el pipeline de GNN en cora"
"$REPO/scripts/hnsw_gnn_smoke.sh" --keep >/tmp/hnsw_emb_pipe.log 2>&1
DB1=$(grep -oP "DB en \K/tmp/\S+" /tmp/hnsw_emb_pipe.log) || fail "no salio la base"
[ -d "$DB1" ] || fail "base no encontrada"
G="$DB1/projections/cora/gnn_output/default"

# 2) dispersar embeddings a orden de nodo y guardarlos como .npy de entrada
say "2/5 dispersando embeddings a orden de nodo"
python3 - "$G" <<'PY' || exit 1
import numpy as np, sys
G = sys.argv[1]
E = np.load(f"{G}/embeddings.npy")                       # [N,256] en orden de BATCH
S = np.load(f"{G}/seed_ids.npy").astype(np.uint64)
ordinals = (S & np.uint64(0x00FFFFFFFFFFFFFF)).astype(np.int64)
N, D = E.shape
out = np.zeros((N, D), dtype=np.float32)
out[ordinals] = E                                        # scatter: fila = ordinal del nodo
assert len(np.unique(ordinals)) == N, "ordinales no biyectivos"
np.save("/tmp/cora_embeddings_nodeorder.npy", np.ascontiguousarray(out))
print(f"    [{N},{D}] float32 C-order, ordinales {ordinals.min()}..{ordinals.max()}")
PY

# 3) segunda base donde los embeddings ocupan el lugar de las features
say "3/5 importando la base espejo con los embeddings como features"
DB2=$(mktemp -d /tmp/cora_emb.XXXXXX)
"$MDB" import "$CORA_GQL" "$DB2" --with-tensors /tmp/cora_embeddings_nodeorder.npy \
  >/dev/null 2>&1 || fail "import"

say "4/5 server e indice HNSW de 256 dims"
"$MDB" server "$DB2" -p $PORT -t 900 --browser false >"$DB2/server.log" 2>&1 &
SRVPID=$!
for _ in $(seq 1 60); do
  curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
    -H 'Content-Type: application/gql' --data-binary 'RETURN 1' >/dev/null 2>&1 && break
  sleep 0.5
done
q() { curl -s --max-time 300 -X POST "http://localhost:$PORT/gql" \
        -H 'Content-Type: application/gql' -H 'Accept: text/csv' --data-binary "$1"; }

q "CALL gnn_hnsw_create('emb_idx','node_features',{metric:'cosine'}) YIELD * RETURN *"
q "CALL gnn_hnsw_list() YIELD * RETURN *"

# 5) concordancia de clase de los vecinos que devuelve el INDICE
say "5/5 midiendo concordancia de clase de los vecinos del indice"
CLS=/tmp/cora_classes.csv
q "MATCH (n) RETURN n, n.class_name AS cls" > "$CLS"
: > /tmp/hnsw_neighbors.csv
for NID in $(seq 0 199); do
  q "CALL gnn_hnsw_find_similar('emb_idx',$NID,11,200) YIELD similar_node, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> /tmp/hnsw_neighbors.csv
done

python3 - <<'PY'
import csv, collections
cls={}
for r in csv.reader(open("/tmp/cora_classes.csv")):
    if len(r)==2 and r[0].startswith("_n"): cls[r[0]]=r[1].strip('"')
agree=tot=0
per=collections.defaultdict(list)
for r in csv.reader(open("/tmp/hnsw_neighbors.csv")):
    if len(r)!=3: continue
    src,nb,_=r; src=f"_n{src}"
    if nb==src: continue                       # el propio nodo
    if src in cls and nb in cls:
        per[src].append(cls[nb]==cls[src])
for s,v in per.items():
    agree+=sum(v[:10]); tot+=len(v[:10])
print(f"    HNSW sobre embeddings de 256 dims: {agree/tot*100:.2f}% de los 10 vecinos comparten clase  (n={len(per)} nodos)")
print( "    referencia medida antes por escaneo exacto: embeddings 95.65% | entrada 1433 dims 54.84% | azar 17.96%")
PY

RC=0
say "LISTO"
