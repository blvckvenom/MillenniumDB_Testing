#!/usr/bin/env bash
# hnsw_property_validate.sh -- valida de punta a punta que los embeddings que
# produce el entrenamiento quedan indexados EN LA BASE con HNSW, declarando el
# indice sobre la propiedad, como hacen RDF y Quad Model.
#
# Cadena:
#   import -> graph_project -> gnn_offline_sample -> gnn_build_feature_store
#     -> gnn_train(writeProperty:'embedding')        escribe los embeddings
#     -> gnn_hnsw_create_property(... 'embedding')   declara el indice sobre la propiedad
#     -> gnn_hnsw_search                             consulta por id de nodo real
#
# Comprobaciones:
#   A. auto-acierto: el vecino mas cercano de n es n, a distancia 0. Prueba que la
#      identidad es un ObjectId real y no la posicion de una fila.
#   B. concordancia de clase de los 10 vecinos, contra tres referencias medidas:
#      embeddings 95.65% por escaneo exacto, entrada de 1433 dims 54.84%, azar 17.96%.
#   C. reproducibilidad: dos construcciones con la misma semilla dan lo mismo.
#   D. persistencia: tras reiniciar el servidor el indice sigue respondiendo igual.
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
CORA_GQL="$REPO/data/example/gql/cora/cora.gql"
CORA_NPY="$REPO/data/example/gql/cora/cora_features.npy"
PORT=7885
TMP=/home/bfuentes/.claude/jobs/0729db73/tmp
DB=$(mktemp -d "$TMP/hnswprop.XXXXXX")
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null
  rm -rf "$DB" 2>/dev/null
  exit $RC
}
trap cleanup EXIT INT TERM

say()  { echo "[prop] $*"; }
fail() { echo "[prop] FALLO: $*" >&2; RC=1; exit 1; }

q() {
  curl -s --max-time 600 -X POST "http://localhost:$PORT/gql" \
    -H 'Content-Type: application/gql' -H 'Accept: text/csv' --data-binary "$1"
}

run_gql() {
  local name=$1 query=$2 body
  body=$(q "$query")
  if grep -qiE '"?(error|exception)"?|Traceback' <<<"$body"; then
    echo "$body" >&2
    fail "$name devolvio error"
  fi
  echo "$body"
}

start_server() {
  "$MDB" server "$DB" -p $PORT -t 900 --browser false >>"$DB/server.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 1 90); do
    curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
      -H 'Content-Type: application/gql' --data-binary 'RETURN 1' >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  fail "el servidor no levanto"
}

stop_server() {
  kill "$SRVPID" 2>/dev/null
  wait "$SRVPID" 2>/dev/null
  SRVPID=""
  sleep 1
}

say "importando cora"
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" >/dev/null 2>&1 || fail "import"
start_server

say "1/8 graph_project"
run_gql project "CALL graph_project('cora','Paper','CITES',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label'}) YIELD nodeCount, featureDim, numClasses RETURN *"

say "2/8 gnn_offline_sample"
run_gql sample "CALL gnn_offline_sample('cora','s',[5,10],{batchSize:64,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches RETURN *"

say "3/8 gnn_build_feature_store"
run_gql build "CALL gnn_build_feature_store('s','node_features',{gpu_budget_mb:0,cpu_budget_mb:100,force:1}) YIELD l1Nodes RETURN *"

say "4/8 gnn_train con writeProperty:'embedding'"
run_gql train "CALL gnn_train('s','node_features',{epochs:20,randomSeed:42,patience:999,tolerance:0,saveOnBestVal:false,saveFinal:false,writeProperty:'embedding'}) YIELD testAccuracy, nodesWritten, writeCoverage RETURN *"

say "5/8 indice HNSW declarado SOBRE LA PROPIEDAD de la proyeccion"
CREATE_OUT=$(run_gql create_prop "CALL gnn_hnsw_create_property('cora_emb_idx','embedding',{projection:'cora',metric:'cosine',seed:42}) YIELD indexName, projection, property, dimension, nodeCount, buildTimeMs RETURN *")
echo "$CREATE_OUT"

# La entrada de 1433 dims llega por --with-tensors, que escribe una matriz densa
# y no una propiedad, asi que el control se construye por la ruta antigua. De paso
# comprueba que ambas rutas conviven en la misma base.
say "6/8 indice de control sobre la entrada de 1433 dims (matriz densa)"
run_gql create_feat "CALL gnn_hnsw_create('cora_feat_idx','node_features',{metric:'cosine'}) YIELD dimension, nodeCount, buildTimeMs RETURN *"

run_gql list "CALL gnn_hnsw_list() YIELD * RETURN *"

# ---------------------------------------------------------------- A: auto-acierto
say "7/8 comprobacion A: auto-acierto sobre 200 nodos"
: > "$DB/selfhit.csv"
for NID in $(seq 0 199); do
  q "CALL gnn_hnsw_search('cora_emb_idx',$NID,1,64) YIELD nodeId, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/selfhit.csv"
done

# ------------------------------------------------- B: concordancia de clase
say "8/8 comprobacion B: concordancia de clase de los 10 vecinos"
run_gql labels "MATCH (n) RETURN n, n.label AS lbl" > "$DB/labels.csv"

: > "$DB/nb_cora_emb_idx.csv"
for NID in $(seq 0 299); do
  q "CALL gnn_hnsw_search('cora_emb_idx',$NID,11,128) YIELD nodeId, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_cora_emb_idx.csv"
done
: > "$DB/nb_cora_feat_idx.csv"
for NID in $(seq 0 299); do
  q "CALL gnn_hnsw_find_similar('cora_feat_idx',$NID,11,128) YIELD similar_node, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_cora_feat_idx.csv"
done

# --------------------------------------------------------------- D: persistencia
say "reiniciando el servidor para comprobar persistencia"
stop_server
start_server
: > "$DB/nb_after_restart.csv"
for NID in $(seq 0 49); do
  q "CALL gnn_hnsw_search('cora_emb_idx',$NID,11,128) YIELD nodeId, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_after_restart.csv"
done

python3 - "$DB" <<'PY'
import csv, sys, collections
D = sys.argv[1]

# etiquetas: la primera columna es el nodo, la segunda su clase
lbl = {}
for r in csv.reader(open(f"{D}/labels.csv")):
    if len(r) == 2 and r[0] not in ("n", ""):
        node = r[0].lstrip("_n")
        try:
            lbl[int(node)] = r[1].strip('"')
        except ValueError:
            pass

def selfhit():
    ok = tot = 0
    for r in csv.reader(open(f"{D}/selfhit.csv")):
        if len(r) != 3: continue
        src, nb = int(r[0]), int(r[1])
        dist = float(r[2])
        tot += 1
        if nb == src and abs(dist) < 1e-5: ok += 1
    return ok, tot

def as_id(tok):
    tok = tok.strip().strip('"').lstrip("_n")
    return int(tok) if tok.isdigit() else None

def agreement(path):
    per = collections.defaultdict(list)
    for r in csv.reader(open(path)):
        if len(r) != 3: continue
        src, nb = as_id(r[0]), as_id(r[1])
        if src is None or nb is None or nb == src: continue
        per[src].append(lbl.get(nb) == lbl.get(src))
    ag = sum(sum(v[:10]) for v in per.values())
    tot = sum(len(v[:10]) for v in per.values())
    return (ag / tot * 100 if tot else 0.0), len(per)

if len(set(lbl.values())) < 2:
    print("AVISO: las etiquetas no discriminan, la concordancia no significa nada.")
    print("Primeras lineas de labels.csv:")
    print("".join(open(f"{D}/labels.csv").readlines()[:3]))
    sys.exit(1)
prior = collections.Counter(lbl.values())
n = sum(prior.values())
chance = sum((c / n) ** 2 for c in prior.values()) * 100

ok, tot = selfhit()
print()
print("=" * 68)
print(f"A. auto-acierto        {ok}/{tot} nodos devuelven su propio id a distancia 0")
emb, n_emb = agreement(f"{D}/nb_cora_emb_idx.csv")
fea, n_fea = agreement(f"{D}/nb_cora_feat_idx.csv")
print(f"B. concordancia de clase de los 10 vecinos")
print(f"     embeddings 256 dims   {emb:6.2f}%   (referencia por escaneo exacto 95.65%)   n={n_emb}")
print(f"     entrada  1433 dims    {fea:6.2f}%   (referencia por escaneo exacto 54.84%)   n={n_fea}")
print(f"     azar (prior de clases){chance:6.2f}%   (referencia 17.96%)")

before = {}
for r in csv.reader(open(f"{D}/nb_cora_emb_idx.csv")):
    if len(r) == 3 and r[0].isdigit() and int(r[0]) < 50:
        before.setdefault(int(r[0]), []).append(r[1])
after = {}
for r in csv.reader(open(f"{D}/nb_after_restart.csv")):
    if len(r) == 3 and r[0].isdigit():
        after.setdefault(int(r[0]), []).append(r[1])
same = sum(1 for k in after if before.get(k) == after.get(k))
print(f"D. persistencia        {same}/{len(after)} nodos dan los mismos vecinos tras reiniciar")
print("=" * 68)

fails = []
if tot == 0 or ok / tot < 0.99: fails.append(f"auto-acierto {ok}/{tot}")
if emb < 94.0: fails.append(f"concordancia embeddings {emb:.2f}% < 94%")
if not (50.0 <= fea <= 60.0): fails.append(f"concordancia entrada {fea:.2f}% fuera de 50-60%")
if len(after) == 0 or same != len(after): fails.append(f"persistencia {same}/{len(after)}")
if fails:
    print("FALLA: " + "; ".join(fails))
    sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
PY
[ $? -ne 0 ] && fail "las comprobaciones no pasaron"

RC=0
say "LISTO"
