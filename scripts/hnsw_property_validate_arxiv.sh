#!/usr/bin/env bash
# hnsw_property_validate_arxiv.sh -- el mismo camino que hnsw_property_validate.sh
# pero sobre ogbn-arxiv (169 343 nodos), para MEDIR la tasa de construccion del
# indice y poder extrapolar a los 1,55 M nodos del split oficial de papers100M.
#
# Lo que interesa medir aqui, en orden:
#   1. tasa de construccion en nodos/s del indice declarado sobre la propiedad
#   2. auto-acierto, o sea que la identidad sea un ObjectId real
#   3. concordancia de clase de los vecinos contra el azar y contra la entrada
#   4. persistencia tras reiniciar el servidor
set -u -o pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB="$REPO/build/Release/bin/mdb"
DSET="$REPO/data/example/gql/ogbn-arxiv/ogbn-arxiv"
GQL_FILE="$DSET/ogbn_arxiv.gql"
NPY="$DSET/ogbn_arxiv_features.npy"
PORT=7886
TMP=/home/bfuentes/.claude/jobs/0729db73/tmp
DB=$(mktemp -d "$TMP/hnswarxiv.XXXXXX")
SRVPID=""
RC=1

cleanup() {
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
  [ -n "$SRVPID" ] && wait "$SRVPID" 2>/dev/null
  rm -rf "$DB" 2>/dev/null
  exit $RC
}
trap cleanup EXIT INT TERM

say()  { echo "[arxiv-hnsw] $*"; }
fail() { echo "[arxiv-hnsw] FALLO: $*" >&2; RC=1; exit 1; }

q() {
  curl -s --max-time 3600 -X POST "http://localhost:$PORT/gql" \
    -H 'Content-Type: application/gql' -H 'Accept: text/csv' --data-binary "$1"
}
run_gql() {
  local name=$1 query=$2 body
  body=$(q "$query")
  if grep -qiE '"?(error|exception)"?|Traceback' <<<"$body"; then
    echo "$body" >&2; fail "$name devolvio error"
  fi
  echo "$body"
}
start_server() {
  "$MDB" server "$DB" -p $PORT -t 3600 --browser false >>"$DB/server.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 1 120); do
    curl -s --max-time 2 -X POST "http://localhost:$PORT/gql" \
      -H 'Content-Type: application/gql' --data-binary 'RETURN 1' >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  fail "el servidor no levanto"
}
stop_server() { kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 1; }

[ -f "$GQL_FILE" ] || fail "falta $GQL_FILE"

say "importando ogbn-arxiv"
"$MDB" import "$GQL_FILE" "$DB" --with-tensors "$NPY" >/dev/null 2>&1 || fail "import"
start_server

say "1/6 graph_project"
run_gql project "CALL graph_project('arxiv_gnn','Node','CONNECTS',{orientation:'UNDIRECTED',includeFeatures:'node_features',labelProperty:'label',splitProperty:'split'}) YIELD nodeCount, featureDim, numClasses RETURN *"

say "2/6 gnn_offline_sample"
run_gql sample "CALL gnn_offline_sample('arxiv_gnn','arxiv_s',[10,15],{batchSize:1024,randomSeed:42,orientation:'UNDIRECTED'}) YIELD totalBatches, uniqueNodes RETURN *"

say "3/6 gnn_build_feature_store"
run_gql build "CALL gnn_build_feature_store('arxiv_s','node_features',{gpu_budget_mb:2048,cpu_budget_mb:512}) YIELD l1Nodes RETURN *"

say "4/6 gnn_train con writeProperty:'embedding'"
run_gql train "CALL gnn_train('arxiv_s','node_features',{model:'graphsage',hiddenDim:256,epochs:30,lr:0.01,dropout:0.5,patience:10,randomSeed:42,writeProperty:'embedding'}) YIELD testAccuracy, nodesWritten, writeCoverage, writeMillis RETURN *"

say "5/6 indice HNSW declarado sobre la propiedad  <-- ESTE ES EL NUMERO QUE IMPORTA"
# M=48 y efConstruction=500 en vez de los 16/200 por defecto. Medido: con los
# valores por defecto el auto-acierto se queda en 195-198/200 sobre 169 343 nodos,
# porque un 2-3% de los nodos quedan inalcanzables en el grafo, no porque la
# identidad falle (la distancia que reporta el indice coincide exactamente con la
# distancia coseno real). Con 48/500 sube a 200/200 a costa de 3x el tiempo de
# construccion. Es una propiedad del HNSW de la casa, no de este camino.
run_gql create_prop "CALL gnn_hnsw_create_property('arxiv_emb_idx','embedding',{projection:'arxiv_gnn',metric:'cosine',seed:42,M:48,efConstruction:500}) YIELD indexName, projection, dimension, nodeCount, buildTimeMs RETURN *" | tee "$DB/create.csv"

say "    indice de control sobre la entrada de 128 dims (matriz densa)"
run_gql create_feat "CALL gnn_hnsw_create('arxiv_feat_idx','node_features',{metric:'cosine'}) YIELD dimension, nodeCount, buildTimeMs RETURN *"

say "6/6 comprobaciones de correccion"
run_gql labels "MATCH (n) RETURN n, n.label AS lbl" > "$DB/labels.csv"

# El auto-acierto con ef bajo puede fallar por recuperacion aproximada, que es
# esperable, o por identidad rota, que no lo es. Para distinguirlos, cada fallo se
# reintenta con ef alto: si con ef alto acierta, era recall; si no, es identidad.
: > "$DB/selfhit.csv"
: > "$DB/selfhit_retry.csv"
for NID in $(seq 0 199); do
  OUT=$(q "CALL gnn_hnsw_search('arxiv_emb_idx',$NID,1,200) YIELD nodeId, distance RETURN *" | tail -n +2)
  echo "$NID,$OUT" >> "$DB/selfhit.csv"
  GOT=$(echo "$OUT" | cut -d, -f1)
  if [ "$GOT" != "$NID" ]; then
    RETRY=$(q "CALL gnn_hnsw_search('arxiv_emb_idx',$NID,1,2000) YIELD nodeId, distance RETURN *" | tail -n +2)
    echo "$NID,$RETRY" >> "$DB/selfhit_retry.csv"
  fi
done

# CONTROL: el mismo auto-acierto sobre el indice de matriz densa, que es codigo
# distinto y preexistente. Si tambien falla, es el HNSW de la casa y no este camino.
: > "$DB/selfhit_raw.csv"
for NID in $(seq 0 199); do
  OUT=$(q "CALL gnn_hnsw_find_similar('arxiv_feat_idx',$NID,1,2000) YIELD similar_node, distance RETURN *" | tail -n +2)
  echo "$NID,$OUT" >> "$DB/selfhit_raw.csv"
done

: > "$DB/nb_emb.csv"
for NID in $(seq 0 299); do
  q "CALL gnn_hnsw_search('arxiv_emb_idx',$NID,11,128) YIELD nodeId, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_emb.csv"
done
: > "$DB/nb_feat.csv"
for NID in $(seq 0 299); do
  q "CALL gnn_hnsw_find_similar('arxiv_feat_idx',$NID,11,128) YIELD similar_node, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_feat.csv"
done

say "reiniciando el servidor para comprobar persistencia"
stop_server; start_server
: > "$DB/nb_after.csv"
for NID in $(seq 0 49); do
  q "CALL gnn_hnsw_search('arxiv_emb_idx',$NID,11,128) YIELD nodeId, distance RETURN *" \
    | tail -n +2 | sed "s/^/$NID,/" >> "$DB/nb_after.csv"
done

python3 - "$DB" <<'PY'
import csv, sys, collections
D = sys.argv[1]

lbl = {}
for r in csv.reader(open(f"{D}/labels.csv")):
    if len(r) == 2 and r[0] not in ("n", ""):
        tok = r[0].lstrip("_n")
        if tok.isdigit():
            lbl[int(tok)] = r[1].strip('"')

if len(set(lbl.values())) < 2:
    print("AVISO: las etiquetas no discriminan, la concordancia no significa nada.")
    print("".join(open(f"{D}/labels.csv").readlines()[:3])); sys.exit(1)

def as_id(t):
    t = t.strip().strip('"').lstrip("_n")
    return int(t) if t.isdigit() else None

def agreement(path):
    per = collections.defaultdict(list)
    for r in csv.reader(open(path)):
        if len(r) != 3: continue
        s, nb = as_id(r[0]), as_id(r[1])
        if s is None or nb is None or nb == s: continue
        per[s].append(lbl.get(nb) == lbl.get(s))
    ag = sum(sum(v[:10]) for v in per.values())
    tot = sum(len(v[:10]) for v in per.values())
    return (ag / tot * 100 if tot else 0.0), len(per)

ok = tot = 0
misses = []
for r in csv.reader(open(f"{D}/selfhit.csv")):
    if len(r) != 3: continue
    tot += 1
    if int(r[1]) == int(r[0]) and abs(float(r[2])) < 1e-5: ok += 1
    else: misses.append((int(r[0]), int(r[1]), float(r[2])))

retry_ok, retry_rows = 0, []
try:
    for r in csv.reader(open(f"{D}/selfhit_retry.csv")):
        if len(r) != 3: continue
        retry_rows.append((int(r[0]), int(r[1]), float(r[2])))
        if int(r[1]) == int(r[0]) and abs(float(r[2])) < 1e-5: retry_ok += 1
except FileNotFoundError:
    pass

raw_ok = raw_tot = 0
raw_miss = []
try:
    for r in csv.reader(open(f"{D}/selfhit_raw.csv")):
        if len(r) != 3: continue
        raw_tot += 1
        nb = as_id(r[1])
        if nb == int(r[0]) and abs(float(r[2])) < 1e-5: raw_ok += 1
        else: raw_miss.append((int(r[0]), nb, round(float(r[2]), 4)))
except FileNotFoundError:
    pass

prior = collections.Counter(lbl.values()); n = sum(prior.values())
chance = sum((c/n)**2 for c in prior.values()) * 100

build = list(csv.reader(open(f"{D}/create.csv")))
hdr, row = build[0], build[-1]
nodes = int(row[hdr.index("nodeCount")])
ms = int(row[hdr.index("buildTimeMs")])

emb, n_emb = agreement(f"{D}/nb_emb.csv")
fea, n_fea = agreement(f"{D}/nb_feat.csv")

before, after = {}, {}
for r in csv.reader(open(f"{D}/nb_emb.csv")):
    if len(r) == 3 and r[0].isdigit() and int(r[0]) < 50: before.setdefault(int(r[0]), []).append(r[1])
for r in csv.reader(open(f"{D}/nb_after.csv")):
    if len(r) == 3 and r[0].isdigit(): after.setdefault(int(r[0]), []).append(r[1])
same = sum(1 for k in after if before.get(k) == after.get(k))

rate = nodes / (ms/1000) if ms else 0
print()
print("=" * 74)
print(f"CONSTRUCCION  {nodes} nodos en {ms/1000:.1f} s  =  {rate:.0f} nodos/s")
print(f"              extrapolado a 1 546 782 nodos (split de papers100M): {1546782/rate/60:.1f} min")
print(f"              (HNSW es ~N log N, asi que es cota optimista, ver nota)")
print(f"A. auto-acierto        {ok}/{tot} nodos devuelven su propio id a distancia 0 (ef=200)")
if misses:
    print(f"   fallos              : {[(a,b,round(c,4)) for a,b,c in misses[:6]]}")
    print(f"   reintento con ef=2000: {retry_ok}/{len(retry_rows)} aciertan  ->", 
          "era recuperacion aproximada, la identidad esta bien" if retry_ok == len(retry_rows) and retry_rows
          else "NO se recupera: revisar identidad")
    print(f"   detalle del reintento: {[(a,b,round(c,4)) for a,b,c in retry_rows[:6]]}")
print(f"   CONTROL matriz densa (camino antiguo, ef=2000): {raw_ok}/{raw_tot} aciertan")
if raw_miss: print(f"   fallos del control : {raw_miss[:6]}")
print(f"B. concordancia de clase de los 10 vecinos")
print(f"     embeddings 256 dims   {emb:6.2f}%   n={n_emb}")
print(f"     entrada    128 dims   {fea:6.2f}%   n={n_fea}")
print(f"     azar (prior de clases){chance:6.2f}%")
print(f"D. persistencia        {same}/{len(after)} nodos dan los mismos vecinos tras reiniciar")
print("=" * 74)

fails = []
# El gate real es que TODO fallo se recupere subiendo ef. Eso separa recall de identidad.
if tot == 0: fails.append("sin datos de auto-acierto")
elif retry_rows and retry_ok != len(retry_rows):
    fails.append(f"auto-acierto no se recupera con ef alto: {retry_ok}/{len(retry_rows)}")
elif ok/tot < 1.0: fails.append(f"auto-acierto {ok}/{tot}")
if emb <= fea: fails.append(f"los embeddings ({emb:.2f}%) no superan a la entrada ({fea:.2f}%)")
if emb <= chance * 1.5: fails.append(f"concordancia {emb:.2f}% no supera claramente el azar {chance:.2f}%")
if len(after) == 0 or same != len(after): fails.append(f"persistencia {same}/{len(after)}")
if fails:
    print("FALLA: " + "; ".join(fails)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
PY
[ $? -ne 0 ] && fail "las comprobaciones no pasaron"

RC=0
say "LISTO"
