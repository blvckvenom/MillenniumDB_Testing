#!/usr/bin/env bash
# migrate.sh: mueve las bases de datos del proyecto a la raíz /data.
# Diseño y justificación: docs/superpowers/specs/2026-09-02-data-root-design.md (§8).
#
# Uso:   migrate.sh --step N [--dry-run] [--commit]   N en 0..9
#        --commit: en el paso 4, además de copiar y verificar, borra el origen y enlaza.
# Cada paso comprueba su postcondición y salta lo ya hecho, así que se puede
# repetir. Solo el paso 4 cruza discos (rsync con segunda pasada de checksum);
# todo lo demás es rename dentro del mismo sistema de archivos.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data}
DISK2=${DISK2:-/mnt/data_disk/graphdata}
GROUP=${GROUP:-graphdata}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${MDB_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
LOG=${LOG:-$HOME/data-root-migrate.log}
JOB=$HOME/.claude/jobs/0729db73/tmp
CAMP=$DATA_ROOT/campaigns/2026-07-14-accuracy-small
S="sudo -n"
DRY=0; STEP=""; COMMIT=0

while [ $# -gt 0 ]; do
  case $1 in
    --step)    STEP=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --commit)  COMMIT=1; shift ;;
    *) echo "argumento desconocido: $1" >&2; exit 2 ;;
  esac
done
[ -n "$STEP" ] || { echo "uso: $0 --step N [--dry-run]" >&2; exit 2; }

log() { printf '%s %s\n' "$(date +%FT%T)" "$*" | tee -a "$LOG" >&2; }
run() { log "+ $*"; [ $DRY = 1 ] || "$@"; }
die() { log "ERROR: $*"; exit 1; }
dev() { stat -c %d "$1"; }

# sudo solo cuando el directorio afectado no es escribible por este proceso
# (el grupo nuevo no entra en vigor hasta la próxima sesión).
mvx()    { if [ -w "$(dirname "$1")" ] && [ -w "$(dirname "$2")" ]; then mv "$1" "$2"; else $S mv "$1" "$2"; fi; }
lnx()    { if [ -w "$(dirname "$2")" ]; then ln -s "$1" "$2"; else $S ln -s "$1" "$2"; fi; }
mkdirx() { if [ -e "$1" ]; then return 0; fi; if [ -w "$(dirname "$1")" ]; then mkdir -p "$1"; else $S mkdir -p "$1"; fi; }

# mv_link ORIGEN DESTINO_REAL DESTINO_LOGICO: mueve (mismo disco) y deja en
# ORIGEN un enlace al destino lógico. Idempotente.
mv_link() {
  local src=$1 dst=$2 tgt=$3
  if [ -L "$src" ]; then log "ya enlazado: $src -> $(readlink "$src")"; return 0; fi
  [ -e "$src" ] || die "no existe el origen $src"
  [ -e "$dst" ] && die "el destino ya existe: $dst"
  run mkdirx "$(dirname "$dst")"
  if [ $DRY = 0 ]; then
    [ "$(dev "$src")" = "$(dev "$(dirname "$dst")")" ] || die "$src y $dst están en discos distintos: mv copiaría"
  fi
  run mvx "$src" "$dst"
  run lnx "$tgt" "$src"
  if [ $DRY = 0 ]; then
    [ -d "$dst" ] && [ -L "$src" ] && [ -e "$src" ] || die "postcondición fallida: $src"
  fi
}

# ensure_link ENLACE DESTINO: crea el enlace si falta; falla si existe y apunta a otro lado.
ensure_link() {
  local link=$1 tgt=$2
  if [ -L "$link" ]; then
    [ "$(readlink "$link")" = "$tgt" ] || die "$link apunta a $(readlink "$link"), esperaba $tgt"
    return 0
  fi
  if [ -e "$link" ]; then
    [ $DRY = 1 ] && { log "(ensayo) $link aún existe; en la ejecución real se enlazaría a $tgt"; return 0; }
    die "$link existe y no es enlace"
  fi
  run mkdirx "$(dirname "$link")"
  run lnx "$tgt" "$link"
}

step0() {
  log "== paso 0: precondiciones =="
  if pgrep -a -x mdb >/dev/null; then pgrep -a -x mdb; die "hay un servidor mdb corriendo; detenlo antes"; fi
  if pgrep -a -f 'org.neo4j' >/dev/null; then pgrep -a -f 'org.neo4j'; die "hay un neo4j corriendo; detenlo antes"; fi
  $S -v >/dev/null 2>&1 || die "sudo sin contraseña no disponible"
  df -h / /mnt/data_disk | tee -a "$LOG"
  git -C "$REPO" status --short > "$HOME/data-root-git-status.before"
  log "git status guardado en $HOME/data-root-git-status.before ($(wc -l < "$HOME/data-root-git-status.before") líneas)"
}

step1() {
  log "== paso 1: grupo y esqueleto =="
  getent group "$GROUP" >/dev/null || run $S groupadd "$GROUP"
  id -nG "$USER" | grep -qw "$GROUP" || run $S usermod -aG "$GROUP" "$USER"
  local d
  for d in "$DATA_ROOT" "$DATA_ROOT"/{mdb,import,import/gql,raw,baselines,campaigns,scratch,bin} \
           "$DISK2" "$DISK2"/{mdb,raw,baselines}; do
    [ -d "$d" ] || run $S mkdir -p "$d"
    run $S chown "root:$GROUP" "$d"
    run $S chmod 2775 "$d"
  done
  run $S chmod 1777 "$DATA_ROOT/scratch"
  [ $DRY = 1 ] || stat -c '%a %U:%G %n' "$DATA_ROOT" "$DATA_ROOT"/* "$DISK2" "$DISK2"/* | tee -a "$LOG"
}

step2() {
  log "== paso 2: limpieza aprobada =="
  local orig="$REPO/docs/research/2026-07-14-accuracy-small/mdb_dbs" ds
  [ -e "$orig" ] || orig="$CAMP/mdb_dbs"
  # B1: NO es un duplicado. El cmp del 2026-09-02 mostró 93,98 M bytes distintos en
  # arxiv/sample_seed42/batches.dat y 596 archivos presentes en un solo lado: es una
  # repetición del 27-jul con otra configuración. Se conserva como campaña propia.
  local adj="$DATA_ROOT/campaigns/2026-07-27-accuracy-small-adj"
  if [ -d "$JOB/adj/dbs" ] && [ ! -L "$JOB/adj/dbs" ]; then
    log "B1 conservado: $(du -sh "$JOB/adj/dbs" | cut -f1) se mueven a $adj/mdb_dbs"
    mv_link "$JOB/adj/dbs" "$adj/mdb_dbs" "$adj/mdb_dbs"
  else log "B1 ya hecho"; fi
  # B2, B3: residuo del job cerrado.
  [ -d "$JOB/products_cmp" ] && { log "B2 borrando $(du -sh "$JOB/products_cmp" | cut -f1)"; run rm -rf "$JOB/products_cmp"; } || log "B2 ya hecho"
  local h; for h in "$JOB"/hnswarxiv.*; do [ -e "$h" ] && { log "B3 borrando $(du -sh "$h" | cut -f1) $h"; run rm -rf "$h"; }; done
  # B4, B5: efímeros de /tmp.
  [ -d /tmp/bench_sandbox ] && { log "B4 borrando $(du -sh /tmp/bench_sandbox | cut -f1)"; run rm -rf /tmp/bench_sandbox; } || log "B4 ya hecho"
  local old=/tmp/claude-1001/-home-bfuentes-MillenniumDB-Testing/702f768c-f941-4ba9-ae95-7bf7b222e38b/scratchpad
  [ -d "$old" ] && { log "B5 borrando $(du -sh "$old" | cut -f1)"; run rm -rf "$old"; } || log "B5 scratchpad ya hecho"
  run rm -rf /tmp/radix_* /tmp/test_output9.* /tmp/pf_gather
}

step3() {
  log "== paso 3: movimientos dentro del mismo disco =="
  # D5 campaña (antes que B1's enlace de compatibilidad)
  mv_link "$REPO/docs/research/2026-07-14-accuracy-small/mdb_dbs" "$CAMP/mdb_dbs" "$CAMP/mdb_dbs"
  mv_link "$REPO/docs/research/2026-07-14-accuracy-small/data"    "$CAMP/data"    "$CAMP/data"
  # D6 bases y fuentes del repo
  mv_link "$REPO/data/dbs/gql/cora_gnn" "$DATA_ROOT/mdb/cora_gnn" "$DATA_ROOT/mdb/cora_gnn"
  local ds; for ds in papers100M ogbn-arxiv cora; do
    mv_link "$REPO/data/example/gql/$ds" "$DATA_ROOT/import/gql/$ds" "$DATA_ROOT/import/gql/$ds"
  done
  # D4 neo4j cora
  mv_link "$HOME/neo4j_cora" "$DATA_ROOT/baselines/neo4j_cora" "$DATA_ROOT/baselines/neo4j_cora"
  # D7 cachés DGL y PyG
  mv_link "$HOME/.dgl/cora_v2_d697a464"           "$DATA_ROOT/raw/dgl/cora_v2_d697a464" "$DATA_ROOT/raw/dgl/cora_v2_d697a464"
  mv_link "$HOME/cora_xframework/planetoid/Cora"  "$DATA_ROOT/raw/planetoid/Cora"       "$DATA_ROOT/raw/planetoid/Cora"
  # D2, D3 baselines en el segundo disco: primero el enlace lógico en /data, luego el mv
  local b; for b in offgs_dataset offgs_small graphbolt_dataset; do
    ensure_link "$DATA_ROOT/baselines/$b" "$DISK2/baselines/$b"
    mv_link "/mnt/data_disk/bfuentes/$b" "$DISK2/baselines/$b" "$DATA_ROOT/baselines/$b"
  done
  ensure_link "$HOME/diskgnn_data" "$DATA_ROOT/baselines"
  # D1 OGB oficial
  ensure_link "$DATA_ROOT/raw/ogb" "$DISK2/raw/ogb"
  mv_link "/mnt/data_disk/svergara_tests/ogb" "$DISK2/raw/ogb" "$DATA_ROOT/raw/ogb"
  local oa=/mnt/data_disk/svergara_tests/ogb_arxiv
  if [ -d "$oa" ] && [ ! -L "$oa" ]; then
    if [ -d "$oa/ogbn_arxiv" ] && [ ! -e "$DISK2/raw/ogb/ogbn_arxiv" ] && [ "$(ls -A "$oa" | wc -l)" = 1 ]; then
      run mvx "$oa/ogbn_arxiv" "$DISK2/raw/ogb/ogbn_arxiv"
      run rmdir "$oa"
      run lnx "$DATA_ROOT/raw/ogb" "$oa"
    else
      mv_link "$oa" "$DISK2/raw/ogb_arxiv" "$DATA_ROOT/raw/ogb_arxiv"
      ensure_link "$DATA_ROOT/raw/ogb_arxiv" "$DISK2/raw/ogb_arxiv"
    fi
  fi
}

step4() {
  log "== paso 4: papers100M al segundo disco (único cruce de disco) =="
  local src="$REPO/data/dbs/gql/papers100M" dst="$DISK2/mdb/papers100M"
  if [ -L "$src" ]; then log "ya enlazado: $src -> $(readlink "$src")"; return 0; fi
  [ -d "$src" ] || die "no existe $src"
  # El marcador lleva la hora en que EMPEZÓ la última verificación limpia. Si ningún
  # archivo de origen o destino es más nuevo que él, la copia sigue verificada y no
  # hace falta repetir los ~9 min de checksum al volver con --commit.
  local marker="$HOME/data-root-checksum-ok.papers100M" verified=0
  if [ -f "$marker" ] && [ -d "$dst" ] && [ -z "$(find "$src" "$dst" -newer "$marker" -print -quit 2>/dev/null)" ]; then
    verified=1; log "checksum ya verificado (marcador del $(date -r "$marker" +%FT%T)) y nada cambió desde entonces: se omite la copia"
  fi
  if [ $verified = 0 ]; then
    run $S rsync -aHAX --info=progress2 "$src/" "$dst/"
    local started; started=$(date +%FT%T)
    log "segunda pasada con checksum (debe imprimir vacío)"
    local diff; diff=$($S rsync -aHAX --checksum --dry-run -i "$src/" "$dst/" || true)
    [ $DRY = 1 ] || [ -z "$diff" ] || die "checksum detectó diferencias, NO se borra el origen:
$diff"
    log "checksum limpio: $(du -sh "$dst" | cut -f1) en $dst"
    [ $DRY = 1 ] || touch -d "$started" "$marker"
  fi
  if [ $COMMIT = 0 ]; then log "copia verificada. Para borrar el origen y enlazar: migrate.sh --step 4 --commit"; return 0; fi
  run rm -rf "$src"
  ensure_link "$DATA_ROOT/mdb/papers100M" "$dst"
  ensure_link "$src" "$DATA_ROOT/mdb/papers100M"
}

step5() {
  log "== paso 5: ningún enlace colgante =="
  local d; d=$(find -L "$DATA_ROOT" "$REPO/data" "$REPO/docs/research/2026-07-14-accuracy-small" "$HOME/diskgnn_data" /mnt/data_disk/bfuentes /mnt/data_disk/svergara_tests -maxdepth 4 \( -name .venv -prune \) -o -type l -print 2>/dev/null || true)
  [ -z "$d" ] || die "enlaces colgantes:
$d"
  log "ok"
}

step6() {
  log "== paso 6: enlaces del repositorio y git status =="
  run bash "$HERE/link-repo.sh"
  git -C "$REPO" status --short > "$HOME/data-root-git-status.after"
  # cambios esperados de esta misma migración: .gitignore, onboard.sh, los scripts nuevos y la spec
  local expect='\.gitignore|scripts/onboard\.sh|scripts/data-root|docs/superpowers'
  if ! diff <(grep -vE "$expect" "$HOME/data-root-git-status.before") <(grep -vE "$expect" "$HOME/data-root-git-status.after"); then
    die "git status cambió (arriba); revisar .gitignore antes de seguir"
  fi
  log "git status idéntico al inicial"
}

step7() {
  log "== paso 7: rutas absolutas en store.meta =="
  run python3 "$HERE/fix_store_meta_path.py" "$DATA_ROOT" \
     --rewrite-prefix "$REPO/data/dbs/gql/papers100M" "$DATA_ROOT/mdb/papers100M" \
     --rewrite-prefix "$REPO/data/dbs/gql/cora_gnn"   "$DATA_ROOT/mdb/cora_gnn" \
     --rewrite-prefix "$REPO/docs/research/2026-07-14-accuracy-small/mdb_dbs" "$CAMP/mdb_dbs" \
     --rewrite-prefix "$JOB/adj/dbs" "$DATA_ROOT/campaigns/2026-07-27-accuracy-small-adj/mdb_dbs"
}

step8() {
  log "== paso 8: aceptación =="
  local fail=0
  # A1 contrato del refactor
  local acc; acc=$(cd "$REPO" && ./scripts/cora_fastloop.sh 2>&1 | tail -30)
  if echo "$acc" | grep -q 0.8648649; then log "A1 ok cora_fastloop = 0.8648649"; else log "A1 FALLA:"; echo "$acc" | tail -15 | tee -a "$LOG"; fail=1; fi
  # A2 papers100M abre y cuenta
  local port=1299 pid body
  "$REPO/build/Release/bin/mdb" server "$DATA_ROOT/mdb/papers100M" -p $port >"$HOME/data-root-a2-server.log" 2>&1 &
  pid=$!
  local i; for i in $(seq 1 60); do curl -s -o /dev/null "http://localhost:$port/" && break; sleep 1; done
  body=$(curl -s --max-time 300 -X POST "http://localhost:$port/gql" -H "Content-Type: text/plain" --data-binary 'MATCH (n) RETURN count(n)' || true)
  kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
  if echo "$body" | grep -q 111059956; then log "A2 ok count(n)=111059956"; else log "A2 FALLA: $body"; fail=1; fi
  "$REPO/build/Release/bin/mdb" list-projections "$DATA_ROOT/mdb/papers100M" 2>&1 | tee -a "$LOG" | grep -q papers100M_e2e_opt && log "A2 ok proyecciones" || { log "A2 FALLA proyecciones"; fail=1; }
  # A3, A4
  step5 && log "A3 ok" || fail=1
  diff -q <(grep -vE '\.gitignore|scripts/onboard\.sh|scripts/data-root|docs/superpowers' "$HOME/data-root-git-status.before") <(grep -vE '\.gitignore|scripts/onboard\.sh|scripts/data-root|docs/superpowers' "$HOME/data-root-git-status.after") >/dev/null && log "A4 ok git status" || { log "A4 FALLA"; fail=1; }
  # A6 espacio
  df -h / /mnt/data_disk | tee -a "$LOG"
  # A7 otro usuario lee
  $S -u svergara test -r "$DATA_ROOT/mdb/cora_gnn/catalog.dat" && log "A7 ok svergara lee mdb/cora_gnn" || { log "A7 FALLA"; fail=1; }
  # A8 referencias antes rotas
  local p; for p in "$HOME/diskgnn_data/offgs_dataset" "$HOME/diskgnn_data/graphbolt_dataset" "$HOME/diskgnn_data/offgs_dataset/ogbn-papers100M-offgs/graph.pth"; do
    [ -e "$p" ] && log "A8 ok $p" || { log "A8 FALLA $p"; fail=1; }
  done
  # A9 store.meta
  python3 "$HERE/fix_store_meta_path.py" "$DATA_ROOT" --expect-prefix "$DATA_ROOT" >/dev/null && log "A9 ok store.meta" || { log "A9 FALLA store.meta"; fail=1; }
  [ $fail = 0 ] || die "aceptación con fallas"
  log "aceptación completa"
}

step9() {
  log "== paso 9: documentación, índice y permisos =="
  run $S install -m 0644 -o root -g "$GROUP" "$HERE/README.data.md" "$DATA_ROOT/README.md"
  local f; for f in migrate.sh link-repo.sh check.sh data-index.sh fix_store_meta_path.py; do
    run $S install -m 0755 -o root -g "$GROUP" "$HERE/$f" "$DATA_ROOT/bin/$f"
  done
  # grupo y setgid en todo lo movido (chgrp -R no sigue enlaces: se hace en las dos raíces físicas)
  run $S chgrp -R "$GROUP" "$DATA_ROOT" "$DISK2"
  run $S find "$DATA_ROOT" "$DISK2" -type d -exec chmod g+ws {} +
  run $S chmod 1777 "$DATA_ROOT/scratch"
  run bash "$HERE/data-index.sh" "$DATA_ROOT/INDEX.md"
}

case $STEP in
  0) step0 ;; 1) step1 ;; 2) step2 ;; 3) step3 ;; 4) step4 ;;
  5) step5 ;; 6) step6 ;; 7) step7 ;; 8) step8 ;; 9) step9 ;;
  *) die "paso desconocido: $STEP" ;;
esac
