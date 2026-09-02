#!/usr/bin/env bash
# check.sh: salud de la raíz de datos /data. Nunca modifica nada. Exit 1 si algo falla.
# Lo llama scripts/onboard.sh; también sirve solo.
set -uo pipefail
DATA_ROOT=${DATA_ROOT:-/data}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${MDB_REPO:-$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HOME/MillenniumDB_Testing")}
fail=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  FALLA $*"; fail=1; }
warn() { echo "  aviso $*"; }

[ -d "$DATA_ROOT" ] || { echo "no existe $DATA_ROOT"; exit 1; }

echo "[check] enlaces colgantes"
d=$(find -L "$DATA_ROOT" "$REPO/data" "$REPO/docs/research/2026-07-14-accuracy-small" "$HOME/diskgnn_data" -maxdepth 4 \( -name .venv -prune \) -o -type l -print 2>/dev/null)
if [ -z "$d" ]; then ok "ninguno"; else bad "$d"; fi

echo "[check] cada base en data/dbs/gql debe ser un enlace a $DATA_ROOT"
for p in "$REPO"/data/dbs/gql/*/; do
  p=${p%/}; [ -e "$p" ] || continue
  if [ -L "$p" ]; then ok "$(basename "$p") -> $(readlink "$p")"; else bad "$(basename "$p") es un directorio real dentro del repo: muévelo a $DATA_ROOT/mdb y enlaza"; fi
done

echo "[check] rutas grabadas en store.meta"
python3 "$HERE/fix_store_meta_path.py" "$DATA_ROOT" --expect-prefix "$DATA_ROOT" | grep -E 'INESPERADO|existe en disco: False|revisados' | sed 's/^/  /'
python3 "$HERE/fix_store_meta_path.py" "$DATA_ROOT" --expect-prefix "$DATA_ROOT" >/dev/null 2>&1 && ok "todas bajo $DATA_ROOT" || bad "hay store.meta con prefijo antiguo: fix_store_meta_path.py --rewrite-prefix"

echo "[check] espacio libre"
for m in / /mnt/data_disk; do
  free=$(df --output=avail -BG "$m" 2>/dev/null | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -ge 50 ]; then ok "$m ${free} G"; else warn "$m ${free} G (< 50 G: los datasets nuevos van al otro disco)"; fi
done

echo "[check] lectura por otro usuario"
if sudo -n true 2>/dev/null; then
  sudo -n -u nobody test -r "$DATA_ROOT/mdb/cora_gnn/catalog.dat" && ok "nobody lee mdb/cora_gnn" || bad "otro usuario no puede leer mdb/cora_gnn"
else warn "sin sudo: comprobación omitida"; fi

echo "[check] servidores"
pgrep -f 'org.neo4j' >/dev/null && warn "hay un neo4j corriendo: 7687/7474 ocupados, no arranques el otro" || ok "ningún neo4j corriendo"
pgrep -a -x mdb >/dev/null && warn "servidores mdb: $(pgrep -a -x mdb | tr '\n' ';')" || ok "ningún mdb server corriendo"

exit $fail
