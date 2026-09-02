#!/usr/bin/env bash
# link-repo.sh: (re)crea los enlaces del repositorio hacia /data. Idempotente.
# Úsalo después de un `git clean -fdx`, que borra los enlaces (ignorados) pero no los datos.
# Verifica DESPUÉS de crear cada enlace que git lo ignora: un patrón con barra final
# (`dir/`) no casa con un enlace simbólico, y aparecería como sin seguimiento.
set -uo pipefail
DATA_ROOT=${DATA_ROOT:-/data}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${MDB_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
CAMP=$DATA_ROOT/campaigns/2026-07-14-accuracy-small
rc=0
links=(
  "data/dbs/gql/papers100M|$DATA_ROOT/mdb/papers100M"
  "data/dbs/gql/cora_gnn|$DATA_ROOT/mdb/cora_gnn"
  "data/example/gql/papers100M|$DATA_ROOT/import/gql/papers100M"
  "data/example/gql/ogbn-arxiv|$DATA_ROOT/import/gql/ogbn-arxiv"
  "data/example/gql/cora|$DATA_ROOT/import/gql/cora"
  "docs/research/2026-07-14-accuracy-small/mdb_dbs|$CAMP/mdb_dbs"
  "docs/research/2026-07-14-accuracy-small/data|$CAMP/data"
)
for e in "${links[@]}"; do
  rel=${e%%|*}; tgt=${e#*|}; link=$REPO/$rel
  if [ ! -e "$tgt" ]; then echo "FALTA destino: $tgt (para $rel)"; rc=1; continue; fi
  if [ -L "$link" ]; then
    if [ "$(readlink "$link")" = "$tgt" ]; then :; else echo "ERROR $rel apunta a $(readlink "$link"), esperaba $tgt"; rc=1; continue; fi
  elif [ -e "$link" ]; then echo "ERROR $rel existe y no es enlace"; rc=1; continue
  else
    mkdir -p "$(dirname "$link")" && ln -s "$tgt" "$link" && echo "creado  $rel -> $tgt"
  fi
  if git -C "$REPO" check-ignore -q "$rel"; then echo "ok      $rel (ignorado por git)"
  else echo "ERROR git NO ignora $rel: agregar un patrón sin barra final a .gitignore"; rc=1; fi
done
exit $rc
