#!/usr/bin/env bash
# data-index.sh: genera el inventario de /data (INDEX.md). Nunca modifica datos.
# Clasifica cada directorio por su archivo marcador (ver spec, apéndice C).
set -uo pipefail
DATA_ROOT=${DATA_ROOT:-/data}
OUT=${1:-$DATA_ROOT/INDEX.md}
tmp=$(mktemp)

classify() {  # imprime "formato|detalle" o nada
  local d=$1
  if [ -f "$d/catalog.dat" ]; then
    case "$d" in
      */projections/*) echo "MDB proyección|" ;;
      */samples/*)     echo "MDB muestra|" ;;
      *) echo "MDB base|proyecciones=$(ls "$d/projections" 2>/dev/null | wc -l) muestras=$(ls "$d/samples" 2>/dev/null | wc -l)" ;;
    esac
  elif [ -f "$d/preprocessed/metadata.yaml" ]; then echo "GraphBolt|"
  elif [ -f "$d/conf.json" ] && [ -f "$d/features.bin" ]; then echo "OffGS store|$(python3 -c "import json;c=json.load(open('$d/conf.json'));print('nodos=%d dim=%d'%(c['num_nodes'],c['features_shape'][1]))" 2>/dev/null)"
  elif [ -f "$d/cached_feats.bin" ]; then echo "OffGS caché|"
  elif [ -f "$d/RELEASE_v1.txt" ] && [ -d "$d/raw" ]; then echo "OGB|"
  elif [ -f "$d/data/databases/neo4j/neostore.nodestore.db" ]; then echo "Neo4j|"
  elif ls "$d"/ind.*.x >/dev/null 2>&1 || [ -f "$d/raw/ind.cora.x" ]; then echo "Planetoid/DGL|"
  elif ls "$d"/*.gql >/dev/null 2>&1; then echo "fuente GQL|$(ls "$d"/*.npy 2>/dev/null | wc -l) npy"
  elif ls "$d"/*.csv >/dev/null 2>&1 && [ "$(ls "$d" | wc -l)" -le 20 ]; then echo "CSV|"
  fi
}

{
  echo "# Inventario de $DATA_ROOT"
  echo
  echo "Generado $(date '+%F %T') por bin/data-index.sh. No editar a mano: se regenera."
  echo "Cómo abrir cada formato: README.md. Salud: bin/check.sh."
  echo
  echo '```'; df -h / /mnt/data_disk; echo '```'
  for fam in mdb import raw baselines campaigns scratch; do
    [ -d "$DATA_ROOT/$fam" ] || continue
    echo; echo "## $fam"; echo
    echo "| Ruta | Formato | Disco | Tamaño | Dueño | Modificado | Detalle |"
    echo "|---|---|---|---|---|---|---|"
    find -L "$DATA_ROOT/$fam" -mindepth 1 -maxdepth 6 -type d 2>/dev/null | sort | while read -r d; do
      c=$(classify "$d"); [ -n "$c" ] || continue
      fmt=${c%%|*}; detail=${c#*|}
      disk=$(df --output=source "$d" 2>/dev/null | tail -1); disk=${disk##*/}
      size=$(du -sh "$d" 2>/dev/null | cut -f1)
      owner=$(stat -c %U "$d" 2>/dev/null)
      mod=$(find "$d" -maxdepth 2 -type f -printf '%TY-%Tm-%Td\n' 2>/dev/null | sort | tail -1)
      echo "| ${d#$DATA_ROOT/} | $fmt | $disk | $size | $owner | ${mod:-?} | $detail |"
    done
  done
} > "$tmp"
if [ -w "$(dirname "$OUT")" ] || [ -w "$OUT" ]; then cat "$tmp" > "$OUT"; else sudo -n install -m 0644 "$tmp" "$OUT"; fi
rm -f "$tmp"
echo "escrito $OUT ($(grep -c '^|' "$OUT") filas)"
