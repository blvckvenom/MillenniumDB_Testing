#!/usr/bin/env python3
"""Inspecciona o reescribe la ruta absoluta grabada en los archivos *_store.meta.

Formato GFLS (src/gnn/storage/four_level_store.h): encabezado de 312 bytes; el
campo `packed_slim_dir` son 256 bytes terminados en nulo en el desplazamiento 56.
Es la única ruta absoluta que MillenniumDB persiste (four_level_store.cc:2411 la
lee y :2456 deriva de ella el directorio de la muestra), así que hay que
actualizarla cuando un dataset cambia de directorio. Se reescribe en sitio sin
alterar el tamaño ni la versión del formato.

Uso:
  fix_store_meta_path.py RAIZ                         lista cada store.meta bajo RAIZ
  fix_store_meta_path.py RAIZ --expect-prefix P       exit 1 si alguna ruta grabada no empieza por P
  fix_store_meta_path.py RAIZ --rewrite-prefix V N    reescribe (copia .bak) las rutas que empiecen por V
"""
import argparse
import os
import shutil
import struct
import sys

MAGIC = 0x47464C53
SIZE = 312
OFF = 56
LEN = 256


def find_metas(root):
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for f in files:
            if f.endswith("_store.meta"):
                yield os.path.join(dirpath, f)


def read_path(meta):
    with open(meta, "rb") as fh:
        data = fh.read()
    if len(data) != SIZE or struct.unpack_from("<I", data, 0)[0] != MAGIC:
        return None
    return data[OFF:OFF + LEN].split(b"\0")[0].decode(errors="replace")


def write_path(meta, new):
    raw = new.encode()
    if len(raw) >= LEN:
        raise SystemExit(f"ruta demasiado larga ({len(raw)} >= {LEN}): {new}")
    shutil.copy2(meta, meta + ".bak")
    with open(meta, "r+b") as fh:
        fh.seek(OFF)
        fh.write(raw.ljust(LEN, b"\0"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--expect-prefix")
    ap.add_argument("--rewrite-prefix", nargs=2, action="append", default=[], metavar=("VIEJO", "NUEVO"))
    args = ap.parse_args()
    rc = 0
    n = 0
    for meta in sorted(find_metas(args.root)):
        n += 1
        stored = read_path(meta)
        if stored is None:
            print(f"{meta}: formato desconocido, sin tocar")
            continue
        for old, new in args.rewrite_prefix:
            if stored.startswith(old):
                target = new + stored[len(old):]
                write_path(meta, target)
                print(f"{meta}\n  reescrito: {stored}\n         -> {target}")
                stored = target
                break
        exists = os.path.exists(stored)
        print(f"{meta}\n  grabada: {stored}\n  existe en disco: {exists}")
        if args.expect_prefix and not stored.startswith(args.expect_prefix):
            print(f"  PREFIJO INESPERADO (esperaba {args.expect_prefix})")
            rc = 1
    print(f"{n} archivos store.meta revisados")
    sys.exit(rc)


if __name__ == "__main__":
    main()
