# Raíz de datos `/data`: diseño

Fecha: 2026-09-02. Rama: `feature-GNN`. Estado: **aprobado y ejecutado el 2026-09-02** (las desviaciones encontradas al ejecutar están marcadas "Ejecución:").

## 1. Problema

443 GB de bases de datos y artefactos viven físicamente dentro del repositorio
(`data/dbs`, `data/example`, `docs/research/2026-07-14-accuracy-small`), otros
392 GB están sueltos en `/mnt/data_disk/bfuentes`, 142 GB en un directorio con
nombre de otro usuario, 63 GB duplicados en `~/.claude/jobs`, y dos
instalaciones de Neo4j. No hay un punto de entrada, no hay convención de
nombres, y no hay manual de acceso. El disco raíz está al 84 %.

Git no es el problema: `.gitignore:91` (`/data/*`) y `:135` (`docs/research/*`)
ya excluyen todo. El desorden es físico.

## 2. Objetivos y no objetivos

Objetivos:

- O1. Un único punto de entrada, `/data`, con taxonomía estable y nombres predecibles.
- O2. Los datos salen físicamente del repositorio; el repositorio sigue funcionando sin cambiar un script.
- O3. Compartido entre los cuatro usuarios de la máquina, con un modelo de permisos honesto.
- O4. El árbol abarca los dos NVMe: cada dataset vive en el disco que le convenga.
- O5. Cada paso de la migración es verificable antes de destruir su origen, y reversible.
- O6. El árbol se autodocumenta: un índice generado por script, y un README con la receta de acceso de cada formato.

No objetivos:

- N1. Reescribir los 139 scripts para leer una variable de entorno. Es fase 2.
- N2. Mover `/var/lib/neo4j` (lo administra apt; requiere editar `neo4j.conf`).
- N3. Mover código (`~/DiskGNN`, `~/dgl`, `~/mdb_oom_fixes`).
- N4. Deduplicar datasets de terceros entre formatos (OGB vs GraphBolt vs OffGS son representaciones distintas del mismo grafo y cada baseline necesita la suya).

## 3. Invariantes

- I1. **Un dataset es un directorio autocontenido.** Todo lo que el motor necesita para abrirlo está debajo de su raíz. Verificado: `projection_manager.cc:22` deriva `projections/` de `db_folder`, y `catalog.dat` más `gnn_meta.bin` no contienen rutas.
- I2. **Ninguna operación del motor cruza sistemas de archivos.** Los 19 `rename()` atómicos crean el `.tmp` junto al destino (`block_store.cc:31`, `addr_table_writer.cc:31`, `projection_catalog.cc:247`); los temporales del sort van dentro de la proyección (`projection_storage.cc:1462`, `native_projection_builder.cc:1933`); las cubetas del ext_sort dentro de `gnn_features` (`feature_matrix.cc:768`); cero enlaces duros; cero manejo explícito de symlinks. Por tanto un dataset puede estar detrás de un enlace simbólico a otro disco.
- I3. **La única ruta absoluta persistida** es `char packed_slim_dir[256]` en el desplazamiento 56 de `<feature>_store.meta` (`four_level_store.h:63`), leída en `four_level_store.cc:2411` y usada para derivar `sample_dir_` en `:2456`. Mover un dataset obliga a reescribir ese campo. Es de tamaño fijo y terminado en nulo: se reescribe en sitio sin cambiar el formato.
- I4. **Máximo dos saltos de enlace** entre el repositorio y el archivo físico: `repo → /data/<familia>/<dataset> → /mnt/data_disk/graphdata/...`. Nunca más.
- I5. **Dentro de un dataset de terceros no se renombra nada.** La librería `ogb` exige `ogbn_papers100M/` con guion bajo; GraphBolt exige `metadata.yaml` en `preprocessed/`; OffGS exige `conf.json` junto a `features.bin`. El nombre canónico nuestro vive solo en el primer nivel bajo la familia.
- I6. **Un escritor por dataset.** El motor crea archivos con modo `0644` (27 sitios). El grupo lee; escribe el dueño. Dos procesos escribiendo la misma base MillenniumDB es inseguro con o sin `/data`.
- I7. **El origen no se borra hasta que una segunda pasada con checksum confirme la copia.** Nunca `mv` entre discos.

## 4. Taxonomía

```
/data/                                  root:graphdata 2775
├── mdb/          bases MillenniumDB nativas (catalog.dat + B+Tree + gnn_features)
├── import/       fuentes de importación de MillenniumDB, por modelo: gql/ qm/ rdf/
├── raw/          datasets de terceros en su formato de distribución: ogb/ planetoid/ dgl/
├── baselines/    formatos propios de los sistemas contra los que comparamos:
│                 offgs_dataset/ offgs_small/ graphbolt_dataset/ neo4j_cora/
├── campaigns/    binarios de campañas de medición, por fecha: 2026-07-14-accuracy-small/
├── scratch/      efímero; cualquiera puede borrar sin preguntar
├── README.md     manual de acceso (apéndice A de este documento)
├── INDEX.md      inventario generado por bin/data-index.sh; no editar a mano
└── bin/          data-index.sh  check.sh  link-repo.sh  fix_store_meta_path.py
```

Semántica de las familias, para que un dataset nuevo sepa dónde va:

| Familia | Criterio de entrada | Ejemplo |
|---|---|---|
| `mdb/` | tiene `catalog.dat` en su raíz | `papers100M/`, `cora_gnn/` |
| `import/<modelo>/` | es lo que se le pasa a `mdb import` | `gql/papers100M/papers100M.gql` |
| `raw/<distribuidor>/` | tal como se descargó, sin procesar por nosotros | `ogb/ogbn_papers100M/` |
| `baselines/<sistema>_*` | formato interno de un sistema comparado | `offgs_dataset/ogbn-papers100M-offgs/` |
| `campaigns/<fecha>-<nombre>/` | salida binaria de una campaña cerrada | `2026-07-14-accuracy-small/mdb_dbs/` |
| `scratch/` | nada de lo anterior, o no sabes todavía | |

Nombres canónicos de dataset en `mdb/` e `import/`: `papers100M`, `products`,
`arxiv`, `cora`. El nombre `papers100M` en `mdb/` es obligatorio porque
`REPRODUCE.md:74` lo cita.

## 5. Mecanismo: enlaces simbólicos por dataset

Se evaluaron tres mecanismos. Se elige el enlace simbólico **por dataset** (no
sobre `data/dbs`).

| Criterio | Enlace por dataset | Montaje bind | Variable de entorno |
|---|---|---|---|
| Modo de falla si falta el destino | ruidoso: enlace colgante, `ls` en rojo, `ENOENT` | **silencioso**: directorio vacío; un `mdb import` crearía la base en el disco raíz bajo el punto de montaje | ruidoso |
| Reversión | `rm` del enlace y `mv` de vuelta | editar `fstab` y desmontar | revertir 97 referencias |
| Root en runtime | no | sí (arranque) | no |
| `REPRODUCE.md` | válido palabra por palabra | válido | hay que reescribirlo |
| Visible para git | no (`/data/*` ignorado) | no | sí, 139 scripts |
| Granularidad por disco | por dataset | por dataset, una línea de `fstab` cada uno | por dataset |
| Sobrevive `git clean -fdx` | los enlaces se borran, los datos no; `link-repo.sh` los recrea | sí | sí |

Por qué **por dataset** y no en `data/dbs`: `data/dbs/gql/.keep` está
versionado. Reemplazar `data/dbs` por un enlace borraría archivos rastreados.
`data/dbs/gql/papers100M` como enlace lo captura `.gitignore:91`.

Enlaces en el repositorio (todos ignorados por git; se verifica con
`git check-ignore -v` en el paso 6):

```
data/dbs/gql/papers100M          → /data/mdb/papers100M
data/dbs/gql/cora_gnn            → /data/mdb/cora_gnn
data/example/gql/papers100M      → /data/import/gql/papers100M     (.git/info/exclude:11)
data/example/gql/ogbn-arxiv      → /data/import/gql/ogbn-arxiv     (.gitignore:158)
data/example/gql/cora            → /data/import/gql/cora           (.gitignore:159)
docs/research/2026-07-14-accuracy-small/mdb_dbs → /data/campaigns/2026-07-14-accuracy-small/mdb_dbs
docs/research/2026-07-14-accuracy-small/data    → /data/campaigns/2026-07-14-accuracy-small/data
```

`data/example/qm/cora` y `data/example/rdf/cora` están rastreados por git (4 archivos):
se quedan.

**Ejecución:** los patrones `.gitignore:158-159` y `.git/info/exclude:11` terminan en
`/` y casan solo con directorios; git trató los tres enlaces de `data/example/gql`
como archivos y los mostró como sin seguimiento. Se agregaron `.gitignore:163-165`
sin barra final. `link-repo.sh` verifica el ignorado **después** de crear cada
enlace, con el tipo real que git ve.

Enlaces de compatibilidad en las rutas viejas, para que ningún script de nadie
se rompa:

```
/mnt/data_disk/bfuentes/offgs_dataset      → /data/baselines/offgs_dataset
/mnt/data_disk/bfuentes/offgs_small        → /data/baselines/offgs_small
/mnt/data_disk/bfuentes/graphbolt_dataset  → /data/baselines/graphbolt_dataset
/mnt/data_disk/svergara_tests/ogb          → /data/raw/ogb
~/neo4j_cora                               → /data/baselines/neo4j_cora
~/diskgnn_data                             → /data/baselines      (repara 8 referencias hoy rotas en scripts/)
```

## 6. Distribución física entre discos

`/data` es un directorio real en `/`. Las hojas pesadas viven en
`/mnt/data_disk/graphdata/` con la misma taxonomía, enlazadas desde `/data`.

| Destino lógico | Disco físico | Tamaño | Método |
|---|---|---|---|
| `/data/mdb/papers100M` | data_disk | 261 G | rsync + checksum (único cruce de disco) |
| `/data/mdb/cora_gnn` | `/` | 33 M | mv |
| `/data/import/gql/papers100M` | `/` | 98 G | mv |
| `/data/import/gql/{ogbn-arxiv,cora}` | `/` | 146 M | mv |
| `/data/raw/ogb` | data_disk | 135 G | mv (mismo disco); **Ejecución:** absorbió `svergara_tests/ogb_arxiv/ogbn_arxiv` (183 M), y ambas rutas viejas enlazan a `/data/raw/ogb` |
| `/data/raw/planetoid`, `/data/raw/dgl` | `/` | 32 M | mv |
| `/data/baselines/offgs_dataset` | data_disk | 314 G | mv (mismo disco) |
| `/data/baselines/offgs_small` | data_disk | 6,8 G | mv (mismo disco) |
| `/data/baselines/graphbolt_dataset` | data_disk | 72 G | mv (mismo disco) |
| `/data/baselines/neo4j_cora` | `/` | 1,1 G | mv |
| `/data/campaigns/2026-07-14-accuracy-small/{mdb_dbs,data}` | `/` | 64 G | mv |

Balance esperado:

| Momento | `/` libre | data_disk libre |
|---|---|---|
| hoy | 144 G | 357 G |
| tras limpieza (§7 paso 2) | ≥ 200 G | 357 G |
| tras mover papers100M | ≥ 460 G | 96 G |

Regla para datasets nuevos: si pesa más de lo que le queda libre a data_disk
menos 20 G, va físicamente a `/` bajo `/data` directo. `check.sh` avisa cuando
cualquiera de los dos discos baja de 50 G.

## 7. Modelo de permisos

- Grupo `graphdata`. Miembros iniciales: `bfuentes`. Se agrega a quien lo pida con `usermod -aG graphdata <u>`.
- `/data` y `/mnt/data_disk/graphdata`: `root:graphdata`, modo `2775`. El setgid hace que todo lo creado debajo herede el grupo.
- Cada dataset: dueño quien lo creó, grupo `graphdata`, directorios `2775`, archivos `0644` (lo que el motor produce).
- Consecuencia (I6): todos leen; escribe el dueño. Para ceder escritura de un dataset concreto: `chmod -R g+w <dataset>`, decisión explícita del dueño.
- `scratch/`: `1777` con sticky bit, como `/tmp`.

## 8. Procedimiento de migración

Cada paso es idempotente (se puede repetir sin daño), tiene verificación, y
tiene reversión. `migrate.sh --step N` ejecuta uno; `--dry-run` imprime sin hacer.

**Paso 0. Precondiciones.** `pgrep -a mdb` vacío (ningún servidor abierto sobre lo que se mueve). `pgrep -a neo4j` vacío. `df` registrado en `migrate.log`. Reversión: ninguna.

**Paso 1. Esqueleto.** `groupadd graphdata`; `usermod -aG graphdata bfuentes`; crear las dos raíces con la taxonomía de §4 y los permisos de §7. Verificación: `stat -c '%U:%G %a'` en cada directorio. Reversión: `rmdir` en orden inverso (vacíos).

**Paso 2. Limpieza aprobada.** Se borra, en este orden, y solo tras imprimir el listado y el tamaño exacto:

  a. `~/.claude/jobs/0729db73/tmp/adj/dbs` (44 G). **Ejecución: NO era un duplicado.** El diseño lo llamó "verificado por inodo y tamaño", pero lo verificado era solo el tamaño (dos muestras con la misma semilla y batch size pesan igual). El `cmp` previo al borrado encontró 93 981 002 bytes distintos en `arxiv/sample_seed42/batches.dat`, entre 240 k y 323 k en cada muestra de cora, 596 archivos presentes en un solo lado, y `store.meta` apuntando a semillas distintas: es una repetición del 27-jul (12:14, contra 11:36 la original) con otra configuración. Se conservó como `campaigns/2026-07-27-accuracy-small-adj/mdb_dbs` con enlace en la ruta vieja. Lección: "mismo tamaño" no es "mismo contenido"; la salvaguarda de comparar antes de borrar fue la que evitó perder evidencia.
  b. `~/.claude/jobs/0729db73/tmp/{products_cmp,hnswarxiv.PWtfwz}` (≈ 18,5 G): residuo del mismo job cerrado (comparaciones de embeddings y una base de arxiv de prueba). **No es duplicado verificado**; se borra por ser salida de un job terminado. Si hay duda, se mueve a `scratch/` en vez de borrarse.
  c. `/tmp/bench_sandbox` (7,0 G), `/tmp/claude-1001/*/702f768c-*/scratchpad` (1,3 G), `/tmp/radix_*`, `/tmp/test_output9.*`, `/tmp/pf_gather`.
  d. **Retirado.** Las 8.744 bases de test bajo `build/` pesan 0,0 GB en total (fixtures de kilobytes); los 15 GB de `build/` son binarios. No hay nada que ganar.

  **Ejecución:** el clasificador del modo automático bloquea `rm -rf` lanzado por el agente. El paso 2 lo ejecuta el usuario desde el chat: `! bash scripts/data-root/migrate.sh --step 2`. El paso 3 se desacopló del 2 (el enlace de compatibilidad sobre `adj/dbs` se crea en el paso que llegue segundo).

  Reversión de 2a: ninguna necesaria (queda el original). De 2b: ninguna (si se movió a scratch, `mv` de vuelta). De 2c y 2d: se regeneran.

**Paso 3. Movimientos dentro del mismo disco.** `mv` de cada fila de §6 marcada "mismo disco" o "/". Un `mv` dentro del mismo sistema de archivos es un `rename(2)`: atómico e instantáneo. Verificación: `test -d destino && ! test -e origen`. Reversión: `mv` inverso.

**Paso 4. papers100M al segundo disco.** El único cruce de disco. **Ejecución:** sin `--commit` el paso copia y verifica y se detiene; `--step 4 --commit` repite el checksum, borra el origen y crea los enlaces. Así la parte destructiva puede lanzarla el usuario aunque el clasificador bloquee al agente.

```
rsync -aHAX --info=progress2 data/dbs/gql/papers100M/ /mnt/data_disk/graphdata/mdb/papers100M/
rsync -aHAX --checksum --dry-run -i data/dbs/gql/papers100M/ /mnt/data_disk/graphdata/mdb/papers100M/  # debe imprimir vacío
```

Solo si la segunda pasada no lista ningún archivo se ejecuta
`rm -rf data/dbs/gql/papers100M`. Tiempo estimado: 261 G a 1,5 GB/s ≈ 3 min de
copia, ≈ 10 min de checksum. Reversión antes del `rm`: borrar el destino.
Después: rsync inverso.

**Paso 5. Enlaces en `/data` hacia data_disk.** Los cinco de §6 con disco físico data_disk (se crean dentro de los pasos 3 y 4). Verificación: `find -L /data -type l` vacío. **Ejecución:** el primer detector usaba `-xtype l` bajo `-L`, que es verdadero para *todo* enlace simbólico (marcó 21 enlaces sanos); bajo `-L` solo `-type l` casa con los que no se pudieron seguir. Corregido en `migrate.sh` y `check.sh`, con `.venv` podado. Reversión: `rm` del enlace.

**Paso 6. Enlaces en el repositorio y de compatibilidad** (§5). Antes de cada enlace del repo: `git check-ignore -v <ruta>` debe responder. Verificación: `git status --short` idéntico al de antes de empezar. Reversión: `rm` del enlace, `mv` de vuelta.

**Paso 7. Corregir `store.meta`.** `fix_store_meta_path.py --scan /data/mdb` lista todo `*_store.meta`, muestra la ruta grabada en el desplazamiento 56 y si existe. Con `--rewrite-prefix VIEJO NUEVO` reescribe en sitio el campo de 256 bytes (respaldo `.bak` antes). Para papers100M la ruta grabada apunta a `samples/vss555/packed_slim`, borrado el 2026-07-08: se reescribe el prefijo igualmente para que, cuando se regenere una muestra, el campo sea coherente; y se documenta en `INDEX.md` que el almacén de features de papers100M requiere `gnn_offline_sample` + `gnn_build_feature_store` antes de entrenar.

**Ejecución del paso 7:** debe correr **después** del paso 4 `--commit`; corrió antes y el `store.meta` de papers100M conservó el prefijo viejo hasta repetirlo (A9 falló la primera vez). Es idempotente, repetirlo no cuesta nada.

**Paso 8. Aceptación** (§10).

**Paso 9. Documentación.** Copiar el apéndice A a `/data/README.md`; generar `/data/INDEX.md` con `data-index.sh`; agregar a `scripts/onboard.sh` la llamada a `/data/bin/check.sh`.

## 9. Modos de falla y detección

| Falla | Cómo se detecta | Cómo se repara |
|---|---|---|
| Enlace colgante (alguien movió o borró el destino) | `check.sh`: `find -L /data ~/MillenniumDB_Testing/data -type l` | restaurar destino o `link-repo.sh` |
| `store.meta` con ruta que no existe | `fix_store_meta_path.py --scan` (lo llama `check.sh`) | `--rewrite-prefix` |
| Copia parcial a data_disk | segunda pasada `rsync --checksum` no vacía | repetir rsync; el origen sigue intacto |
| Servidor abierto durante el movimiento | paso 0 | esperar o detener |
| Otro usuario no puede leer | `check.sh` corre `sudo -u nobody test -r` sobre un archivo por familia | `chmod`/`chgrp` |
| data_disk sin espacio para una muestra nueva | `check.sh` avisa bajo 50 G | crear el dataset nuevo en `/` |
| Respaldo que copia el enlace y no el dato | regla en README: `rsync -aL` o respaldar las raíces físicas | |
| Dos Neo4j a la vez | ambos usan 7687/7474 por defecto | README: nunca los dos |
| `git clean -fdx` borra los enlaces del repo | `git status` limpio pero `cora_fastloop` falla con `ENOENT` | `link-repo.sh` (idempotente) |
| Alguien crea una base nueva en `data/dbs/gql/` real | `check.sh`: todo subdirectorio de `data/dbs/gql` debe ser enlace | mover a `/data/mdb` y enlazar |

## 10. Criterios de aceptación

- A1. `scripts/cora_fastloop.sh` imprime `0.8648649` (contrato del refactor; también prueba que GPU sigue activa, porque `0.8747` significaría CPU).
- A2. `mdb server /data/mdb/papers100M -p 1234` arranca; `MATCH (n) RETURN count(n)` devuelve 111 059 956; `mdb list-projections` lista `papers100M_e2e_opt` y `m1_medicion`.
- A3. `find -L /data ~/MillenniumDB_Testing/data -xtype l` vacío.
- A4. `git status --short` idéntico al inicial salvo los cambios de esta misma migración (`.gitignore`, `scripts/onboard.sh`, `scripts/data-root/`, la spec).
- A5. Retirado junto con el paso 2d.
- A6. `df`: `/` ≥ 400 G libres, data_disk ≥ 90 G libres.
- A7. `sudo -u svergara ls -R /data/mdb/cora_gnn` funciona y puede leer `catalog.dat`.
- A8. Las 8 referencias a `~/diskgnn_data/...` en `scripts/` resuelven (`test -e`).
- A9. `fix_store_meta_path.py --scan` no reporta prefijos viejos.

**Resultado medido el 2026-09-02 13:30:** A1 `0.8648649`; A2 `count(n) = 111 059 956` y las dos proyecciones listadas; A3, A4, A7, A8, A9 ok; A6 `/` 431 G libres (antes 144), `/mnt/data_disk` 97 G (antes 357). Borrado en el paso 2: `products_cmp` 17 G, `hnswarxiv` 1,3 G, `/tmp/bench_sandbox` 7 G, scratchpad 1,3 G. Conservado: `adj/dbs` 44 G como campaña propia.

## 11. Entregables en el repositorio

`scripts/data-root/`: `migrate.sh`, `link-repo.sh`, `check.sh`, `data-index.sh`,
`fix_store_meta_path.py`, `README.data.md` (se instala como `/data/README.md`).
El gancho en `scripts/onboard.sh` va justo antes del resumen final ("Binary:") y
solo corre si `/data` existe. Los scripts se copian a `/data/bin/` en el paso 9 (el repo es la
fuente; `/data/bin` la copia instalada). Ningún script fija `/home/bfuentes`:
usan `${MDB_REPO:-$(git rev-parse --show-toplevel)}` y `${DATA_ROOT:-/data}`.

---

## Apéndice A. `/data/README.md` (texto que se instala)

```markdown
# /data: datos del proyecto MillenniumDB + GNN

Un dataset es un directorio autocontenido. Las familias son: mdb/ import/ raw/
baselines/ campaigns/ scratch/. El inventario vivo está en INDEX.md (generado;
no editar). La verificación de salud es `bin/check.sh`.

Reglas:
1. Nada nuevo en la raíz de una familia sin nombre canónico (papers100M,
   products, arxiv, cora) o el nombre original del distribuidor.
2. No renombrar nada dentro de un dataset de terceros (ogb, graphbolt, offgs).
3. Un escritor por dataset. Para escribir en uno ajeno, pídele al dueño
   `chmod -R g+w`.
4. Respaldos: `rsync -aL` (sigue enlaces), o respaldar /data y
   /mnt/data_disk/graphdata por separado.
5. scratch/ se puede borrar sin avisar.

## Cómo abrir cada formato

### MillenniumDB (mdb/)  formato: catalog.dat + B+Tree .leaf/.dir + tensores
Binario: ~/MillenniumDB_Testing/build/Release/bin/mdb (v1.0.0).
    mdb server /data/mdb/papers100M -p 1234          # puerto por defecto 1234
    curl -s -X POST http://localhost:1234/gql -H "Content-Type: text/plain" \
         --data-binary 'MATCH (n) RETURN count(n)'
    mdb cli /data/mdb/papers100M                     # consola interactiva
    mdb list-projections /data/mdb/papers100M
    # sobre una proyección:  USE papers100M_e2e_opt MATCH (n) RETURN count(n)
Cuidado: curl devuelve exit 0 aunque el servidor responda error; revisa el cuerpo.
Modelos: gql/ se consulta en GQL; una base importada de .qm se consulta en MQL y
una de .ttl en SPARQL (usar `mdb cli`; el endpoint HTTP de esos dos modelos no
está verificado en este README).
Importar:
    mdb import /data/import/gql/papers100M/papers100M.gql /data/mdb/papers100M \
        --with-tensors /data/import/gql/papers100M/papers100M_features.npy
Python (modelos entrenados por gnn_train, fuera del DBMS):
    ~/MillenniumDB_Testing/scripts/gnn_datasets/mdb_gnn.py
    funciones: load_params, to_dgl, to_pyg, evaluate_offline(run_dir)
Procedimiento canónico de papers100M:
    ~/MillenniumDB_Testing/docs/research/2026-07-01-papers100m-repro/REPRODUCE.md

### OGB oficial (raw/ogb/)  formato: raw/data.npz + processed/*.pt + split/
    from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
    ds = NodePropPredDataset('ogbn-papers100M', root='/data/raw/ogb')
    # root debe contener ogbn_papers100M/ (guion bajo: lo exige la librería)
Entornos con `ogb`: diskgnn_cu124, pytorch2, svergara_tests/.venv,
scripts/gnn_datasets/.venv.

### DGL GraphBolt (baselines/graphbolt_dataset/)  formato: metadata.yaml + fused_csc_sampling_graph.pt + .npy
    import dgl.graphbolt as gb
    ds = gb.OnDiskDataset('/data/baselines/graphbolt_dataset/datasets/ogbn-papers100M-seeds').load()
Entornos con `dgl`: diskgnn_cu124, pytorch2.

### OffGS / DiskGNN (baselines/offgs_dataset/)  formato: features.bin + graph.pth + labels.pth + split_idx.pth + conf.json, más cache-size-*/blowup-*/cached_feats.bin
    conda activate diskgnn_cu124
    cd ~/DiskGNN/DiskGNN/examples
    python train_single_thread.py --dir /data/baselines/offgs_dataset --dataset ogbn-papers100M ...
El `--dir` por defecto del código es /nvme1n1/offgs_dataset: pásalo siempre.
Campaña vigente de DiskGNN: ~/logs_diskgnn/.

### DGL clásico (raw/dgl/)  formato: cora_v2_dgl_graph.bin + ind.cora_v2.*
    import dgl; dgl.data.CoraGraphDataset(raw_dir='/data/raw/dgl')

### PyTorch Geometric Planetoid (raw/planetoid/)  formato: raw/ind.* + processed/*.pt
    from torch_geometric.datasets import Planetoid
    Planetoid(root='/data/raw/planetoid', name='Cora')

### NumPy (.npy / .npz en import/ y raw/)
    import numpy as np; x = np.load(path, mmap_mode='r')   # 53 G sin cargar a RAM
Los features de papers100M están en orden C (el import los exige así).

### Neo4j
Dos instalaciones; ambas usan bolt 7687 y http 7474: **nunca las dos a la vez**.
- Sistema (products cargado, GDS 2026.06.0): datos en /var/lib/neo4j/data,
  CSV de importación en /var/lib/neo4j/import/products.
      sudo systemctl start neo4j;  cypher-shell -u neo4j -a bolt://localhost:7687
- Cora (tarball 5.26.0, GDS 2.13.2): /data/baselines/neo4j_cora/neo4j-community-5.26.0
      bin/neo4j console;  bin/cypher-shell
Python: paquete `neo4j` en el entorno diskgnn_cu124.

### Checkpoints y embeddings (dentro de cada proyección, gnn_output/<run>/)
model.pt (LibTorch), checkpoints/*.pt + *.ckptmeta (con estado Adam),
embeddings.npy, training_log.json. Abrir con mdb_gnn.py o torch.load.
```

## Apéndice B. Inventario definitivo (medido 2026-09-02)

Un dataset aparece una vez por formato. "Destino" es la ruta lógica bajo `/data`.

### papers100M (ogbn-papers100M: 111 059 956 nodos, 128 features, 172 clases)

| Formato | Ubicación actual | Tamaño | Destino |
|---|---|---|---|
| MillenniumDB nativo (base 46 G + proyecciones 96 G + gnn_features 120 G) | `data/dbs/gql/papers100M` | 261 G | `mdb/papers100M` (data_disk) |
| fuente de importación (.gql 44 G + features.npy 53 G + labels.npy) | `data/example/gql/papers100M` | 98 G | `import/gql/papers100M` |
| OGB oficial (data.npz 57 G + geometric_data_processed.pt 78 G) | `/mnt/data_disk/svergara_tests/ogb/ogbn_papers100M` | 135 G | `raw/ogb/ogbn_papers100M` |
| GraphBolt (node-feat.npy 53 G + fused_csc 13 G) | `/mnt/data_disk/bfuentes/graphbolt_dataset/datasets/ogbn-papers100M-seeds` | 68 G | `baselines/graphbolt_dataset/...` |
| OffGS store (features.bin 53 G + graph.pth 13 G) | `/mnt/data_disk/bfuentes/offgs_dataset/ogbn-papers100M-offgs` | 67 G | `baselines/offgs_dataset/...` |
| OffGS cachés (3 presupuestos) | `.../offgs_dataset/ogbn-papers100M-1024-10,15,20-1.0` | 247 G | idem |

Estado del almacén de features MillenniumDB: `samples/` vacío desde 2026-07-08;
`store.meta` registra 93,6 M nodos en L3/L4 sin muestra que los sirva. Entrenar
requiere regenerar muestra y almacén.

### products (ogbn-products)

| Formato | Ubicación actual | Tamaño | Destino |
|---|---|---|---|
| MillenniumDB nativo (base + proj, pwb, pwbfix + 4 muestras) | `docs/research/2026-07-14-accuracy-small/mdb_dbs/products/db` | 42 G | `campaigns/2026-07-14-accuracy-small/mdb_dbs/products` |
| repetición del 27-jul con otra configuración (no duplicado, ver §8 paso 2a) | `~/.claude/jobs/0729db73/tmp/adj/dbs/products` | 40 G | `campaigns/2026-07-27-accuracy-small-adj/mdb_dbs/products` |
| fuente .gql | `docs/research/.../data/gql_products_out/ogbn-products/ogbn_products.gql` | 1,5 G | `campaigns/.../data` |
| OGB caché | `docs/research/.../data/dl_cache_products` | 9,8 G | `campaigns/.../data` |
| copia de sandbox | `/tmp/bench_sandbox` | 7,0 G | **se borra** |
| GraphBolt | `graphbolt_dataset/datasets/ogbn-products-seeds` | 4,1 G | `baselines/graphbolt_dataset/...` |
| OffGS | `offgs_small/ogbn-products-offgs` + `ogbn-products-512-5,10-1.0` | 5,4 G | `baselines/offgs_small/...` |
| Neo4j 5 store (system) | `/var/lib/neo4j/data/databases/neo4j` | 9,3 G | no se mueve |
| Neo4j CSV de importación | `/var/lib/neo4j/import/products` y `docs/research/.../data/neo4j` | 3,8 G + 4,0 G | el segundo a `campaigns/.../data` |

### arxiv (ogbn-arxiv: 169 343 nodos)

| Formato | Ubicación actual | Tamaño | Destino |
|---|---|---|---|
| MillenniumDB nativo (base + proj + 10 muestras por semilla) | `docs/research/.../mdb_dbs/arxiv/db` | 4,1 G | `campaigns/.../mdb_dbs/arxiv` |
| repetición del 27-jul (no duplicado) | `~/.claude/jobs/0729db73/tmp/adj/dbs/arxiv` | 4,4 G | `campaigns/2026-07-27-accuracy-small-adj/mdb_dbs/arxiv` |
| base de prueba HNSW | `~/.claude/jobs/0729db73/tmp/hnswarxiv.PWtfwz` | 1,3 G | **se borra** (residuo de job) |
| fuente .gql | `data/example/gql/ogbn-arxiv` | 119 M | `import/gql/ogbn-arxiv` |
| OGB | `docs/research/.../data/{ogb_arxiv,ogb_arxiv_dgl}`, `/mnt/data_disk/svergara_tests/ogb_arxiv` | 183 M + 286 M + 183 M | `campaigns/.../data`; el tercero a `raw/ogb` |
| GraphBolt | `graphbolt_dataset/datasets/ogbn-arxiv-seeds` | 213 M | `baselines/graphbolt_dataset/...` |
| OffGS | `offgs_small/ogbn-arxiv-offgs` + `ogbn-arxiv-1024-10,15-1.0` | 1,5 G | `baselines/offgs_small/...` |
| DiskGNN arxiv | `docs/research/.../data/diskgnn_arxiv` | 992 M | `campaigns/.../data` |

### cora (2 708 nodos)

| Formato | Ubicación actual | Tamaño | Destino |
|---|---|---|---|
| MillenniumDB GQL (base de trabajo + proyección smoke6) | `data/dbs/gql/cora_gnn` | 33 M | `mdb/cora_gnn` |
| MillenniumDB GQL (campaña, 10 muestras) | `docs/research/.../mdb_dbs/cora/db` | 95 M | `campaigns/.../mdb_dbs/cora` |
| MillenniumDB GQL (8 bases de la campaña de medición) | `~/.claude/jobs/3a6fa3a4/tmp/*` | 366 M | no se mueve (pequeño, ligado a un job) |
| MillenniumDB GQL/QM/RDF (18 bases efímeras) | `/tmp/claude-1001/.../702f768c-*/scratchpad` | 1,3 G | **se borra** |
| fuente GQL | `data/example/gql/cora` | 27 M | `import/gql/cora` |
| fuente Quad Model (.qm, lenguaje MQL) | `data/example/qm/cora/cora.qm` | 12 M | `import/qm/cora` si no está en git |
| fuente RDF (.ttl, lenguaje SPARQL) | `data/example/rdf/cora/cora.ttl` | 12 M | `import/rdf/cora` si no está en git |
| DGL | `~/.dgl/cora_v2_d697a464` | 16 M | `raw/dgl/cora_v2_d697a464` |
| PyG Planetoid | `~/cora_xframework/planetoid/Cora` | 16 M | `raw/planetoid/Cora` |
| GraphBolt | `graphbolt_dataset/datasets/cora-seeds` | 31 M | `baselines/graphbolt_dataset/...` |
| OffGS | `offgs_small/{cora-offgs,cora-64-5,10-1.0}` | 37 M | `baselines/offgs_small/...` |
| Neo4j 5 store (tarball) | `~/neo4j_cora/neo4j-community-5.26.0/data` | 582 M | `baselines/neo4j_cora` |

### Pequeños y versionados (no se mueven)

`data/example/gql/{books,posts,hetero_test,transacciones}`,
`data/example/qm/{books,posts}`: CSV y .gql rastreados por git, < 50 KB cada uno.
`~/Desktop/neo4j-mpgnn-thesis/data/coauthor/Physics/raw/ms_academic_phy.npz`
(15,6 M): repo de la tesis de Lund, se queda con su repo.

### Formatos distintos presentes en la máquina

1. MillenniumDB nativo, en tres modelos: GQL, Quad Model (MQL), RDF (SPARQL).
2. Artefactos GNN de MillenniumDB: `.fmat`, `.rmap`, `.csr`, `batches.dat`, `.blk`, `store.meta`, cachés `.bin`.
3. Fuentes de importación MillenniumDB: `.gql`, `.qm`, `.ttl`, `.csv`, `.npy`.
4. OGB: `.npz` + PyG `.pt` + `split/`.
5. DGL GraphBolt: `metadata.yaml` + `fused_csc_sampling_graph.pt` + `.npy`.
6. DGL clásico: `.bin` + pickles `ind.*`.
7. PyG Planetoid: `ind.*` + `processed/*.pt`.
8. OffGS / DiskGNN: `features.bin` + `graph.pth` + `conf.json` + cachés por presupuesto.
9. Neo4j 5: `neostore.*` + transacciones, y CSV de importación.
10. Checkpoints y embeddings: `.pt`, `.ckptmeta`, `.npy`, `training_log.json`.

### Fuera del alcance de la medición

Los homes de `sferrada`, `svergara` y `vesteban` (modo 750) no se pudieron leer.
Si tienen bases propias, no están en este inventario.

## Apéndice C. Contrato de `data-index.sh`

Entrada: `DATA_ROOT` (por defecto `/data`). Recorre con `find -L` hasta
profundidad 6. Clasifica cada directorio por marcador: `catalog.dat` → base
MillenniumDB (y distingue raíz / `projections/*` / `samples/*`); `metadata.yaml`
→ GraphBolt; `conf.json` + `features.bin` → OffGS; `RELEASE_v1.txt` + `raw/` →
OGB; `neostore.nodestore.db` → Neo4j. Para cada hoja imprime familia, nombre,
formato, disco físico (`df --output=source`), tamaño (`du -sb`), dueño, fecha
de modificación más reciente, y para bases MillenniumDB el número de
proyecciones y muestras. Salida: `INDEX.md` con una tabla por familia y un
encabezado con `df` de ambos discos y la fecha. Código de salida 0; nunca
modifica nada. `check.sh` lo llama y además ejecuta las detecciones de §9 y
devuelve 1 si alguna falla.
