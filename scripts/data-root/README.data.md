# /data: datos del proyecto MillenniumDB + GNN

Un dataset es un directorio autocontenido. Familias: `mdb/` `import/` `raw/`
`baselines/` `campaigns/` `scratch/`. El inventario vivo está en `INDEX.md`
(generado por `bin/data-index.sh`; no editar). La verificación de salud es
`bin/check.sh`. El diseño completo y la justificación:
`MillenniumDB_Testing/docs/superpowers/specs/2026-09-02-data-root-design.md`.

## Reglas

1. Nada nuevo en la raíz de una familia sin nombre canónico (`papers100M`,
   `products`, `arxiv`, `cora`) o el nombre original del distribuidor.
2. No renombrar nada dentro de un dataset de terceros (`ogb`, `graphbolt`,
   `offgs`): sus librerías buscan nombres fijos.
3. Un escritor por dataset. Para escribir en uno ajeno, pide al dueño
   `chmod -R g+w`. Dos procesos escribiendo la misma base MillenniumDB es
   inseguro con o sin `/data`.
4. Respaldos: `rsync -aL` (sigue enlaces), o respaldar `/data` y
   `/mnt/data_disk/graphdata` por separado. Un `rsync -a` sin `-L` copia el
   enlace y no los datos.
5. `scratch/` se puede borrar sin avisar.
6. Dónde va un dataset nuevo: si pesa más que lo libre en `/mnt/data_disk`
   menos 20 G, créalo físicamente bajo `/data` (disco raíz); si no, bajo
   `/mnt/data_disk/graphdata/<familia>/` y enlázalo desde `/data/<familia>/`.
   `bin/check.sh` avisa cuando cualquiera de los dos discos baja de 50 G.
7. Dentro del repositorio, `data/dbs/gql/<base>` y `data/example/gql/<dataset>`
   son enlaces a `/data`. Si un `git clean -fdx` los borra, corre
   `scripts/data-root/link-repo.sh` (idempotente).

## Qué hay

| Familia | Criterio | Contenido hoy |
|---|---|---|
| `mdb/` | tiene `catalog.dat` en su raíz | `papers100M` (segundo disco), `cora_gnn` |
| `import/gql/` | lo que se le pasa a `mdb import` | `papers100M` (.gql 44 G + features.npy 53 G), `ogbn-arxiv`, `cora` |
| `raw/` | tal como se descargó | `ogb/{ogbn_papers100M,ogbn_arxiv}` (segundo disco), `dgl/`, `planetoid/` |
| `baselines/` | formato interno de un sistema comparado | `offgs_dataset`, `offgs_small`, `graphbolt_dataset` (segundo disco), `neo4j_cora` |
| `campaigns/` | salida binaria de una campaña cerrada | `2026-07-14-accuracy-small/{mdb_dbs,data}` (products, arxiv, cora en MillenniumDB, más OGB y Neo4j CSV) |

Los CSV, logs y notas citables de cada campaña siguen en el repositorio
(`docs/research/<campaña>/`); aquí solo viven los binarios.

## Cómo abrir cada formato

### MillenniumDB (`mdb/`): `catalog.dat` + B+Tree `.leaf`/`.dir` + tensores

Binario: `~/MillenniumDB_Testing/build/Release/bin/mdb` (v1.0.0).

    mdb server /data/mdb/papers100M -p 1234          # puerto por defecto 1234
    curl -s -X POST http://localhost:1234/gql -H "Content-Type: text/plain" \
         --data-binary 'MATCH (n) RETURN count(n)'
    mdb cli /data/mdb/papers100M                     # consola interactiva
    mdb list-projections /data/mdb/papers100M
    # sobre una proyección:  USE papers100M_e2e_opt MATCH (n) RETURN count(n)

`curl` devuelve exit 0 aunque el servidor responda error: revisa el cuerpo.

Modelos: una base importada de `.gql` se consulta en GQL (endpoint `/gql`);
una de `.qm` en MQL y una de `.ttl` en SPARQL (usa `mdb cli`; el endpoint HTTP
de esos dos modelos no está verificado en este README).

Importar:

    mdb import /data/import/gql/papers100M/papers100M.gql /data/mdb/papers100M \
        --with-tensors /data/import/gql/papers100M/papers100M_features.npy

Python, para usar modelos entrenados por `gnn_train` fuera del DBMS (PyTorch,
DGL, PyTorch Geometric): `~/MillenniumDB_Testing/scripts/gnn_datasets/mdb_gnn.py`
con `load_params`, `to_dgl`, `to_pyg`, `evaluate_offline(run_dir)`.

Procedimiento canónico de papers100M:
`~/MillenniumDB_Testing/docs/research/2026-07-01-papers100m-repro/REPRODUCE.md`.
Estado del almacén de features de papers100M: `samples/` está vacío desde
2026-07-08 y `store.meta` registra 93,6 M nodos en niveles de disco sin
muestra. Entrenar requiere `gnn_offline_sample` + `gnn_build_feature_store`.

Única ruta absoluta que el motor persiste: el campo de 256 bytes en el
desplazamiento 56 de `<feature>_store.meta`. Si mueves un dataset:
`bin/fix_store_meta_path.py /data --rewrite-prefix VIEJO NUEVO`.

### OGB oficial (`raw/ogb/`): `raw/data.npz` + `processed/*.pt` + `split/`

    from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
    ds = NodePropPredDataset('ogbn-papers100M', root='/data/raw/ogb')
    # root debe contener ogbn_papers100M/ (guion bajo: lo exige la librería)

### DGL GraphBolt (`baselines/graphbolt_dataset/`): `preprocessed/metadata.yaml` + `fused_csc_sampling_graph.pt` + `.npy`

    import dgl.graphbolt as gb
    ds = gb.OnDiskDataset('/data/baselines/graphbolt_dataset/datasets/ogbn-papers100M-seeds').load()

### OffGS / DiskGNN (`baselines/offgs_dataset/`): `features.bin` + `graph.pth` + `labels.pth` + `split_idx.pth` + `conf.json`, más `cache-size-*/blowup-*/cached_feats.bin`

    conda activate diskgnn_cu124
    cd ~/Desktop/DiskGNN/DiskGNN/examples      # checkout canónico (commit 2a6e1f5, el de la campaña vigente)
    python train_single_thread.py --dir /data/baselines/offgs_dataset --dataset ogbn-papers100M ...

El `--dir` por defecto del código es `/nvme1n1/offgs_dataset`, un disco que no
existe aquí: pásalo siempre. Campaña vigente de DiskGNN: `~/logs_diskgnn/`. Hay un
segundo checkout en `~/DiskGNN/DiskGNN` con parches locales sin commitear: no confundirlos.
`~/diskgnn_data` es un enlace a `/data/baselines`, por los scripts que lo citan.

### DGL clásico (`raw/dgl/`): `cora_v2_dgl_graph.bin` + `ind.cora_v2.*`

    import dgl; dgl.data.CoraGraphDataset(raw_dir='/data/raw/dgl')

### PyTorch Geometric Planetoid (`raw/planetoid/`): `raw/ind.*` + `processed/*.pt`

    from torch_geometric.datasets import Planetoid
    Planetoid(root='/data/raw/planetoid', name='Cora')

### NumPy (`.npy` / `.npz` en `import/` y `raw/`)

    import numpy as np; x = np.load(path, mmap_mode='r')   # 53 G sin cargar a RAM

Los features de papers100M están en orden C; el import los exige así.

### Neo4j

Dos instalaciones. Ambas usan bolt 7687 y http 7474: **nunca las dos a la vez**.

- Sistema (products cargado, GDS 2026.06.0): datos en `/var/lib/neo4j/data`,
  CSV de importación en `/var/lib/neo4j/import/products`. No está bajo `/data`
  porque lo administra apt.

      sudo systemctl start neo4j
      cypher-shell -u neo4j -a bolt://localhost:7687

- Cora (tarball 5.26.0, GDS 2.13.2): `/data/baselines/neo4j_cora/neo4j-community-5.26.0`

      bin/neo4j console
      bin/cypher-shell

Python: paquete `neo4j` en el entorno `diskgnn_cu124`.

### Checkpoints y embeddings (dentro de cada proyección, `gnn_output/<run>/`)

`model.pt` (LibTorch), `checkpoints/*.pt` + `*.ckptmeta` (con estado Adam),
`embeddings.npy`, `training_log.json`. Abrir con `mdb_gnn.py` o `torch.load`.

## Entornos Python y qué tiene cada uno

| Entorno | torch | dgl | pyg | ogb | offgs | neo4j |
|---|---|---|---|---|---|---|
| `~/miniconda3/envs/diskgnn_cu124` | sí | sí | sí | sí | sí | sí |
| `~/miniconda3/envs/pytorch2` | sí | sí | sí | sí | sí | no |
| `~/MillenniumDB_Testing/scripts/gnn_datasets/.venv` | sí | no | no | sí | no | no |
| `/mnt/data_disk/svergara_tests/.venv` | sí | no | sí | sí | no | no |

## Permisos

Grupo `graphdata`. Raíces `root:graphdata 2775` con setgid: lo nuevo hereda el
grupo. Cada dataset: dueño quien lo creó, archivos `0644` (todos leen), escribe
el dueño. Para entrar al grupo: `sudo usermod -aG graphdata <usuario>` y volver
a iniciar sesión.
