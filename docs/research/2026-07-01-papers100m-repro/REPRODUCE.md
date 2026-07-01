# papers100M — procedimiento exacto y reproducible (test 0.6527 / val 0.6836)

Registro consolidado de **los 4 procedimientos exactos** del pipeline GNN sobre papers100M que
producen el resultado de accuracy canónico actual. Antes esto estaba disperso en ~4 scripts (varios
apuntando al sample viejo `e2e5ep` que ya no existe). Este doc es la fuente única.

- **Fecha de medición:** 2026-07-01
- **Commit:** `e3b71984` (rama `feature-GNN`, 14 commits locales, sin pushear)
- **Host:** celebi — GPU 16 GB (RTX 5070 Ti), 30 GB RAM, NVMe Gen4, 20 cores
- **Binario:** `build/Release/bin/mdb` (Release)

## Resultado

| Métrica | Valor |
|---|---|
| **Test accuracy (final, época 50)** | **0.6527** |
| **Test accuracy (mejor modelo registrado, época 47)** | **0.6527** |
| **Best validation accuracy** | **0.6836** (época 47) |
| Épocas | 50 (patience 999, sin early-stop) |
| Wall-clock train | 2204.5 s = 36.7 min |
| Época steady | ~40.5 s (48 s en épocas con eval de test) |
| L1 hit ratio | 90.0 % |

Referencia externa: DiskGNN paper best_test 0.6591 / best_val 0.6908 → quedamos a ~0.64 pt en test
(dentro de la varianza irreducible de protocolo/muestreo ya caracterizada). Supera el canónico MDB
previo (~0.6406/0.6742 sobre el sample viejo `e2e5ep`).

## Nomenclatura importante (resuelve una ambigüedad real)

El sample se llama **`rev_10_15_20`** pero **NO es `orientation:REVERSE`**. Verificado decodificando
`samples/rev_10_15_20/catalog.dat`: tiene **48,730,271 nodos únicos**, que corresponde a
**`orientation:UNDIRECTED`** (REVERSE con el mismo fanout da ~9.16M, medido en la validación Plan F).
El "rev" del nombre significa "reverse edges added" = undirected (el paper DiskGNN agrega aristas
inversas para papers100M). **Orientación real = UNDIRECTED.** Fanout `[10,15,20]`, batchSize 1024,
seed 42, 1512 batches (train 1179 / val 123 / test 210). (catalog `GNNS` v3.)

---

## Prerrequisito — DB importada

La DB `data/dbs/gql/papers100M` debe tener importados los tensores de features + las propiedades
`label` y `split` (train/valid/test de OGB):

```
mdb import data.gql data/dbs/gql/papers100M --with-tensors node_features.npy
```

(La preparación de datos OGB→MDB es un procedimiento aparte; aquí se asume la DB ya importada.)

---

## Etapa 1 — `graph_project` (one-time, ~44.5 min)

Crea la proyección `papers100M_e2e_opt` (subgrafo en disco con la topología + features + label + split).

```gql
CALL graph_project('papers100M_e2e_opt','Node','CITES',{
    orientation:'NATURAL',
    indexSet:'GNN_MINIMAL',
    buildTopologySnapshot:false,
    includeFeatures:'node_features',
    labelProperty:'label',
    splitProperty:'split'
}) YIELD graphName,nodeCount,relationshipCount,featureDim,numClasses,projectMillis RETURN *
```

**Salida esperada:** `"papers100M_e2e_opt", 111059956, 1615685872, 128, 172, ~2673076`
(111.06M nodos, 1.6157B aristas — = raw OGB exacto tras el fix del Bloom filter; featureDim 128,
numClasses 172).

> Nota: `orientation:'NATURAL'` en la proyección es correcto — la simetrización a UNDIRECTED la hace
> el muestreo (Etapa 2), no la proyección.

---

## Etapa 2 — `gnn_offline_sample` (one-time, ~2.5–13 min)

Produce el sample `rev_10_15_20` (los k-hop batches + splits predefinidos).

```gql
CALL gnn_offline_sample('papers100M_e2e_opt','rev_10_15_20',[10,15,20],{
    batchSize:1024,
    randomSeed:42,
    orientation:'UNDIRECTED',
    usePredefinedSplits:true,
    useFourLevelTopologyStore:true,
    useAdjacencyCache:true,
    useL3MmapSidecar:true,
    l1CacheMb:512,
    l2CacheMb:6144,
    numWorkers:16,
    force:true
}) YIELD totalBatches,trainBatches,validationBatches,testBatches,uniqueNodes,numWorkersUsed,computeMillis,sampleContentFp RETURN *
```

**Salida esperada:** `totalBatches 1512, train 1179, val 123, test 210, uniqueNodes ~48.73M`.

> ⚠️ **Determinismo del sample:** con `useFourLevelTopologyStore:true` + el feedback warm-start de
> `node_counts.bin` (el default), el sample **NO es bit-reproducible run-to-run** — `uniqueNodes`
> deriva ~±2k entre regeneraciones (48,724,978 vs 48,726,532 medido) porque cada corrida reescribe
> `node_counts.bin` y eso mueve nodos entre tiers, cambiando el orden de vecinos que ve el sampler.
> El accuracy varía dentro de ±0.01. Para bit-reproducibilidad: `MDB_GNN_FREEZE_NODE_COUNTS=1` o
> cold-start (borrar `node_counts.bin`).

---

## Etapa 3 — `gnn_build_feature_store` (one-time, ~19–34 min)

Construye el four-level feature store + hornea los bloques offline (self-contained → train rápido).

**IMPORTANTE — el accuracy es INVARIANTE al tamaño de cache.** Los presupuestos L1/L2 solo eligen en
qué tier vive cada nodo (L1 GPU / L2 CPU / L3 disco / L4 packed), nunca los VALORES de las features.
Así que 0.6527 se reproduce con cualquier `l1CacheMb`; el tamaño solo afecta velocidad/OOMs de
cold-start, no el resultado.

**Primera vez (sin calibración previa)** — presupuesto por defecto o explícito:

```gql
CALL gnn_build_feature_store('rev_10_15_20','node_features',{
    bakeBlocks:true
}) YIELD l1Nodes,l2Nodes,l3Nodes,l4Nodes,gpuCacheMb,blocksBuiltOk,buildTimeMs RETURN *
```

**Loop auto-calibrante (opcional, para velocidad óptima OOM-free)** — tras una corrida de train
(Etapa 4) que mide el pico de VRAM y escribe `node_features_vram_calib.txt`, reconstruir con
`autoCache:true` dimensiona L1 al máximo OOM-free medido para este host:

```gql
CALL gnn_build_feature_store('rev_10_15_20','node_features',{
    autoCache:true, force:true, force_reorder:false, bakeBlocks:true
}) YIELD l1Nodes,gpuCacheMb,blocksBuiltOk RETURN *
```

> **Recipe de rebuild a un L1 nuevo (verificado en código):** `force:true, force_reorder:false,
> bakeBlocks:true`, dejando `force_packed_slim`/`force_caches`/`force_meta` en su default `true`.
> Solo el reorder MinHash es layout-independiente (reusable con `force_reorder:false`). Usar
> `force_packed_slim:false` dejaría un slim viejo → store inconsistente. Verificar
> `blocksBuiltOk=true` (si no, el train cae a modo online ~2× más lento).

El store medido en este run: L1=5426 MiB (11,112,448 nodos), gpu_cache.bin 5510 MiB.

---

## Etapa 4 — `gnn_train` (50 épocas, ~36.7 min a 41.5 s/ép)

Config canónica (lr 0.001, dropout 0.2, hidden 256, seed 42, N=8). `trackTestAtBestVal:true` es lo
que da el yield `testAccuracyAtBestVal` = test accuracy del **mejor modelo registrado**.

```gql
CALL gnn_train('rev_10_15_20','node_features',{
    model:'graphsage',
    hiddenDim:256,
    epochs:50,
    lr:0.001,
    dropout:0.2,
    normalize:false,
    patience:999,
    tolerance:0,
    randomSeed:42,
    useAsyncPrefetcher:true,
    prefetchNumWorkers:8,
    prefetchQueueSize:6,
    sampleCacheMb:2048,
    saveOnBestVal:true,
    saveFinal:true,
    exportEmbeddings:false,
    trackTestAtBestVal:true,
    outputDir:'acc50_rev'
}) YIELD testAccuracy,testAccuracyAtBestVal,bestValAccuracy,bestValEpoch,ranEpochs,trainSeconds,l1HitRatio,bestCheckpointPath,finalCheckpointPath RETURN *
```

**Salida esperada:** `testAccuracy ~0.6527, testAccuracyAtBestVal ~0.6527, bestValAccuracy ~0.6836,
bestValEpoch 47, ranEpochs 50, trainSeconds ~2205, l1HitRatio ~0.90`. Checkpoints en
`<proj_dir>/gnn_output/acc50_rev/checkpoints/{best_model,final_model}`.

Convergencia (val_acc por época): ép1 0.5646 → ép5 0.6504 → ép12 0.6703 → ép18 0.6768 →
ép38 0.6834 → **ép47 0.6836 (best)** → ép50 0.6834 (plateau ~0.683 desde ép38).

---

## Caveats (para no confundirse)

1. **`rev_10_15_20` = UNDIRECTED**, no REVERSE (ver "Nomenclatura" arriba). 48.73M nodos únicos.
2. **Accuracy invariante al cache L1/L2** — solo tiering, no valores de features. 0.6527 se reproduce a
   cualquier tamaño de cache.
3. **Sample no bit-reproducible run-to-run** con warm-start `node_counts.bin` (±~2k nodos, accuracy
   ±0.01). Usar `MDB_GNN_FREEZE_NODE_COUNTS=1` o cold-start para bit-identidad.
4. **Cold-start OOM en época 1** si L1 está al límite de VRAM (p.ej. 5426 en 16 GB): 1-6 OOMs
   transitorios → 1-2 batches zero-filled UNA vez, luego self-heal (real-sample-read). Impacto en
   accuracy: nulo. Evitable con el margen OOM-free (autoCache tras la calibración baja L1 a ~4974).
5. **`e2e5ep` está obsoleto** — cualquier script viejo con `SAMPLE=e2e5ep` apunta a un sample que ya
   no existe. Usar `rev_10_15_20`.

## Resumen de tiempos (one-time prep + train repetible)

| Etapa | Wall-clock | Tipo |
|---|---|---|
| graph_project | ~44.5 min | one-time |
| gnn_offline_sample | ~2.5–13 min | one-time |
| gnn_build_feature_store (+ bake) | ~19–34 min | one-time |
| gnn_train (50 ép) | ~36.7 min (41.5 s/ép) | repetible |
