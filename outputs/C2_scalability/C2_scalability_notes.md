# C2 — Caracterización experimental preliminar

## Objetivo

El objetivo de C2 es pasar desde la validación funcional inicial sobre Cora hacia una caracterización experimental del pipeline, midiendo tiempo, memoria y almacenamiento generado.

## C2 sobre Cora

Se reprodujeron las cinco configuraciones de C1:

1. Cora original.
2. Atributos estructurales nativos.
3. Atributos estructurales generados desde consultas GQL.
4. Atributos originales + estructurales nativos.
5. Atributos originales + atributos generados desde GQL.

### Resultados principales

- Cora original obtuvo el mejor accuracy.
- Las rutas estructurales-only obtuvieron menor accuracy, como era esperable.
- La ruta GQL combinada obtuvo un resultado cercano a la ruta nativa combinada.
- La memoria máxima fue similar entre configuraciones, alrededor de 1.4 GB.
- Las diferencias de almacenamiento reflejan principalmente la dimensión de las matrices de features.
- Las rutas GQL y nativa equivalentes tuvieron tamaños de almacenamiento prácticamente iguales.

## C2 sobre ogbn-arxiv

Se ejecutó el pipeline end-to-end sobre ogbn-arxiv como dataset de mayor escala.

### Características del dataset

- Nodos: 169,343.
- Relaciones: 1,166,243.
- Dimensión de features: 128.
- Clases: 40.
- Batches totales: 167.

### Resultados principales

- testAccuracy: aproximadamente 0.653--0.654.
- Tiempo total: entre 11.46 s y 14.51 s.
- Memoria máxima: aproximadamente 1.89 GB.
- Tamaño total de DB generada: 868.55 MB.
- Tamaño total de archivos `.fmat`: 165.37 MB.
- Tamaño total de archivos `.rmap`: 7.75 MB.
- Tamaño total de cachés: 83.98 MB.

## Interpretación preliminar

Los resultados muestran que el pipeline GNN de MillenniumDB funciona end-to-end tanto en Cora como en ogbn-arxiv. En Cora se valida la comparación entre rutas nativas y rutas GQL para generación de features. En ogbn-arxiv se obtiene una línea base de escalabilidad sobre un grafo considerablemente mayor, que permitirá comparar futuras extensiones basadas en GQL.

## Próximos pasos

1. Repetir la generación de atributos mediante GQL sobre un dataset mayor.
2. Evaluar si consultas GQL más complejas pueden generar features estructurales útiles.
3. Comparar tiempo, memoria y almacenamiento entre ruta base, ruta nativa y ruta GQL en escala mayor.
4. Explorar optimizaciones adicionales del pipeline, como definición de subgrafos, selección de vecindarios o generación en streaming.
