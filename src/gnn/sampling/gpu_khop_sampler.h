#pragma once

// gpu_khop_sampler.h
//
// Fachada host (sin CUDA en la firma, includible sin nvcc) del muestreo k-hop en
// GPU. El kernel hace la parte cara y paralela: por cada nodo del frontier de una
// capa, muestrea hasta `fanout` vecinos SIN REEMPLAZO (reservoir de Vitter) leyendo
// el CSR de adyacencia pineado en RAM via UVA, con RNG contador-based (Philox), de
// modo que la muestra de un nodo depende solo de `(random_seed XOR batch_id, nodo,
// capa)` — invariante al scheduling de hilos/bloques y al numero de workers, igual
// que el `reseed_for_batch` del muestreador CPU.
//
// NO es bit-identico al muestreador CPU: el CPU usa un `mt19937_64` serial con
// Fisher-Yates, imposible de replicar en paralelo. Ambos honran la MISMA
// distribucion (uniforme-sin-reemplazo, `k = min(fanout, deg)`, short-circuit
// `k==deg` que emite todos los vecinos sin sacar ningun random), asi que la
// accuracy cae en la misma banda. El RNG elegido queda registrado en la metadata
// del sampleo para que un consumidor sepa que lo produjo.
//
// Diseno hibrido GPU+host: el GPU samplea los vecinos por capa (el expand); el host
// reusa la logica de ensamblaje del muestreador CPU (mapas de indice local +
// dedup por primera aparicion) para construir el `GraphSample`, garantizando que la
// FORMA de la salida sea identica — solo difieren los vecinos sorteados.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gnn/sampling/graph_sample.h"

namespace mdb::gnn {

class PinnedTopologyView;
struct SamplingBackendPlan;

/**
 * @brief Muestreo k-hop completo en GPU → `GraphSample` host.
 *
 * Mira `nodes_per_layer[0] = seeds` verbatim; expande capa por capa con el kernel
 * de reservoir sobre el CSR pineado del `view`; ensambla edges + all_unique_nodes
 * en host con la misma semantica que `BasicKHopSampler`. `plan.directions` decide
 * que direcciones del grafo sirve el GPU (REVERSE por defecto; BOTH para
 * UNDIRECTED).
 *
 * Precondicion: `view.is_registered()` (hay GPU y el CSR esta pineado). El llamador
 * (engine) solo invoca esto cuando el plan eligio GPU y el view se registro.
 */
GraphSample sample_khop_gpu(const std::vector<ObjectId>&     seeds,
                            uint64_t                         batch_id,
                            SplitType                        split,
                            const std::vector<int>&          fanouts,
                            const PinnedTopologyView&        view,
                            const SamplingBackendPlan&       plan,
                            uint64_t                         random_seed);

/**
 * @brief Entrada SOLO-TEST del primitivo de muestreo por nodo.
 *
 * Expone el kernel de reservoir+Philox aislado para validacion estadistica sin el
 * pipeline completo: sube un CSR sintetico (host) a device, samplea hasta `fanout`
 * vecinos de cada nodo de `nodes`, y devuelve por nodo la lista de ids vecinos
 * muestreados (dense uint32). Determinista por `(batch_seed, nodo, layer)`.
 *
 * @param row_ptr   CSR row_ptr host (longitud N+1).
 * @param col_idx   CSR col_idx host (longitud M), ids densos tag-stripped.
 * @param nodes     ids de nodos del frontier (densos) a samplear.
 * @param fanout    maximo de vecinos por nodo.
 * @param batch_seed  `random_seed XOR batch_id`, la key de Philox.
 * @param layer     indice de capa (entra en el counter de Philox).
 * @return por cada nodo de entrada, los vecinos muestreados (vacio si 0 grado).
 */
std::vector<std::vector<uint32_t>> gpu_sample_neighbors_for_test(
    const std::vector<uint64_t>& row_ptr,
    const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes,
    int                          fanout,
    uint64_t                     batch_seed,
    int                          layer);

/**
 * @brief Variante TILED del seam de test: mismo contrato que
 *        gpu_sample_neighbors_for_test, pero stagea COL_IDX en ventanas alineadas
 *        a nodos de cota blanda `window_cap_edges` (un buffer pinned reusable).
 *
 * Prueba que el camino tiled (ROW_PTR entero + COL_IDX por ventanas) produce un
 * resultado BIT-IDENTICO al de whole-CSR para cualquier window cap, demostrando
 * la correctitud del windowing/particion/reensamblado sin tocar produccion ni
 * requerir un topology_sym.csr en disco.
 *
 * @param window_cap_edges cota blanda de aristas por ventana (un nodo de grado
 *        mayor a la cota forma su propia ventana; el buffer real se dimensiona a
 *        max(cap, grado maximo)).
 */
std::vector<std::vector<uint32_t>> gpu_sample_neighbors_tiled_for_test(
    const std::vector<uint64_t>& row_ptr,
    const std::vector<uint32_t>& col_idx,
    const std::vector<uint32_t>& nodes,
    int                          fanout,
    uint64_t                     batch_seed,
    int                          layer,
    std::size_t                  window_cap_edges);

}  // namespace mdb::gnn
