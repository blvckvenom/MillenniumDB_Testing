#pragma once

// sampling_backend_plan.h
//
// Decide, en el momento de construir el muestreo offline, si el fetch de
// candidatos de vecinos para el muestreo k-hop debe correr en GPU (leyendo el
// CSR de adyacencia uint32 pineado en RAM vía cudaHostRegister/UVA, o copiado a
// VRAM) o en la vía CPU out-of-core ya probada (el muestreador por nodo sobre el
// almacen de topologia jerarquico, con derrame a sidecar mmap / B+Tree para la
// cola fria).
//
// `plan_sampling_backend` es una funcion PURA: no llama a CUDA, no cachea
// punteros; solo razona sobre tamaños (los del CSR compacto en RAM, L2CompactCsr)
// y los recursos de hardware reportados por `mdb::gpu::detect_resources()`. Es la
// hermana de `plan_sort()` (src/gpu/resource_planner.h), que ya decide GPU/CPU
// para el ordenamiento de proyeccion consumiendo el mismo `SystemResources`.
// Asi es unit-testeable en CI sin GPU, alimentando un `SystemResources` sintetico.

#include <cstddef>
#include <cstdint>
#include <string>

#include "gnn/projection/edge_orientation.h"
#include "gpu/gpu_device.h"

namespace mdb::gnn {

// Dimensiones de una direccion del CSR global de topologia que la GPU pinea (el
// sidecar `topology_*.csr` narrow uint32). La decision DEBE dimensionarse sobre
// ESTE sustrato — el que `enable_pinned_gpu_view` realmente registra — y no sobre
// el tier-2 warm: pinear es `cudaHostRegister` de paginas ya residentes, que las
// vuelve no-swappables durante toda la corrida, asi que el gate de RAM debe contar
// los bytes globales completos `(n_rows+1)*8 + n_edges*4`, no el subconjunto warm.
// `present` es false cuando esa direccion no tiene sidecar narrow pineable.
struct DirCsrDims {
    std::uint64_t n_rows  = 0;  // N global (nodos)
    std::uint64_t n_edges = 0;  // aristas de esa direccion (grafo completo)
    bool          present = false;
};

// Backend de muestreo elegido por la decision por hardware.
enum class SamplingBackend {
    CPU_OUT_OF_CORE,  // muestreador por nodo sobre el almacen jerarquico (vía probada)
    GPU_UVA,          // GPU camina el CSR pineado en RAM via PCIe (zero-copy)
    GPU_VRAM_COPY,    // el CSR cabe en VRAM: copia a device, gather mas rapido
};

// Eleccion del usuario (override de la decision automatica).
enum class SamplingBackendChoice {
    AUTO,        // decidir por hardware
    FORCE_CPU,   // forzar la vía CPU (referencia bit-reproducible)
    FORCE_GPU,   // exigir GPU (error duro si no hay GPU capaz)
};

// Que direcciones del grafo sirve la GPU. Para UNDIRECTED, una direccion puede
// caber en RAM y la otra caer al host; la union de vecinos sigue completa.
enum class GpuDirections {
    NONE,
    FORWARD_ONLY,
    REVERSE_ONLY,
    BOTH,
};

// Parametros de la decision (primeras aproximaciones copiadas de la filosofia del
// planner de ordenamiento; tunear con A/B antes de fijar defaults).
struct SamplingBackendConfig {
    // El CSR + el mapa denso global->fila deben caber en esta fraccion de la RAM
    // disponible. 0.60 (mas estricto que el 0.70 del sort) porque el muestreo
    // tambien tiene residentes el almacen de features, los stores de etiqueta/
    // particion, y (en UVA) las paginas pineadas viven toda la corrida.
    double   ram_headroom_factor    = 0.60;
    // Piso de capacidad de computo CUDA (Volta+, el piso de mdb_gnn_core).
    int      min_compute_capability = 70;
    // Bajo este nº de aristas la vía CPU es sub-segundo; el pin/copy PCIe
    // dominaria. Espeja PlannerConfig.min_records_gpu del ordenamiento.
    std::uint64_t min_edges_for_gpu = 2'000'000;
    // Permitir, en UNDIRECTED, acelerar en GPU solo la direccion que cabe.
    bool     allow_single_direction = true;
};

// Resultado de la decision. `reason` es legible y se loguea una vez. Los tamaños
// quedan expuestos para los yields/telemetria del procedimiento.
struct SamplingBackendPlan {
    SamplingBackend backend              = SamplingBackend::CPU_OUT_OF_CORE;
    GpuDirections   directions           = GpuDirections::NONE;
    // Set by the ENGINE (not the pure planner) when the symmetric pre-merged
    // undirected slice is served: the plan then carries directions==FORWARD_ONLY
    // and the pinned view is the single undirected CSR (never BOTH — that would
    // double-count an already-merged list). plan_sampling_backend leaves false.
    bool            use_symmetric        = false;
    std::string     reason;
    std::size_t     fwd_csr_bytes        = 0;  // (n_rows+1)*8 + n_edges*4, dir fwd
    std::size_t     rev_csr_bytes        = 0;  // (n_rows+1)*8 + n_edges*4, dir rev
    std::size_t     node_map_bytes       = 0;  // 0 con substrato A (sidecar global,
                                               // indexado por fila densa, sin mapa)
    std::size_t     estimated_vram_bytes = 0;  // lo que residiria en device si GPU
};

// Decide el backend a partir de los recursos del sistema y los tamaños del CSR.
//
// PURA: sin llamadas CUDA, sin punteros cacheados. Solo sizing.
//
// @param res          recursos del sistema (de detect_resources()).
// @param orientation  orientacion del muestreo (NATURAL/REVERSE/UNDIRECTED).
// @param fwd_dims      dimensiones del sidecar global direccion natural.
// @param rev_dims      dimensiones del sidecar global direccion reverse.
// @param choice       override del usuario (AUTO/FORCE_CPU/FORCE_GPU).
// @param cfg          umbrales de la decision.
SamplingBackendPlan plan_sampling_backend(
    const mdb::gpu::SystemResources& res,
    EdgeOrientation                  orientation,
    DirCsrDims                       fwd_dims,
    DirCsrDims                       rev_dims,
    SamplingBackendChoice            choice = SamplingBackendChoice::AUTO,
    const SamplingBackendConfig&     cfg    = {});

// Helpers de texto para logs/yields.
const char* to_string(SamplingBackend backend) noexcept;
const char* to_string(GpuDirections directions) noexcept;

} // namespace mdb::gnn
