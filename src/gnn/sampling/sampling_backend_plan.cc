#include "gnn/sampling/sampling_backend_plan.h"

namespace mdb::gnn {

namespace {

// Tamaños de una direccion del CSR global a pinear. `present` es false cuando esa
// direccion no tiene sidecar narrow pineable.
struct DirSizing {
    std::size_t   csr_bytes    = 0;  // (n_rows+1)*8 + n_edges*4 (el arreglo completo)
    std::size_t   locked_bytes = 0;  // RAM sostenida no-reclamable de servir esta dir
    std::uint64_t edges        = 0;
    bool          present      = false;
};

// Mismo computo EXACTO que registra pinned_topology_view: ROW_PTR uint64 + COL_IDX
// uint32 sobre TODO el grafo (no el subconjunto warm). El costo BLOQUEADO difiere
// por modo de consumo: pin completo => csr_bytes; slice horneado consumido tiled
// desde mmap => solo ROW_PTR + ventana de staging (el COL_IDX queda en page cache
// reclamable, ya contado por MemAvailable).
DirSizing size_dir(DirCsrDims d, std::size_t tiled_stage_bytes) {
    if (!d.present) {
        return {};
    }
    const std::size_t row_ptr_bytes = (d.n_rows + 1) * sizeof(std::uint64_t);
    const std::size_t col_idx_bytes = d.n_edges * sizeof(std::uint32_t);
    DirSizing s;
    s.csr_bytes    = row_ptr_bytes + col_idx_bytes;
    s.locked_bytes = d.tiled_mmap ? row_ptr_bytes + tiled_stage_bytes : s.csr_bytes;
    s.edges        = d.n_edges;
    s.present      = true;
    return s;
}

} // namespace

const char* to_string(SamplingBackend backend) noexcept {
    switch (backend) {
        case SamplingBackend::CPU_OUT_OF_CORE: return "CPU_OUT_OF_CORE";
        case SamplingBackend::GPU_UVA:         return "GPU_UVA";
        case SamplingBackend::GPU_VRAM_COPY:   return "GPU_VRAM_COPY";
    }
    return "UNKNOWN";
}

const char* to_string(GpuDirections directions) noexcept {
    switch (directions) {
        case GpuDirections::NONE:         return "NONE";
        case GpuDirections::FORWARD_ONLY: return "FORWARD_ONLY";
        case GpuDirections::REVERSE_ONLY: return "REVERSE_ONLY";
        case GpuDirections::BOTH:         return "BOTH";
    }
    return "UNKNOWN";
}

SamplingBackendPlan plan_sampling_backend(
    const mdb::gpu::SystemResources& res,
    EdgeOrientation                  orientation,
    DirCsrDims                       fwd_dims,
    DirCsrDims                       rev_dims,
    SamplingBackendChoice            choice,
    const SamplingBackendConfig&     cfg) {

    SamplingBackendPlan plan;  // defaults: CPU_OUT_OF_CORE / NONE

    // Alcance segun orientacion: que direcciones del grafo se muestrean.
    const bool use_fwd = orientation == EdgeOrientation::NATURAL
                      || orientation == EdgeOrientation::UNDIRECTED;
    const bool use_rev = orientation == EdgeOrientation::REVERSE
                      || orientation == EdgeOrientation::UNDIRECTED;

    const DirSizing fwd = use_fwd ? size_dir(fwd_dims, cfg.tiled_stage_bytes) : DirSizing{};
    const DirSizing rev = use_rev ? size_dir(rev_dims, cfg.tiled_stage_bytes) : DirSizing{};
    plan.fwd_csr_bytes = fwd.csr_bytes;
    plan.rev_csr_bytes = rev.csr_bytes;
    // Substrato A (sidecar global): el CSR ya esta indexado por fila densa global,
    // asi que el kernel no necesita un mapa global->fila => node_map_bytes = 0.
    plan.node_map_bytes = 0;

    // Override: forzar CPU salta todas las gates.
    if (choice == SamplingBackendChoice::FORCE_CPU) {
        plan.reason = "forced CPU by user (FORCE_CPU)";
        return plan;
    }

    const bool force_gpu = choice == SamplingBackendChoice::FORCE_GPU;

    // Gate A: hay GPU y es capaz (Volta+). FORCE_GPU sin GPU capaz = error duro
    // (se señala en reason con prefijo ERROR: para que el motor lo eleve).
    const bool gate_a = res.has_gpu
                     && res.gpu.compute_capability >= cfg.min_compute_capability;
    if (!gate_a) {
        const std::string detail = "has_gpu=" + std::string(res.has_gpu ? "true" : "false")
                                 + ", cc=" + std::to_string(res.gpu.compute_capability)
                                 + " < " + std::to_string(cfg.min_compute_capability);
        plan.reason = (force_gpu ? "ERROR: FORCE_GPU but no capable GPU (" : "no capable GPU (")
                    + detail + ")";
        return plan;  // CPU (el motor convierte el prefijo ERROR: en hard-error)
    }

    // Gate B: workload suficientemente grande (salvo FORCE_GPU, que lo ignora).
    const std::uint64_t total_edges = fwd.edges + rev.edges;
    if (!force_gpu && total_edges < cfg.min_edges_for_gpu) {
        plan.reason = "workload too small (" + std::to_string(total_edges)
                    + " edges < " + std::to_string(cfg.min_edges_for_gpu) + ")";
        return plan;  // CPU
    }

    // Gate C: el costo de RAM SOSTENIDA de servir el CSR cabe en la fraccion de
    // headroom. Para el pin completo ese costo es el CSR entero (paginas no-
    // swappables toda la corrida); para un slice horneado consumido tiled desde
    // mmap es solo ROW_PTR + ventana de staging (el COL_IDX vive en page cache
    // reclamable). Si NO cabe, la cola fria ya esta derramada => CPU.
    const std::size_t headroom = static_cast<std::size_t>(
        cfg.ram_headroom_factor * static_cast<double>(res.ram_available));
    const auto fits = [&](const DirSizing& d) {
        return d.present && d.locked_bytes <= headroom;
    };
    // El detalle "locked X of csr Y" en reason hace observable el modo de costo.
    const auto need_str = [](const DirSizing& a, const DirSizing& b) {
        return std::to_string(a.locked_bytes + b.locked_bytes) + " locked (csr "
             + std::to_string(a.csr_bytes + b.csr_bytes) + ")";
    };

    GpuDirections dirs            = GpuDirections::NONE;
    std::size_t   device_resident = 0;

    if (orientation == EdgeOrientation::UNDIRECTED) {
        const std::size_t both_locked = fwd.locked_bytes + rev.locked_bytes;
        if (fwd.present && rev.present && both_locked <= headroom) {
            dirs            = GpuDirections::BOTH;
            device_resident = fwd.csr_bytes + rev.csr_bytes;
        } else if (cfg.allow_single_direction && (fits(fwd) || fits(rev))) {
            // Solo cabe una direccion: acelerar la de MAYOR edge_count (mas
            // lookups que acelerar); la otra cae al host. La union sigue completa.
            const bool fwd_fits = fits(fwd);
            const bool rev_fits = fits(rev);
            const bool pick_fwd = fwd_fits && (!rev_fits || fwd.edges >= rev.edges);
            if (pick_fwd) {
                dirs            = GpuDirections::FORWARD_ONLY;
                device_resident = fwd.csr_bytes;
            } else {
                dirs            = GpuDirections::REVERSE_ONLY;
                device_resident = rev.csr_bytes;
            }
        } else {
            plan.reason = "CSR exceeds RAM headroom (need "
                        + need_str(fwd, rev) + " > "
                        + std::to_string(headroom) + ") -> already out-of-core";
            return plan;  // CPU
        }
    } else {
        // NATURAL o REVERSE: una sola direccion.
        const DirSizing& d = (orientation == EdgeOrientation::NATURAL) ? fwd : rev;
        if (fits(d)) {
            dirs            = (orientation == EdgeOrientation::NATURAL)
                                ? GpuDirections::FORWARD_ONLY
                                : GpuDirections::REVERSE_ONLY;
            device_resident = d.csr_bytes;
        } else {
            plan.reason = "CSR exceeds RAM headroom (need "
                        + need_str(d, DirSizing{}) + " > "
                        + std::to_string(headroom) + ") -> already out-of-core";
            return plan;  // CPU
        }
    }

    // Gate D: la VRAM decide UVA vs copia a VRAM (no GPU vs CPU). free_vram ya
    // viene derateado por el factor de seguridad en detect_resources.
    plan.directions           = dirs;
    plan.estimated_vram_bytes = device_resident;
    // VRAM-resident cuando el CSR + una reserva absoluta (contexto CUDA + scratch)
    // caben en la VRAM libre CRUDA. Durante el muestreo no hay tensores de modelo/
    // features residentes, asi que el derate ciego (compartido con otros planners)
    // es demasiado conservador aqui. Si no cabe, leerlo por PCIe via UVA.
    const std::size_t vram_need = device_resident + cfg.vram_abs_headroom_bytes;
    if (vram_need <= res.gpu.raw_free_vram) {
        plan.backend = SamplingBackend::GPU_VRAM_COPY;
        plan.reason  = "GPU: CSR fits VRAM device-resident (" + std::to_string(device_resident)
                     + " + headroom " + std::to_string(cfg.vram_abs_headroom_bytes)
                     + " <= raw_free_vram " + std::to_string(res.gpu.raw_free_vram)
                     + "), directions=" + to_string(dirs);
    } else {
        plan.backend = SamplingBackend::GPU_UVA;
        plan.reason  = "GPU via UVA (device_resident " + std::to_string(device_resident)
                     + " + headroom > raw_free_vram " + std::to_string(res.gpu.raw_free_vram)
                     + "), directions=" + to_string(dirs);
    }
    return plan;
}

} // namespace mdb::gnn
