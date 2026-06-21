#pragma once

// pinned_topology_view.h
//
// Registra los arrays CSR uint32 de adyacencia (ya residentes en RAM) como
// visibles por la GPU mediante cudaHostRegister/UVA, SIN duplicarlos: la GPU los
// recorre por PCIe (zero-copy), igual que un muestreador out-of-core que pinea su
// grafo en RAM del host y lo lee desde el device. Es el sustrato que consumira el
// kernel de muestreo k-hop; aqui solo se registran/desregistran las paginas host
// y se exponen los punteros device-visibles.
//
// Sustratos posibles para los arrays (el llamador decide cual pasa):
//   - Sidecar global de topologia (ROW_PTR[N+1] uint64 + COL_IDX[M] uint32):
//     cubre TODOS los nodos indexados por fila densa, correctness-complete.
//   - CSR compacto del tier-2 (L2CompactCsr): solo la fraccion warm.
// En ambos casos el descriptor es el mismo (punteros host + tamaños + tag), de
// modo que esta clase es agnostica al sustrato.
//
// Sin GPU en runtime (o build compilado sin CUDA) es un no-op total:
// `build_and_register` no registra nada, `is_registered()` queda false y los
// punteros device quedan null — la via de muestreo CPU no cambia en absoluto y la
// salida es byte-identica.

#include <cstddef>
#include <cstdint>

namespace mdb::gnn {

/**
 * @brief Vista device de una direccion del CSR (forward o reverse).
 *
 * Los punteros son device-visibles sobre las MISMAS paginas host (no son copias;
 * el modo zero-copy hace que la GPU camine la RAM por PCIe). `dst_type_tag` es el
 * tag de tipo ObjectId pre-desplazado (`tag << 56`) que el kernel re-OR'ea al
 * materializar un ObjectId de salida, porque `d_col_idx` guarda ordinales uint32
 * tag-stripped.
 */
struct PinnedDirView {
    const uint64_t* d_row_ptr    = nullptr;  // device-ptr, longitud n_rows + 1
    const uint32_t* d_col_idx    = nullptr;  // device-ptr, longitud n_edges
    uint64_t        dst_type_tag = 0;        // tag << 56
    std::size_t     n_rows       = 0;        // N (sidecar global) o N_L2 (compacto)
    std::size_t     n_edges      = 0;
};

/**
 * @brief Descriptor host de una direccion: punteros a los arrays CSR ya en RAM.
 *
 * `row_ptr`/`col_idx` == nullptr => esa direccion no se registra (p.ej. NATURAL
 * solo pasa forward, o un nodo sin aristas en una direccion). Los punteros deben
 * ser estables durante toda la vida del view (post-freeze / post-mmap); registrar
 * un arreglo que luego se reubica deja punteros device colgantes.
 */
struct HostCsrArrays {
    const uint64_t* row_ptr      = nullptr;  // host, longitud n_rows + 1
    const uint32_t* col_idx      = nullptr;  // host, longitud n_edges
    std::size_t     n_rows       = 0;
    std::size_t     n_edges      = 0;
    uint64_t        dst_type_tag = 0;        // tag << 56
};

/**
 * @brief Registra/desregistra las paginas host del CSR como device-visibles.
 *
 * No copiable ni movible: gestiona registro de paginas con el runtime CUDA, donde
 * un doble-unregister seria un error. El propietario la mantiene tras un
 * `std::unique_ptr` (el move del puntero transfiere la propiedad sin mover el
 * objeto). Construir solo sobre los arrays FINALES (post-freeze/post-mmap).
 */
class PinnedTopologyView {
public:
    PinnedTopologyView() = default;
    ~PinnedTopologyView();

    PinnedTopologyView(const PinnedTopologyView&)            = delete;
    PinnedTopologyView& operator=(const PinnedTopologyView&) = delete;
    PinnedTopologyView(PinnedTopologyView&&)                 = delete;
    PinnedTopologyView& operator=(PinnedTopologyView&&)      = delete;

    /**
     * @brief Registra forward (si trae punteros) y reverse (opcional).
     *
     * No-op silencioso (`is_registered()` queda false, punteros device null)
     * cuando el build es sin CUDA o no hay GPU capaz en runtime — la via CPU no
     * cambia. Con GPU presente registra cada region con `cudaHostRegisterMapped`
     * y obtiene los punteros device via `cudaHostGetDevicePointer`.
     *
     * @throws std::logic_error si ya hay un registro activo (llamar `release()`
     *         primero; re-registrar paginas ya pineadas es un bug del llamador).
     * @throws CudaException si una llamada CUDA falla; cualquier registro parcial
     *         se deshace antes de propagar.
     */
    void build_and_register(const HostCsrArrays& fwd, const HostCsrArrays& rev);

    /**
     * @brief Desregistra todas las regiones. Idempotente y noexcept.
     */
    void release() noexcept;

    bool is_registered() const noexcept { return registered_; }

    /// nullptr si esa direccion no se registro.
    const PinnedDirView* fwd() const noexcept { return fwd_active_ ? &fwd_ : nullptr; }
    const PinnedDirView* rev() const noexcept { return rev_active_ ? &rev_ : nullptr; }

private:
    // Punteros HOST originales por direccion, retenidos para `cudaHostUnregister`
    // (que toma el puntero host, no el device).
    struct HostRegistration {
        void* row_ptr = nullptr;
        void* col_idx = nullptr;
    };

    // Registra una direccion. Rellena `reg` incrementalmente (tras cada
    // `cudaHostRegister` exitoso) para que `release()` deshaga registros parciales
    // si una llamada posterior lanza.
    void register_dir_(const HostCsrArrays& src,
                       PinnedDirView&       out,
                       bool&                active_flag,
                       HostRegistration&    reg);

    bool             registered_ = false;
    bool             fwd_active_ = false;
    bool             rev_active_ = false;
    PinnedDirView    fwd_;
    PinnedDirView    rev_;
    HostRegistration fwd_reg_;
    HostRegistration rev_reg_;
};

}  // namespace mdb::gnn
