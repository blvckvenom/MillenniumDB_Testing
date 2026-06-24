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

    // --- Modo tiled (path lean simetrico) -----------------------------------
    // En modo tiled COL_IDX NO se registra entero: ROW_PTR se pinea completo y
    // COL_IDX se transmite en ventanas alineadas a nodos a traves de un unico
    // buffer pinned reutilizable. `tiled==false` => los campos de arriba se
    // comportan EXACTAMENTE como el path no-tiled (d_col_idx es el COL_IDX
    // entero device-visible) y los de abajo quedan sin usar.
    const uint64_t* h_row_ptr      = nullptr;  // ROW_PTR host (scan de grados por ventana)
    const uint32_t* h_col_src      = nullptr;  // COL_IDX host fuente (mmap/heap base); NUNCA se registra
    uint32_t*       h_col_window   = nullptr;  // buffer pinned mapped de staging (host ptr)
    uint32_t*       d_col_window   = nullptr;  // su puntero device
    std::size_t     window_cap_edges = 0;      // capacidad del buffer (en aristas)
    bool            tiled          = false;

    // --- Modo resident (CSR entero en VRAM) ---------------------------------
    // Cuando el grafo cabe en VRAM, ROW_PTR y COL_IDX se copian a memoria DEVICE
    // propia (cudaMalloc) y d_row_ptr/d_col_idx apuntan ahí: el kernel lee de HBM
    // (~cientos de GB/s) en vez de páginas host por PCIe (UVA). resident y tiled
    // son mutuamente excluyentes (resident => tiled=false, d_col_idx no-null).
    bool            resident       = false;
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
     * @brief Variante tiled: pinea ROW_PTR entero pero NO COL_IDX; transmite
     *        COL_IDX en ventanas alineadas a nodos via un buffer pinned reusable.
     *
     * Pensada para el path lean simetrico en grafos enormes (papers100M): pinear
     * el COL_IDX entero (~12.9 GB) como copia host pineada es justo lo que dispara
     * el OOM. Aqui ROW_PTR (~0.9 GB) se pinea entero (cudaHostRegisterMapped) y por
     * cada direccion activa se aloca UN buffer pinned mapped de `window_cap_edges`
     * aristas (cudaHostAllocMapped); el COL_IDX se copia ventana a ventana con
     * `map_col_window`. Deja `fwd()->d_col_idx == nullptr`: el llamador alimenta el
     * COL_IDX por lanzamiento via `map_col_window`. No-op silencioso sin GPU/CUDA
     * (igual que build_and_register).
     *
     * @param window_cap_edges capacidad del buffer de ventana en aristas (debe ser
     *        >= el grado del nodo de mayor grado, para que ninguna ventana de un
     *        solo nodo exceda el buffer).
     * @throws std::logic_error si ya hay un registro activo.
     * @throws CudaException si una llamada CUDA falla (registro parcial deshecho).
     */
    void build_and_register_tiled(const HostCsrArrays& fwd,
                                  const HostCsrArrays& rev,
                                  std::size_t          window_cap_edges);

    /**
     * @brief Stagea col_idx[edge_lo, edge_hi) de `dir` en su buffer pinned y
     *        devuelve el device-ptr desde donde leer.
     *
     * Para una direccion no-tiled devuelve `dir.d_col_idx` sin tocar (edge_lo/
     * edge_hi se ignoran): el COL_IDX entero ya es device-visible. Para una tiled
     * copia la ventana al buffer pinned (memcpy host->host pinned, visible por el
     * device via el mapeo) y devuelve `dir.d_col_window`. El buffer es UNICO y
     * reusable: cada llamada lo sobreescribe, asi que el kernel de la ventana
     * anterior debe haber sincronizado antes de re-mapear (lo hace
     * sample_layer_on_device_ con cudaDeviceSynchronize).
     *
     * @throws std::logic_error si la ventana excede la capacidad del buffer o los
     *         punteros tiled estan sin setear.
     */
    const uint32_t* map_col_window(const PinnedDirView& dir,
                                   std::uint64_t        edge_lo,
                                   std::uint64_t        edge_hi) const;

    /**
     * @brief Variante resident: copia el CSR ENTERO a memoria device (cudaMalloc
     *        + cudaMemcpy) y deja al kernel leerlo de HBM.
     *
     * Para grafos que caben en VRAM: el camino más rápido (lee de HBM, no por
     * PCIe vía UVA, ni stagea ventanas). Una sola subida amortizada al inicio.
     * Deja `resident=true, tiled=false` y d_row_ptr/d_col_idx apuntando a los
     * buffers device propios. No-op silencioso sin GPU/CUDA. La salida del kernel
     * es byte-idéntica a la de build_and_register (el RNG no depende de dónde vive
     * col_idx).
     *
     * @throws std::logic_error si ya hay un registro activo.
     * @throws CudaException si una llamada CUDA falla (buffers parciales liberados).
     */
    void build_and_register_resident(const HostCsrArrays& fwd,
                                     const HostCsrArrays& rev);

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
        void* row_ptr     = nullptr;
        void* col_idx     = nullptr;
        void* col_window  = nullptr;  // buffer cudaHostAlloc (tiled); cudaFreeHost en release()
        void* d_row_owned = nullptr;  // cudaMalloc ROW_PTR (resident); cudaFree en release()
        void* d_col_owned = nullptr;  // cudaMalloc COL_IDX (resident); cudaFree en release()
    };

    // Registra una direccion. Rellena `reg` incrementalmente (tras cada
    // `cudaHostRegister` exitoso) para que `release()` deshaga registros parciales
    // si una llamada posterior lanza.
    void register_dir_(const HostCsrArrays& src,
                       PinnedDirView&       out,
                       bool&                active_flag,
                       HostRegistration&    reg);

    // Variante tiled de register_dir_: pinea ROW_PTR entero, aloca el buffer de
    // ventana pinned mapped, deja d_col_idx null y setea los campos tiled de out.
    void register_dir_tiled_(const HostCsrArrays& src,
                             PinnedDirView&       out,
                             bool&                active_flag,
                             HostRegistration&    reg,
                             std::size_t          window_cap_edges);

    // Variante resident de register_dir_: cudaMalloc ROW_PTR+COL_IDX en device y
    // cudaMemcpy desde el mmap/heap fuente; setea resident y los d_*_owned de reg.
    void register_dir_resident_(const HostCsrArrays& src,
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
