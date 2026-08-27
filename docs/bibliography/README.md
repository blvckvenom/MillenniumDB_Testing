# Bibliography of verified external sources

This document exists so that the citations in the source code are auditable. Every factual
claim that a code comment attributes to a public source (a standard, a peer-reviewed paper, a
product manual, system documentation) has an entry here recording the exact section as it
appears in the source, the verbatim quote that was read, the date and method of verification,
and — just as important — an explicit "What it does not say" boundary separating what the
source actually states from what is our own design choice, measurement, or inference. A claim
whose boundary is not recorded tends to grow over time until the source is being credited with
things it never said; the boundary field is what prevents that.

The reading direction is strictly one-way: code comments cite the public sources directly, and
this file records the verification behind those citations. The code NEVER cites this document,
and no comment may say "see docs/bibliography/". This file lives only on the development
branch and its deletion must not invalidate a single comment. Finally, an entry without a
verification date and method is not valid: if you add a citation, you must have opened and
read the source yourself, and you must record here when and how you did it. An undated entry
is to be treated as unverified and re-checked before anyone relies on it.

## Consolidation findings (2026-08-27)

1. **Contradiction found and resolved — DiskGNN queue-size sentence.** One raw entry placed
   the verbatim sentence "the sizes of all shared queues are set to 2" in Section 5.3
   "Training Pipeline"; another placed it in Section 6 "Implementation". Re-fetched
   https://arxiv.org/html/2405.05231 during consolidation: the sentence is in **Section 6
   "Implementation"** ("For the producer-consumer-based pipeline, the sizes of all shared
   queues are set to 2. This is observed to fully overlap the stages and consume a small
   amount of memory."). The merged entry below carries the corrected attribution.
2. **Provenance discrepancy — Neo4j "Graph management" quote.** The identical verbatim quote
   was attributed to two different URLs (`.../management-ops/` and
   `.../management-ops/graph-catalog-ops/`), both claimed WebFetch-verified on 2026-08-27.
   The claims themselves are identical, so this is not a contradiction of content, but the
   exact page could not be disambiguated from the raw records; both URLs are kept in the
   merged entry pending a re-check.
3. **Not WebFetch-verifiable — ISO/IEC 39075:2024.** The standard is paywalled (iso.org
   returns 403); every ISO claim was verified against the extracted full text of the
   purchased copy shipped in `docs/external_references/ISO_IEC_39075_extracted/`. The
   standard's existence, title, and publication date (12 April 2024) were confirmed publicly
   via Wikipedia. If strict WebFetch-only provenance is ever required, the ISO entry is the
   one to flag.

---

## 1. Standards

### ISO/IEC 39075:2024 — GQL

- **Source:** ISO/IEC (International Organization for Standardization / International
  Electrotechnical Commission), "ISO/IEC 39075:2024 Information technology — Database
  languages — GQL", 2024. https://www.iso.org/standard/76120.html
- **Verified:** 2026-08-27, read from the extracted full text of the purchased standard
  shipped in the repo (`docs/external_references/ISO_IEC_39075_extracted/`: TOC p. iii,
  clauses 3.4 pp. 24-25, 4.3.5 p. 29, clause 15 pp. 199-204, and the sections files for 4.4,
  17.2, and the optional-features annex). WebFetch of the standard's body is impossible
  (paywalled; iso.org returns 403). Designation, title, and publication date (12 April 2024)
  confirmed publicly via WebFetch of https://en.wikipedia.org/wiki/Graph_Query_Language.
- **Claims:**
  1. **Section:** 3.4.12 "directed edge"; 3.4.13 "undirected edge"; 4.3.5 "Graphs".
     **Quote:** "undirected edge: edge (3.4.11) that does not distinguish between its
     endpoints (3.4.14). Note 1 to entry: An undirected edge expresses a relationship that is
     necessarily symmetric. [...] The indication of whether the edge is a directed edge or an
     undirected edge (which is also called the directionality of the edge)."
     **What it does not say:** It does not define a traversal orientation
     (NATURAL/REVERSE/UNDIRECTED) applied to stored directed edges — that is an
     implementation-side notion of our projection layer (mirroring GDS-style orientation).
     Also, 4.3.5 is titled "Graphs", not "Undirected Edge Handling": the section name
     previously cited in a comment does not exist in the standard.
     **Cited from:** `src/gnn/projection/edge_orientation.h`
  2. **Section:** 3.4.13 "undirected edge", Note 1 to entry.
     **Quote:** "An undirected edge expresses a relationship that is necessarily symmetric."
     **What it does not say:** The standard does not define or endorse the canonical storage
     order (lower node id first) the code applies; that is a memory optimization of ours,
     presented in the comment as something that preserves the symmetry the standard states.
     It also backs the justification that UNDIRECTED orientation is a no-op over undirected
     sources, nothing more.
     **Cited from:** `src/graph_models/gql/projection/native_projection_builder.cc`,
     `src/query/procedure/builtin/project_procedure.h`
  3. **Section:** Clause 15 "Procedure calling" (15.1 \<call procedure statement\> and
     \<procedure call\>, 15.3 \<named procedure call\>).
     **Quote:** clause and subclause headings as listed.
     **What it does not say:** The standard does not define graph projections, parallel-edge
     aggregation, or projection orientations; it only backs the CALL/YIELD semantics.
     **Cited from:** `src/query/procedure/builtin/project_procedure.h`
  4. **Section:** 15.1 \<call procedure statement\> and \<procedure call\>.
     **Quote:** "If CPS immediately contains OPTIONAL and RESULT is an empty binding table
     result, then OPTIONAL_RESULT comprises a new record of type OTARTPC in which every field
     value is the null value."
     **What it does not say:** It defines the null-record for the EMPTY-RESULT case only; it
     does not define what to do when the procedure raises an error. Our suppression of errors
     into that same shape is our own extension, and the comment says so explicitly.
     **Cited from:** `src/query/executor/binding_iter/gql/call_procedure.cc`
  5. **Section:** 17.2 \<graph reference\> and \<catalog graph parent and name\>.
     **Quote:** "\<graph reference\> ::= \<catalog object parent reference\> \<graph name\> |
     \<delimited graph name\> | \<home graph\> | \<reference parameter specification\>"
     **What it does not say:** It does not say a flat namespace is a valid implementation of
     \<graph reference\>, and it says nothing about projections: it defines graph-reference
     resolution against the GQL-catalog. Ignoring the catalog/schema prefix is our current
     limitation, declared as such in the comments.
     **Cited from:** `src/graph_models/gql/graph_reference.h`,
     `src/query/parser/grammar/gql/query_visitor.cc`
  6. **Section:** 4.4 "Values".
     **Quote:** "4.4 Values" (subclause heading, with 4.4.1..4.4.5 beneath it).
     **What it does not say:** The subclause enumerates GQL's value model; it does not define
     MillenniumDB's GenericType enum or its numeric ids — that mapping is ours, and the @note
     presents it as a mapping, not as a definition from the standard.
     **Cited from:** `src/graph_models/gql/gql_object_id.h`
  7. **Section:** Feature GT03 (optional-features annex); official name read from the full
     text: 'Specifications for Feature GT03, "Use of multiple graphs in a transaction"'.
     **Quote:** "Feature GT03, \"Use of multiple graphs in a transaction\""
     **What it does not say:** GT03 is about using several graphs in one transaction; it does
     not describe or require a catalog/schema hierarchy. The original comment said
     "multi-graph transactions" informally; it now uses the standard's literal title.
     **Cited from:** `src/query/parser/grammar/gql/query_visitor.cc`
- **Notes:** All pre-existing ISO citations were confirmed literal against the purchased
  copy; none had to be altered beyond the two corrections recorded above (nonexistent section
  name "Undirected Edge Handling"; informal GT03 wording). See consolidation finding 3 for
  the verification-method caveat.

---

## 2. Peer-reviewed articles

### DiskGNN (Liu et al., SIGMOD 2025)

- **Source:** Renjie Liu, Yichuan Wang, Xiao Yan, Haitian Jiang, Zhenkun Cai, Minjie Wang,
  Bo Tang, Jinyang Li. "DiskGNN: Bridging I/O Efficiency and Model Accuracy for Out-of-Core
  GNN Training". Proc. ACM Manag. Data, Vol. 3, No. 1 (SIGMOD), Article 34, February 2025.
  arXiv:2405.05231. https://arxiv.org/abs/2405.05231
- **Verified:** 2026-08-27, WebFetch of https://arxiv.org/abs/2405.05231 (title, authors,
  abstract) and https://arxiv.org/html/2405.05231 / v2 (sections 5.1-5.3, 6, 7.3, Table 1,
  Algorithm 1), plus pdftotext search over the downloaded PDF v2 (Algorithm 1 and the
  reordering section read in full; the PDF footer confirms the SIGMOD venue string) and the
  paper's LaTeX source shipped in the repo for verbatim phrase checks. The section placement
  of the queue-size sentence was re-verified by a fresh WebFetch during consolidation (see
  finding 1).
- **Claims:**
  1. **Section:** Abstract.
     **Quote:** "The key technique used by DiskGNN is offline sampling, which helps decouple
     graph sampling from model computation."
     **Cited from:** `src/gnn/sampling/offline_sampling_engine.h`,
     `src/gnn/sampling/offline_sampling_engine.cc`, `src/gnn/sampling/seed_selector.h`,
     `src/gnn/sampling/basic_khop_sampler.h`, `src/gnn/sampling/basic_khop_sampler.cc`,
     `src/gnn/sampling/sample_storage.h`
  2. **Section:** Section 5.1, Algorithm 1 "Disk Cache Reordering using MinHash"; "Node
     reordering for disk cache".
     **Quote:** "similar sets are more likely to have the same hash value. Lines 2-3 of
     Algorithm 1 generate k MinHash functions by permuting the IDs." — and — "we choose
     HashOrder [71] over more complex algorithms (e.g., Gorder [62]) because it is
     lightweight and shown to produce high-quality ordering. [ref 71:] Tianyi Zhang, Aditya
     Desai, Gaurav Gupta, and Anshumali Shrivastava. 2024. HashOrder: Accelerating Graph..."
     **What it does not say:** The paper does NOT fix k=2 hash functions (k is left as a
     parameter; the 2 comes from the DiskGNN repository code, and the comment in
     `minhash_reorderer.h` distinguishes the two). It does not prescribe our global
     single-file consolidation of per-segment caches (the composite key
     `(segment_id<<32) | minhash` is ours — the paper reorders within per-segment disk
     caches). It does not define a per-visit access count: its tally is per batch appearance.
     It does not state our post-reorder read-amplification figures (our measurements, stated
     with their conditions in `cache_stats_snapshot.h`).
     **Cited from:** `src/gnn/sampling/minhash_reorderer.h`,
     `src/gnn/sampling/minhash_reorderer.cc`, `src/gnn/storage/batch_materializer.h`
  3. **Section:** Section 5.1 (segmented disk cache).
     **Quote:** "we search for the minimum s that satisfies the space constraint"
     **What it does not say:** It does not define our `disk_budget_bytes` semantics
     (warn-only today, heuristic search deferred); the paper's `disk_size` actively
     constrains its cache search.
     **Cited from:** `src/gnn/storage/four_level_store.h` (conceptual reference)
  4. **Section:** Section 5.2 (feature packing); abstract.
     **Quote:** "we can first collect all node features it requires and store them
     contiguously as a disk block (called feature packing) beforehand"
     **What it does not say:** Names no on-disk files (no "train-aux" appears in the paper);
     the consolidated.slim single-file layout, 4096-byte alignment, and fingerprint-based
     stale rejection are our design.
     **Cited from:** `src/gnn/storage/consolidated_slim.h`,
     `src/gnn/storage/batch_materializer.h`
  5. **Section:** Section 5.3 "Training Pipeline".
     **Quote:** "As GNN models and graph samples are typically small and can not saturate GPU
     computation, we run the model trainer and feature assembler on separate CUDA streams to
     improve GPU utilization." Same section also verbatim contains "The feature assembler can
     also assemble the complete features for mini-batch b+1 in parallel".
     **What it does not say:** Does not name SAGE/GAT specifically, does not give stream
     counts or priorities; our high-priority train-stream option and the c10 stream-pool
     mechanics are our implementation choices.
     **Cited from:** `src/gnn/core/stage3_streams.h`
  6. **Section:** Section 6 "Implementation".
     **Quote:** "DiskGNN leverages io_uring and uses 4 threads with each thread holding a
     ring to launch concurrent I/O requests"
     **What it does not say:** Specifies NO submission-queue depth; our QUEUE_DEPTH=1024 (and
     the ~4096 in-flight target) is our own sizing, labeled as such in the comment.
     **Cited from:** `src/gnn/storage/direct_io_reader.h`,
     `src/gnn/storage/direct_io_reader.cc`
  7. **Section:** Section 6 "Implementation".
     **Quote:** "For feature assembling on GPU, DiskGNN uses Unified Virtual Addressing (UVA)
     to fetch the node features resident on CPU memory."
     **What it does not say:** Does not describe our pinned-region pointer-lifetime contract
     (pointers valid for CpuCache lifetime) nor the lookup_uva/find_index API split; those
     are ours.
     **Cited from:** `src/gnn/storage/cpu_cache.h`
  8. **Section:** Section 6 "Implementation" (API listing).
     **Quote:** "def DiskGNN_train(dataset_pth : PATH, disk_size : int, cpu_size : int,
     gpu_size : int, kwargs)"
     **What it does not say:** Does not define our `disk_budget_bytes` semantics; the paper's
     `disk_size` actively constrains its cache search.
     **Cited from:** `src/gnn/storage/four_level_store.h`
  9. **Section:** Section 6 "Implementation" (item "Training pipeline"). **[Corrected during
     consolidation — one raw entry placed this in Section 5.3; see finding 1.]**
     **Quote:** "For the producer-consumer-based pipeline, the sizes of all shared queues are
     set to 2."
     **Cited from:** `src/query/procedure/builtin/gnn_train_procedure.cc`,
     `src/query/procedure/builtin/gnn_predict_procedure.cc`,
     `src/query/procedure/builtin/gnn_offline_sample_procedure.h`
  10. **Section:** Section 7 "Evaluation" (Experiment Settings).
     **Quote:** "we report the test accuracy at the epoch when the highest validation
     accuracy is achieved for each system."
     **What it does not say:** The paper does not prescribe a learning-rate schedule as part
     of the protocol; our cosine default and its justification (~0.640 -> ~0.654
     test-at-best-val on ogbn-papers100M) are our own measurement, declared as such in
     `gnn_train_procedure.cc`.
     **Cited from:** `src/query/procedure/builtin/gnn_train_procedure.cc`,
     `src/query/procedure/builtin/gnn_predict_procedure.cc`
  11. **Section:** Section 7.3 "Microbenchmarks" (training time breakdown).
     **Quote:** "computations are lightweight for GNN models"
     **What it does not say:** Does not mention spare SMs or any GPU-occupancy percentage;
     our "spare SMs" framing is an inference and is worded as ours in the comment. Replaces a
     previous paraphrase that over-attributed "SAGE/GAT graph samples don't saturate GPU" to
     the paper.
     **Cited from:** `src/gnn/core/stage3_streams.h`
  12. **Section:** Table 1, caption "Execution statistics of disk-based GNN systems and our
     DiskGNN on the Ogbn-papers100M graph", row "Disk access volume (GB)".
     **Quote:** "Disk access volume (GB)"
     **What it does not say:** Does not define whether alignment padding counts toward the
     volume; our bytes_disk counts aligned-up O_DIRECT regions, which is our accounting
     convention for comparability. It is a ROW, not a column (systems are the columns) — the
     code comments previously said "column" and were corrected.
     **Cited from:** `src/gnn/storage/cache_stats_snapshot.h`,
     `src/gnn/storage/four_level_store.h`
  13. **Section:** pre-processing access-frequency collection.
     **Quote:** "During pre-processing, DiskGNN first collects the access frequencies of all
     node features. This procedure is lightweight, as it simply keeps a counter for each node
     and streams the graph samples from disk."
     **What it does not say:** Does not describe frequency-band binning as a stand-in for
     per-batch access sets (our fallback in `compute_l3_minhash_reorder_`), nor the
     four-level L1/L2/L3/L4 topology tiering itself.
     **Cited from:** `src/gnn/projection/topology_frequency_profiler.h`,
     `src/gnn/projection/four_level_topology_store.h`,
     `src/gnn/projection/four_level_topology_store.cc`
- **Notes:** The arXiv HTML header shows PACMMOD (SIGMOD) as the journal; the exact
  volume/year shorthand "SIGMOD'25" used in the codebase was confirmed from the PDF footer in
  one verification pass but not independently from the arXiv pages. The "train-aux" filename
  previously attributed to the paper comes from the DiskGNN code repository and was removed
  from the code rather than re-sourced. The pre-existing citation chain DiskGNN -> HashOrder
  (Zhang et al., 2024) in `minhash_reorderer.h` was confirmed exact.

### SALIENT (Kaler et al., MLSys 2022)

- **Source:** Tim Kaler, Nickolas Stathas, Anne Ouyang, Alexandros-Stavros Iliopoulos, Tao
  B. Schardl, Charles E. Leiserson, Jie Chen. "Accelerating Training and Inference of Graph
  Neural Networks with Fast Sampling and Pipelining" (SALIENT), MLSys 2022.
  https://arxiv.org/abs/2110.08450
- **Verified:** 2026-08-27, WebFetch of the arXiv abstract (two reads, full abstract quoted
  verbatim).
- **Section:** Abstract.
- **Quote:** "We present a sequence of improvements to mitigate these bottlenecks, including
  a performance-engineered neighborhood sampler, a shared-memory parallelization strategy,
  and the pipelining of batch transfer with GPU computation. [...] our system SALIENT
  achieves a speedup of 3x over a standard PyTorch-Geometric implementation with a single GPU
  and a further 8x parallel speedup with 16 GPUs."
- **What it does not say:** It does NOT define any 6-12x speedup range (that figure was
  previously attributed to it jointly with NextDoor and has been removed), and it does not
  prescribe our pool layout (shared atomic batch counter + mutex-serialized writes); the code
  comment states this explicitly.
- **Cited from:** `src/gnn/sampling/sampling_config.h`
- **Notes:** The abstract phrase "shared-memory parallelization strategy" is the only thing
  the code now attributes to it.

### NextDoor (Jangda et al., EuroSys 2021)

- **Source:** Abhinav Jangda, Sandeep Polisetty, Arjun Guha, Marco Serafini. "Accelerating
  Graph Sampling for Graph Machine Learning using GPUs" (NextDoor), EuroSys 2021.
  https://arxiv.org/abs/2009.06693
- **Verified:** 2026-08-27, WebFetch of the arXiv abstract.
- **Section:** Abstract.
- **Quote:** "NextDoor runs them orders of magnitude faster than existing systems"
- **What it does not say:** Gives NO numeric speedup figures in the abstract; it does not
  back the 6-12x range the code used to attribute to it. It is also a GPU-based sampling
  system, not a CPU shared-memory thread pool.
- **Cited from:** (none — citation REMOVED from `src/gnn/sampling/sampling_config.h` because
  the claim it backed does not exist in the source)
- **Notes:** Entry kept as a record of the verification that motivated the removal.

### Vitter, "Random Sampling with a Reservoir" (1985)

- **Source:** Jeffrey Scott Vitter. "Random Sampling with a Reservoir", ACM Transactions on
  Mathematical Software, Vol. 11, No. 1, March 1985, pp. 37-57.
  https://www.ittc.ku.edu/~jsv/Papers/Vit85.Reservoir.pdf
- **Verified:** 2026-08-27, WebFetch of the PDF + reading of pages 1-3 (cover, Table I,
  section 2).
- **Section:** 2. "RESERVOIR ALGORITHMS AND ALGORITHM R".
- **Quote:** "Algorithm R (which is a reservoir algorithm due to Alan Waterman) works as
  follows: When the (t + 1)st record in the file is being processed, for t >= n, the n
  candidates form a random sample of the first t records. The (t + 1)st record has a
  n/(t + 1) chance of being in a random sample of size n of the first t + 1 records, and so
  it is made a candidate with probability n/(t + 1). The candidate it replaces is chosen
  randomly from the n candidates."
- **What it does not say:** The paper does NOT claim Algorithm R as Vitter's invention: it
  credits Alan Waterman (the paper's main contribution is Algorithm Z). Citing
  "(Vitter, 1985)" is correct as the source where the algorithm is defined and analyzed.
- **Cited from:** `src/gnn/sampling/reservoir_sampler.h`,
  `src/gnn/sampling/reservoir_sampler.cc`, `src/gnn/sampling/sorted_batch_sampler.cc`,
  `src/gnn/sampling/leapfrog_gnn_sampler.cc`, `src/gnn/sampling/seek_based_gnn_sampler.cc`,
  `src/gnn/sampling/gpu_khop_sampler.h`, `src/gnn/sampling/gpu_khop_sampler.cu`
- **Notes:** The paper's pseudocode (M := TRUNC(t x RANDOM()); if M < n then replace C[M])
  matches the description carried in our comments.

### Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11)

- **Source:** John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw. "Parallel Random
  Numbers: As Easy as 1, 2, 3", SC'11 (Proceedings of 2011 International Conference for High
  Performance Computing, Networking, Storage and Analysis), Seattle, 2011.
  https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- **Verified:** 2026-08-27, WebFetch of the PDF + reading of pages 1, 6-7 (abstract,
  sections 4.2 and 4.3).
- **Section:** 4.2 "Fast PRNGs by reducing cryptographic strength: ARS, Threefry"; 4.3 "The
  Philox PRNG".
- **Quote:** "We have verified the Crush-resistance of ARS-5 [...] with the constants
  0xBB67AE8584CAA73B (sqrt(3)-1) and 0x9E3779B97F4A7C15 (the golden ratio) for the upper and
  lower halves of the round key. [...] This procedure yielded the following multipliers: for
  Philox-4x32, 0xCD9E8D57 and 0xD2511F53"
- **What it does not say:** The paper prints the Weyl constants in their 64-bit form
  (section 4.2, for ARS); the 32-bit truncations used by the Philox 4x32 key schedule
  (0x9E3779B9, 0xBB67AE85) come from the Random123 code — the kernel comment presents them
  as "32-bit truncations". The paper also says nothing about graph sampling.
- **Cited from:** `src/gnn/sampling/gpu_khop_sampler.cu`
- **Notes:** Corrected the previous comment that called 0xBB67AE85 "sqrt5": it is sqrt(3)-1.

### Lemire, "Fast Random Integer Generation in an Interval" (2019)

- **Source:** Daniel Lemire. "Fast Random Integer Generation in an Interval", ACM
  Transactions on Modeling and Computer Simulation (TOMACS), 2019.
  https://arxiv.org/abs/1805.10941
- **Verified:** 2026-08-27, WebFetch of the arXiv abstract.
- **Section:** Abstract.
- **Quote:** "We review an unbiased function to generate ranged integers from a source of
  random words that avoids integer divisions with high probability."
- **What it does not say:** The paper's UNBIASED version includes a rejection step; our
  kernel deliberately omits it (residual bias <= m/2^32 per draw, negligible for
  fanout-sized m). That omission and its justification are our decision, not the paper's —
  the kernel comment declares it.
- **Cited from:** `src/gnn/sampling/gpu_khop_sampler.cu`
- **Notes:** —

### Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (AISTATS 2010)

- **Source:** Xavier Glorot & Yoshua Bengio, "Understanding the difficulty of training deep
  feedforward neural networks", Proceedings of the Thirteenth International Conference on
  Artificial Intelligence and Statistics (AISTATS), PMLR v9, 2010.
  https://proceedings.mlr.press/v9/glorot10a.html
- **Verified:** 2026-08-27, WebFetch of the PMLR abstract page.
- **Section:** Abstract (PMLR landing page).
- **Quote:** "Based on these considerations, we propose a new initialization scheme that
  brings substantially faster convergence."
- **What it does not say:** The fetched page (abstract only) does not state the
  Var(W)=2/(n_in+n_out) formula or the uniform bound sqrt(6/(fan_in+fan_out)) — those appear
  in the paper body, which was not fetched; the concrete bound used in our comment is instead
  sourced from the PyTorch xavier_uniform_ documentation, which cites this paper. The page
  also never uses the phrase "normalized initialization".
- **Cited from:** `src/gnn/models/graphsage_model.cc`
- **Notes:** Used as the attribution for the Xavier bound; the formula itself is quoted from
  the PyTorch documentation entry below.

---

## 3. Product manuals and library documentation

### Neo4j Graph Data Science Manual (current)

- **Source:** Neo4j, Neo4j Graph Data Science Manual (current). Pages: "Native projection"
  (https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/),
  "Graph management" (https://neo4j.com/docs/graph-data-science/current/management-ops/ —
  also recorded as .../management-ops/graph-catalog-ops/, see finding 2), "Listing graphs"
  (https://neo4j.com/docs/graph-data-science/current/management-ops/graph-list/).
- **Verified:** 2026-08-27, WebFetch of each URL.
- **Claims:**
  1. **Section:** "Native projection" page, Relationship projection.
     **Quote:** "Handling of multiple instances of all the relationship properties associated
     to the relationship. Allowed values: NONE (default), SINGLE, COUNT, MIN, MAX, SUM."
     **What it does not say:** Does NOT define what SINGLE does (neither which edge is kept,
     nor whether it fails, nor any tie-breaking). Our SINGLE = strict rejection of parallel
     edges, our default of SINGLE (Neo4j lists NONE as the default), and accepting NONE as an
     alias of SINGLE are our own definitions, declared as such in the comment in
     `project_procedure.h`. The same section does define orientation: NATURAL (default),
     UNDIRECTED, REVERSE.
     **Cited from:** `src/query/procedure/builtin/project_procedure.h`
  2. **Section:** "Graph management" (opening section).
     **Quote:** "Each graph has a name that can be used as a reference for management
     operations, or in analytical workflows that require the same graph to be processed
     several times. These references are stored in the graph catalog."
     **What it does not say:** The manual never uses the phrase "flat namespace" and never
     states that no catalog.schema.graph hierarchy exists; it simply describes name-based
     references in the catalog, with no hierarchical qualifier. The comment's "no
     catalog/schema hierarchy" phrasing is our inference from absence, worded as our own
     comparison, not as a quote.
     **Cited from:** `src/graph_models/gql/graph_reference.h`,
     `src/query/parser/grammar/gql/query_visitor.cc`
  3. **Section:** "Listing graphs".
     **Quote:** "Referring to a named graph in a procedure is only allowed on the database it
     has been projected on."
     **What it does not say:** Does not mention qualified names or hierarchical resolution
     (they do not exist in GDS), and defines no semantics for catalog/schema prefixes. It
     supports only that the name's scope is the database that projected it.
     **Cited from:** `src/query/parser/grammar/gql/query_visitor.cc`
- **Notes:** The "Graph management" quote was recorded under two different URLs by two
  independent verification passes (see consolidation finding 2); the claims are identical.
  The comment in `graph_reference.h` is prose comparison, not a formal citation with a
  section.

### PyTorch documentation v2.13

- **Source:** PyTorch contributors, PyTorch documentation v2.13. Pages: "Linear —
  torch.nn.Linear" (https://docs.pytorch.org/docs/2.13/generated/torch.nn.Linear.html) and
  "torch.nn.init" (https://docs.pytorch.org/docs/2.13/nn.init.html).
- **Verified:** 2026-08-27, WebFetch of both v2.13 pages (the stable URL redirects there).
- **Claims:**
  1. **Section:** "Linear" page, Variables (weight, bias).
     **Quote:** "The values are initialized from U(−k,k), where k=1/in_features (bounds are
     ±sqrt(k)); same for bias when bias is True."
     **What it does not say:** The docs do not mention that the implementation achieves this
     via kaiming_uniform_(a=sqrt(5)) (a source-code detail; the old comment's phrasing was
     replaced by the documented distribution), and they make no claim that this default is
     small for ReLU networks — that comparison is our own arithmetic in the comment.
     **Cited from:** `src/gnn/models/graphsage_model.cc`
  2. **Section:** "torch.nn.init" page: calculate_gain (gain table), xavier_uniform_,
     kaiming_uniform_.
     **Quote:** "gain for ReLU = sqrt(2); xavier_uniform_: a = gain × sqrt(6 / (fan_in +
     fan_out)), 'described in Understanding the difficulty of training deep feedforward
     neural networks - Glorot, X. & Bengio, Y. (2010)'; kaiming_uniform_: bound = gain ×
     sqrt(3 / fan_mode), 'described in Delving deep into rectifiers ... He, K. et al.
     (2015)'."
     **What it does not say:** Says nothing about which initialization suits GraphSAGE or any
     accuracy consequence; the depth-compounding argument (r = fan_in*Var(W)/2 per ReLU
     layer, r^L) in our comment is our own derivation, presented as reasoning, not attributed
     to this page. He et al. 2015 is cited only via this page's reference line — the He paper
     itself was not fetched.
     **Cited from:** `src/gnn/models/graphsage_model.cc`
- **Notes:** —

### DGL SAGEConv source (master branch)

- **Source:** DGL (Deep Graph Library) contributors, SAGEConv source,
  `python/dgl/nn/pytorch/conv/sageconv.py`, master branch.
  https://raw.githubusercontent.com/dmlc/dgl/master/python/dgl/nn/pytorch/conv/sageconv.py
- **Verified:** 2026-08-27, WebFetch of the raw source file on GitHub (docs.dgl.ai was
  unreachable, ECONNRESET on three attempts).
- **Section:** SAGEConv.reset_parameters (method body and docstring).
- **Quote:** "gain = nn.init.calculate_gain(\"relu\") ...
  nn.init.xavier_uniform_(self.fc_self.weight, gain=gain);
  nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain); docstring: 'The linear weights
  W(l) are initialized using Glorot uniform initialization.'"
- **What it does not say:** reset_parameters does not touch biases — the bias zeroing in our
  MDB_GNN_XAVIER_INIT branch is our own choice, and the comment now says so explicitly. The
  fetched code is the master branch, not a pinned release tag.
- **Cited from:** `src/gnn/models/graphsage_model.cc`
- **Notes:** Verified against source rather than the manual because docs.dgl.ai reset the
  connection on three attempts.

### DiskGNN repository (Liu-rj/DiskGNN, GitHub)

- **Source:** DiskGNN repository (Liu-rj/DiskGNN), GitHub, MIT license.
  https://github.com/Liu-rj/DiskGNN
- **Verified:** 2026-08-27, WebFetch of the repo page (MIT license badge) + raw fetches of
  `examples/load_graph.py` and `examples/mega_batch_sampling.py`; the 2-hash count further
  corroborated in the local clone `~/DiskGNN` (`examples/feat_packing.py`, h1/h2 with
  `_CAPI_SegmentedMinHash`).
- **Section:** `examples/load_graph.py` (function load_ogb); `examples/mega_batch_sampling.py`;
  `examples/feat_packing.py` (section "# calculate minhash").
- **Quote:** "g = dgl.remove_self_loop(g); g = dgl.add_self_loop(g)  |
  node_counts[input_nodes.cuda()] += 1  |  h1 = torch.randperm(num_batches, device=device);
  h1_res, _ = torch.ops.offgs._CAPI_SegmentedMinHash(...); h2 = torch.randperm(...)"
- **What it does not say:** The repo's line numbers are NOT stable (add_self_loop is today at
  line 26, not 24; the tally at 59, not 50): the code therefore cites file+function instead
  of line numbers. The repo's tally counts once per batch appearance, not per visit — that
  difference is declared in the comment.
- **Cited from:** `src/gnn/sampling/basic_khop_sampler.cc`,
  `src/gnn/sampling/minhash_reorderer.h`
- **Notes:** MIT license confirmed on the repo page, consistent with the pre-existing
  "Original: ... (MIT License)" note in `minhash_reorderer.h`.

---

## 4. System documentation

### Linux kernel documentation — "The /proc Filesystem"

- **Source:** The Linux Kernel documentation, "The /proc Filesystem"
  (Documentation/filesystems/proc.rst), kernel.org.
  https://docs.kernel.org/filesystems/proc.html
- **Verified:** 2026-08-27, WebFetch of the URL.
- **Section:** meminfo.
- **Quote:** "An estimate of how much memory is available for starting new applications,
  without swapping. Calculated from MemFree, SReclaimable, the size of the file LRU lists,
  and the low watermarks in each zone."
- **What it does not say:** It does not prescribe what fraction of MemAvailable a process may
  take, nor any minimum floor: the 3/4 ratio and the 256 MB floor are our decisions, and the
  comment in `available_ram.h` declares that explicitly.
- **Cited from:** `src/misc/available_ram.h`
- **Notes:** The same definition backs using MemAvailable (not MemFree, not total RAM) to
  size the sort buffer mid-session.

---

## 5. Verification-support sources (not cited from code)

### Wikipedia, "Graph Query Language"

- **Source:** Wikipedia, "Graph Query Language".
  https://en.wikipedia.org/wiki/Graph_Query_Language
- **Verified:** 2026-08-27, WebFetch of the URL.
- **Section:** main article (publication of the standard).
- **Quote:** "The GQL standard, ISO/IEC 39075:2024 Information technology – Database
  languages – GQL, was officially published by ISO on 12 April 2024."
- **What it does not say:** It does not contain the text of clauses 15.1 or 17.2; it only
  confirms the standard's designation, title, and date.
- **Cited from:** (none — used only as public verification of the ISO standard's existence;
  never cited from code)
- **Notes:** Supports the ISO/IEC 39075:2024 entry above.
