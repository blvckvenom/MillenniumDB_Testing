#pragma once

#include "query/procedure/procedure.h"

namespace GQL::Procedures {

class GnnTrainProcedure : public Procedure {
public:
    std::string name() const override { return "gnn_train"; }
    std::string qualified_name() const override { return "gnn_train"; }
    std::string description() const override {
        return "Train a GraphSAGE model on a pre-sampled graph projection with offline "
               "batches, labels, and train/val/test splits. Exports model checkpoint, "
               "training log, and optionally node embeddings.";
    }

    std::vector<Parameter> parameters() const override {
        return {
            Parameter("sampleName",  ParamType::STRING, true,  "Existing sample set name"),
            Parameter("featureName", ParamType::STRING, true,  "Registered feature name (e.g. 'node_features')"),
            Parameter("options",     ParamType::ANY,    false,
                "Options: {model, hiddenDim, dropout, epochs, lr, weightDecay, "
                "patience, tolerance, normalize, randomSeed, outputDir, "
                "exportEmbeddings, writeProperty, writeCoverage, inferenceBatchSize, resumeFrom, "
                "saveOnBestVal, saveFinal, sampleCacheMb, useAsyncPrefetcher, "
                "useCudaStreams, prefetchNumWorkers, prefetchQueueSize, profileLog, "
                "readOnlyBench, noBlocks, noSelfContained, noPackedFull, "
                "trackTestAtBestVal, lrSchedule}"),
        };
    }

    std::vector<YieldField> yield_fields() const override {
        return {
            YieldField{"modelName",       YieldType::STRING, "Model identifier (e.g. 'graphsage')"},
            YieldField{"ranEpochs",       YieldType::INT,    "Number of epochs executed"},
            YieldField{"didConverge",     YieldType::BOOL,   "Stopped by convergence or early stopping"},
            YieldField{"bestValAccuracy", YieldType::FLOAT,  "Best validation accuracy achieved"},
            YieldField{"testAccuracy",    YieldType::FLOAT,  "Test accuracy at the FINAL epoch (-1.0 if no test split)"},
            YieldField{"testAccuracyAtBestVal", YieldType::FLOAT, "Test accuracy at the best-validation epoch (paper protocol); -1.0 if trackTestAtBestVal is off or no test split"},
            YieldField{"bestValEpoch",    YieldType::INT,    "0-indexed epoch that produced bestValAccuracy (0 if trackTestAtBestVal is off)"},
            YieldField{"trainSeconds",    YieldType::FLOAT,  "Total wall-clock training time in seconds"},
            YieldField{"assembleSeconds", YieldType::FLOAT,  "Cumulative batch assemble time across all train batches in seconds"},
            YieldField{"forwardSeconds",  YieldType::FLOAT,  "Cumulative model forward time across all train batches in seconds"},
            YieldField{"backwardSeconds", YieldType::FLOAT,  "Cumulative model backward time across all train batches in seconds"},
            YieldField{"l1HitRatio",      YieldType::FLOAT,  "FourLevelStore L1 (GPU) cache hit ratio"},
            YieldField{"l2HitRatio",      YieldType::FLOAT,  "FourLevelStore L2 (CPU pinned) cache hit ratio"},
            YieldField{"l3Reads",         YieldType::INT,    "FourLevelStore L3 (disk reordered) read count"},
            YieldField{"l4Reads",         YieldType::INT,    "FourLevelStore L4 (packed batch) read count"},
            YieldField{"l3BytesDisk",     YieldType::INT,    "Bytes read from disk by L3 (page-granular)"},
            YieldField{"l4BytesDisk",     YieldType::INT,    "Bytes read from disk by L4 packed batches"},
            YieldField{"totalBytesDisk",  YieldType::INT,    "Total feature-store disk traffic in bytes (l3BytesDisk + l4BytesDisk)"},
            YieldField{"l3ReadAmplification", YieldType::FLOAT, "L3 bytes read from disk / bytes actually wanted"},
            YieldField{"nodesWritten",    YieldType::INT,    "Nodes whose embeddings were written to projection (0 if writeProperty not set)"},
            YieldField{"nodesInferred",   YieldType::INT,    "Non-seed nodes inferred during write-back (0 if writeProperty not set)"},
            YieldField{"inferenceMillis", YieldType::FLOAT,  "Wall-clock time for non-seed inference in ms (0.0 if writeProperty not set)"},
            YieldField{"writeMillis",     YieldType::FLOAT,  "Wall-clock time for projection writes in ms (0.0 if writeProperty not set)"},
            YieldField{"writeCoverage",   YieldType::STRING, "Node set the write-back covered: 'all' or 'seeds' (empty if writeProperty not set)"},
            YieldField{"bestCheckpointPath",  YieldType::STRING, "Absolute path (no extension) to best_model checkpoint; empty if disabled or no improvement"},
            YieldField{"finalCheckpointPath", YieldType::STRING, "Absolute path (no extension) to final_model checkpoint; empty if disabled"},
            YieldField{"resumedFromEpoch",    YieldType::INT,    "Epoch index from which training resumed (0 if fresh training)"},
            YieldField{"effectivePrefetchWorkers", YieldType::INT, "Actual number of AsyncBatchPrefetcher workers used (resolved from prefetchNumWorkers or env MDB_GNN_PREFETCH_WORKERS; N>1 is faster but not bit-reproducible)"},
            YieldField{"useAddrTablesEffective", YieldType::BOOL, "True if the v2 addr-table fast path served at least one batch"},
            YieldField{"addrTableLoadUs", YieldType::DOUBLE, "Mean per-batch addr-table load time in microseconds"},
        };
    }

    void execute(ProcedureContext& ctx) override;
};

} // namespace GQL::Procedures
