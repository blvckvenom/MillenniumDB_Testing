#!/usr/bin/env bash
# Path 4 bit-identical gate: gnn_train val_acc matches between
# v2 (addr_tables enabled) and legacy (addr_tables removed) paths.
#
# Cora schema: Paper nodes, CITES edges, features imported as 'node_features'
# (via --with-tensors at DB creation), 'label' property.
# No predefined split property in cora.gql — uses ratio-based splits (70/15/15).
#
# NOTE: The pre-built data/dbs/gql/cora_gnn DB has tensors.dat=0 bytes
# (imported without --with-tensors). This script creates a fresh temp DB
# with proper feature import so graph_project succeeds.
set -euo pipefail

REPO=/home/bfuentes/MillenniumDB_Testing
MDB=$REPO/build/Release/bin/mdb
CORA_GQL=$REPO/data/example/gql/cora/cora.gql
CORA_NPY=$REPO/data/example/gql/cora/cora_features.npy
PORT=9123
OUT=/tmp/addrtab_bit_identical_$$
DB=/tmp/cora_gnn_addrtest_$$
mkdir -p "$OUT"

# Samples are stored at <db>/samples/<name> (not inside the projection dir).
SAMPLE_DIR="$DB/samples/cora_sample"

cleanup() {
    pkill -INT -f "bin/mdb server $DB" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "bin/mdb server $DB" 2>/dev/null || true
    rm -rf "$DB"
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# Step 0: Generate cora_features.npy if not already present
# -----------------------------------------------------------------------
if [ ! -f "$CORA_NPY" ]; then
    echo "=== Generating cora_features.npy from cora.gql ==="
    python3 - <<'PYEOF'
import re, sys, numpy as np
gql = '/home/bfuentes/MillenniumDB_Testing/data/example/gql/cora/cora.gql'
out = '/home/bfuentes/MillenniumDB_Testing/data/example/gql/cora/cora_features.npy'
nodes = {}
with open(gql) as f:
    for line in f:
        m = re.match(r'^(\d+) :Paper .*features:\[([^\]]+)\]', line.strip())
        if m:
            idx = int(m.group(1))
            feat = [float(x.strip()) for x in m.group(2).split(',')]
            nodes[idx] = feat
n = max(nodes.keys()) + 1
d = len(nodes[0])
mat = np.zeros((n, d), dtype=np.float32)
for i, f in nodes.items():
    mat[i] = f
np.save(out, mat)
print(f'Saved ({n}, {d}) float32 -> {out}')
PYEOF
else
    echo "=== cora_features.npy already exists ($CORA_NPY) ==="
fi

# -----------------------------------------------------------------------
# Step 1: Create fresh temp DB with --with-tensors
# -----------------------------------------------------------------------
echo "=== Creating temp DB with tensor import ==="
"$MDB" import "$CORA_GQL" "$DB" --with-tensors "$CORA_NPY" > "$OUT/00_import.log" 2>&1
echo "  Import log tail:"
tail -5 "$OUT/00_import.log"
echo

# Verify feature was registered as 'node_features'
if [ ! -f "$DB/gnn_features/node_features.fmat" ]; then
    echo "FAIL: node_features.fmat not found after import"
    cat "$OUT/00_import.log"
    exit 1
fi
echo "  node_features.fmat OK."

# -----------------------------------------------------------------------
# Step 2: Start server on temp DB
# -----------------------------------------------------------------------
echo "=== Starting mdb server on temp cora DB ==="
"$MDB" server "$DB" -p $PORT -j 4 > "$OUT/server.log" 2>&1 &
sleep 5

# Verify server up
curl -s -X POST -H "Content-Type: application/gql" --data-binary "RETURN 1;" \
     http://localhost:$PORT/query > /dev/null || { echo "FAIL: server not up"; cat "$OUT/server.log"; exit 1; }
echo "  Server is up."

# -----------------------------------------------------------------------
# Step 3: Create projection
# -----------------------------------------------------------------------
echo "=== Step 3: Create projection ==="
# Feature name is 'node_features' (hardcoded by import). No split property.
cat > "$OUT/01_project.gql" <<'EOF'
CALL graph_project('cora_proj', 'Paper', 'CITES',
    {orientation: 'UNDIRECTED',
     includeFeatures: 'node_features',
     labelProperty: 'label'})
YIELD graphName, nodeCount, relCount RETURN *;
EOF
curl -s -X POST -H "Content-Type: application/gql" --data-binary @"$OUT/01_project.gql" \
     http://localhost:$PORT/query > "$OUT/01_project.out"
echo "  Project output:"
cat "$OUT/01_project.out"
echo

# Sanity: check nodeCount > 0.
# Response is CSV: header "graphName,nodeCount,relCount" then data line.
# Data line: "cora_proj",2708,NULL — nodeCount is the second comma-separated field.
NODE_COUNT=$(tail -1 "$OUT/01_project.out" | cut -d',' -f2)
if [ -z "$NODE_COUNT" ] || { [ "$NODE_COUNT" -eq "$NODE_COUNT" ] 2>/dev/null && [ "$NODE_COUNT" -eq 0 ]; }; then
    echo "FAIL: projection returned no nodes (nodeCount='$NODE_COUNT')."
    exit 1
fi
echo "  nodeCount=$NODE_COUNT OK."

# -----------------------------------------------------------------------
# Step 4: Build sample
# -----------------------------------------------------------------------
echo "=== Step 4: Build sample ==="
# No usePredefinedSplits — cora.gql has no split property, use 70/15/15 ratios.
cat > "$OUT/02_sample.gql" <<'EOF'
CALL gnn_offline_sample('cora_proj', 'cora_sample', [10, 10],
    {batchSize: 32, randomSeed: 42, orientation: 'UNDIRECTED'})
YIELD totalBatches, trainBatches, valBatches, testBatches RETURN *;
EOF
curl -s -X POST -H "Content-Type: application/gql" --data-binary @"$OUT/02_sample.gql" \
     http://localhost:$PORT/query > "$OUT/02_sample.out"
echo "  Sample output:"
cat "$OUT/02_sample.out"
echo

# CSV response: header "totalBatches,trainBatches,valBatches,testBatches" then data.
TOTAL_BATCHES=$(tail -1 "$OUT/02_sample.out" | cut -d',' -f1)
if [ -z "$TOTAL_BATCHES" ] || { [ "$TOTAL_BATCHES" -eq "$TOTAL_BATCHES" ] 2>/dev/null && [ "$TOTAL_BATCHES" -eq 0 ]; }; then
    echo "FAIL: sampling returned no batches (totalBatches='$TOTAL_BATCHES')."
    exit 1
fi
echo "  totalBatches=$TOTAL_BATCHES OK."

# -----------------------------------------------------------------------
# Step 5: Build feature store WITH addr_tables (v2)
# -----------------------------------------------------------------------
echo "=== Step 5: Build feature store WITH addr_tables (v2) ==="
cat > "$OUT/03_build_v2.gql" <<'EOF'
CALL gnn_build_feature_store('cora_sample', 'node_features',
    {buildAddrTables: true})
YIELD addrTablesMb, addrTablesBuiltOk RETURN *;
EOF
curl -s -X POST -H "Content-Type: application/gql" --data-binary @"$OUT/03_build_v2.gql" \
     http://localhost:$PORT/query > "$OUT/03_build_v2.out"
echo "  Build (v2) output:"
cat "$OUT/03_build_v2.out"
echo

# CSV response: "addrTablesMb,addrTablesBuiltOk" header then "0,true" data.
ADDR_OK=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++) if($i=="addrTablesBuiltOk") col=i} NR==2{print $col}' "$OUT/03_build_v2.out")
if [ "$ADDR_OK" != "true" ]; then
    echo "FAIL: addrTablesBuiltOk is not true (got '$ADDR_OK') — addr_table phase did not complete."
    exit 1
fi
echo "  addrTablesBuiltOk=$ADDR_OK"

ADDR_COUNT=$(ls "$SAMPLE_DIR/addr_tables/"*.addr 2>/dev/null | wc -l)
if [ "$ADDR_COUNT" -eq 0 ]; then
    echo "FAIL: no .addr files found at $SAMPLE_DIR/addr_tables/"
    exit 1
fi
echo "  Found $ADDR_COUNT .addr file(s) OK."

# -----------------------------------------------------------------------
# Step 6: Train (V2 path, addr_tables present)
# -----------------------------------------------------------------------
echo "=== Step 6: Train (V2 path, addr_tables present) ==="
cat > "$OUT/04_train_v2.gql" <<'EOF'
CALL gnn_train('cora_sample', 'node_features',
    {epochs: 1, batchSize: 32, randomSeed: 42, patience: 999,
     hiddenDim: 64, dropout: 0.2, lr: 0.003,
     useAsyncPrefetcher: false, saveOnBestVal: false, saveFinal: false,
     outputDir: 'bit_identical_v2'})
YIELD bestValAccuracy, testAccuracy, useAddrTablesEffective, addrTableLoadUs RETURN *;
EOF
curl -s -X POST -H "Content-Type: application/gql" --data-binary @"$OUT/04_train_v2.gql" \
     http://localhost:$PORT/query > "$OUT/04_train_v2.out"
echo "  Train (v2) output:"
cat "$OUT/04_train_v2.out"
echo

# -----------------------------------------------------------------------
# Step 7: Remove addr_tables/ to force legacy path
# -----------------------------------------------------------------------
echo "=== Step 7: Remove addr_tables/ to force legacy path ==="
if [ -d "$SAMPLE_DIR/addr_tables" ]; then
    rm -rf "$SAMPLE_DIR/addr_tables"
    echo "  Removed $SAMPLE_DIR/addr_tables"
else
    echo "  WARNING: $SAMPLE_DIR/addr_tables not found — already absent?"
fi

# -----------------------------------------------------------------------
# Step 8: Train (LEGACY path, no addr_tables)
# -----------------------------------------------------------------------
echo "=== Step 8: Train (LEGACY path, no addr_tables) ==="
cat > "$OUT/05_train_legacy.gql" <<'EOF'
CALL gnn_train('cora_sample', 'node_features',
    {epochs: 1, batchSize: 32, randomSeed: 42, patience: 999,
     hiddenDim: 64, dropout: 0.2, lr: 0.003,
     useAsyncPrefetcher: false, saveOnBestVal: false, saveFinal: false,
     outputDir: 'bit_identical_legacy'})
YIELD bestValAccuracy, testAccuracy, useAddrTablesEffective, addrTableLoadUs RETURN *;
EOF
curl -s -X POST -H "Content-Type: application/gql" --data-binary @"$OUT/05_train_legacy.gql" \
     http://localhost:$PORT/query > "$OUT/05_train_legacy.out"
echo "  Train (legacy) output:"
cat "$OUT/05_train_legacy.out"
echo

# -----------------------------------------------------------------------
# Step 9: Compare val_acc
# -----------------------------------------------------------------------
echo "=== Step 9: Compare val_acc ==="
# CSV extraction helper: awk -F',' matching header column by name.
# YIELD clause produces only the requested columns in header order.
csv_field() {
    local file=$1 colname=$2
    awk -F',' -v col="$colname" '
        NR==1 { for(i=1;i<=NF;i++) if($i==col) { idx=i; break } }
        NR==2 { print $idx }
    ' "$file"
}
VAL_V2=$(csv_field "$OUT/04_train_v2.out" "bestValAccuracy")
VAL_LEGACY=$(csv_field "$OUT/05_train_legacy.out" "bestValAccuracy")
USED_V2_EFFECTIVE=$(csv_field "$OUT/04_train_v2.out" "useAddrTablesEffective")
USED_LEGACY_EFFECTIVE=$(csv_field "$OUT/05_train_legacy.out" "useAddrTablesEffective")
ADDR_LOAD_US=$(csv_field "$OUT/04_train_v2.out" "addrTableLoadUs")

echo "  V2 path val_acc:          $VAL_V2  (useAddrTablesEffective=$USED_V2_EFFECTIVE, addrTableLoadUs=${ADDR_LOAD_US}us)"
echo "  Legacy path val_acc:      $VAL_LEGACY  (useAddrTablesEffective=$USED_LEGACY_EFFECTIVE)"

if [ -z "$VAL_V2" ] || [ -z "$VAL_LEGACY" ]; then
    echo "FAIL: could not extract val_acc from training outputs."
    echo "  V2 output: $(cat "$OUT/04_train_v2.out")"
    echo "  Legacy output: $(cat "$OUT/05_train_legacy.out")"
    exit 1
fi

# Compute |val_v2 - val_legacy|
DIFF=$(awk -v a="$VAL_V2" -v b="$VAL_LEGACY" 'BEGIN { d = a - b; if (d < 0) d = -d; printf "%.6f", d }')
echo "  |Delta val_acc|:          $DIFF"

# Spec: bit-identical or within 0.0001 for float ordering
THRESHOLD=0.0001
WITHIN=$(awk -v d="$DIFF" -v t="$THRESHOLD" 'BEGIN { print (d <= t ? "1" : "0") }')

# Warn if flag behavior is unexpected
if [ "$USED_V2_EFFECTIVE" != "true" ]; then
    echo "WARN: useAddrTablesEffective=$USED_V2_EFFECTIVE on v2 run (expected true)"
fi
if [ "$USED_LEGACY_EFFECTIVE" != "false" ]; then
    echo "WARN: useAddrTablesEffective=$USED_LEGACY_EFFECTIVE on legacy run (expected false)"
fi

if [ "$WITHIN" = "1" ]; then
    echo
    echo "PASS: val_acc match (Delta=$DIFF <= $THRESHOLD)"
    echo "  v2 useAddrTablesEffective:      $USED_V2_EFFECTIVE  (expected true)"
    echo "  legacy useAddrTablesEffective:  $USED_LEGACY_EFFECTIVE  (expected false)"
    echo "  addrTableLoadUs (v2):           ${ADDR_LOAD_US}us"
    exit 0
else
    echo
    echo "FAIL: val_acc differ (Delta=$DIFF > $THRESHOLD)"
    exit 1
fi
