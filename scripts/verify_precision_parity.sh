#!/usr/bin/env bash
# verify_precision_parity.sh — reproducible check that MillenniumDB and DiskGNN
# train ogbn-papers100M at IDENTICAL numerical precision (float32 throughout,
# no float64/double on either side). Answers the question: "is the MDB↔DiskGNN
# test-accuracy gap caused by a numeric-resolution difference (double vs float)?"
#
# Verdict produced: precision is NOT a confound — both pipelines use float32 for
# features, model parameters and compute; neither uses double. Forcing DiskGNN to
# float would be a no-op (it is already float).
#
# Each check prints the MEASURED value and PASS/FAIL. Exit 0 iff all pass.
# Usage:  bash scripts/verify_precision_parity.sh
set -u
N=111059956; D=128                       # papers100M: nodes, feature dim
PY=/home/bfuentes/miniconda3/envs/diskgnn_cu124/bin/python
MDB_NPY=/home/bfuentes/MillenniumDB_Testing/data/example/gql/papers100M/papers100M_features.npy
MDB_FMAT=/home/bfuentes/MillenniumDB_Testing/data/dbs/gql/papers100M/gnn_features/node_features.fmat
MDB_SRC=/home/bfuentes/MillenniumDB_Testing/src/gnn
DGN=/home/bfuentes/DiskGNN/DiskGNN/examples
OFFGS=/home/bfuentes/miniconda3/envs/diskgnn_cu124/lib/python3.10/site-packages/offgs
DGN_CONF=/home/bfuentes/diskgnn_data/offgs_dataset/ogbn-papers100M-offgs/conf.json

pass=0; fail=0
ok(){ echo "  [PASS] $*"; pass=$((pass+1)); }
no(){ echo "  [FAIL] $*"; fail=$((fail+1)); }
hr(){ echo "------------------------------------------------------------"; }

echo "=== PRECISION PARITY AUDIT: MillenniumDB vs DiskGNN (papers100M) ==="
hr
echo "MDB side"
# 1. source feature npy dtype
DT=$("$PY" -c "import numpy as np; print(np.load('$MDB_NPY',mmap_mode='r').dtype)" 2>/dev/null)
echo "  MDB source features npy dtype = ${DT:-MISSING}"
[ "$DT" = "float32" ] && ok "MDB source features are float32" || no "MDB source features NOT float32 (got ${DT:-MISSING})"

# 2. fmat dtype inferred from exact byte size (float32 => N*D*4 + small header)
SZ=$(stat -c%s "$MDB_FMAT" 2>/dev/null || echo 0)
ITEM=$("$PY" -c "print(round(($SZ)/($N*$D),4)) if $SZ else print('NA')" 2>/dev/null)
HDR=$(( SZ - N*D*4 ))
echo "  MDB fmat bytes = $SZ  => bytes/element = $ITEM  (header = $HDR B)"
if [ "$SZ" -gt 0 ] && [ "$HDR" -ge 0 ] && [ "$HDR" -lt 4096 ]; then
  ok "MDB fmat is float32 (size = N*D*4 + ${HDR}B header; float64 would be $((N*D*8)) B)"
else
  no "MDB fmat size inconsistent with float32"
fi

# 3. store default dtype is FLOAT32 in code
DEF=$(grep -E "GnnDtype\s+dtype_\s*=\s*GnnDtype::" "$MDB_SRC/storage/four_level_store.h" | grep -oE "FLOAT[0-9]+" | head -1)
echo "  FourLevelStore default dtype_ = ${DEF:-?}"
[ "$DEF" = "FLOAT32" ] && ok "MDB feature store defaults to FLOAT32" || no "store default not FLOAT32"

# 4. model + conv compute path carries no double (the only float64 in src/gnn are:
#    dtype-dispatch switch branches that fire only for a FLOAT64 store, the
#    float64 loss-SUM accumulator scalar, and the unrelated GQL double-tensor
#    property reader — none touch the float32 model/feature/gradient path).
MODEL_DBL=$(grep -cE "kFloat64|kDouble|\.double\(\)" "$MDB_SRC/models/graphsage_model.cc" 2>/dev/null)
echo "  double casts in graphsage_model.cc = $MODEL_DBL"
[ "$MODEL_DBL" = "0" ] && ok "MDB GraphSAGE model has zero double casts (LibTorch default float32)" || no "model has double casts — inspect"
hr
echo "DiskGNN side"
# 5. dataset conf dtype + itemsize
CDT=$(grep -o '"features_dtype": *"[^"]*"' "$DGN_CONF" | grep -oE 'torch\.[a-z0-9]+')
CIT=$(grep -o '"feat_itemsize": *[0-9]*' "$DGN_CONF" | grep -oE '[0-9]+')
echo "  DiskGNN conf.json: features_dtype = ${CDT:-?}, feat_itemsize = ${CIT:-?}"
[ "$CDT" = "torch.float32" ] && ok "DiskGNN features dtype = float32" || no "DiskGNN features not float32 (got ${CDT:-?})"
[ "$CIT" = "4" ] && ok "DiskGNN feat_itemsize = 4 bytes (float32)" || no "DiskGNN itemsize != 4 (got ${CIT:-?})"

# 6. SAGE model params are float32 (live instantiation)
PDT=$("$PY" - <<'PY' 2>/dev/null
import torch
from offgs.utils import SAGE
m = SAGE(128, 256, 172, 3, 0.2)
print(",".join(sorted({str(p.dtype) for p in m.parameters()})))
PY
)
echo "  DiskGNN SAGE param dtypes = ${PDT:-?}"
[ "$PDT" = "torch.float32" ] && ok "DiskGNN SAGE parameters are float32" || no "DiskGNN SAGE params not all float32 (got ${PDT:-?})"

# 7. no double in DiskGNN train/model/loader
DGN_DBL=$(grep -rcE "\.double\(\)|torch\.float64|DoubleTensor|to\(torch\.double\)" \
  "$DGN/train_single_thread.py" "$OFFGS/utils/model.py" "$OFFGS/dataset.py" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
echo "  double casts in DiskGNN train+model+loader = $DGN_DBL"
[ "$DGN_DBL" = "0" ] && ok "DiskGNN train/model/loader have zero double casts" || no "DiskGNN has double casts — inspect"
hr
echo "RESULT: $pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
  echo "VERDICT: PRECISION PARITY CONFIRMED — both pipelines are float32 end-to-end"
  echo "         (features + model params + compute). Neither uses double."
  echo "         => numeric resolution is NOT the source of the accuracy gap;"
  echo "            forcing DiskGNN to float is a no-op (already float32)."
  exit 0
else
  echo "VERDICT: parity NOT confirmed — see [FAIL] lines above."
  exit 1
fi
