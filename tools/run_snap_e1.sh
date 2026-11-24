#!/usr/bin/env bash
# Production MS recipe (fm_build HANDOFF 2026-11-18) + E1 tile snapping, one slice at a time, on olympus-compute.
# Each slice writes two arms into $ROOT/$DS: <id> (the builder's own EDOF, arm A) and <id>__snapE1 (arm B), plus
# _qc/$DS/snap/<id>_snap_tiles.npz and the probe-cell crops. Nothing is written outside $ROOT.
#   QUEUE  TSV: image_id <TAB> raw folder under the VA root <TAB> name filter or -
#   bash tools/run_snap_e1.sh QUEUE CELLS_TSV ROOT [GPU]
# Slices whose log already says "all variants done" are skipped; a failed slice is retried twice, 20 min apart.
# Before each slice the script waits while any build_ms.py of another user is running (one raw reader at a time).
set -uo pipefail
QUEUE=$1; CELLS=$2; ROOT=$3; GPU=${4:-1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/mwang/MagicScan_cancer_detection/prostate_env/bin/python
V="/sci-it/hosts/olympus/magicscan/SIM/Prostate study - VA"
DS=Prostate_VA_MS_SYBR_EO_snapE1
LOG=$ROOT/logs; mkdir -p "$LOG" "$ROOT/hf_home"
RECIPE="--workers 6 --ff select --strip_match gain --mask clipseg --reg_mode ours --reg_workers 24 --read_workers 6 --edof_fallback on --edof_flat_min_mm2 1.0 --z_variants off"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
others() { ps -eo user=,args= | awk -v me="$USER" '$1 != me && /[b]uild_ms\.py/' | wc -l; }
for pass in 1 2 3; do
  left=0
  while IFS=$'\t' read -r id folder filt; do
    case "$id" in ''|\#*) continue;; esac
    grep -q "all variants done" "$LOG/$id.log" 2>/dev/null && continue
    while [ "$(others)" -gt 0 ]; do echo "[$(date '+%F %T')] $id: another user's build_ms.py is running; waiting" >> "$LOG/_batch.log"; sleep 600; done
    echo "[$(date '+%F %T')] start $id pass $pass gpu $GPU (repo $(git -C "$REPO" rev-parse --short HEAD))" >> "$LOG/_batch.log"
    NF=(); [ "${filt:--}" != "-" ] && NF=(--name_filter "$filt")
    "$PY" "$REPO/build_registered.py" --builder fm --register builtin --snap e1 --snap_probe_cells "$CELLS" -- \
      "$V/$folder" --image_id "$id" --dataset "$DS" --dataset_root "$ROOT" --io_lock "$ROOT/nfs_read.lock" \
      "${NF[@]}" $RECIPE > "$LOG/$id.log" 2>&1 < /dev/null
    rc=$?
    grep -q "all variants done" "$LOG/$id.log" && echo "[$(date '+%F %T')] done $id" >> "$LOG/_batch.log" \
      || { echo "[$(date '+%F %T')] FAILED $id rc $rc: $(tail -n 2 "$LOG/$id.log" | tr '\n' ' ')" >> "$LOG/_batch.log"; left=$((left+1)); }
  done < "$QUEUE"
  [ "$left" -eq 0 ] && break
  [ "$pass" -lt 3 ] && sleep 1200
done
echo "[$(date '+%F %T')] SNAP_E1 BATCH END ($left not done)" >> "$LOG/_batch.log"
