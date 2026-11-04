#!/bin/bash
# Rebuild + compare MagicScan slides on voodoo (CPU only).
#   cd <repo>; sbatch -o LOGDIR/ms-rebuild-%j.out -e LOGDIR/ms-rebuild-%j.err tools/voodoo_rebuild.sbatch SLIDES_TSV OUT_DIR
# SLIDES_TSV: image_id <TAB> slice dir under /mnt/arpa-h-data <TAB> optional name filter <TAB> optional extra
#             rebuild_edof.py arguments (e.g. --no_prefetch for slides with many long planes)
# Note: docker containers are not inside the SLURM memory limit; keep concurrent jobs within the machine's RAM.
#SBATCH --job-name=ms-rebuild
#SBATCH --output=ms-rebuild-%j.out
#SBATCH --error=ms-rebuild-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
set -uo pipefail
SLIDES=$1; OUT=$2
REPO=${MSPROG_REPO:-$SLURM_SUBMIT_DIR}     # submit from the repo root (SLURM runs a copy of this script)
IMG=${MSPROG_IMAGE:-$USER/msprog:v1}
mkdir -p "$OUT"
run() {
  srun --ntasks=1 docker run --rm --user "$(id -u):$(id -g)" --ipc=host \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
    -v /mnt/arpa-h-data:/mnt/arpa-h-data:ro -v "$REPO":/repo:ro -v "$OUT":"$OUT" -w /repo "$IMG" python "$@"
}
while IFS=$'\t' read -r ID DIR FILT EXTRA; do
  [ -z "$ID" ] || [ "${ID:0:1}" = "#" ] && continue
  echo "=== $ID $(date)"
  ARGS=("tools/rebuild_edof.py" "$DIR" --image_id "$ID" --out "$OUT/$ID" --workers 32)
  [ -n "${FILT:-}" ] && ARGS+=(--name_filter "$FILT")
  [ -n "${EXTRA:-}" ] && read -r -a XA <<< "$EXTRA" && ARGS+=("${XA[@]}")
  run "${ARGS[@]}" || { echo "REBUILD FAILED $ID"; continue; }
  A=$(awk -F'\t' '$1=="off"{print $2}' "$OUT/$ID/specs.tsv"); B=$(awk -F'\t' '$1=="pearson"{print $2}' "$OUT/$ID/specs.tsv")
  run edof_align_check.py compare "$A" "$B" --out "$OUT/$ID/compare" --workers 32 || { echo "CHECK FAILED $ID"; continue; }
  run tools/preview_pair.py "$OUT/$ID/compare" --out "$OUT/$ID/preview" || echo "PREVIEW FAILED $ID"
done < "$SLIDES"
echo "ALL DONE $(date)"
