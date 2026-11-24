#!/usr/bin/env bash
# The 3-slide stage checks of E1 tile snapping, after tools/run_snap_e1.sh built the slides (olympus-compute).
#   bash tools/run_snap_checks.sh ROOT GPU ID [ID ...]
# 1 exposure / identity / sharpness (CPU)       -> ROOT/eval/*.tsv
# 2 features of both arms with Mei's extractor: student twin2i (gray) and the 7B teacher (packed), at the E1 coords
#                                               -> ROOT/feat/<enc>_<arm>/<id>.h5
# 3 whole-slide per-cell P(cancer), arm A vs B   -> ROOT/eval/score_<enc>.tsv + _summary.json
# 4 boundary-sensitivity probe on the probe cells -> ROOT/eval/probe/
# Read-only use of Mei's code, environment, checkpoints, heads, labels and HF cache; writes only under ROOT.
set -uo pipefail
ROOT=$1; GPU=$2; shift 2; IDS=("$@")
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/mwang/MagicScan_cancer_detection/prostate_env/bin/python
FM=/sci-it/hosts/olympus/magicscan/SIM/3_SIM_foundation_model
DS=Prostate_VA_MS_SYBR_EO_snapE1
COORD=/sci-it/hosts/olympus/magicscan/SIM/0_Datasets/0.5_mpp/0_coord_h5
CK2I=/scratch/scratch_nvme/mwang/fm_phase4/runs/twin_2i/ckpt_080000.pt
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_CACHE=/scratch/scratch_nvme/mwang/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
E=$ROOT/eval; mkdir -p $E $ROOT/feat
log() { echo "[$(date '+%F %T')] $*" | tee -a $E/_checks.log; }
log "checks for ${IDS[*]} (repo $(git -C $REPO rev-parse --short HEAD), gpu $GPU)"
for c in exposure sharpness identity; do
  $PY $REPO/tools/snap_eval.py $c $ROOT "${IDS[@]}" --out $E > $E/$c.log 2>&1 && log "$c done" || log "$c FAILED (see $E/$c.log)"
done
for arm in A B; do
  sfx=""; [ $arm = B ] && sfx="__snapE1"
  : > $E/pairs_$arm.tsv
  for id in "${IDS[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\n" $id $ROOT/$DS/grayscale/${id}${sfx}_470.tif $ROOT/$DS/grayscale/${id}${sfx}_528.tif \
      $COORD/${id}_fullstack_he_patches.h5 $id >> $E/pairs_$arm.tsv
  done
  $PY $FM/distill/extract_fm_bench.py --pairs $E/pairs_$arm.tsv --out $ROOT/feat/student_$arm --encoder student --ckpt $CK2I \
    > $ROOT/feat/student_$arm.log 2>&1 && log "student features $arm" || log "student features $arm FAILED"
  $PY $FM/distill/extract_fm_bench.py --pairs $E/pairs_$arm.tsv --out $ROOT/feat/teacher_$arm --encoder dinov3_7b --view packed \
    > $ROOT/feat/teacher_$arm.log 2>&1 && log "teacher features $arm" || log "teacher features $arm FAILED"
done
$PY $REPO/tools/snap_score.py "${IDS[@]}" --heads mw_e1_twin2i_080k_e1reg --dim 768 --feat_a $ROOT/feat/student_A \
  --feat_b $ROOT/feat/student_B --out $E/score_student.tsv > $E/score_student.log 2>&1 && log "score student" || log "score student FAILED"
$PY $REPO/tools/snap_score.py "${IDS[@]}" --heads mw_e1_7b_packed --dim 4096 --feat_a $ROOT/feat/teacher_A \
  --feat_b $ROOT/feat/teacher_B --out $E/score_teacher.tsv > $E/score_teacher.log 2>&1 && log "score teacher" || log "score teacher FAILED"
$PY $REPO/tools/snap_probe.py $ROOT "${IDS[@]}" --out $E/probe > $E/probe.log 2>&1 && log "probe done" || log "probe FAILED (see $E/probe.log)"
log "CHECKS END"
