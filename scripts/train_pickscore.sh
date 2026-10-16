#!/usr/bin/env bash
# PickScore: continuous-reward trajectory bank (24 prompts x 24 trajectories, eta = 0.03, head-8 rows), 200 updates.
# usage: bash scripts/train_pickscore.sh OUTPUT_DIR [extra trainer flags]
cd "$(dirname "$0")/.."; source scripts/_common.sh
out="${1:?usage: train_pickscore.sh OUTPUT_DIR [flags]}"; shift
run --output "$out" "${COMMON[@]}" --reward pickscore --panel dataset/pickscore/panel.json \
  --group-size 24 --groups-per-update 24 --expected-trajectories-per-update 576 --unique-prompts-per-update \
  --noise-radius 0.03 --reward-normalization gain_uncentered --lr 3e-4 --updates 200 --save-every 10 \
  --eval-prompts-file dataset/drawbench_test_unique.txt --eval-prompt-limit 200 --eval-policy both "$@"
