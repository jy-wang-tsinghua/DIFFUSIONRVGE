#!/usr/bin/env bash
# Evaluate a saved checkpoint (the averaged weights in its `ema` slot) on a test split: 40-step ODE, no CFG.
# usage: bash scripts/evaluate.sh {pickscore|ocr|geneval} CHECKPOINT_STATE_PT OUTPUT_DIR
#   pickscore: DrawBench, 200 prompts x 1 image      ocr: 1,016 test prompts      geneval: 553 test prompts x 4 seeds
cd "$(dirname "$0")/.."; source scripts/_common.sh
reward="${1:?reward}"; ckpt="${2:?checkpoint state.pt}"; out="${3:?output dir}"
case "$reward" in
  pickscore) args=(--panel dataset/pickscore/panel.json --eval-prompts-file dataset/drawbench_test_unique.txt --eval-prompt-limit 200);;
  ocr)       args=(--panel dataset/ocr/panel.json --eval-prompt-limit 1016 --ocr-backend "${OCR_BACKEND:-cpu}");;
  geneval)   args=(--panel dataset/geneval/panel_test.json --eval-prompt-limit 553 --eval-seeds 4 --prepare-local-prompts --shard-mode block);;
  *) echo "unknown reward $reward"; exit 1;;
esac
run --output "$out" --method endpoint --reward "$reward" "${args[@]}" --resume "$ckpt" --updates 100000 --allow-resume-lr-change --eval-only --eval-policy ema
