#!/usr/bin/env bash
# OCR: trajectory bank (8 prompts x 12 trajectories, eta = 0.5, head-8 rows), 120 updates. The gain scale is the running
# scale for the first 20 updates and fixed at 0.2 afterwards; the learning rate follows a cosine from 3e-4 to 3e-5 after update 20.
# OCR scoring uses PaddleOCR. OCR_BACKEND=cpu (default) scores in CPU worker processes; OCR_BACKEND=gpu runs PaddleOCR on
# each rank's GPU in a separate interpreter given by RVGE_OCR_PYTHON (a Python with paddlepaddle-gpu and paddleocr).
# usage: bash scripts/train_ocr.sh OUTPUT_DIR [extra trainer flags]
cd "$(dirname "$0")/.."; source scripts/_common.sh
out="${1:?usage: train_ocr.sh OUTPUT_DIR [flags]}"; shift
backend="${OCR_BACKEND:-cpu}"; workers=16; [ "$backend" = gpu ] && workers=1
run --output "$out" "${COMMON[@]}" --reward ocr --panel dataset/ocr/panel.json \
  --group-size 12 --groups-per-update 8 --expected-trajectories-per-update 96 --unique-prompts-per-update \
  --noise-radius 0.5 --reward-normalization gain_uncentered --gain-std-fixed 0.2 --gain-std-fixed-start 20 \
  --lr 3e-4 --lr-schedule cosine --lr-decay-start 20 --lr-final 3e-5 --updates 120 --save-every 20 \
  --ocr-backend "$backend" --ocr-workers "$workers" --eval-prompt-limit 1016 "$@"
