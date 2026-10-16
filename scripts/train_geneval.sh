#!/usr/bin/env bash
# GenEval: discrete-reward independent-seed groups (48 prompts x 24 ODE trajectories; each prompt trains its 6 lowest-reward
# trajectories, regressing on the other 23 clean images centred at their leave-one-out mean), 400 updates.
# Requires the GenEval reward server (see README): RVGE_GENEVAL_SERVER=<checkout>, RVGE_GENEVAL_PYTHON=<its interpreter>.
# usage: bash scripts/train_geneval.sh OUTPUT_DIR [extra trainer flags]
cd "$(dirname "$0")/.."; source scripts/_common.sh
out="${1:?usage: train_geneval.sh OUTPUT_DIR [flags]}"; shift
: "${RVGE_GENEVAL_SERVER:?set RVGE_GENEVAL_SERVER to the GenEval reward-server checkout (see README)}"
run --output "$out" "${COMMON[@]}" --reward geneval --panel dataset/geneval/panel_train.json \
  --group-size 24 --groups-per-update 48 --expected-trajectories-per-update 1152 \
  --noise-radius 0.5 --rollout-mode ode_group --train-per-group 6 --train-select low --sibling-centre mean \
  --shard-mode block --prepare-local-prompts --reward-scale 1 --adam-eps 1e-15 \
  --lr 3e-4 --updates 400 --save-every 10 --eval-prompt-limit 440 --eval-policy both "$@"
