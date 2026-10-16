# Shared launcher settings. Usage: source scripts/_common.sh (from the RVGE root).
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="${PYTHON:-python}"
NGPU="${NGPU:-8}"
PORT="${PORT:-29500}"
run() { exec "$PYTHON" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port "$PORT" scripts/train.py "$@"; }
# Options shared by every recipe: 10-step rollouts, rank-32 LoRA, one AdamW step per collection, linear actor loss.
COMMON=(--method endpoint --time-exponent 1.0 --direction-normalization none --actor-loss fdfo_first_order
        --ridge 0.3 --fit-scope head --independent-roots --batching rows --train-microbatch 6
        --sde-noise-mode constant --grad-clip 15 --trainable-dtype float32 --scorer-dtype float32
        --rollout-batching grouped --rollout-chunk 18 --score-chunk 33 --eval-chunk 32 --no-gradient-checkpointing
        --paper-max-btau inf --eval-every 10 --eval-seeds 1 --save-lora-every 1 --ema-decay 0.98)
