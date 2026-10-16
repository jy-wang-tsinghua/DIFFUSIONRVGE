# RVGE: single-reward training code

Code for the single-reward experiments of *DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient
Estimation* (anonymous submission). It fine-tunes Stable Diffusion 3.5 Medium with a rank-32 LoRA on one reward at a
time: PickScore, OCR, or GenEval.

## Contents

| Path | Content |
|---|---|
| `scripts/train.py` | Trainer (single GPU or `torchrun`): rollouts, reward scoring, clean-image regression, linear actor update |
| `scripts/train_{pickscore,ocr,geneval}.sh` | The three single-reward recipes of the paper |
| `scripts/evaluate.sh` | Test-split readout of a saved checkpoint |
| `rvge/endpoint_gradient.py` | Normalized ridge regression of reward differences on clean-image differences |
| `rvge/stepwise_rows.py`, `rvge/stepwise_credit.py` | SDE trajectory bank with ODE completions; independent-seed ODE groups; sampler coefficients |
| `rvge/stepwise_group_fit.py` | Group regression for discrete rewards (leave-one-out centring, lowest-reward query selection) |
| `rvge/stepwise_normalization.py` | Reward-gain preprocessing of the trajectory bank |
| `rvge/fdfo_aligned_loss.py` | Linear velocity loss with a stopped regression direction |
| `rvge/stepwise_runtime.py` | SD3.5 + LoRA runtime, decoding and reward scoring |
| `rvge/pickscore.py`, `rvge/stepwise_ocr.py`, `rvge/stepwise_geneval.py` | Reward wrappers |
| `scripts/prepare_data.py` | Builds the prompt panels in `dataset/` from Flow-GRPO's public prompt files |
| `dataset/` | DrawBench evaluation prompts; held-out GenEval validation indices; the panels after `prepare_data.py` |

The six source prompt files come from the public Flow-GRPO repository at a pinned commit. The URL in `scripts/prepare_data.py` identifies the benchmark data source; it is not a repository for this submission's code.

## Released weights

**Five-reward LoRA:**
[roundman/DiffusionRVGE-SD3.5M-MultiReward](https://huggingface.co/roundman/DiffusionRVGE-SD3.5M-MultiReward/tree/main).
This is the joint model of the paper's multi-reward comparison: a rank-32 LoRA for SD3.5 Medium trained jointly on five
rewards, namely PickScore, CLIPScore and HPSv2.1 (combined, on Pick-a-Pic prompts), GenEval, and OCR. This repository
contains the single-reward recipes only; the multi-reward training script is not included.

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("roundman/DiffusionRVGE-SD3.5M-MultiReward")  # alpha / r is folded in; keep the default scale 1.0
image = pipe("a photo of a red bicycle next to a blue umbrella",
             num_inference_steps=40, guidance_scale=1.0, height=512, width=512).images[0]
```

## Setup

```bash
pip install -r requirements.txt
python scripts/prepare_data.py   # downloads the Pick-a-Pic, OCR and GenEval prompt files and builds dataset/*/panel*.json
```

Models are downloaded from Hugging Face on first use: `stabilityai/stable-diffusion-3.5-medium` (accept its license
first), `yuvalkirstain/PickScore_v1`, and `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`.

**OCR reward.** PaddleOCR scores rendered text by normalized edit distance. By default it runs in CPU worker processes
(`OCR_BACKEND=cpu`). To run it on each rank's GPU instead, create a separate interpreter with `paddlepaddle-gpu` and
`paddleocr`, then set `OCR_BACKEND=gpu RVGE_OCR_PYTHON=/path/to/that/python`.

**GenEval reward.** GenEval runs out of process in its own environment (Mask2Former via mmdet, CLIP ViT-L/14), exactly
as in Flow-GRPO. Install the GenEval reward server used by Flow-GRPO following the Flow-GRPO instructions, then set
`RVGE_GENEVAL_SERVER` to the checkout that contains `reward_server/gen_eval.py` and `RVGE_GENEVAL_PYTHON` to the Python
interpreter of that environment.

## Training

All recipes use 8 GPUs with 40 GB each (`NGPU` and `PORT` override the defaults).

```bash
bash scripts/train_pickscore.sh runs/pickscore   # 200 updates, about 110 GPU-hours
bash scripts/train_ocr.sh runs/ocr               # 120 updates, about 20 GPU-hours
bash scripts/train_geneval.sh runs/geneval       # 400 updates
```

A run writes `resolved_config.json`, per-update `metrics.jsonl`, periodic validation readouts in `eval_metrics.jsonl`,
full checkpoints under `checkpoints/checkpoint-XXXXXX/state.pt`, and per-update LoRA weights. `--resume <state.pt>`
continues a run. Extra flags after the output directory are passed to the trainer (`python scripts/train.py --help`).

## Evaluation

```bash
bash scripts/evaluate.sh pickscore runs/pickscore/checkpoints/checkpoint-000200/state.pt eval/pickscore
bash scripts/evaluate.sh ocr       runs/ocr/checkpoints/checkpoint-000120/state.pt       eval/ocr
bash scripts/evaluate.sh geneval   runs/geneval/checkpoints/checkpoint-000400/state.pt   eval/geneval
```

Evaluation uses a 40-step ODE sampler without guidance: PickScore on 200 DrawBench prompts, OCR on the 1,016 test
prompts, and GenEval on the 553 test prompts with four seeds each. The eight-metric panel in the paper uses
DiffusionNFT's public evaluation script, which is not included here.
