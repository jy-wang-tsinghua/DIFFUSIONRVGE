"""Frozen, raw PickScore reward (no /26 display normalization)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def differentiable_clip_preprocess(images: torch.Tensor) -> torch.Tensor:
    """Apply PickScore's CLIP-H image transform without breaking autograd.

    The formal SD3.5 runs generate square images, so the public processor's
    resize-short-edge followed by center crop is exactly a 224 x 224 resize.
    Keeping the input as a floating-point tensor intentionally omits the
    uint8/PIL quantization used by the ordinary read-only reward wrapper.
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError("differentiable PickScore requires a torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "images must have shape [batch,3,height,width], got "
            f"{tuple(images.shape)}"
        )
    if not images.is_floating_point():
        raise TypeError("differentiable PickScore requires floating-point images")
    resized = F.interpolate(
        images.float().clamp(0.0, 1.0),
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    mean = resized.new_tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


class RawPickScore:
    """PickScore without the legacy `/26` display normalization.

    This scorer deliberately has no trainable path.  Images are converted to
    CPU PIL before preprocessing, which prevents reward-model graph retention
    and keeps the reward model's fp16 memory separate from SD3.5's bf16 model.
    """

    processor_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id = "yuvalkirstain/PickScore_v1"

    def __init__(
        self,
        device: torch.device | str,
        chunk_size: int = 8,
        local_files_only: bool = True,
    ):
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.processor = CLIPProcessor.from_pretrained(
            self.processor_id, local_files_only=local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            self.model_id, local_files_only=local_files_only
        ).eval().to(
            self.device, dtype=torch.float16
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        if images.shape[0] != len(prompts):
            raise ValueError("PickScore image/prompt batch mismatch")
        scores = []
        for start in range(0, len(prompts), self.chunk_size):
            end = min(start + self.chunk_size, len(prompts))
            pil_images = _as_pil(images[start:end])
            image_inputs = self.processor(images=pil_images, return_tensors="pt")
            text_inputs = self.processor(
                text=list(prompts[start:end]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {
                key: value.to(self.device, dtype=torch.float16)
                for key, value in image_inputs.items()
            }
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            image_emb = self.model.get_image_features(**image_inputs)
            text_emb = self.model.get_text_features(**text_inputs)
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            scores.append((self.model.logit_scale.exp() * (text_emb * image_emb).sum(-1)).float())
        return torch.cat(scores).detach()


class DifferentiableRawPickScore(torch.nn.Module):
    """Raw PickScore with gradients preserved from scores to float images.

    Model parameters and cached text features stay frozen.  Scores use the
    same unnormalized PickScore scale as :class:`RawPickScore` (roughly
    20--30 rather than division by 26).
    """

    processor_id = RawPickScore.processor_id
    model_id = RawPickScore.model_id

    def __init__(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(
            self.processor_id, local_files_only=local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        features = self.model.get_text_features(**inputs)
        return features / features.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(1e-12)

    def score_with_text_features(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(images) != len(text_features):
            raise ValueError("images and text_features must have equal batch size")
        pixel_values = differentiable_clip_preprocess(images).to(
            device=self.device, dtype=self.dtype
        )
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        paired_similarity = (
            text_features.to(image_features) * image_features
        ).sum(-1)
        return self.model.logit_scale.exp() * paired_similarity

    def forward(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_with_text_features(images, self.encode_text(prompts))


def _as_pil(images: torch.Tensor) -> list[Image.Image]:
    uint8 = (images.detach().float().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    return [Image.fromarray(image.permute(1, 2, 0).numpy()) for image in uint8]
