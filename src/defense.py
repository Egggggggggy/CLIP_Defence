from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageFilter

from .config import DefenseConfig

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - allows image-only helper tests
    torch = None
    F = None

if torch is None:  # pragma: no cover
    def _no_grad():
        def _decorator(func):
            return func
        return _decorator
else:
    _no_grad = torch.no_grad


@dataclass
class DefenseOutput:
    logits: torch.Tensor
    suspicious_mask: torch.Tensor
    consistency_score: torch.Tensor


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def preprocess_image(image: Image.Image, quality: int) -> Image.Image:
    compressed = jpeg_compress(image, quality=quality)
    return compressed.filter(ImageFilter.MedianFilter(size=3))


@_no_grad()
def encode_prompt_ensemble(model, class_prompt_ensemble: list[list[str]]) -> torch.Tensor:
    class_vectors = []
    for prompts in class_prompt_ensemble:
        embeds = model.encode_text(prompts)
        class_vectors.append(F.normalize(embeds.mean(dim=0, keepdim=True), p=2, dim=-1))
    return torch.cat(class_vectors, dim=0)


@_no_grad()
def defended_logits(
    model,
    image_embeds: torch.Tensor,
    class_text_embeds: torch.Tensor,
    config: DefenseConfig,
) -> DefenseOutput:
    if torch is None:
        raise ImportError("PyTorch is required for defended_logits.")
    batch_size, dim = image_embeds.shape
    samples = []
    for _ in range(config.smoothing_samples):
        noise = torch.randn(batch_size, dim, device=image_embeds.device) * config.smoothing_sigma
        noisy = F.normalize(image_embeds + noise, p=2, dim=-1)
        samples.append(model.similarity_logits(noisy, class_text_embeds))

    stacked = torch.stack(samples, dim=0)
    logits = stacked.mean(dim=0)

    clean_logits = model.similarity_logits(image_embeds, class_text_embeds)
    consistency_score = (logits - clean_logits).abs().mean(dim=-1)
    suspicious = consistency_score > config.detector_cosine_threshold
    return DefenseOutput(logits=logits, suspicious_mask=suspicious, consistency_score=consistency_score)
