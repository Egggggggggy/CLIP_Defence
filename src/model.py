from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - allows lightweight tests without torch installed
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
class ClipOutputs:
    image_embeds: torch.Tensor
    text_embeds: torch.Tensor
    logits_per_image: torch.Tensor


class CLIPWrapper:
    def __init__(self, model_name: str, device: torch.device) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to initialize CLIPWrapper.")
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @_no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.model.get_text_features(**inputs)
        return F.normalize(text_embeds, p=2, dim=-1)

    @_no_grad()
    def encode_image_pil(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_embeds = self.model.get_image_features(**inputs)
        return F.normalize(image_embeds, p=2, dim=-1)

    def prepare_image_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.to(self.device)

    def image_features_from_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(image_embeds, p=2, dim=-1)

    def similarity_logits(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_embeds @ text_embeds.T

    def forward(self, images: list[Image.Image], prompts: list[str]) -> ClipOutputs:
        image_embeds = self.encode_image_pil(images)
        text_embeds = self.encode_text(prompts)
        logits_per_image = self.similarity_logits(image_embeds, text_embeds)
        return ClipOutputs(image_embeds=image_embeds, text_embeds=text_embeds, logits_per_image=logits_per_image)


def build_prompt_ensemble(class_names: list[str], templates: list[str]) -> list[list[str]]:
    return [[template.format(name) for template in templates] for name in class_names]
