from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.enable_grad()
def pgd_embedding_attack(
    model,
    pixel_values: torch.Tensor,
    class_text_embeds: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    x = pixel_values.detach()
    x_adv = x.clone()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        image_embeds = model.image_features_from_tensor(x_adv)
        logits = model.similarity_logits(image_embeds, class_text_embeds)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()
