from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from .attacks import pgd_embedding_attack
from .config import ExperimentConfig
from .defense import defended_logits, encode_prompt_ensemble, preprocess_image
from .model import CLIPWrapper, build_prompt_ensemble
from .utils import ensure_dirs, get_device, set_seed


def _to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
    to_pil = transforms.ToPILImage()
    return [to_pil(img.cpu()) for img in images]


def load_cifar10(root: Path):
    transform = transforms.ToTensor()
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return ds


def run_eval(config: ExperimentConfig) -> dict:
    set_seed(config.seed)
    device = get_device()
    ensure_dirs(config.output_dir / "metrics")

    model = CLIPWrapper(config.model_name, device)
    ds = load_cifar10(config.data_root)
    class_names = ds.classes

    class_prompts = build_prompt_ensemble(class_names, config.defense.prompt_templates)
    class_embeds = encode_prompt_ensemble(model, class_prompts)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    total = 0
    clean_correct = 0
    attack_correct = 0
    defended_correct = 0
    flagged = 0

    for images, labels in tqdm(loader, desc="Evaluating"):
        if total >= config.max_eval_samples:
            break

        images = images.to(device)
        labels = labels.to(device)
        batch_n = labels.shape[0]

        pil_images = _to_pil_batch(images)
        clean_preprocessed = [preprocess_image(img, config.defense.jpeg_quality) for img in pil_images]

        clean_embeds = model.encode_image_pil(clean_preprocessed)
        clean_logits = model.similarity_logits(clean_embeds, class_embeds)
        clean_pred = clean_logits.argmax(dim=-1)
        clean_correct += (clean_pred == labels).sum().item()

        adv_images = pgd_embedding_attack(
            model=model,
            pixel_values=images,
            class_text_embeds=class_embeds,
            labels=labels,
            epsilon=config.attack.epsilon,
            alpha=config.attack.alpha,
            steps=config.attack.steps,
            random_start=config.attack.random_start,
        )

        adv_pil = _to_pil_batch(adv_images)
        adv_processed = [preprocess_image(img, config.defense.jpeg_quality) for img in adv_pil]
        adv_embeds = model.encode_image_pil(adv_processed)

        attack_logits = model.similarity_logits(adv_embeds, class_embeds)
        attack_pred = attack_logits.argmax(dim=-1)
        attack_correct += (attack_pred == labels).sum().item()

        out = defended_logits(model, adv_embeds, class_embeds, config.defense)
        defended_pred = out.logits.argmax(dim=-1)
        defended_correct += (defended_pred == labels).sum().item()
        flagged += out.suspicious_mask.sum().item()

        total += batch_n

    total = max(total, 1)
    metrics = {
        "samples": total,
        "device": str(device),
        "clean_accuracy": clean_correct / total,
        "attacked_accuracy": attack_correct / total,
        "defended_accuracy": defended_correct / total,
        "suspicious_rate": flagged / total,
    }

    out_file = config.output_dir / "metrics" / "evaluation.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and defended CLIP.")
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(max_eval_samples=args.max_eval_samples, batch_size=args.batch_size)
    cfg.attack.epsilon = args.epsilon
    cfg.attack.steps = args.steps
    metrics = run_eval(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
