from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AttackConfig:
    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 10
    random_start: bool = True


@dataclass
class DefenseConfig:
    jpeg_quality: int = 70
    smoothing_samples: int = 8
    smoothing_sigma: float = 0.05
    detector_cosine_threshold: float = 0.2
    prompt_templates: list[str] = field(
        default_factory=lambda: [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a close-up photo of a {}.",
            "a clean photo of a {}.",
            "a low-resolution photo of a {}.",
        ]
    )


@dataclass
class ExperimentConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    dataset_name: str = "cifar10"
    data_root: Path = Path("data")
    output_dir: Path = Path("results")
    batch_size: int = 32
    num_workers: int = 2
    max_eval_samples: int = 1000
    seed: int = 42
    use_amp: bool = True
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
