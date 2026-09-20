from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .evaluate import run_eval
from .utils import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight defense calibration/eval loop.")
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--smoothing-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(max_eval_samples=args.max_eval_samples, batch_size=args.batch_size)
    cfg.defense.jpeg_quality = args.jpeg_quality
    cfg.defense.smoothing_samples = args.smoothing_samples

    ensure_dirs(cfg.output_dir / "checkpoints", cfg.output_dir / "metrics")
    metrics = run_eval(cfg)

    calib_file = cfg.output_dir / "checkpoints" / "defense_calibration.json"
    calib_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved calibration to {calib_file}")


if __name__ == "__main__":
    main()
