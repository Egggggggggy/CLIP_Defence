# CLIP Defense (Windows 11, Local, Defensive Research)

This repository provides a complete, reproducible, **defensive-only** CLIP robustness workflow for local Windows 11 use with VS Code.

## 1. Conceptual explanation

### How CLIP works
CLIP uses two encoders: an image encoder (ViT-B/32) and a text encoder. Each encoder maps its input into the same embedding space.

### How embeddings, similarity, and logits are computed
- Image/text embeddings are L2-normalized vectors.
- Similarity is cosine similarity via dot product of normalized vectors.
- Logits are scaled similarities: `logits = exp(logit_scale) * image_embeds @ text_embeds.T`.

### How adversarial examples fool CLIP
Small pixel perturbations can push image embeddings toward wrong text directions, changing retrieval ranking and zero-shot predictions while looking visually unchanged.

### Chosen defense: Hybrid CLIP Shield
This implementation combines:
1. **Input purification** (JPEG recompression + median filtering)
2. **Robust prompt ensembling** (multiple templates per class)
3. **Embedding-space randomized smoothing**
4. **Consistency-based suspicious-input detection**

Why suitable for CLIP:
- CLIP is strongly embedding-driven, so embedding-space smoothing and prompt ensembling directly stabilize similarity outcomes.
- Purification helps against high-frequency perturbations with low implementation complexity.
- Detection adds a reject option for uncertain/adaptive attacks.

### Threat model
- **Attacker capabilities:** white-box/strong gray-box adversary crafting bounded perturbations (PGD-style) to reduce correct class similarity.
- **Attacker limitations:** perturbation budget bounded by L∞ epsilon; no physical-world guarantees assumed.
- **Defense assumptions:** attacker does not perfectly optimize through full stochastic defense stack at deployment time.
- **Protects against:** common digital perturbation attacks that exploit embedding sensitivity.
- **Does not protect against:** unrestricted perturbations, patch/backdoor attacks, severe distribution shift, full adaptive EOT attackers.
- **Adaptive attacks expected:** EOT through purification/smoothing, prompt-targeted optimization, detector evasion.

### Trade-offs
- **Accuracy:** usually slight clean drop possible.
- **Robustness:** improved under bounded perturbations.
- **Latency:** higher due to smoothing samples.
- **Memory:** modest increase for batched noisy embeddings.
- **Complexity:** moderate; still practical for local experimentation.

### Defended pipeline (textual diagram)
`input image -> JPEG+median preprocess -> CLIP image embedding -> noisy embedding ensemble -> averaged logits with prompt-ensemble text embeddings -> prediction + suspicious-score gate`

## 2. Software, tools, and hardware requirements

- OS: Windows 11 (native, no WSL2)
- Python: 3.11
- PyTorch: 2.6.0
- Transformers: 5.10.0
- CUDA: 12.4 compatible build (if GPU used)
- GPU recommendation: NVIDIA GPU with >= 8GB VRAM (4GB minimum for smaller batch)
- CPU fallback: supported, slower
- Dataset: CIFAR-10 auto-download via torchvision (or adapt loader for ImageNet subset)
- Approx. disk: 5-12 GB (env + model + dataset + outputs)
- Typical runtime:
  - CPU (500 samples): ~20-60 min
  - Mid-range GPU (500 samples): ~5-20 min

### Required Python packages
See `/home/runner/work/CLIP_Defence/CLIP_Defence/requirements.txt`.

### Optional
- Conda env via `/home/runner/work/CLIP_Defence/CLIP_Defence/environment.yml`
- Jupyter for notebook experimentation

### VS Code extensions (recommended)
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)
- GitLens (eamodio.gitlens)

### Verify PyTorch and CUDA
Run in VS Code terminal:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 3. Recommended project structure

```text
clip-defense/
├── README.md
├── requirements.txt
├── .venv/
├── data/
├── notebooks/
│   └── 01_test_clip.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── defense.py
│   ├── attacks.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── config.py
├── scripts/
│   └── run_experiments.bat
├── results/
│   ├── metrics/
│   ├── plots/
│   └── checkpoints/
└── tests/
```

## Quick start (Windows 11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.evaluate --max-eval-samples 500 --batch-size 16 --steps 10
```

Metrics are saved to:
- `/home/runner/work/CLIP_Defence/CLIP_Defence/results/metrics/evaluation.json`

## Essential clarification notes
If you want ImageNet-subset experiments, certified robustness methods, or full adversarial fine-tuning of CLIP, share your preferred dataset split and compute budget, and this scaffold can be extended.
