# jina-eurobert

Distill [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) into base [EuroBERT/EuroBERT-210m](https://huggingface.co/EuroBERT/EuroBERT-210m) without LoRA adapters, following the [Jina v5 distillation recipe](https://arxiv.org/abs/2602.15547).

## Features

- Teacher MRL targets at dims `32, 64, 128, 256, 512, 768`
- Multi-objective training: Matryoshka distill + InfoNCE + CoSENT + GOR
- Fixed **512-token** max sequence length for train, eval, and teacher precompute
- 8×A100 DDP via `torchrun`

## Setup

```bash
pip install -e .
```

## Offline datasets

Training and MTEB evaluation can load Hugging Face datasets from a local directory instead of the Hub API at runtime.

```bash
# 1. Download all datasets referenced by the default config
python scripts/download_datasets.py \
  --output-dir data/hf_datasets \
  --config configs/distill_8xa100.yaml

# Training only (smaller/faster)
python scripts/download_datasets.py --output-dir data/hf_datasets --training-only

# MTEB only
python scripts/download_datasets.py --output-dir data/hf_datasets --mteb-only

# List repos without downloading
python scripts/download_datasets.py --output-dir data/hf_datasets --config configs/distill_8xa100.yaml --dry-run
```

Point training/eval at the local cache via config (`data.datasets_dir`), CLI (`--datasets-dir`), or env:

```bash
export JINA_EUROBERT_DATASETS_DIR=data/hf_datasets
```

### Check training mixture size

Training also prints per-split row counts at startup. To inspect sizes without launching training:

```bash
PYTHONPATH=src:scripts python3 -c "
from jina_eurobert.config import load_config
from jina_eurobert.data import build_training_mixture, load_teacher_embedding_index, summarize_training_mixture
config = load_config()
idx = load_teacher_embedding_index(config['data']['teacher_embeddings_dir'])
mix = build_training_mixture(config, teacher_index=idx, max_samples_per_source=5000)
print(summarize_training_mixture(mix))
print('teacher_index=', len(idx))
"
```

Example output:

```
{'distill': 15000, 'retrieval': 10000, 'sts': 5749}
teacher_index= 25000
```

If counts are very small (e.g. 8 per split), datasets were not found locally — run `download_datasets.py` first.

## 1. Precompute teacher MRL embeddings

```bash
torchrun --nproc_per_node=8 scripts/precompute_teacher_mrl.py \
  --config configs/distill_8xa100.yaml \
  --output-dir data/teacher_embeddings \
  --batch-size 64
```

## 2. Train distilled EuroBERT

```bash
torchrun --nproc_per_node=8 scripts/train_distill.py \
  --config configs/distill_8xa100.yaml \
  --output-dir output/distilled-eurobert
```

Smoke test (CPU/single GPU):

```bash
PYTHONPATH=src:scripts python scripts/train_distill.py --smoke-test --max-samples 32
```

## 3. Evaluate on MTEB

```bash
PYTHONPATH=src:scripts python scripts/eval_mteb.py \
  --model-path output/distilled-eurobert/final \
  --output-dir output/mteb_results \
  --datasets-dir data/hf_datasets
```

Runs retrieval + STS tasks from `MTEB(eng, v2)` and `MTEB(Multilingual, v2)` (configurable in `configs/distill_8xa100.yaml`). Prefix routing: `Query:` / `Document:` for retrieval, `Document:` for STS.

Smoke test (single task):

```bash
PYTHONPATH=src:scripts python scripts/eval_mteb.py \
  --model-path output/distilled-eurobert/final \
  --smoke-test
```

## Layout

```
configs/distill_8xa100.yaml   # hyperparameters
src/jina_eurobert/            # library code
scripts/                      # CLI entrypoints
```

## Inference prefixes

| Task | Prefix |
|------|--------|
| Retrieval queries | `Query: ` |
| Retrieval documents | `Document: ` |
| STS / text-matching | `Document: ` on both sides |

## License

Training uses Apache-2.0 models (EuroBERT, Qwen3-Embedding-4B).
