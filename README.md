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
  --output-dir output/mteb_results
```

Runs retrieval + STS tasks from `MTEB(eng, v2)` and `MTEB(multilingual, v2)` (configurable in `configs/distill_8xa100.yaml`). Prefix routing: `Query:` / `Document:` for retrieval, `Document:` for STS.

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
