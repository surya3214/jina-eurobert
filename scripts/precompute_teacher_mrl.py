#!/usr/bin/env python3
"""Distributed Qwen3 teacher MRL embedding precomputation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from tqdm import tqdm

from jina_eurobert.config import load_config, matryoshka_dims
from jina_eurobert.data import build_training_mixture, load_gooaq_pair_dataset, load_nq_pair_dataset, teacher_index_key, text_hash
from jina_eurobert.datasets_registry import manifest_path_for, read_manifest
from jina_eurobert.hf_datasets import resolve_datasets_dir
from jina_eurobert.models import build_teacher_model


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def unique_texts_with_prompts(datasets: list[Dataset]) -> list[tuple[str, str]]:
    """Return unique (text, prompt_type) pairs for query anchors and document positives."""
    pairs: set[tuple[str, str]] = set()
    for dataset in datasets:
        for row in dataset:
            pairs.add((row["anchor"], "query"))
            pairs.add((row["positive"], "document"))
    return sorted(pairs)


def write_smoke_test_embeddings(output_dir: Path, dims: list[int]) -> None:
    rows = []
    for text, prompt_type in [
        ("What is Paris?", "query"),
        ("Paris is the capital of France.", "document"),
    ]:
        row = {"text": text, "text_hash": text_hash(text), "prompt_type": prompt_type}
        for dim in dims:
            vector = np.random.randn(dim).astype(np.float32)
            vector /= np.linalg.norm(vector) + 1e-8
            row[f"embedding_{dim}"] = vector.astype(np.float16).tolist()
        rows.append(row)
    Dataset.from_list(rows).to_parquet(str(output_dir / "teacher_embeddings_rank0.parquet"))


def encode_texts_mrl(
    model,
    items: list[tuple[str, str]],
    dims: list[int],
    batch_size: int,
    rank: int,
    world_size: int,
) -> Dataset:
    shard = items[rank::world_size]
    rows: list[dict] = []

    index = 0
    progress = tqdm(total=len(shard), desc=f"rank {rank}", disable=rank != 0)
    while index < len(shard):
        prompt_name = shard[index][1]
        batch_items: list[tuple[str, str]] = []
        while index < len(shard) and shard[index][1] == prompt_name and len(batch_items) < batch_size:
            batch_items.append(shard[index])
            index += 1

        batch_texts = [text for text, _ in batch_items]
        batch_embeddings: dict[int, list] = {}
        for dim in dims:
            embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                prompt_name=prompt_name,
                truncate_dim=dim,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            batch_embeddings[dim] = [emb.astype("float16").tolist() for emb in embeddings]

        for item_idx, (text, item_prompt) in enumerate(batch_items):
            rows.append(
                {
                    "text": text,
                    "text_hash": text_hash(text),
                    "prompt_type": item_prompt,
                    **{f"embedding_{dim}": batch_embeddings[dim][item_idx] for dim in dims},
                }
            )
        progress.update(len(batch_items))
    progress.close()

    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--datasets-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir or config["data"]["teacher_embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = matryoshka_dims(config)

    datasets_dir = resolve_datasets_dir(args.datasets_dir, config)
    if datasets_dir and rank == 0:
        manifest = read_manifest(manifest_path_for(datasets_dir))
        print(f"Using local datasets from {datasets_dir} ({len(manifest)} repos in manifest)")

    if args.smoke_test:
        if rank == 0:
            write_smoke_test_embeddings(output_dir, dims)
            print(f"Wrote smoke-test teacher embeddings to {output_dir}")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    teacher_cfg = config["teacher"]
    pair_sets = []
    try:
        pair_sets.append(
            load_gooaq_pair_dataset(max_samples=args.max_samples, datasets_dir=datasets_dir, config=config)
        )
        pair_sets.append(
            load_nq_pair_dataset(max_samples=args.max_samples, datasets_dir=datasets_dir, config=config)
        )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            print(f"Warning: could not load HF pair datasets ({exc}); using smoke mixture.")
        distill = build_training_mixture(config, smoke_test=True)["distill"]
        pair_sets.append(distill)

    items = unique_texts_with_prompts(pair_sets)
    teacher = build_teacher_model(
        model_name=teacher_cfg["model"],
        max_seq_length=teacher_cfg["max_seq_length"],
        device=device,
    )
    teacher.eval()

    dataset = encode_texts_mrl(
        teacher,
        items,
        dims=dims,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )

    shard_path = output_dir / f"teacher_embeddings_rank{rank}.parquet"
    dataset.to_parquet(str(shard_path))
    if rank == 0:
        print(f"Wrote {len(dataset)} embeddings to {shard_path} (rank 0/{world_size})")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
