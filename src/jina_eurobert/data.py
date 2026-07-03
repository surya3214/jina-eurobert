from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def teacher_index_key(text: str, prompt_type: str = "document") -> str:
    return f"{text_hash(text)}:{prompt_type}"


def load_teacher_embedding_index(embeddings_dir: str | Path) -> dict[str, np.ndarray]:
    """Load parquet shards into a (text-hash, prompt) -> embedding index (prefers dim 768)."""
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        return {}

    index: dict[str, np.ndarray] = {}
    for parquet_path in sorted(embeddings_dir.glob("**/*.parquet")):
        dataset = Dataset.from_parquet(str(parquet_path))
        for row in dataset:
            text = row.get("text") or row.get("anchor") or row.get("sentence")
            if text is None:
                continue
            prompt_type = row.get("prompt_type", "document")
            key = teacher_index_key(text, prompt_type)
            if key in index:
                continue
            for dim_key in ("embedding_768", "embedding_512", "embedding"):
                if dim_key in row:
                    index[key] = np.asarray(row[dim_key], dtype=np.float32)
                    break
            else:
                for col, value in row.items():
                    if col.startswith("embedding_"):
                        index[key] = np.asarray(value, dtype=np.float32)
                        break
    return index


def _zero_teacher(dim: int) -> list[float]:
    return [0.0] * dim


def _normalize_pair_dataset(dataset: Dataset, anchor_col: str, positive_col: str) -> Dataset:
    return dataset.rename_columns({anchor_col: "anchor", positive_col: "positive"}).select_columns(
        ["anchor", "positive"]
    )


def load_gooaq_pair_dataset(split: str = "train", max_samples: int | None = None) -> Dataset:
    dataset = load_dataset("sentence-transformers/gooaq", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if "question" in dataset.column_names and "answer" in dataset.column_names:
        return _normalize_pair_dataset(dataset, "question", "answer")
    return _normalize_pair_dataset(dataset, "anchor", "positive")


def load_nq_pair_dataset(split: str = "train", max_samples: int | None = None) -> Dataset:
    dataset = load_dataset("sentence-transformers/natural-questions", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    anchor_col = "query" if "query" in dataset.column_names else "question"
    positive_col = "answer" if "answer" in dataset.column_names else "positive"
    return _normalize_pair_dataset(dataset, anchor_col, positive_col)


def load_stsb_dataset(split: str = "train", max_samples: int | None = None) -> Dataset:
    dataset = load_dataset("sentence-transformers/stsb", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    mapped = dataset.rename_columns({"sentence1": "anchor", "sentence2": "positive", "score": "score"})
    return mapped.select_columns(["anchor", "positive", "score"])


def load_msmarco_triplet_dataset(max_samples: int | None = None) -> Dataset:
    dataset = load_dataset("sentence-transformers/msmarco-hard-negatives", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if "query" in dataset.column_names:
        return dataset.rename_columns({"query": "anchor", "positive": "positive", "negative": "negative"})
    return dataset


def attach_teacher_embeddings(
    dataset: Dataset,
    text_columns: list[str],
    teacher_index: dict[str, np.ndarray],
    dim: int = 768,
) -> Dataset:
    """Add teacher_anchor and teacher_positive embedding columns."""

    def _map_row(row: dict[str, Any]) -> dict[str, Any]:
        prompt_types = ["query", "document"]
        for target_col, source_col, prompt_type in zip(
            ["teacher_anchor", "teacher_positive"],
            text_columns,
            prompt_types,
            strict=True,
        ):
            text = row[source_col]
            embedding = teacher_index.get(teacher_index_key(text, prompt_type))
            if embedding is None:
                embedding = teacher_index.get(text_hash(text))
            if embedding is None:
                row[target_col] = [0.0] * dim
            else:
                vector = embedding[:dim].astype(np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                row[target_col] = vector.tolist()
        return row

    return dataset.map(_map_row)


def prepare_distill_dataset(
    dataset: Dataset,
    teacher_index: dict[str, np.ndarray],
    teacher_dim: int = 768,
) -> Dataset:
    if teacher_index:
        dataset = attach_teacher_embeddings(dataset, ["anchor", "positive"], teacher_index, teacher_dim)
    else:
        dataset = dataset.add_column("teacher_anchor", [_zero_teacher(teacher_dim)] * len(dataset))
        dataset = dataset.add_column("teacher_positive", [_zero_teacher(teacher_dim)] * len(dataset))
    return dataset.select_columns(["anchor", "positive", "teacher_anchor", "teacher_positive"])


def prepare_retrieval_dataset(dataset: Dataset) -> Dataset:
    return dataset.select_columns(["anchor", "positive", "negative"])


def prepare_sts_dataset(dataset: Dataset) -> Dataset:
    if "label" in dataset.column_names and "score" not in dataset.column_names:
        dataset = dataset.rename_column("label", "score")
    return dataset.select_columns(["anchor", "positive", "score"])


def _smoke_test_datasets(teacher_dim: int) -> DatasetDict:
    zero = _zero_teacher(teacher_dim)
    n = 8
    return DatasetDict(
        {
            "distill": Dataset.from_dict(
                {
                    "anchor": [f"What is city {i}?" for i in range(n)],
                    "positive": [f"City {i} is a European capital." for i in range(n)],
                    "teacher_anchor": [zero] * n,
                    "teacher_positive": [zero] * n,
                }
            ),
            "retrieval": Dataset.from_dict(
                {
                    "anchor": [f"topic {i}" for i in range(n)],
                    "positive": [f"relevant passage about topic {i}" for i in range(n)],
                    "negative": [f"unrelated text {i}" for i in range(n)],
                }
            ),
            "sts": Dataset.from_dict(
                {
                    "anchor": [f"sentence A variant {i}" for i in range(n)],
                    "positive": [f"sentence B variant {i}" for i in range(n)],
                    "score": [float(i % 5) for i in range(n)],
                }
            ),
        }
    )


def build_training_mixture(
    config: dict[str, Any],
    teacher_index: dict[str, np.ndarray] | None = None,
    max_samples_per_source: int | None = 5000,
    smoke_test: bool = False,
) -> DatasetDict:
    """Build per-task datasets for homogeneous batches (distill / retrieval / sts)."""
    teacher_index = teacher_index or {}
    dims = config.get("matryoshka_dims", [768])
    teacher_dim = max(dims)
    seed = config.get("training", {}).get("seed", 42)

    if smoke_test:
        return _smoke_test_datasets(teacher_dim)

    distill_sets: list[Dataset] = []
    for name in config.get("data", {}).get("pair_datasets", []):
        try:
            if "gooaq" in name:
                dataset = load_gooaq_pair_dataset(max_samples=max_samples_per_source)
            elif "natural-questions" in name:
                dataset = load_nq_pair_dataset(max_samples=max_samples_per_source)
            else:
                continue
            distill_sets.append(prepare_distill_dataset(dataset, teacher_index, teacher_dim))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping pair dataset {name}: {exc}")

    retrieval_sets: list[Dataset] = []
    for name in config.get("data", {}).get("triplet_datasets", []):
        try:
            if "msmarco" in name:
                triplets = load_msmarco_triplet_dataset(max_samples=max_samples_per_source)
                retrieval_sets.append(prepare_retrieval_dataset(triplets))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping triplet dataset {name}: {exc}")

    sts_sets: list[Dataset] = []
    for name in config.get("data", {}).get("sts_datasets", []):
        try:
            if "stsb" in name:
                stsb = load_stsb_dataset(max_samples=max_samples_per_source)
                sts_sets.append(prepare_sts_dataset(stsb))
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping STS dataset {name}: {exc}")

    datasets: dict[str, Dataset] = {}
    if distill_sets:
        datasets["distill"] = concatenate_datasets(distill_sets).shuffle(seed=seed)
    if retrieval_sets:
        datasets["retrieval"] = concatenate_datasets(retrieval_sets).shuffle(seed=seed)
    if sts_sets:
        datasets["sts"] = concatenate_datasets(sts_sets).shuffle(seed=seed)

    if not datasets:
        return _smoke_test_datasets(teacher_dim)

    return DatasetDict(datasets)


def save_dataset_manifest(dataset: Dataset | DatasetDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, DatasetDict):
        manifest = {name: {"num_rows": len(ds), "columns": ds.column_names} for name, ds in dataset.items()}
    else:
        manifest = {"num_rows": len(dataset), "columns": dataset.column_names}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
