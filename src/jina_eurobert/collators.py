from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from jina_eurobert.device import infer_dataset_name, normalize_dataset_name
from sentence_transformers.base.data_collator import BaseDataCollator

TEACHER_COLUMNS = frozenset({"teacher_anchor", "teacher_positive"})


@dataclass
class DistillationDataCollator(BaseDataCollator):
    """Collator that strips teacher metadata and routes columns by dataset name."""

    prompts: str | dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}

        dataset_names = [infer_dataset_name(row) for row in features]
        batch_dataset = dataset_names[0]
        if len(set(dataset_names)) != 1:
            raise ValueError(f"Mixed dataset types in one batch: {sorted(set(dataset_names))}")

        extras: dict[str, Any] = {}
        if batch_dataset == "distill" and all("teacher_anchor" in row for row in features):
            extras["teacher_anchor"] = torch.tensor(
                [row["teacher_anchor"] for row in features],
                dtype=torch.float32,
            )
            extras["teacher_positive"] = torch.tensor(
                [row["teacher_positive"] for row in features],
                dtype=torch.float32,
            )

        token_rows: list[dict[str, Any]] = []
        for row, dataset_name in zip(features, dataset_names, strict=True):
            token_row: dict[str, Any] = {
                "dataset_name": normalize_dataset_name(dataset_name),
                "anchor": row["anchor"],
                "positive": row["positive"],
            }
            if dataset_name == "retrieval" and row.get("negative"):
                token_row["negative"] = row["negative"]
            if dataset_name == "sts":
                token_row["score"] = row.get("score", row.get("label", 0.0))
            token_rows.append(token_row)

        batch = super().__call__(token_rows)
        batch.update(extras)
        return batch
