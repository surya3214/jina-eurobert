from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from jina_eurobert.device import normalize_dataset_name
from sentence_transformers.base.data_collator import BaseDataCollator

TEACHER_COLUMNS = frozenset({"teacher_anchor", "teacher_positive"})


@dataclass
class DistillationDataCollator(BaseDataCollator):
    """Collator that strips teacher metadata and routes columns by dataset name."""

    prompts: str | dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}

        dataset_name = normalize_dataset_name(features[0].get("dataset_name", "distill"))
        extras: dict[str, Any] = {}

        if "teacher_anchor" in features[0]:
            extras["teacher_anchor"] = torch.tensor(
                [row["teacher_anchor"] for row in features],
                dtype=torch.float32,
            )
            extras["teacher_positive"] = torch.tensor(
                [row["teacher_positive"] for row in features],
                dtype=torch.float32,
            )

        token_rows: list[dict[str, Any]] = []
        for row in features:
            token_row: dict[str, Any] = {
                "dataset_name": normalize_dataset_name(row.get("dataset_name", dataset_name)),
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
