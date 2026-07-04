from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jina_eurobert.device import model_device
from sentence_transformers.sentence_transformer.losses import (
    CosineSimilarityLoss,
    GlobalOrthogonalRegularizationLoss,
    MultipleNegativesRankingLoss,
)


MODEL_FEATURE_KEYS = frozenset({"input_ids", "attention_mask", "token_type_ids"})


def _model_feature_inputs(features: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in features.items() if key in MODEL_FEATURE_KEYS and torch.is_tensor(value)}


def _feature_labels(features: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    if key not in features:
        return None
    value = features[key]
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=features["sentence_embedding"].device)
    return value


def _loss_fp32_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _normalize_sts_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    if labels.numel() == 0:
        return labels
    if labels.dim() != 1:
        raise ValueError(f"STS labels must be a 1D score tensor, got shape {tuple(labels.shape)}")
    if float(labels.max()) > 1.0:
        labels = labels / 5.0
    return labels.clamp(0.0, 1.0)


def _embeddings_from_features(model: nn.Module, sentence_features: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
    device = model_device(model)
    embeddings: list[torch.Tensor] = []
    with _loss_fp32_context(device):
        for sentence_feature in sentence_features:
            model_inputs = _model_feature_inputs(sentence_feature)
            if "sentence_embedding" in sentence_feature and not model_inputs:
                embedding = sentence_feature["sentence_embedding"].float()
            else:
                embedding = model(model_inputs)["sentence_embedding"].float()
            if not torch.isfinite(embedding).all():
                raise FloatingPointError("Model produced non-finite sentence embeddings for the current batch.")
            embeddings.append(embedding)
    return embeddings


def _ensure_finite(loss: torch.Tensor, term_name: str) -> torch.Tensor:
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite {term_name} loss.")
    return loss


class MRLEmbedDistillLoss(nn.Module):
    """Distill student embeddings to precomputed teacher MRL targets at multiple dims."""

    def __init__(
        self,
        model: nn.Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        if isinstance(matryoshka_dims, int):
            matryoshka_dims = [matryoshka_dims]
        self.matryoshka_dims = [int(dim) for dim in matryoshka_dims]
        if matryoshka_weights is None:
            matryoshka_weights = [1.0] * len(matryoshka_dims)
        weight_sum = sum(matryoshka_weights)
        self.matryoshka_weights = [weight / weight_sum for weight in matryoshka_weights]

    def forward(self, sentence_features: list[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:
        if labels is None or not torch.is_tensor(labels):
            raise ValueError("MRL distillation requires teacher embedding labels.")

        if isinstance(sentence_features, dict):
            sentence_features = [sentence_features]
        elif not isinstance(sentence_features, (list, tuple)):
            raise TypeError(
                f"Expected sentence_features to be a list of feature dicts, got {type(sentence_features).__name__}"
            )

        embeddings = _embeddings_from_features(self.model, list(sentence_features))
        labels = labels.float()

        losses: list[torch.Tensor] = []
        for col_idx, student in enumerate(embeddings):
            if labels.dim() == 3:
                teacher = labels[:, col_idx, :].to(student.device)
            elif labels.dim() == 2 and len(embeddings) == 1:
                teacher = labels.to(student.device)
            else:
                teacher = labels[:, col_idx, :].to(student.device)

            for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights, strict=True):
                student_d = F.normalize(student[..., :dim], p=2, dim=-1)
                teacher_d = teacher[..., :dim]
                teacher_norm = teacher_d.norm(p=2, dim=-1, keepdim=True)
                valid = teacher_norm.squeeze(-1) > 1e-6
                if not valid.any():
                    continue
                teacher_d = teacher_d / teacher_norm.clamp(min=1e-12)
                cosine = F.cosine_similarity(student_d, teacher_d, dim=-1)
                cosine = cosine[valid]
                losses.append(weight * (1.0 - cosine).mean())
        if not losses:
            raise ValueError(
                "No valid teacher embeddings in distill batch (all zero vectors). "
                "Run precompute_teacher_mrl.py before training."
            )
        return torch.stack(losses).sum()


def _scalar_loss(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return sum(output.values())
    return output


class CombinedDistillationLoss(nn.Module):
    """Multi-objective loss: MRL distill + InfoNCE + STS cosine + GOR with batch routing."""

    def __init__(
        self,
        model: nn.Module,
        matryoshka_dims: list[int],
        loss_weights: dict[str, float],
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_weights = loss_weights
        self.distill_loss = MRLEmbedDistillLoss(model, matryoshka_dims)
        self.infonce_loss = MultipleNegativesRankingLoss(model, scale=20.0)
        self.sts_loss = CosineSimilarityLoss(model)
        self.gor_loss = GlobalOrthogonalRegularizationLoss(model, mean_weight=0.0)
        self._batch_type = "distill"
        self.last_batch_type: str | None = None
        self.last_loss_terms: dict[str, float] = {}

    def set_batch_type(self, batch_type: str) -> None:
        self._batch_type = batch_type

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss_weights": self.loss_weights,
            "matryoshka_dims": self.distill_loss.matryoshka_dims,
        }

    def forward(self, sentence_features: list[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:
        batch_type = self._batch_type
        self.last_batch_type = batch_type
        self.last_loss_terms = {}
        if isinstance(sentence_features, dict):
            sentence_features = [sentence_features]
        elif not isinstance(sentence_features, (list, tuple)):
            raise TypeError(
                f"Expected sentence_features to be a list of feature dicts, got {type(sentence_features).__name__}"
            )
        sentence_features = [_model_feature_inputs(feature) for feature in sentence_features]
        terms: list[torch.Tensor] = []

        if batch_type == "distill":
            if (
                labels is not None
                and torch.is_tensor(labels)
                and labels.dim() == 3
                and labels.shape[1] == len(sentence_features)
            ):
                distill_term = self.loss_weights["distill_mrl"] * self.distill_loss(sentence_features, labels)
                terms.append(_ensure_finite(distill_term, "distill_mrl"))
                self.last_loss_terms["distill_mrl"] = float(distill_term.detach())
            else:
                raise ValueError(
                    "Distill batch is missing 3D teacher labels. "
                    "Run precompute_teacher_mrl.py and ensure teacher embeddings match training texts."
                )
        elif batch_type == "retrieval":
            embeddings = _embeddings_from_features(self.model, sentence_features)
            infonce_term = self.loss_weights["infonce"] * self.infonce_loss.compute_loss_from_embeddings(
                embeddings, labels
            )
            terms.append(_ensure_finite(infonce_term, "infonce"))
            self.last_loss_terms["infonce"] = float(infonce_term.detach())

            gor_output = self.gor_loss.compute_loss_from_embeddings(embeddings, labels)
            gor_term = self.loss_weights["gor"] * _scalar_loss(gor_output)
            terms.append(_ensure_finite(gor_term, "gor"))
            self.last_loss_terms["gor"] = float(gor_term.detach())
        elif batch_type == "sts":
            if labels is None or not torch.is_tensor(labels):
                raise ValueError("STS batch is missing score labels.")
            sts_labels = _normalize_sts_labels(labels)
            embeddings = _embeddings_from_features(self.model, sentence_features)
            sts_term = self.loss_weights["cosent"] * self.sts_loss.compute_loss_from_embeddings(
                embeddings, sts_labels
            )
            terms.append(_ensure_finite(sts_term, "sts"))
            self.last_loss_terms["sts"] = float(sts_term.detach())
        else:
            raise ValueError(f"Unknown batch_type: {batch_type}")

        if not terms:
            raise RuntimeError(f"No loss terms computed for batch_type={batch_type!r}")
        total = torch.stack(terms).sum()
        self.last_loss_terms["total"] = float(total.detach())
        return total


class MixedBatchRouter:
    """Sample batch types according to configured routing probabilities."""

    def __init__(self, routing: dict[str, float], seed: int = 42) -> None:
        self.routes = list(routing.keys())
        weights = [routing[key] for key in self.routes]
        self.rng = random.Random(seed)
        self.weights = weights

    def sample(self) -> str:
        return self.rng.choices(self.routes, weights=self.weights, k=1)[0]
