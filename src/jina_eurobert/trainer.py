from __future__ import annotations

import math
import os
from typing import Any

import torch
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

from jina_eurobert.device import model_device, normalize_dataset_name


class DistillationTrainer(SentenceTransformerTrainer):
    """Trainer that routes multi-dataset batches to the combined distillation loss."""

    eurobert_base_model: str = "EuroBERT/EuroBERT-210m"

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        from jina_eurobert.models import bundle_eurobert_custom_code

        super()._save(output_dir=output_dir, state_dict=state_dict)
        if output_dir is not None:
            bundle_eurobert_custom_code(output_dir, source_model=self.eurobert_base_model)

    def prepare_loss(self, loss, model):
        if isinstance(loss, torch.nn.Module):
            loss = loss.to(model_device(model))
        else:
            loss = loss(model).to(model_device(model))

        if getattr(loss, "requires_media_counts", False):
            from sentence_transformers.base.modules.router import Router

            if Router in [module.__class__ for module in model.children()]:
                input_modules = [route[0] for route in model[0].sub_modules.values()]  # type: ignore[index]
            else:
                input_modules = [model[0]]
            for module in input_modules:
                if hasattr(module, "track_media_counts"):
                    module.track_media_counts = True

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch_type = normalize_dataset_name(inputs.pop("dataset_name", None))
        if hasattr(self.loss, "set_batch_type"):
            self.loss.set_batch_type(batch_type)

        teacher_anchor = inputs.pop("teacher_anchor", None)
        teacher_positive = inputs.pop("teacher_positive", None)

        if batch_type == "distill":
            if teacher_anchor is None or teacher_positive is None:
                raise ValueError(
                    "Distill batch is missing teacher embeddings. "
                    "Run precompute_teacher_mrl.py and ensure teacher embeddings match training texts."
                )
            device = model_device(model)
            inputs["label"] = torch.stack(
                [
                    teacher_anchor.to(device=device, dtype=torch.float32),
                    teacher_positive.to(device=device, dtype=torch.float32),
                ],
                dim=1,
            )
        elif batch_type == "retrieval":
            inputs.pop("label", None)
        elif batch_type == "sts" and "label" not in inputs:
            raise ValueError("STS batch is missing score labels.")

        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if (
            not return_outputs
            and hasattr(self.loss, "last_loss_terms")
            and self.loss.last_loss_terms
            and os.environ.get("RANK", "0") == "0"
        ):
            self._pending_loss_terms = dict(self.loss.last_loss_terms)
            self._pending_batch_type = batch_type
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if os.environ.get("RANK", "0") == "0":
            if self.state.max_steps and self.state.max_steps > 0:
                logs = dict(logs)
                logs["progress"] = round(self.state.global_step / self.state.max_steps, 4)
            pending_terms = getattr(self, "_pending_loss_terms", None)
            pending_batch = getattr(self, "_pending_batch_type", None)
            if pending_terms and "loss" in logs:
                logs = dict(logs)
                logs["batch_type"] = pending_batch or "unknown"
                for key, value in pending_terms.items():
                    if key != "total":
                        logs[f"loss/{key}"] = round(value, 4)
            if "loss" in logs and (math.isnan(logs["loss"]) or math.isinf(logs["loss"])):
                logs = dict(logs)
                logs["loss"] = 0.0
        if start_time is not None:
            return super().log(logs, start_time)
        return super().log(logs)

    def train(self, *args, **kwargs):
        if os.environ.get("RANK", "0") == "0" and self.train_dataset is not None:
            train_dataloader = self.get_train_dataloader()
            steps_per_epoch = len(train_dataloader) // max(self.args.gradient_accumulation_steps, 1)
            max_steps = self.args.max_steps
            print(
                "Training schedule: "
                f"{steps_per_epoch:,} optimizer steps per epoch, "
                f"max_steps={max_steps:,}. "
                "The logged `epoch` is fractional because training is step-based.",
                flush=True,
            )
        return super().train(*args, **kwargs)
