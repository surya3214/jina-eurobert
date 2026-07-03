from __future__ import annotations

from typing import Any

import torch
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

from jina_eurobert.device import model_device, normalize_dataset_name


class DistillationTrainer(SentenceTransformerTrainer):
    """Trainer that routes multi-dataset batches to the combined distillation loss."""

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
        dataset_name = normalize_dataset_name(inputs.pop("dataset_name", None))

        batch_type = dataset_name
        if hasattr(self.loss, "set_batch_type"):
            self.loss.set_batch_type(batch_type)

        if batch_type == "distill" and "teacher_anchor" in inputs and "teacher_positive" in inputs:
            device = model_device(model)
            anchor = inputs.pop("teacher_anchor").to(device)
            positive = inputs.pop("teacher_positive").to(device)
            inputs["label"] = torch.stack([anchor, positive], dim=1)
        else:
            inputs.pop("teacher_anchor", None)
            inputs.pop("teacher_positive", None)

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
