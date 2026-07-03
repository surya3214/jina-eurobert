from __future__ import annotations

import torch
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

from jina_eurobert.device import model_device


class DistillationTrainer(SentenceTransformerTrainer):
    """Trainer that routes multi-dataset batches to the combined distillation loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dataset_name = inputs.pop("dataset_name", None)
        if isinstance(dataset_name, list):
            dataset_name = dataset_name[0]

        batch_type = str(dataset_name) if dataset_name else "distill"
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
