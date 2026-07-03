from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


def model_device(model: nn.Module) -> torch.device:
    """Resolve device for plain, DataParallel, or DDP-wrapped modules."""
    if hasattr(model, "device"):
        device = model.device
        if isinstance(device, torch.device):
            return device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def normalize_dataset_name(dataset_name: Any) -> str:
    if dataset_name is None:
        return "distill"
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0] if dataset_name else "distill"
    if torch.is_tensor(dataset_name):
        dataset_name = dataset_name.reshape(-1)[0].item()
    return str(dataset_name)
