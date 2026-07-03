from __future__ import annotations

import torch
import torch.nn as nn


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
