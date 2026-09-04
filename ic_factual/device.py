"""Device selection for laptops and CUDA clusters."""

from __future__ import annotations

import os

import torch

ENV_DEVICE = "IC_FACTUAL_DEVICE"


def resolve_device(explicit: str | None = None) -> str:
    """
    Pick the best available device.

    Priority: explicit argument > IC_FACTUAL_DEVICE > cuda > mps > cpu.
    cuda is selected when drivers are visible.
    """
    if explicit:
        return explicit

    env = os.environ.get(ENV_DEVICE)
    if env:
        return env

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


