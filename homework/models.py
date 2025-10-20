from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path



NUM_CLASSES = 6
IMAGE_C = 3
IMAGE_H = 64
IMAGE_W = 64
FLAT_D = IMAGE_C * IMAGE_H * IMAGE_W


# ------------------------------
# Loss
# ------------------------------
class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss on logits (implements the negative log-likelihood of a softmax classifier).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, NUM_CLASSES) unnormalized class scores
            labels: (B,) integer class labels in [0, NUM_CLASSES-1]
        Returns:
            scalar loss tensor
        """
        return F.cross_entropy(logits, labels)


# ------------------------------
# Models
# ------------------------------
class LinearClassifier(nn.Module):
    """
    Single linear layer mapping flattened pixels -> class logits.
    Input:  (B, 3, 64, 64)
    Output: (B, 6)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_D, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    """
    One hidden-layer MLP: Flatten -> Linear(hidden) -> ReLU -> Linear(6)
    """
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_D, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifierDeep(nn.Module):
    """
    Deep MLP with >=4 Linear layers (no residuals).
    Structure: Flatten -> Linear -> [Linear+ReLU]*k -> Linear -> logits
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 5,  # total Linear layers including first & output; must be >= 4
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 4, "num_layers must be >= 4"
        layers = [nn.Flatten(), nn.Linear(FLAT_D, hidden_dim), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # add (num_layers - 3) additional hidden Linear layers
        for _ in range(num_layers - 3):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Simple 2-layer MLP residual block:
        y = x + Linear(ReLU(Linear(x)))
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.fc2(out)
        return x + out


class MLPClassifierDeepResidual(nn.Module):
    """
    Deep MLP with residual connections.
    Structure: Flatten -> Linear(input->hidden) -> [ResidualBlock]*n -> Linear(hidden->classes)
    Ensures >=4 Linear layers total.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_blocks: int = 3,   # each block = 2 linear layers
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_blocks >= 2, "num_blocks must be >= 2 to be meaningfully deep"

        self.flatten = nn.Flatten()
        self.input_proj = nn.Linear(FLAT_D, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)  # helps training stability
        self.head = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.input_proj(x), inplace=True)
        for blk in self.blocks:
            x = blk(x)
            x = F.relu(x, inplace=True)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ------------------------------
# Utilities
# ------------------------------
def save_model(model: nn.Module, path: str) -> None:
    """
    Save model weights (state_dict). Path should usually end with .th
    """
    torch.save(model.state_dict(), path)


_ALLOWED_HPARAMS = {"hidden_dim", "num_layers", "num_blocks", "dropout"}

def model_factory(name: str, **kwargs) -> nn.Module:
    """
    Create a model by name. Extra kwargs are filtered so unexpected keys
    like 'with_weights' don't reach the constructors.
    """
    clean = {k: v for k, v in kwargs.items() if k in _ALLOWED_HPARAMS}
    name = name.lower()
    if name == "linear":
        return LinearClassifier()
    if name == "mlp":
        return MLPClassifier(**clean)
    if name == "mlp_deep":
        return MLPClassifierDeep(**clean)
    if name == "mlp_deep_residual":
        return MLPClassifierDeepResidual(**clean)
    raise ValueError(f"Unknown model name: {name}")

# Alias some code might expect
build_model = model_factory


# ------------------------------
# Flexible loader (works with both call styles)
# ------------------------------
def load_model(
    model_or_name,
    with_weights: bool = True,
    path: str | None = None,
    map_location: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Compatible with the grader:

    - load_model('linear', with_weights=False)  -> fresh Linear model
    - load_model('linear', with_weights=True)   -> loads from linear.th (or other common paths)
    - load_model(model_instance, with_weights=True/False) -> optionally loads into provided instance

    Also robust to the grader's working directory by searching multiple locations.
    """
    # 1) Build/accept the model instance
    if isinstance(model_or_name, nn.Module):
        model = model_or_name
        model_name = getattr(model, "__class__", type(model)).__name__.lower()
        # normalize common class names to checkpoint stem
        if "linear" in model_name: model_name = "linear"
        elif "residual" in model_name: model_name = "mlp_deep_residual"
        elif "deep" in model_name: model_name = "mlp_deep"
        elif "mlp" in model_name: model_name = "mlp"
    else:
        model_name = str(model_or_name).lower()
        # assumes you have model_factory in this module
        model = model_factory(model_name, **kwargs)

    # 2) If weights are not requested, just return the fresh model
    if not with_weights:
        return model

    # 3) Determine candidate checkpoint filenames
    ckpt = Path(path) if path is not None else Path(f"{model_name}.th")
    tried = []

    # project root = folder that contains homework/, grader/, bundle.py
    root = Path(__file__).resolve().parents[1]

    candidates = []
    # as given
    candidates.append(ckpt)
    # current working dir
    candidates.append(Path.cwd() / ckpt.name)
    # project root + common siblings
    candidates += [
        root / ckpt.name,
        root / "homework" / ckpt.name,
        root / "grader" / ckpt.name,
        root / "logs" / ckpt.name,
    ]
    # logs/<stem>_*/<stem>.th
    stem = ckpt.stem  # e.g., 'linear', 'mlp'
    candidates += sorted((root / "logs").glob(f"{stem}_*/{ckpt.name}"))
    # generic fallback
    candidates.append(root / "model.th")

    # 4) Try to load
    for c in candidates:
        try:
            if c.exists():
                state = torch.load(str(c), map_location=map_location)
                model.load_state_dict(state)
                return model
            tried.append(str(c))
        except Exception:
            tried.append(str(c))
            continue

    raise FileNotFoundError(
        f"Could not find checkpoint for '{model_name}'. "
        f"Tried: {' | '.join(tried)}"
    )


def calculate_model_size_mb(state_dict: Optional[Dict[str, torch.Tensor]] = None, model: Optional[nn.Module] = None) -> float:
    """
    Utility to estimate model size in MB for the 40MB submission limit.
    """
    if state_dict is None:
        if model is None:
            return 0.0
        state_dict = model.state_dict()
    total = 0
    for t in state_dict.values():
        total += t.numel() * t.element_size()
    return total / (1024 ** 2)
