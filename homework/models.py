from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    path: Optional[str] = None,
    map_location: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Two modes:
      1) load_model(model, path) -> loads weights into existing model
      2) load_model(name, **kwargs) -> builds model; if 'with_weights' or 'ckpt/path' provided,
         tries to load checkpoint.
    """
    # Mode 1: original contract (model instance + checkpoint path)
    if isinstance(model_or_name, nn.Module):
        model = model_or_name
        if path is not None:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"load_model: checkpoint not found: {path}")
            state = torch.load(path, map_location=map_location or "cpu")
            model.load_state_dict(state)
        return model

    # Mode 2: name string
    if isinstance(model_or_name, str):
        name = model_or_name

        # Consume grader/driver flags so they don't hit constructors
        with_weights = bool(kwargs.pop("with_weights", False))
        ckpt = kwargs.pop("ckpt", None) or path

        # Build model with only allowed hyperparameters
        model = model_factory(name, **kwargs)

        # If weights requested but no ckpt passed, try common default paths (optional)
        if with_weights and not ckpt:
            for p in (
                f"pretrained/{name}.pt",
                f"pretrained/{name}.pth",
                f"assets/{name}.th",
                f"{name}.pt",
            ):
                if os.path.isfile(p):
                    ckpt = p
                    break

        # Load if we have a checkpoint file
        if ckpt:
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"load_model: checkpoint not found: {ckpt}")
            state = torch.load(ckpt, map_location=map_location or "cpu")
            model.load_state_dict(state)

        return model

    raise TypeError(
        "load_model: first argument must be an nn.Module (to load weights) "
        "or a str model name (to construct a new model)."
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
