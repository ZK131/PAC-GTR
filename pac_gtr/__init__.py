"""PAC-GTR code-only reference implementation."""

from .core import Component, Protocol, predict_components, robust_diagonal
from .evaluation import evaluate_components

__all__ = [
    "Component",
    "Protocol",
    "evaluate_components",
    "predict_components",
    "robust_diagonal",
]

