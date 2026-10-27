"""Mise en forme et contrôle des piles d'images en entrée des mesures."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_stack(frames: ArrayLike | Sequence[ArrayLike], *, min_frames: int, name: str = "frames") -> NDArray[np.float64]:
    """
    Convertit une pile en tableau `float64` de forme (K, H, W).

    Accepte un tableau (K, H, W) ou une séquence d'images 2D de même forme.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"{name} doit être une pile (K, H, W), reçu une forme {stack.shape}")
    if stack.shape[0] < min_frames:
        raise ValueError(f"{name} doit contenir au moins {min_frames} images, reçu {stack.shape[0]}")
    if not np.all(np.isfinite(stack)):
        raise ValueError(f"{name} contient des valeurs non finies")
    return stack


def pushbroom_as_stack(lines: ArrayLike, *, min_lines: int, name: str = "lines") -> NDArray[np.float64]:
    """
    Convertit un bloc de lignes push broom (M, W) en pile (M, 1, W).

    Chaque ligne est une acquisition ; chaque colonne, un détecteur.
    """
    block = np.asarray(lines, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError(f"{name} doit être un bloc de lignes (M, W), reçu une forme {block.shape}")
    return as_stack(block[:, None, :], min_frames=min_lines, name=name)
