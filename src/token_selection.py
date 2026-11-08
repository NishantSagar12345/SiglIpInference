"""Deterministic, spatially uniform patch-token selection on a ViT patch grid.

For SigLIP-Base/16 at 224x224 the patch grid is 14x14 = 196 tokens, flattened
in row-major order (index = row * 14 + col), which is the order produced by
``patch_embeds.flatten(2).transpose(1, 2)`` in ``SiglipVisionEmbeddings``.

Selection rule (one rule for every configuration)
-------------------------------------------------
For a target sub-grid of k x k tokens we choose k row indices and k column
indices with

    rows = cols = round(linspace(0, G - 1, k))      (G = 14)

and keep the Cartesian product rows x cols, sorted in row-major order. The
retained patches therefore always include the four corners and are spread as
evenly as an integer grid allows. Because the spacing (G-1)/(k-1) is > 1 for
every k < G, rounding can never map two linspace points to the same integer,
so the indices are unique by construction; this is still asserted explicitly.

  k = 14 -> 0..13 (identity, 196 tokens)
  k = 12 -> 0 1 2 4 5 6 7 8 9 11 12 13           (144 tokens)
  k = 10 -> 0 1 3 4 6 7 9 10 12 13               (100 tokens)
  k =  7 -> 0 2 4 6 9 11 13                      ( 49 tokens)

The same index tensor is used for every image in a configuration. No
randomness is involved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TokenConfig:
    name: str
    side: int          # k: retained tokens form a k x k sub-grid
    grid_size: int     # G: original grid side (14 for SigLIP-B/16 @ 224)

    @property
    def num_tokens(self) -> int:
        return self.side * self.side


def axis_indices(grid_size: int, side: int) -> np.ndarray:
    """Return ``side`` evenly spaced integer positions in [0, grid_size - 1]."""
    if not (1 <= side <= grid_size):
        raise ValueError(f"side must be in [1, {grid_size}], got {side}")
    if side == 1:
        return np.array([grid_size // 2], dtype=np.int64)
    pos = np.round(np.linspace(0, grid_size - 1, side)).astype(np.int64)
    assert len(np.unique(pos)) == side, f"duplicate axis indices for side={side}: {pos}"
    return pos


def grid_indices(grid_size: int, side: int) -> torch.Tensor:
    """Flattened row-major indices of a uniform ``side x side`` sub-grid."""
    rows = axis_indices(grid_size, side)
    cols = axis_indices(grid_size, side)
    idx = (rows[:, None] * grid_size + cols[None, :]).reshape(-1)
    idx = torch.as_tensor(np.sort(idx), dtype=torch.long)
    validate_indices(idx, grid_size * grid_size, side * side)
    return idx


def validate_indices(idx: torch.Tensor, num_patches: int, expected: int) -> None:
    assert idx.dtype == torch.long, idx.dtype
    assert idx.ndim == 1, idx.shape
    assert idx.numel() == expected, f"expected {expected} indices, got {idx.numel()}"
    assert torch.unique(idx).numel() == idx.numel(), "duplicate token indices"
    assert int(idx.min()) >= 0 and int(idx.max()) < num_patches, "index out of range"
    assert torch.all(idx[1:] > idx[:-1]), "indices must be strictly increasing"


def build_configs(grid_size: int = 14, sides=(14, 12, 10, 7)) -> dict[str, torch.Tensor]:
    """Map config name (e.g. '196') -> index tensor."""
    out = {}
    for s in sides:
        out[str(s * s)] = grid_indices(grid_size, s)
    return out


def describe(idx: torch.Tensor, grid_size: int = 14) -> str:
    """ASCII map of retained patches (X) on the original grid."""
    keep = set(idx.tolist())
    lines = []
    for r in range(grid_size):
        lines.append(" ".join("X" if r * grid_size + c in keep else "." for c in range(grid_size)))
    return "\n".join(lines)


def selection_table(configs: dict[str, torch.Tensor], grid_size: int = 14) -> list[dict]:
    rows = []
    for name, idx in configs.items():
        side = int(round(math.sqrt(idx.numel())))
        rows.append({
            "config": name,
            "num_tokens": idx.numel(),
            "grid": f"{side}x{side}",
            "axis_positions": " ".join(map(str, axis_indices(grid_size, side).tolist())),
            "fraction_kept": idx.numel() / grid_size ** 2,
            "indices": idx.tolist(),
        })
    return rows
