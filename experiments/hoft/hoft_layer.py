"""Householder Orthogonal Fine-Tuning for a dense layer.

The compact-WY first-order inverse used here follows the HOFT paper and its
official implementation. It applies ``W' = Q_U W Q_V`` without materializing
either large orthogonal matrix during the forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class HOFTLinear(nn.Module):
    """Apply learned Householder products to both sides of a frozen weight."""

    def __init__(self, in_features: int, out_features: int, *, rank: int) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError("HOFT rank must be in [1, min(in_features, out_features)].")
        if rank % 2:
            raise ValueError("This experiment uses an even HOFT rank for identity initialization.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.base = nn.Linear(in_features, out_features)
        self.left_vectors = nn.Parameter(torch.empty(out_features, rank))
        self.right_vectors = nn.Parameter(torch.empty(in_features, rank))
        self.reset_adapter_parameters()

    @staticmethod
    def _paired_vectors(dimension: int, rank: int) -> torch.Tensor:
        half = torch.empty(dimension, rank // 2)
        nn.init.normal_(half, mean=0.0, std=1.0 / math.sqrt(dimension))
        half = F.normalize(half, dim=0, eps=1e-12)
        return torch.repeat_interleave(half, 2, dim=1)

    def reset_adapter_parameters(self) -> None:
        with torch.no_grad():
            self.left_vectors.copy_(self._paired_vectors(self.out_features, self.rank))
            self.right_vectors.copy_(self._paired_vectors(self.in_features, self.rank))

    @staticmethod
    def _compact_wy_factor(vectors: torch.Tensor) -> torch.Tensor:
        """Return the HOFT first-order coefficient S in Q = I + V S V^T."""
        gram = vectors.T @ vectors
        diagonal_inverse = 2.0 * gram.diagonal().reciprocal()
        # This deliberately mirrors the official HOFT broadcasting rule:
        # each column of the strict upper triangle is scaled twice.
        factor = diagonal_inverse * torch.triu(gram, diagonal=1) * diagonal_inverse
        factor = factor.clone()
        factor.diagonal().sub_(diagonal_inverse)
        return factor

    @classmethod
    def apply_transpose(cls, inputs: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """Apply ``Q^T`` to row vectors using the compact representation."""
        factor = cls._compact_wy_factor(vectors)
        return inputs + (inputs @ vectors) @ factor @ vectors.T

    @classmethod
    def dense_matrix(cls, vectors: torch.Tensor) -> torch.Tensor:
        factor = cls._compact_wy_factor(vectors)
        identity = torch.eye(vectors.shape[0], device=vectors.device, dtype=vectors.dtype)
        return identity + vectors @ factor @ vectors.T

    def effective_weight(self) -> torch.Tensor:
        # The compact matrices below act on PyTorch row vectors. Transposing
        # the complete row-vector computation gives the effective nn.Linear
        # weight stored in [out_features, in_features] convention.
        left = self.dense_matrix(self.left_vectors)
        right = self.dense_matrix(self.right_vectors)
        return left.T @ self.base.weight @ right.T

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        right_rotated = self.apply_transpose(inputs, self.right_vectors)
        outputs = F.linear(right_rotated, self.base.weight, bias=None)
        outputs = self.apply_transpose(outputs, self.left_vectors)
        if self.base.bias is not None:
            outputs = outputs + self.base.bias
        return outputs
