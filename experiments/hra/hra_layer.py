"""A small, self-contained Householder Reflection Adaptation layer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class HRALinear(nn.Module):
    """Adapt a frozen linear map as ``W' = W H_r ... H_1``.

    Each reflection is ``H_i = I - 2 u_i u_i^T / ||u_i||^2``.  Repeating
    every initial vector twice makes the adapter exactly the identity at
    initialization while keeping every vector independently trainable.
    """

    def __init__(self, in_features: int, out_features: int, *, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("HRA rank must be positive.")
        if rank % 2:
            raise ValueError("This experiment uses an even HRA rank for identity initialization.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.base = nn.Linear(in_features, out_features)
        self.householder_vectors = nn.Parameter(torch.empty(in_features, rank))
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        half = torch.empty(self.in_features, self.rank // 2)
        nn.init.kaiming_uniform_(half, a=math.sqrt(5))
        with torch.no_grad():
            self.householder_vectors.copy_(torch.repeat_interleave(half, 2, dim=1))

    def normalized_vectors(self) -> torch.Tensor:
        return F.normalize(self.householder_vectors, dim=0, eps=1e-12)

    def compact_wy_factor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return V, T such that the reflection product is I - V T V^T."""
        vectors = self.normalized_vectors()
        inverse_factor = 0.5 * torch.eye(
            self.rank,
            device=vectors.device,
            dtype=vectors.dtype,
        ) + torch.triu(vectors.T @ vectors, diagonal=1)
        factor = torch.linalg.solve_triangular(
            inverse_factor,
            torch.eye(self.rank, device=vectors.device, dtype=vectors.dtype),
            upper=True,
        )
        return vectors, factor

    def orthogonal_matrix(self, *, reverse: bool = False) -> torch.Tensor:
        """Materialize the reflection product for tests and diagnostics."""
        if not reverse:
            vectors, factor = self.compact_wy_factor()
            identity = torch.eye(
                self.in_features,
                device=vectors.device,
                dtype=vectors.dtype,
            )
            return identity - vectors @ factor @ vectors.T

        vectors = self.normalized_vectors()
        matrix = torch.eye(
            self.in_features,
            device=vectors.device,
            dtype=vectors.dtype,
        )
        indices = range(self.rank - 1, -1, -1) if reverse else range(self.rank)
        for index in indices:
            vector = vectors[:, index : index + 1]
            matrix = matrix - 2.0 * (matrix @ vector) @ vector.T
        return matrix

    def effective_weight(self) -> torch.Tensor:
        return self.base.weight @ self.orthogonal_matrix()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # W' = W H implies x W'^T = x H^T W^T. Applying the reflections to
        # activations avoids a dense W @ H multiplication at every step. The
        # compact-WY form is exactly equal to the sequential reflections while
        # retaining only a rank-r intermediate tensor for backpropagation.
        vectors, factor = self.compact_wy_factor()
        transformed = inputs - ((inputs @ vectors) @ factor.T) @ vectors.T
        return F.linear(transformed, self.base.weight, self.base.bias)
