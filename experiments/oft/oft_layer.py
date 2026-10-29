"""Standalone block-diagonal Orthogonal Fine-Tuning layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BlockOFTLinear(nn.Module):
    """A frozen linear layer preceded by one trainable block-orthogonal map."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if in_features % block_size != 0:
            raise ValueError("in_features must be divisible by block_size.")

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = in_features // block_size
        self.base = nn.Linear(in_features, out_features, bias=bias)

        independent_entries = block_size * (block_size - 1) // 2
        self.skew_parameters = nn.Parameter(
            torch.zeros(1, self.num_blocks, independent_entries)
        )

        upper_row, upper_col = torch.triu_indices(
            block_size,
            block_size,
            offset=1,
        )
        self.register_buffer("upper_row", upper_row, persistent=False)
        self.register_buffer("upper_col", upper_col, persistent=False)
        self._rotation_cache: torch.Tensor | None = None

    def skew_matrices(self) -> torch.Tensor:
        """Construct the skew-symmetric generator of every block."""
        shape = (1, self.num_blocks, self.block_size, self.block_size)
        skew = self.skew_parameters.new_zeros(shape)
        skew[..., self.upper_row, self.upper_col] = self.skew_parameters
        skew[..., self.upper_col, self.upper_row] = -self.skew_parameters
        return skew

    def orthogonal_blocks(self) -> torch.Tensor:
        """Map every skew-symmetric block through the Cayley transform."""
        if self._rotation_cache is not None:
            return self._rotation_cache

        with torch.autocast(
            device_type=self.skew_parameters.device.type,
            enabled=False,
        ):
            skew = self.skew_matrices().float()
            identity = torch.eye(
                self.block_size,
                dtype=skew.dtype,
                device=skew.device,
            )
            return torch.linalg.solve(identity + skew, identity - skew)

    def set_rotation_cache(self, rotations: torch.Tensor) -> None:
        """Temporarily reuse rotations produced by a model-level solve."""
        expected_shape = (1, self.num_blocks, self.block_size, self.block_size)
        if rotations.shape != expected_shape:
            raise ValueError(
                f"OFT rotation cache must have shape {expected_shape}, "
                f"got {tuple(rotations.shape)}."
            )
        self._rotation_cache = rotations

    def clear_rotation_cache(self) -> None:
        """Drop the temporary graph-bearing cache after one forward pass."""
        self._rotation_cache = None

    def rotate_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the block-diagonal orthogonal map to the final tensor axis."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected final dimension {self.in_features}, "
                f"got {inputs.shape[-1]}."
            )

        blocks = inputs.reshape(
            *inputs.shape[:-1],
            self.num_blocks,
            self.block_size,
        )
        rotations = self.orthogonal_blocks()[0].to(dtype=inputs.dtype)
        rotated = torch.einsum("...ki,kij->...kj", blocks, rotations)
        return rotated.flatten(-2)

    def dense_orthogonal_matrix(self) -> torch.Tensor:
        """Materialize the block-diagonal transformation for verification."""
        identity = torch.eye(
            self.in_features,
            dtype=self.skew_parameters.dtype,
            device=self.skew_parameters.device,
        )
        return self.rotate_features(identity)

    def effective_weight(self) -> torch.Tensor:
        """Return the dense adapted weight without changing the base weight."""
        rotation = self.dense_orthogonal_matrix()
        return self.base.weight @ rotation.transpose(0, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            self.rotate_features(inputs),
            self.base.weight,
            self.base.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, bias={self.base.bias is not None}"
        )
