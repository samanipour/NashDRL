from __future__ import annotations

from torch import Tensor, nn

from .common.mlp import MLP


class DeepSetEncoder(nn.Module):
    """Permutation-invariant encoder for the other-agent feature set.

    defines the invariant stream as ``[N, N-1, F]``: for each
    focal agent, the N-1 rival feature vectors are independently transformed
    by a shared embedding function phi and then summed.  The sum is therefore
    invariant to any permutation of the rival-agent ordering.

    A leading batch dimension is also supported for training:
    ``[B, N, N-1, F] -> [B, N, D]``.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        embedding_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if embedding_hidden_layers < 0:
            raise ValueError("embedding_hidden_layers must be non-negative")

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        # phi is shared for every rival j and every focal agent i.
        self.phi = MLP(
            feature_dim,
            hidden_dim=hidden_dim,
            hidden_layers=embedding_hidden_layers,
            output_dim=embedding_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(
                "Deep Sets input must be [N,N-1,F] or [B,N,N-1,F], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension F={self.feature_dim}, got {x.shape[-1]}"
            )

        # Preserve the leading dimensions and apply the same phi to every
        # element in the rival-agent set.
        flat = x.reshape(-1, self.feature_dim)
        embedded = self.phi(flat).reshape(*x.shape[:-1], self.embedding_dim)

        # Sum aggregation is the permutation-invariant operation specified in
        # Section 4 of NashDRL-v12.
        return embedded.sum(dim=-2)
