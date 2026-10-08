import mlx.core as mx
from mlx import nn


class MixtureDensityNetwork(nn.Module):
    def __init__(self, components: int = 2, hidden_dims: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, 3 * components),
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Return log weights, means, and positive scales, each shaped (B, K)."""
        logits, means, raw_scales = mx.split(self.network(x), 3, axis=-1)
        log_weights = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scales = nn.softplus(raw_scales) + 1e-3
        return log_weights, means, scales
