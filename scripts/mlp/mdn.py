"""Fit a Gaussian mixture to y = +/- sin(x) + noise using MLX."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.optimizers import Adam

from neurnet.arch.mlp import MixtureDensityNetwork


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-plot", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.epochs, args.batch_size, args.components) <= 0:
        parser.error("samples, epochs, batch size, and components must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")

    return args


def log_density(outputs: tuple[mx.array, mx.array, mx.array], y: mx.array) -> mx.array:
    log_weights, means, scales = outputs
    log_normal = (
        -0.5 * ((y - means) / scales) ** 2
        - mx.log(scales)
        - 0.5 * math.log(2 * math.pi)
    )
    return mx.logsumexp(log_weights + log_normal, axis=-1)


def loss_fn(model: MixtureDensityNetwork, x: mx.array, y: mx.array) -> mx.array:
    return -mx.mean(log_density(model(x), y))


def make_data(rng: np.random.Generator, samples: int) -> tuple[mx.array, mx.array]:
    x = rng.uniform(-math.pi, math.pi, size=(samples, 1))
    branch = rng.choice([-1.0, 1.0], size=(samples, 1))
    y = branch * np.sin(x) + rng.normal(0, 0.1, size=(samples, 1))
    return mx.array(x, dtype=mx.float32), mx.array(y, dtype=mx.float32)


def plot_density(
    model: MixtureDensityNetwork,
    x: mx.array,
    y: mx.array,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot the learned conditional density over the training samples."""
    grid_x = mx.linspace(-math.pi, math.pi, 200)[:, None]
    grid_y = mx.linspace(-1.5, 1.5, 200)
    outputs = tuple(value[None, :, :] for value in model(grid_x))
    density = np.array(mx.exp(log_density(outputs, grid_y[:, None, None])))
    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap = ax.pcolormesh(
        np.array(grid_x[:, 0]), np.array(grid_y), density, shading="auto", cmap="Blues"
    )
    ax.scatter(np.array(x[:, 0]), np.array(y[:, 0]), s=3, alpha=0.15, color="black")
    ax.set(xlabel="x", ylabel="y", title="Mixture Density Network: p(y | x)")
    fig.colorbar(heatmap, ax=ax, label="Conditional density")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_arguments()

    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)
    x, y = make_data(rng, args.samples)
    test_x, test_y = make_data(rng, 1024)

    model = MixtureDensityNetwork(components=args.components)
    optimizer = Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    print(f"Initial test NLL: {loss_fn(model, test_x, test_y).item():.4f}")

    for epoch in range(args.epochs):
        order = mx.array(rng.permutation(args.samples))
        total_loss = 0.0
        for start in range(0, args.samples, args.batch_size):
            indices = order[start : start + args.batch_size]
            loss, grads = loss_and_grad(model, x[indices], y[indices])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += loss.item() * indices.size
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            test_loss = loss_fn(model, test_x, test_y).item()
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train NLL {total_loss / args.samples:.4f}, test NLL {test_loss:.4f}"
            )

    plot_density(model, x, y, save_path=args.save_plot, show=not args.no_show)


if __name__ == "__main__":
    main()
