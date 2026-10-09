import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.datasets import MLXDataLoader
from neurnet.nn import Flatten

# ===---------------------------------------------------------------------------===
# Convolutional Encoder and Decoder
# ===---------------------------------------------------------------------------===


class ConvEncoder(nn.Module):
    """Map NHWC or flattened image batches to convolutional features."""

    def __init__(self, image_dims: tuple[int, int] = (24, 24), in_channels: int = 3):
        super().__init__()
        assert all(d > 0 and d % 4 == 0 for d in image_dims), (
            "Image dimensions must be positive multiples of 4."
        )
        assert in_channels > 0, "in_channels must be positive."
        self.image_shape = (*image_dims, in_channels)
        self.feature_shape = (image_dims[0] // 4, image_dims[1] // 4, 32)
        self.feature_size = self.feature_shape[0] * self.feature_shape[1] * 32
        # 16 and 32 are feature-channel counts. Kernel 4, stride 2, padding 1
        # halve spatial dimensions twice: e.g. 28 -> 14 -> 7 for MNIST.
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(start_axis=1),  # Preserve batch axis 0.
        )

    def prepare_images(self, x: mx.array) -> mx.array:
        """Restore flattened dataset batches to the configured NHWC shape."""
        if x.ndim == 2:
            height, width, channels = self.image_shape
            assert x.shape[1] == height * width * channels, (
                f"Expected {height * width * channels} pixels per image, got {x.shape[1]}."
            )
            x = x.reshape((x.shape[0], *self.image_shape))
        assert x.ndim == 4 and x.shape[1:] == self.image_shape, (
            f"Expected input shape (batch, {', '.join(map(str, self.image_shape))}), got {x.shape}."
        )
        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self.prepare_images(x)
        h = self.layers(x)
        # Flattened features: 6 * 6 * 32 = 1152 for 24 x 24, or 1568 for 28 x 28.
        assert h.shape == (x.shape[0], self.feature_size), (
            f"Expected encoded features shape ({x.shape[0]}, {self.feature_size}), got {h.shape}."
        )
        return h


class ConvDecoder(nn.Module):
    """Map vector codes to NHWC reconstructions in [0, 1]."""

    def __init__(
        self,
        code_size: int,
        *,
        image_dims: tuple[int, int] = (24, 24),
        in_channels: int = 3,
    ):
        super().__init__()
        assert code_size > 0, "code_size must be positive."
        self.code_size = code_size
        assert all(d > 0 and d % 4 == 0 for d in image_dims), (
            "Image dimensions must be positive multiples of 4."
        )
        assert in_channels > 0, "in_channels must be positive."
        self.image_shape = (*image_dims, in_channels)
        self.feature_shape = (image_dims[0] // 4, image_dims[1] // 4, 32)
        feature_size = self.feature_shape[0] * self.feature_shape[1] * 32

        self.fc = nn.Linear(input_dims=code_size, output_dims=feature_size)
        # Double spatial dimensions twice; restore the original channel count.
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def __call__(self, z: mx.array) -> mx.array:
        assert z.ndim == 2 and z.shape[1] == self.code_size, (
            f"Expected latent shape (batch, {self.code_size}), got {z.shape}."
        )
        # Restore the NHWC feature map with the original batch size.
        h = self.fc(z).reshape((z.shape[0], *self.feature_shape))
        reconstruction = self.layers(h)
        assert reconstruction.shape == (z.shape[0], *self.image_shape), (
            f"Expected reconstruction shape ({z.shape[0]}, {self.image_shape}), "
            f"got {reconstruction.shape}."
        )
        return reconstruction


# ===---------------------------------------------------------------------------===
# Convolutional Autoencoder
# ===---------------------------------------------------------------------------===


class ConvAE(nn.Module):
    """Deterministic vector autoencoder for NHWC images in [0, 1]."""

    def __init__(
        self,
        code_size: int,
        *,
        image_dims: tuple[int, int] = (24, 24),
        in_channels: int = 3,
    ):
        super().__init__()
        assert code_size > 0, "code_size must be positive."
        self.code_size = code_size
        self.encoder = ConvEncoder(image_dims, in_channels)
        self.code = nn.Linear(
            input_dims=self.encoder.feature_size, output_dims=code_size
        )
        self.decoder = ConvDecoder(
            code_size, image_dims=image_dims, in_channels=in_channels
        )

    def encode(self, x: mx.array) -> mx.array:
        return self.code(self.encoder(x))

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)

    def __call__(self, x: mx.array) -> mx.array:
        return self.decode(self.encode(x))


def ae_loss(model: ConvAE, images: mx.array) -> mx.array:
    images = model.encoder.prepare_images(images)
    reconstructed = model(images)
    return mx.mean((reconstructed - images) ** 2)


def train_autoencoder(
    epochs: int,
    model: ConvAE,
    train_loader: MLXDataLoader,
    test_loader: MLXDataLoader,
    optimizer: Optimizer,
) -> ConvAE:

    loss_and_grad = nn.value_and_grad(model, ae_loss)
    for e in range(1, epochs + 1):
        train_loader.reset()
        model.train()
        train_loss_sum = mx.array(0.0)
        train_count = 0

        for imgs, _ in train_loader:
            loss, grads = loss_and_grad(model, imgs)
            optimizer.update(model, grads)
            train_loss_sum += loss * imgs.shape[0]
            train_count += imgs.shape[0]
            mx.eval(model.parameters(), optimizer.state, train_loss_sum)

        # Evaluate the model
        test_loader.reset()
        model.eval()
        test_loss_sum = mx.array(0.0)
        pixel_square_sum = mx.array(0.0)
        test_count = 0
        for imgs, _ in test_loader:
            images = model.encoder.prepare_images(imgs)
            test_loss_sum += ae_loss(model, images) * images.shape[0]
            pixel_square_sum += mx.sum(images**2)
            test_count += images.shape[0]
            mx.eval(test_loss_sum, pixel_square_sum)

        if not train_count or not test_count:
            raise ValueError("Training and test loaders must contain images.")
        train_mse = train_loss_sum / train_count
        test_mse = test_loss_sum / test_count
        print(
            f"[Epoch {e} / {epochs}] Train MSE = {train_mse.item():.6f}, "
            f"Test MSE = {test_mse.item():.6f}, "
        )

    return model


# ===---------------------------------------------------------------------------===
# Convolutional Variational Autoencoder
# ===---------------------------------------------------------------------------===


class ConvVAE(nn.Module):
    """Variational vector autoencoder with the same convolutional architecture."""

    def __init__(
        self,
        code_size: int,
        *,
        image_dims: tuple[int, int] = (24, 24),
        in_channels: int = 3,
    ):
        super().__init__()
        assert code_size > 0, "code_size must be positive."
        self.code_size = code_size
        self.encoder = ConvEncoder(image_dims, in_channels)
        # Predict code_size-dimensional mean and log-variance vectors.
        self.mu = nn.Linear(input_dims=self.encoder.feature_size, output_dims=code_size)
        self.logvar = nn.Linear(
            input_dims=self.encoder.feature_size, output_dims=code_size
        )
        self.decoder = ConvDecoder(
            code_size, image_dims=image_dims, in_channels=in_channels
        )

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        mu, logvar = self.encode(x)
        # Standard deviation = exp(log(variance) / 2), hence the factor 0.5.
        z = mu + mx.random.normal(shape=mu.shape, dtype=mu.dtype) * mx.exp(0.5 * logvar)
        return self.decode(z), mu, logvar
