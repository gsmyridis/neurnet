import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neurnet.nn import DeserializableTorchModel, SerializableTorchModel


class LeNet(SerializableTorchModel, DeserializableTorchModel):
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.cn_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.cn_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully-connected layers
        self.fc_1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc_2 = nn.Linear(in_features=120, out_features=84)
        self.fc_3 = nn.Linear(in_features=84, out_features=10)

    def _forward_features(self, input: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        feature_maps = {}

        # Covolution 1 with 5 x 5 kernel
        x = F.relu(self.cn_1(input))
        feature_maps["Convolution 1"] = x
        # Max pooling 1 over a (2, 2) window
        x = F.max_pool2d(input=x, kernel_size=(2, 2))
        # Convolusion 2 with 5 x 5 kernel
        x = F.relu(self.cn_2(x))
        feature_maps["Convolution 2"] = x
        # Max pooling 2 over a (2, 2) window
        x = F.max_pool2d(input=x, kernel_size=(2, 2))

        return x, feature_maps

    def extract_feature_maps(self, input: Tensor) -> dict[str, Tensor]:
        """Return the activations produced by both convolutional layers."""
        _, feature_maps = self._forward_features(input)
        return feature_maps

    def forward(self, input: Tensor) -> Tensor:
        x, _ = self._forward_features(input)

        # Flatten spatial and depth dimension into a single vector
        x = x.view(-1, self._n_features(x))
        # Fully connected operations
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x

    def _n_features(self, x: Tensor) -> int:
        # The first dimension expresses the number of images in the batch.
        size = x.size()[1:]
        num_feats = 1
        for s in size:
            num_feats *= s

        return num_feats


def train_lenet(net: LeNet, train_loader: DataLoader, optim: Optimizer, epoch: int):
    loss_total = 0.0

    for i, (input, ground_truth) in enumerate(train_loader):
        optim.zero_grad()
        infered = net(input)
        loss = nn.CrossEntropyLoss().forward(infered, ground_truth)
        loss.backward()
        optim.step()
        loss_total += loss.item()

        if (i + 1) % 1000 == 0:
            print(
                f"[Epoch number: {epoch + 1}, Mini-batches: {i + 1}, loss: {loss_total:.3f}"
            )
            loss_total = 0.0


def test_lenet(net: LeNet, test_loader: DataLoader):
    success = 0
    counter = 0

    with torch.no_grad():
        for input, ground_truth in test_loader:
            infered = net(input)
            _, pred = torch.max(infered.data, 1)
            counter += ground_truth.size(0)
            success += (pred == ground_truth).sum().item()

    pct = 100 * success / counter
    print(f"LeNet accuracy on 10000 images from test dataset: {pct:.2f}%")
