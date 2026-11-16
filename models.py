"""CNN profiles for Section VI and the CIFAR WideResNet-28-4."""

from torch import nn


def cnn(channels, classes, profile):
    widths, kernel, padding, side = {
        "emnist": ((32, 64), 5, 0, 4),
        "cifar10": ((32, 64), 5, 2, 8),
        "cifar100": ((32, 64, 128), 3, 1, 4),
    }[profile]
    layers = []
    for width in widths:
        layers.extend((
            nn.Conv2d(channels, width, kernel, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ))
        channels = width
    layers.extend((nn.Flatten(), nn.Linear(channels * side * side, 128),
                   nn.ReLU(inplace=True), nn.Linear(128, classes)))
    return nn.Sequential(*layers)


class WideBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.shortcut = None
        if cin != cout or stride != 1:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)

    def forward(self, x):
        activated = self.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(activated)
        return self.conv2(self.relu(self.bn2(self.conv1(activated)))) + shortcut


def wide_resnet(classes):
    layers = [nn.Conv2d(3, 16, 3, padding=1, bias=False)]
    previous = 16
    for group, width in enumerate((64, 128, 256)):
        for block in range(4):
            stride = 2 if group and block == 0 else 1
            layers.append(WideBlock(previous, width, stride))
            previous = width
    layers.extend((nn.BatchNorm2d(previous), nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(previous, classes)))
    return nn.Sequential(*layers)


def make_model(dataset, architecture, classes):
    if architecture == "wrn28-4":
        return wide_resnet(classes)
    profile = "emnist" if dataset == "synthetic" else dataset
    return cnn(1 if profile == "emnist" else 3, classes, profile)
