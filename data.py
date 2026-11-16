"""Datasets and deterministic, disjoint client and server splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST


@dataclass
class Images:
    pixels: torch.Tensor
    labels: torch.Tensor
    classes: int
    mean: tuple
    std: tuple

    def __post_init__(self):
        self.labels = self.labels.to(torch.long)
        self.mean = torch.tensor(self.mean).view(1, -1, 1, 1)
        self.std = torch.tensor(self.std).view(1, -1, 1, 1)

    def batch(self, indices, device, labels=None):
        x = self.pixels[indices].to(torch.float32).div_(255)
        x.sub_(self.mean).div_(self.std)
        if labels is None:
            labels = self.labels
        return x.to(device), labels[indices].to(device)


def load_data(name, directory, download):
    if name == "synthetic":
        return synthetic_images(80, 110), synthetic_images(20, 120)
    dataset = {"emnist": EMNIST, "cifar10": CIFAR10, "cifar100": CIFAR100}[name]
    options = {"split": "balanced"} if name == "emnist" else {}
    normalization = {
        "emnist": ((0.1751,), (0.3332,)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }[name]
    result = []
    for training in (True, False):
        split = dataset(str(directory), train=training, download=download, **options)
        images = torch.as_tensor(split.data)
        if name == "emnist":
            images = images.transpose(-1, -2).unsqueeze(1)
        else:
            images = images.permute(0, 3, 1, 2)
        result.append(Images(images, torch.as_tensor(split.targets), len(split.classes), *normalization))
    return result


def synthetic_images(per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), per_class)
    images = rng.integers(0, 36, (len(labels), 1, 28, 28), dtype=np.uint8)
    for label in range(10):
        row, col = (label // 5) * 12 + 2, (label % 5) * 5 + 2
        images[labels == label, 0, row:row + 8, col:col + 4] += 200
    return Images(torch.from_numpy(images), torch.from_numpy(labels), 10, (0.1,), (0.25,))


def class_balanced_pool(labels, size, rng, vetted_indices=None):
    labels = np.asarray(labels)
    eligible = np.arange(len(labels))
    if vetted_indices:
        eligible = np.loadtxt(Path(vetted_indices), dtype=np.int64, ndmin=1)
        if (eligible.ndim != 1 or np.unique(eligible).size != eligible.size
                or np.any(eligible < 0) or np.any(eligible >= len(labels))):
            raise ValueError("vetted indices must be unique, zero-based training indices")
    groups = [rng.permutation(eligible[labels[eligible] == label]) for label in np.unique(labels)]
    return balanced_prefix(groups, size)


def balanced_prefix(groups, size):
    if not size:
        return np.empty(0, dtype=np.int64)
    quotient, remainder = divmod(size, len(groups))
    lengths = [quotient + (idx < remainder) for idx in range(len(groups))]
    if any(count > len(group) for group, count in zip(groups, lengths)):
        raise ValueError("insufficient eligible samples for a class-balanced root pool")
    # Interleave classes so every prefix is also balanced.
    core = np.stack([group[:quotient] for group in groups], axis=1).ravel()
    tail = np.asarray([group[quotient] for group in groups[:remainder]], dtype=np.int64)
    return np.concatenate((core, tail))


def dirichlet_split(labels, available, workers, beta, rng):
    labels = np.asarray(labels)
    if len(available) < workers:
        raise ValueError("the client pool must contain at least one sample per worker")
    for _ in range(10_000):
        partitions = [[] for _ in range(workers)]
        for label in np.unique(labels):
            indices = rng.permutation(available[labels[available] == label])
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(workers, beta)))
            for bucket, chunk in zip(partitions, np.split(indices, counts.cumsum()[:-1])):
                bucket.extend(chunk)
        if all(partitions):
            return [np.asarray(partition, dtype=np.int64) for partition in partitions]
    raise ValueError("could not draw nonempty client partitions; increase beta or reduce workers")


def corrupt_labels(labels, indices, ratio, classes, rng, reverse=False):
    """Select a fixed subset per experiment, keeping the source labels intact."""
    result = labels.clone()
    chosen = rng.choice(indices, int(len(indices) * ratio), replace=False)
    if reverse:
        result[chosen] = classes - 1 - labels[chosen]
    else:
        offset = torch.from_numpy(rng.integers(1, classes, len(chosen)))
        result[chosen] = (labels[chosen] + offset) % classes
    return result
