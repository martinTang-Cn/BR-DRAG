"""Run one reproducible federated learning experiment."""

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from aggregation import aggregate, directions, perturb
from data import class_balanced_pool, corrupt_labels, dirichlet_split, load_data
from models import make_model


DATASETS = ("emnist", "cifar10", "cifar100", "synthetic")
METHODS = ("drag", "br-drag", "fedavg", "fedprox", "scaffold", "fedexp", "fedacg",
           "fltrust", "rfa", "raga", "br-trust", "root-only")
ATTACKS = ("none", "noise", "sign-flip", "label-flip", "min-max", "min-sum")
ROOT_METHODS = ("br-drag", "fltrust", "br-trust", "root-only")
METRICS = (
    "round", "test_accuracy", "test_loss", "malicious_fraction", "reference_norm",
    "benign_rho_min", "benign_rho_max", "benign_cos_rho_mean",
    "client_seconds_mean", "server_seconds", "server_client_ratio",
)


@dataclass(frozen=True)
class Experiment:
    dataset: str = "cifar10"
    method: str = "drag"
    model: str = "cnn"
    attack: str = "none"
    attacker_ratio: float = 0.0
    workers: int = 40
    participants: int = 10
    beta: float = 0.1
    rounds: Optional[int] = None
    local_steps: int = 5
    batch_size: int = 10
    lr: float = 0.01
    alpha: float = 0.25
    c: Optional[float] = None
    root_size: Optional[int] = None
    root_pool_size: Optional[int] = None
    root_corruption: float = 0.0
    root_missing: int = 0
    vetted_indices: Optional[str] = None
    seed: int = 0
    eval_every: int = 10
    device: str = "auto"
    threads: int = 4
    data_dir: str = "data"
    download: bool = False

    def resolved(self):
        values = asdict(self)
        if self.rounds is None:
            values["rounds"] = {"emnist": 3000, "synthetic": 40}.get(self.dataset, 4000)
        if self.c is None:
            values["c"] = (0.25 if self.beta <= 0.1 else 0.1) if self.method == "drag" else 0.5
        if self.root_size is None:
            if self.attack != "none" or self.method in ROOT_METHODS:
                values["root_size"] = 80 if self.dataset == "synthetic" else 3000
            else:
                values["root_size"] = 0
        if self.root_pool_size is None:
            values["root_pool_size"] = values["root_size"]
        resolved = Experiment(**values)
        resolved.validate()
        return resolved

    def validate(self):
        if self.dataset not in DATASETS or self.method not in METHODS or self.attack not in ATTACKS:
            raise ValueError("unknown dataset, method, or attack")
        if self.model not in ("cnn", "wrn28-4"):
            raise ValueError("unknown model")
        if self.model == "wrn28-4" and self.dataset not in ("cifar10", "cifar100"):
            raise ValueError("WRN-28-4 requires a CIFAR dataset")
        if not 1 <= self.participants <= self.workers:
            raise ValueError("participants must be between 1 and workers")
        for name in ("rounds", "local_steps", "batch_size", "eval_every", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (math.isfinite(self.beta) and self.beta > 0 and math.isfinite(self.lr) and self.lr > 0):
            raise ValueError("beta and lr must be positive finite numbers")
        for name in ("c", "alpha", "attacker_ratio", "root_corruption"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.attack == "none" and self.attacker_ratio:
            raise ValueError("attacker_ratio requires an attack")
        if not 0 <= self.root_size <= self.root_pool_size:
            raise ValueError("root_size must be between 0 and root_pool_size")
        if self.method in ROOT_METHODS and self.root_size == 0:
            raise ValueError("this method requires a nonempty root dataset")
        if self.root_missing < 0 or self.seed < 0:
            raise ValueError("root_missing and seed must be nonnegative")


class LocalSGD:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = tuple(model.parameters())
        self.buffers = tuple(model.buffers())
        self.config = config
        self.device = device
        self.optimizer = torch.optim.SGD(self.parameters, lr=config.lr)

    @torch.no_grad()
    def copy_parameters(self, vector):
        start = 0
        for parameter in self.parameters:
            end = start + parameter.numel()
            parameter.copy_(vector[start:end].view_as(parameter))
            start = end

    @torch.no_grad()
    def copy_buffers(self, buffers):
        for destination, source in zip(self.buffers, buffers):
            destination.copy_(source)

    def snapshot_buffers(self):
        return tuple(buffer.detach().clone() for buffer in self.buffers)

    def update(self, initial, buffers, data, pool, rng, labels=None, prox=0.0, control=None):
        self.copy_parameters(initial)
        self.copy_buffers(buffers)
        self.model.train()
        first_gradient = None
        for step in range(self.config.local_steps):
            indices = rng.choice(
                pool, self.config.batch_size, replace=len(pool) < self.config.batch_size
            )
            x, y = data.batch(indices, self.device, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(x), y)
            if prox:
                delta = parameters_to_vector(self.parameters) - initial
                loss = loss + (prox / 2) * delta.square().sum()
            loss.backward()
            if control is not None:
                gradients = [parameter.grad for parameter in self.parameters]
                if step == 0:
                    first_gradient = parameters_to_vector(gradients)
                offset = 0
                for gradient in gradients:
                    end = offset + gradient.numel()
                    gradient.add_(control[offset:end].view_as(gradient))
                    offset = end
            self.optimizer.step()
        with torch.no_grad():
            update = parameters_to_vector(self.parameters) - initial
        return update, self.snapshot_buffers(), first_gradient

    @torch.inference_mode()
    def evaluate(self, weights, buffers, data):
        self.copy_parameters(weights)
        self.copy_buffers(buffers)
        self.model.eval()
        loss = correct = 0
        for start in range(0, len(data.labels), 256):
            indices = slice(start, start + 256)
            x, y = data.batch(indices, self.device)
            logits = self.model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            loss += F.cross_entropy(logits, y, reduction="sum").item()
        count = len(data.labels)
        return {"test_accuracy": 100 * correct / count, "test_loss": loss / count}


def report_root_stats(updates, reference, malicious):
    benign = updates[[not flag for flag in malicious]]
    if not len(benign):
        return {}
    norms, root_norm, _, _, cosine = directions(benign, reference)
    valid = norms > 0
    if not valid.any():
        return {}
    rho = root_norm / norms[valid]
    return {
        "benign_rho_min": rho.min().item(),
        "benign_rho_max": rho.max().item(),
        "benign_cos_rho_mean": (rho * cosine[valid]).sum().item() / len(benign),
    }


def rng_for(config, *stream):
    return np.random.default_rng(np.random.SeedSequence((config.seed, *stream)))


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_splits(config, training):
    reserved = class_balanced_pool(
        training.labels, config.root_pool_size, rng_for(config, 1), config.vetted_indices
    )
    available = np.setdiff1d(np.arange(len(training.labels)), reserved)
    partitions = dirichlet_split(training.labels, available, config.workers, config.beta, rng_for(config, 2))
    root = reserved[:config.root_size]
    if config.root_missing:
        missing_classes = np.arange(1, config.root_missing + 1)
        if config.root_missing >= training.classes:
            raise ValueError("cannot remove that many classes")
        root = root[~np.isin(training.labels[root].numpy(), missing_classes)]
    if config.method in ROOT_METHODS and not len(root):
        raise ValueError("no root samples remain after class removal")
    attacker_ids = sorted(rng_for(config, 3).choice(
        config.workers, round(config.workers * config.attacker_ratio), replace=False
    ).tolist())
    return partitions, root, attacker_ids


def run(config, output):
    config = config.resolved()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already contains files: {output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-enabled PyTorch device is available")
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    training, test = load_data(config.dataset, Path(config.data_dir), config.download)
    splits, root, attackers = prepare_splits(config, training)
    root_only = config.method == "root-only"
    attacker_set = set(attackers)
    root_labels = training.labels
    if config.root_corruption:
        root_labels = corrupt_labels(root_labels, root, config.root_corruption,
                                     training.classes, rng_for(config, 4))
    client_labels = training.labels
    if config.attack == "label-flip" and not root_only:
        for number in attackers:
            client_labels = corrupt_labels(client_labels, splits[number], 0.5,
                                           training.classes, rng_for(config, 5, number), reverse=True)

    model = make_model(config.dataset, config.model, training.classes).to(device)
    trainer = LocalSGD(model, config, device)
    weights = parameters_to_vector(trainer.parameters).detach()
    buffers = trainer.snapshot_buffers()
    momentum = global_control = torch.zeros_like(weights)
    controls = {}
    reference = None
    loss_prox = 0.2 if config.method in ("fedprox", "fedacg") else 0.0

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config))
    write_json(output / "split.json", {
        "device": device, "torch": torch.__version__, "numpy": np.__version__,
        "train_examples": len(training.labels), "test_examples": len(test.labels),
        "client_sizes": [len(part) for part in splits], "attacker_ids": attackers,
        "root_labels": np.bincount(training.labels[root].numpy(), minlength=training.classes).tolist(),
    })

    def synchronize():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRICS)
        writer.writeheader()
        last = trainer.evaluate(weights, buffers, test)
        writer.writerow({"round": 0, **last})
        failed_round = None
        for step in range(config.rounds):
            server_seconds = 0.0
            if config.method in ROOT_METHODS:
                synchronize()
                start = perf_counter()
                reference, root_buffers, _ = trainer.update(
                    weights, buffers, training, root, rng_for(config, 7, step), root_labels
                )
                synchronize()
                server_seconds = perf_counter() - start
            if root_only:
                start = perf_counter()
                delta = reference
                buffers = root_buffers
            else:
                selected = rng_for(config, 6, step).choice(
                    config.workers, config.participants, replace=False
                )
                malicious = [int(number) in attacker_set for number in selected]
                if config.method == "drag" and step == 0:
                    reference = torch.stack([
                        trainer.update(weights, buffers, training, splits[number],
                                       rng_for(config, 8, int(number)))[0]
                        for number in selected
                    ]).mean(dim=0)

                base = weights + 0.85 * momentum if config.method == "fedacg" else weights
                updates = torch.empty((config.participants, weights.numel()), device=device)
                client_buffers = []
                control_deltas = []
                synchronize()
                start = perf_counter()
                for index, number in enumerate(selected):
                    control = None
                    if config.method == "scaffold":
                        previous = controls.get(number)
                        control = global_control if previous is None else global_control - previous.to(device)
                    update, local_buffers, first = trainer.update(
                        base, buffers, training, splits[number], rng_for(config, 9, step, int(number)),
                        client_labels, loss_prox, control
                    )
                    updates[index].copy_(update)
                    client_buffers.append(local_buffers)
                    if first is not None:
                        control_deltas.append(first if previous is None else first - previous.to(device))
                        controls[number] = first.cpu()
                synchronize()
                client_seconds = (perf_counter() - start) / config.participants
                updates = perturb(updates, malicious, config.attack, rng_for(config, 10, step))

                synchronize()
                start = perf_counter()
                delta = aggregate(config.method, updates, reference, config.c)
                if config.method == "fedacg":
                    momentum = 0.85 * momentum + delta
                    delta = momentum
                if control_deltas:
                    global_control = global_control + torch.stack(control_deltas).sum(dim=0) / config.workers
                buffers = tuple(
                    torch.stack(values).mean(dim=0) if values[0].is_floating_point()
                    else torch.stack(values).amax(dim=0)
                    for values in zip(*client_buffers)
                )
            weights = weights + delta
            synchronize()
            server_seconds += perf_counter() - start

            if not torch.isfinite(weights).all():
                failed_round = step + 1
                break
            row = {"round": step + 1, "server_seconds": server_seconds}
            if not root_only:
                row.update(
                    malicious_fraction=sum(malicious) / config.participants,
                    client_seconds_mean=client_seconds,
                    server_client_ratio=server_seconds / client_seconds,
                )
                if config.method in ROOT_METHODS:
                    row.update(report_root_stats(updates, reference, malicious))
            if reference is not None:
                row["reference_norm"] = reference.norm().item()
            if config.method == "drag":
                reference = (1 - config.alpha) * reference + config.alpha * delta
            if (step + 1) % config.eval_every == 0 or step + 1 == config.rounds:
                evaluation = trainer.evaluate(weights, buffers, test)
                if not math.isfinite(evaluation["test_loss"]):
                    failed_round = step + 1
                    break
                last = evaluation
                row.update(last)
                print(f"{config.method} | {step + 1:4d}/{config.rounds} | {last['test_accuracy']:6.2f}%", flush=True)
            writer.writerow(row)
            file.flush()
    summary = {"status": "completed", "rounds": config.rounds, **last}
    if failed_round is not None:
        summary.update(status="diverged", rounds=failed_round - 1, failed_round=failed_round)
        print(f"{config.method} | numerical divergence at round {failed_round}", flush=True)
    write_json(output / "complete.json", summary)
    return last


def add_runtime_args(parser):
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--vetted-indices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--method", choices=METHODS, default="drag")
    parser.add_argument("--model", choices=("cnn", "wrn28-4"), default="cnn")
    parser.add_argument("--attack", choices=ATTACKS, default="none")
    for name in ("workers", "participants", "local_steps", "batch_size", "root_size",
                 "root_pool_size", "root_missing", "seed", "beta", "lr", "alpha", "c",
                 "attacker_ratio", "root_corruption"):
        field = next(field for field in fields(Experiment) if field.name == name)
        number_type = int if name in ("root_size", "root_pool_size") else field.type
        if name == "c":
            number_type = float
        parser.add_argument("--" + name.replace("_", "-"), type=number_type, default=field.default)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        run(Experiment(**args), output)
    except (ValueError, FileExistsError, RuntimeError, FloatingPointError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
