"""Run the paper experiments and the BR-DRAG root-only comparison."""

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from train import METHODS, Experiment, add_runtime_args, run


CLEAN = ("drag", "fedavg", "fedprox", "scaffold", "fedexp", "fedacg")
ROBUST = ("br-drag", "fedavg", "fltrust", "rfa", "raga")
LABEL = dict(zip(METHODS, (
    "DRAG", "BR-DRAG", "FedAvg", "FedProx", "SCAFFOLD", "FedExP", "FedACG",
    "FLTrust", "RFA", "RAGA", "BR-DRAG (clipped cosine)", "Server root only",
)))


@dataclass(frozen=True)
class Case:
    panel: str
    label: str
    config: Experiment
    sweep: float = 0.0


def comparison(base, methods):
    return [
        Case(f"beta={beta:g}", LABEL[method], replace(base, method=method, beta=beta))
        for beta in (0.1, 0.5) for method in methods
    ]


def figure_cases(figure):
    if figure == "br-drag-root-only":
        return comparison(Experiment(attack="sign-flip", attacker_ratio=0.3), ("br-drag", "root-only"))
    if figure == "smoke":
        return [
            Case("Clean clients" if attack == "none" else "Sign flipping (60% of clients)",
                 LABEL[method], Experiment(
                dataset="synthetic", attack=attack, method=method, attacker_ratio=ratio,
                root_size=80, rounds=60, eval_every=5,
            ))
            for attack, ratio, methods in (
                ("none", 0.0, ("drag", "fedavg")),
                ("sign-flip", 0.6, ("br-drag", "fedavg", "fltrust")),
            )
            for method in methods
        ]
    if figure in ("3", "4", "5", "6"):
        dataset = {"3": "emnist", "4": "cifar10", "5": "cifar100", "6": "cifar100"}[figure]
        architecture = "wrn28-4" if figure == "6" else "cnn"
        return comparison(Experiment(dataset=dataset, model=architecture), CLEAN)
    if figure in ("7", "8", "9"):
        parameter, values = {
            "7": ("participants", (5, 15, 25, 35)),
            "8": ("alpha", (0.01, 0.05, 0.25, 0.45, 0.65, 0.85)),
            "9": ("c", (0.05, 0.25, 0.45, 0.65, 0.85)),
        }[figure]
        return [
            Case(f"beta={beta:g}", f"{parameter}={value:g}",
                 replace(Experiment(beta=beta), **{parameter: value}))
            for beta in (0.1, 0.5) for value in values
        ]
    if figure == "10":
        return comparison(Experiment(dataset="cifar100", method="br-drag"), ("br-drag",))
    if figure in {str(number) for number in range(11, 20)}:
        attack = {
            "11": "noise", "12": "sign-flip", "13": "label-flip", "14": "sign-flip",
            "15": "min-max", "16": "min-sum", "17": "noise", "18": "sign-flip", "19": "label-flip",
        }[figure]
        dataset = "cifar100" if figure in ("13", "14", "15", "16") else "cifar10"
        ratio = 0.6 if int(figure) >= 17 else 0.3
        model = "wrn28-4" if figure == "14" else "cnn"
        return comparison(Experiment(dataset=dataset, model=model, attack=attack, attacker_ratio=ratio), ROBUST)
    if figure in ("20", "22"):
        attack = "noise" if figure == "20" else "sign-flip"
        other = "fltrust" if figure == "20" else "br-trust"
        return [
            Case(dataset, LABEL[method], Experiment(
                dataset=dataset, attack=attack, attacker_ratio=0.3, method=method
            ))
            for dataset in ("cifar10", "cifar100") for method in ("br-drag", other)
        ]
    if figure == "21":
        base = Experiment(attack="noise", attacker_ratio=0.3, root_pool_size=6000)
        sweeps = (
            ("Root size", "root_size", (2000, 3000, 4000, 5000, 6000)),
            ("Corrupted labels (%)", "root_corruption", (0, 0.025, 0.05, 0.075, 0.10)),
            ("Missing classes", "root_missing", (0, 1, 2, 3)),
        )
        return [
            Case(panel, LABEL[method], replace(base, method=method, **{parameter: value}),
                 100 * value if parameter == "root_corruption" else value)
            for panel, parameter, values in sweeps for value in values for method in ("br-drag", "fltrust")
        ]
    raise ValueError(f"unknown figure {figure}")


def figure_name(figure):
    return f"fig{int(figure):02d}" if figure.isdecimal() else figure


def read_series(directory, key):
    points = []
    if (directory / "complete.json").exists():
        with (directory / "metrics.csv").open(encoding="utf-8", newline="") as file:
            points = [(int(row["round"]), float(row[key])) for row in csv.DictReader(file) if row[key]]
    if not points:
        raise ValueError(f"no completed {key} series in {directory}")
    return np.asarray(points)


def is_diverged(directory):
    with (directory / "complete.json").open(encoding="utf-8") as file:
        return json.load(file).get("status") == "diverged"


def combine_series(directories, key):
    series = [read_series(directory, key) for directory in directories]
    length = min(len(values) for values in series)
    series = [values[:length] for values in series]
    rounds = series[0][:, 0]
    if not all(np.array_equal(values[:, 0], rounds) for values in series):
        raise ValueError(f"rounds differ across seeds for {key}")
    scores = np.stack([values[:, 1] for values in series])
    return rounds, scores.mean(axis=0), scores.std(axis=0)


def plot(figure, records, folder):
    # Keep the renderer local to the experiment directory.
    os.environ.setdefault("MPLCONFIGDIR", str((folder.parent / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = defaultdict(list)
    for case, directories in records:
        panels[case.panel].append((case, directories))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 3.8), squeeze=False,
                             layout="constrained")
    if figure == "smoke":
        fig.suptitle("Synthetic classification · pipeline check")
    else:
        datasets = ", ".join(dict.fromkeys(case.config.dataset.upper() for case, _ in records))
        rounds = read_series(records[0][1][0], "test_accuracy")[-1, 0]
        experiment = records[0][0].config
        subject = "BR-DRAG vs. server root only" if figure == "br-drag-root-only" else f"Figure {figure}"
        title = f"{subject} · {datasets} · {experiment.model.upper()}"
        if experiment.attacker_ratio:
            title += f" · {experiment.attack} ({experiment.attacker_ratio:.0%})"
        if rounds < experiment.resolved().rounds and not is_diverged(records[0][1][0]):
            title += f" · {int(rounds)}-round run"
        fig.suptitle(title)
    for axis, (panel, entries) in zip(axes[0], panels.items()):
        axis.set_title(panel)
        axis.grid(alpha=0.2)
        if figure == "21":
            curves = defaultdict(list)
            for case, directories in entries:
                mean, std = np.nan, np.nan
                if not any(map(is_diverged, directories)):
                    _, means, stds = combine_series(directories, "test_accuracy")
                    mean, std = means[-1], stds[-1]
                curves[case.label].append((case.sweep, mean, std))
            for label, points in curves.items():
                x, y, spread = np.asarray(points).T
                axis.errorbar(x, y, yerr=spread, marker="o", capsize=2, label=label)
            axis.set(xlabel=panel, ylabel="Final test accuracy (%)")
        else:
            key = "server_client_ratio" if figure == "20" else "test_accuracy"
            axis.set(xlabel="Communication rounds",
                     ylabel="Server time / mean client time" if figure == "20" else "Test accuracy (%)")
            for case, directories in entries:
                traces = [(key, case.label)]
                if figure == "10":
                    traces = [("benign_rho_min", "Min. norm ratio"),
                              ("benign_rho_max", "Max. norm ratio"),
                              ("benign_cos_rho_mean", "Mean cosine × norm ratio")]
                    axis.set_ylabel("Benign update diagnostics")
                for metric, label in traces:
                    rounds, means, stds = combine_series(directories, metric)
                    diverged = sum(map(is_diverged, directories))
                    if diverged:
                        label += f" (diverged {diverged}/{len(directories)})"
                    line, = axis.plot(rounds, means, label=label)
                    if diverged:
                        axis.scatter(rounds[-1], means[-1], color=line.get_color(), marker="x")
                    if len(directories) > 1:
                        axis.fill_between(rounds, means - stds, means + stds,
                                          color=line.get_color(), alpha=0.15, linewidth=0)
        axis.legend(fontsize=8)
    path = folder / f"{figure_name(figure)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Plot: {path}", flush=True)


def case_config(case, args, seed):
    settings = {key: getattr(args, key) for key in (
        "data_dir", "download", "device", "threads", "vetted_indices",
    )}
    if args.rounds is not None:
        settings["rounds"] = args.rounds
    if args.eval_every is not None:
        settings["eval_every"] = args.eval_every
    return replace(case.config, seed=seed, **settings).resolved()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.set_defaults(eval_every=None)
    parser.add_argument(
        "--figure", nargs="+", required=True,
        choices=["all", "smoke", "br-drag-root-only", *(str(number) for number in range(3, 23))],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    figures = list(dict.fromkeys(
        figure for selection in args.figure
        for figure in (map(str, range(3, 23)) if selection == "all" else (selection,))
    ))
    seeds = list(dict.fromkeys(args.seeds))
    for number in figures:
        cases = figure_cases(number)
        folder = args.output / figure_name(number)
        configs = [[case_config(case, args, seed) for seed in seeds] for case in cases]
        if args.plan:
            total = sum(config.rounds for group in configs for config in group)
            print(f"{folder.name}: {len(cases) * len(seeds)} runs, {total} communication rounds")
            continue
        records = []
        for index, (case, group) in enumerate(zip(cases, configs), 1):
            directories = []
            for config in group:
                directory = folder / f"{index:02d}-{config.method}-seed{config.seed}"
                print(f"{folder.name} | {case.panel} | {case.label} | seed {config.seed}", flush=True)
                if not args.plot_only:
                    if (directory / "complete.json").exists():
                        with (directory / "config.json").open(encoding="utf-8") as file:
                            previous = json.load(file)
                        current = vars(config)
                        if any(previous[key] != value for key, value in current.items() if key != "download"):
                            parser.exit(1, f"Run configuration differs in {directory}. Use a different output directory.\n")
                    else:
                        run(config, directory)
                directories.append(directory)
            records.append((case, directories))
        plot(number, records, folder)


if __name__ == "__main__":
    main()
