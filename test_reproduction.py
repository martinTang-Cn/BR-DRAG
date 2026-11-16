import csv
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from aggregation import aggregate, distance_stress_update, divergence_correction, perturb
from data import load_data
from models import make_model
from reproduce import figure_cases
from train import Experiment, prepare_splits, run


class AggregationTest(unittest.TestCase):
    def test_drag_remaps_orthogonal_and_opposed_updates(self):
        updates = torch.tensor([[0., 2.], [-3., 0.], [0., 0.]])
        root = torch.tensor([1., 0.])
        expected = torch.tensor([[0.5, 1.5], [-0., 0.], [0., 0.]])
        result = divergence_correction(updates, root, scale=0.25, robust=False)
        torch.testing.assert_close(result, expected)

    def test_br_drag_uses_root_magnitude_and_corrects_opposite(self):
        root = torch.tensor([3., 0.])
        updates = torch.tensor([[0., 200.], [-80., 0.], [0., 0.]])
        expected = torch.tensor([[1.5, 1.5], [3., 0.], [0., 0.]])
        actual = divergence_correction(updates, root, scale=0.5, robust=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aggregate("br-drag", updates, root), expected.mean(dim=0))

    def test_zero_reference_and_zero_trust(self):
        vectors = torch.tensor([[-5., 0.], [0., 0.]])
        zero = torch.zeros(2)
        torch.testing.assert_close(aggregate("drag", vectors, zero), vectors.mean(dim=0))
        torch.testing.assert_close(aggregate("br-drag", vectors, zero), zero)
        torch.testing.assert_close(aggregate("fltrust", vectors, torch.tensor([1., 0.])), zero)

    def test_fltrust_denominator_and_ablation(self):
        vectors = torch.tensor([[2., 0.], [0., -1.]])
        root = torch.tensor([4., 0.])
        torch.testing.assert_close(aggregate("fltrust", vectors, root), root)
        torch.testing.assert_close(aggregate("br-trust", vectors, root), root / 2)

    def test_geometric_medians_resist_single_outlier(self):
        vectors = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [500., 0.]])
        for method in ("rfa", "raga"):
            with self.subTest(method=method):
                self.assertLess(aggregate(method, vectors).norm().item(), 5)

    def test_distance_perturbations_satisfy_constraints(self):
        honest = torch.tensor([[2., 0.], [3., 1.], [4., -2.]], dtype=torch.float64)
        pairwise = torch.cdist(honest, honest)
        for family in ("min-max", "min-sum"):
            with self.subTest(family=family):
                candidate = distance_stress_update(honest, family)
                distances = (honest - candidate).norm(dim=1)
                actual = distances.max() if family == "min-max" else distances.sum()
                limit = pairwise.max() if family == "min-max" else pairwise.sum(dim=1).max()
                torch.testing.assert_close(actual, limit)
        updates = torch.randn(4, 2)
        self.assertEqual(perturb(updates, [True] * 4, "min-max", np.random.default_rng(0)).norm(), 0)


class ExperimentTest(unittest.TestCase):
    def test_data_split_is_disjoint_and_comparable_across_methods(self):
        train, test = load_data("synthetic", None, False)
        config = Experiment(dataset="synthetic", method="br-drag", attack="sign-flip",
                            attacker_ratio=0.3, root_size=80).resolved()
        clients, root, attackers = prepare_splits(config, train)
        control = Experiment(**{**vars(config), "method": "fedavg"})
        clients_again, root_again, attackers_again = prepare_splits(control, train)
        self.assertEqual(len(attackers), 12)
        self.assertEqual(attackers, attackers_again)
        np.testing.assert_array_equal(root, root_again)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(clients, clients_again)))
        np.testing.assert_array_equal(np.sort(np.concatenate([*clients, root])), np.arange(len(train.labels)))
        np.testing.assert_array_equal(np.bincount(train.labels[root]), [8] * 10)
        self.assertEqual(len(test.labels), 200)

    def test_input_output_shapes(self):
        torch.set_num_threads(2)
        cases = (
            ("emnist", "cnn", 47, 1, 28), ("cifar10", "cnn", 10, 3, 32),
            ("cifar100", "cnn", 100, 3, 32), ("cifar100", "wrn28-4", 100, 3, 32),
        )
        with torch.inference_mode():
            for dataset, architecture, classes, channels, side in cases:
                with self.subTest(dataset=dataset, architecture=architecture):
                    model = make_model(dataset, architecture, classes).eval()
                    self.assertEqual(model(torch.zeros(2, channels, side, side)).shape, (2, classes))

    def test_figure_presets_have_paper_budgets(self):
        for figure in range(3, 23):
            for case in figure_cases(str(figure)):
                config = case.config.resolved()
                self.assertEqual(config.rounds, 3000 if figure == 3 else 4000)
                self.assertEqual(config.workers, 40)
                self.assertEqual(config.local_steps, 5)
                self.assertEqual(config.batch_size, 10)

    def test_br_drag_root_only_pairs_share_experimental_settings(self):
        cases = figure_cases("br-drag-root-only")
        self.assertEqual([case.config.beta for case in cases], [0.1, 0.1, 0.5, 0.5])
        for baseline, ablation in zip(cases[::2], cases[1::2]):
            config = baseline.config.resolved()
            self.assertEqual(config.method, "br-drag")
            self.assertEqual((config.root_size, config.root_corruption), (3000, 0.0))
            self.assertEqual(ablation.config.resolved(), replace(config, method="root-only"))

    @patch("train.perturb")
    @patch("train.aggregate")
    @patch("train.make_model")
    @patch("train.LocalSGD")
    def test_root_only_applies_server_updates_without_client_aggregation(
        self, mock_sgd, _mock_model, mock_aggregate, mock_perturb,
    ):
        config = Experiment(dataset="synthetic", method="root-only", attack="sign-flip",
                            attacker_ratio=0.3, rounds=2, eval_every=1, device="cpu")
        trainer = mock_sgd.return_value
        trainer.parameters = (torch.tensor([2., 4.]),)
        trainer.snapshot_buffers.return_value = (torch.tensor(0),)
        trainer.update.side_effect = [
            (torch.tensor([1., -3.]), (torch.tensor(1),), None),
            (torch.tensor([-2., 2.]), (torch.tensor(2),), None),
        ]
        trainer.evaluate.return_value = {"test_accuracy": 10.0, "test_loss": 2.0}

        with TemporaryDirectory() as output:
            run(config, output)
            with (Path(output) / "metrics.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(trainer.update.call_count, config.rounds)
        mock_aggregate.assert_not_called()
        mock_perturb.assert_not_called()
        first_update, second_update = trainer.update.call_args_list
        self.assertEqual(len(first_update.args[3]), 80)
        np.testing.assert_array_equal(first_update.args[3], second_update.args[3])
        torch.testing.assert_close(second_update.args[0], torch.tensor([3., 1.]))
        torch.testing.assert_close(second_update.args[1], (torch.tensor(1),))
        last_eval = trainer.evaluate.call_args
        torch.testing.assert_close(last_eval.args[0], torch.tensor([1., 3.]))
        torch.testing.assert_close(last_eval.args[1], (torch.tensor(2),))
        self.assertTrue(all(not row["client_seconds_mean"] for row in rows))


if __name__ == "__main__":
    unittest.main()
