"""CPU-only tests for the timestep-EOT scheduling and logging helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from attackers.text_grad_eot_utils import (  # noqa: E402
    evaluation_steps,
    make_timestep_schedule,
    split_timestep_strata,
)
from utils.plot_attack_loss import _loss_series  # noqa: E402


class TimestepScheduleTests(unittest.TestCase):
    def test_fifty_timesteps_split_17_17_16(self):
        strata = split_timestep_strata(range(10, 1000, 20))
        self.assertEqual([len(group) for group in strata], [17, 17, 16])
        self.assertLess(max(strata[0]), min(strata[1]))
        self.assertLess(max(strata[1]), min(strata[2]))
        self.assertEqual(sum((list(group) for group in strata), []), list(range(10, 1000, 20)))

    def test_schedule_is_seeded_and_covers_all_strata(self):
        sampled = range(10, 1000, 20)
        first = make_timestep_schedule(sampled, 40, seed=123)
        second = make_timestep_schedule(sampled, 40, seed=123)
        self.assertEqual(first, second)
        strata = split_timestep_strata(sampled)
        for update in first:
            self.assertEqual(len(update), 3)
            self.assertIn(update[0], strata[0])
            self.assertIn(update[1], strata[1])
            self.assertIn(update[2], strata[2])

    def test_schedule_rejects_non_three_eot(self):
        with self.assertRaises(ValueError):
            make_timestep_schedule(range(10, 1000, 20), 1, timesteps_per_update=2)

    def test_evaluation_checkpoints_include_final_step(self):
        self.assertEqual(evaluation_steps(400, 10), tuple(range(10, 401, 10)))
        self.assertEqual(evaluation_steps(7, 10), (7,))


class PlotCompatibilityTests(unittest.TestCase):
    def test_eot_loss_window_expands_to_optimizer_steps(self):
        records = [
            {"success": False},
            {
                "record_type": "evaluation",
                "optimization_step": 10,
                "loss_window": [
                    {"optimization_step": 1, "mean_loss": 0.9},
                    {"optimization_step": 2, "mean_loss": 0.8},
                ],
            },
            {
                "record_type": "evaluation",
                "optimization_step": 20,
                "loss_window": [
                    {"optimization_step": 3, "mean_loss": 0.7},
                    {"optimization_step": 2, "mean_loss": 0.81},
                ],
            },
        ]
        losses, x_values, mode = _loss_series(records)
        self.assertEqual(x_values, [1, 2, 3])
        self.assertEqual(losses, [0.9, 0.81, 0.7])
        self.assertEqual(mode, "EOT optimizer step")

    def test_baseline_scalar_loss_stays_compatible(self):
        records = [{"success": False}, {"loss": 1.2}, {"loss": 0.8}]
        losses, x_values, mode = _loss_series(records)
        self.assertEqual(losses, [1.2, 0.8])
        self.assertEqual(x_values, [0, 1])
        self.assertIn("Outer", mode)


class ConfigTests(unittest.TestCase):
    def test_eot_config_isolated_from_baseline(self):
        eot_path = REPO_ROOT / "configs/nudity/text_grad_eot_esd_nudity_classifier.json"
        baseline_path = REPO_ROOT / "configs/nudity/text_grad_esd_nudity_classifier.json"
        eot = json.loads(eot_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual(eot["overall"]["attacker"], "text_grad_eot")
        self.assertEqual(eot["attacker"]["iteration"], 400)
        self.assertEqual(eot["attacker"]["text_grad_eot"]["eval_interval"], 10)
        self.assertEqual(baseline["overall"]["attacker"], "text_grad")
        self.assertNotEqual(
            eot["logger"]["json"]["root"], baseline["logger"]["json"]["root"]
        )


if __name__ == "__main__":
    unittest.main()
