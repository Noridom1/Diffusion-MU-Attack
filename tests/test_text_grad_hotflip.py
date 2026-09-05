"""CPU tests for the discrete HotFlip utilities and config wiring."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The repository's lightweight test environment may omit torch.
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

if torch is not None:
    from attackers.hotflip_utils import (  # noqa: E402
        make_candidate_ids,
        replacement_scores,
        top_replacements,
    )


@unittest.skipIf(torch is None, 'PyTorch is required for HotFlip tensor tests')
class HotFlipUtilityTests(unittest.TestCase):
    def test_replacement_scores_and_global_topk(self):
        grad = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        vocab = torch.tensor([
            [0.0, 0.0],
            [-2.0, 0.0],
            [0.0, -1.0],
            [4.0, 4.0],
        ])
        scores = replacement_scores(grad, vocab)
        positions, token_ids, values = top_replacements(
            scores,
            torch.tensor([0, 1]),
            torch.ones_like(scores, dtype=torch.bool),
            topk=2,
        )
        self.assertEqual(list(zip(positions.tolist(), token_ids.tolist())), [(0, 1), (1, 2)])
        self.assertTrue(torch.allclose(values, torch.tensor([-2.0, -1.0])))

    def test_current_and_forbidden_tokens_are_not_proposed(self):
        scores = torch.zeros((2, 4))
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, 3] = False
        positions, token_ids, _ = top_replacements(
            scores, torch.tensor([0, 1]), mask, topk=10
        )
        self.assertNotIn(0, [int(token_ids[i]) for i in range(len(token_ids)) if int(positions[i]) == 0])
        self.assertNotIn(1, [int(token_ids[i]) for i in range(len(token_ids)) if int(positions[i]) == 1])
        self.assertNotIn(3, token_ids.tolist())

    def test_candidate_ids_change_one_position(self):
        current = torch.tensor([10, 11, 12])
        candidates = make_candidate_ids(
            current, torch.tensor([2, 0]), torch.tensor([99, 88])
        )
        self.assertEqual(candidates.tolist(), [[10, 11, 99], [88, 11, 12]])


class HotFlipConfigTests(unittest.TestCase):
    def test_only_requested_baseline_failures_are_in_scope(self):
        config_path = REPO_ROOT / 'configs/nudity/text_grad_hotflip_esd_nudity_classifier.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        self.assertEqual(config['overall']['attacker'], 'text_grad_hotflip')
        self.assertEqual(config['attacker']['text_grad_hotflip']['candidate_topk'], 8)
        self.assertEqual(config['attacker']['attack_idx'], 7)
        self.assertEqual(config['attacker']['text_grad_hotflip']['max_timesteps'], 50)


if __name__ == '__main__':
    unittest.main()
