import math
import unittest

import torch
import torch.nn.functional as F

from Evaluate import evaluate
from GPT import GPT


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(20, 12, 8, 3, 2)

    def test_matches_direct_loss_and_restores_mode(self):
        ids = [1, 2, 3, 4, 5, 6]
        self.model.train()
        loss, perplexity, tokens = evaluate(self.model, ids, 8)
        self.model.eval()
        logits = self.model(torch.tensor([ids[:-1]]))
        expected = F.cross_entropy(logits.flatten(0, 1), torch.tensor(ids[1:])).item()
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertAlmostEqual(perplexity, math.exp(expected), places=4)
        self.assertEqual(tokens, 5)
        evaluate(self.model.train(), ids, 8)
        self.assertTrue(self.model.training)

    def test_windows_cover_every_token_once_and_can_be_limited(self):
        ids = list(range(1, 20))
        self.assertEqual(evaluate(self.model, ids, 8)[2], 18)
        self.assertEqual(evaluate(self.model, ids, 8, max_windows=1)[2], 8)

    def test_rejects_too_short_text(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, [1], 8)


if __name__ == "__main__":
    unittest.main()
