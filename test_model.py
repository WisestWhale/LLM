import unittest
from unittest.mock import patch

import torch

from GPT import GPT
from Generate import generate
from SelfAttention import MultiHeadAttention


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(20, 12, 8, 3, 2)

    def test_output_shape_and_feed_forward_gradients(self):
        logits = self.model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(logits.shape, (1, 3, 20))
        logits.square().mean().backward()
        for block in self.model.blocks:
            gradient = block.ff.fc1.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_future_tokens_do_not_change_earlier_predictions(self):
        first = self.model(torch.tensor([[1, 2, 3, 4]]))
        second = self.model(torch.tensor([[1, 2, 9, 8]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_context_limit_and_head_validation(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones((1, 9), dtype=torch.long))
        with self.assertRaises(ValueError):
            MultiHeadAttention(12, 13, 8, 3)

    def test_generation_preserves_prompt_crops_context_and_restores_mode(self):
        prompt = torch.tensor([[1, 2, 3]])
        output = generate(self.model, prompt, 10, 8, top_k=1)
        self.assertEqual(output.shape, (1, 13))
        torch.testing.assert_close(output[:, :3], prompt)
        self.assertTrue(self.model.training)
        again = generate(self.model, prompt, 10, 8, top_k=1)
        torch.testing.assert_close(output, again)

    def test_nucleus_sampling_keeps_threshold_token_for_each_batch_row(self):
        class FixedModel(torch.nn.Module):
            def forward(self, ids):
                probabilities = torch.tensor([[0.1, 0.6, 0.3], [0.3, 0.1, 0.6]])
                return probabilities.log().unsqueeze(1)

        model = FixedModel()
        prompt = torch.tensor([[0], [1]])
        with patch("Generate.torch.multinomial", return_value=torch.tensor([[1], [2]])) as sample:
            generate(model, prompt, 1, 8, top_p=0.8)
            torch.testing.assert_close(sample.call_args.args[0], torch.tensor([
                [0.0, 2 / 3, 1 / 3], [1 / 3, 0.0, 2 / 3],
            ]))
        with patch("Generate.torch.multinomial", return_value=torch.tensor([[1], [2]])) as sample:
            generate(model, prompt, 1, 8, top_p=0.01)
            torch.testing.assert_close(sample.call_args.args[0], torch.tensor([
                [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            ]))

    def test_top_p_one_preserves_seeded_generation(self):
        prompt = torch.tensor([[1, 2]])
        torch.manual_seed(42)
        default = generate(self.model, prompt, 3, 8, top_k=5)
        torch.manual_seed(42)
        explicit = generate(self.model, prompt, 3, 8, top_k=5, top_p=1.0)
        torch.testing.assert_close(default, explicit)

    def test_invalid_generation_settings(self):
        prompt = torch.tensor([[1]])
        for settings in ({"temperature": 0}, {"top_k": 0}, {"max_new_tokens": -1},
                         {"top_p": 0}, {"top_p": 1.1}, {"top_p": float("nan")}):
            kwargs = {"max_new_tokens": 2, "max_length": 8, **settings}
            with self.assertRaises(ValueError):
                generate(self.model, prompt, **kwargs)


if __name__ == "__main__":
    unittest.main()
