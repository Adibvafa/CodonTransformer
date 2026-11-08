import unittest
from unittest.mock import Mock, patch

import torch

import finetune
import pretrain
from CodonTransformer.CodonUtils import MAX_LEN, TOKEN2MASK


class TestTrainingCollators(unittest.TestCase):
    def check_collator(self, module, random_matches_original=False):
        original = torch.tensor(
            [
                [1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 2, 3, 0, 4],
                [1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 2, 3, 0, 4],
            ]
        )
        attention_mask = (original != 3).long()
        tokenizer = Mock(
            return_value={
                "input_ids": original.clone(),
                "attention_mask": attention_mask.clone(),
                "token_type_ids": torch.zeros_like(original),
            }
        )
        examples = [
            {"codons": "first sequence", "organism": 11},
            {"codons": "second sequence", "organism": 42},
        ]

        # Ten codons per row are selected, with one eligible codon left out.
        selection_draws = torch.full(original.shape, 0.05)
        selection_draws[:, 11] = 0.2
        # Eight of those ten codons are masked; two remain for random/keep.
        mask_draws = torch.full(original.shape, 0.1)
        mask_draws[:, 9:11] = 0.9
        # A draw of 0.3 distinguishes the required 0.5 from the old 0.1.
        random_draws = torch.full(original.shape, 0.3)
        random_draws[:, 10] = 0.7
        draws = iter([selection_draws, mask_draws, random_draws])

        def bernoulli(probabilities):
            return (next(draws) < probabilities).float()

        random_tokens = (
            original.clone()
            if random_matches_original
            else torch.full_like(original, 89)
        )
        with (
            patch.object(module.torch, "bernoulli", side_effect=bernoulli) as sample,
            patch.object(
                module.torch, "randint", return_value=random_tokens
            ) as randint,
        ):
            result = module.MaskedTokenizerCollator(tokenizer)(examples)

        expected = original.clone()
        expected[:, 1:9] = torch.tensor(
            [[TOKEN2MASK[token.item()] for token in row[1:9]] for row in original]
        )
        expected[:, 9] = random_tokens[:, 9]
        torch.testing.assert_close(result["input_ids"], expected)

        labels = torch.full_like(original, -100)
        labels[:, 1:11] = original[:, 1:11]
        torch.testing.assert_close(result["labels"], labels)
        torch.testing.assert_close(result["attention_mask"], attention_mask)
        torch.testing.assert_close(
            result["token_type_ids"],
            torch.tensor([[11], [42]]).repeat(1, original.shape[1]),
        )
        tokenizer.assert_called_once_with(
            [example["codons"] for example in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.assertEqual(sample.call_count, 3)
        selection_probabilities = torch.full(original.shape, 0.15)
        selection_probabilities[original < 5] = 0.0
        for call, probabilities in zip(
            sample.call_args_list,
            [
                selection_probabilities,
                torch.full(original.shape, 0.8),
                torch.full(original.shape, 0.5),
            ],
        ):
            torch.testing.assert_close(call.args[0], probabilities)
        randint.assert_called_once_with(26, 90, original.shape, dtype=torch.long)

    def test_mask_random_keep_branches(self):
        for module in (pretrain, finetune):
            with self.subTest(script=module.__name__):
                self.check_collator(module)

    def test_random_replacement_can_match_original(self):
        # Randomized branch membership need not visibly change the input token.
        for module in (pretrain, finetune):
            with self.subTest(script=module.__name__):
                self.check_collator(module, random_matches_original=True)


if __name__ == "__main__":
    unittest.main()
