import unittest
import warnings

import torch

from CodonTransformer.CodonPrediction import (
    load_model,
    load_tokenizer,
    predict_dna_sequence,
)


class TestCodonPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Suppress warnings about loading from HuggingFace
        for message in [
            "Tokenizer path not provided. Loading from HuggingFace.",
            "Model path not provided. Loading from HuggingFace.",
        ]:
            warnings.filterwarnings("ignore", message=message)

        cls.model = load_model(device=cls.device)
        cls.tokenizer = load_tokenizer()

    def test_predict_dna_sequence_valid_input(self):
        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"
        result = predict_dna_sequence(
            protein_sequence,
            organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.assertIsInstance(result.predicted_dna, str)
        self.assertTrue(
            all(nucleotide in "ATCG" for nucleotide in result.predicted_dna)
        )
        self.assertEqual(result.predicted_dna, "ATGTGGTGGATGTGGTGA")

    def test_predict_dna_sequence_non_deterministic(self):
        protein_sequence = "MFWY"
        organism = "Escherichia coli general"
        num_iterations = 100
        temperatures = [0.2, 0.5, 0.8]
        possible_outputs = set()
        possible_encodings_wo_stop = {
            "ATGTTTTGGTAT",
            "ATGTTCTGGTAT",
            "ATGTTTTGGTAC",
            "ATGTTCTGGTAC",
        }

        for _ in range(num_iterations):
            for temperature in temperatures:
                result = predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism,
                    device=self.device,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    deterministic=False,
                    temperature=temperature,
                )
                possible_outputs.add(result.predicted_dna[:-3])  # Remove stop codon

        self.assertEqual(possible_outputs, possible_encodings_wo_stop)

    def test_predict_dna_sequence_invalid_inputs(self):
        test_cases = [
            ("MKTZZFVLLL", "Escherichia coli general", "invalid protein sequence"),
            ("MKTFFVLLL", "Alien $%#@!", "invalid organism code"),
            ("", "Escherichia coli general", "empty protein sequence"),
        ]

        for protein_sequence, organism, error_type in test_cases:
            with self.subTest(error_type=error_type):
                with self.assertRaises(ValueError):
                    predict_dna_sequence(
                        protein_sequence,
                        organism,
                        device=self.device,
                        tokenizer=self.tokenizer,
                        model=self.model,
                    )


if __name__ == "__main__":
    unittest.main()
