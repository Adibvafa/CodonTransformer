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

    def test_predict_dna_sequence_top_p_effect(self):
        """Test that changing top_p affects the diversity of outputs."""
        protein_sequence = "MFWY"
        organism = "Escherichia coli general"
        num_iterations = 50
        temperature = 0.5
        top_p_values = [0.8, 0.95]
        outputs_by_top_p = {top_p: set() for top_p in top_p_values}

        for top_p in top_p_values:
            for _ in range(num_iterations):
                result = predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism,
                    device=self.device,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    deterministic=False,
                    temperature=temperature,
                    top_p=top_p,
                )
                outputs_by_top_p[top_p].add(
                    result.predicted_dna[:-3]
                )  # Remove stop codon

        # Assert that higher top_p results in more diverse outputs
        diversity_lower_top_p = len(outputs_by_top_p[0.8])
        diversity_higher_top_p = len(outputs_by_top_p[0.95])
        self.assertGreaterEqual(
            diversity_higher_top_p,
            diversity_lower_top_p,
            "Higher top_p should result in more diverse outputs",
        )

    def test_predict_dna_sequence_invalid_temperature_and_top_p(self):
        """Test that invalid temperature and top_p values raise ValueError."""
        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"
        invalid_params = [
            {"temperature": -0.1, "top_p": 0.95},
            {"temperature": 0, "top_p": 0.95},
            {"temperature": 0.5, "top_p": -0.1},
            {"temperature": 0.5, "top_p": 1.1},
        ]

        for params in invalid_params:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    predict_dna_sequence(
                        protein=protein_sequence,
                        organism=organism,
                        device=self.device,
                        tokenizer=self.tokenizer,
                        model=self.model,
                        deterministic=False,
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                    )

    def test_predict_dna_sequence_translation_consistency(self):
        """Test that the predicted DNA translates back to the original protein."""
        from CodonTransformer.CodonData import get_amino_acid_sequence

        protein_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVE"
        organism = "Escherichia coli general"
        result = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=True,
        )

        # Translate predicted DNA back to protein
        translated_protein = get_amino_acid_sequence(result.predicted_dna[:-3])

        self.assertEqual(
            translated_protein,
            protein_sequence,
            "Translated protein does not match the original protein sequence",
        )


if __name__ == "__main__":
    unittest.main()
