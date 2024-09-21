import random
import unittest
import warnings

import torch

from CodonTransformer.CodonData import get_amino_acid_sequence
from CodonTransformer.CodonPrediction import (
    load_model,
    load_tokenizer,
    predict_dna_sequence,
)
from CodonTransformer.CodonUtils import (
    AMINO_ACIDS,
    ORGANISM2ID,
    STOP_SYMBOLS,
    DNASequencePrediction,
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
            ("MKTZZFVLLL?", "Escherichia coli general", "invalid protein sequence"),
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

    def test_predict_dna_sequence_long_protein_sequence(self):
        """Test the function with a very long protein sequence to check performance and correctness."""
        protein_sequence = (
            "M"
            + "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
            * 20
            + STOP_SYMBOLS[0]
        )
        organism = "Escherichia coli general"
        result = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=True,
        )

        # Check that the predicted DNA translates back to the original protein
        dna_sequence = result.predicted_dna[:-3]
        translated_protein = get_amino_acid_sequence(dna_sequence)
        self.assertEqual(
            translated_protein,
            protein_sequence[:-1],
            "Translated protein does not match the original long protein sequence",
        )

    def test_predict_dna_sequence_edge_case_organisms(self):
        """Test the function with organism IDs at the boundaries of the mapping."""
        protein_sequence = "MWWMW"
        # Assuming ORGANISM2ID has IDs starting from 0 to N
        min_organism_id = min(ORGANISM2ID.values())
        max_organism_id = max(ORGANISM2ID.values())
        organisms = [min_organism_id, max_organism_id]

        for organism_id in organisms:
            with self.subTest(organism_id=organism_id):
                result = predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism_id,
                    device=self.device,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    deterministic=True,
                )
                self.assertIsInstance(result.predicted_dna, str)
                self.assertTrue(
                    all(nucleotide in "ATCG" for nucleotide in result.predicted_dna)
                )

    def test_predict_dna_sequence_concurrent_calls(self):
        """Test the function's behavior under concurrent execution."""
        import threading

        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"
        results = []

        def call_predict():
            result = predict_dna_sequence(
                protein=protein_sequence,
                organism=organism,
                device=self.device,
                tokenizer=self.tokenizer,
                model=self.model,
                deterministic=True,
            )
            results.append(result.predicted_dna)

        threads = [threading.Thread(target=call_predict) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(dna == results[0] for dna in results))

    def test_predict_dna_sequence_random_seed_consistency(self):
        """Test that setting a random seed results in consistent outputs in non-deterministic mode."""
        protein_sequence = "MFWY"
        organism = "Escherichia coli general"
        temperature = 0.5
        top_p = 0.95
        torch.manual_seed(42)

        result1 = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=False,
            temperature=temperature,
            top_p=top_p,
        )

        torch.manual_seed(42)

        result2 = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=False,
            temperature=temperature,
            top_p=top_p,
        )

        self.assertEqual(
            result1.predicted_dna,
            result2.predicted_dna,
            "Outputs should be consistent when random seed is set",
        )

    def test_predict_dna_sequence_invalid_tokenizer_and_model(self):
        """Test that providing invalid tokenizer or model raises appropriate exceptions."""
        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"

        with self.subTest("Invalid tokenizer"):
            with self.assertRaises(Exception):
                predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism,
                    device=self.device,
                    tokenizer="invalid_tokenizer_path",
                    model=self.model,
                )

        with self.subTest("Invalid model"):
            with self.assertRaises(Exception):
                predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism,
                    device=self.device,
                    tokenizer=self.tokenizer,
                    model="invalid_model_path",
                )

    def test_predict_dna_sequence_stop_codon_handling(self):
        """Test the function's handling of protein sequences ending with a non '_' or '*' stop symbol."""
        protein_sequence = "MWW/"
        organism = "Escherichia coli general"

        with self.assertRaises(ValueError):
            predict_dna_sequence(
                protein=protein_sequence,
                organism=organism,
                device=self.device,
                tokenizer=self.tokenizer,
                model=self.model,
            )

    def test_predict_dna_sequence_device_compatibility(self):
        """Test that the function works correctly on both CPU and GPU devices."""
        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device):
                result = predict_dna_sequence(
                    protein=protein_sequence,
                    organism=organism,
                    device=device,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    deterministic=True,
                )
                self.assertIsInstance(result.predicted_dna, str)
                self.assertTrue(
                    all(nucleotide in "ATCG" for nucleotide in result.predicted_dna)
                )

    def test_predict_dna_sequence_random_proteins(self):
        """Test random proteins to ensure translated DNA matches the original protein."""
        organism = "Escherichia coli general"
        num_tests = 200

        for _ in range(num_tests):
            # Generate a random protein sequence of random length between 10 and 50
            protein_length = random.randint(10, 500)
            protein_sequence = "M" + "".join(
                random.choices(AMINO_ACIDS, k=protein_length - 1)
            )
            protein_sequence += random.choice(STOP_SYMBOLS)

            result = predict_dna_sequence(
                protein=protein_sequence,
                organism=organism,
                device=self.device,
                tokenizer=self.tokenizer,
                model=self.model,
                deterministic=True,
            )

            # Remove stop codon from predicted DNA
            dna_sequence = result.predicted_dna[:-3]

            # Translate predicted DNA back to protein
            translated_protein = get_amino_acid_sequence(dna_sequence)
            self.assertEqual(
                translated_protein,
                protein_sequence[:-1],  # Remove stop symbol
                f"Translated protein does not match the original protein sequence for protein: {protein_sequence}",
            )

    def test_predict_dna_sequence_long_protein_over_max_length(self):
        """Test that the model handles protein sequences longer than 2048 amino acids."""
        # Create a protein sequence longer than 2048 amino acids
        base_sequence = (
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
        )
        protein_sequence = base_sequence * 100  # Length > 2048 amino acids
        organism = "Escherichia coli general"

        result = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=True,
        )

        # Remove stop codon from predicted DNA
        dna_sequence = result.predicted_dna[:-3]
        translated_protein = get_amino_acid_sequence(dna_sequence)

        # Due to potential model limitations, compare up to the model's max supported length
        max_length = len(translated_protein)
        self.assertEqual(
            translated_protein[:max_length],
            protein_sequence[:max_length],
            "Translated protein does not match the original protein sequence up to the maximum length supported.",
        )

    def test_predict_dna_sequence_multi_output(self):
        """Test that the function returns multiple sequences when num_sequences > 1."""
        protein_sequence = "MFQLLAPWY"
        organism = "Escherichia coli general"
        num_sequences = 20

        result = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=False,
            num_sequences=num_sequences,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_sequences)

        for prediction in result:
            self.assertIsInstance(prediction, DNASequencePrediction)
            self.assertTrue(
                all(nucleotide in "ATCG" for nucleotide in prediction.predicted_dna)
            )

            # Check that all predicted DNA sequences translate back to the original protein
            translated_protein = get_amino_acid_sequence(prediction.predicted_dna[:-3])
            self.assertEqual(translated_protein, protein_sequence)

    def test_predict_dna_sequence_deterministic_multi_raises_error(self):
        """Test that requesting multiple sequences in deterministic mode raises an error."""
        protein_sequence = "MFWY"
        organism = "Escherichia coli general"

        with self.assertRaises(ValueError):
            predict_dna_sequence(
                protein=protein_sequence,
                organism=organism,
                device=self.device,
                tokenizer=self.tokenizer,
                model=self.model,
                deterministic=True,
                num_sequences=3,
            )

    def test_predict_dna_sequence_multi_diversity(self):
        """Test that multiple sequences generated are diverse."""
        protein_sequence = "MFWYMFWY"
        organism = "Escherichia coli general"
        num_sequences = 10

        result = predict_dna_sequence(
            protein=protein_sequence,
            organism=organism,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            deterministic=False,
            num_sequences=num_sequences,
            temperature=0.8,
        )

        unique_sequences = set(prediction.predicted_dna for prediction in result)

        self.assertGreater(
            len(unique_sequences),
            2,
            "Multiple sequence generation should produce diverse results",
        )

        # Check that all sequences are valid translations of the input protein
        for prediction in result:
            translated_protein = get_amino_acid_sequence(prediction.predicted_dna[:-3])
            self.assertEqual(translated_protein, protein_sequence)


if __name__ == "__main__":
    unittest.main()
