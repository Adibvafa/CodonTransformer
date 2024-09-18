import unittest
import torch
from CodonTransformer.CodonPrediction import (
    predict_dna_sequence,
    # add other imported functions or classes as needed
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCodonPrediction(unittest.TestCase):
    def test_predict_dna_sequence_valid_input(self):
        # Test predict_dna_sequence with a valid protein sequence and organism code
        protein_sequence = "MWWMW"
        organism = "Escherichia coli general"
        result = predict_dna_sequence(protein_sequence, organism, device)
        # Test if the output is a string and contains only A, T, C, G.
        self.assertIsInstance(result.predicted_dna, str)
        self.assertEqual(result.predicted_dna, "ATGTGGTGGATGTGGTGA")

    def test_predict_dna_sequence_invalid_protein_sequence(self):
        # Test predict_dna_sequence with an invalid protein sequence
        protein_sequence = "MKTZZFVLLL"  # 'Z' is not a valid amino acid
        organism = "Escherichia coli general"
        with self.assertRaises(ValueError):
            predict_dna_sequence(protein_sequence, organism, device)

    def test_predict_dna_sequence_invalid_organism_code(self):
        # Test predict_dna_sequence with an invalid organism code
        protein_sequence = "MKTFFVLLL"
        organism = "Alien $%#@!"
        with self.assertRaises(ValueError):
            predict_dna_sequence(protein_sequence, organism, device)

    def test_predict_dna_sequence_empty_protein_sequence(self):
        # Test predict_dna_sequence with an empty protein sequence
        protein_sequence = ""
        organism = "Escherichia coli general"
        with self.assertRaises(ValueError):
            predict_dna_sequence(protein_sequence, organism, device)

    def test_predict_dna_sequence_none_protein_sequence(self):
        # Test predict_dna_sequence with None as protein sequence
        protein_sequence = None
        organism = "Escherichia coli general"
        with self.assertRaises(ValueError):
            predict_dna_sequence(protein_sequence, organism, device)


if __name__ == "__main__":
    unittest.main()
