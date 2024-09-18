import tempfile
import unittest
import pandas as pd
from CodonTransformer.CodonData import (
    read_fasta_file,
    build_amino2codon_skeleton,
    get_amino_acid_sequence,
    is_correct_seq,
)
from Bio.Data.CodonTable import TranslationError


class TestCodonData(unittest.TestCase):
    def test_read_fasta_file(self):
        fasta_content = ">sequence1\n" "ATGATGATGATGATG\n" ">sequence2\n" "TGATGATGATGA"

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".fasta"
        ) as temp_file:
            temp_file.write(fasta_content)
            temp_file_name = temp_file.name

        try:
            sequences = read_fasta_file(temp_file_name, save_to_file=None)
            self.assertIsInstance(sequences, pd.DataFrame)
            self.assertEqual(len(sequences), 2)
            self.assertEqual(sequences.iloc[0]["dna"], "ATGATGATGATGATG")
            self.assertEqual(sequences.iloc[1]["dna"], "TGATGATGATGA")
        finally:
            import os

            os.unlink(temp_file_name)

    def test_build_amino2codon_skeleton(self):
        organism = "Homo sapiens"
        codon_skeleton = build_amino2codon_skeleton(organism)

        expected_amino_acids = "ARNDCQEGHILKMFPSTWYV_"

        for amino_acid in expected_amino_acids:
            self.assertIn(amino_acid, codon_skeleton)
            codons, frequencies = codon_skeleton[amino_acid]
            self.assertIsInstance(codons, list)
            self.assertIsInstance(frequencies, list)
            self.assertEqual(len(codons), len(frequencies))
            self.assertTrue(all(isinstance(codon, str) for codon in codons))
            self.assertTrue(all(freq == 0 for freq in frequencies))

        all_codons = set(
            codon for codons, _ in codon_skeleton.values() for codon in codons
        )
        self.assertEqual(len(all_codons), 64)  # There should be 64 unique codons

    def test_get_amino_acid_sequence(self):
        dna = "ATGGCCTGA"
        protein, is_correct = get_amino_acid_sequence(dna, return_correct_seq=True)
        self.assertEqual(protein, "MA_")
        self.assertTrue(is_correct)

    def test_is_correct_seq(self):
        dna = "ATGGCCTGA"
        protein = "MA_"
        self.assertTrue(is_correct_seq(dna, protein))

    def test_read_fasta_file_raises_exception_for_non_dna(self):
        non_dna_content = ">sequence1\nATGATGATGXYZATG\n>sequence2\nTGATGATGATGA"

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".fasta"
        ) as temp_file:
            temp_file.write(non_dna_content)
            temp_file_name = temp_file.name

        try:
            with self.assertRaises(TranslationError) as context:
                read_fasta_file(temp_file_name)
            self.assertIn("Codon 'XYZ' is invalid", str(context.exception))
        finally:
            import os

            os.unlink(temp_file_name)


if __name__ == "__main__":
    unittest.main()
