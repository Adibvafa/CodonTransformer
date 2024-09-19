import os
import pickle
import tempfile
import unittest

from CodonTransformer.CodonUtils import (
    find_pattern_in_fasta,
    get_organism2id_dict,
    get_taxonomy_id,
    load_pkl_from_url,
    load_python_object_from_disk,
    save_python_object_to_disk,
    sort_amino2codon_skeleton,
)


class TestCodonUtils(unittest.TestCase):
    def test_load_python_object_from_disk(self):
        test_obj = {"key1": "value1", "key2": 2}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(test_obj, temp_file_name)
        loaded_obj = load_python_object_from_disk(temp_file_name)
        self.assertEqual(test_obj, loaded_obj)
        os.remove(temp_file_name)

    def test_save_python_object_to_disk(self):
        test_obj = [1, 2, 3, 4, 5]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(test_obj, temp_file_name)
            self.assertTrue(os.path.exists(temp_file_name))
        os.remove(temp_file_name)

    def test_find_pattern_in_fasta(self):
        text = (
            ">seq1 [keyword=value1]\nATGCGTACGTAGCTAG\n"
            ">seq2 [keyword=value2]\nGGTACGATCGATCGAT"
        )
        self.assertEqual(find_pattern_in_fasta("keyword", text), "value1")
        self.assertEqual(find_pattern_in_fasta("nonexistent", text), "")

    def test_get_organism2id_dict(self):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=True, suffix=".csv"
        ) as temp_file:
            temp_file.write("0,Escherichia coli\n1,Homo sapiens\n2,Mus musculus")
            temp_file.flush()
            organism2id = get_organism2id_dict(temp_file.name)
            self.assertEqual(
                organism2id,
                {"Escherichia coli": 0, "Homo sapiens": 1, "Mus musculus": 2},
            )

    def test_get_taxonomy_id(self):
        taxonomy_dict = {
            "Escherichia coli": 562,
            "Homo sapiens": 9606,
            "Mus musculus": 10090,
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as temp_file:
            temp_file_name = temp_file.name
            save_python_object_to_disk(taxonomy_dict, temp_file_name)
            self.assertEqual(get_taxonomy_id(temp_file_name, "Escherichia coli"), 562)
            self.assertEqual(
                get_taxonomy_id(temp_file_name, return_dict=True), taxonomy_dict
            )

    def test_sort_amino2codon_skeleton(self):
        amino2codon = {
            "A": (["GCT", "GCC", "GCA", "GCG"], [0.0, 0.0, 0.0, 0.0]),
            "C": (["TGT", "TGC"], [0.0, 0.0]),
        }
        sorted_amino2codon = sort_amino2codon_skeleton(amino2codon)
        self.assertEqual(
            sorted_amino2codon,
            {
                "A": (["GCA", "GCC", "GCG", "GCT"], [0.0, 0.0, 0.0, 0.0]),
                "C": (["TGC", "TGT"], [0.0, 0.0]),
            },
        )

    def test_load_pkl_from_url(self):
        url = "https://example.com/test.pkl"
        expected_obj = {"key": "value"}
        with unittest.mock.patch("requests.get") as mock_get:
            mock_get.return_value.content = pickle.dumps(expected_obj)
            loaded_obj = load_pkl_from_url(url)
        self.assertEqual(loaded_obj, expected_obj)


if __name__ == "__main__":
    unittest.main()
