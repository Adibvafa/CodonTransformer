"""
File: CodonUtils.py
---------------------
Includes constants and helper functions used by other Python scripts.
"""

import os
import re
import pickle
import requests
import itertools

import torch
import pandas as pd

from typing import Any, List, Dict, Tuple, Optional, Iterator


# Token-to-index mapping for amino acids and special tokens
TOKEN2INDEX: Dict[str, int] = {
    "[UNK]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[PAD]": 3,
    "[MASK]": 4,
    "a_unk": 5,
    "c_unk": 6,
    "d_unk": 7,
    "e_unk": 8,
    "f_unk": 9,
    "g_unk": 10,
    "h_unk": 11,
    "i_unk": 12,
    "k_unk": 13,
    "l_unk": 14,
    "m_unk": 15,
    "n_unk": 16,
    "p_unk": 17,
    "q_unk": 18,
    "r_unk": 19,
    "s_unk": 20,
    "t_unk": 21,
    "v_unk": 22,
    "w_unk": 23,
    "y_unk": 24,
    "__unk": 25,
    "k_aaa": 26,
    "n_aac": 27,
    "k_aag": 28,
    "n_aat": 29,
    "t_aca": 30,
    "t_acc": 31,
    "t_acg": 32,
    "t_act": 33,
    "r_aga": 34,
    "s_agc": 35,
    "r_agg": 36,
    "s_agt": 37,
    "i_ata": 38,
    "i_atc": 39,
    "m_atg": 40,
    "i_att": 41,
    "q_caa": 42,
    "h_cac": 43,
    "q_cag": 44,
    "h_cat": 45,
    "p_cca": 46,
    "p_ccc": 47,
    "p_ccg": 48,
    "p_cct": 49,
    "r_cga": 50,
    "r_cgc": 51,
    "r_cgg": 52,
    "r_cgt": 53,
    "l_cta": 54,
    "l_ctc": 55,
    "l_ctg": 56,
    "l_ctt": 57,
    "e_gaa": 58,
    "d_gac": 59,
    "e_gag": 60,
    "d_gat": 61,
    "a_gca": 62,
    "a_gcc": 63,
    "a_gcg": 64,
    "a_gct": 65,
    "g_gga": 66,
    "g_ggc": 67,
    "g_ggg": 68,
    "g_ggt": 69,
    "v_gta": 70,
    "v_gtc": 71,
    "v_gtg": 72,
    "v_gtt": 73,
    "__taa": 74,
    "y_tac": 75,
    "__tag": 76,
    "y_tat": 77,
    "s_tca": 78,
    "s_tcc": 79,
    "s_tcg": 80,
    "s_tct": 81,
    "__tga": 82,
    "c_tgc": 83,
    "w_tgg": 84,
    "c_tgt": 85,
    "l_tta": 86,
    "f_ttc": 87,
    "l_ttg": 88,
    "f_ttt": 89,
}

# Index-to-token mapping, reverse of TOKEN2INDEX
INDEX2TOKEN: Dict[int, str] = {i: c for c, i in TOKEN2INDEX.items()}

# Mask token mapping
TOKEN2MASK: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 13,
    27: 16,
    28: 13,
    29: 16,
    30: 21,
    31: 21,
    32: 21,
    33: 21,
    34: 19,
    35: 20,
    36: 19,
    37: 20,
    38: 12,
    39: 12,
    40: 15,
    41: 12,
    42: 18,
    43: 11,
    44: 18,
    45: 11,
    46: 17,
    47: 17,
    48: 17,
    49: 17,
    50: 19,
    51: 19,
    52: 19,
    53: 19,
    54: 14,
    55: 14,
    56: 14,
    57: 14,
    58: 8,
    59: 7,
    60: 8,
    61: 7,
    62: 5,
    63: 5,
    64: 5,
    65: 5,
    66: 10,
    67: 10,
    68: 10,
    69: 10,
    70: 22,
    71: 22,
    72: 22,
    73: 22,
    74: 25,
    75: 24,
    76: 25,
    77: 24,
    78: 20,
    79: 20,
    80: 20,
    81: 20,
    82: 25,
    83: 6,
    84: 23,
    85: 6,
    86: 14,
    87: 9,
    88: 14,
    89: 9,
}

# List of organisms used for fine-tuning
FINE_TUNE_ORGANISMS: List[str] = [
    "Arabidopsis thaliana",
    "Bacillus subtilis",
    "Caenorhabditis elegans",
    "Chlamydomonas reinhardtii",
    "Chlamydomonas reinhardtii chloroplast",
    "Danio rerio",
    "Drosophila melanogaster",
    "Homo sapiens",
    "Mus musculus",
    "Nicotiana tabacum",
    "Nicotiana tabacum chloroplast",
    "Pseudomonas putida",
    "Saccharomyces cerevisiae",
    "Escherichia coli O157-H7 str. Sakai",
    "Escherichia coli general",
    "Escherichia coli str. K-12 substr. MG1655",
    "Thermococcus barophilus MPT",
]

# Type alias for amino acid to codon mapping
AMINO2CODON_TYPE = Dict[str, Tuple[List[str], List[float]]]

# Constants for the number of organisms and sequence lengths
NUM_ORGANISMS = 164
MAX_LEN = 2048
MAX_AMINO_ACIDS = MAX_LEN - 2  # Without special tokens [CLS] and [SEP]
STOP_SYMBOL = "_"


class IterableData(torch.utils.data.IterableDataset):
    """
    Defines the logic for iterable datasets (working over streams of
    data) in parallel multi-processing environments, e.g., multi-GPU.

    Args:
        dist_env (Optional[str]): The distribution environment identifier (e.g., "slurm").

    Credit: Guilluame Filion
    """

    def __init__(self, dist_env: Optional[str] = None):
        super().__init__()
        self.world_size_handle, self.rank_handle = {
            "slurm": ("SLURM_NTASKS", "SLURM_PROCID")
        }.get(dist_env, ("WORLD_SIZE", "LOCAL_RANK"))

    @property
    def iterator(self) -> Iterator:
        """Define the stream logic for the dataset. Should be implemented in subclasses."""
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        """
        Create an iterator for the dataset, handling multi-processing contexts.

        Returns:
            Iterator: The iterator for the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.iterator

        # In multi-processing context, use 'os.environ' to
        # find global worker rank. Then use 'islice' to allocate
        # the items of the stream to the workers.
        world_size = int(os.environ.get(self.world_size_handle))
        global_rank = int(os.environ.get(self.rank_handle))
        local_rank = worker_info.id
        local_num_workers = worker_info.num_workers

        # Assume that each process has the same number of local workers.
        worker_rk = global_rank * local_num_workers + local_rank
        worker_nb = world_size * local_num_workers
        return itertools.islice(self.iterator, worker_rk, None, worker_nb)


class IterableJSONData(IterableData):
    """
    Iterate over the lines of a JSON file and uncompress if needed.

    Args:
        data_path (str): The path to the JSON data file.
        train (bool): Flag indicating if the dataset is for training.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(self, data_path: str, train: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.train = train


def load_python_object_from_disk(file_path: str) -> Any:
    """
    Load a Pickle object from disk and return it as a Python object.

    Args:
        file_path (str): The path to the Pickle file.

    Returns:
        Any: The loaded Python object.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_python_object_to_disk(input_object: Any, file_path: str) -> None:
    """
    Save a Python object to disk using Pickle.

    Args:
        input_object (Any): The Python object to save.
        file_path (str): The path where the object will be saved.
    """
    with open(file_path, "wb") as file:
        pickle.dump(input_object, file)


def find_pattern_in_fasta(keyword: str, text: str) -> str:
    """
    Find a specific keyword pattern in text. Helpful for identifying parts of a FASTA sequence.

    Args:
        keyword (str): The keyword pattern to search for.
        text (str): The text to search within.

    Returns:
        str: The found pattern or an empty string if not found.
    """
    # Search for the keyword pattern in the text using regex
    result = re.search(keyword + r"=(.*?)]", text)
    return result.group(1) if result else ""


def get_organism2id_dict(organism_reference: str) -> Dict[str, int]:
    """
    Return a dictionary mapping each organism in training data to an index used for training.

    Args:
        organism_reference (str): Path to a CSV file containing a list of all organisms.

    Returns:
        Dict[str, int]: A dictionary mapping organism names to their respective indices.
    """
    # Read the CSV file and create a dictionary mapping organisms to their indices
    organisms = pd.read_csv(organism_reference, index_col=0, header=None)
    organism2id = {organisms.iloc[i].values[0]: i for i in organisms.index}

    return organism2id


def get_taxonomy_id(
    taxonomy_reference: str, organism: Optional[str] = None, return_dict: bool = False
) -> Any:
    """
    Return the taxonomy id of a given organism using a reference file.
    Optionally, return the whole dictionary instead if return_dict is True.

    Args:
        taxonomy_reference (str): Path to the taxonomy reference file.
        organism (Optional[str]): The name of the organism to look up.
        return_dict (bool): Whether to return the entire dictionary.

    Returns:
        Any: The taxonomy id of the organism or the entire dictionary.
    """
    # Load the organism-to-taxonomy mapping from a Pickle file
    organism2taxonomy = load_python_object_from_disk(taxonomy_reference)

    if return_dict:
        return dict(sorted(organism2taxonomy.items()))

    return organism2taxonomy[organism]


def sort_amino2codon_skeleton(amino2codon: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort the amino2codon dictionary alphabetically by amino acid and by codon name.

    Args:
        amino2codon (Dict[str, Any]): The amino2codon dictionary to sort.

    Returns:
        Dict[str, Any]: The sorted amino2codon dictionary.
    """
    # Sort the dictionary by amino acid and then by codon name
    amino2codon = dict(sorted(amino2codon.items()))
    amino2codon = {
        amino: (
            [codon for codon, _ in sorted(zip(codons, frequencies))],
            [freq for _, freq in sorted(zip(codons, frequencies))],
        )
        for amino, (codons, frequencies) in amino2codon.items()
    }

    return amino2codon


def load_pkl_from_url(url: str) -> Any:
    """
    Download a Pickle file from a URL and return the loaded object.

    Args:
        url (str): The URL to download the Pickle file from.

    Returns:
        Any: The loaded Python object from the Pickle file.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Load the Pickle object from the response content
    return pickle.loads(response.content)
