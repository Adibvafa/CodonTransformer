"""
File: CodonEvaluation.py
---------------------------
Includes functions to calculate various evaluation metrics along with helper functions.
"""

import pandas as pd

from CAI import CAI, relative_adaptiveness

from typing import List, Dict, Tuple
from tqdm import tqdm

from CodonTransformer.CodonUtils import AMINO2CODON_TYPE
from CodonTransformer.CodonData import build_amino2codon_skeleton, get_codon_frequencies


def get_CSI_weights(sequences: List[str]) -> Dict[str, float]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of DNA sequences.

    Args:
        sequences (List[str]): List of DNA sequences.

    Returns:
        dict: The CSI weights.
    """
    return relative_adaptiveness(sequences=sequences)


def get_CSI_value(dna: str, weights: Dict[str, float]) -> float:
    """
    Calculate the Codon Similarity Index (CSI) for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        weights (dict): The CSI weights from get_CSI_weights.

    Returns:
        float: The CSI value.
    """
    return CAI(dna, weights)


def get_organism_to_CSI_weights(
    dataset: pd.DataFrame, organisms: List[str]
) -> Dict[str, dict]:
    """
    Calculate the Codon Similarity Index (CSI) weights for a list of organisms.

    Args:
        dataset (pd.DataFrame): The dataset containing organism and DNA sequence information.
        organisms (List[str]): List of organism names.

    Returns:
        Dict[str, dict]: A dictionary mapping each organism to its CSI weights.
    """
    organism2weights = {}

    # Iterate through each organism to calculate its CSI weights
    for organism in tqdm(organisms, desc="Calculating CSI Weights: ", unit="Organism"):
        organism_data = dataset.loc[dataset["organism"] == organism]
        sequences = organism_data["dna"].to_list()
        weights = get_CSI_weights(sequences)
        organism2weights[organism] = weights

    return organism2weights


def get_GC_content(dna: str, lower: bool = False) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        lower (bool): If True, converts DNA sequence to lowercase before calculation.

    Returns:
        float: The GC content as a percentage.
    """
    if lower:
        dna = dna.lower()
    return (dna.count("G") + dna.count("C")) / len(dna) * 100


def get_cfd(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    threshold: float = 0.3,
) -> float:
    """
    Calculate the codon frequency distribution (CFD) metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon frequency distribution per amino acid.
        threshold (float): Frequency threshold for counting rare codons.

    Returns:
        float: The CFD metric as a percentage.
    """
    # Get a dictionary mapping each codon to its normalized frequency
    codon2frequency = {
        codon: freq / max(frequencies)
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon, freq in zip(codons, frequencies)
    }

    cfd = 0

    # Iterate through the DNA sequence in steps of 3 to process each codon
    for i in range(0, len(dna), 3):
        codon = dna[i : i + 3]
        codon_frequency = codon2frequency[codon]

        if codon_frequency < threshold:
            cfd += 1

    return cfd / (len(dna) / 3) * 100


def get_min_max_percentage(
    dna: str,
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
    window_size: int = 18,
) -> List[float]:
    """
    Calculate the %MinMax metric for a DNA sequence.

    Args:
        dna (str): The DNA sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon frequency distribution per amino acid.
        window_size (int): Size of the window to calculate %MinMax.

    Returns:
        List[float]: List of %MinMax values for the sequence.

    Credit: https://github.com/chowington/minmax
    """
    # Get a dictionary mapping each codon to its respective amino acid
    codon2amino = {
        codon: amino
        for amino, (codons, frequencies) in codon_frequencies.items()
        for codon in codons
    }

    min_max_values = []
    codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]  # Split DNA into codons

    # Iterate through the DNA sequence using the specified window size
    for i in range(len(codons) - window_size + 1):
        codon_window = codons[
            i : i + window_size
        ]  # List of the codons in the current window

        Actual = 0.0  # Average of the actual codon frequencies
        Max = 0.0  # Average of the min codon frequencies
        Min = 0.0  # Average of the max codon frequencies
        Avg = 0.0  # Average of the averages of all the frequencies associated with each amino acid

        # Sum the frequencies for codons in the current window
        for codon in codon_window:
            aminoacid = codon2amino[codon]
            frequencies = codon_frequencies[aminoacid][1]
            codon_index = codon_frequencies[aminoacid][0].index(codon)
            codon_frequency = codon_frequencies[aminoacid][1][codon_index]

            Actual += codon_frequency
            Max += max(frequencies)
            Min += min(frequencies)
            Avg += sum(frequencies) / len(frequencies)

        # Divide by the window size to get the averages
        Actual = Actual / window_size
        Max = Max / window_size
        Min = Min / window_size
        Avg = Avg / window_size

        # Calculate %MinMax
        percentMax = ((Actual - Avg) / (Max - Avg)) * 100
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100

        # Append the appropriate %MinMax value
        if percentMax >= 0:
            min_max_values.append(percentMax)
        else:
            min_max_values.append(-percentMin)

    # Populate the last floor(window_size / 2) entries of min_max_values with None
    for i in range(int(window_size / 2)):
        min_max_values.append(None)

    return min_max_values


def get_sequence_complexity(dna: str) -> float:
    """
    Calculate the sequence complexity score of a DNA sequence.

    Args:
        dna (str): The DNA sequence.

    Returns:
        float: The sequence complexity score.
    """

    def sum_up_to(x):
        """Recursive function to calculate the sum of integers from 1 to x."""
        if x <= 1:
            return 1
        else:
            return x + sum_up_to(x - 1)

    def f(x):
        """Function that returns 4 if x is greater than or equal to 4, else returns x."""
        if x >= 4:
            return 4
        elif x < 4:
            return x

    unique_subseq_length = []

    # Calculate unique subsequences lengths
    for i in range(1, len(dna) + 1):
        unique_subseq = set()
        for j in range(len(dna) - (i - 1)):
            unique_subseq.add(dna[j : (j + i)])
        unique_subseq_length.append(len(unique_subseq))

    # Calculate complexity score
    complexity_score = (
        sum(unique_subseq_length) / (sum_up_to(len(dna) - 1) + f(len(dna)))
    ) * 100

    return complexity_score


def get_sequence_similarity(
    original: str, predicted: str, truncate: bool = True, window_length: int = 1
) -> float:
    """
    Calculate the sequence similarity between two sequences.

    Args:
        original (str): The original sequence.
        predicted (str): The predicted sequence.
        truncate (bool): If True, truncate the original sequence to match the length of the predicted sequence.
        window_length (int): Length of the window for comparison (1 for amino acids, 3 for codons).

    Returns:
        float: The sequence similarity as a percentage.

    Preconditions:
        len(predicted) <= len(original).
    """
    if not truncate and len(original) != len(predicted):
        raise ValueError(
            "Set truncate to True if the length of sequences do not match."
        )

    identity = 0.0
    original = original.strip()
    predicted = predicted.strip()

    if truncate:
        original = original[: len(predicted)]

    if window_length == 1:
        # Simple comparison for amino acid
        for i in range(len(predicted)):
            if original[i] == predicted[i]:
                identity += 1
    else:
        # Comparison for substrings based on window_length
        for i in range(0, len(original) - window_length + 1, window_length):
            if original[i : i + window_length] == predicted[i : i + window_length]:
                identity += 1

    return (identity / (len(predicted) / window_length)) * 100
