"""
File: CodonEvaluation.py
---------------------------
Includes functions to calculate various evaluation metrics along with helper functions.
"""

import numpy as np
import pandas as pd

from CAI import relative_adaptiveness
from Bio.Seq import Seq

from typing import List, Dict, Tuple
from tqdm import tqdm


def get_organism_to_CAI_weights(dataset: pd.DataFrame,
                                organisms: List[str]) -> Dict[str, dict]:
    """
    Return the appropriate weights dictionary for calculating Codon Adaptation Index (CAI)
    for a given list of organisms. Expects dataset dataframe to have columns named "organism" and "dna".
    """
    organism2weights = {}

    for organism in tqdm(organisms, desc="Calculating CAI Weights: ", unit="Organism"):
        organism_data = dataset.loc[dataset["organism"] == organism]
        sequences = organism_data['dna'].to_list()
        weights = relative_adaptiveness(sequences=sequences)
        organism2weights[organism] = weights

    return organism2weights


def get_GC_content(dna: str, lower: bool = False) -> float:
    """
    Return the GC content of given dna sequence, calculated as number of Gs and Cs over length.
    """
    if lower:
        dna = dna.lower()
    return (dna.count("G") + dna.count("C")) / len(dna) * 100


def get_cfd(dna: str,
            codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
            threshold: float = 0.3) -> float:
    """
    Return the codon frequency distribution metric of given input sequence.
    codon_frequencies represent the codon frequency distribution per amino acid of
    the organism to which dna, protein pair belong. You can call the helper function
    named get_codon_frequency_distribution in CodonPrediction.py to create it.
    """
    # Get a dictionary mapping each codon to its normalized frequency
    codon2frequency = {codon: freq / max(frequencies)
                       for amino, (codons, frequencies) in codon_frequencies.items()
                       for codon, freq in zip(codons, frequencies)}

    cfd = 0

    for i in range(0, len(dna), 3):
        codon = dna[i: i+3]
        codon_frequency = codon2frequency[codon]

        if codon_frequency < threshold:
            cfd += 1

    return cfd / (len(dna) / 3) * 100


def get_min_max_percentage(dna: str,
                           codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
                           window_size: int = 18) -> List[float]:
    """
    Get the %MinMax metric for given DNA sequence, using codon_frequencies of the organism.
    Credit: https://github.com/chowington/minmax
    """
    # Get a dictionary mapping each codon to its respective amino acid
    codon2amino = {codon: amino
                   for amino, (codons, frequencies) in codon_frequencies.items()
                   for codon in codons}

    min_max_values = []
    codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]

    for i in range(len(codons) - window_size + 1):
        codon_window = codons[i:i + window_size]        # List of the codons in the current window

        Actual = 0.0    # Average of the actual codon frequencies
        Max = 0.0       # Average of the min codon frequencies
        Min = 0.0       # Average of the max codon frequencies
        Avg = 0.0       # Average of the averages of all the frequencies associated with each amino acid

        # Sum the frequencies
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

        percentMax = ((Actual - Avg) / (Max - Avg)) * 100
        percentMin = ((Avg - Actual) / (Avg - Min)) * 100

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
    Calculates the sequence complexity score based on the provided input dna.
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
            unique_subseq.add(dna[j:(j + i)])
        unique_subseq_length.append(len(unique_subseq))

    # Calculate complexity score
    complexity_score = (sum(unique_subseq_length) / (sum_up_to(len(dna) - 1) + f(len(dna)))) * 100

    return complexity_score


def get_sequence_similarity(original: str,
                            predicted: str,
                            truncate: bool = True,
                            window_length: int = 1) -> float:
    """
    Return the percentage of amino acids in common between two protein sequences or
    the percentage of nucleotides in common between two DNA sequences.

    If window_length is set to 3, it will compare triplets (e.g. codons in DNA) to calculate identity.

    We expect len(predicted) <= len(original).
    """
    if not truncate and len(original) != len(predicted):
        raise ValueError('Set truncate to True if the length of sequences do not match.')

    identity = 0.0
    original = original.strip()
    predicted = predicted.strip()

    if truncate:
        original = original[:len(predicted)]

    if window_length == 1:
        # Simple comparison for single characters
        for i in range(len(predicted)):
            if original[i] == predicted[i]:
                identity += 1
    else:
        # Comparison for substrings based on window_length
        for i in range(0, len(original) - window_length + 1, window_length):
            if original[i:i + window_length] == predicted[i:i + window_length]:
                identity += 1

    return (identity / (len(predicted) / window_length)) * 100


