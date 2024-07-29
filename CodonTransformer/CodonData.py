"""
File: CodonData.py
---------------------
Includes helper functions for preprocessing NCBI or Kazusa databases and
preparing the data for training and inference of the CodonTransformer model.
"""

import os
import json
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle

from CodonTransformer.CodonUtils import (
    AMINO_ACIDS,
    START_CODONS,
    STOP_CODONS,
    STOP_SYMBOL,
    AMINO2CODON_TYPE,
    AMBIGUOUS_AMINOACID_MAP,
    ORGANISM2ID,
    find_pattern_in_fasta,
    sort_amino2codon_skeleton,
    get_taxonomy_id,
)

from Bio import SeqIO
from Bio.Seq import Seq

import python_codon_tables as pct

from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm


def prepare_training_data(
    dataset: Union[str, pd.DataFrame], output_file: str, shuffle: bool = True
) -> None:
    """
    Prepare a JSON dataset for training the CodonTransformer model.

    Input dataset should have columns below:
        - dna: str (DNA sequence)
        - protein: str (Protein sequence)
        - organism: Union[int, str] (ID or Name of the organism)

    The output JSON dataset will have the following format:
        {"idx": 0, "codons": "M_ATG R_AGG L_TTG L_CTA R_CGA __TAG", "organism": 51}
        {"idx": 1, "codons": "M_ATG K_AAG C_TGC F_TTT F_TTC __TAA", "organism": 59}

    Args:
        dataset (Union[str, pd.DataFrame]): Input dataset in CSV or DataFrame format.
        output_file (str): Path to save the output JSON dataset.
        shuffle (bool, optional): Whether to shuffle the dataset before saving. Defaults to True.

    Returns:
        None
    """
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    required_columns = {"dna", "protein", "organism"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Input dataset must have columns: {required_columns}")

    # Prepare the dataset for finetuning
    dataset["codons"] = dataset.apply(
        lambda row: get_merged_seq(row["protein"], row["dna"], separator="_"), axis=1
    )

    # Replace organism str with organism id using ORGANISM2ID
    dataset["organism"] = dataset["organism"].apply(
        lambda org: process_organism(org, ORGANISM2ID)
    )

    # Save the dataset to a JSON file
    dataframe_to_json(dataset[["codons", "organism"]], output_file, shuffle=shuffle)


def dataframe_to_json(df: pd.DataFrame, output_file: str, shuffle: bool = True) -> None:
    """
    Convert a pandas DataFrame to a JSON file format suitable for training CodonTransformer.

    This function takes a preprocessed DataFrame and writes it to a JSON file
    where each line is a JSON object representing a single record.

    Args:
        df (pd.DataFrame): The input DataFrame with 'codons' and 'organism' columns.
        output_file (str): Path to the output JSON file.
        shuffle (bool, optional): Whether to shuffle the dataset before saving. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = {"codons", "organism"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    print(f"\nStarted writing to {output_file}...")

    # Shuffle the DataFrame if requested
    if shuffle:
        df = sk_shuffle(df)

    # Write the DataFrame to a JSON file
    with open(output_file, "w") as f:
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Writing JSON...", unit=" records"
        ):
            doc = {"idx": idx, "codons": row["codons"], "organism": row["organism"]}
            f.write(json.dumps(doc) + "\n")

    print(f"\nTotal Entries Saved: {len(df)}, JSON data saved to {output_file}")


def process_organism(organism: Union[str, int], organism_to_id: Dict[str, int]) -> int:
    """
    Process and validate the organism input, converting it to a valid organism ID.

    This function handles both string (organism name) and integer (organism ID) inputs.
    It validates the input against a provided mapping of organism names to IDs.

    Args:
        organism (Union[str, int]): The input organism, either as a name (str) or ID (int).
        organism_to_id (Dict[str, int]): A dictionary mapping organism names to their corresponding IDs.

    Returns:
        int: The validated organism ID.

    Raises:
        ValueError: If the input is an invalid organism name or ID.
        TypeError: If the input is neither a string nor an integer.
    """
    if isinstance(organism, str):
        if organism not in organism_to_id:
            raise ValueError(f"Invalid organism name: {organism}")
        return organism_to_id[organism]

    elif isinstance(organism, int):
        if organism not in organism_to_id.values():
            raise ValueError(f"Invalid organism ID: {organism}")
        return organism

    raise TypeError(
        f"Organism must be a string or integer, not {type(organism).__name__}"
    )


def preprocess_protein_sequence(protein: str) -> str:
    """
    Preprocess a protein sequence by cleaning, standardizing, and handling ambiguous amino acids.

    Args:
        protein (str): The input protein sequence.

    Returns:
        str: The preprocessed protein sequence.

    Raises:
        ValueError: If the protein sequence is invalid.
    """
    if not protein:
        raise ValueError("Protein sequence is empty.")

    # Clean and standardize the protein sequence
    protein = (
        protein.upper().strip().replace("\n", "").replace(" ", "").replace("\t", "")
    )

    # Replace ambiguous amino acids with standard 20 amino acids
    protein = "".join(
        AMBIGUOUS_AMINOACID_MAP.get(aminoacid, aminoacid) for aminoacid in protein
    )

    # Check for sequence validity
    if any(
        aminoacid not in AMINO_ACIDS + ["*", STOP_SYMBOL] for aminoacid in protein[:-1]
    ):
        raise ValueError("Invalid characters in protein sequence.")

    if protein[-1] not in AMINO_ACIDS + ["*", STOP_SYMBOL]:
        raise ValueError("Protein sequence must end with *, or _, or an amino acid.")

    # Replace '*' at the end of protein with STOP_SYMBOL if present
    if protein[-1] == "*":
        protein = protein[:-1] + STOP_SYMBOL

    # Add stop symbol to end of protein
    if protein[-1] != STOP_SYMBOL:
        protein += STOP_SYMBOL

    return protein


def replace_ambiguous_codons(dna: str) -> str:
    """
    Replaces ambiguous codons in a DNA sequence with "UNK".

    Args:
        dna (str): The DNA sequence to process.

    Returns:
        str: The processed DNA sequence with ambiguous codons replaced by "UNK".
    """
    result = []
    dna = dna.upper()

    # Check codons in DNA sequence
    for i in range(0, len(dna), 3):
        codon = dna[i : i + 3]

        if len(codon) == 3 and all(nucleotide in "ATCG" for nucleotide in codon):
            result.append(codon)
        else:
            result.append("UNK")

    return "".join(result)


def preprocess_dna_sequence(dna: str) -> str:
    """
    Cleans and preprocesses a DNA sequence by standardizing it and replacing ambiguous codons.

    Args:
        dna (str): The DNA sequence to preprocess.

    Returns:
        str: The cleaned and preprocessed DNA sequence.
    """
    if not dna:
        return ""

    # Clean and standardize the DNA sequence
    dna = dna.upper().strip().replace("\n", "").replace(" ", "").replace("\t", "")

    # Replace codons with ambigous nucleotides with "UNK"
    dna = replace_ambiguous_codons(dna)

    # Add unkown stop codon to end of DNA sequence if not present
    if dna[-3:] not in STOP_CODONS + ["UNK"]:
        dna += "UNK"

    return dna


def get_merged_seq(protein: str, dna: str = "", separator: str = "_") -> str:
    """
    Return the merged sequence of protein amino acids and DNA codons in the form of tokens
    separated by space, where each token is composed of an amino acid + separator + codon.

    Args:
        protein (str): Protein sequence.
        dna (str): DNA sequence.
        separator (str): Separator between amino acid and codon.

    Returns:
        str: Merged sequence.

    Example:
        >>> get_merged_seq(protein="MAV_", dna="ATGGCTGTGTAA", separator="_")
        'M_ATG A_GCT V_GTG __TAA'

        >>> get_merged_seq(protein="QHH_", dna="", separator="_")
        'Q_UNK H_UNK H_UNK __UNK'
    """
    merged_seq = ""

    # Prepare protein and dna sequences
    dna = preprocess_dna_sequence(dna)
    protein = preprocess_protein_sequence(protein)

    # Check if the length of protein and dna sequences are equal
    if len(dna) > 0 and len(protein) != len(dna) / 3:
        raise ValueError(
            'Length of protein (including stop symbol such as "_") and \
                         the number of codons in DNA sequence (including stop codon) must be equal.'
        )

    # Merge protein and DNA sequences into tokens
    for i, aminoacid in enumerate(protein):
        merged_seq += f'{aminoacid}{separator}{dna[i * 3:i * 3 + 3] if dna else "UNK"} '

    return merged_seq.strip()


def is_correct_seq(dna: str, protein: str, stop_symbol: str = STOP_SYMBOL) -> bool:
    """
    Check if the given DNA and protein pair is correct, that is:
        1. The length of dna is divisible by 3
        2. There is an initiator codon in the beginning of dna
        3. There is only one stop codon in the sequence
        4. The only stop codon is the last codon

    Note since in Codon Table 3, 'TGA' is interpreted as Triptophan (W),
    there is a separate check to make sure those sequences are considered correct.

    Args:
        dna (str): DNA sequence.
        protein (str): Protein sequence.
        stop_symbol (str): Stop symbol.

    Returns:
        bool: True if the sequence is correct, False otherwise.
    """
    return (
        len(dna) % 3 == 0  # Check if DNA length is divisible by 3
        and dna[:3].upper() in START_CODONS  # Check for initiator codon
        and protein[-1]
        == stop_symbol  # Check if the last protein symbol is the stop symbol
        and protein.count(stop_symbol) == 1  # Check if there is only one stop symbol
        and len(set(dna))
        == 4  # Check if DNA consists of 4 unique nucleotides (A, T, C, G)
    )


def get_amino_acid_sequence(
    dna: str,
    stop_symbol: str = "_",
    codon_table: int = 1,
    return_correct_seq: bool = True,
) -> Union[Tuple[str, bool], str]:
    """
    Return the translated protein sequence given a DNA sequence and codon table.

    Args:
        dna (str): DNA sequence.
        stop_symbol (str): Stop symbol.
        codon_table (int): Codon table number.
        return_correct_seq (bool): Whether to return if the sequence is correct.

    Returns:
        Union[Tuple[str, bool], str]: Protein sequence and correctness flag if return_correct_seq is True,
                                      otherwise just the protein sequence.
    """
    dna_seq = Seq(dna).strip()

    # Translate the DNA sequence to a protein sequence
    protein_seq = str(
        dna_seq.translate(
            stop_symbol=stop_symbol,  # Symbol to use for stop codons
            to_stop=False,  # Translate the entire sequence, including any stop codons
            cds=False,  # Do not assume the input is a coding sequence
            table=codon_table,  # Codon table to use for translation
        )
    ).strip()

    correct_seq = is_correct_seq(dna_seq, protein_seq, stop_symbol)

    return (protein_seq, correct_seq) if return_correct_seq else protein_seq


def read_fasta_file(
    input_file: str,
    output_path: str,
    organism: str = "",
    return_dataframe: bool = True,
    buffer_size: int = 50000,
) -> pd.DataFrame:
    """
    Read a FASTA file of DNA sequences and save it to a Pandas DataFrame.

    Args:
        input_file (str): Path to the input FASTA file.
        output_path (str): Path to save the output DataFrame.
        organism (str): Name of the organism.
        return_dataframe (bool): Whether to return the DataFrame.
        buffer_size (int): Buffer size for reading the file.

    Returns:
        pd.DataFrame: DataFrame containing the DNA sequences.
    """
    buffer = []
    columns = [
        "dna",
        "protein",
        "correct_seq",
        "organism",
        "GeneID",
        "description",
        "tokenized",
    ]

    # Read the FASTA file and process each sequence record
    with open(input_file, "r") as fasta_file:
        for record in tqdm(
            SeqIO.parse(fasta_file, "fasta"), desc=f"{organism}", unit=" Rows"
        ):
            dna = str(record.seq).strip()

            # Determine the organism from the record if not provided
            if not organism:
                organism = find_pattern_in_fasta("organism", record.description)
            GeneID = find_pattern_in_fasta("GeneID", record.description)

            # Get the appropriate codon table for the organism
            codon_table = get_codon_table(organism)

            # Translate DNA to protein sequence
            protein, correct_seq = get_amino_acid_sequence(
                dna, stop_symbol=STOP_SYMBOL, codon_table=codon_table
            )
            description = str(record.description[: record.description.find("[")])
            tokenized = get_merged_seq(protein, dna, seperator=STOP_SYMBOL)

            # Create a data row for the current sequence
            data_row = {
                "dna": dna,
                "protein": protein,
                "correct_seq": correct_seq,
                "organism": organism,
                "GeneID": GeneID,
                "description": description,
                "tokenized": tokenized,
            }
            buffer.append(data_row)

            # Write buffer to CSV file when buffer size is reached
            if len(buffer) >= buffer_size:
                buffer_df = pd.DataFrame(buffer, columns=columns)
                buffer_df.to_csv(
                    output_path,
                    mode="a",
                    header=(not os.path.exists(output_path)),
                    index=True,
                )
                buffer = []

        # Write remaining buffer to CSV file
        if buffer:
            buffer_df = pd.DataFrame(buffer, columns=columns)
            buffer_df.to_csv(
                output_path,
                mode="a",
                header=(not os.path.exists(output_path)),
                index=True,
            )

    if return_dataframe:
        return pd.read_csv(output_path, index_col=0)


def download_codon_frequencies_from_kazusa(
    taxonomy_id: Optional[int] = None,
    organism: Optional[str] = None,
    taxonomy_reference: Optional[str] = None,
    return_original_format: bool = False,
) -> AMINO2CODON_TYPE:
    """
    Return the codon table of the given taxonomy ID from the Kazusa Database.

    Args:
        taxonomy_id (Optional[int]): Taxonomy ID.
        organism (Optional[str]): Name of the organism.
        taxonomy_reference (Optional[str]): Taxonomy reference.
        return_original_format (bool): Whether to return in the original format.

    Returns:
        AMINO2CODON_TYPE: Codon table.
    """
    if taxonomy_reference:
        taxonomy_id = get_taxonomy_id(taxonomy_reference, organism=organism)

    kazusa_amino2codon = pct.get_codons_table(table_name=taxonomy_id)

    if return_original_format:
        return kazusa_amino2codon

    # Replace "*" with STOP_SYMBOL in the codon table
    kazusa_amino2codon[STOP_SYMBOL] = kazusa_amino2codon.pop("*")

    # Create amino2codon dictionary
    amino2codon = {
        aminoacid: (list(codon2freq.keys()), list(codon2freq.values()))
        for aminoacid, codon2freq in kazusa_amino2codon.items()
    }

    return sort_amino2codon_skeleton(amino2codon)


def build_amino2codon_skeleton(organism: str) -> AMINO2CODON_TYPE:
    """
    Return the empty skeleton of the amino2codon dictionary, needed for get_codon_frequencies.

    Args:
        organism (str): Name of the organism.

    Returns:
        AMINO2CODON_TYPE: Empty amino2codon dictionary.
    """
    amino2codon = {}
    possible_codons = [f"{i}{j}{k}" for i in "ACGT" for j in "ACGT" for k in "ACGT"]
    possible_aminoacids = get_amino_acid_sequence(
        dna="".join(possible_codons),
        codon_table=get_codon_table(organism),
        return_correct_seq=False,
    )

    # Initialize the amino2codon skeleton with all possible codons and set their frequencies to 0
    for i, (codon, amino) in enumerate(zip(possible_codons, possible_aminoacids)):
        if amino not in amino2codon:
            amino2codon[amino] = ([], [])

        amino2codon[amino][0].append(codon)
        amino2codon[amino][1].append(0)

    # Sort the dictionary and each list of codon frequency alphabetically
    amino2codon = sort_amino2codon_skeleton(amino2codon)

    return amino2codon


def get_codon_frequencies(
    dna_sequences: List[str],
    protein_sequences: Optional[List[str]] = None,
    organism: Optional[str] = None,
) -> AMINO2CODON_TYPE:
    """
    Return a dictionary mapping each codon to its respective frequency based on
    the collection of DNA sequences and protein sequences.

    Args:
        dna_sequences (List[str]): List of DNA sequences.
        protein_sequences (Optional[List[str]]): List of protein sequences.
        organism (Optional[str]): Name of the organism.

    Returns:
        AMINO2CODON_TYPE: Dictionary mapping each amino acid to a tuple of codons and frequencies.
    """
    if organism:
        codon_table = get_codon_table(organism)
        protein_sequences = [
            get_amino_acid_sequence(
                dna, codon_table=codon_table, return_correct_seq=False
            )
            for dna in dna_sequences
        ]

    amino2codon = build_amino2codon_skeleton(organism)

    # Count the frequencies of each codon for each amino acid
    for dna, protein in zip(dna_sequences, protein_sequences):
        for i, amino in enumerate(protein):
            codon = dna[i * 3 : (i + 1) * 3]
            codon_loc = amino2codon[amino][0].index(codon)
            amino2codon[amino][1][codon_loc] += 1

    # Normalize codon frequencies per amino acid so they sum to 1
    amino2codon = {
        amino: (codons, [freq / (sum(frequencies) + 1e-100) for freq in frequencies])
        for amino, (codons, frequencies) in amino2codon.items()
    }

    return amino2codon


def get_organism_to_codon_frequencies(
    dataset: pd.DataFrame, organisms: List[str]
) -> Dict[str, AMINO2CODON_TYPE]:
    """
    Return a dictionary mapping each organism to their codon frequency distribution.

    Args:
        dataset (pd.DataFrame): DataFrame containing DNA sequences.
        organisms (List[str]): List of organisms.

    Returns:
        Dict[str, AMINO2CODON_TYPE]: Dictionary mapping each organism to its codon frequency distribution.
    """
    organism2frequencies = {}

    # Calculate codon frequencies for each organism in the dataset
    for organism in tqdm(
        organisms, desc="Calculating Codon Frequencies: ", unit="Organism"
    ):
        organism_data = dataset.loc[dataset["organism"] == organism]

        dna_sequences = organism_data["dna"].to_list()
        protein_sequences = organism_data["protein"].to_list()

        codon_frequencies = get_codon_frequencies(dna_sequences, protein_sequences)
        organism2frequencies[organism] = codon_frequencies

    return organism2frequencies


def get_codon_table(organism: str) -> int:
    """
    Return the appropriate NCBI codon table for a given organism.

    Args:
        organism (str): Name of the organism.

    Returns:
        int: Codon table number.
    """
    # Common codon table (Table 1) for many model organisms
    if organism in [
        "Arabidopsis thaliana",
        "Caenorhabditis elegans",
        "Chlamydomonas reinhardtii",
        "Saccharomyces cerevisiae" "Danio rerio",
        "Drosophila melanogaster",
        "Homo sapiens",
        "Mus musculus",
        "Nicotiana tabacum",
        "Solanum tuberosum",
        "Solanum lycopersicum",
        "Oryza sativa",
        "Glycine max",
        "Zea mays",
    ]:
        codon_table = 1

    # Chloroplast codon table (Table 11)
    elif organism in [
        "Chlamydomonas reinhardtii chloroplast",
        "Nicotiana tabacum chloroplast",
    ]:
        codon_table = 11

    # Default to Table 11 for other bacteria and archaea
    else:
        codon_table = 11

    return codon_table
