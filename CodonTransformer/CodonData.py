"""
File: CodonData.py
---------------------
Includes helper functions for preprocessing NCBI or Kazusa databases.
"""

import os
import pandas as pd

from CodonTransformer.CodonUtils import (
    STOP_SYMBOL,
    AMINO2CODON_TYPE,
    find_pattern_in_fasta,
    sort_amino2codon_skeleton,
    load_python_object_from_disk,
    get_taxonomy_id
)

from Bio import SeqIO
from Bio.Seq import Seq

import python_codon_tables as pct

from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm


def get_codon_table(organism: str) -> int:
    """
    Return the appropriate NCBI codon table for given organism
    """
    if organism in ['Arabidopsis thaliana', 'Caenorhabditis elegans', 'Chlamydomonas reinhardtii',
                    'Saccharomyces cerevisiae' 'Danio rerio', 'Drosophila melanogaster', 'Homo sapiens',
                    'Mus musculus', 'Nicotiana tabacum', 'Solanum tuberosum', 'Solanum lycopersicum',
                    'Oryza sativa', 'Glycine max', 'Zea mays']:
        codon_table = 1

    elif organism in ['Chlamydomonas reinhardtii chloroplast', 'Nicotiana tabacum chloroplast']:
        codon_table = 11

    else:  # Other Bacteria including E. coli and Archaebacteria
        codon_table = 11

    return codon_table


def get_merged_seq(protein: str, dna: str = "", seperator: str = "_") -> str:
    """
    Return the merged sequence of protein aminoacids and dna codons in the form of tokens
    separated by space, where each token is composed of an aminoacid + seperator + codon

    >>> get_merged_seq(protein="MAV", dna="ATGGCTGTG", seperator="_")
    'M_ATG A_GCT V_GTG'
    """
    merged_seq = ''

    if protein[-1] == '*':
        protein[-1] = STOP_SYMBOL

    if protein[-1] != STOP_SYMBOL:
        protein += STOP_SYMBOL
    
    protein = protein.upper().strip().replace('\n', '').replace(' ', '').replace('\t', '')

    for i, aminoacid in enumerate(protein):
        merged_seq += f'{aminoacid}{seperator}{dna[i * 3:i * 3 + 3] if dna else "UNK"} '

    return merged_seq.strip()


def is_correct_seq(dna: str, protein: str, stop_symbol: str = STOP_SYMBOL) -> bool:
    """
    Return if the given dna, protein pair is correct, that is:
    1. The length of dna is divisible by 3
    2. There is an initiator codon in the beginning of dna
    3. There is only one stop codon in the sequence
    4. The only stop codon is the last codon
    Note since in Codon Table 3, 'TGA' is interpreted as Triptophan (W),
    we have included a separate check to make sure those sequences are considered correct.
    """
    return (len(dna) % 3 == 0 and
            dna[:3].upper() in ('ATG', 'TTG', 'CTG', 'GTG') and
            protein[-1] == stop_symbol and
            protein.count(stop_symbol) == 1 and
            len(set(dna)) == 4)


def get_amino_acid_sequence(dna: str,
                            stop_symbol: str = "_",
                            codon_table: int = 1,
                            return_correct_seq: bool = True
                            ) -> Union[Tuple[str, bool], str]:
    """
    Return the translated protein sequence given a dna sequence and codon table.
    """
    dna_seq = Seq(dna).strip()
    protein_seq = str(dna_seq.translate(stop_symbol=stop_symbol,
                                        to_stop=False,
                                        cds=False,
                                        table=codon_table)).strip()

    correct_seq = is_correct_seq(dna_seq, protein_seq, stop_symbol)

    return (protein_seq, correct_seq) if return_correct_seq else protein_seq


def read_fasta_file(input_file: str,
                    output_path: str,
                    organism: str = '',
                    return_dataframe: bool = True,
                    buffer_size: int = 50000) -> pd.DataFrame:
    """
    Read a FASTA file of DNA sequences and save it to a Pandas DataFrame.
    Columns indicates the output columns in that DataFrame.
    Return the created Pandas DataFrame if needed by function call.
    """

    buffer = []
    columns = ['dna', 'protein', 'correct_seq', 'organism', 'GeneID', 'description', 'tokenized']

    with open(input_file, "r") as fasta_file:

        for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc=f'{organism}', unit=' Rows'):
            dna = str(record.seq).strip()

            if not organism:
                organism = find_pattern_in_fasta('organism', record.description)
            GeneID = find_pattern_in_fasta('GeneID', record.description)

            codon_table = get_codon_table(organism)
            protein, correct_seq = get_amino_acid_sequence(dna, stop_symbol=STOP_SYMBOL, codon_table=codon_table)
            description = str(record.description[:record.description.find('[')])
            tokenized = get_merged_seq(protein, dna, seperator=STOP_SYMBOL)

            data_row = {'dna': dna,
                        'protein': protein,
                        'correct_seq': correct_seq,
                        'organism': organism,
                        'GeneID': GeneID,
                        'description': description,
                        'tokenized': tokenized}
            buffer.append(data_row)

            if len(buffer) >= buffer_size:
                buffer_df = pd.DataFrame(buffer, columns=columns)
                buffer_df.to_csv(output_path, mode='a',
                                 header=(not os.path.exists(output_path)), index=True)
                buffer = []

        if buffer:
            buffer_df = pd.DataFrame(buffer, columns=columns)
            buffer_df.to_csv(output_path, mode='a',
                             header=(not os.path.exists(output_path)), index=True)

    if return_dataframe:
        return pd.read_csv(output_path, index_col=0)


def download_codon_frequencies_from_kazusa(taxonomy_id: Optional[int] = None,
                                           organism: Optional[str] = None,
                                           taxonomy_reference: Optional[str] = None,
                                           return_original_format=False) -> AMINO2CODON_TYPE:
    """
    Return the codon table of given taxonomy id from Kazsua Database.
    If organism and taxonomy_reference are provided, it will find the taxonomy_id
    using the given references.
    """
    if taxonomy_reference:
        taxonomy_id = get_taxonomy_id(taxonomy_reference, organism=organism)

    kazusa_amino2codon = pct.get_codons_table(table_name=taxonomy_id)

    if return_original_format:
        return kazusa_amino2codon

    kazusa_amino2codon[STOP_SYMBOL] = kazusa_amino2codon.pop('*')

    amino2codon = {aminoacid: (list(codon2freq.keys()), list(codon2freq.values()))
                   for aminoacid, codon2freq in kazusa_amino2codon.items()}

    return sort_amino2codon_skeleton(amino2codon)


def build_amino2codon_skeleton(organism: str) -> AMINO2CODON_TYPE:
    """
    Return the empty skeleton of amino2codon dictionary, which is needed for get_codon_frequencies
    """
    amino2codon = {}
    possible_codons = [f'{i}{j}{k}' for i in 'ACGT' for j in 'ACGT' for k in 'ACGT']
    possible_aminoacids = get_amino_acid_sequence(dna=''.join(possible_codons),
                                                  codon_table=get_codon_table(organism),
                                                  return_correct_seq=False)

    for i, (codon, amino) in enumerate(zip(possible_codons, possible_aminoacids)):

        if amino not in amino2codon:
            amino2codon[amino] = ([], [])

        amino2codon[amino][0].append(codon)
        amino2codon[amino][1].append(0)

    # Sort the dictionary and each list of codon frequency alphabetically
    amino2codon = sort_amino2codon_skeleton(amino2codon)

    return amino2codon


def get_codon_frequencies(dna_sequences: List[str],
                          protein_sequences: Optional[List[str]] = None,
                          organism: Optional[str] = None) -> AMINO2CODON_TYPE:
    """
    Return a dictionary mapping each codon to its respective frequency based on
    the collection of dna sequences and protein sequences entered.
    If the protein sequences are not provided, the organism input will be used to generate them.
    The returned dictionary maps each amino acid to a tuple of two lists, the first list represents all
    possible codons of that amino acid and the second list represents their frequencies which sums to one.
    """
    if organism:
        codon_table = get_codon_table(organism)
        protein_sequences = [
            get_amino_acid_sequence(dna, codon_table=codon_table, return_correct_seq=False)
            for dna in dna_sequences
        ]

    amino2codon = build_amino2codon_skeleton(organism)

    for dna, protein in zip(dna_sequences, protein_sequences):
        for i, amino in enumerate(protein):
            codon = dna[i * 3: (i + 1) * 3]
            codon_loc = amino2codon[amino][0].index(codon)
            amino2codon[amino][1][codon_loc] += 1

    # Normalize codon frequencies per amino acid so they sum to 1
    amino2codon = {amino: (codons,
                           [freq / (sum(frequencies) + 1e-100) for freq in frequencies])
                   for amino, (codons, frequencies) in amino2codon.items()}

    return amino2codon


def get_organism_to_codon_frequencies(dataset: pd.DataFrame,
                                      organisms: List[str]) -> Dict[str, AMINO2CODON_TYPE]:
    """
    Return a dictionary mapping each organism to their codon frequency distribution.
    Expects dataset dataframe to have columns named "organism" and "dna".
    """
    organism2frequencies = {}

    for organism in tqdm(organisms, desc="Calculating Codon Frequencies: ", unit="Organism"):
        organism_data = dataset.loc[dataset["organism"] == organism]

        dna_sequences = organism_data['dna'].to_list()
        protein_sequences = organism_data['protein'].to_list()

        codon_frequencies = get_codon_frequencies(dna_sequences, protein_sequences)
        organism2frequencies[organism] = codon_frequencies

    return organism2frequencies


def get_cousin(dna, organism, ref_freq):
    que_freq = get_codon_frequencies([dna], organism=organism)

    weight_ref = build_amino2codon_skeleton(organism)
    weight_ref = {amino: [0] for amino in weight_ref}

    weight_que = build_amino2codon_skeleton(organism)
    weight_que = {amino: [0] for amino in weight_que}

    aminos = []

    for ((amino_ref, (codons_ref, frequencies_ref)), (amino_que, (codons_que, frequencies_que))) in zip(
            ref_freq.items(), que_freq.items()):
        if amino_ref != '_' and len(frequencies_ref) > 1 and sum(frequencies_ref) != 0.0 and sum(
                frequencies_que) != 0.0:
            aminos.append(amino_ref)
            weight_ref[amino_ref] = [ref * round((ref - (1 / len(codons_ref))), 3) for (ref, que) in
                                     zip(frequencies_ref, frequencies_que)]
            weight_que[amino_que] = [que * round((ref - (1 / len(codons_ref))), 3) for (ref, que) in
                                     zip(frequencies_ref, frequencies_que)]

    cousin = 0
    for ((amino_ref, ref), (amino_que, que)) in zip(weight_ref.items(), weight_que.items()):

        if sum(ref) == 0:
            if amino_ref in aminos:
                aminos.remove(amino_ref)
            continue

        cousin_aa = round((sum(que) / (sum(ref) + 1e-100)), 3)
        # cousin_aa = np.clip(cousin_aa, a_min=-3, a_max=4)
        cousin += cousin_aa  # round(cousin_aa, 3)

    cousin = cousin / (len(aminos))
    return cousin
