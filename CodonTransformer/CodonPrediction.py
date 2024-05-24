"""
File: CodonPrediction.py
---------------------------
Includes functions to tokenize input, load models, infer predicted dna sequences and
helper functions related to processing data for passing to the model.
"""

from typing import Any, List, Dict, Tuple
import onnxruntime as rt

import torch
import transformers
from transformers import BatchEncoding, PreTrainedTokenizerFast, BigBirdConfig
import numpy as np

from CodonTransformer.CodonData import get_codon_table, get_amino_acid_sequence, get_merged_seq
from CodonTransformer.CodonUtils import TOKEN2INDEX, INDEX2TOKEN, NUM_ORGANISMS


def load_model(
        path: str,
        device: torch.device = None,
        num_organisms: int = None,
        remove_prefix: bool = True,
        attention_type: str = 'original_full'
    ) -> torch.nn.Module:
    """
    Load a BigBirdForMaskedLM model from a model file or checkpoint based on the file extension.

    Args:
        path (str): Path to the model file or checkpoint.
        device (torch.device, optional): The device to load the model onto.
        num_organisms (int, optional): Number of organisms, needed if loading from a checkpoint that requires this.
        remove_prefix (bool, optional): Whether to remove the "model." prefix from the keys in the state dict.
        attention_type (str, optional): The type of attention, 'block_sparse' or 'original_full'.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Load from checkpoint
    if path.endswith('.ckpt'):
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        # Remove the "model." prefix from the keys
        if remove_prefix:
            state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

        if num_organisms is None:
            num_organisms = NUM_ORGANISMS
        
        config = load_bigbird_config(num_organisms)
        model = transformers.BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict)

    # Load model directly
    elif path.endswith('.pt'):
        state_dict = torch.load(path)
        config = state_dict.pop('self.config')
        model = transformers.BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict)

    else:
        raise ValueError("Unsupported file type. Please provide a .ckpt or .pt file.")

    model.bert.set_attention_type(attention_type) 
    model.eval()
    if device:
        model.to(device)

    return model


def load_bigbird_config(num_organisms: int) -> BigBirdConfig:
    """
    Load the config object used to train the BigBird transformer.
    """
    config = transformers.BigBirdConfig(
        vocab_size=len(TOKEN2INDEX),  # Equal to len(tokenizer)
        type_vocab_size=num_organisms,
        sep_token_id=2,
    )
    return config


def create_model_from_checkpoint(checkpoint_dir: str, output_model_dir: str, num_organisms: int) -> None:
    """
    Save a model to disk using a previous checkpoint.
    """
    checkpoint = load_model(checkpoint_dir, num_organisms=num_organisms)
    state_dict = checkpoint.state_dict()
    state_dict['self.config'] = load_bigbird_config(num_organisms=num_organisms)
    torch.save(state_dict, output_model_dir)


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """
    Create and return a tokenizer object from given tokenizer_path.
    """
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    return tokenizer


def tokenize(
        batch: List[Dict],
        tokenizer_path: str = '',
        tokenizer_object: PreTrainedTokenizerFast = None,
        max_len: int = 2048
) -> BatchEncoding:
    """
    Returned the tokenized sequences given batch of input data.
    Each data in batch is expected to be a dictionary with "codons" and "organism" keys.
    """
    if not tokenizer_object:
        tokenizer_object = load_tokenizer(tokenizer_path)

    tokenized = tokenizer_object(
        [data["codons"] for data in batch],
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )

    seq_len = tokenized["input_ids"].shape[-1]
    species_index = torch.tensor([[data["organism"]] for data in batch])
    tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

    return tokenized


def predict_dna_sequence(
        protein: str,
        organism_id: int,
        device: torch.device,
        tokenizer_path: str = '',
        tokenizer_object: PreTrainedTokenizerFast = None,
        model_path: str = '',
        model_object: torch.nn.Module = None,
        attention_type: str = 'original_full'
) -> str:
    """
    Return the predicted dna sequence for a given protein based on a Transformer model.
    Uses INDEX2TOKEN dictionary which maps each index to respective token of tokenizer.

    Args:
        protein (str): The protein sequence to predict the dna sequence for.
        organism_id (int): The organism id to predict the dna sequence for.
        device (torch.device): The device to run the model on.
        tokenizer_path (str, optional): The path to the tokenizer file.
        tokenizer_object (PreTrainedTokenizerFast, optional): The tokenizer object.
        model_path (str, optional): The path to the model file.
        model_object (torch.nn.Module, optional): The model object.
        attention_type (str, optional): The type of attention, 'block_sparse' or 'original_full'.
    
    Returns:
        str: The predicted dna sequence.
    """
    if not tokenizer_object:
        tokenizer_object = load_tokenizer(tokenizer_path)

    if not model_object:
        model_object = load_model(model_path, device)

    if protein == '' or protein is None:
        raise ValueError("Protein sequence cannot be empty.")
    
    if not isinstance(organism_id, int) or organism_id < 0 or organism_id >= NUM_ORGANISMS:
        raise ValueError("Invalid organism ID. Please select a valid organism id.")
    
    model_object.bert.set_attention_type(attention_type) 
    model_object.eval()
    model_object.to(device)

    with torch.no_grad():
        merged_seq = get_merged_seq(protein=protein, dna='')
        input_dict = {"idx": 0,  # sample index
                      "codons": merged_seq,
                      "organism": organism_id}

        tokenized_input = tokenize([input_dict],
                                   tokenizer_object=tokenizer_object).to(device)

        output_dict = model_object(**tokenized_input, return_dict=True)
        output = output_dict.logits.detach().cpu().numpy()

        predicted_dna = list((map(INDEX2TOKEN.__getitem__, output.argmax(axis=-1).squeeze().tolist())))
        predicted_dna = ''.join([token[-3:] for token in predicted_dna[1:-1]]).strip().upper()

        return predicted_dna


def get_high_frequency_choice_sequence(
        protein: str,
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the dna sequence optimized using High Frequency Choice (HFC) approach in which
    the most frequent codon for a given amino acid is always chosen.
    """
    dna_codons = [codon_frequencies[aminoacid][0][np.argmax(codon_frequencies[aminoacid][1])]
                  for aminoacid in protein]
    return ''.join(dna_codons)


def precompute_most_frequent_codons(
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> Dict[str, str]:
    """
    Precompute the most frequent codon for each amino acid.
    """
    return {aminoacid: codons[np.argmax(frequencies)]
            for aminoacid, (codons, frequencies) in codon_frequencies.items()}


def get_high_frequency_choice_sequence_optimized(
        protein: str,
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_high_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x10 faster speed.
    """
    most_frequent_codons = precompute_most_frequent_codons(codon_frequencies)
    return ''.join(most_frequent_codons[aminoacid] for aminoacid in protein)


def get_background_frequency_choice_sequence(
        protein: str,
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the dna sequence optimized using Background Frequency Choice (BFC) approach in which
    a random codon for a given amino acid is chosen using the codon frequencies probability distribution.
    """
    dna_codons = [np.random.choice(codon_frequencies[aminoacid][0], p=codon_frequencies[aminoacid][1])
                  for aminoacid in protein]
    return ''.join(dna_codons)


def precompute_cdf(
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> Dict[str, Tuple[List[str], Any]]:
    """
    Precompute the cumulative distribution function (CDF) for each amino acid.
    """
    cdf = {}
    for aminoacid, (codons, frequencies) in codon_frequencies.items():
        cdf[aminoacid] = (codons, np.cumsum(frequencies))
    return cdf


def get_background_frequency_choice_sequence_optimized(
        protein: str,
        codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_background_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x8 faster speed.
    """
    dna_codons = []
    cdf = precompute_cdf(codon_frequencies)

    for aminoacid in protein:
        codons, cumulative_prob = cdf[aminoacid]
        selected_codon_index = np.searchsorted(cumulative_prob, np.random.rand())
        dna_codons.append(codons[selected_codon_index])

    return ''.join(dna_codons)


def get_icor_prediction(input_seq: str, model_path: str, stop_symbol: str) -> str:
    """
    Return the optimized codon sequence for the given protein sequence.

    Credit: ICOR: improving codon optimization with recurrent neural networks
            Rishab Jain, Aditya Jain, Elizabeth Mauro, Kevin LeShane, Douglas Densmore
    """
    input_seq = input_seq.strip().upper()
    input_seq = input_seq.replace(stop_symbol, '*')

    # Define categorical labels from when model was trained.
    labels = ['AAA', 'AAC','AAG','AAT','ACA','ACG','ACT','AGC','ATA','ATC','ATG',
              'ATT','CAA','CAC','CAG','CCG','CCT','CTA','CTC','CTG','CTT','GAA',
              'GAT','GCA','GCC','GCG','GCT','GGA','GGC','GTC','GTG','GTT','TAA',
              'TAT','TCA','TCG','TCT','TGG','TGT','TTA','TTC','TTG','TTT','ACC',
              'CAT','CCA','CGG','CGT','GAC','GAG','GGT','AGT','GGG','GTA','TGC',
              'CCC','CGA','CGC','TAC','TAG','TCC','AGA','AGG','TGA']

    # Define aa to integer table
    def aa2int(seq: str) -> List[int]:
        _aa2int = {
            'A': 1,
            'R': 2,
            'N': 3,
            'D': 4,
            'C': 5,
            'Q': 6,
            'E': 7,
            'G': 8,
            'H': 9,
            'I': 10,
            'L': 11,
            'K': 12,
            'M': 13,
            'F': 14,
            'P': 15,
            'S': 16,
            'T': 17,
            'W': 18,
            'Y': 19,
            'V': 20,
            'B': 21,
            'Z': 22,
            'X': 23,
            '*': 24,
            '-': 25,
            '?': 26
        }
        return [_aa2int[i] for i in seq]


    # Create empty array to fill
    oh_array = np.zeros(shape=(26, len(input_seq)))

    # Load placements from aa2int
    aa_placement = aa2int(input_seq)

    # One-hot encode the amino acid sequence:

    # style nit: more pythonic to write for i in range(0, len(aa_placement)):
    for i in range(0, len(aa_placement)):
        oh_array[aa_placement[i], i] = 1
        i += 1

    oh_array = [oh_array]
    x = np.array(np.transpose(oh_array))

    y = x.astype(np.float32)

    y = np.reshape(y, (y.shape[0], 1, 26))

    # Start ICOR session using model.
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    # Get prediction:
    pred_onx = sess.run(None, {input_name: y})

    # Get the index of the highest probability from softmax output:
    pred_indices = []
    for pred in pred_onx[0]:
        pred_indices.append(np.argmax(pred))

    out_str = ""
    for index in pred_indices:
        out_str += labels[index]
    
    return out_str
