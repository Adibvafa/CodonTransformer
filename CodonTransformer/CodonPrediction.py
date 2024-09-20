"""
File: CodonPrediction.py
---------------------------
Includes functions to tokenize input, load models, infer predicted dna sequences and
helper functions related to processing data for passing to the model.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as rt
import torch
import transformers
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BigBirdConfig,
    BigBirdForMaskedLM,
    PreTrainedTokenizerFast,
)

from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonUtils import (
    INDEX2TOKEN,
    NUM_ORGANISMS,
    ORGANISM2ID,
    TOKEN2INDEX,
    DNASequencePrediction,
)


def predict_dna_sequence(
    protein: str,
    organism: Union[int, str],
    device: torch.device,
    tokenizer: Union[str, PreTrainedTokenizerFast] = None,
    model: Union[str, torch.nn.Module] = None,
    attention_type: str = "original_full",
    deterministic: bool = True,
    temperature: float = 0.2,
) -> DNASequencePrediction:
    """
    Predict the DNA sequence for a given protein using the CodonTransformer model.

    This function takes a protein sequence and an organism (as ID or name) as input
    and returns the predicted DNA sequence using the CodonTransformer model. It can use
    either provided tokenizer and model objects or load them from specified paths.

    Args:
        protein (str): The input protein sequence for which to predict the DNA sequence.
        organism (Union[int, str]): Either the ID of the organism or its name (e.g.,
            "Escherichia coli general"). If a string is provided, it will be converted
            to the corresponding ID using `ORGANISM2ID`.
        device (torch.device): The device (CPU or GPU) to run the model on.
        tokenizer (Union[str, PreTrainedTokenizerFast, None], optional): Either a file
            path to load the tokenizer from, a pre-loaded tokenizer object, or None. If
            None, it will be loaded from HuggingFace. Defaults to None.
        model (Union[str, torch.nn.Module, None], optional): Either a file path to load
            the model from, a pre-loaded model object, or None. If None, it will be
            loaded from HuggingFace. Defaults to None.
        attention_type (str, optional): The type of attention mechanism to use in the
            model. Can be either 'block_sparse' or 'original_full'. Defaults to
            "original_full".
        deterministic (bool, optional): Whether to use deterministic decoding (most
            likely tokens). If False, samples tokens according to their probabilities
            adjusted by the temperature. Defaults to True.
        temperature (float, optional): A value controlling the randomness of predictions
            during non-deterministic decoding. Lower values (e.g., 0.2) make the model
            more conservative, while higher values (e.g., 0.8) increase randomness.
            Using high temperatures may result in prediction of DNA sequences that
            do not translate to the input protein.
            Recommended values are:
                - Low randomness: 0.2
                - Medium randomness: 0.5
                - High randomness: 0.8
            The temperature must be a positive float. Defaults to 0.2.

    Returns:
        DNASequencePrediction: An object containing the prediction results:
            - organism (str): Name of the organism used for prediction.
            - protein (str): Input protein sequence for which DNA sequence is predicted.
            - processed_input (str): Processed input sequence (merged protein and DNA).
            - predicted_dna (str): Predicted DNA sequence.

    Raises:
        ValueError: If the protein sequence is empty, if the organism is invalid,
            or if the temperature is not a positive float.

    Note:
        This function uses `ORGANISM2ID` and `INDEX2TOKEN` dictionaries imported from
        `CodonTransformer.CodonUtils`. `ORGANISM2ID` maps organism names to their
        corresponding IDs. `INDEX2TOKEN` maps model output indices (token IDs) to
        respective codons.

    Example:
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForMaskedLM
        >>> from CodonTransformer.CodonPrediction import predict_dna_sequence
        >>> from CodonTransformer.CodonJupyter import format_model_output
        >>>
        >>> # Set up device
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>>
        >>> # Load tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        >>> model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        >>> model = model.to(device)
        >>>
        >>> # Define protein sequence and organism
        >>> protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"
        >>> organism = "Escherichia coli general"
        >>>
        >>> # Predict DNA sequence with deterministic decoding
        >>> output = predict_dna_sequence(
        ...     protein=protein,
        ...     organism=organism,
        ...     device=device,
        ...     tokenizer=tokenizer,
        ...     model=model,
        ...     attention_type="original_full",
        ...     deterministic=True
        ... )
        >>>
        >>> # Predict DNA sequence with low randomness
        >>> output_random = predict_dna_sequence(
        ...     protein=protein,
        ...     organism=organism,
        ...     device=device,
        ...     tokenizer=tokenizer,
        ...     model=model,
        ...     attention_type="original_full",
        ...     deterministic=False,
        ...     temperature=0.2
        ... )
        >>>
        >>> print(format_model_output(output))
        >>> print(format_model_output(output_random))
    """
    if not protein:
        raise ValueError("Protein sequence cannot be empty.")

    # Ensure the protein sequence contains only valid amino acids
    if not all(aminoacid in AMINO_ACIDS for aminoacid in protein):
        raise ValueError("Invalid amino acid found in protein sequence.")

    # Validate temperature
    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("Temperature must be a positive float.")

    # Load tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = load_tokenizer(tokenizer)

    # Load model
    if not isinstance(model, torch.nn.Module):
        model = load_model(model, device=device, attention_type=attention_type)
    else:
        model.eval()
        model.bert.set_attention_type(attention_type)
        model.to(device)

    # Validate organism and convert to organism_id and organism_name
    organism_id, organism_name = validate_and_convert_organism(organism)

    # Inference loop
    with torch.no_grad():
        # Tokenize the input sequence
        merged_seq = get_merged_seq(protein=protein, dna="")
        input_dict = {
            "idx": 0,  # sample index
            "codons": merged_seq,
            "organism": organism_id,
        }
        tokenized_input = tokenize([input_dict], tokenizer=tokenizer).to(device)

        # Get the model predictions
        output_dict = model(**tokenized_input, return_dict=True)
        logits = output_dict.logits.detach().cpu()

        # Decode the predicted DNA sequence from the model output
        if deterministic:
            # Select the most probable tokens (argmax)
            predicted_indices = logits.argmax(dim=-1).squeeze().tolist()
        else:
            # Sample tokens according to their probability distribution
            # Apply temperature scaling and convert logits to probabilities
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)

            # Sample from the probability distribution at each position
            probabilities = probabilities.squeeze(0)  # Shape: [seq_len, vocab_size]
            predicted_indices = (
                torch.multinomial(probabilities, num_samples=1).squeeze(-1).tolist()
            )

        predicted_dna = list(map(INDEX2TOKEN.__getitem__, predicted_indices))

        # Skip special tokens [CLS] and [SEP] to create the predicted_dna
        predicted_dna = (
            "".join([token[-3:] for token in predicted_dna[1:-1]]).strip().upper()
        )

    return DNASequencePrediction(
        organism=organism_name,
        protein=protein,
        processed_input=merged_seq,
        predicted_dna=predicted_dna,
    )


def load_model(
    model_path: Optional[str] = None,
    device: torch.device = None,
    attention_type: str = "original_full",
    num_organisms: int = None,
    remove_prefix: bool = True,
) -> torch.nn.Module:
    """
    Load a BigBirdForMaskedLM model from a model file, checkpoint, or HuggingFace.

    Args:
        model_path (Optional[str]): Path to the model file or checkpoint. If None,
            load from HuggingFace.
        device (torch.device, optional): The device to load the model onto.
        attention_type (str, optional): The type of attention, 'block_sparse'
            or 'original_full'.
        num_organisms (int, optional): Number of organisms, needed if loading from a
            checkpoint that requires this.
        remove_prefix (bool, optional): Whether to remove the "model." prefix from the
            keys in the state dict.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if not model_path:
        warnings.warn("Model path not provided. Loading from HuggingFace.", UserWarning)
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")

    elif model_path.endswith(".ckpt"):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["state_dict"]

        # Remove the "model." prefix from the keys
        if remove_prefix:
            state_dict = {
                key.replace("model.", ""): value for key, value in state_dict.items()
            }

        if num_organisms is None:
            num_organisms = NUM_ORGANISMS

        # Load model configuration and instantiate the model
        config = load_bigbird_config(num_organisms)
        model = BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict)

    elif model_path.endswith(".pt"):
        state_dict = torch.load(model_path)
        config = state_dict.pop("self.config")
        model = BigBirdForMaskedLM(config=config)
        model.load_state_dict(state_dict)

    else:
        raise ValueError(
            "Unsupported file type. Please provide a .ckpt or .pt file, "
            "or None to load from HuggingFace."
        )

    # Prepare model for evaluation
    model.bert.set_attention_type(attention_type)
    model.eval()
    if device:
        model.to(device)

    return model


def load_bigbird_config(num_organisms: int) -> BigBirdConfig:
    """
    Load the config object used to train the BigBird transformer.

    Args:
        num_organisms (int): The number of organisms.

    Returns:
        BigBirdConfig: The configuration object for BigBird.
    """
    config = transformers.BigBirdConfig(
        vocab_size=len(TOKEN2INDEX),  # Equal to len(tokenizer)
        type_vocab_size=num_organisms,
        sep_token_id=2,
    )
    return config


def create_model_from_checkpoint(
    checkpoint_dir: str, output_model_dir: str, num_organisms: int
) -> None:
    """
    Save a model to disk using a previous checkpoint.

    Args:
        checkpoint_dir (str): Directory where the checkpoint is stored.
        output_model_dir (str): Directory where the model will be saved.
        num_organisms (int): Number of organisms.
    """
    checkpoint = load_model(model_path=checkpoint_dir, num_organisms=num_organisms)
    state_dict = checkpoint.state_dict()
    state_dict["self.config"] = load_bigbird_config(num_organisms=num_organisms)

    # Save the model state dict to the output directory
    torch.save(state_dict, output_model_dir)


def load_tokenizer(tokenizer_path: Optional[str] = None) -> PreTrainedTokenizerFast:
    """
    Create and return a tokenizer object from tokenizer path or HuggingFace.

    Args:
        tokenizer_path (Optional[str]): Path to the tokenizer file. If None,
        load from HuggingFace.

    Returns:
        PreTrainedTokenizerFast: The tokenizer object.
    """
    if not tokenizer_path:
        warnings.warn(
            "Tokenizer path not provided. Loading from HuggingFace.", UserWarning
        )
        return AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

    return transformers.PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )


def tokenize(
    batch: List[Dict[str, Any]],
    tokenizer: Union[PreTrainedTokenizerFast, str] = None,
    max_len: int = 2048,
) -> BatchEncoding:
    """
    Return the tokenized sequences given a batch of input data.
    Each data in the batch is expected to be a dictionary with "codons" and
    "organism" keys.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries with "codons" and
            "organism" keys.
        tokenizer (PreTrainedTokenizerFast, str, optional): The tokenizer object or
            path to the tokenizer file.
        max_len (int, optional): Maximum length of the tokenized sequence.

    Returns:
        BatchEncoding: The tokenized batch.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = load_tokenizer(tokenizer)

    tokenized = tokenizer(
        [data["codons"] for data in batch],
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Add token type IDs for species
    seq_len = tokenized["input_ids"].shape[-1]
    species_index = torch.tensor([[data["organism"]] for data in batch])
    tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

    return tokenized


def validate_and_convert_organism(organism: Union[int, str]) -> Tuple[int, str]:
    """
    Validate and convert the organism input to both ID and name.

    This function takes either an organism ID or name as input and returns both
    the ID and name. It performs validation to ensure the input corresponds to
    a valid organism in the ORGANISM2ID dictionary.

    Args:
        organism (Union[int, str]): Either the ID of the organism (int) or its
        name (str).

    Returns:
        Tuple[int, str]: A tuple containing the organism ID (int) and name (str).

    Raises:
        ValueError: If the input is neither a string nor an integer, if the
        organism name is not found in ORGANISM2ID, if the organism ID is not a
        value in ORGANISM2ID, or if no name is found for a given ID.

    Note:
        This function relies on the ORGANISM2ID dictionary imported from
        CodonTransformer.CodonUtils, which maps organism names to their
        corresponding IDs.
    """
    if isinstance(organism, str):
        if organism not in ORGANISM2ID:
            raise ValueError(
                f"Invalid organism name: {organism}. "
                "Please use a valid organism name or ID."
            )
        organism_id = ORGANISM2ID[organism]
        organism_name = organism

    elif isinstance(organism, int):
        if organism not in ORGANISM2ID.values():
            raise ValueError(
                f"Invalid organism ID: {organism}. "
                "Please use a valid organism name or ID."
            )

        organism_id = organism
        organism_name = next(
            (name for name, id in ORGANISM2ID.items() if id == organism), None
        )
        if organism_name is None:
            raise ValueError(f"No organism name found for ID: {organism}")

    return organism_id, organism_name


def get_high_frequency_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using High Frequency Choice (HFC) approach
    in which the most frequent codon for a given amino acid is always chosen.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select the most frequent codon for each amino acid in the protein sequence
    dna_codons = [
        codon_frequencies[aminoacid][0][np.argmax(codon_frequencies[aminoacid][1])]
        for aminoacid in protein
    ]
    return "".join(dna_codons)


def precompute_most_frequent_codons(
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
) -> Dict[str, str]:
    """
    Precompute the most frequent codon for each amino acid.

    Args:
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        Dict[str, str]: The most frequent codon for each amino acid.
    """
    # Create a dictionary mapping each amino acid to its most frequent codon
    return {
        aminoacid: codons[np.argmax(frequencies)]
        for aminoacid, (codons, frequencies) in codon_frequencies.items()
    }


def get_high_frequency_choice_sequence_optimized(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_high_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x10 faster speed.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Precompute the most frequent codons for each amino acid
    most_frequent_codons = precompute_most_frequent_codons(codon_frequencies)

    return "".join(most_frequent_codons[aminoacid] for aminoacid in protein)


def get_background_frequency_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using Background Frequency Choice (BFC)
    approach in which a random codon for a given amino acid is chosen using
    the codon frequencies probability distribution.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select a random codon for each amino acid based on the codon frequencies
    # probability distribution
    dna_codons = [
        np.random.choice(
            codon_frequencies[aminoacid][0], p=codon_frequencies[aminoacid][1]
        )
        for aminoacid in protein
    ]
    return "".join(dna_codons)


def precompute_cdf(
    codon_frequencies: Dict[str, Tuple[List[str], List[float]]],
) -> Dict[str, Tuple[List[str], Any]]:
    """
    Precompute the cumulative distribution function (CDF) for each amino acid.

    Args:
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        Dict[str, Tuple[List[str], Any]]: CDFs for each amino acid.
    """
    cdf = {}

    # Calculate the cumulative distribution function for each amino acid
    for aminoacid, (codons, frequencies) in codon_frequencies.items():
        cdf[aminoacid] = (codons, np.cumsum(frequencies))

    return cdf


def get_background_frequency_choice_sequence_optimized(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Efficient implementation of get_background_frequency_choice_sequence that uses
    vectorized operations and helper functions, achieving up to x8 faster speed.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    dna_codons = []
    cdf = precompute_cdf(codon_frequencies)

    # Select a random codon for each amino acid using the precomputed CDFs
    for aminoacid in protein:
        codons, cumulative_prob = cdf[aminoacid]
        selected_codon_index = np.searchsorted(cumulative_prob, np.random.rand())
        dna_codons.append(codons[selected_codon_index])

    return "".join(dna_codons)


def get_uniform_random_choice_sequence(
    protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]
) -> str:
    """
    Return the DNA sequence optimized using Uniform Random Choice (URC) approach
    in which a random codon for a given amino acid is chosen using a uniform
    prior.

    Args:
        protein (str): The protein sequence.
        codon_frequencies (Dict[str, Tuple[List[str], List[float]]]): Codon
        frequencies for each amino acid.

    Returns:
        str: The optimized DNA sequence.
    """
    # Select a random codon for each amino acid using a uniform prior distribution
    dna_codons = [
        np.random.choice(codon_frequencies[aminoacid][0]) for aminoacid in protein
    ]
    return "".join(dna_codons)


def get_icor_prediction(input_seq: str, model_path: str, stop_symbol: str) -> str:
    """
    Return the optimized codon sequence for the given protein sequence using ICOR.

    Credit: ICOR: improving codon optimization with recurrent neural networks
            Rishab Jain, Aditya Jain, Elizabeth Mauro, Kevin LeShane, Douglas
            Densmore

    Args:
        input_seq (str): The input protein sequence.
        model_path (str): The path to the ICOR model.
        stop_symbol (str): The symbol representing stop codons in the sequence.

    Returns:
        str: The optimized DNA sequence.
    """
    input_seq = input_seq.strip().upper()
    input_seq = input_seq.replace(stop_symbol, "*")

    # Define categorical labels from when model was trained.
    labels = [
        "AAA",
        "AAC",
        "AAG",
        "AAT",
        "ACA",
        "ACG",
        "ACT",
        "AGC",
        "ATA",
        "ATC",
        "ATG",
        "ATT",
        "CAA",
        "CAC",
        "CAG",
        "CCG",
        "CCT",
        "CTA",
        "CTC",
        "CTG",
        "CTT",
        "GAA",
        "GAT",
        "GCA",
        "GCC",
        "GCG",
        "GCT",
        "GGA",
        "GGC",
        "GTC",
        "GTG",
        "GTT",
        "TAA",
        "TAT",
        "TCA",
        "TCG",
        "TCT",
        "TGG",
        "TGT",
        "TTA",
        "TTC",
        "TTG",
        "TTT",
        "ACC",
        "CAT",
        "CCA",
        "CGG",
        "CGT",
        "GAC",
        "GAG",
        "GGT",
        "AGT",
        "GGG",
        "GTA",
        "TGC",
        "CCC",
        "CGA",
        "CGC",
        "TAC",
        "TAG",
        "TCC",
        "AGA",
        "AGG",
        "TGA",
    ]

    # Define aa to integer table
    def aa2int(seq: str) -> List[int]:
        _aa2int = {
            "A": 1,
            "R": 2,
            "N": 3,
            "D": 4,
            "C": 5,
            "Q": 6,
            "E": 7,
            "G": 8,
            "H": 9,
            "I": 10,
            "L": 11,
            "K": 12,
            "M": 13,
            "F": 14,
            "P": 15,
            "S": 16,
            "T": 17,
            "W": 18,
            "Y": 19,
            "V": 20,
            "B": 21,
            "Z": 22,
            "X": 23,
            "*": 24,
            "-": 25,
            "?": 26,
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
