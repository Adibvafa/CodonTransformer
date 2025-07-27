<p align="center">
  <img src="https://github.com/Adibvafa/CodonTransformer/raw/main/src/banner_final.png" alt="CodonTransformer Logo" width="100%" height="100%" style="vertical-align: middle;"/>
</p>

<p align="center">
  <a href="https://www.nature.com/articles/s41467-025-58588-7" target="_blank"><img src="https://img.shields.io/badge/Nature-Paper-2C974B?style=for-the-badge&logo=nature&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/Adibvafa/CodonTransformer"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://adibvafa.github.io/CodonTransformer/"><img src="https://img.shields.io/badge/Website-Online-00B89E?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/adibvafa/CodonTransformer"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
  <a href="https://adibvafa.github.io/CodonTransformer/GoogleColab"><img src="https://img.shields.io/badge/Colab-Notebook-e2006a?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a>
</p>

## Table of Contents
- [Overview](#overview)
- [Use Case](#use-case)
- [Installation](#installation)
- [Finetuning](#finetuning-codontransformer)
- [Key Features](#key-features)
  - [CodonData](#codondata)
  - [CodonPrediction](#codonprediction)
  - [CodonEvaluation](#codonevaluation)
  - [CodonUtils](#codonutils)
  - [CodonJupyter](#codonjupyter)
- [Contribution](#contribution)
- [Citation](#citation)
<br></br>

## Abstract
The genetic code is degenerate allowing a multitude of possible DNA sequences to encode the same protein. This degeneracy impacts the efficiency of heterologous protein production due to the codon usage preferences of each organism. The process of tailoring organism-specific synonymous codons, known as codon optimization, must respect local sequence patterns that go beyond global codon preferences. As a result, the search space faces a combinatorial explosion that makes exhaustive exploration impossible. Nevertheless, throughout the diverse life on Earth, natural selection has already optimized the sequences, thereby providing a rich source of data allowing machine learning algorithms to explore the underlying rules. Here, we introduce CodonTransformer, a multispecies deep learning model trained on over 1 million DNA-protein pairs from 164 organisms spanning all kingdoms of life. The model demonstrates context-awareness thanks to the attention mechanism and bidirectionality of the Transformers we used, and to a novel sequence representation that combines organism, amino acid, and codon encodings. CodonTransformer generates host-specific DNA sequences with natural-like codon distribution profiles and with negative cis-regulatory elements. This work introduces a novel strategy of Shared Token Representation and Encoding with Aligned Multi-masking (STREAM) and provides a state-of-the-art codon optimization framework with a customizable open-access model and a user-friendly interface.
<br></br>


## Use Case
**For a user-friendly interface, check out our [Google Colab Notebook.](https://adibvafa.github.io/CodonTransformer/GoogleColab)**
<br></br>
After installing CodonTransformer, you can use:

```python
import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonJupyter import format_model_output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(device)


# Set your input data
protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
organism = "Escherichia coli general"


# Predict with CodonTransformer
output = predict_dna_sequence(
    protein=protein,
    organism=organism,
    device=device,
    tokenizer=tokenizer,
    model=model,
    attention_type="original_full",
    deterministic=True
)
print(format_model_output(output))
```
The output is:
```
-----------------------------
|          Organism         |
-----------------------------
Escherichia coli general

-----------------------------
|       Input Protein       |
-----------------------------
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG

-----------------------------
|      Processed Input      |
-----------------------------
M_UNK A_UNK L_UNK W_UNK M_UNK R_UNK L_UNK L_UNK P_UNK L_UNK L_UNK A_UNK L_UNK L_UNK A_UNK L_UNK W_UNK G_UNK P_UNK D_UNK P_UNK A_UNK A_UNK A_UNK F_UNK V_UNK N_UNK Q_UNK H_UNK L_UNK C_UNK G_UNK S_UNK H_UNK L_UNK V_UNK E_UNK A_UNK L_UNK Y_UNK L_UNK V_UNK C_UNK G_UNK E_UNK R_UNK G_UNK F_UNK F_UNK Y_UNK T_UNK P_UNK K_UNK T_UNK R_UNK R_UNK E_UNK A_UNK E_UNK D_UNK L_UNK Q_UNK V_UNK G_UNK Q_UNK V_UNK E_UNK L_UNK G_UNK G_UNK __UNK

-----------------------------
|       Predicted DNA       |
-----------------------------
ATGGCTTTATGGATGCGTCTGCTGCCGCTGCTGGCGCTGCTGGCGCTGTGGGGCCCGGACCCGGCGGCGGCGTTTGTGAATCAGCACCTGTGCGGCAGCCACCTGGTGGAAGCGCTGTATCTGGTGTGCGGTGAGCGCGGCTTCTTCTACACGCCCAAAACCCGCCGCGAAGCGGAAGATCTGCAGGTGGGCCAGGTGGAGCTGGGCGGCTAA
```

### Generating Multiple Variable Sequences

Set `deterministic=False` to generate variable sequences. Control the variability using `temperature`:

- `temperature`: (recommended between 0.2 and 0.8)
  - Lower values (e.g., 0.2): More conservative predictions
  - Higher values (e.g., 0.8): More diverse predictions

Using high temperatures (e.g. more than 1) might result in prediction of DNA sequences that do not translate to the input protein.<br>
You can set `match_protein=True` to ensure predicted DNA sequences translate to the input protein.<br>
Generate multiple sequences by setting `num_sequences` to a value greater than 1.
<br><br>


### Batch Inference
**You can use the [inference template](https://github.com/Adibvafa/CodonTransformer/raw/main/src/CodonTransformer_inference_template.xlsx) to setup your dataset for batch inference in [Google Colab](https://adibvafa.github.io/CodonTransformer/GoogleColab).**
A sample dataset is provided under [\demo](https://github.com/Adibvafa/CodonTransformer/tree/main/demo) . A typical inference might take 1-3 seconds based on available compute.

<br>

### Arguments of predict_dna_sequence
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `protein` | `str` | Input protein sequence | Required |
| `organism` | `Union[int, str]` | Organism ID (integer) or name (string) (e.g., "Escherichia coli general") | Required |
| `device` | `torch.device` | PyTorch device object specifying whether to run on CPU or GPU | Required |
| `tokenizer` | `Union[str, PreTrainedTokenizerFast, None]` | Either a file path to load tokenizer from, a pre-loaded tokenizer object, or None to load from HuggingFace's "adibvafa/CodonTransformer" | `None` |
| `model` | `Union[str, torch.nn.Module, None]` | Either a file path to load model from, a pre-loaded model object, or None to load from HuggingFace's "adibvafa/CodonTransformer" | `None` |
| `attention_type` | `str` | Type of attention mechanism to use in model - 'block_sparse' for memory efficient or 'original_full' for standard attention | `"original_full"` |
| `deterministic` | `bool` | If True, uses deterministic decoding (picks most likely tokens). If False, samples tokens based on probabilities adjusted by temperature | `True` |
| `temperature` | `float` | Controls randomness in non-deterministic mode. Lower values (0.2) are conservative and pick high probability tokens, while higher values (0.8) allow more diversity. Must be positive | `0.2` |
| `top_p` | `float` | Nucleus sampling threshold - only tokens with cumulative probability up to this value are considered. Balances diversity and quality of predictions. Must be between 0 and 1 | `0.95` |
| `num_sequences` | `int` | Number of different DNA sequences to generate. Only works when deterministic=False. Each sequence will be sampled based on the temperature and top_p parameters. Must be positive | `1` |
| `match_protein` | `bool` | Constrains predictions to only use codons that translate back to the exact input protein sequence. Only recommended when using high temperatures or error prone input proteins (e.g. not starting with methionine or having numerous repetitions) | `False` |

Returns: `Union[DNASequencePrediction, List[DNASequencePrediction]]` containing predicted DNA sequence(s) and metadata.
<br>

## Installation
Install CodonTransformer via pip:

```sh
pip install CodonTransformer
```

Or clone the repository:

```sh
git clone https://github.com/adibvafa/CodonTransformer.git
cd CodonTransformer
pip install -r requirements.txt
```

The package requires `python>=3.9`, supports all major operating systems, and takes about 10-30 seconds depending on already installed requirements, [availabe here](requirements.txt).
<br><br><br>


## Finetuning CodonTransformer
To finetune CodonTransformer on your own data, follow these steps:

1. **Prepare your dataset**

   Create a CSV file with the following columns:
   - `dna`: DNA sequences (string, preferably uppercase ATCG)
   - `protein`: Protein sequences (string, preferably uppercase amino acid letters)
   - `organism`: Target organism (string or int, must be from `ORGANISM2ID` in `CodonUtils`)


   Note:
   - Use organisms from the `FINE_TUNE_ORGANISMS` list for best results.
   - For E. coli, use `Escherichia coli general`.
   - DNA sequences should ideally contain only A, T, C, and G. Ambiguous codons are replaced with 'UNK' for tokenization.
   - Protein sequences should contain standard amino acid letters from `AMINO_ACIDS` in `CodonUtils`. Ambiguous amino acids are replaced according to the `AMBIGUOUS_AMINOACID_MAP` in `CodonUtils`.
   - End your DNA sequences with a stop codon from `STOP_CODONS` in `CodonUtils`. If not present, a 'UNK' stop codon will be addded in preprocessing.
   - End your protein sequence with `_` or `*`. If either is not present, a `_` will be added in preprocessing.
<br>

2. **Prepare training data**

   Use the `prepare_training_data` function from `CodonData` to prepare training data from your dataset.

   ```python
   from CodonTransformer.CodonData import prepare_training_data
   prepare_training_data('your_data.csv', 'your_dataset_directory/training_data.json')
   ```
<br>

3. **Run the finetuning script**
   Execute finetune.py with appropriate arguments: (an example)
    ```bash
     python finetune.py \
        --dataset_dir 'your_dataset_directory/training_data.json' \
        --checkpoint_dir 'your_checkpoint_directory' \
        --checkpoint_filename 'finetune.ckpt' \
        --batch_size 6 \
        --max_epochs 15 \
        --num_workers 5 \
        --accumulate_grad_batches 1 \
        --num_gpus 4 \
        --learning_rate 0.00005 \
        --warmup_fraction 0.1 \
        --save_every_n_steps 512 \
        --seed 23
    ```
   This script automatically loads the pretrained model from Hugging Face and finetunes it on your dataset.
   For an example of a SLURM job request, see the `slurm` directory in the repository.
<br></br>

## Handling Ambiguous Amino Acids

CodonTransformer provides a flexible system for handling ambiguous amino acids through the `ProteinConfig` class. By default, CodonUtils includes a [predefined mapping for ambiguous amino acids](https://github.com/Adibvafa/CodonTransformer/blob/main/CodonTransformer/CodonUtils.py#L45), but users can customize this behavior:

```python
from CodonTransformer.CodonUtils import ProteinConfig

# Configure protein preprocessing
config = ProteinConfig()
config.set('ambiguous_aminoacid_behavior', 'standardize_random')
config.set('ambiguous_aminoacid_map_override', {'X': ['A', 'G', 'S']})

# Run CodonTransformer
...
```

Options for `ambiguous_aminoacid_behavior`:
- `standardize_random` (default): Randomly selects a random amino acid from the mapping list.
- `standardize_deterministic`: Selects the first amino acid from the mapping list.
- `raise_error`: Treats ambiguous amino acids as invalid.

Users can override the default mapping with `ambiguous_aminoacid_map_override`.
<br></br>


## Organisms

This project utilizes a dataset of 164 distinct organisms for training and fine-tuning. The full dataset, including genomic sequences and codon adaptation indices, can be found at [Zenodo (DOI: 10.5281/zenodo.12509224)](https://zenodo.org/records/12509224).

### Summary by Kingdom

| Kingdom  | Number of Organisms |
| :------- | :------------------ |
| Bacteria | 142                 |
| Archaea  | 11                  |
| Plantae  | 5                   |
| Animalia | 5                   |
| Fungi    | 1                   |
| **Total**| **164** |

### Detailed Breakdown by Clade

#### Animalia

| Class          | Number of Organisms | Organism Names                                   |
| :------------- | :------------------ | :----------------------------------------------- |
| Mammalia       | 2                   | *Homo sapiens*, *Mus musculus* |
| Insecta        | 1                   | *Drosophila melanogaster* |
| Actinopterygii | 1                   | *Danio rerio* (zebrafish)                        |
| Chromadorea    | 1                   | *Caenorhabditis elegans* |

#### Plantae

The list includes two plant species, a green alga, and their corresponding chloroplasts, which are treated as separate entries in the dataset.

| Phylum       | Class         | Number of Organisms | Organism Names                                                                |
| :----------- | :------------ | :------------------ | :---------------------------------------------------------------------------- |
| Tracheophyta | Magnoliopsida | 3                   | *Arabidopsis thaliana*, *Nicotiana tabacum*, *Nicotiana tabacum chloroplast* |
| Chlorophyta  | Chlorophyceae | 2                   | *Chlamydomonas reinhardtii*, *Chlamydomonas reinhardtii chloroplast* |

#### Fungi

| Phylum     | Class           | Number of Organisms | Organism Name              |
| :--------- | :-------------- | :------------------ | :------------------------- |
| Ascomycota | Saccharomycetes | 1                   | *Saccharomyces cerevisiae* |

#### Archaea

The archaeal organisms in this dataset are all extremophiles.

| Phylum       | Class        | Number of Organisms | Genera                          |
| :----------- | :----------- | :------------------ | :------------------------------ |
| Euryarchaeota| Thermococci  | 10                  | *Pyrococcus*, *Thermococcus* |
| Crenarchaeota| Thermoprotei | 1                   | *Saccharolobus* |

#### Bacteria

The vast majority of the organisms in the dataset are bacteria, primarily from the phylum Pseudomonadota (formerly Proteobacteria).

| Phylum        | Number of Organisms | Notable Genera                                                                                    |
| :------------ | :------------------ | :------------------------------------------------------------------------------------------------ |
| Pseudomonadota| 141                 | *Escherichia*, *Salmonella*, *Klebsiella*, *Pseudomonas*, *Yersinia*, *Serratia*, *Enterobacter*, *Proteus*, etc. |
| Bacillota     | 1                   | *Bacillus* |

### Full Organism List

The model supports the following 164 organisms. Organisms can be referenced by their name or by their corresponding ID (0-163) when using the tool.

| | | | |
| :-- | :-- | :-- | :-- |
| Arabidopsis thaliana | Enterobacter hormaechei | Klebsiella variicola | Proteus penneri |
| Atlantibacter hermannii | Enterobacter kobei | Kosakonia cowanii | Proteus terrae subsp. cibarius |
| Bacillus subtilis | Enterobacter ludwigii | Kosakonia radicincitans | Proteus vulgaris |
| Brenneria goodwinii | Enterobacter mori | Leclercia adecarboxylata | Providencia alcalifaciens |
| Buchnera aphidicola (Schizaphis graminum) | Enterobacter quasiroggenkampii | Lelliottia amnigena | Providencia heimbachae |
| Caenorhabditis elegans | Enterobacter roggenkampii | Lonsdalea populi | Providencia rettgeri |
| Candidatus Erwinia haradaeae | Enterobacter sichuanensis | Moellerella wisconsensis | Providencia rustigianii |
| Candidatus Hamiltonella defensa 5AT (Acyrthosiphon pisum) | Erwinia amylovora CFBP1430 | Morganella morganii | Providencia stuartii |
| Chlamydomonas reinhardtii | Erwinia persicina | Mus musculus | Providencia thailandensis |
| Chlamydomonas reinhardtii chloroplast | Escherichia albertii | Nicotiana tabacum | Pseudomonas putida |
| Citrobacter amalonaticus | Escherichia coli general | Nicotiana tabacum chloroplast | Pyrococcus furiosus |
| Citrobacter braakii | Escherichia coli O157-H7 str. Sakai | Obesumbacterium proteus | Pyrococcus horikoshii |
| Citrobacter cronae | Escherichia coli str. K-12 substr. MG1655 | Pantoea agglomerans | Pyrococcus yayanosii |
| Citrobacter europaeus | Escherichia fergusonii | Pantoea allii | Rahnella aquatilis CIP 78.65 = ATCC 33071 |
| Citrobacter farmeri | Escherichia marmotae | Pantoea ananatis PA13 | Raoultella ornithinolytica |
| Citrobacter freundii | Escherichia ruysiae | Pantoea dispersa | Raoultella planticola |
| Citrobacter koseri ATCC BAA-895 | Ewingella americana | Pantoea stewartii | Raoultella terrigena |
| Citrobacter portucalensis | Hafnia alvei | Pantoea vagans | Rosenbergiella epipactidis |
| Citrobacter werkmanii | Hafnia paralvei | Pectobacterium aroidearum | Rouxiella badensis |
| Citrobacter youngae | Homo sapiens | Pectobacterium atrosepticum | Saccharomyces cerevisiae |
| Cronobacter dublinensis subsp. dublinensis LMG 23823 | Kalamiella piersonii | Pectobacterium brasiliense | Saccharolobus solfataricus |
| Cronobacter malonaticus LMG 23826 | Klebsiella aerogenes | Pectobacterium carotovorum | Salmonella bongori N268-08 |
| Cronobacter sakazakii | Klebsiella grimontii | Pectobacterium odoriferum | Salmonella enterica subsp. enterica serovar Typhimurium str. LT2 |
| Cronobacter turicensis | Klebsiella michiganensis | Pectobacterium parmentieri | Serratia bockelmannii |
| Danio rerio | Klebsiella oxytoca | Pectobacterium polaris | Serratia entomophila |
| Dickeya dadantii 3937 | Klebsiella pasteurii | Pectobacterium versatile | Serratia ficaria |
| Dickeya dianthicola | Klebsiella pneumoniae subsp. pneumoniae HS11286 | Photorhabdus laumondii subsp. laumondii TTO1 | Serratia fonticola |
| Dickeya fangzhongdai | Klebsiella quasipneumoniae | Plesiomonas shigelloides | Serratia grimesii |
| Dickeya solani | Klebsiella quasivariicola | Pluralibacter gergoviae | Serratia liquefaciens |
| Dickeya zeae | Thermoccoccus kodakarensis | Proteus faecis | Serratia marcescens |
| Drosophila melanogaster | Thermococcus barophilus MPT | Proteus mirabilis HI4320 | Serratia nevei |
| Edwardsiella anguillarum ET080813 | Thermococcus chitonophagus | Yersinia aldovae 670-83 | Serratia plymuthica AS9 |
| Edwardsiella ictaluri | Thermococcus gammatolerans | Yersinia aleksiciae | Serratia proteamaculans |
| Edwardsiella piscicida | Thermococcus litoralis | Yersinia alsatica | Serratia quinivorans |
| Edwardsiella tarda | Thermococcus onnurineus | Yersinia enterocolitica | Serratia rubidaea |
| Enterobacter asburiae | Thermococcus sibiricus | Yersinia frederiksenii ATCC 33641 | Serratia ureilytica |
| Enterobacter bugandensis | Xenorhabdus bovienii str. feltiae Florida | Yersinia intermedia | Shigella boydii |
| Enterobacter cancerogenus | Yersinia kristensenii | Yersinia massiliensis CCUG 53443 | Shigella dysenteriae |
| Enterobacter chengduensis | Yersinia mollaretii ATCC 43969 | Yersinia pestis A1122 | Shigella flexneri 2a str. 301 |
| Enterobacter cloacae | Yersinia proxima | Yersinia pseudotuberculosis IP 32953 | Shigella sonnei |
| Yersinia rochesterensis | Yersinia rohdei | Yersinia ruckeri | Yokenella regensburgei |

<br><br>

## Star History
<p align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=adibvafa/codontransformer&type=Date)](https://star-history.com/#adibvafa/codontransformer&Date)

</p>
<br>


## Key Features
- **CodonData** <br>
CodonData facilitates preprocessing of genetic information by cleaning and translating DNA and protein sequences, handling FASTA files, and managing codon frequencies from databases like NCBI and Kazusa.

- **CodonPrediction** <br>
CodonPrediction enables preprocessing of sequences, the prediction of optimized DNA sequences tailored to specific organisms using the CodonTransformer model, and supports various other optimization strategies.

- **CodonEvaluation** <br>
CodonEvaluation provides tools to compute evaluation metrics such as Codon Similarity Index (CSI), GC content, and Codon Frequency Distribution, allowing for detailed assessment of optimized sequences.

- **CodonUtils** <br>
CodonUtils offers essential constants and helper functions for genetic sequence analysis, including amino acid mappings, codon tables, taxonomy ID management, and sequence validation.

- **CodonJupyter** <br>
CodonJupyter enhances Jupyter notebook workflows with interactive widgets for selecting organisms and inputting protein sequences, as well as formatting and displaying optimized DNA sequence outputs.
<br></br>


## CodonData

The CodonData subpackage offers tools for preprocessing NCBI or Kazusa databases and managing codon-related data operations. It includes comprehensive functions for working with DNA sequences, protein sequences, and codon frequencies, providing a robust toolkit for sequence preprocessing and codon frequency analysis across different organisms.

### Overview

This subpackage is suitable for:

- Preparing data for model training and inference
- Preprocessing and cleaning DNA and protein sequences
- Translating DNA sequences to protein sequences
- Reading and processing FASTA files
- Downloading and processing codon frequency data from the Kazusa database
- Calculating codon frequencies from given sequences
- Handling organism-specific codon tables and translations

### Available Functions

- `prepare_training_data(dataset: Union[str, pd.DataFrame], output_file: str, shuffle: bool = True) -> None`

  Prepare a JSON dataset for training the CodonTransformer model. Process the input dataset, create the 'codons' column, handle organism IDs, and save the result to a JSON file.

- `dataframe_to_json(df: pd.DataFrame, output_file: str, shuffle: bool = True) -> None`

  Convert a pandas DataFrame to a JSON file format suitable for training CodonTransformer. Write each row of the DataFrame as a JSON object to the output file, with an option to shuffle the data.

- `process_organism(organism: Union[str, int], organism_to_id: Dict[str, int]) -> int`

  Process and validate the organism input, converting it to a valid organism ID. Handle both string (organism name) and integer (organism ID) inputs.

- `get_codon_table(organism: str) -> int`

  Return the appropriate NCBI codon table number for a given organism.

- `preprocess_protein_sequence(protein: str) -> str`

  Clean, standardize, and handle ambiguous amino acids in a protein sequence. Utilizes the `ProteinConfig` to handle ambiguous amino acids based on user-defined settings.

- `replace_ambiguous_codons(dna: str) -> str`

  Replace ambiguous codons in a DNA sequence with "UNK".

- `preprocess_dna_sequence(dna: str) -> str`

  Clean and preprocess a DNA sequence by standardizing it and replacing ambiguous codons.

- `get_merged_seq(protein: str, dna: str = "", separator: str = "_") -> str`

  Merge protein and DNA sequences into a single string of tokens.

- `is_correct_seq(dna: str, protein: str, stop_symbol: str = STOP_SYMBOL) -> bool`

  Check if the given DNA and protein pair is correct based on specific criteria.

- `get_amino_acid_sequence(dna: str, stop_symbol: str = "_", codon_table: int = 1, return_correct_seq: bool = True) -> Union[Tuple[str, bool], str]`

  Translate a DNA sequence to a protein sequence using a specified codon table.

- `read_fasta_file(input_file: str, output_path: str, organism: str = "", return_dataframe: bool = True, buffer_size: int = 50000) -> pd.DataFrame`

  Read a FASTA file of DNA sequences and saves it to a Pandas DataFrame.

- `download_codon_frequencies_from_kazusa(taxonomy_id: Optional[int] = None, organism: Optional[str] = None, taxonomy_reference: Optional[str] = None, return_original_format: bool = False) -> AMINO2CODON_TYPE`

  Download and process codon frequency data from the Kazusa database for a given taxonomy ID or organism.

- `build_amino2codon_skeleton(organism: str) -> AMINO2CODON_TYPE`

  Create an empty skeleton of the amino2codon dictionary for a given organism.

- `get_codon_frequencies(dna_sequences: List[str], protein_sequences: Optional[List[str]] = None, organism: Optional[str] = None) -> AMINO2CODON_TYPE`

  Calculate codon frequencies based on a collection of DNA and protein sequences.

- `get_organism_to_codon_frequencies(dataset: pd.DataFrame, organisms: List[str]) -> Dict[str, AMINO2CODON_TYPE]`

  Generate a dictionary mapping each organism to its codon frequency distribution.
<br></br>


## CodonPrediction

The CodonPrediction subpackage is an essential component of CodonTransformer, used for tokenizing input, loading models, predicting DNA sequences, and providing helper functions for data processing. It offers a comprehensive toolkit for working with the CodonTransformer model, covering tasks from model loading and configuration to various types of codon optimization and DNA sequence prediction.

### Overview

This subpackage contains functions and classes that handle the core prediction functionality of CodonTransformer. It includes tools for working with the BigBird transformer model, tokenization, and various codon optimization strategies.

### Available Functions and Classes

- `predict_dna_sequence(protein: str, organism: Union[int, str], device: torch.device, tokenizer: Union[str, PreTrainedTokenizerFast], model: Union[str, torch.nn.Module], attention_type: str = "original_full", deterministic: bool = True, temperature: float = 0.2, top_p: float = 0.95, num_sequences: int = 1, match_protein: bool = False) -> DNASequencePrediction`

  Predict the DNA sequence for a given protein using the CodonTransformer model.

- `sample_non_deterministic(logits: torch.Tensor, temperature: float = 0.2, top_p: float = 0.95) -> List[int]`

  Sample token indices from logits using temperature scaling and nucleus (top-p) sampling.

- `load_model(path: str, device: torch.device = None, num_organisms: int = None, remove_prefix: bool = True, attention_type: str = "original_full") -> torch.nn.Module`

  Load a BigBirdForMaskedLM model from a file or checkpoint.

- `load_bigbird_config(num_organisms: int) -> BigBirdConfig`

  Load the configuration object used to train the BigBird transformer.

- `create_model_from_checkpoint(checkpoint_dir: str, output_model_dir: str, num_organisms: int) -> None`

  Save a model to disk using a previous checkpoint.

- `load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast`

  Create and return a tokenizer object from the given tokenizer path.

- `tokenize(batch: List[Dict[str, Any]], tokenizer_path: str = "", tokenizer_object: Optional[PreTrainedTokenizerFast] = None, max_len: int = 2048) -> BatchEncoding`

  Tokenize sequences given a batch of input data.

- `validate_and_convert_organism(organism: Union[int, str]) -> Tuple[int, str]`

  Validate and convert the organism input to both ID and name.

- `get_high_frequency_choice_sequence(protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> str`

  Return a DNA sequence optimized using the High Frequency Choice (HFC) approach.

- `precompute_most_frequent_codons(codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> Dict[str, str]`

  Precompute the most frequent codon for each amino acid.

- `get_high_frequency_choice_sequence_optimized(protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> str`

  An efficient implementation of the HFC approach, up to 10 times faster than the original.

- `get_background_frequency_choice_sequence(protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> str`

  Return a DNA sequence optimized using the Background Frequency Choice (BFC) approach.

- `precompute_cdf(codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> Dict[str, Tuple[List[str], Any]]`

  Precompute the cumulative distribution function (CDF) for each amino acid.

- `get_background_frequency_choice_sequence_optimized(protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> str`

  An efficient implementation of the BFC approach, up to 8 times faster than the original.

- `get_uniform_random_choice_sequence(protein: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]]) -> str`

  Return a DNA sequence optimized using the Uniform Random Choice (URC) approach.

- `get_icor_prediction(input_seq: str, model_path: str, stop_symbol: str) -> str`

  Return an optimized codon sequence for the given protein sequence using ICOR (Improving Codon Optimization with Recurrent Neural Networks).
<br></br>


## CodonEvaluation

The CodonEvaluation subpackage offers functions for calculating evaluation metrics related to codon usage and DNA sequence analysis, used for assessing the quality and characteristics of DNA sequences, especially in codon optimization. It provides a comprehensive toolkit for evaluating DNA sequences and codon usage, performingng genetic data analysis within the CodonTransformer package.


### Overview

The CodonEvaluation module includes functions to compute metrics such as Codon Adaptation Index (CAI)/Codon Similarity Index (CSI) weights, GC content, codon frequency distribution (CFD), %MinMax, sequence complexity, and sequence similarity. These metrics are valuable for analyzing and comparing DNA sequences across different organisms.

### Available Functions

- `get_organism_to_CAI_weights(dataset: pd.DataFrame, organisms: List[str]) -> Dict[str, dict]`

  Calculate the Codon Adaptation Index (CAI) weights for a list of organisms.

- `get_GC_content(dna: str, lower: bool = False) -> float`

  Compute the GC content of a DNA sequence.

- `get_cfd(dna: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]], threshold: float = 0.3) -> float`

  Calculate the codon frequency distribution (CFD) metric for a DNA sequence.

- `get_min_max_percentage(dna: str, codon_frequencies: Dict[str, Tuple[List[str], List[float]]], window_size: int = 18) -> List[float]`

  Calculate the %MinMax metric for a DNA sequence.

- `get_sequence_complexity(dna: str) -> float`

  Compute the sequence complexity score of a DNA sequence.

- `get_sequence_similarity(original: str, predicted: str, truncate: bool = True, window_length: int = 1) -> float`

  Calculate the sequence similarity between two sequences.
<br></br>


## CodonUtils

The CodonUtils subpackage contains constants and helper functions essential for working with genetic sequences, amino acids, and organism data in the CodonTransformer package. It provides tools for genetic sequence analysis, organism identification, and data processing, forming the foundation for many core functionalities within the CodonTransformer package.

### Contents

#### Constants

- `AMINO_ACIDS`: List of all standard amino acids
- `STOP_SYMBOLS`: List of possible stop symbols to end the protein with
- `AMBIGUOUS_AMINOACID_MAP`: Mapping of ambiguous amino acids to standard amino acids
- `START_CODONS` and `STOP_CODONS`: Lists of start and stop codons
- `TOKEN2INDEX` and `INDEX2TOKEN`: Mappings between tokens and their indices
- `AMINO_ACID_TO_INDEX`: Dictionary mapping each amino acid and stop symbol to indices of codon tokens that translate to it
- `TOKEN2MASK`: Mapping for mask tokens
- `FINE_TUNE_ORGANISMS`: List of organisms used for fine-tuning
- `ORGANISM2ID`: Dictionary mapping organisms to their respective IDs
- `NUM_ORGANISMS`, `MAX_LEN`, `MAX_AMINO_ACIDS`, `STOP_SYMBOL`: Various constants for sequence processing

#### Classes

- `DNASequencePrediction`

  Dataclass for holding DNA sequence prediction outputs.

- `IterableData`

  Base class for iterable datasets in parallel multi-processing environments.

- `IterableJSONData`

  Class for iterating over lines of a JSON file.

- `ConfigManager`

  An abstract base class for managing configuration settings.

- `ProteinConfig`:

  A singleton class derived from `ConfigManager` for managing protein-specific configurations. It allows customization of ambiguous amino acid handling.


#### Functions

- `load_python_object_from_disk(file_path: str) -> Any`

  Load a Pickle object from disk.

- `save_python_object_to_disk(input_object: Any, file_path: str) -> None`

  Save a Python object to disk using Pickle.

- `find_pattern_in_fasta(keyword: str, text: str) -> List[str]`

  Find a specific keyword pattern in text (useful for FASTA sequences).

- `get_organism2id_dict(organism_reference: str) -> Dict[str, int]`

  Get a dictionary mapping organisms to their indices.

- `get_taxonomy_id(taxonomy_reference: str, organism: str, return_dict: bool = False) -> Union[int, Dict[str, int]]`

  Get taxonomy ID for an organism or return the entire mapping.

- `sort_amino2codon_skeleton(amino2codon: Dict[str, List[str]]) -> Dict[str, List[str]]`

  Sort the amino2codon dictionary alphabetically.

- `load_pkl_from_url(url: str) -> Any`

  Download and load a Pickle file from a URL.
<br></br>


## CodonJupyter

The CodonJupyter subpackage offers Jupyter-specific functions for displaying interactive widgets, facilitating user interaction with the CodonTransformer package in a Jupyter notebook environment. It improves the user experience by providing interactive and visually appealing interfaces for input and output.

### Overview

This subpackage can be used for:

- Creating and displaying interactive widgets for organism selection and protein sequence input
- Handling user inputs and storing them in a container
- Formatting and displaying the model output in a visually appealing manner

### Classes and Functions

- `UserContainer`

  A container class to store user inputs for organism and protein sequence.

  **Attributes:**
  - `organism (int)`: The selected organism ID
  - `protein (str)`: The input protein sequence

- `display_organism_dropdown(organism2id: Dict[str, int], container: UserContainer) -> None`

  Display a dropdown widget for selecting an organism from a list and updates the organism ID in the provided container.

- `display_protein_input(container: UserContainer) -> None`

  Display a widget for entering a protein sequence and saves the entered sequence to the container.

- `format_model_output(output: DNASequencePrediction) -> str`

  Format the DNA sequence prediction output in a visually appealing and easy-to-read manner. Take a `DNASequencePrediction` object and return a formatted string.

### Usage
Checkout our [Google Colab Notebook](https://adibvafa.github.io/CodonTransformer/GoogleColab) for an example use case!
<br></br>

## Contribution
We welcome contributions to CodonTransformer! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
<br></br>

## Citation
If you use CodonTransformer or our data in your research, please cite our work:

```sh
@article{Fallahpour_Gureghian_Filion_Lindner_Pandi_2025,
  title={CodonTransformer: a multispecies codon optimizer using context-aware neural networks},
  volume={16},
  ISSN={2041-1723},
  url={https://www.nature.com/articles/s41467-025-58588-7},
  DOI={10.1038/s41467-025-58588-7},
  number={1},
  journal={Nature Communications},
  author={Fallahpour, Adibvafa and Gureghian, Vincent and Filion, Guillaume J. and Lindner, Ariel B. and Pandi, Amir},
  year={2025},
  month=apr,
  pages={3205},
  language={en}
}
```
