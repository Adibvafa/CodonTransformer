<p align="center">
  <img src="src/codontransformer2.png" alt="CodonTransformer Logo" width="100%" height="100%" style="vertical-align: middle;"/>
</p>

<p align="center">
    <a href="https://adibvafa.github.io/CodonTransformer/"><img alt="Documentation" src="https://img.shields.io/website/http/adibvafa.github.io/CodonTransformer/index.svg?color=00B89E&down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/adibvafa/CodonTransformer/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/adibvafa/CodonTransformer.svg?color=E80070"></a>
    <a href="https://github.com/adibvafa/CodonTransformer/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/adibvafa/CodonTransformer.svg?color=00BEE8"></a>
    <a href="https://pypi.org/project/CodonTransformer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/CodonTransformer?color=E8C800"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg?color=E8C800" alt="DOI"></a>
</p>

## Table of Contents
TODO
<br></br>

## Overview
CodonTransformer is the ultimate tool for codon optimization, transforming protein sequences into optimized DNA sequences specific for your target organisms. Whether you are a researcher or a practitioner in genetic engineering, CodonTransformer provides a comprehensive suite of features to facilitate your work. By leveraging the Transformer architecture and a user-friendly Jupyter notebook, it reduces the complexity of codon optimization, saving you time and effort.
<br></br>


## Use Case
**For an interactive demo, check out our [Google Colab Notebook.](https://adibvafa.github.io/CodonTransformer/GoogleColab)**

After installing CodonTransformer, you can use:
```python
import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import ORGANISM2ID
from CodonTransformer.CodonJupyter import format_model_output
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(DEVICE)


# Set your input data
protein = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG"
organism = "Escherichia coli general"


# Predict with CodonTransformer
output = predict_dna_sequence(
    protein=protein,
    organism=organism,
    device=DEVICE,
    tokenizer_object=tokenizer,
    model_object=model,
    attention_type="original_full",
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

<br></br>


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

The package requires `python>=3.9`. The requirements are [availabe here](requirements.txt).
<br></br>


## Key Features
- **CodonData**: Simplifies the process of gathering, preprocessing, and representing data, ensuring you have the clean and well-structured data needed for your research.
- **CodonEvaluation**: Offers robust scripts to evaluate various metrics for codon optimized sequences, helping you assess the quality and effectiveness of your optimizations.
- **CodonPrediction**: Enables easy tokenization, loading, and utilization of CodonTransformer for making predictions. It includes various approaches such as High Frequency Choice (HFC), Background Frequency Choice (BFC), random selection, and ICOR.
- **CodonUtils**: Provides essential utility functions and constants that streamline your workflow and improve efficiency.
- **CodonJupyter**: Comes with tools for creating demo notebooks, allowing you to quickly set up and demonstrate the capabilities of CodonTransformer in an interactive environment.
<br></br>


## Contribution
We welcome contributions to CodonTransformer! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
<br></br>

## Citation
If you use CodonTransformer or our data in your research, please cite our work:

```sh
TBD
```
