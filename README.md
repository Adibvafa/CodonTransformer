<p align="center">
  <img src="src/codontransformer.png" alt="CodonTransformer Logo" width="100%" height="100%" style="vertical-align: middle;"/>
</p>

<p align="center">
    <a href="https://adibvafa.github.io/CodonTransformer/"><img alt="Documentation" src="https://img.shields.io/website/http/adibvafa.github.io/CodonTransformer/index.svg?color=00B89E&down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/adibvafa/CodonTransformer/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/adibvafa/CodonTransformer.svg?color=E80070"></a>
    <a href="https://github.com/adibvafa/CodonTransformer/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/adibvafa/CodonTransformer.svg?color=00BEE8"></a>
    <a href="https://pypi.org/project/CodonTransformer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/CodonTransformer?color=E8C800"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg?color=E8C800" alt="DOI"></a>
</p>

<h3 align="center">
    <p>The Ultimate Tool or Codon Optimization.</p>
</h3>


## Table of Contents
TODO

## Overview
CodonTransformer is the ultimate tool for codon optimization, transforming protein sequences into optimized DNA sequences specific for your target organisms. Whether you are a researcher or a practitioner in genetic engineering, CodonTransformer provides a comprehensive suite of features to facilitate your work. CodonTransformer is built to make codon optimization accessible and efficient. By leveraging the Transformer architecture and a user-friendly Jupyter notebook, it reduces the complexity of genetic sequence optimization, saving you time and effort.


## Use Case
```python
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import ORGANISM2ID
from CodonTransformer.CodonJupyter import (
    UserContainer,
    display_protein_input,
    display_organism_dropdown,
    format_model_output,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(DEVICE)

# Example use case
output = predict_dna_sequence(
    protein="MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGG",
    organism="Escherichia coli general",
    device=DEVICE,
    tokenizer_object=tokenizer,
    model_object=model,
    attention_type="original_full",
)
print(format_model_output(output))
```
For a more interactive experience, check out our [Google Colab Notebook](https://adibvafa.github.io/CodonTransformer/GoogleColab)

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

## Key Features

- **CodonData**: Simplifies the process of gathering, preprocessing, and representing data, ensuring you have the clean and well-structured data needed for your research.
- **CodonEvaluation**: Offers robust scripts to evaluate various metrics for codon optimized sequences, helping you assess the quality and effectiveness of your optimizations.
- **CodonPrediction**: Enables easy tokenization, loading, and utilization of CodonTransformer for making predictions. It includes various approaches such as High Frequency Choice (HFC), Background Frequency Choice (BFC), random selection, and ICOR.
- **CodonUtils**: Provides essential utility functions and constants that streamline your workflow and improve efficiency.
- **CodonJupyter**: Comes with tools for creating demo notebooks, allowing you to quickly set up and demonstrate the capabilities of CodonTransformer in an interactive environment.


## Contribution
We welcome contributions to CodonTransformer! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you use CodonTransformer or our data in your research, please cite our work:

```sh
@misc{CodonTransformer2024,
  author = {Adibvafa, Fallahpour},
  title = {CodonTransformer: An Ultimate Tool for Codon Optimization},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/adibvafa/CodonTransformer}},
}
```
