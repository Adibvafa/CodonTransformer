# setup.py

from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="CodonTransformer",
    version="1.2.4",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Adibvafa Fallahpour",
    author_email="Adibvafa.fallahpour@mail.utoronto.ca",
    description="The ultimate tool for codon optimization, transforming protein sequences into optimized DNA sequences specific for your target organisms.",
    long_description=(
        "CodonTransformer is ultimate tool for codon optimization, transforming protein sequences into optimized DNA sequences specific for your target organisms. "
        "Whether you are a researcher or a practitioner in genetic engineering, CodonTransformer provides a comprehensive suite of features to facilitate your work.\n\n"
        "### Key Features\n\n"
        "- **CodonData**: Simplifies the process of gathering, preprocessing, and representing data, ensuring you have the clean and well-structured data needed for your training.\n"
        "- **CodonEvaluation**: Offers robust scripts to evaluate various metrics for codon optimized sequences, helping you assess the quality and effectiveness of your optimizations.\n"
        "- **CodonPrediction**: Enables easy tokenization, loading, and utilization of CodonTransformer for making predictions. It includes various approaches such as High Frequency Choice (HFC), Background Frequency Choice (BFC), random selection, and ICOR.\n"
        "- **CodonUtils**: Provides essential utility functions and constants that streamline your workflow and improve efficiency.\n"
        "- **CodonJupyter**: Comes with tools for creating demo notebooks, allowing you to quickly set up and demonstrate the capabilities of CodonTransformer in an interactive environment.\n\n"
        "### Why Choose CodonTransformer?\n\n"
        "CodonTransformer is built to make codon optimization accessible and efficient. By leveraging advanced algorithms and a user-friendly interface, it reduces the complexity of genetic sequence optimization, saving you time and effort."
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/adibvafa/CodonTransformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
