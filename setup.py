# setup.py
import os
from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    # Get the directory where setup.py is located
    here = os.path.abspath(os.path.dirname(__file__))
    # Construct the path to README.md
    readme_path = os.path.join(here, 'README.md')
    
    # Read and return the content of README.md
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="CodonTransformer",
    version="1.2.7",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Adibvafa Fallahpour",
    author_email="Adibvafa.fallahpour@mail.utoronto.ca",
    description="The ultimate tool for codon optimization, transforming protein sequences into optimized DNA sequences specific for your target organisms.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adibvafa/CodonTransformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
