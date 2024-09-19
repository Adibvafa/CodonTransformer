"""
File: CodonJupyter.py
---------------------
Includes Jupyter-specific functions for displaying interactive widgets.
"""

from typing import Dict, List, Tuple

import ipywidgets as widgets
from IPython.display import HTML, display

from CodonTransformer.CodonUtils import (
    COMMON_ORGANISMS,
    ID2ORGANISM,
    ORGANISM2ID,
    DNASequencePrediction,
)


class UserContainer:
    """
    A container class to store user inputs for organism and protein sequence.
    Attributes:
        organism (int): The selected organism id.
        protein (str): The input protein sequence.
    """

    def __init__(self) -> None:
        self.organism: int = -1
        self.protein: str = ""


def create_styled_options(
    organisms: list, organism2id: Dict[str, int], is_fine_tuned: bool = False
) -> list:
    """
    Create styled options for the dropdown widget.

    Args:
        organisms (list): List of organism names.
        organism2id (Dict[str, int]): Dictionary mapping organism names to their IDs.
        is_fine_tuned (bool): Whether these are fine-tuned organisms.

    Returns:
        list: Styled options for the dropdown widget.
    """
    styled_options = []
    for organism in organisms:
        organism_id = organism2id[organism]
        if is_fine_tuned:
            if organism_id < 10:
                styled_options.append(f"\u200b{organism_id:>6}.  {organism}")
            elif organism_id < 100:
                styled_options.append(f"\u200b{organism_id:>5}.  {organism}")
            else:
                styled_options.append(f"\u200b{organism_id:>4}.  {organism}")
        else:
            if organism_id < 10:
                styled_options.append(f"{organism_id:>6}.  {organism}")
            elif organism_id < 100:
                styled_options.append(f"{organism_id:>5}.  {organism}")
            else:
                styled_options.append(f"{organism_id:>4}.  {organism}")
    return styled_options


def create_dropdown_options(organism2id: Dict[str, int]) -> list:
    """
    Create the full list of dropdown options, including section headers.

    Args:
        organism2id (Dict[str, int]): Dictionary mapping organism names to their IDs.

    Returns:
        list: Full list of dropdown options.
    """
    fine_tuned_organisms = sorted(
        [org for org in organism2id.keys() if org in COMMON_ORGANISMS]
    )
    all_organisms = sorted(organism2id.keys())

    fine_tuned_options = create_styled_options(
        fine_tuned_organisms, organism2id, is_fine_tuned=True
    )
    all_organisms_options = create_styled_options(
        all_organisms, organism2id, is_fine_tuned=False
    )

    return (
        [""]
        + ["Selected Organisms"]
        + fine_tuned_options
        + [""]
        + ["All Organisms"]
        + all_organisms_options
    )


def create_organism_dropdown(container: UserContainer) -> widgets.Dropdown:
    """
    Create and configure the organism dropdown widget.

    Args:
        container (UserContainer): Container to store the selected organism.

    Returns:
        widgets.Dropdown: Configured dropdown widget.
    """
    dropdown = widgets.Dropdown(
        options=create_dropdown_options(ORGANISM2ID),
        description="",
        layout=widgets.Layout(width="40%", margin="0 0 10px 0"),
        style={"description_width": "initial"},
    )

    def show_organism(change: Dict[str, str]) -> None:
        """
        Update the container with the selected organism and print to terminal.

        Args:
            change (Dict[str, str]): Information about the change in dropdown value.
        """
        dropdown_choice = change["new"]
        if dropdown_choice and dropdown_choice not in [
            "Selected Organisms",
            "All Organisms",
        ]:
            organism = "".join(filter(str.isdigit, dropdown_choice))
            organism_id = ID2ORGANISM[int(organism)]
            container.organism = organism_id
        else:
            container.organism = None

    dropdown.observe(show_organism, names="value")
    return dropdown


def get_dropdown_style() -> str:
    """
    Return the custom CSS style for the dropdown widget.

    Returns:
        str: CSS style string.
    """
    return """
    <style>
        .widget-dropdown > select {
            font-size: 16px;
            font-weight: normal;
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 5px;
        }
        .widget-label {
            font-size: 18px;
            font-weight: bold;
        }
        .custom-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .widget-dropdown option[value^="\u200b"] {
            font-family: sans-serif;
            font-weight: bold;
            font-size: 18px;
            padding: 510px;
        }
        .widget-dropdown option[value*="Selected Organisms"],
        .widget-dropdown option[value*="All Organisms"] {
            text-align: center;
            font-family: Arial, sans-serif;
            font-weight: bold;
            font-size: 20px;
            color: #6900A1;
            background-color: #00D8A1;
        }
    </style>
    """


def display_organism_dropdown(container: UserContainer) -> None:
    """
    Display the organism dropdown widget and apply custom styles.

    Args:
        container (UserContainer): Container to store the selected organism.
    """
    dropdown = create_organism_dropdown(container)
    header = widgets.HTML(
        '<b style="font-size:20px;">Select Organism:</b>'
        '<div style="height:10px;"></div>'
    )
    container_widget = widgets.VBox(
        [header, dropdown],
        layout=widgets.Layout(padding="12px 0 12px 25px"),
    )
    display(container_widget)
    display(HTML(get_dropdown_style()))


def display_protein_input(container: UserContainer) -> None:
    """
    Display a widget for entering a protein sequence and save it to the container.

    Args:
        container (UserContainer): A container to store the entered protein sequence.
    """
    protein_input = widgets.Textarea(
        value="",
        placeholder="Enter here...",
        description="",
        layout=widgets.Layout(width="100%", height="100px", margin="0 0 10px 0"),
        style={"description_width": "initial"},
    )

    # Custom CSS for the input widget
    input_style = """
        <style>
            .widget-textarea > textarea {
                font-size: 12px;
                font-family: Arial, sans-serif;
                font-weight: normal;
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 10px;
            }
            .widget-label {
                font-size: 18px;
                font-weight: bold;
            }
            .custom-container {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }
        </style>
    """

    # Function to save the input protein sequence to the container
    def save_protein(change: Dict[str, str]) -> None:
        """
        Save the input protein sequence to the container.

        Args:
            change (Dict[str, str]): A dictionary containing information about
            the change in textarea value.
        """
        container.protein = (
            change["new"]
            .upper()
            .strip()
            .replace("\n", "")
            .replace(" ", "")
            .replace("\t", "")
        )

    # Attach the function to the input widget
    protein_input.observe(save_protein, names="value")

    # Display the input widget
    header = widgets.HTML(
        '<b style="font-size:20px;">Enter Protein Sequence:</b>'
        '<div style="height:18px;"></div>'
    )
    container_widget = widgets.VBox(
        [header, protein_input], layout=widgets.Layout(padding="12px 12px 0 25px")
    )

    display(container_widget)
    display(widgets.HTML(input_style))


def format_model_output(output: DNASequencePrediction) -> str:
    """
    Format DNA sequence prediction output in an appealing and easy-to-read manner.

    This function takes the prediction output and formats it into
    a structured string with clear section headers and separators.

    Args:
        output (DNASequencePrediction): Object containing the prediction output.
            Expected attributes:
            - organism (str): The organism name.
            - protein (str): The input protein sequence.
            - processed_input (str): The processed input sequence.
            - predicted_dna (str): The predicted DNA sequence.

    Returns:
        str: A formatted string containing the organized output.
    """

    def format_section(title: str, content: str) -> str:
        """Helper function to format individual sections."""
        separator = "-" * 29
        title_line = f"| {title.center(25)} |"
        return f"{separator}\n{title_line}\n{separator}\n{content}\n\n"

    sections: List[Tuple[str, str]] = [
        ("Organism", output.organism),
        ("Input Protein", output.protein),
        ("Processed Input", output.processed_input),
        ("Predicted DNA", output.predicted_dna),
    ]

    formatted_output = ""
    for title, content in sections:
        formatted_output += format_section(title, content)

    # Remove the last newline to avoid extra space at the end
    return formatted_output.rstrip()
