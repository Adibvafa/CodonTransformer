"""
File: CodonJupyter.py
---------------------
Includes Jupyter-specific functions for displaying interactive widgets.
"""

from typing import Dict, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from CodonTransformer.CodonUtils import FINE_TUNE_ORGANISMS


class UserContainer:
    """
    A container class to store user inputs for organism and protein sequence.
    Attributes:
        organism (Optional[str]): The selected organism name.
        organism_id (Optional[int]): The ID corresponding to the selected organism.
        protein_sequence (str): The input protein sequence.
        predicted_dna (str): The predicted DNA sequence.
    """
    def __init__(self) -> None:
        self.organism: Optional[str] = None
        self.organism_id: Optional[int] = None
        self.protein_sequence: str = ''
        self.predicted_dna: str = ''


def display_organism_dropdown(organism2id: Dict[str, int], container: UserContainer) -> None:
    """
    Display a dropdown widget for selecting an organism from a list and 
    update the organism ID in the provided container.

    Args:
        organism2id (Dict[str, int]): A dictionary mapping organism names to their IDs.
        container (UserContainer): A container to store the selected organism and its ID.
    """
    organism_names = sorted(organism2id.keys())

    # Modify the names of commonly used organisms for better display
    organism_names_styled = [
        ('          ' + organism) if organism in FINE_TUNE_ORGANISMS else organism
        for organism in organism_names
    ]

    # Create a dropdown widget with organism names
    organism_dropdown = widgets.Dropdown(
        options=[''] + organism_names_styled,
        description='',
        layout=widgets.Layout(width='40%', margin='0 0 10px 0'),
        style={'description_width': 'initial'}
    )

    # Custom CSS for the dropdown widget
    dropdown_style = """
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
            .bold-option {
                font-weight: normal;
            }
        </style>
    """

    # Output widget to display the result
    output = widgets.Output()

    # Function to display the corresponding ID and update the container
    def show_organism_id(change: Dict[str, str]) -> None:
        """
        Display the corresponding ID and update the container with the selected organism.

        Args:
            change (Dict[str, str]): A dictionary containing information about the change in dropdown value.
        """
        organism = change['new']
        if organism != '':
            organism = organism.strip()
            organism_id = organism2id.get(organism, None)
            with output:
                clear_output()
            if organism_id is not None:
                container.organism = organism
                container.organism_id = organism_id
        else:
            with output:
                clear_output()
            container.organism_id = None

    # Attach the function to the dropdown
    organism_dropdown.observe(show_organism_id, names='value')

    # Display the dropdown widget and the output
    header = widgets.HTML('<b style="font-size:20px;">Select Organism:</b><div style="height:10px;"></div>')
    container_widget = widgets.VBox([header, organism_dropdown, output], layout=widgets.Layout(padding='12px 0 0 25px'))

    # Apply custom styles
    display(container_widget)
    display(widgets.HTML(dropdown_style))


def display_protein_sequence_input(container: UserContainer) -> None:
    """
    Display a widget for entering a protein sequence and save it to the container.

    Args:
        container (UserContainer): A container to store the entered protein sequence.
    """
    protein_input = widgets.Textarea(
        value='',
        placeholder='Enter here...',
        description='',
        layout=widgets.Layout(width='100%', height='100px', margin='0 0 10px 0'),
        style={'description_width': 'initial'}
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
    def save_protein_sequence(change: Dict[str, str]) -> None:
        """
        Save the input protein sequence to the container.

        Args:
            change (Dict[str, str]): A dictionary containing information about the change in textarea value.
        """
        container.protein_sequence = change['new'].upper().strip().replace('\n', '').replace(' ', '').replace('\t', '')

    # Attach the function to the input widget
    protein_input.observe(save_protein_sequence, names='value')

    # Display the input widget
    header = widgets.HTML('<b style="font-size:20px;">Enter Protein Sequence:</b><div style="height:18px;"></div>')
    container_widget = widgets.VBox([header, protein_input], layout=widgets.Layout(padding='12px 12px 0 25px'))

    display(container_widget)
    display(widgets.HTML(input_style))
