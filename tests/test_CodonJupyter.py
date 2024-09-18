import unittest
import ipywidgets
from CodonTransformer.CodonJupyter import (
    UserContainer,
    create_organism_dropdown,
    create_dropdown_options,
    display_organism_dropdown,
    display_protein_input,
    format_model_output,
    DNASequencePrediction,
)
from CodonTransformer.CodonUtils import ORGANISM2ID


class TestCodonJupyter(unittest.TestCase):
    def test_UserContainer(self):
        user_container = UserContainer()
        self.assertEqual(user_container.organism, -1)
        self.assertEqual(user_container.protein, "")

    def test_create_organism_dropdown(self):
        container = UserContainer()
        dropdown = create_organism_dropdown(container)

        self.assertIsInstance(dropdown, ipywidgets.Dropdown)
        self.assertGreater(len(dropdown.options), 0)
        self.assertEqual(dropdown.description, "")
        self.assertEqual(dropdown.layout.width, "40%")
        self.assertEqual(dropdown.layout.margin, "0 0 10px 0")
        self.assertEqual(dropdown.style.description_width, "initial")

        # Test the dropdown options
        options = dropdown.options
        self.assertIn("", options)
        self.assertIn("Selected Organisms", options)
        self.assertIn("All Organisms", options)

    def test_create_dropdown_options(self):
        options = create_dropdown_options(ORGANISM2ID)
        self.assertIsInstance(options, list)
        self.assertGreater(len(options), 0)

    def test_display_organism_dropdown(self):
        container = UserContainer()
        with unittest.mock.patch(
            "CodonTransformer.CodonJupyter.display"
        ) as mock_display:
            display_organism_dropdown(container)

        # Check that display was called twice (for container_widget and HTML)
        self.assertEqual(mock_display.call_count, 2)

        # Check that the first call to display was with a VBox widget
        self.assertIsInstance(mock_display.call_args_list[0][0][0], ipywidgets.VBox)

        # Check that the VBox contains a Dropdown
        dropdown = mock_display.call_args_list[0][0][0].children[1]
        self.assertIsInstance(dropdown, ipywidgets.Dropdown)
        self.assertGreater(len(dropdown.options), 0)

    def test_display_protein_input(self):
        container = UserContainer()
        with unittest.mock.patch(
            "CodonTransformer.CodonJupyter.display"
        ) as mock_display:
            display_protein_input(container)

        # Check that display was called twice (for container_widget and HTML)
        self.assertEqual(mock_display.call_count, 2)

        # Check that the first call to display was with a VBox widget
        self.assertIsInstance(mock_display.call_args_list[0][0][0], ipywidgets.VBox)

        # Check that the VBox contains a Textarea
        textarea = mock_display.call_args_list[0][0][0].children[1]
        self.assertIsInstance(textarea, ipywidgets.Textarea)

        # Verify the properties of the Textarea
        self.assertEqual(textarea.value, "")
        self.assertEqual(textarea.placeholder, "Enter here...")
        self.assertEqual(textarea.description, "")
        self.assertEqual(textarea.layout.width, "100%")
        self.assertEqual(textarea.layout.height, "100px")
        self.assertEqual(textarea.layout.margin, "0 0 10px 0")
        self.assertEqual(textarea.style.description_width, "initial")

    def test_format_model_output(self):
        output = DNASequencePrediction(
            organism="Escherichia coli",
            protein="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            processed_input="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            predicted_dna="ATGAAAACTGTTCGTCAGGAACGTCTGAAATCTATTGTTCGTATTCTGGAACGTTCTAAAGAACCGGTTTCTGGTGCTCAACTGGCTGAAGAACTGTCTGTTTCTCGTCAGGTTATTGTTCAGGACATTGCTTACCTGCGTTCTCTGGGTTATAA",
        )
        formatted_output = format_model_output(output)
        self.assertIsInstance(formatted_output, str)
        self.assertIn("Organism", formatted_output)
        self.assertIn("Escherichia coli", formatted_output)
        self.assertIn("Input Protein", formatted_output)
        self.assertIn(
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            formatted_output,
        )
        self.assertIn("Processed Input", formatted_output)
        self.assertIn(
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            formatted_output,
        )
        self.assertIn("Predicted DNA", formatted_output)
        self.assertIn(
            "ATGAAAACTGTTCGTCAGGAACGTCTGAAATCTATTGTTCGTATTCTGGAACGTTCTAAAGAACCGGTTTCTGGTGCTCAACTGGCTGAAGAACTGTCTGTTTCTCGTCAGGTTATTGTTCAGGACATTGCTTACCTGCGTTCTCTGGGTTATAA",
            formatted_output,
        )


if __name__ == "__main__":
    unittest.main()
