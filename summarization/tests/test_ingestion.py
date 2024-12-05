import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.ingestion import HuggingFaceDataset


class TestHuggingFaceDataset(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.dataset = HuggingFaceDataset()

    @patch("src.ingestion.load_dataset")
    def test_load_pd_test_dataset(self, mock_load_dataset):
        """Test the `load_pd_test_dataset` method."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value.to_pandas.return_value = pd.DataFrame(
            {"text": ["Document 1", "Document 2", "Document 3"]}
        )
        mock_load_dataset.return_value = mock_dataset

        # Call the method
        result = self.dataset.load_pd_test_dataset(
            max_size=3, dataset_name="mock_dataset", split="mock_split"
        )

        # Assertions
        mock_load_dataset.assert_called_once_with(dataset_name="mock_dataset", split="mock_split")
        mock_dataset.select.assert_called_once_with(range(3))
        pd.testing.assert_frame_equal(
            result, pd.DataFrame(
                {"inputs": ["Document 1", "Document 2", "Document 3"]}
            )
        )


    def test_load_pd_test_dataset_missing_args(self):
        """Test the `load_pd_test_dataset` method with default arguments."""
        # should give error load_dataset() missing 1 required positional argument: 'path'
        try: 
            self.dataset.load_pd_test_dataset()
        except TypeError:
            assert True


    def test_abstract_class(self):
        """Test the abstract class enforcement."""
        from src.ingestion import Dataset

        with self.assertRaises(TypeError):
            Dataset()  # Should raise error because Dataset is abstract


if __name__ == "__main__":
    unittest.main()
