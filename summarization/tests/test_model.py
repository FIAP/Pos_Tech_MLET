import unittest
from unittest.mock import MagicMock, patch

from src.model import HuggingFaceModel  # Adjust import path as needed


class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.task = "summarization"
        self.model = HuggingFaceModel(
            model_name=self.model_name,
            task=self.task,
        )

    @patch("src.model.pipeline")
    @patch("torch.cuda.is_available", return_value=True)
    def test_load_context_gpu(self, mock_cuda_available, mock_pipeline):
        """Test `load_context` when GPU is available."""
        mock_pipeline.return_value = MagicMock()

        self.model.load_context(context=None)

        mock_cuda_available.assert_called_once()
        mock_pipeline.assert_called_once_with(
            task=self.task,
            model=self.model_name,
            device=0,  # GPU
            truncation=True,
        )
        self.assertIsNotNone(self.model.pipeline)

    @patch("src.model.pipeline")
    @patch("torch.cuda.is_available", return_value=False)
    def test_load_context_cpu(self, mock_cuda_available, mock_pipeline):
        """Test `load_context` when GPU is not available."""
        mock_pipeline.return_value = MagicMock()

        self.model.load_context(context=None)

        mock_cuda_available.assert_called_once()
        mock_pipeline.assert_called_once_with(
            task=self.task,
            model=self.model_name,
            device=-1,  # CPU
            truncation=True,
        )
        self.assertIsNotNone(self.model.pipeline)

    @patch("src.model.pipeline")
    def test_predict_with_string(self, mock_pipeline):
        """Test `predict` with a string input."""
        mock_pipeline.return_value = MagicMock()
        self.model.load_context(context=None)

        self.model.pipeline = MagicMock()
        self.model.pipeline.return_value = [{"summary_text": "Summarized text"}]

        result = self.model.predict(context=None, model_input="Test input")

        self.model.pipeline.assert_called_once_with("Test input", truncation=True)
        self.assertEqual(result, ["Summarized text"])

    @patch("src.model.pipeline")
    def test_predict_with_list(self, mock_pipeline):
        """Test `predict` with a list input."""
        mock_pipeline.return_value = MagicMock()
        self.model.load_context(context=None)

        self.model.pipeline = MagicMock()
        self.model.pipeline.return_value = [
            {"summary_text": "Summarized text 1"},
            {"summary_text": "Summarized text 2"},
        ]

        result = self.model.predict(context=None, model_input=["Input 1", "Input 2"])

        self.model.pipeline.assert_called_once_with(
            ["Input 1", "Input 2"], truncation=True
        )
        self.assertEqual(result, ["Summarized text 1", "Summarized text 2"])

    @patch("src.model.pipeline")
    def test_predict_with_dataframe(self, mock_pipeline):
        """Test `predict` with a pandas DataFrame input."""
        import pandas as pd

        mock_pipeline.return_value = MagicMock()
        self.model.load_context(context=None)

        self.model.pipeline = MagicMock()
        self.model.pipeline.return_value = [
            {"summary_text": "Summarized text 1"},
            {"summary_text": "Summarized text 2"},
        ]

        test_df = pd.DataFrame({"text": ["Input 1", "Input 2"]})
        result = self.model.predict(context=None, model_input=test_df)

        self.model.pipeline.assert_called_once_with(
            ["Input 1", "Input 2"], truncation=True
        )
        self.assertEqual(result, ["Summarized text 1", "Summarized text 2"])


if __name__ == "__main__":
    unittest.main()
