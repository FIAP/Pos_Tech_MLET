# import unittest
# from unittest.mock import patch, MagicMock
# from service import Summarization  # Ensure the correct import path


# class TestSummarizationService(unittest.TestCase):
#     def setUp(self):
#         """Set up test variables."""
#         self.example_input = [
#             "Breaking News: The small town of Willow Creek was taken by storm."
#         ]
#         self.example_output = ["Breaking News: Willow Creek was taken by storm."]

#     @patch("bentoml")
#     def test_initialize_model(self, mock_bentoml):
#         """Test model initialization."""
#         # Arrange
#         mock_model = MagicMock()
#         mock_bentoml.models.get.return_value = "mock_model_path"
#         mock_bentoml.mlflow.load_model.return_value = mock_model

#         # Act
#         summarization_service = Summarization()

#         # Assert
#         print(f"Called bentoml.models.get: {mock_bentoml.models.get.call_count}")  # Debugging
#         mock_bentoml.models.get.assert_called_once_with("summarization:latest")
#         mock_bentoml.mlflow.load_model.assert_called_once_with("mock_model_path")
#         self.assertEqual(summarization_service.model, mock_model)

# @patch("service.bentoml.models.get", autospec=True)
# @patch("service.bentoml.mlflow.load_model", autospec=True)
# def test_summarize(self, mock_load_model, mock_get_model):
#     """Test the summarize method."""
#     # Arrange
#     mock_model = MagicMock()
#     mock_model.predict.return_value = self.example_output
#     mock_get_model.return_value = "mock_model_path"
#     mock_load_model.return_value = mock_model

#     summarization_service = Summarization()

#     # Act
#     result = summarization_service.summarize(texts=self.example_input)

#     # Assert
#     mock_model.predict.assert_called_once_with(self.example_input)
#     self.assertEqual(result, self.example_output)


# if __name__ == "__main__":
#     unittest.main()
