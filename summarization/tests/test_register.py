import unittest
from unittest.mock import patch, MagicMock
from src.register import Register


class TestRegister(unittest.TestCase):
    def setUp(self):
        """Set up test variables."""
        self.title = "test_experiment"
        self.run_id = "12345"
        self.model_uri = f"runs:/{self.run_id}/model"

    @patch("src.register.mlflow.register_model")
    @patch("src.register.bentoml.mlflow.import_model")
    def test_register_model(self, mock_import_model, mock_register_model):
        """Test the register_model method."""
        # Arrange
        register_instance = Register(title=self.title)

        # Act
        register_instance.register_model(run_id=self.run_id)

        # Assert
        mock_register_model.assert_called_once_with(
            self.model_uri,
            name=self.title,
            tags={"status": "demo", "owner": "renata-gotler"},
        )
        mock_import_model.assert_called_once_with(self.title, self.model_uri)


if __name__ == "__main__":
    unittest.main()
