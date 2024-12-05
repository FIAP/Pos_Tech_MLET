import unittest
from unittest.mock import MagicMock, patch

from src.experiment import Experiment


class TestExperiment(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.mock_model = MagicMock()
        self.mock_model.model_name = "mock_model"
        self.mock_model.task = "classification"
        self.experiment = Experiment(model=self.mock_model, title="Mock Experiment")

    @patch("mlflow.set_experiment")
    @patch("mlflow.models.infer_signature")
    @patch("mlflow.pyfunc.log_model")
    @patch("mlflow.log_params")
    @patch("mlflow.start_run")
    def test_track(
        self,
        mock_start_run,
        mock_log_params,
        mock_log_model,
        mock_infer_signature,
        mock_set_experiment,
    ):
        """Test the `track` method."""
        mock_signature = MagicMock()
        mock_infer_signature.return_value = mock_signature
        mock_model_info = MagicMock()
        mock_log_model.return_value = mock_model_info

        result = self.experiment.track(run_name="test_run")

        mock_set_experiment.assert_called_once_with("Mock Experiment")
        mock_infer_signature.assert_called_once_with(
            model_input="What are the three primary colors?",
            model_output="The three primary colors are red, yellow, and blue.",
        )
        mock_log_model.assert_called_once_with(
            python_model=self.mock_model,
            signature=mock_signature,
            artifact_path="model",
            pip_requirements=["-r requirements/requirements.txt"],
            code_paths=["src"],
        )
        mock_log_params.assert_called_once_with(
            {"model_name": "mock_model", "task": "classification"}
        )
        self.assertEqual(result, mock_model_info)

    @patch("mlflow.metrics.latency")
    @patch("mlflow.evaluate")
    @patch("mlflow.start_run")
    def test_evaluate(self, mock_start_run, mock_evaluate, mock_latency):
        """Test the `evaluate` method."""
        mock_results = MagicMock()
        metrics = {"accuracy": 0.95, "latency": 200}
        mock_results.metrics = metrics
        mock_evaluate.return_value = mock_results
        test_df = MagicMock()
        result = self.experiment.evaluate(
            model_uri="runs:/12345/model", test_df=test_df
        )

        mock_start_run.assert_called_once_with(run_id="12345")
        mock_evaluate.assert_called_once_with(
            "runs:/12345/model",
            test_df,
            evaluators="default",
            model_type="text-summarization",
            targets="summary",
            extra_metrics=[mock_latency()],
        )
        self.assertEqual(result, metrics)

    @patch("mlflow.search_runs")
    def test_search_finished_experiments(self, mock_search_runs):
        """Test the `search_runs` method."""
        mock_runs = MagicMock()
        mock_search_runs.return_value = mock_runs

        result = self.experiment.search_finished_experiments(run_name="test_run")

        mock_search_runs.assert_called_once_with(
            experiment_names=["Mock Experiment"],
            filter_string="attributes.run_name = 'test_run' and attributes.status = 'FINISHED'",
        )
        self.assertEqual(result, mock_runs)


if __name__ == "__main__":
    unittest.main()
