"""Module responsible for Experiment methods."""

from typing import Dict
import mlflow
from mlflow.models.model import ModelInfo
from datasets import load_dataset
import pandas as pd

class Experiment:
    """Experiment."""
    def __init__(self, model: object):
        """Initialize Experiment.

        Args:
            model (object): model to experiment with.
        """
        self.model = model

    def track(self, title: str, run_name: str, **kwargs) -> ModelInfo:
        """Track experiment to mlflow.

        Args:
            title (str): Experiment title.
            run_name (str): Experiment run name.

        Returns:
            ModelInfo: Model info.
        """
        mlflow.set_experiment(title)

        signature = mlflow.models.infer_signature(
            model_input="What are the three primary colors?",
            model_output="The three primary colors are red, yellow, and blue.",
        )

        with mlflow.start_run(run_name=run_name):
            model_info = mlflow.pyfunc.log_model(
                python_model=self.model,
                signature=signature,
                artifact_path="model",
                pip_requirements=["-r requirements/requirements.txt"],
                code_paths=["src"],
                **kwargs,
            )
        return model_info

    def load_test_dataset(self, max_size: int) -> pd.DataFrame:
        """Load test dataset.

        Args:
            max_size (int): Maximum quantity of rows to retrieve.

        Returns:
            pd.DataFrame: Test dataset
        """
        dataset = (
            load_dataset(
                "billsum", split="ca_test"
            ).select(range(max_size)).to_pandas()
        )
        return dataset.rename(columns={"text": "inputs"})

    def evaluate(self, model_uri: str, max_size: int = 20) -> Dict[str, int]:
        """Evaluate model.

        Args:
            model_uri (str): MLFlow model uri for evaluation.
            max_size (int, optional): Maximum quantity of rows to retrieve
                for test dataset. Defaults to 20.

        Returns:
            Dict[str, int]: Metrics.
        """
        run_id = model_uri.split("/")[1]
        with mlflow.start_run(run_id=run_id):
            results = mlflow.evaluate(
                model_uri,
                self.load_test_dataset(max_size=max_size),
                evaluators="default",
                model_type="text-summarization",
                targets="summary",
                extra_metrics=[mlflow.metrics.latency()],
            )
        return results.metrics
