"""Module responsible for Experiment methods."""

from typing import Dict
import mlflow
from mlflow.models.model import ModelInfo



class Experiment:
    """Experiment."""
    def __init__(self, model: object, title: str):
        """Initialize Experiment.

        Args:
            model (object): model to experiment with.
            title (str): Experiment title.
        """
        self.model = model
        self.title = title

    def track(self, run_name: str, **kwargs) -> ModelInfo:
        """Track experiment to mlflow.

        Args:
            run_name (str): Experiment run name.

        Returns:
            ModelInfo: Model info.
        """
        mlflow.set_experiment(self.title)

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
            mlflow.log_params({"model_name": self.model.model_name, "task": self.model.task})
        return model_info

    def evaluate(self, model_uri: str, test_df: pd.DataFrame) -> Dict[str, int]:
        """Evaluate model.

        Args:
            model_uri (str): MLFlow model uri for evaluation.

        Returns:
            Dict[str, int]: Metrics.
        """
        run_id = model_uri.split("/")[1]
        with mlflow.start_run(run_id=run_id):
            results = mlflow.evaluate(
                model_uri,
                test_df,
                evaluators="default",
                model_type="text-summarization",
                targets="summary",
                extra_metrics=[mlflow.metrics.latency()],
            )
        return results.metrics

