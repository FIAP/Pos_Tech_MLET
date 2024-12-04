
import mlflow
import pandas as pd
import bentoml


class Register():
    """Register."""
    def __init__(self, title: str):
        """Initialize Register.

        Args:
            title (str): Experiment title.
        """
        self.title = title

    def search_finished_runs(self, run_name: str, **kwargs) -> pd.DataFrame:
        """Search finished runs.

        Args:
            run_name (str): Run name.

        Returns:
            pd.DataFrame
        """
        filters = (
            f"attributes.run_name = '{run_name}')" +
            "and attributes.status = 'FINISHED'"
        )
        return mlflow.search_runs(
            experiment_names=[self.title],
            filter_string=filters,
            **kwargs
        )

    def register_model(self, run_id: str):
        """Register model in mlflow and bentoml.

        Args:
            run_id (str): Run id.
        """
        model_uri = f'runs:/{run_id}/model'
        mlflow.register_model(
            model_uri,
            name=self.title,
            tags={"status": "demo", "owner": "renata-gotler"}
        )
        bentoml.mlflow.import_model(self.title, model_uri)
