import bentoml
import mlflow


class Register:
    """Register."""

    def __init__(self, title: str):
        """Initialize Register.

        Args:
            title (str): Experiment title.
        """
        self.title = title

    def register_model(self, run_id: str):
        """Register model in mlflow and bentoml.

        Args:
            run_id (str): Run id.
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(
            model_uri,
            name=self.title,
            tags={"status": "demo", "owner": "renata-gotler"},
        )
        bentoml.mlflow.import_model(self.title, model_uri)
