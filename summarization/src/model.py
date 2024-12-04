"""Module responsible for HuggingFaceModel methods"""

from typing import List, Any
import torch
from transformers import pipeline
from mlflow.pyfunc import PythonModel
import mlflow


class HuggingFaceModel(PythonModel):
    """HuggingFaceModel."""
    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        task: str = "summarization",
    ):
        """Initialize HuggingFaceModel.

        Args:
            model_name (str, optional): Model name from hugging face.
                Defaults to "sshleifer/distilbart-cnn-12-6".
            task (str, optional): Task. Defaults to "summarization".
        """
        self.model_name = model_name
        self.task = task

    def load_context(self, context):
        """Load context."""
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            task=self.task,
            model=self.model_name,
            device=device,
            truncation=True,
        )

    @mlflow.trace
    def predict(self, context: Any, model_input: Any):
        """Summarize texts.

        Args:
            model_input: Text, List of texts or Series with texts to be summarized.

        Returns:
            List[str]: Texts summarized.
        """
        if not isinstance(model_input, str) and not isinstance(model_input, list):
            model_input = model_input.iloc[:, 0].values.tolist()
        summaries = self.pipeline(model_input, truncation=True)
        return [summary["summary_text"] for summary in summaries]
