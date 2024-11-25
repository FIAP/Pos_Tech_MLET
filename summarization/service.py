"""Module responsible for serving with BentoML"""
from __future__ import annotations

import bentoml


@bentoml.service(resources={"cpu": "4"})
class Summarization:
    """Summarization."""
    bento_model = bentoml.models.get("summarization:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(batchable=True)
    def summarize(self, texts: list[str]) -> list[str]:
        """Summarize texts.

        Args:
            texts (list[str]): Texts to be summarized.

        Returns:
            list[str]: Summarized texts.
        """
        return self.model.predict(texts)
