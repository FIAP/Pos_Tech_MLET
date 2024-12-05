"""Module responsible for serving with BentoML."""

from __future__ import annotations

import bentoml

EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.'"


@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 10},
    monitoring={"enabled": True},
    metrics={
        "enabled": True,
        "namespace": "bentoml_service",
    },
)
class Summarization:
    """Summarization."""

    bento_model = bentoml.models.get("summarization:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(batchable=True)
    def summarize(self, texts: list[str] = EXAMPLE_INPUT) -> list[str]:
        """Summarize texts.

        Args:
            texts (list[str]): Texts to be summarized.

        Returns:
            list[str]: Summarized texts.
        """
        with bentoml.monitor("text_summarization") as mon:
            mon.log(texts, name="request", role="input", data_type="list")

            summary_texts = self.model.predict(texts)
            mon.log(summary_texts, name="response", role="prediction", data_type="list")
            return summary_texts
