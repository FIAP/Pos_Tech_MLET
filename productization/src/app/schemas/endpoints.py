"""endpoints.py

Pydantic request schemas for the LSTM Service API.

Each class in this module represents the JSON body of a specific endpoint,
providing automatic validation, serialisation, and OpenAPI documentation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """Request payload for real-time inference via ``POST /infer``.

    Attributes:
        strategy: Optional strategy/model name to select a specific trained
            artifact.  When omitted the most recently trained model is used.
        sequence: Input sequence with shape ``[seq_len, input_size]``.
        y_true: Observed target value for online quality monitoring.
        y_pred_old: Prediction from the baseline model for quality comparison.

    Note:
        ``y_true`` and ``y_pred_old`` must be supplied **together** to activate
        the quality monitoring pipeline.  Providing only one raises HTTP 400.
    """

    strategy: Optional[str] = Field(
        default=None,
        description="Optional strategy/model name to select a specific trained artifact.",
        examples=["NoProcessingSimple"],
    )
    sequence: list[list[float]] = Field(
        ...,
        description=(
            "Input sequence with shape [seq_len, input_size] used by the trained model."
        ),
    )
    y_true: Optional[float] = Field(
        default=None,
        description="Observed target value to evaluate model quality online.",
    )
    y_pred_old: Optional[float] = Field(
        default=None,
        description="Prediction from the baseline/old model for quality comparison.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "strategy": "NoProcessingSimple",
                    "sequence": [
                        [187.11, 184.50, 186.80, 45000000.0],
                        [188.20, 185.10, 187.95, 42000000.0],
                        [189.00, 186.00, 188.75, 39800000.0],
                    ],
                    "y_true": 189.45,
                    "y_pred_old": 190.18,
                },
                {
                    "sequence": [
                        [150.20, 147.90, 149.10, 32500000.0],
                        [151.10, 148.50, 150.40, 31200000.0],
                        [151.80, 149.00, 150.95, 30100000.0],
                    ]
                },
            ]
        }
    }

