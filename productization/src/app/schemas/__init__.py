"""
A package that holds response schemas and models.
"""

__all__ = [
    "RESPONSES",
    "ErrorMessage",
    "SuccessMessage",
    "InferRequest",
]

from .responses import RESPONSES, ErrorMessage, SuccessMessage
from .endpoints import InferRequest
