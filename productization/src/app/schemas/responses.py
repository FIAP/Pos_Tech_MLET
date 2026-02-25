"""responses.py

Standardised HTTP response bodies and the ``RESPONSES`` mapping used
by FastAPI to generate OpenAPI documentation.

Classes:
    SuccessMessage: Base body for 2xx responses.
    ErrorMessage: Base body for 4xx/5xx error responses.

Constants:
    RESPONSES: ``dict[int, dict]`` mapping HTTP status codes to their
        Pydantic model for use in ``FastAPI(responses=...)``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_202_ACCEPTED,
    HTTP_301_MOVED_PERMANENTLY,
    HTTP_302_FOUND,
    HTTP_307_TEMPORARY_REDIRECT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_418_IM_A_TEAPOT,
    HTTP_422_UNPROCESSABLE_ENTITY,
)


@dataclass
class SuccessMessage:
    """Base body for successful (2xx) HTTP responses.

    Attributes:
        title: Short human-readable title (e.g. ``"Model trained"``).
        message: Longer descriptive message.
        content: Arbitrary payload — a dictionary or list of dictionaries.
    """

    title: Optional[str]
    message: Optional[str]
    content: Optional[Union[Dict[str, Union[str, List]], List[Dict[str, Union[str, List]]]]]


@dataclass
class ErrorMessage:
    """Base body for error (4xx / 5xx) HTTP responses.

    Attributes:
        success: Always ``False`` for error responses.
        type: Error category (e.g. ``"Validation Error"``).
        title: Short title describing the problem.
        detail: Detailed information — a dictionary or list of dictionaries
            containing invalid parameters or stack traces.
    """

    success: bool
    type: Optional[str]
    title: Optional[str]
    detail: Optional[Union[Dict[str, Union[str, List]], List[Dict[str, Union[str, List]]]]]


RESPONSES = {
    HTTP_200_OK: {"model": SuccessMessage},
    HTTP_201_CREATED: {"model": SuccessMessage},
    HTTP_202_ACCEPTED: {"model": SuccessMessage},
    HTTP_302_FOUND: {"model": SuccessMessage},
    HTTP_301_MOVED_PERMANENTLY: {"model": ErrorMessage},
    HTTP_307_TEMPORARY_REDIRECT: {"model": ErrorMessage},
    HTTP_400_BAD_REQUEST: {"model": ErrorMessage},
    HTTP_401_UNAUTHORIZED: {"model": ErrorMessage},
    HTTP_403_FORBIDDEN: {"model": ErrorMessage},
    HTTP_418_IM_A_TEAPOT: {"model": ErrorMessage},
    HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorMessage},
}
