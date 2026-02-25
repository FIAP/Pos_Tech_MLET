"""
main.py

Entry point for the LSTM Service FastAPI application.

This module boots the API, registers middleware, error handlers</s>
and all endpoint routes for training, inference, and monitoring of
LSTM-based stock price prediction models.

Key responsibilities:
    - Health / readiness / startup probes (``GET /``, ``/ready``, ``/startup``).
    - Asynchronous model training via ``POST /train`` (runs in a
      ``ProcessPoolExecutor``).
    - Real-time inference via ``POST /infer`` with optional quality
      monitoring side-car.
    - Model quality evaluation via ``POST /evaluate_quality``.
"""

import os
from typing import Any
from pathlib import Path
from threading import Lock
from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __app__, __author__, __version__, logger
from app.schemas import RESPONSES, ErrorMessage, InferRequest
from app.model.lstm import LSTMFactory
from app.model.lstm_params import LSTMParams
from app.inference import evaluate_quality, MIN_SAMPLE_SIZE

from app import train


tags_metadata: list[dict] = [
    {
        "name": "Treinamento",
        "description": """
    Endpoints para o treinamento e a otimização de um novo modelo.
        """,
    },
    {
        "name": "Inferencia",
        "description": """
    Endpoints de inferência do modelo.
        """,
    },
    {
        "name": "Atualização",
        "description": """
    Endpoints para a atualização do modelo, como fine tunning, prunning e quantization.
        """,
    },
    {
        "name": "Monitoramento",
        "description": """
    Endpoints de monitoramento do modelos.
        """,
    },
    {
        "name": "Configuração",
        "description": """
    Endpoints para configuração do Serviço.
        """,
    },
]

description: str = """

"""


app: FastAPI = FastAPI(
    title=__app__,
    version=__version__,
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
    swagger_ui_oauth2_redirect_url='/oauth2-redirect',
    swagger_ui_init_oauth={
        'usePkceWithAuthorizationCodeGrant': True,
        'clientId': f'{os.getenv("MSFT_CLIENT_ID", "")}',
    },
    docs_url=None,
    redoc_url=None
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


TRAINING_EXECUTOR: ProcessPoolExecutor = ProcessPoolExecutor()
_ACTIVE_TRAINING_JOBS: set[Future] = set()
_MODEL_CACHE: dict[str, Any] = {}
_LAST_TRAINING_CONFIG_BY_STRATEGY: dict[str, dict[str, Any]] = {}
_QUALITY_MONITOR_LOCK = Lock()
_QUALITY_MONITOR_STATE: dict[str, deque[float]] = {
    "y_true": deque(maxlen=5000),
    "y_pred_new": deque(maxlen=5000),
    "y_pred_old": deque(maxlen=5000),
}


def _get_model_artifact_dir() -> Path:
    """Return the directory where trained model artifacts (``.pt`` files) are stored.

    Returns:
        Path: Absolute path to the ``.models`` directory inside the ``train`` package.
    """
    return Path(train.__file__).resolve().parent / ".models"


def _resolve_model_artifact_path(strategy_name: str | None = None) -> Path:
    """Resolve the path to a trained model artifact.

    When *strategy_name* is provided the corresponding ``.pt`` file is
    returned.  Otherwise the most recently modified artifact is selected.

    Args:
        strategy_name (str | None): Optional strategy identifier.

    Returns:
        Path: Path to the ``.pt`` model file.

    Raises:
        HTTPException: 404 if no artifact is found; 409 if training is
            still in progress.
    """
    model_dir = _get_model_artifact_dir()

    if strategy_name:
        model_path = model_dir / f"{strategy_name}.pt"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Model artifact for strategy '{strategy_name}' not found. "
                    "Train the model first using /train."
                ),
            )
        return model_path

    model_paths = list(model_dir.glob("*.pt")) if model_dir.exists() else []
    if not model_paths:
        if _ACTIVE_TRAINING_JOBS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No trained model is available yet. A training job is still running.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model found. Execute /train before calling /infer.",
        )

    return max(model_paths, key=lambda candidate: candidate.stat().st_mtime)


def _load_inference_model(
    model_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load (or retrieve from cache) a trained LSTM model for inference.

    Uses a module-level cache keyed on file path and modification time to
    avoid redundant ``torch.load`` calls.

    Args:
        model_path (Path): Path to the ``.pt`` artifact.

    Returns:
        tuple[torch.nn.Module, dict, dict]: A triple of
            (model in ``eval`` mode, LSTM parameter dict, training parameter dict).

    Raises:
        HTTPException: 500 if the artifact format is unsupported or metadata
            is missing.
    """
    model_mtime = model_path.stat().st_mtime
    cached_path = _MODEL_CACHE.get("path")
    cached_mtime = _MODEL_CACHE.get("mtime")
    cached_model = _MODEL_CACHE.get("model")
    cached_lstm_params = _MODEL_CACHE.get("lstm_params")
    cached_training_params = _MODEL_CACHE.get("training_params")

    if (
        cached_model is not None
        and cached_path == str(model_path)
        and cached_mtime == model_mtime
        and isinstance(cached_lstm_params, dict)
        and isinstance(cached_training_params, dict)
    ):
        return cached_model, cached_lstm_params, cached_training_params

    raw_artifact = torch.load(model_path, map_location="cpu")
    strategy_name = model_path.stem

    state_dict: dict[str, Any]
    layer_config: dict[str, Any]
    lstm_params_raw: dict[str, Any]
    training_params: dict[str, Any]

    if isinstance(raw_artifact, dict) and "state_dict" in raw_artifact:
        state_dict = raw_artifact["state_dict"]
        layer_config = raw_artifact.get("layer_config", {})
        lstm_params_raw = raw_artifact.get("lstm_params", {})
        training_params = raw_artifact.get("training_params", {})
    elif isinstance(raw_artifact, dict):
        state_dict = raw_artifact
        fallback_config = _LAST_TRAINING_CONFIG_BY_STRATEGY.get(strategy_name, {})
        layer_config = fallback_config.get("layer_config", {})
        lstm_params_raw = fallback_config.get("lstm_params", {})
        training_params = fallback_config.get("training_params", {})
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported model artifact format at '{model_path}'.",
        )

    if not layer_config or not lstm_params_raw:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Model metadata not found in artifact. Retrain the model with the current "
                "version and try again."
            ),
        )

    lstm_params = LSTMParams.model_validate(lstm_params_raw)
    factory = LSTMFactory(layer_config, lstm_params)
    model = factory.create()
    model.load_state_dict(state_dict)
    model.eval()

    lstm_params_dict = lstm_params.model_dump()
    _MODEL_CACHE.update(
        {
            "path": str(model_path),
            "mtime": model_mtime,
            "model": model,
            "lstm_params": lstm_params_dict,
            "training_params": training_params,
        }
    )

    return model, lstm_params_dict, training_params


def _cleanup_future(future: Future) -> None:
    """Callback to remove a training ``Future`` from the active set when it completes.

    Args:
        future (Future): The finished future to discard.
    """
    _ACTIVE_TRAINING_JOBS.discard(future)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        detail={"invalid-params": list(exc.errors())},
    )

    logger.error(
        f"Validation error: {exc.errors()}",
        extra={
            "request": {
                "method": request.method,
                "url": request.url,
                "headers": request.headers,
                "body": await request.json(),
            }
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.exception_handler(ResponseValidationError)
async def response_exception_handler(
    request: Request, exc: ResponseValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    response_exception_handler Exception handler for response validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Response Error",
        title="Found Errors on processing your requests.",
        detail={"invalid-params": list(exc.errors())},
    )

    logger.error(
        f"Response validation error: {exc.errors()}",
        extra={
            "request": {
                "method": request.method,
                "url": request.url,
                "headers": request.headers,
                "body": await request.json(),
            }
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.get("/", tags=["Configuração"], summary="Health Check Endpoint")
async def health_check() -> dict[str, str]:
    """Return a simple ``{"status": "healthy"}`` payload.

    Used by container orchestrators (Docker / Kubernetes) as a
    **liveness probe** to verify the service is running.

    Returns:
        dict[str, str]: ``{"status": "healthy"}``.
    """
    return {"status": "healthy"}


@app.get("/ready", tags=["Configuração"], summary="Readiness Check Endpoint")
async def readiness_check() -> dict[str, str]:
    """Return a simple ``{"status": "ready"}`` payload.

    Used as a **readiness probe** to confirm the service can accept requests.

    Returns:
        dict[str, str]: ``{"status": "ready"}``.
    """
    return {"status": "ready"}


@app.get("/startup", tags=["Configuração"], summary="Startup Check Endpoint")
async def startup_check() -> dict[str, str]:
    """Return a simple ``{"status": "started"}`` payload.

    Used as a **startup probe** to confirm the application has initialised.

    Returns:
        dict[str, str]: ``{"status": "started"}``.
    """
    return {"status": "started"}


@app.post("/train", tags=["Treinamento"], summary="Train a new LSTM model")
async def train_model(strategy: str, params: train.TrainingParams) -> dict[str, str]:
    """Schedule an asynchronous training job for a new LSTM model.

    The training is dispatched to a ``ProcessPoolExecutor`` so the API
    remains responsive. Once the job finishes, the model artifact is
    persisted in ``train/.models/<strategy>.pt``.

    Args:
        strategy (str): Name of the ``TrainingStrategy`` class to use
            (e.g. ``"RangeMultipleStrategy"``).
        params (TrainingParams): Hyperparameters and configuration for
            the training run (tickers, period, seq_len, etc.).

    Returns:
        dict[str, str]: Confirmation message with paths to the MLflow
            tracking directory and the expected model artifact.

    Raises:
        HTTPException: 400 if the strategy name is unknown.
    """

    if not hasattr(train, strategy):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown training strategy '{strategy}'."
        )

    strategy_instance = getattr(train, strategy)(params)
    _LAST_TRAINING_CONFIG_BY_STRATEGY[strategy_instance.name] = {
        "layer_config": strategy_instance.layer_config,
        "lstm_params": strategy_instance.lstm_params,
        "training_params": strategy_instance.get_training_params(),
    }
    context = train.TrainerContext(strategy_instance)

    future: Future = TRAINING_EXECUTOR.submit(context.train)
    _ACTIVE_TRAINING_JOBS.add(future)
    future.add_done_callback(_cleanup_future)

    train_module_dir = Path(train.__file__).resolve().parent
    mlflow_directory = train_module_dir / "mlruns"
    expected_model_path = train_module_dir / ".models" / f"{strategy_instance.name}.pt"

    return {
        "message": "Training started. The model will be available after the run in the MLflow directory.",
        "mlflow_directory": str(mlflow_directory),
        "expected_model_path": str(expected_model_path)
    }


@app.post("/infer", tags=["Inferencia"], summary="Make a prediction using the LSTM model")
async def infer_model(data: InferRequest) -> dict[str, Any]:
    """Perform a real-time prediction with an optional quality monitoring side-car.

    The endpoint loads the trained LSTM artifact (cached for performance),
    validates the input sequence shape, and returns the prediction.

    When *y_true* **and** *y_pred_old* are also supplied, the response
    includes running quality metrics (mean error comparison +
    Kolmogorov–Smirnov test).

    Args:
        data (InferRequest): Request body containing the input sequence
            and optional quality fields.

    Returns:
        dict[str, Any]: ``{"prediction": float}`` plus an optional
            ``"quality_monitoring"`` sub-document.

    Raises:
        HTTPException: 400 for invalid input; 404/409 if no model is
            available.
    """
    strategy_name = data.strategy

    model_path = _resolve_model_artifact_path(strategy_name)
    model, lstm_params, training_params = _load_inference_model(model_path)

    raw_sequence = data.sequence

    try:
        input_tensor = torch.tensor(raw_sequence, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'sequence' must contain only numeric values.",
        ) from exc

    if input_tensor.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'sequence' must be a 2D list with shape [seq_len, input_size].",
        )

    expected_input_size = int(lstm_params.get("input_size", 0))
    if input_tensor.shape[1] != expected_input_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid input_size: received {input_tensor.shape[1]}, "
                f"expected {expected_input_size}."
            ),
        )

    expected_seq_len = training_params.get("seq_len")
    if isinstance(expected_seq_len, int) and input_tensor.shape[0] != expected_seq_len:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid seq_len: received {input_tensor.shape[0]}, "
                f"expected {expected_seq_len}."
            ),
        )

    model_input = input_tensor.unsqueeze(0)
    with torch.inference_mode():
        output = model(model_input)

    prediction_result = float(output.reshape(-1)[0].item())

    if (data.y_true is None) != (data.y_pred_old is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Fields 'y_true' and 'y_pred_old' must be provided together "
                "to run quality monitoring."
            ),
        )

    response: dict[str, Any] = {"prediction": prediction_result}

    if data.y_true is not None and data.y_pred_old is not None:
        with _QUALITY_MONITOR_LOCK:
            _QUALITY_MONITOR_STATE["y_true"].append(float(data.y_true))
            _QUALITY_MONITOR_STATE["y_pred_old"].append(float(data.y_pred_old))
            _QUALITY_MONITOR_STATE["y_pred_new"].append(prediction_result)

            quality_metrics = evaluate_quality(
                y_true=list(_QUALITY_MONITOR_STATE["y_true"]),
                y_pred_new=list(_QUALITY_MONITOR_STATE["y_pred_new"]),
                y_pred_old=list(_QUALITY_MONITOR_STATE["y_pred_old"]),
                min_sample_size=MIN_SAMPLE_SIZE,
            )

        response["quality_monitoring"] = quality_metrics

    return response


@app.post("/evaluate_quality", tags=["Monitoramento"], summary="Evaluate the quality of the new model")
async def evaluate_quality_endpoint(data: dict[str, Any]) -> dict[str, Any]:
    """Run a full quality evaluation of accumulated new-model predictions.

    Compares the recent predictions stored in the in-memory quality
    monitor against user-supplied ground-truth and baseline predictions.

    Args:
        data (dict[str, Any]): JSON body with keys ``y_true`` (list[float])
            and ``y_pred_old`` (list[float]).

    Returns:
        dict[str, Any]: ``{"quality_monitoring": {...}}`` containing
            mean errors, KS statistics, and a boolean quality gate.

    Raises:
        HTTPException: 400 if required fields are missing or no new
            predictions exist.
    """
    y_true = data.get("y_true")
    y_pred_old = data.get("y_pred_old")

    if y_true is None or y_pred_old is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fields 'y_true' and 'y_pred_old' must be provided together for quality evaluation.",
        )

    y_pred_new = list(_QUALITY_MONITOR_STATE["y_pred_new"])

    if not y_pred_new:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No new predictions available for quality evaluation. Make inference calls to generate predictions before evaluating quality.",
        )

    evaluation_results = evaluate_quality(y_true, y_pred_new, y_pred_old)

    return {"quality_monitoring": evaluation_results}