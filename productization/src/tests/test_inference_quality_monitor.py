from pathlib import Path

import torch
from fastapi.testclient import TestClient

from app.inference.quality import evaluate_quality
from app.main import app
import app.main as app_main


class DummyModel(torch.nn.Module):
    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return model_input[:, -1, 0:1]


def test_evaluate_quality_replicates_mean_and_ks_criteria() -> None:
    y_true = [0.0] * 40
    y_pred_old = [1.0] * 40
    y_pred_new = [0.2] * 40

    result = evaluate_quality(y_true=y_true, y_pred_new=y_pred_new, y_pred_old=y_pred_old)

    assert result["sample_size"] == 40
    assert bool(result["mean_error_improved"]) is True
    assert bool(result["ks_improved"]) is True
    assert bool(result["quality_gate_passed"]) is True


def test_infer_endpoint_returns_quality_monitoring_when_metrics_are_sent(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "_resolve_model_artifact_path", lambda strategy: Path("dummy.pt"))
    monkeypatch.setattr(
        app_main,
        "_load_inference_model",
        lambda model_path: (
            DummyModel(),
            {"input_size": 4},
            {"seq_len": 3},
        ),
    )

    with app_main._QUALITY_MONITOR_LOCK:
        app_main._QUALITY_MONITOR_STATE["y_true"].clear()
        app_main._QUALITY_MONITOR_STATE["y_pred_new"].clear()
        app_main._QUALITY_MONITOR_STATE["y_pred_old"].clear()

    client = TestClient(app)
    payload = {
        "sequence": [
            [10.0, 9.0, 9.5, 1000.0],
            [10.5, 9.5, 10.0, 1200.0],
            [11.0, 10.0, 10.5, 1400.0],
        ],
        "y_true": 10.8,
        "y_pred_old": 11.4,
    }

    response = client.post("/infer", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "quality_monitoring" in body
    assert body["quality_monitoring"]["sample_size"] == 1
    assert body["quality_monitoring"]["ks_pvalue"] is None


def test_infer_endpoint_rejects_partial_quality_payload(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "_resolve_model_artifact_path", lambda strategy: Path("dummy.pt"))
    monkeypatch.setattr(
        app_main,
        "_load_inference_model",
        lambda model_path: (
            DummyModel(),
            {"input_size": 4},
            {"seq_len": 3},
        ),
    )

    client = TestClient(app)
    payload = {
        "sequence": [
            [10.0, 9.0, 9.5, 1000.0],
            [10.5, 9.5, 10.0, 1200.0],
            [11.0, 10.0, 10.5, 1400.0],
        ],
        "y_true": 10.8,
    }

    response = client.post("/infer", json=payload)

    assert response.status_code == 400
    assert "must be provided together" in response.json()["detail"]
