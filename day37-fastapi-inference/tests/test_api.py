from fastapi.testclient import TestClient

from main import app


def test_health(monkeypatch):
    monkeypatch.delenv(
        "SIMULATE_MODEL_UNAVAILABLE",
        raising=False,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_predict(monkeypatch):
    monkeypatch.delenv(
        "SIMULATE_MODEL_UNAVAILABLE",
        raising=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == 2.4


def test_invalid_features(monkeypatch):
    monkeypatch.delenv(
        "SIMULATE_MODEL_UNAVAILABLE",
        raising=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0]},
        )

    assert response.status_code == 422


def test_model_unavailable(monkeypatch):
    monkeypatch.setenv(
        "SIMULATE_MODEL_UNAVAILABLE",
        "1",
    )

    with TestClient(app) as client:
        ready = client.get("/ready")
        predict = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert ready.status_code == 503
    assert predict.status_code == 503
