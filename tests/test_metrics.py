import re

import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app
from src.utils import metrics
from src.modeling.train import train_model
from src.modeling.evaluate import evaluate_model

client = TestClient(app)


def test_metrics_endpoint_and_score():
    # ensure /metrics exists and contains expected metric names
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "api_request_count" in text
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json().get("status") == "ok"
    from src.api.app import _model
    class Dummy:
        def predict(self, X):
            return [0]
    # monkeypatch the model for a normal request
    app.dependency_overrides = {}
    import src.api.app as _app_module
    _app_module._model = Dummy()
    payload = {"feature1": 1}
    r2 = client.post("/score", json=payload)
    assert r2.status_code == 200
    # after a request the metrics endpoint should show a count >=1
    r3 = client.get("/metrics")
    assert "api_request_count" in r3.text
    assert re.search(r"api_request_count\{endpoint=\"/score\",method=\"POST\",http_status=\"200\"\} \d+", r3.text)


def test_score_no_model():
    # make sure when no model is loaded we return 500 and metrics updated
    import src.api.app as _app_module
    _app_module._model = None
    r = client.post("/score", json={"a": 1})
    assert r.status_code == 500
    assert "Modelo não carregado" in r.text


def test_score_incompatible_model():
    # model object has neither predict nor predict_proba
    class Bad:
        pass
    import src.api.app as _app_module
    _app_module._model = Bad()
    r = client.post("/score", json={"a": 1})
    assert r.status_code == 500
    assert "Modelo incompatível" in r.text


def test_train_evaluate_metrics():
    # create tiny dataset
    import numpy as np
    X = pd.DataFrame(np.random.rand(4, 2), columns=["a", "b"])
    y = pd.Series([0, 1, 0, 1])
    model = train_model(X, y)
    # get metrics text after training
    mtxt = metrics.generate_latest(metrics.registry).decode()
    assert "model_train_duration_seconds" in mtxt

    # evaluate and check gauge values appear
    res = evaluate_model(model, X, y)
    mtxt2 = metrics.generate_latest(metrics.registry).decode()
    assert "model_f1_score" in mtxt2
    assert "model_roc_auc" in mtxt2

    # manually exercise pipeline-data gauge
    metrics.DATA_ROWS.labels(stage="dummy").set(123)
    mtxt3 = metrics.generate_latest(metrics.registry).decode()
    assert "pipeline_data_rows" in mtxt3
    # stage label should include equals sign
    assert 'stage="dummy"' in mtxt3

