import os
import joblib
import json
import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app

client = TestClient(app)


def test_load_artifacts_missing(tmp_path, monkeypatch):
    # ensure models folder is empty
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    # calling load_artifacts should raise because model.pkl doesn't exist
    with pytest.raises(FileNotFoundError):
        app_module.load_artifacts()


def test_load_artifacts_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    # also adjust the configured MODELS_DIR so features.json lookup uses our temp
    monkeypatch.setattr(app_module, 'MODELS_DIR', str(models_dir))
    # dump a dummy model
    dummy = {'x': 1}
    joblib.dump(dummy, models_dir / 'model.pkl')
    # features file optional
    features = ['a', 'b']
    with open(models_dir / 'features.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(features))
    # should load without error and set global vars
    app_module._model = None
    app_module._features = None
    app_module.load_artifacts()
    assert app_module._model == dummy
    assert app_module._features == features


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_score_no_model():
    app_module._model = None
    r = client.post("/score", json={"a": 1})
    assert r.status_code == 500
    assert "Modelo não carregado" in r.text


def test_score_incompatible_model():
    class Bad:
        pass

    app_module._model = Bad()
    r = client.post("/score", json={"a": 1})
    assert r.status_code == 500
    assert "Modelo incompatível" in r.text


def test_score_ok():
    class Dummy:
        def predict(self, X):
            return [0]

        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.7, 0.3]])

    app_module._model = Dummy()
    app_module._features = None
    r = client.post("/score", json={"feature1": 1})
    assert r.status_code == 200
    data = r.json()
    assert "classe_predita" in data
    assert "score_risco" in data
