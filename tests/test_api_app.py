import os
import joblib
import json
import pytest
from pathlib import Path

from src.api import app as app_module


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
