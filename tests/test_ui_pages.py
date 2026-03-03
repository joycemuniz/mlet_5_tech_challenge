import sys
import types
import pandas as pd
import json
from pathlib import Path

def make_dummy_st():
    class Dummy:
        def __getattr__(self, name):
            # provide sensible defaults for some methods
            if name == 'slider':
                return lambda *args, **kwargs: 50
            if name == 'form_submit_button':
                return lambda *args, **kwargs: True
            if name == 'selectbox':
                return lambda *args, **kwargs: args[1][0] if len(args) > 1 and args[1] else None
            if name == 'columns':
                def _columns(spec):
                    if isinstance(spec, int):
                        n = spec
                    elif isinstance(spec, (list, tuple)):
                        n = len(spec)
                    else:
                        n = 1
                    return tuple(Dummy() for _ in range(n))
                return _columns
            if name == 'expander':
                return lambda *args, **kwargs: Dummy()
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def dataframe(self, *args, **kwargs):
            return self

        def bar_chart(self, *args, **kwargs):
            return self

        def pyplot(self, *args, **kwargs):
            return self

    return Dummy()


def test_import_home(monkeypatch):
    monkeypatch.setitem(sys.modules, 'streamlit', make_dummy_st())
    import src.app.Home as home
    assert hasattr(home, 'ROOT')  # mantenho como você está usando hoje


def test_import_score_page(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'streamlit', make_dummy_st())
    fake_req = types.SimpleNamespace()

    def post(url, json=None, timeout=None):
        return types.SimpleNamespace(status_code=200, json=lambda: {'score_risco': 50, 'classe_predita': 0})

    fake_req.post = post
    monkeypatch.setitem(sys.modules, 'requests', fake_req)

    import importlib
    score_module = importlib.import_module('src.app.pages.01_Score')
    assert hasattr(score_module, 'API_BASE')


def test_import_metrics_page(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'streamlit', make_dummy_st())

    metrics_path = tmp_path / 'metrics.json'
    preds_path = tmp_path / 'predictions.csv'
    metrics_path.write_text(json.dumps({'f1': 1, 'roc_auc': 0.5, 'confusion_matrix': [[1, 0], [0, 1]]}), encoding='utf-8')
    pd.DataFrame({'y_true': [0], 'y_pred': [1], 'score_risco': [0.3]}).to_csv(preds_path, index=False)

    import src.utils.config as config
    monkeypatch.setattr(config, 'METRICS_PATH', str(metrics_path))
    monkeypatch.setattr(config, 'PREDICTIONS_PATH', str(preds_path))

    import importlib
    metrics_module = importlib.import_module('src.app.pages.02_Metricas')
    assert hasattr(metrics_module, 'METRICS_PATH')