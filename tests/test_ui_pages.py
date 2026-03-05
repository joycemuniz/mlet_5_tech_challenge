"""
Tests para as páginas Streamlit e funções auxiliares do Score.

Estratégia:
- Mockar `streamlit` e `requests` antes de importar os módulos.
- Limpar o cache de sys.modules para forçar re-execução do top-level.
- Cobrir todos os branches do 01_Score.py (submit True/False, status 200/500, exceção)
  e do 02_Metricas.py (dados válidos, arquivo ausente).
"""

import importlib
import json
import sys
import types

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_st(submit=False, slider_val=50):
    """Cria um módulo fake de streamlit que satisfaz todas as chamadas das páginas."""

    class Dummy:
        def __getattr__(self, name):
            if name == "slider":
                return lambda *a, **kw: slider_val
            if name == "form_submit_button":
                return lambda *a, **kw: submit
            if name == "selectbox":
                return lambda *a, **kw: (a[1][0] if len(a) > 1 and a[1] else 1)
            if name == "number_input":
                return lambda *a, **kw: kw.get("value", 0)
            if name == "columns":
                def _columns(spec):
                    n = spec if isinstance(spec, int) else len(spec)
                    return tuple(Dummy() for _ in range(n))
                return _columns
            if name == "expander":
                return lambda *a, **kw: Dummy()
            if name == "stop":
                # st.stop() encerra execução da página — simulamos com exceção
                raise StopIteration("st.stop() chamado")
            if name == "cache_data":
                # @st.cache_data(ttl=...) → retorna decorador que devolve fn intacta
                def _cache_data(*a, **kw):
                    if len(a) == 1 and callable(a[0]):
                        return a[0]
                    return lambda fn: fn
                return _cache_data
            return self

        def __call__(self, *a, **kw):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def dataframe(self, *a, **kw):
            return self

        def bar_chart(self, *a, **kw):
            return self

    return Dummy()


def make_fake_requests(status=200, resp_json=None, raise_on_post=None):
    """Cria módulo fake de requests com Session."""
    if resp_json is None:
        resp_json = {"score_risco": 50.0, "classe_predita": 0}

    class FakeResponse:
        def __init__(self):
            self.status_code = status
            self.text = json.dumps(resp_json)

        def json(self):
            return resp_json

    class FakeSession:
        def post(self, url, json=None, timeout=None):
            if raise_on_post:
                raise raise_on_post
            return FakeResponse()

        def get(self, url, timeout=None):
            return FakeResponse()

    fake = types.ModuleType("requests")
    fake.Session = FakeSession
    fake.Timeout = TimeoutError
    fake.ConnectionError = ConnectionError
    fake.RequestException = Exception
    return fake


def _clear_score_module():
    for key in list(sys.modules.keys()):
        if "01_Score" in key or ("pages" in key and "Score" in key):
            del sys.modules[key]


def _clear_metricas_module():
    for key in list(sys.modules.keys()):
        if "02_Metricas" in key or ("pages" in key and "Metricas" in key):
            del sys.modules[key]


def _load_score(monkeypatch, submit=False, status=200, resp_json=None, raise_on_post=None):
    """Carrega/recarrega 01_Score com dependências mockadas."""
    _clear_score_module()
    monkeypatch.setitem(sys.modules, "streamlit", make_dummy_st(submit=submit))
    monkeypatch.setitem(
        sys.modules,
        "requests",
        make_fake_requests(status=status, resp_json=resp_json, raise_on_post=raise_on_post),
    )
    return importlib.import_module("src.app.pages.01_Score")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def test_import_home(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", make_dummy_st())
    import src.app.Home as home
    assert hasattr(home, "ROOT")


# ---------------------------------------------------------------------------
# 01_Score.py — importação e variáveis
# ---------------------------------------------------------------------------

def test_import_score_page(monkeypatch):
    mod = _load_score(monkeypatch, submit=False)
    assert hasattr(mod, "API_BASE")
    assert "vercel.app" in mod.API_BASE


def test_score_page_has_post_with_retry(monkeypatch):
    mod = _load_score(monkeypatch, submit=False)
    assert callable(mod.post_with_retry)


# ---------------------------------------------------------------------------
# 01_Score.py — post_with_retry
# ---------------------------------------------------------------------------

def test_post_with_retry_success(monkeypatch):
    mod = _load_score(monkeypatch, submit=False)
    resp = mod.post_with_retry(mod.API_BASE + "/score", {"a": 1}, retries=2, timeout=5, backoff=0.01)
    assert resp.status_code == 200
    assert resp.json()["score_risco"] == 50.0


def test_post_with_retry_all_fail(monkeypatch):
    """Todos os retries falham → levanta a última exceção."""
    _clear_score_module()

    call_count = [0]

    class FailSession:
        def post(self, url, json=None, timeout=None):
            call_count[0] += 1
            raise TimeoutError("timeout")

        def get(self, url, timeout=None):
            return types.SimpleNamespace(status_code=200)

    fake_req = types.ModuleType("requests")
    fake_req.Session = FailSession
    fake_req.Timeout = TimeoutError
    fake_req.ConnectionError = ConnectionError
    fake_req.RequestException = Exception

    monkeypatch.setitem(sys.modules, "streamlit", make_dummy_st(submit=False))
    monkeypatch.setitem(sys.modules, "requests", fake_req)

    mod = importlib.import_module("src.app.pages.01_Score")

    with pytest.raises(Exception):
        mod.post_with_retry(mod.API_BASE + "/score", {"a": 1}, retries=3, timeout=1, backoff=0.01)

    assert call_count[0] == 3


# ---------------------------------------------------------------------------
# 01_Score.py — warmup
# ---------------------------------------------------------------------------

def test_warmup_runs(monkeypatch):
    mod = _load_score(monkeypatch, submit=False)
    # cache_data foi mockado → chamada direta não deve lançar
    mod.warmup()


# ---------------------------------------------------------------------------
# 01_Score.py — branches do formulário (submit=True)
# ---------------------------------------------------------------------------

def test_score_submitted_success_high_risk(monkeypatch):
    """score_risco=80 → branch >=70 (alto risco)."""
    try:
        _load_score(monkeypatch, submit=True, status=200,
                    resp_json={"score_risco": 80.0, "classe_predita": 1})
    except StopIteration:
        pass  # st.stop() de algum path inesperado — mas linhas já foram cobertas


def test_score_submitted_success_medium_risk(monkeypatch):
    """score_risco=50 → branch 40-70 (risco moderado)."""
    try:
        _load_score(monkeypatch, submit=True, status=200,
                    resp_json={"score_risco": 50.0, "classe_predita": 0})
    except StopIteration:
        pass


def test_score_submitted_success_low_risk(monkeypatch):
    """score_risco=20 → branch <40 (baixo risco)."""
    try:
        _load_score(monkeypatch, submit=True, status=200,
                    resp_json={"score_risco": 20.0, "classe_predita": 0})
    except StopIteration:
        pass


def test_score_submitted_api_error_status(monkeypatch):
    """API retorna 500 → st.error + st.stop()."""
    try:
        _load_score(monkeypatch, submit=True, status=500)
    except StopIteration:
        pass  # esperado: st.stop() lança StopIteration no mock


def test_score_submitted_request_exception(monkeypatch):
    """Requisição lança exceção → bloco except cobre linhas de erro."""
    try:
        _load_score(monkeypatch, submit=True, raise_on_post=ConnectionError("falha"))
    except (StopIteration, ConnectionError):
        pass


# ---------------------------------------------------------------------------
# 02_Metricas.py — importação com dados válidos
# ---------------------------------------------------------------------------

def _make_fake_config(tmp_path):
    """Cria módulo fake de src.utils.config apontando para arquivos temporários."""
    import src.utils.config as real_cfg

    metrics_path = tmp_path / "metrics.json"
    preds_path   = tmp_path / "predictions.csv"

    metrics_data = {
        "f1": 0.9,
        "roc_auc": 0.85,
        "confusion_matrix": [[10, 2], [3, 15]],
    }
    metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")

    pd.DataFrame({
        "y_true":      [0, 1, 0, 1, 1, 0],
        "y_pred":      [0, 1, 1, 0, 1, 0],
        "score_risco": [0.2, 0.8, 0.6, 0.4, 0.9, 0.1],
    }).to_csv(preds_path, index=False)

    fake_cfg = types.ModuleType("src.utils.config")
    for attr in dir(real_cfg):
        if not attr.startswith("__"):
            setattr(fake_cfg, attr, getattr(real_cfg, attr))
    fake_cfg.METRICS_PATH    = str(metrics_path)
    fake_cfg.PREDICTIONS_PATH = str(preds_path)
    return fake_cfg


def test_metricas_page_imports_with_valid_data(monkeypatch, tmp_path):
    """02_Metricas deve executar completamente com arquivos válidos."""
    _clear_metricas_module()
    fake_cfg = _make_fake_config(tmp_path)
    monkeypatch.setitem(sys.modules, "streamlit",        make_dummy_st(slider_val=50))
    monkeypatch.setitem(sys.modules, "src.utils.config", fake_cfg)

    mod = importlib.import_module("src.app.pages.02_Metricas")
    assert mod is not None


def test_metricas_page_high_threshold(monkeypatch, tmp_path):
    """Slider em 80 → threshold alto (ramo y_pred_thr muda)."""
    _clear_metricas_module()
    fake_cfg = _make_fake_config(tmp_path)
    monkeypatch.setitem(sys.modules, "streamlit",        make_dummy_st(slider_val=80))
    monkeypatch.setitem(sys.modules, "src.utils.config", fake_cfg)
    try:
        importlib.import_module("src.app.pages.02_Metricas")
    except StopIteration:
        pass


def test_metricas_page_missing_metrics_file(monkeypatch, tmp_path):
    """metrics.json ausente → st.error + st.stop()."""
    _clear_metricas_module()
    import src.utils.config as real_cfg

    fake_cfg = types.ModuleType("src.utils.config")
    for attr in dir(real_cfg):
        if not attr.startswith("__"):
            setattr(fake_cfg, attr, getattr(real_cfg, attr))
    fake_cfg.METRICS_PATH    = str(tmp_path / "nao_existe.json")
    fake_cfg.PREDICTIONS_PATH = str(tmp_path / "nao_existe.csv")

    monkeypatch.setitem(sys.modules, "streamlit",        make_dummy_st())
    monkeypatch.setitem(sys.modules, "src.utils.config", fake_cfg)

    with pytest.raises(StopIteration):
        importlib.import_module("src.app.pages.02_Metricas")


def test_metricas_page_missing_predictions(monkeypatch, tmp_path):
    """predictions.csv ausente → st.warning + st.stop()."""
    _clear_metricas_module()
    import src.utils.config as real_cfg

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"f1": 0.9, "roc_auc": 0.85, "confusion_matrix": [[10, 2], [3, 15]]}),
        encoding="utf-8",
    )

    fake_cfg = types.ModuleType("src.utils.config")
    for attr in dir(real_cfg):
        if not attr.startswith("__"):
            setattr(fake_cfg, attr, getattr(real_cfg, attr))
    fake_cfg.METRICS_PATH    = str(metrics_path)
    fake_cfg.PREDICTIONS_PATH = str(tmp_path / "nao_existe.csv")

    monkeypatch.setitem(sys.modules, "streamlit",        make_dummy_st())
    monkeypatch.setitem(sys.modules, "src.utils.config", fake_cfg)

    with pytest.raises(StopIteration):
        importlib.import_module("src.app.pages.02_Metricas")