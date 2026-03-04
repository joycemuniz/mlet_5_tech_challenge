"""Tests for src/utils/metrics – in-memory stubs (no Prometheus dependency)."""

import pandas as pd
import numpy as np

from src.utils import metrics
from src.modeling.train import train_model
from src.modeling.evaluate import evaluate_model


def test_counter_labels_inc():
    c = metrics._Counter()
    c.labels(endpoint="/test", method="GET", http_status="200").inc()
    c.labels(endpoint="/test", method="GET", http_status="200").inc(2)
    key = tuple(sorted({"endpoint": "/test", "method": "GET", "http_status": "200"}.items()))
    assert c._values[key] == 3


def test_histogram_is_noop():
    h = metrics._Histogram()
    # labels() returns something with time() and observe()
    labelled = h.labels(endpoint="/x")
    ctx = labelled.time()
    ctx.__enter__()
    ctx.__exit__(None, None, None)
    labelled.observe(0.5)
    # direct usage
    with h.time():
        pass
    h.observe(1.0)


def test_gauge_set():
    g = metrics._Gauge()
    g.set(3.14)
    assert g._value == 3.14


def test_gauge_labels_set():
    g = metrics._Gauge()
    g.labels(stage="test").set(99)
    key = tuple(sorted({"stage": "test"}.items()))
    assert g._values[key] == 99


def test_global_objects_exist():
    assert metrics.REQUEST_COUNT is not None
    assert metrics.REQUEST_LATENCY is not None
    assert metrics.TRAIN_DURATION is not None
    assert metrics.EVAL_F1 is not None
    assert metrics.EVAL_ROC_AUC is not None
    assert metrics.DATA_ROWS is not None


def test_push_metrics_noop():
    # Should not raise under any circumstance
    metrics.push_metrics()
    metrics.push_metrics(job="custom_job")


def test_timer_context_manager():
    """Timer deve registrar duração (cobre __enter__/__exit__)."""
    import time as _time

    hist = metrics._Histogram()
    timer = metrics.Timer(hist)
    with timer:
        _time.sleep(0.01)
    # No exception means it worked


def test_train_uses_no_prometheus():
    """train_model deve executar sem erros (sem prometheus_client)."""
    X = pd.DataFrame(np.random.rand(4, 2), columns=["a", "b"])
    y = pd.Series([0, 1, 0, 1])
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_evaluate_uses_no_prometheus():
    """evaluate_model deve executar sem erros (sem prometheus_client)."""
    X = pd.DataFrame(np.random.rand(4, 2), columns=["a", "b"])
    y = pd.Series([0, 1, 0, 1])
    model = train_model(X, y)
    res = evaluate_model(model, X, y)
    assert "f1" in res
    assert "roc_auc" in res


def test_request_count_global():
    """REQUEST_COUNT global deve aceitar labels e inc sem erros."""
    metrics.REQUEST_COUNT.labels(endpoint="/score", method="POST", http_status="200").inc()
    metrics.REQUEST_COUNT.labels(endpoint="/score", method="POST", http_status="500").inc()


def test_data_rows_global():
    """DATA_ROWS global deve aceitar labels e set sem erros."""
    metrics.DATA_ROWS.labels(stage="refined").set(500)
    metrics.DATA_ROWS.labels(stage="train").set(400)
    metrics.DATA_ROWS.labels(stage="test").set(100)

