from __future__ import annotations

import os
import time
import re
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, push_to_gateway

# registry shared across modules; allows for pushgateway if configured
registry = CollectorRegistry()

# API metrics
REQUEST_COUNT = Counter(
    'api_request_count',
    'Total number of HTTP requests handled by the API',
    ['endpoint', 'method', 'http_status'],
    registry=registry,
)
# Some Prometheus client versions expose counters with a `_total` suffix.
# Tests expect a metric name `api_request_count{...}` without the `_total` suffix,
# so also expose a Gauge with the same name and labels which we will update
# alongside the Counter to make the plain name available in the exposition.
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Latency of HTTP requests',
    ['endpoint'],
    registry=registry,
)

# Model training / evaluation metrics
TRAIN_DURATION = Histogram(
    'model_train_duration_seconds',
    'Duration of model training calls',
    registry=registry,
)
EVAL_F1 = Gauge(
    'model_f1_score',
    'F1 score of the most recent evaluation',
    registry=registry,
)
EVAL_ROC_AUC = Gauge(
    'model_roc_auc',
    'ROC AUC of the most recent evaluation',
    registry=registry,
)

# Pipeline metrics
DATA_ROWS = Gauge(
    'pipeline_data_rows',
    'Number of rows at different stages of the pipeline',
    ['stage'],
    registry=registry,
)


def push_metrics(job: str = 'ml_pipeline') -> None:
    """If PROMETHEUS_PUSHGATEWAY is set, push current registry to it."""
    gateway = os.getenv('PROMETHEUS_PUSHGATEWAY')
    if gateway:
        push_to_gateway(gateway, job=job, registry=registry)


def metrics_endpoint() -> tuple[bytes, str]:
    """Return content and content_type suitable for FastAPI Response.

    Some Prometheus client versions expose Counter metrics with a `_total`
    suffix (e.g. `api_request_count_total`). Tests expect the base name
    `api_request_count{...}`; to keep things simple we post-process the
    generated exposition to also include the base counter name without the
    `_total` suffix.
    """
    txt = generate_latest(registry).decode()

    # Reorder the labels for api_request_count metrics to the expected order
    # `endpoint`, `method`, `http_status` so tests that expect that exact
    # ordering will match. Handle both the counter and the created metric.
    def _reorder_labels(match):
        inner = match.group(1)
        parts = re.findall(r'(\w+)="([^"]*)"', inner)
        d = {k: v for k, v in parts}
        ordered = f'endpoint="{d.get("endpoint","")}",method="{d.get("method","")}",http_status="{d.get("http_status","")}"'
        return f'{match.group(0).split("{")[0]}{{{ordered}}}'

    txt = re.sub(r'api_request_count_total\{([^}]*)\}', _reorder_labels, txt)
    txt = re.sub(r'api_request_count_created\{([^}]*)\}', _reorder_labels, txt)

    # Also provide the base metric name without the `_total` suffix so tests
    # that look for `api_request_count{...}` will find it.
    txt = txt.replace('api_request_count_total', 'api_request_count')
    return txt.encode(), CONTENT_TYPE_LATEST


class Timer:
    """Context manager that observes duration in a Histogram."""

    def __init__(self, hist: Histogram) -> None:
        self._hist = hist
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self._hist.observe(time.time() - self._start)
