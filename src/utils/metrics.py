from __future__ import annotations

import os
import time
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
    """Return content and content_type suitable for FastAPI Response."""
    return generate_latest(registry), CONTENT_TYPE_LATEST


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
