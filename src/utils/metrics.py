from __future__ import annotations

import time


class _CounterChild:
    """Single labelled counter instance."""

    def __init__(self, store: dict, key: tuple) -> None:
        self._store = store
        self._key = key

    def inc(self, amount: float = 1) -> None:
        self._store[self._key] = self._store.get(self._key, 0) + amount


class _Counter:
    """In-memory request counter (no external dependencies)."""

    def __init__(self) -> None:
        self._values: dict = {}

    def labels(self, **kwargs) -> _CounterChild:
        key = tuple(sorted(kwargs.items()))
        return _CounterChild(self._values, key)


class _NullContextManager:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _HistogramLabelled:
    def time(self) -> _NullContextManager:
        return _NullContextManager()

    def observe(self, value: float) -> None:
        pass


class _Histogram:
    """In-memory histogram stub."""

    def labels(self, **kwargs) -> _HistogramLabelled:
        return _HistogramLabelled()

    def time(self) -> _NullContextManager:
        return _NullContextManager()

    def observe(self, value: float) -> None:
        pass


class _GaugeChild:
    def __init__(self, store: dict, key: tuple) -> None:
        self._store = store
        self._key = key

    def set(self, value: float) -> None:
        self._store[self._key] = value


class _Gauge:
    """In-memory gauge stub."""

    def __init__(self) -> None:
        self._value: float = 0.0
        self._values: dict = {}

    def labels(self, **kwargs) -> _GaugeChild:
        key = tuple(sorted(kwargs.items()))
        return _GaugeChild(self._values, key)

    def set(self, value: float) -> None:
        self._value = value


# ---------------------------------------------------------------------------
# Public metric objects (same names as before so existing imports still work)
# ---------------------------------------------------------------------------

REQUEST_COUNT = _Counter()
REQUEST_LATENCY = _Histogram()
TRAIN_DURATION = _Histogram()
EVAL_F1 = _Gauge()
EVAL_ROC_AUC = _Gauge()
DATA_ROWS = _Gauge()


def push_metrics(job: str = 'ml_pipeline') -> None:
    """No-op: Prometheus/Grafana integration removed."""
    pass


class Timer:
    """Context manager that measures elapsed time."""

    def __init__(self, hist) -> None:
        self._hist = hist
        self._start: float | None = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self._hist.observe(time.time() - self._start)
