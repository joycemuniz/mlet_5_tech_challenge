from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from src.modeling.train import split_by_year, TrainConfig

from src.utils.metrics import EVAL_F1, EVAL_ROC_AUC, push_metrics


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)

    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)

    report_txt = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(report_txt)
    print("Confusion Matrix:\n", cm)
    print(f"F1-score: {f1:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.3f}")

    EVAL_F1.set(float(f1))
    if roc_auc is not None:
        EVAL_ROC_AUC.set(float(roc_auc))
    push_metrics()

    return {
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_txt,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_reports(results: dict, X_test: pd.DataFrame, y_test: pd.Series, out_dir: str = None):
    from src.utils.config import REPORTS_DIR
    if out_dir is None:
        out = Path(REPORTS_DIR)
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "f1": results["f1"],
        "roc_auc": results["roc_auc"],
        "confusion_matrix": results["confusion_matrix"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")  
    df_pred = X_test.copy()
    df_pred["y_true"] = y_test.values
    df_pred["y_pred"] = results["y_pred"]
    if results["y_proba"] is not None:
        df_pred["score_risco"] = (results["y_proba"] * 100).round(2)

    df_pred.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    print(f"Reports salvos em: {out.resolve()}")


if __name__ == "__main__":
    from src.utils.utils import load_data

    cfg = TrainConfig()
    df = load_data("data/refined/dados_modelo.csv")
    X_train, X_test, y_train, y_test = split_by_year(df, cfg=cfg)

    model = joblib.load("models/model.pkl")
    results = evaluate_model(model, X_test, y_test)
    save_reports(results, X_test, y_test, out_dir="reports")