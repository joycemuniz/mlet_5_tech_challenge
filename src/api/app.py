from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.utils.config import MODEL_PATH, MODELS_DIR


app = FastAPI(title="MLET 5 - Modelo de Score de Defasagem Escolar", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}

_model = None
_features = None


def load_artifacts():
    global _model, _features

    model_path = Path("models/model.pkl")
    features_path = Path(MODELS_DIR) / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path.resolve()}")

    _model = joblib.load("models/model.pkl")

    if features_path.exists():
        _features = json.loads(features_path.read_text(encoding="utf-8"))
    else:
        _features = None


@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.post("/score")
def score(payload: Dict[str, Any]):
    if _model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    df = pd.DataFrame([payload])

    if _features is not None:
        for col in _features:
            if col not in df.columns:
                df[col] = 0
        df = df[_features]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(df)[:, 1][0])
        pred = int(_model.predict(df)[0]) if hasattr(_model, "predict") else int(proba >= 0.5)

    elif hasattr(_model, "predict"):
        pred = int(_model.predict(df)[0])
        proba = float(pred)

    else:
        raise HTTPException(status_code=500, detail="Modelo incompatível.")

    return {"classe_predita": pred, "score_risco": round(proba * 100, 2)}

