from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import SplitConfig, ModelConfig, FEATURE_COLUMNS
from src.utils.metrics import TRAIN_DURATION, push_metrics


@dataclass(frozen=True)
class TrainConfig:
    train_years: Tuple[int, ...] = (2022, 2023)
    test_year: int = 2024
    target_col: str = "POSSUI_DEFASAGEM"
    drop_cols: Tuple[str, ...] = ("ANO", "IAN")
    n_estimators: int = 200
    random_state: int = 42
    class_weight: str = "balanced"

def split_by_year(df: pd.DataFrame, cfg: TrainConfig = None):
    if cfg is None:
        split_cfg = SplitConfig()
        cfg = TrainConfig(
            train_years=split_cfg.train_years,
            test_year=split_cfg.test_year,
            target_col=split_cfg.target_col,
            drop_cols=split_cfg.drop_cols,
        )

    if "ANO" not in df.columns:
        raise ValueError("Coluna 'ANO' não encontrada no DataFrame.")

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target '{cfg.target_col}' não encontrado no DataFrame.")

    train_mask = df["ANO"].isin(cfg.train_years)
    test_mask = df["ANO"] == cfg.test_year

    if train_mask.sum() == 0:
        raise ValueError(f"Nenhum dado encontrado para treino nos anos {cfg.train_years}.")
    if test_mask.sum() == 0:
        raise ValueError(f"Nenhum dado encontrado para teste no ano {cfg.test_year}.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        # Add missing features to the dataframe filled with zeros.  This keeps
        # the feature vector dimension constant and lets tests use toy data
        # without all real feature columns.
        for c in missing:
            df[c] = 0

    X = df[FEATURE_COLUMNS].copy()
    y = df[cfg.target_col].astype(int)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: TrainConfig = None):
    if cfg is None:
        model_cfg = ModelConfig()
        split_cfg = SplitConfig()
        cfg = TrainConfig(
            n_estimators=model_cfg.n_estimators,
            random_state=model_cfg.random_state,
            class_weight=model_cfg.class_weight,
            train_years=split_cfg.train_years,
            test_year=split_cfg.test_year,
            target_col=split_cfg.target_col,
            drop_cols=split_cfg.drop_cols,
        )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        class_weight=cfg.class_weight,
        n_jobs=-1,
    )

    with TRAIN_DURATION.time():
        model.fit(X_train, y_train)

    push_metrics()
    return model

def save_model(model, path: str):
    joblib.dump(model, path)

if __name__ == "__main__":
    from src.utils.utils import load_data
    from src.utils.config import REFINED_DATASET_PATH, MODEL_PATH

    df = load_data(str(REFINED_DATASET_PATH))
    cfg = TrainConfig()

    X_train, X_test, y_train, y_test = split_by_year(df, cfg=cfg)

    model = train_model(X_train, y_train, cfg=cfg)
    save_model(model, str(MODEL_PATH))

    print(f"Modelo treinado e salvo em {MODEL_PATH}")