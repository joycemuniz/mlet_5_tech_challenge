import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if "IPP_IMPUTADO" in df.columns:
        df["IPP_FALTANTE"] = (df["IPP_IMPUTADO"] == 1).astype(int)

    if "INDE_ATUAL" in df.columns and "IDA" in df.columns:
        df["INDE_X_IDA"] = df["INDE_ATUAL"] * df["IDA"]

    return df


def select_features(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, k: int = 15):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from src.utils.config import ModelConfig

    X_train_num = X_train.select_dtypes(include=["number"]).copy()
    X_test_num = X_test[X_train_num.columns].copy()

    X_train_num = X_train_num.fillna(0)
    X_test_num = X_test_num.fillna(0)

    cfg = ModelConfig()
    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        n_jobs=-1
    )
    rf.fit(X_train_num, y_train)

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:k]
    top_cols = X_train_num.columns[idx].tolist()

    return X_train_num[top_cols], X_test_num[top_cols], top_cols