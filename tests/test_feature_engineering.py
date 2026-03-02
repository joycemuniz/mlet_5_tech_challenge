import pandas as pd
from src.pipeline.feature_engineering import create_features, select_features

def test_create_features():
    df = pd.DataFrame({'IPP_IMPUTADO': [1, 0], 'INDE_ATUAL': [2, 3], 'IDA': [4, 5]})
    df2 = create_features(df.copy())
    assert 'IPP_FALTANTE' in df2.columns
    assert 'INDE_X_IDA' in df2.columns

def test_select_features():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=20, n_features=12, random_state=42)
    import numpy as np
    import pandas as pd
    X_train = pd.DataFrame(X[:10], columns=[f'f{i}' for i in range(12)])
    X_test = pd.DataFrame(X[10:], columns=[f'f{i}' for i in range(12)])
    y_train = pd.Series(y[:10])
    X_train_sel, X_test_sel, top_cols = select_features(X_train, y_train, X_test, k=5)
    assert len(top_cols) == 5
    assert X_train_sel.shape[1] == 5
    assert X_test_sel.shape[1] == 5
