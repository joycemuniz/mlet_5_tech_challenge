import pandas as pd
from src.modeling.train import split_by_year, train_model

def test_split_by_year():
    df = pd.DataFrame({
        'ANO': [2022, 2023, 2024],
        'POSSUI_DEFASAGEM': [1, 0, 1],
        'IAN': [1, 2, 3],
        'f1': [1, 2, 3],
        'f2': [4, 5, 6]
    })
    X_train, X_test, y_train, y_test = split_by_year(df)
    assert len(X_train) == 2
    assert len(X_test) == 1
    assert y_train.tolist() == [1, 0]
    assert y_test.tolist() == [1]

def test_train_model():
    import numpy as np
    X = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])
    y = pd.Series([0, 1]*5)
    model = train_model(X, y)
    assert hasattr(model, 'predict')
