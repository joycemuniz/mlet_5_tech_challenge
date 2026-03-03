import pandas as pd
import pytest
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


def test_split_by_year_errors():
    df = pd.DataFrame({'POSSUI_DEFASAGEM': [1], 'ANO': [2022]})
    # remove ANO column
    with pytest.raises(ValueError):
        split_by_year(df.drop(columns=['ANO']))
    # missing target
    df2 = pd.DataFrame({'ANO': [2022]})
    with pytest.raises(ValueError):
        split_by_year(df2)

def test_train_model():
    import numpy as np
    X = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])
    y = pd.Series([0, 1]*5)
    model = train_model(X, y)
    assert hasattr(model, 'predict')


def test_split_no_test_year_raises():
    """Nenhum dado no ano de teste → ValueError (linha 43/45 de train.py)."""
    df = pd.DataFrame({
        'ANO': [2022, 2023, 2022],
        'POSSUI_DEFASAGEM': [1, 0, 1],
    })
    # test_year=2024 mas não existe → deve lançar
    with pytest.raises(ValueError, match="Nenhum dado encontrado para teste"):
        split_by_year(df)


def test_split_no_train_year_raises():
    """Nenhum dado nos anos de treino → ValueError."""
    df = pd.DataFrame({
        'ANO': [2024, 2024],
        'POSSUI_DEFASAGEM': [1, 0],
    })
    with pytest.raises(ValueError, match="Nenhum dado encontrado para treino"):
        split_by_year(df)


def test_save_model(tmp_path):
    """save_model persiste e o arquivo pode ser recarregado (linha 93 de train.py)."""
    import numpy as np
    import joblib
    from src.modeling.train import save_model

    X = pd.DataFrame(np.random.rand(6, 2), columns=['a', 'b'])
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = train_model(X, y)

    out = tmp_path / 'model.pkl'
    save_model(model, str(out))
    assert out.exists()

    loaded = joblib.load(str(out))
    assert hasattr(loaded, 'predict')


def test_train_model_with_custom_config():
    """Treinar passando TrainConfig explícito (cobre ramo cfg != None)."""
    import numpy as np
    from src.modeling.train import TrainConfig

    X = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])
    y = pd.Series([0, 1] * 5)
    cfg = TrainConfig(n_estimators=10, random_state=0)
    model = train_model(X, y, cfg=cfg)
    assert hasattr(model, 'predict')
