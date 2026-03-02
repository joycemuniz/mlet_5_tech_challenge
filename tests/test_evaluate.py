import pandas as pd
from src.modeling.evaluate import evaluate_model
from src.modeling.train import train_model

def test_evaluate_model():
    import numpy as np
    X = pd.DataFrame(np.random.rand(20, 4), columns=['a', 'b', 'c', 'd'])
    y = pd.Series([0, 1]*10)
    model = train_model(X, y)
    results = evaluate_model(model, X, y)
    assert 'f1' in results
    assert 'confusion_matrix' in results
    assert 'classification_report' in results
