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


def test_evaluate_no_proba_and_save(tmp_path):
    # dummy model without predict_proba
    class Dummy:
        def predict(self, X):
            return [0]*len(X)
    import numpy as np
    X = pd.DataFrame(np.random.rand(5, 2), columns=['a', 'b'])
    y = pd.Series([0, 1, 0, 1, 0])
    results = evaluate_model(Dummy(), X, y)
    assert results['roc_auc'] is None

    # check save_reports writes expected files and contents
    outdir = tmp_path / 'out'
    evaluate_model(Dummy(), X, y)  # update metrics registry but not needed
    from src.modeling.evaluate import save_reports
    save_reports(results, X, y, out_dir=str(outdir))
    assert (outdir / 'metrics.json').exists()
    assert (outdir / 'predictions.csv').exists()
    df = pd.read_csv(outdir / 'predictions.csv')
    assert 'y_true' in df.columns
    assert 'y_pred' in df.columns
