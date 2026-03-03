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


def test_evaluate_with_proba_and_save(tmp_path):
    """Modelo COM predict_proba → cobre linha 60 (print ROC) e linha 74 (score_risco)."""
    import numpy as np
    from src.modeling.evaluate import save_reports

    X = pd.DataFrame(np.random.rand(20, 4), columns=['a', 'b', 'c', 'd'])
    y = pd.Series([0, 1] * 10)
    model = train_model(X, y)
    results = evaluate_model(model, X, y)

    # roc_auc deve ser calculado pois RandomForest tem predict_proba
    assert results['roc_auc'] is not None
    assert results['y_proba'] is not None

    outdir = tmp_path / 'with_proba'
    save_reports(results, X, y, out_dir=str(outdir))
    df = pd.read_csv(outdir / 'predictions.csv')
    # score_risco só aparece quando y_proba não é None (linha 74)
    assert 'score_risco' in df.columns


def test_save_reports_default_dir(monkeypatch, tmp_path):
    """save_reports sem out_dir → usa REPORTS_DIR da config (linha 57)."""
    import numpy as np
    from src.modeling.evaluate import save_reports
    import src.utils.config as cfg

    monkeypatch.setattr(cfg, 'REPORTS_DIR', tmp_path)

    class Dummy:
        def predict(self, X):
            return [0] * len(X)

    X = pd.DataFrame(np.random.rand(4, 2), columns=['a', 'b'])
    y = pd.Series([0, 1, 0, 1])
    results = evaluate_model(Dummy(), X, y)
    save_reports(results, X, y)  # sem out_dir
    assert (tmp_path / 'metrics.json').exists()
