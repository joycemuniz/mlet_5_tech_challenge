from src.utils.config import (
    ensure_dirs,
    RAW_XLSX_PATH,
    INTERIM_DATASET_PATH,
    REFINED_DATASET_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SplitConfig,
    ModelConfig,
    FEATURE_COLUMNS,  # <- precisa existir no config.py
)

from src.pipeline.data_consolidation import consolidate_data
from src.pipeline.preprocessing import preprocess
from src.pipeline.feature_engineering import create_features

from src.utils.utils import load_data
from src.modeling.train import split_by_year, train_model, save_model, TrainConfig
from src.modeling.evaluate import evaluate_model, save_reports

# opcional (se você já criou o arquivo feature_importance.py)
# from src.modeling.feature_importance import save_feature_importance


def main():
    # 0) Garante pastas
    ensure_dirs()

    # 1) Consolidação: raw (xlsx) -> interim (csv)
    consolidate_data(
        xlsx_path=str(RAW_XLSX_PATH),
        output_path=str(INTERIM_DATASET_PATH),
    )

    # 2) Pré-processamento: interim -> refined
    preprocess(
        input_path=str(INTERIM_DATASET_PATH),
        output_path=str(REFINED_DATASET_PATH),
    )

    # 3) Carregar base refined
    df = load_data(str(REFINED_DATASET_PATH))

    # 4) Feature engineering (não remove colunas)
    df = create_features(df)

    # 5) Config final (split + modelo)
    split_cfg = SplitConfig()
    model_cfg = ModelConfig()
    cfg = TrainConfig(
        train_years=split_cfg.train_years,
        test_year=split_cfg.test_year,
        target_col=split_cfg.target_col,
        drop_cols=split_cfg.drop_cols,
        n_estimators=model_cfg.n_estimators,
        random_state=model_cfg.random_state,
        class_weight=model_cfg.class_weight,
    )

    # 6) Split temporal (saída deve ser X_train/X_test sem target)
    X_train, X_test, y_train, y_test = split_by_year(df, cfg=cfg)

    # 7) FORÇA conjunto fixo de features (contrato do projeto)
    # Isso garante que o modelo sempre treina e prevê com as mesmas variáveis.
    X_train = X_train[FEATURE_COLUMNS].copy()
    X_test = X_test[FEATURE_COLUMNS].copy()

    # 8) Treinar com TODAS as features fixas
    model = train_model(X_train, y_train, cfg=cfg)

    # 9) Avaliar
    results = evaluate_model(model, X_test, y_test)

    # 10) Salvar reports
    save_reports(results, X_test, y_test, out_dir=str(REPORTS_DIR))

    # 11) Salvar modelo
    save_model(model, path=str(MODEL_PATH))

    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()