from pathlib import Path
import pandas as pd


def load_data(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")

    if path.suffix == ".csv":
        return pd.read_csv(path)

    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    else:
        raise ValueError(f"Formato de arquivo não suportado: {path.suffix}")


def drop_unused_columns(df):
    cols_excluir = ["POSSUI_DEFASAGEM", "ANO", "IAN"]
    return df.drop(columns=cols_excluir, errors="ignore")


if __name__ == "__main__":
    from utils.config import REFINED_DATASET_PATH
    df = load_data(REFINED_DATASET_PATH)
    df = drop_unused_columns(df)
    print("Shape:", df.shape)
    print(df.head())