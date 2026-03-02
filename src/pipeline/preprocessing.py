import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import INTERIM_DATASET_PATH, REFINED_DATASET_PATH, ensure_dirs


def preprocess(input_path: str = None, output_path: str = None):
    ensure_dirs()

    if input_path is None:
        input_path = INTERIM_DATASET_PATH
    if output_path is None:
        output_path = REFINED_DATASET_PATH

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # --- validações básicas ---
    if "ANO" not in df.columns:
        raise ValueError("Coluna ANO não encontrada (necessária para split temporal).")
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")

    if "DEFASAGEM" not in df.columns:
        raise ValueError("Coluna DEFASAGEM não encontrada.")
    df["POSSUI_DEFASAGEM"] = (pd.to_numeric(df["DEFASAGEM"], errors="coerce") < 0).astype(int)

    # --- tipagem/encoding ---
    if "IDADE" in df.columns:
        df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")

    if "FASE" in df.columns:
        df["FASE"] = (
            df["FASE"].replace("ALFA", "0")
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )
        df = df[df["FASE"].notna()].copy()
        df["FASE"] = df["FASE"].astype(int)

        # Remover fases sem label
        df = df[~df["FASE"].isin([8, 9])].copy()
    else:
        raise ValueError("Coluna FASE não encontrada (necessária para imputações por fase).")

    if "GENERO" in df.columns:
        df["GENERO"] = df["GENERO"].astype(str).str.strip().str.upper()
        df["GENERO"] = df["GENERO"].replace(
            {"MASCULINO": 0, "MENINO": 0, "FEMININO": 1, "MENINA": 1}
        ).astype("Int64")

    if "INSTITUICAO_ENSINO" in df.columns:
        df["INSTITUICAO_ENSINO"] = df["INSTITUICAO_ENSINO"].astype(str).str.strip()
        map_instituicao = {
            "Escola Pública": 1,
            "Pública": 1,
            "Rede Decisão": 2,
            "Escola JP II": 2,
            "Privada": 2,
            "Privada - Programa de Apadrinhamento": 3,
            "Privada - Programa de apadrinhamento": 3,
            "Privada *Parcerias com Bolsa 100%": 4,
            "Privada - Pagamento por *Empresa Parceira": 5,
            "Concluiu o 3º EM": 6,
            "Bolsista Universitário *Formado (a)": 7,
            "Nenhuma das opções acima": 8,
        }
        df["INSTITUICAO_ENSINO"] = df["INSTITUICAO_ENSINO"].map(map_instituicao).fillna(8).astype(int)

    # --- seleção de colunas: robusta ---
    wanted = [
        "ANO", "FASE", "IDADE", "GENERO", "ANO_INGRESSO", "INSTITUICAO_ENSINO",
        "INDE_2022", "INDE_2023", "INDE_2024",
        "IAA", "IEG", "IPS", "IDA", "IND_PV", "IAN", "IPP",
        "NOTA_MATEM", "NOTA_PORT", "POSSUI_DEFASAGEM"
    ]

    # cria colunas que faltarem como NaN
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    df_modelo = df[wanted].copy()

    # --- INDE_ATUAL ---
    df_modelo["INDE_ATUAL"] = np.where(
        df_modelo["ANO"] == 2022, df_modelo["INDE_2022"],
        np.where(df_modelo["ANO"] == 2023, df_modelo["INDE_2023"],
                 np.where(df_modelo["ANO"] == 2024, df_modelo["INDE_2024"], np.nan))
    )
    df_modelo.drop(columns=["INDE_2022", "INDE_2023", "INDE_2024"], inplace=True, errors="ignore")

    # --- imputações ---
    df_modelo["IDADE"] = pd.to_numeric(df_modelo["IDADE"], errors="coerce")
    df_modelo["IDADE"] = df_modelo.groupby(["FASE", "ANO"])["IDADE"].transform(lambda x: x.fillna(x.median()))
    df_modelo["IDADE"] = df_modelo["IDADE"].fillna(df_modelo["IDADE"].median())

    # IPP: imputar 2022 usando 2023 (treino)
    df_modelo["IPP_IMPUTADO"] = 0
    df_modelo["IPP"] = pd.to_numeric(df_modelo["IPP"], errors="coerce")

    na_ipp_2022 = (df_modelo["ANO"] == 2022) & (df_modelo["IPP"].isna())
    mediana_ipp_por_fase_2023 = df_modelo[df_modelo["ANO"] == 2023].groupby("FASE")["IPP"].median()
    df_modelo.loc[na_ipp_2022, "IPP"] = df_modelo.loc[na_ipp_2022, "FASE"].map(mediana_ipp_por_fase_2023)
    df_modelo.loc[na_ipp_2022, "IPP_IMPUTADO"] = 1

    df_modelo["IPP"] = df_modelo["IPP"].fillna(df_modelo["IPP"].median())

    # imputação por fase/ano: notas e indicadores
    num_cols = ["NOTA_PORT", "NOTA_MATEM", "INDE_ATUAL", "IDA", "IND_PV", "IEG", "IPS", "IAA", "IAN"]
    for col in num_cols:
        df_modelo[col] = pd.to_numeric(df_modelo[col], errors="coerce")
        df_modelo[col] = df_modelo.groupby(["FASE", "ANO"])[col].transform(lambda x: x.fillna(x.median()))
        df_modelo[col] = df_modelo[col].fillna(df_modelo[col].median())

    df_modelo.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Base refinada salva em: {output_path}")


if __name__ == "__main__":
    preprocess()