import pandas as pd
from pathlib import Path


def consolidate_data(xlsx_path, output_path):
    """
    Consolida e padroniza dados de múltiplas sheets do arquivo Excel em um único CSV.
    Aplica renomeação de colunas, inclui coluna de ano, concatena e salva em data/interim.
    """

    colunas_2022 = {
        "RA": "RA",
        "Fase": "FASE",
        "Turma": "TURMA",
        "Nome": "NOME_ALUNO",
        "Ano nasc": "ANO_NASC",
        "Idade 22": "IDADE",
        "Gênero": "GENERO",
        "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO_ENSINO",
        "Pedra 20": "PEDRA_2020",
        "Pedra 21": "PEDRA_2021",
        "Pedra 22": "PEDRA_2022",
        "INDE 22": "INDE_2022",
        "Cg": "CG",  # ✅ corrigido (antes estava GC)
        "Cf": "CF",
        "Ct": "CT",
        "Nº Av": "N_AVALIADOR",
        "Avaliador1": "AVALIADOR_1",
        "Rec Av1": "REC_EQUIPE_1",
        "Avaliador2": "AVALIADOR_2",
        "Rec Av2": "REC_EQUIPE_2",
        "Avaliador3": "AVALIADOR_3",
        "Rec Av3": "REC_EQUIPE_3",
        "Avaliador4": "AVALIADOR_4",
        "Rec Av4": "REC_EQUIPE_4",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "Rec Psicologia": "REC_PSICO",
        "IDA": "IDA",
        "Matem": "NOTA_MATEM",
        "Portug": "NOTA_PORT",
        "Inglês": "NOTA_INGL",
        "Indicado": "INDC_BOLSA",
        "Atingiu PV": "ATING_PV",
        "IPV": "IND_PV",
        "IAN": "IAN",
        "Fase ideal": "NIVEL_IDEAL",
        "Defas": "DEFASAGEM",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
    }

    colunas_2023 = {
        "RA": "RA",
        "Fase": "FASE",
        "INDE 2023": "INDE_2023",
        "Pedra 2023": "PEDRA_2023",
        "Turma": "TURMA",
        "Nome Anonimizado": "NOME_ALUNO",
        "Data de Nasc": "DT_NASC",
        "Idade": "IDADE",
        "Gênero": "GENERO",
        "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO_ENSINO",
        "Pedra 20": "PEDRA_2020",
        "Pedra 21": "PEDRA_2021",
        "Pedra 22": "PEDRA_2022",
        "INDE 22": "INDE_2022",
        "Cg": "CG",
        "Cf": "CF",
        "Ct": "CT",
        "Nº Av": "N_AVALIADOR",
        "Avaliador1": "AVALIADOR_1",
        "Rec Av1": "REC_EQUIPE_1",
        "Avaliador2": "AVALIADOR_2",
        "Rec Av2": "REC_EQUIPE_2",
        "Avaliador3": "AVALIADOR_3",
        "Rec Av3": "REC_EQUIPE_3",
        "Avaliador4": "AVALIADOR_4",
        "Rec Av4": "REC_EQUIPE_4",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IPP": "IPP",
        "Rec Psicologia": "REC_PSICO",
        "IDA": "IDA",
        "Mat": "NOTA_MATEM",
        "Por": "NOTA_PORT",
        "Ing": "NOTA_INGL",
        "Indicado": "INDC_BOLSA",
        "Atingiu PV": "ATING_PV",
        "IPV": "IND_PV",
        "IAN": "IAN",
        "Fase Ideal": "NIVEL_IDEAL",
        "Defasagem": "DEFASAGEM",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
        "Destaque IPV.1": "DESTAQUE_IPV.1",
    }

    colunas_2024 = {
        "RA": "RA",
        "Fase": "FASE",
        "INDE 2024": "INDE_2024",
        "Pedra 2024": "PEDRA_2024",
        "Turma": "TURMA",
        "Nome Anonimizado": "NOME_ALUNO",
        "Data de Nasc": "DT_NASC",
        "Idade": "IDADE",
        "Gênero": "GENERO",
        "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO_ENSINO",
        "Pedra 20": "PEDRA_2020",
        "Pedra 21": "PEDRA_2021",
        "Pedra 22": "PEDRA_2022",
        "Pedra 23": "PEDRA_2023",
        "INDE 22": "INDE_2022",
        "INDE 23": "INDE_2023",
        "Cg": "CG",
        "Cf": "CF",
        "Ct": "CT",
        "Nº Av": "N_AVALIADOR",
        "Avaliador1": "AVALIADOR_1",
        "Rec Av1": "REC_EQUIPE_1",
        "Avaliador2": "AVALIADOR_2",
        "Rec Av2": "REC_EQUIPE_2",  # ✅ corrigido (antes estava REC_EQUIPE_3)
        "Avaliador3": "AVALIADOR_3",
        "Avaliador4": "AVALIADOR_4",
        "Avaliador5": "AVALIADOR_5",
        "Avaliador6": "AVALIADOR_6",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IPP": "IPP",
        "Rec Psicologia": "REC_PSICO",
        "IDA": "IDA",
        "Mat": "NOTA_MATEM",
        "Por": "NOTA_PORT",
        "Ing": "NOTA_INGL",
        "Indicado": "INDC_BOLSA",
        "Atingiu PV": "ATING_PV",
        "IPV": "IND_PV",
        "IAN": "IAN",
        "Fase Ideal": "NIVEL_IDEAL",
        "Defasagem": "DEFASAGEM",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
        "Escola": "ESCOLA",
        "Ativo/ Inativo": "ATIVO/INATIVO",
        "Ativo/ Inativo.1": "ATIVO/INATIVO.1",
    }

    df_2022 = pd.read_excel(xlsx_path, sheet_name="PEDE2022").rename(columns=colunas_2022)
    df_2023 = pd.read_excel(xlsx_path, sheet_name="PEDE2023").rename(columns=colunas_2023)
    df_2024 = pd.read_excel(xlsx_path, sheet_name="PEDE2024").rename(columns=colunas_2024)

    df_2022["ANO"] = 2022
    df_2023["ANO"] = 2023
    df_2024["ANO"] = 2024

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

    if "IDADE" in df.columns:
        df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Base consolidada salva em: {output_path}")


if __name__ == "__main__":
    xlsx_path = "data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    output_path = "data/interim/dataset_concatenado.csv"
    consolidate_data(xlsx_path, output_path)