import pytest
import pandas as pd
from src.pipeline.preprocessing import preprocess
from src.utils.config import REFINED_DATASET_PATH

def test_preprocess_runs(tmp_path):
    # Cria um DataFrame mínimo
    df = pd.DataFrame({
        'DEFASAGEM': [1, -1],
        'IDADE': [10, None],
        'FASE': ['ALFA', '2'],
        'GENERO': ['MASCULINO', 'FEMININO'],
        'INSTITUICAO_ENSINO': ['Pública', 'Privada'],
        'ANO': [2022, 2023],
        'INDE_2022': [1, 2], 'INDE_2023': [3, 4], 'INDE_2024': [5, 6],
        'IAA': [1, 2], 'IEG': [1, 2], 'IPS': [1, 2], 'IDA': [1, 2], 'IND_PV': [1, 2],
        'IAN': [1, 2], 'IPP': [1, 2], 'NOTA_MATEM': [1, 2], 'NOTA_PORT': [1, 2]
    })
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    df.to_csv(input_file, index=False)
    preprocess(input_path=input_file, output_path=output_file)
    assert output_file.exists()
    df_out = pd.read_csv(output_file)
    assert 'POSSUI_DEFASAGEM' in df_out.columns
