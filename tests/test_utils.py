import pandas as pd
import pytest
from src.utils.utils import load_data, drop_unused_columns

def test_load_data_csv(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    df_loaded = load_data(file)
    assert df_loaded.equals(df)

def test_drop_unused_columns():
    df = pd.DataFrame({'POSSUI_DEFASAGEM': [1], 'ANO': [2022], 'IAN': [1], 'a': [5]})
    df2 = drop_unused_columns(df)
    assert 'POSSUI_DEFASAGEM' not in df2.columns
    assert 'ANO' not in df2.columns
    assert 'IAN' not in df2.columns
    assert 'a' in df2.columns


def test_load_data_errors(tmp_path):
    # missing file
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / 'nofile.csv'))
    # unsupported extension
    file = tmp_path / 'foo.txt'
    file.write_text('hello')
    with pytest.raises(ValueError):
        load_data(str(file))


def test_load_data_excel_and_parquet(tmp_path):
    df = pd.DataFrame({'a': [1]})
    xlsx = tmp_path / 'test.xlsx'
    df.to_excel(xlsx, index=False)
    pd.testing.assert_frame_equal(load_data(str(xlsx)), df)
