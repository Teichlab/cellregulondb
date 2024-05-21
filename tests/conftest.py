import pytest
import pandas as pd


@pytest.fixture
def regulon_df():
    df = pd.read_csv("test_data/ionocyte_regulons.csv", index_col=0)
    return df
