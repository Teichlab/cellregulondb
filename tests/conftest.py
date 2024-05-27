import pytest
import pandas as pd


@pytest.fixture
def regulon_df():
    df = pd.read_csv("tests/test_data/ionocyte_regulons.csv", index_col=0)
    df = df.sample(n=1000, random_state=0)
    return df
