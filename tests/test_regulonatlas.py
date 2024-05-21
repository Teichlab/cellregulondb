import pytest
import scipy as sp
import pandas as pd
import scanpy as sc
from cellregulondb.cellregulondb import CellRegulonDB
from cellregulondb.regulonatlas import RegulonAtlas


def test_get_regulons(regulon_df):
    """Test if the regulon dataframe is correctly converted to anndata and back to a dataframe"""
    ra = RegulonAtlas()
    ra.load_from_df(regulon_df)

    ad_tf = ra.get_df()

    check = [
        "transcription_factor",
        "target_gene",
        "regulation",
        "tissue",
        "cell_type",
        "author_cell_type",
        "rss",
    ]

    assert (
        (
            ad_tf[check].sort_values(check).reset_index(drop=True)
            == regulon_df[check].sort_values(check).reset_index(drop=True)
        )
        .all()
        .all()
    ), "dataframes before and after transformation to anndata are not equal"
