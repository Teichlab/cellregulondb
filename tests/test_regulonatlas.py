import pytest
import scipy as sp
import pandas as pd
import scanpy as sc
from cellregulondb.cellregulondb import CellRegulonDB
from cellregulondb.regulonatlas import RegulonAtlas


def test_get_regulons(db_path):
    # Initialize CellRegulonDB object
    cell_regulon_db = CellRegulonDB(db_path)

    # Call get_regulons method
    df = cell_regulon_db.get_regulons(cell_types=["ionocyte"])

    # Assert that the returned object is a DataFrame
    assert isinstance(
        df, pd.DataFrame
    ), "get_regulons method does not return a DataFrame"

    # Check if the adata object is the same as the one returned by the method

    ra = RegulonAtlas()
    ra.load_from_df(df)

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
            == df[check].sort_values(check).reset_index(drop=True)
        )
        .all()
        .all()
    ), "dataframes before and after transformation to anndata are not equal"
