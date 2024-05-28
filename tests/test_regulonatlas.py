import networkx as nx
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


def test_get_target_genes(regulon_df):
    """Test if the target genes are correctly extracted from the regulon dataframe"""
    # get sorted list of target genes
    target_genes_df_sorted = regulon_df["target_gene"].value_counts()

    # convert the regulon dataframe to anndata
    ra = RegulonAtlas()
    ra.load_from_df(regulon_df)

    # check the returned target genes
    target_genes = ra.get_target_genes()

    assert set(target_genes) == set(
        target_genes_df_sorted.index.tolist()
    ), "target genes are not correctly extracted"

    # check the returned target genes with `min_regulon` parameter
    target_genes = ra.get_target_genes(min_regulon=5)

    assert set(target_genes) == set(
        target_genes_df_sorted[target_genes_df_sorted >= 5].index.tolist()
    ), "target genes for `min_regulon=5` are not correctly extracted"

    # check the `top` parameter
    target_genes = ra.get_target_genes(top=10)
    assert (
        len(target_genes) == 10
    ), "target genes for `top=10` are not correctly extracted"

    # check the return_counts parameter
    target_genes = ra.get_target_genes(return_counts=True, min_regulon=0)
    assert isinstance(
        target_genes, dict
    ), "target genes are not returned as a dictionary"
    assert (
        len(target_genes) == regulon_df["target_gene"].nunique()
    ), "counts not returned for all target genes"


def test_get_tf_dict(regulon_df):
    """Test if the regulons are correctly extracted from RegulonAtlas"""
    tf_dict_df = (
        regulon_df.query("regulation == '+'")
        .groupby("transcription_factor")["target_gene"]
        .apply(list)
        .to_dict()
    )

    ra = RegulonAtlas()
    ra.load_from_df(regulon_df)

    tf_dict = ra.get_tf_dict()

    for tf, tgs in tf_dict_df.items():
        assert set(tf_dict[tf]) == set(tgs), f"TF {tf} target genes do not match"


def test_to_networkx(regulon_df):
    """Test if the networkx graph is correctly created from the regulon dataframe"""

    ra = RegulonAtlas()
    ra.load_from_df(regulon_df)
    ra.cell_type_col = "cell_type"

    G = ra.to_networkx()

    assert nx.is_directed(G), "graph is not directed"

    # check if the graph can be converted back to dataframe
    df = nx.to_pandas_edgelist(G)
    assert (
        (
            df.sort_values(["source", "target"])
            .rename(columns={"source": "transcription_factor", "target": "target_gene"})
            .reset_index(drop=True)
            == regulon_df.query("regulation == '+'")[
                ["transcription_factor", "target_gene"]
            ]
            .drop_duplicates()
            .sort_values(["transcription_factor", "target_gene"])
            .reset_index(drop=True)
        )
        .all()
        .all()
    ), "dataframes before and after transformation to networkx are not equal"


def test_get_tables(regulon_df):
    """Test if the tables are correctly extracted from RegulonAtlas"""
    regulon_df = regulon_df.assign(
        cell_type_info=regulon_df["cell_type"].astype(str) + "_info"
    )  # add cell type info column for testing

    ra = RegulonAtlas()
    ra.load_from_df(regulon_df)
    ra.cell_type_col = "cell_type"

    # test get_tables for transcription factors
    links, regs, tgs = ra.get_tables(by=["transcription_factor"])
    assert set(links["source"].tolist()) == set(
        regulon_df["transcription_factor"].tolist()
    ), "sources are not transcription factors"
    assert set(links["target"].tolist()) == set(
        regulon_df["target_gene"].tolist()
    ), "targets are not target genes"
    assert set(regs["source"].tolist()) == set(
        regulon_df["transcription_factor"].tolist()
    ), "regulon node attributes are not for transcription factors"
    assert set(tgs["target"].tolist()) == set(
        regulon_df["target_gene"].tolist()
    ), "gene node attributes are not for target genes"

    # test get_tables for (transcription factors, cell types)
    links, regs, tgs = ra.get_tables(
        by=["transcription_factor", "cell_type"], node_columns=["cell_type_info"]
    )
    parts = links["source"].str.split(" - ", expand=True)
    assert set(parts[0].tolist()) == set(
        regulon_df["transcription_factor"].tolist()
    ), "part 1 of sources are not transcription factors"
    assert set(parts[1].tolist()) == set(
        regulon_df["cell_type"].tolist()
    ), "part 2 of sources are not cell types"
    assert set(links["target"].tolist()) == set(
        regulon_df["target_gene"].tolist()
    ), "targets are not target genes"
    assert set(tgs["target"].tolist()) == set(
        regulon_df["target_gene"].tolist()
    ), "gene node attributes are not for target genes"
    assert (
        "cell_type_info" in regs.columns
    ), "cell_type_info column not assigned to regulon attributes"
