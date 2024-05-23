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
