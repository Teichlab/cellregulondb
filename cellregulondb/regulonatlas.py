import scipy as sp
import pandas as pd
import scanpy as sc


class RegulonAtlas:
    """
    A class to handle regulon data and perform various operations on it.

    This class is a wrapper around `AnnData` and provides methods to load data from a DataFrame,
    convert the data back to a DataFrame, calculate embeddings, plot embeddings, and score gene sets.
    Regulons are stored as a binary matrix (regulons x target genes) in the `X` attribute of the `AnnData` object.

    Attributes:
        adata (sc.AnnData): An AnnData object to store the regulon data.

    Methods:
        load_from_df(df: pd.DataFrame): Loads data from a DataFrame into `self.adata`.
        get_df(): Converts the data in `self.adata` back to a DataFrame.
        calculate_embedding(plot: bool = False, add_leiden: float = None): Calculates UMAP embeddings for the data in `self.adata`.
        plot_embedding(**kwargs): Plots the UMAP embeddings of the data in `self.adata`.
        score_gene_set(gene_set: list): Placeholder method for scoring a gene set.
        perturbation_direction(gene_set: list): Placeholder method for calculating the perturbation direction of a gene set.
    """

    def __init__(self, adata: sc.AnnData = None):
        self.adata: sc.AnnData = adata

    def load_from_df(self, df: pd.DataFrame) -> None:
        """
        Loads data from a DataFrame into `self.adata`.

        This method takes a DataFrame that contains regulon information and converts it into an AnnData object.
        The DataFrame should have columns for 'transcription_factor', 'regulation', 'author_cell_type', and 'tissue'.
        These columns are combined to create a 'regulon' ID.
        The DataFrame is converted into a sparse matrix and stored in the AnnData object's 'X' attribute.
        Additional information from the DataFrame is stored in the AnnData object's 'obs' attribute.

        Args:
            df (pd.DataFrame): A DataFrame containing regulon information.
        """
        df = df.copy()
        df["regulon"] = (
            df["transcription_factor"].astype(str)
            + " - ("
            + df["regulation"]
            + ") - "
            + df["author_cell_type"]
            + " - "
            + df["tissue"]
        )

        # Create an anndata object from the DataFrame
        val_df = pd.DataFrame(
            [1] * df.shape[0],
            index=pd.MultiIndex.from_frame(df[["regulon", "target_gene"]]),
        ).unstack(fill_value=0)

        adata = sc.AnnData(
            X=sp.sparse.csr_matrix(val_df),
            obs=pd.DataFrame(index=val_df.index.get_level_values("regulon")),
            var=pd.DataFrame(index=val_df.columns.get_level_values("target_gene")),
        )

        # Add obs information to adata
        adata.obs = pd.concat(
            [
                adata.obs,
                df.drop(columns=["target_gene", "coexpression"])
                .drop_duplicates("regulon")
                .set_index("regulon"),
            ],
            axis=1,
        )

        # TODO: add gene information from the database to adata.var
        # TODO: add obsm information to adata for coexpression values

        self.adata = adata

    def get_df(self) -> pd.DataFrame:
        """
        Returns the data in `self.adata` as a DataFrame in long format.

        Returns:
            pd.DataFrame: A DataFrame containing regulon information.
        """
        ad_tf = pd.concat([self.adata.to_df(), self.adata.obs], axis=1)
        ad_tf = ad_tf.melt(
            id_vars=ad_tf.columns[-self.adata.obs.shape[1] :], var_name="target_gene"
        )
        ad_tf = ad_tf[ad_tf["value"] == 1].drop(columns="value")
        return ad_tf

    def calculate_embedding(
        self, n_neighbors: int = 10, plot: bool = False, add_leiden: float = None
    ) -> None:
        """
        Calculates a UMAP embedding.

        This method first calculates the neighborhood graph of the data using the Jaccard metric.
        Then, it optionally clusters the data using the Leiden algorithm if `add_leiden` is provided,
        using the value as a clustering resolution.
        Finally, it calculates a UMAP embedding of the data.

        Args:
            n_neighbors (int): The number of neighbors to use for the knn-graph. Defaults to `10`.
            plot (bool, optional): Whether to plot the UMAP embeddings. Defaults to False.
            add_leiden (float, optional): Resolution for the Leiden clustering. If provided, Leiden clustering will be performed. Defaults to None.
        """
        sc.pp.neighbors(
            self.adata, use_rep="X", n_neighbors=n_neighbors, metric="jaccard"
        )
        if add_leiden:
            sc.tl.leiden(self.adata, resolution=add_leiden)
        sc.tl.umap(self.adata)

        if plot:
            self.plot_embedding()

    def plot_embedding(self, **kwargs) -> None:
        """
        Plots the UMAP embedding.

        This method uses the UMAP embedding in `self.adata` (calculated by `calculate_embedding`).
        Additional keyword arguments are passed to `sc.pl.umap`.

        By default, the 'leiden' and 'celltype' attributes are plotted if 'leiden' is present in `self.adata.obs`,
        otherwise only 'celltype' is plotted. Default settings include edges, alpha 0.6, and plotting the legend
        on top of the data (with an outline width of 3).

        Args:
            **kwargs: Additional keyword arguments to customize the plot. These arguments are passed to `sc.pl.umap`.
        """
        kwargs = {
            "color": ["leiden", "celltype"]
            if "leiden" in self.adata.obs
            else "celltype",
            "edges": True,
            "alpha": 0.6,
            "legend_loc": "on data",
            "legend_fontoutline": 3,
            "title": "CellRegulon regulons",
        }.update(kwargs)

        sc.pl.umap(
            self.adata,
            **kwargs,
        )

    def score_gene_set(
        self, gene_set: list, score_name: str = "score", zscore: bool = True
    ) -> None:
        """
        Scores regulons for a given gene set. The scores correspond to the fraction of genes in the gene set
        that are present in each regulon, subtracted by the fraction of genes in a randomly sampled gene set
        of the same size using genes with similar frequency.

        This method is a wrapper around the `score_genes` function from Scanpy.
        The scores are stored in `self.adata.obs` under the column specified by `score_name`.
        If `zscore` is set to True, the scores are then z-score normalized.

        Args:
            gene_set (list): A list of genes to score.
            score_name (str, optional): The name of the column in `self.adata.obs` to store the scores. Defaults to "score".
            zscore (bool, optional): Whether to z-score normalize the scores. Defaults to True.
        """
        sc.tl.score_genes(self.adata, gene_list=gene_set, score_name=score_name)
        if zscore:
            self.adata.obs[score_name] = (
                self.adata.obs[score_name] - self.adata.obs[score_name].mean()
            ) / self.adata.obs[score_name].std()

    def plot_top_scores(
        self,
        sort_by="score",
        show_cat=None,
        top=50,
        search_comp=None,
        search_data=None,
        search_tiss=None,
        search_cell=None,
        search_tf=None,
        space="free_x",
        rotation=60,
    ) -> None:
        """
        Plots the top scoring regulons across various categories.

        This method uses the 'plotnine' package to create a bar plot of the top scoring regulons.
        The regulons are sorted by the column specified by `sort_by` and the top `top` regulons are plotted.
        The plot can be filtered by `search_comp`, `search_data`, `search_tiss`, `search_cell`, and `search_tf`.
        The plot can be customized by providing additional arguments via `show_cat`, `space`, and `rotation`.

        Args:
            sort_by (str, optional): The name of the column in `self.adata.obs` to sort the regulons by. Defaults to "score".
            show_cat (list, optional): A list of categories to show in the plot. If None, defaults to ["celltype", "tissue", "transcription_factor"].
            top (int, optional): The number of top regulons to plot. Defaults to 50.
            search_comp (list, optional): A list of cell compartments to filter the regulons by. Defaults to None.
            search_data (list, optional): A list of datasets to filter the regulons by. Defaults to None.
            search_tiss (list, optional): A list of tissues to filter the regulons by. Defaults to None.
            search_cell (list, optional): A list of cell types to filter the regulons by. Defaults to None.
            search_tf (list, optional): A list of transcription factors to filter the regulons by. Defaults to None.
            space (str, optional): The type of spacing to use for the plot. Defaults to "free_x".
            rotation (int, optional): The angle to rotate the x-axis labels by. Defaults to 60.
        """
        try:
            import plotnine as p9
        except ImportError:
            raise ImportError(
                "The 'plotnine' package is required for this method. "
                "Please install it using 'pip install plotnine' "
                "or 'conda install -c conda-forge plotnine'."
            )

        if show_cat is None:
            show_cat = ["celltype", "tissue", "transcription_factor"]

        plt_df = self.adata.obs

        if search_comp:
            plt_df = plt_df.query("lineage_uni.isin(@search_comp)", engine="python")
        if search_data:
            plt_df = plt_df.query("dataset.isin(@search_data)", engine="python")
        if search_tiss:
            plt_df = plt_df.query("tissue.isin(@search_tiss)", engine="python")
        if search_cell:
            plt_df = plt_df.query("celltype.isin(@search_cell)", engine="python")
        if search_tf:
            plt_df = plt_df.query(
                "transcription_factor.isin(@search_tf)", engine="python"
            )

        df = plt_df[show_cat + [sort_by]].sort_values(sort_by, ascending=False)[:top]

        if space == "free_x":
            space = dict(x=[df[c].unique().size for c in show_cat])

        df = df.melt(id_vars=sort_by)
        df = df.groupby(["variable", "value"])[sort_by].mean().reset_index()
        df.value = df.value.astype("category").cat.reorder_categories(
            df.sort_values(sort_by, ascending=False)["value"].unique().tolist()
        )

        p9.options.figure_size = (15, 8)

        return (
            p9.ggplot(df, p9.aes(x="value", y=sort_by))
            + p9.geom_bar(stat="identity", fill="#3C5488FF")
            + p9.facet_grid(
                (".", "variable"), scales="free_x", space=space
            )  # space only implemented for 'fixed' in plotnine
            + p9.scale_x_discrete(expand=[0, 0.5])
            + p9.theme_linedraw()
            + p9.theme(
                axis_text_x=p9.element_text(angle=rotation, hjust=1),
                strip_background=p9.element_rect(fill="white", colour="black", size=1),
                panel_border=p9.element_rect(size=1),
            )
            + p9.xlab("")
            + p9.ylab("relevance score")
        )

    def perturbation_direction(self, gene_set: list):
        pass
