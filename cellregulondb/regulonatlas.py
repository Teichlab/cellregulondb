from os import PathLike
from typing import Union, List, Optional
import re
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc

import cellregulondb


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

    def __init__(self, adata: Optional[Union[sc.AnnData, str]] = None) -> None:
        self.adata: sc.AnnData = (
            adata if isinstance(adata, Union[sc.AnnData, None]) else sc.read_h5ad(adata)
        )

    def __repr__(self) -> str:
        n_regulons, n_genes = self.adata.shape
        # n_tfs = self.adata.obs["transcription_factor"].nunique()
        # n_tissues = self.adata.obs["tissue"].nunique()
        n_celltypes = self.adata.obs["celltype"].nunique()
        return f"RegulonAtlas object with {n_regulons} regulons, {n_celltypes} cell types and {n_genes} target genes."

    def save(self, filename: PathLike) -> None:
        """
        Saves the data in `self.adata` to an .h5ad file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        self.adata.write(filename)

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

    def subset(
        self,
        regulons: Optional[Union[str, List[str], List[bool]]] = None,
        target_genes: Optional[Union[str, List[str], List[bool]]] = None,
        copy: bool = True,
    ) -> "RegulonAtlas":
        """
        Subsets the data in `self.adata` based on the specified observations and variables.

        This method subsets the data in `self.adata` based on the specified observations and variables.
        The observations and variables to subset by can be specified as a string or a list of strings.
        If a string is provided, the data is subset by the unique values in the specified column.
        If a list is provided, the data is subset by the unique values in all the specified columns.

        Args:
            regulons (str, list, optional): The observation column(s) to subset by. Defaults to None.
            target_genes (str, list, optional): The variable column(s) to subset by. Defaults to None.
            copy (bool, optional): Whether to return a copy of the subsetted data. Defaults to True.

        Returns:
            cellregulondb.RegulonAtlas: A new RegulonAtlas object containing the subsetted data.
        """
        adata = self.adata
        if regulons:
            if isinstance(regulons, str):
                regulons = [regulons]
            adata = adata[regulons, :]
        if target_genes:
            if isinstance(target_genes, str):
                target_genes = [target_genes]
            adata = adata[:, target_genes]
        if copy:
            adata = adata.copy()

        return self.__class__(adata)

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

    def get_target_genes(
        self, regulons: list = None, min_regulon: int = 1, top: int = None
    ) -> list:
        """
        Returns the target genes in the regulon data.

        Computes a list of target genes that are present in at least `min_regulon` regulons.
        If `top` is provided, it returns the top `top` target genes by frequency.

        Args:
            regulons (list, optional): A list of regulons to consider. If None, all regulons are considered. Defaults to None.
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            top (int, optional): The number of top target genes to return. Defaults to None.

        Returns:
            list: A list of target genes.
        """
        if regulons is None:
            regulons = self.adata.obs_names.tolist()

        # get target genes and their frequency
        genes = self.adata[regulons, :].var.index
        freq = np.array(self.adata[regulons, :].X.sum(axis=0)).flatten()

        # sort genes by frequency
        order = freq.argsort()[::-1]
        genes, freq = genes[order], freq[order]

        # filter genes by frequency
        target_genes = genes[freq >= min_regulon].tolist()

        # select top genes
        if top:
            target_genes = target_genes[:top]

        return target_genes

    def get_target_genes_by(
        self,
        by: Union[str, List[str]] = "leiden",
        subset: str = None,
        min_regulon: int = 1,
        top: int = None,
    ) -> dict:
        """
        Returns the target genes for each category in `by`.

        This method groups the regulon data by the categories in `by` and computes the target genes for each group.
        The target genes are computed using the `get_target_genes` method with the `min_regulon` and `top` parameters.
        The results are returned as a dictionary where the keys are the categories in `by` and the values are the target genes.

        Args:
            by (list, optional): A list of categories (columns from `self.adata.obs`) to group the regulons by. Defaults to "leiden".
            subset (str, optional): A query string to filter the data before grouping
                (using `pandas.query(..., engine='python')` on `self.adata.obs`). Defaults to no filtering if None.
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            top (int, optional): The number of top target genes to return. Defaults to None.

        Returns:
            dict: A dictionary where the keys are the categories in `by` and the values are the target genes.
        """
        use_df = self.adata.obs.copy()
        use_df["regulon"] = self.adata.obs_names.tolist()

        if subset:
            use_df = use_df.query(subset, engine="python")

        # TODO: option to return counts per gene instead of list
        return (
            use_df.groupby(by, observed=False)
            .apply(
                lambda df: self.get_target_genes(
                    regulons=df["regulon"].unique().tolist(),
                    min_regulon=min_regulon,
                    top=top,
                ),
            )
            .to_dict()
        )

    def get_tf_dict(self, subset: str = "regulation == '+'", **kwargs) -> dict:
        """
        Returns a dictionary of transcription factors and their target genes.

        This method is a convenience wrapper around the `get_target_genes_by` method,
        grouping the regulon data by the 'transcription_factor' category. The resulting
        dictionary has transcription factors as keys and their corresponding target genes as values.
        By default, it filters the data to only activating regulons, using `subset="regulation == '+'"`.

        Args:
            subset (str, optional): A query string to filter the data before grouping
                (using `pandas.query(filter, engine='python')` on `self.adata.obs`).
                If `None`, no filtering is applied. Defaults to "regulation == '+'"
                (returns activator regulons).
            **kwargs: Additional keyword arguments to pass to the `get_target_genes_by` method.

        Returns:
            dict: A dictionary where the keys are transcription factors and the values are lists of target genes.
        """
        return self.get_target_genes_by(
            by="transcription_factor", subset=subset, **kwargs
        )

    def find_cell_types(
        self, cell_types: list, cell_type_col: str = "celltype"
    ) -> dict:
        """
        Finds string matches for a list of cell types in the regulon data with an adaptive strategy.

        This method takes a list of cell types and searches for name matches with cell type in the regulon data.
        Cell type names are broken down into alphanumeric fragments and matched in a case-insensitive manner.
        If there are more than 20 hits for a query or the query is less than 3 characters long, searches for word matches instead.
        If there are no hits, searches for partial matches.

        The search is performed in the column specified by `cell_type_col`.
        The results are returned as a dictionary where the keys are the input cell types and the values are lists of matching cell types in the data.

        Args:
            cell_types (list): A list of cell types to search for.
            cell_type_col (str, optional): The name of the column in `self.adata.obs` to search in. Defaults to "celltype".

        Returns:
            dict: A dictionary where the keys are the input cell types and the values are lists of matching cell types in the data.
        """
        candidate_cts = self.adata.obs[cell_type_col].unique().tolist()
        cell_type_matches = {}

        for ct in cell_types:
            ct_frags = [x for x in re.split("[^A-Za-z0-9]+", ct.lower()) if x != "cell"]
            hits = [c for c in candidate_cts if all(f in c.lower() for f in ct_frags)]

            if len(hits) > 20 or len(ct) < 3:
                # unspecific hits
                # look for word match
                print(  # TODO: replace with logging
                    f"too many hits ({len(hits)}) for query '{ct}', looking for word matches instead"
                )
                hits = [
                    c
                    for c in candidate_cts
                    if all(
                        f
                        in [
                            x
                            for x in re.split("[^A-Za-z0-9]+", c.lower())
                            if x != "cell"
                        ]
                        for f in ct_frags
                    )
                ]
            elif len(hits) == 0:
                # no hits
                # look for partial matches
                print(  # TODO: replace with logging
                    f"no hits for query {ct}, looking for partial matches {ct_frags} instead"
                )
                hits = [
                    c for c in candidate_cts if any(f in c.lower() for f in ct_frags)
                ]

            cell_type_matches[ct] = hits

        return cell_type_matches

    def find_regulons(
        self,
        target_genes: Union[list, str] = None,
        target_genes_mode: str = "any",
        subset: str = None,
    ) -> pd.DataFrame:
        """
        Finds regulons based on which target genes they regulate or other meta information.

        This method takes a list of target genes and returns a DataFrame containing regulons that are associated with these genes.
        If `target_genes_mode` is set to 'any', it returns regulons that are associated with any of the target genes.
        If `target_genes_mode` is set to 'all', it returns regulons that are associated with all the target genes.
        If `target_genes` is None, it returns all regulons in the data.
        The method can also filter the regulons based on a query string `subset` that is applied to `self.adata.obs`.

        Args:
            target_genes (list, str, optional): A list of target genes to search for. If None, returns all regulons. Defaults to None.
            target_genes_mode (str, optional): The mode to use when filtering regulons based on target genes. Can be 'any' or 'all'. Defaults to 'any'.
            subset (str, optional): A query string to filter the regulons (using `pandas.query(subset, engine='python')` on `self.adata.obs`). Defaults to no filtering if None.

        Returns:
            pd.DataFrame: A DataFrame containing regulons and meta-information associated with the target genes.
        """
        if target_genes_mode not in ["any", "all"]:
            raise ValueError(
                f"target_genes_mode must be one of ['any', 'all'], got {target_genes_mode}"
            )
        if isinstance(target_genes, str):
            target_genes = [target_genes]

        if target_genes is None:
            obs_df = self.adata.obs.copy()
        else:
            mask = (
                self.adata[:, target_genes].X.sum(axis=1) > 0
                if target_genes_mode == "any"
                else self.adata[:, target_genes].X.sum(axis=1) == len(target_genes)
            )
            obs_df = self.adata[mask].obs.copy()

        if subset:
            obs_df = obs_df.query(subset, engine="python")

        return obs_df

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
        umap_kwargs = {
            "color": ["leiden", "celltype"]
            if "leiden" in self.adata.obs
            else "celltype",
            "edges": True,
            "alpha": 0.6,
            "legend_loc": "on data",
            "legend_fontoutline": 3,
            "title": "CellRegulon regulons",
        }
        umap_kwargs.update(kwargs)

        sc.pl.umap(
            self.adata,
            **umap_kwargs,
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
        ad_q = self.adata.copy()
        sc.tl.score_genes(ad_q, gene_list=gene_set, score_name=score_name)
        self.adata.obs[score_name] = ad_q.obs[score_name].tolist()

        if zscore:
            n_score = (
                self.adata.obs[score_name] - self.adata.obs[score_name].mean()
            ) / self.adata.obs[score_name].std()
            self.adata.obs[score_name] = n_score.tolist()

    def plot_top_scores(
        self,
        sort_by: str = "score",
        show_cat: list = None,
        top: int = 50,
        search_comp: list = None,
        search_data: list = None,
        search_tiss: list = None,
        search_cell: list = None,
        search_tf: list = None,
        space: str = "free_x",
        rotation: int = 60,
        figsize: tuple = (15, 4),
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
            figsize (tuple, optional): The size of the figure. Defaults to (15, 8).
        """
        try:
            import plotnine as p9

        except ImportError:
            raise ImportError(
                "The 'plotnine' package is required for this method. "
                "Please install it using 'pip install plotnine' "
                "or 'conda install -c conda-forge plotnine'."
            )

        if figsize:
            p9.options.figure_size = figsize

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
                strip_text=p9.element_text(color="white"),
                strip_background=p9.element_rect(fill="black", colour="black", size=1),
                panel_border=p9.element_rect(size=1),
            )
            + p9.xlab("")
            + p9.ylab("relevance score")
        )

    def perturbation_direction(self, gene_set: list):
        pass
