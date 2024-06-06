import warnings
from os import PathLike
from typing import Union, List, Tuple, Optional
from pathlib import Path
import re
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import scanpy as sc
import matplotlib.pyplot as plt

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

    def __init__(
        self, adata: Optional[Union[sc.AnnData, str]] = None, check: bool = True
    ) -> None:
        self._cell_type_col = "celltype"
        self._tissue_col = "tissue"
        self._transcription_factor_col = "transcription_factor"

        if adata is None or isinstance(adata, sc.AnnData):
            self.adata: sc.AnnData = adata
        else:
            self.adata: sc.AnnData = sc.read_h5ad(adata)

        if self.adata is not None and check:
            self._check_columns()

    def __repr__(self) -> str:
        n_regulons, n_genes = self.adata.shape
        # n_tfs = self.adata.obs["transcription_factor"].nunique()
        # n_tissues = self.adata.obs["tissue"].nunique()
        n_celltypes = self.adata.obs[self.cell_type_col].nunique()
        return f"RegulonAtlas object with {n_regulons} regulons, {n_celltypes} cell types and {n_genes} target genes."

    @property
    def cell_type_col(self):
        return self._cell_type_col

    @cell_type_col.setter
    def cell_type_col(self, value):
        if value not in self.adata.obs.columns:
            raise ValueError(f"Column '{value}' not found in `self.adata.obs`.")
        self._cell_type_col = value

    @property
    def tissue_col(self):
        return self._tissue_col

    @tissue_col.setter
    def tissue_col(self, value):
        if value not in self.adata.obs.columns:
            raise ValueError(f"Column '{value}' not found in `self.adata.obs`.")
        self._tissue_col = value

    @property
    def transcription_factor_col(self):
        return self._transcription_factor_col

    @transcription_factor_col.setter
    def transcription_factor_col(self, value):
        if value not in self.adata.obs.columns:
            raise ValueError(f"Column '{value}' not found in `self.adata.obs`.")
        self._transcription_factor_col = value

    def _check_columns(self) -> None:
        """
        Checks if the required columns are present in `self.adata.obs`.

        Raises:
            ValueError: If any of the required columns are missing.
        """
        if self.cell_type_col not in self.adata.obs:
            warnings.warn(
                f"Column '{self.cell_type_col}' not found in `self.adata.obs`. "
                f"Available columns: {self.adata.obs.columns.tolist()}, "
                f"set `self.cell_type_col` to the correct column name."
            )
        if self.tissue_col not in self.adata.obs:
            warnings.warn(
                f"Column '{self.tissue_col}' not found in `self.adata.obs`. "
                f"Available columns: {self.adata.obs.columns.tolist()}, "
                f"set `self.tissue_col` to the correct column name."
            )
        if self.transcription_factor_col not in self.adata.obs:
            warnings.warn(
                f"Column '{self.transcription_factor_col}' not found in `self.adata.obs`. "
                f"Available columns: {self.adata.obs.columns.tolist()}, "
                f"set `self.transcription_factor_col` to the correct column name."
            )

    def save(self, filename: PathLike) -> None:
        """
        Saves the data in `self.adata` to an .h5ad file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        self.adata.uns["crdb"] = {
            "cell_type_col": self.cell_type_col,
            "tissue_col": self.tissue_col,
            "transcription_factor_col": self.transcription_factor_col,
        }
        self.adata.write(filename)

    @classmethod
    def load_from_file(cls, filename: Union[str, Path]) -> "RegulonAtlas":
        """
        Loads data from an .h5ad file into a new `RegulonAtlas` object.

        Args:
            filename (str, Path): The name of the file to load the data from.

        Returns:
            cellregulondb.RegulonAtlas: A new `RegulonAtlas` object containing the loaded data.
        """
        ra = cls(sc.read_h5ad(filename), check=False)
        ra.cell_type_col = ra.adata.uns["crdb"]["cell_type_col"]
        ra.tissue_col = ra.adata.uns["crdb"]["tissue_col"]
        ra.transcription_factor_col = ra.adata.uns["crdb"]["transcription_factor_col"]
        ra._check_columns()

        return ra

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

        # append number to duplicate regulon names
        if df[["regulon", "target_gene"]].duplicated().any():
            warnings.warn(
                "Duplicate regulon-target pairs detected. Dropping duplicates.",
                category=Warning,
            )
            df = df.drop_duplicates(subset=["regulon", "target_gene"])

        # # Create an anndata object from the DataFrame
        # val_df = pd.DataFrame(
        #     [1] * df.shape[0],
        #     index=pd.MultiIndex.from_frame(df[["regulon", "target_gene"]]),
        # ).unstack(fill_value=0)
        # adata = sc.AnnData(
        #     X=sp.sparse.csr_matrix(val_df),
        #     obs=pd.DataFrame(index=val_df.index.get_level_values("regulon")),
        #     var=pd.DataFrame(index=val_df.columns.get_level_values("target_gene")),
        # )

        # Create an anndata object from the DataFrame (more efficient)
        df = df.astype({"regulon": "category", "target_gene": "category"})
        row_idx, col_idx = (
            df["regulon"].cat.codes.values,
            df["target_gene"].cat.codes.values,
        )
        adata = sc.AnnData(
            X=sp.sparse.csr_matrix(([1] * len(row_idx), (row_idx, col_idx))),
            obs=pd.DataFrame(index=df["regulon"].cat.categories),
            var=pd.DataFrame(index=df["target_gene"].cat.categories),
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
        adata.obs["n_genes"] = adata.X.sum(axis=1)

        # TODO: add gene information from the database to adata.var
        # TODO: add obsm information to adata for coexpression values

        self.adata = adata
        self._check_columns()

    def subset(
        self,
        regulons: Optional[Union[str, List[str], List[bool], pd.DataFrame]] = None,
        target_genes: Optional[Union[str, List[str], List[bool]]] = None,
        shrink: bool = True,
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
            target_genes (str, list, optional): The variable column(s) to subset by. If all target genes should
                be retained regardless of whether they are present in any regulon, also set `shrink=False`. Defaults to None.
            shrink (bool, optional): Whether to remove target genes that are not present in any regulon. Defaults to True.
            copy (bool, optional): Whether to return a copy of the subsetted data. Defaults to True.

        Returns:
            cellregulondb.RegulonAtlas: A new RegulonAtlas object containing the subsetted data.
        """
        adata = self.adata
        if regulons is not None:
            if isinstance(regulons, str):
                regulons = [regulons]
            elif isinstance(regulons, pd.DataFrame):
                regulons = regulons.index.tolist()
            adata = adata[regulons, :]
        if target_genes is not None:
            if isinstance(target_genes, str):
                target_genes = [target_genes]
            adata = adata[:, target_genes]
        if shrink:
            adata = adata[:, adata.X.sum(axis=0) > 0]
        if copy:
            adata = adata.copy()

        ra = self.__class__(adata, check=False)
        ra.cell_type_col = self.cell_type_col
        ra.tissue_col = self.tissue_col
        ra.transcription_factor_col = self.transcription_factor_col

        return ra

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
        self,
        regulons: Optional[Union[str, List[str], List[bool], pd.DataFrame]] = None,
        min_regulon: int = 1,
        top: int = None,
        return_counts: bool = False,
    ) -> Union[list, dict]:
        """
        Returns the target genes in the regulon data.

        Computes a list of target genes that are present in at least `min_regulon` regulons.
        If `top` is provided, it returns the top `top` target genes by frequency.

        Args:
            regulons (list, optional): A list of regulons to consider. If None, all regulons are considered. Defaults to None.
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            top (int, optional): The number of top target genes to return. Defaults to None.
            return_counts (bool, optional): Whether to return the target genes as a dictionary with their frequency instead. Defaults to False.

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
        freq = freq[freq >= min_regulon]

        # select top genes
        if top:
            target_genes = target_genes[:top]
            freq = freq[:top]

        if return_counts:
            return dict(zip(target_genes, freq))
        else:
            return target_genes

    def get_target_genes_by(
        self,
        by: Union[str, List[str]] = "leiden",
        subset: str = None,
        min_regulon: int = 1,
        top: int = None,
        return_counts: bool = False,
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
            return_counts (bool, optional): Whether to return the target genes as a dictionary with their frequency instead. Defaults to False.

        Returns:
            dict: A dictionary where the keys are the categories in `by` and the values are the target genes.
        """
        use_df = self.adata.obs.copy()
        use_df["regulon"] = self.adata.obs_names.tolist()

        if subset:
            use_df = use_df.query(subset, engine="python")

        return (
            use_df.groupby(by, observed=False)
            .apply(
                lambda df: self.get_target_genes(
                    regulons=df["regulon"].unique().tolist(),
                    min_regulon=min_regulon,
                    top=top,
                    return_counts=return_counts,
                ),
            )
            .to_dict()
        )

    def get_tf_dict(
        self, min_regulon: int = 1, subset: str = "regulation == '+'", **kwargs
    ) -> dict:
        """
        Returns a dictionary of transcription factors and their target genes.

        This method is a convenience wrapper around the `get_target_genes_by` method,
        grouping the regulon data by the `self.transcription_factor_col` category. The resulting
        dictionary has transcription factors as keys and their corresponding target genes as values.
        By default, it filters the data to only activating regulons, using `subset="regulation == '+'"`.

        Args:
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            subset (str, optional): A query string to filter the data before grouping
                (using `pandas.query(filter, engine='python')` on `self.adata.obs`).
                If `None`, no filtering is applied. Defaults to "regulation == '+'"
                (returns activator regulons).
            **kwargs: Additional keyword arguments to pass to the `get_target_genes_by` method.

        Returns:
            dict: A dictionary where the keys are transcription factors and the values are lists of target genes.
        """
        kwargs.update(min_regulon=min_regulon, subset=subset)
        return self.get_target_genes_by(by=self.transcription_factor_col, **kwargs)

    def to_networkx(
        self,
        regulons: Optional[Union[str, List[str], List[bool], pd.DataFrame]] = None,
        target_genes: list = None,
        subset: str = "regulation == '+'",
        min_regulon: int = 1,
        min_degree_targets: int = None,
        **kwargs,
    ) -> nx.DiGraph:
        """
        Converts the regulon data to a NetworkX graph.

        This method converts the regulon data in `self.adata` to a directed NetworkX graph.
        The nodes in the graph are the regulons and target genes, and the edges are the regulatory relationships between them.
        The graph is directed from the transcription factors to the target genes.

        Args:
            regulons (list, optional): A list of regulons to include in the graph. If None, includes all regulons. Defaults to None.
            target_genes (list, optional): A list of target genes to include in the graph. If None, includes all target genes. Defaults to None.
            subset (str, optional): A query string to filter the data before creating the graph (using `pandas.query(filter, engine='python')` on `self.adata.obs`). Defaults to "regulation == '+'" (returns activator regulons).
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            min_degree_targets (int, optional): The minimum in-degree of target genes to include in the graph. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the `get_tf_dict` method.

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the regulon data.
        """
        if regulons is None:
            regulons = self.adata.obs_names.tolist()
        if target_genes is None:
            target_genes = self.adata.var_names.tolist()
        kwargs.update(min_regulon=min_regulon, subset=subset)

        # create a directed NetworkX graph
        ra_nx = nx.from_dict_of_lists(
            self.subset(
                regulons=regulons, target_genes=target_genes, copy=False
            ).get_tf_dict(**kwargs),
            create_using=nx.DiGraph,
        )

        tfs = set(self.adata.obs[self.transcription_factor_col].values)
        for node in ra_nx.nodes:
            ra_nx.nodes[node]["transcription_factor"] = node in tfs

        if min_degree_targets is not None:
            remove = [
                node
                for node, degree in ra_nx.in_degree()
                if degree < min_degree_targets
                and not ra_nx.nodes[node]["transcription_factor"]
            ]
            ra_nx.remove_nodes_from(remove)

        return ra_nx

    def get_tables(
        self,
        regulons: Optional[Union[str, List[str], List[bool], pd.DataFrame]] = None,
        target_genes: list = None,
        by: Union[str, List[str]] = None,
        node_columns: List[str] = None,
        feedbacks: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns the link and node tables for the regulon data.
        (These could be used for export to a network visualization tool, such as Cytoscape.)

        Returns three tables in total, a link table, a regulon node table, and a target gene node table.
        IDs of source and target nodes will be in the "source" and "target" columns of the returned tables.
        The link table contains both the source and target IDs.

        A subset of the regulon atlas can be specified by providing lists of regulons and target genes in `regulons` and `target_genes`.

        The source nodes of the tables can be set as combinations of columns in `self.adata.obs` by providing a list of column names in `by`.
        For example, setting `by=["transcription_factor", "cell_type"]` will group the regulons by the unique combinations of transcription factors and cell types.
        The IDs of the source nodes would then be "<transcription factor> - <cell type>".
        If `by` is a single column name, such as "transcription_factor", the source nodes will be the unique values in that column.

        Other columns from `self.adata.obs` will be grouped in lists and become columns of the link table.
        The `node_columns` argument can be used to make them columns of the node tables instead.

        If feedbacks is set to True, targets in the link table that are transcription factors will be replaced by regulons (source nodes), if the `by` columns match.

        Args:
            regulons (list, optional): A list of regulons to include in the tables. If None, includes all regulons. Defaults to None.
            target_genes (list, optional): A list of target genes to include in the tables. If None, includes all target genes. Defaults to None.
            by (str, list, optional): The column(s) to group the regulons by. If None, will be `self.transcription_factor_col`. Defaults to None.
            node_columns (list, optional): A list of columns from `self.adata.obs` to include in the node tables. If None, will be []. Defaults to None.
            feedbacks (bool, optional): Whether to replace target genes that are transcription factors with regulons. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the link table, regulon node table, and target gene node table.
        """
        if regulons is None:
            regulons = self.adata.obs_names.tolist()
        if target_genes is None:
            target_genes = self.adata.var_names.tolist()
        if by is None:
            by = self.transcription_factor_col
        if node_columns is None:
            node_columns = []
        if isinstance(by, str):
            by = [by]

        # get regulon-target gene pairs
        regulon_df = self.subset(
            regulons=regulons, target_genes=target_genes, copy=False
        ).get_df()

        # create link table
        link_df = regulon_df.groupby(by + ["target_gene"]).apply(
            lambda df: df.T.apply(list, axis=1)
        )
        link_df = link_df.drop(columns=by + ["target_gene"] + node_columns)
        link_df = link_df.reset_index(level="target_gene").rename(
            columns={"target_gene": "target"}
        )

        def join_proc(x):
            x = [f"({s})" if re.match(r"[+-]", s) else s for s in x]
            return " - ".join(x)

        if len(by) > 1:
            link_df.index = link_df.index.map(join_proc)  # join multi-index
        link_df = link_df.rename_axis(index="source").reset_index()

        # create node table (regulons)
        reg_df = regulon_df.groupby(by).apply(lambda df: df.T.apply(list, axis=1))
        reg_df = reg_df[node_columns + by]
        if len(by) > 1:
            reg_df.index = reg_df.index.map(join_proc)  # join multi-index
        reg_df = reg_df.rename_axis(index="source").reset_index()

        # create node table (target genes)
        tg_df = self.adata.var.rename_axis(index="target").reset_index()

        # replace target genes with regulons if feedbacks is True
        # (do replacement only if a regulon with the respective `by` columns exists)
        if feedbacks:
            if reg_df["source"].str.contains(r"\([+-]\)", regex=True).any():
                raise ValueError(
                    "setting parameter 'feedbacks' to True with regulons that "
                    "contain the sign ('+' or '-') is not supported"
                )
            src_parts = link_df["source"].str.split(" - ", expand=True)
            src_parts[0] = link_df["target"]
            regs = set(reg_df["source"])
            test_reg = src_parts.apply(" - ".join, axis=1)
            link_df["target"] = [
                reg if reg in regs else trg
                for reg, trg in zip(test_reg, link_df["target"])
            ]

        return link_df, reg_df, tg_df

    def find_cell_types(
        self, cell_types: list, cell_type_col: str = None, tolist: bool = False
    ) -> Union[dict, list]:
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
            cell_type_col (str, optional): The name of the column in `self.adata.obs` to search in. Defaults to `self.cell_type_col`.
            tolist (bool, optional): Whether to return the matching cell types as a list instead of a dictionary. Defaults to False.

        Returns:
            Union[dict, list]: A dictionary where the keys are the input cell types and the values are lists of
                matching cell types in the data or a list of only the matching cell types concatenated.
        """
        if cell_type_col is None:
            cell_type_col = self.cell_type_col

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

        if tolist:
            return [ct for cts in cell_type_matches.values() for ct in cts]
        else:
            return cell_type_matches

    def find_regulons(
        self,
        target_genes: Union[list, str] = None,
        target_genes_mode: str = "any",
        cell_types: list = None,
        transcription_factors: list = None,
        tissues: list = None,
        subset: str = None,
        tolist: bool = False,
    ) -> Union[pd.DataFrame, list]:
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
            cell_types (list, optional): A list of cell types to filter the regulons by. Defaults to None.
            transcription_factors (list, optional): A list of transcription factors to filter the regulons by. Defaults to None.
            tissues (list, optional): A list of tissues to filter the regulons by. Defaults to None.
            subset (str, optional): A query string to filter the regulons (using `pandas.query(subset, engine='python')` on `self.adata.obs`). Defaults to no filtering if None.
            tolist (bool, optional): Whether to return the regulon names as a list instead of a `pd.DataFrame`. Defaults to False.

        Returns:
            Union[pd.DataFrame, list]: A DataFrame containing the regulons that match the specified criteria with meta-information or a list of the regulon names.
        """
        # set arguments
        if target_genes_mode not in ["any", "all"]:
            raise ValueError(
                f"target_genes_mode must be one of ['any', 'all'], got {target_genes_mode}"
            )
        if isinstance(target_genes, str):
            target_genes = [target_genes]

        # filter by target genes
        if target_genes is None:
            obs_df = self.adata.obs.copy()
        else:
            mask = (
                self.adata[:, target_genes].X.sum(axis=1) > 0
                if target_genes_mode == "any"
                else self.adata[:, target_genes].X.sum(axis=1) == len(target_genes)
            )
            obs_df = self.adata[mask].obs.copy()

        # filter by cell types, transcription factors, and tissues
        if cell_types:
            obs_df = obs_df[obs_df[self.cell_type_col].isin(cell_types)]
        if transcription_factors:
            obs_df = obs_df[
                obs_df[self.transcription_factor_col].isin(transcription_factors)
            ]
        if tissues:
            obs_df = obs_df[obs_df[self.tissue_col].isin(tissues)]

        # filter by additional metadata
        if subset:
            obs_df = obs_df.query(subset, engine="python")

        if tolist:
            return obs_df.index.tolist()
        else:
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
        # if "X" is sparse, temporarily save dense array in obsm for metric jaccard
        if sp.sparse.issparse(self.adata.X):
            self.adata.obsm["_X_dense"] = self.adata.X.A
            sc.pp.neighbors(
                self.adata,
                use_rep="_X_dense",
                n_neighbors=n_neighbors,
                metric="jaccard",
            )
            sc.tl.umap(self.adata)  # umap uses `use_rep` from neighbors
            del self.adata.obsm["_X_dense"]  # delete to save ram
        else:
            sc.pp.neighbors(
                self.adata, use_rep="X", n_neighbors=n_neighbors, metric="jaccard"
            )
            sc.tl.umap(self.adata)

        if add_leiden:
            sc.tl.leiden(self.adata, resolution=add_leiden)

        if plot:
            self.plot_embedding()

    def plot_embedding(self, **kwargs) -> None:
        """
        Plots the UMAP embedding.

        This method uses the UMAP embedding in `self.adata` (calculated by `calculate_embedding`).
        Additional keyword arguments are passed to `sc.pl.umap`.

        By default, the 'leiden' and `self.cell_type_col` attributes are plotted if 'leiden' is present in `self.adata.obs`,
        otherwise only `self.cell_type_col` is plotted. Default settings include edges, alpha 0.6, and plotting the legend
        on top of the data (with an outline width of 3).

        Args:
            **kwargs: Additional keyword arguments to customize the plot. These arguments are passed to `sc.pl.umap`.
        """
        umap_kwargs = {
            "color": ["leiden", self.cell_type_col]
            if "leiden" in self.adata.obs
            else self.cell_type_col,
            "edges": True,
            "alpha": 0.6,
            "legend_loc": "on data",
            "legend_fontoutline": 3,
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
            show_cat (list, optional): A list of categories to show in the plot. If None, defaults
                to [`self.cell_type_col`, `self.tissue_col`, `self.transcription_factor_col`].
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
            show_cat = [
                self.cell_type_col,
                self.tissue_col,
                self.transcription_factor_col,
            ]

        plt_df = self.adata.obs

        if search_comp:
            plt_df = plt_df.query("lineage_uni.isin(@search_comp)", engine="python")
        if search_data:
            plt_df = plt_df.query("dataset.isin(@search_data)", engine="python")
        if search_tiss:
            plt_df = plt_df.query(
                f"{self.tissue_col}.isin(@search_tiss)", engine="python"
            )
        if search_cell:
            plt_df = plt_df.query(
                f"{self.cell_type_col}.isin(@search_cell)", engine="python"
            )
        if search_tf:
            plt_df = plt_df.query(
                f"{self.transcription_factor_col}.isin(@search_tf)", engine="python"
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

    def plot_target_gene_heatmap(
        self,
        by: Union[str, List[str]] = None,
        subset: str = None,
        min_regulon: int = 1,
        top: int = None,
        **kwargs,
    ) -> Tuple[plt.Axes, plt.Figure]:
        """
        Plots a heatmap of target genes for each category in `by`.

        This method groups the regulon data by the categories in `by` and computes the target genes for each group.
        The target genes are computed using the `get_target_genes_by` method.
        The results are plotted as a heatmap using the `seaborn` package.

        Args:
            by (str, list, optional): A list of categories (columns from `self.adata.obs`) to group the regulons by.
                If None, uses `self.transcription_factor_col`. Defaults to None.
            subset (str, optional): A query string to filter the data before grouping
                (using `pandas.query(filter, engine='python')` on `self.adata.obs`). Defaults to no filtering if None.
            min_regulon (int, optional): The minimum number of regulons a target gene should be present in. Defaults to 1.
            top (int, optional): The number of top most frequent target genes to return. Defaults to None.
            **kwargs: Additional keyword arguments to customize the heatmap. These arguments are passed to `seaborn.heatmap`.

        Returns:
            Tuple[plt.Axes, plt.Figure]: A tuple containing the heatmap axes and figure.
        """
        if by is None:
            by = self.transcription_factor_col

        sns_kwargs = {
            "cmap": "viridis",
            "cbar_kws": {"label": "number of target genes"},
        }
        sns_kwargs.update(kwargs)

        target_genes = self.get_target_genes_by(
            by=by, subset=subset, min_regulon=min_regulon, top=top, return_counts=True
        )

        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "The 'seaborn' and 'matplotlib' packages are required for this method. "
                "Please install them using 'pip install seaborn matplotlib' "
                "or 'conda install seaborn matplotlib'."
            )

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(
            pd.DataFrame(target_genes).T.fillna(0),
            ax=ax,
            **sns_kwargs,
        )
        ax.set_title(f"Target genes by ({', '.join(by)})")
        ax.set_xlabel("")
        ax.set_ylabel("")

        return ax, fig

    def perturbation_direction(self, gene_set: list):
        pass
