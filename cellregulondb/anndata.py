from functools import partial
from typing import Union, List, Dict, Optional, Literal
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import multiprocessing as mp
import matplotlib.pyplot as plt

import cellregulondb

_PROCESS_OVERWRITE = Literal["neighbors", "leiden", "paga"]


def process(
    adata: sc.AnnData,
    leiden_resolution: float = 1.0,
    groups: str = "leiden",
    overwrite: List[_PROCESS_OVERWRITE] = None,
) -> None:
    """
    Compute relevant information for the AnnData object.

    This includes computing the neighbors, leiden clustering, and PAGA.

    Args:
        adata (sc.AnnData): AnnData object.
        leiden_resolution (float, optional): Resolution for the leiden clustering. Defaults to 1.0.
        groups (str, optional): Column in `adata.obs` that contains the group assignment. Defaults to "leiden".
        overwrite (List[_PROCESS_OVERWRITE], optional): List of processes to overwrite. If None, does not overwrite existing data. Defaults to None.
    """
    if overwrite is None:
        overwrite = []

    if "connectivities" not in adata.obsp or "neighbors" in overwrite:
        sc.pp.neighbors(adata)

    if "leiden" not in adata.obs or "leiden" in overwrite:
        sc.tl.leiden(adata, resolution=leiden_resolution)

    if "paga" not in adata.uns or "paga" in overwrite:
        sc.tl.paga(adata, groups=groups)


def get_transition_genes(
    ad: sc.AnnData,
    group: str = "leiden",
    adj: Union[np.ndarray, sp.sparse.csr_matrix, pd.DataFrame] = None,
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Extracts transition vectors between groups, i.e. vectors connecting the group centroids.

    This method computes the transition vectors between groups by computing the difference between the centroids of the target group and the source group.
    The transition vectors are stored in a DataFrame, where the rows correspond to the groups and the columns correspond to the genes.
    The adjacency matrix needs to represent connections between groups, e.g. the connectivities from PAGA run on `group`.

    Args:
        ad (sc.AnnData): AnnData object.
        group (str): Column in `ad.obs` that contains the group assignment. Defaults to "leiden".
        adj (Union[np.ndarray, sp.sparse.csr_matrix, pd.DataFrame]): Adjacency matrix. If None, uses `ad.uns['paga']['connectivities']`. Defaults to None.
        inplace (bool, optional): Whether to store the transition genes in `ad.uns`, instead of returning a `pd.DataFrame`. Defaults to True.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the transition genes, if `inplace` is False.

    Storage:
        If `inplace` is True, stores the transition genes in `ad.uns['crdb']['transition_genes']`.
    """
    if adj is None:
        adj = ad.uns["paga"]["connectivities"]
    if sp.issparse(adj):
        adj = adj.todense()

    # aggregate counts per group to get centroids
    agg_mat = ad.obs[[group]].assign(assigned=1).pivot(columns=group, values="assigned")
    agg_mat[agg_mat.isna()] = 0
    agg_mat = agg_mat.T.to_numpy()

    annot_centroids = (agg_mat @ ad.raw.X) / agg_mat.sum(axis=1).reshape(-1, 1)

    # extract indices from adjacency matrix
    adj_mat = pd.DataFrame(adj)
    adj_mat[adj_mat == 0] = np.nan
    stack_mat = adj_mat.stack()
    source_idx = stack_mat.index.get_level_values(0).tolist()
    target_idx = stack_mat.index.get_level_values(1).tolist()

    # get vector
    conn_vec = annot_centroids[target_idx, :] - annot_centroids[source_idx, :]

    trans_mat = pd.DataFrame(conn_vec, index=stack_mat.index, columns=ad.raw.var_names)

    if inplace:
        if "crdb" not in ad.uns:
            ad.uns["crdb"] = dict()
        ad.uns["crdb"]["transition_genes"] = trans_mat
    else:
        return trans_mat


def get_regulon_match_parallel(
    conn_vec: Union[pd.DataFrame, sc.AnnData],
    regulon_genes: Dict[str, List[str]],
    n_chunk: int = 10,
    ncpu: int = None,
) -> dict:
    """
    Parallel version of `get_regulon_match`.

    In the parallel version, for `regulon_genes` only a dictionary is supported and there is no inplace storage option yet.

    Args:
        conn_vec (Union[pd.DataFrame, sc.AnnData]): Connection vector. If `sc.AnnData`, takes `ad.uns['crdb']['transition_genes']`.
        regulon_genes (Dict[str, List[str]]): Dictionary of regulons and their target genes.
        n_chunk (int, optional): Number of chunks. Defaults to 10.
        ncpu (int, optional): Number of CPUs. Defaults to using all CPUs.

    Returns:
        dict: Dictionary containing the regulon match.
    """

    assert isinstance(regulon_genes, dict)

    if ncpu is None:
        ncpu = cellregulondb.NCPU

    def pass_params():
        i = dict()
        for k, v in regulon_genes.items():
            i[k] = v
            if len(i) == n_chunk:
                yield [conn_vec, i]
                i = dict()
        yield [conn_vec, i]

    get_regulon_match_outplace = partial(get_regulon_match, inplace=False)

    with mp.Pool(ncpu) as pool:
        res = pool.starmap(get_regulon_match_outplace, pass_params())
    return {k: v for i in res for k, v in i.items()}


def get_regulon_match(
    connections: Union[pd.DataFrame, sc.AnnData],
    regulon_genes: Union[List[list], List[str], Dict[str, List[str]]],
    reg_names: list = None,
    keys: list = None,
    inplace: bool = True,
) -> Union[pd.DataFrame, dict, None]:
    """
    Get regulon match, i.e. the vector product between the connection vector(s) and the regulon vector.

    `connections` is used to pass the vectors connecting the group centroids, as computed by `get_transition_genes`.
    It can be a DataFrame or an AnnData object. If the latter, `ad.uns['crdb']['transition_genes']` from the AnnData object is used.

    `regulon_genes` is used to pass the regulon genes. It can be a list or a dictionary.
    If it is a list of strings, the list contains the target genes of a single regulon and `reg_names` can be used to pass a name for the regulon.
    If it is a list of lists, the regulon names for each inner list can be passed in `reg_names` and the inner lists contain the target genes.
    If it is a dictionary, the keys of the dictionary are used as regulon names and the values are lists of target genes.

    If `inplace` is True, the regulon match is stored in `ad.uns['crdb']['transition_mat']` in the AnnData object.
    In case that multiple regulons are passed, the regulon match is stored as a dictionary, where the keys are the regulon names.
    The `keys` parameter can be used to specify for which regulons the directed PAGA transitions should be stored in the AnnData object.
    Then directed arrows can be plotted with `sc.pl.paga` by passing `transitions=f"{key}_transitions"` as a parameter.

    Args:
        connections (pd.DataFrame): Connection vector. If `sc.AnnData`, takes `ad.uns['crdb']['transition_genes']`.
        regulon_genes (Union[list, dict]): List or dictionary of regulon genes.
        reg_names (list, optional): List of regulon names. If None, uses the keys of `regulon_genes`, if this is a dict. Defaults to None.
        keys (list, optional): List of keys (regulon names) for which to store directed PAGA transitions.
            Only possible, if `connections` is `sc.AnnData`. If None, uses all `reg_names`. Defaults to None.
        inplace (bool, optional): Whether to store the regulon match in `ad.uns`, instead of returning a `pd.DataFrame`. Defaults to True.

    Returns:
        Union[pd.DataFrame, dict, None]: A DataFrame, if `regulon_genes` is a dict. A dictionary, if `regulon_genes` is a list. None, if `inplace == True`.

    Storage:
        If `inplace` is True, stores the regulon match in `ad.uns['crdb']['transition_mat']`.
        If `connections` is an AnnData object and `keys` is not None, stores the directed PAGA transitions in `ad.uns['paga']` for the keys.
    """
    if isinstance(connections, sc.AnnData):
        connections = connections.uns["crdb"]["transition_genes"]

    if isinstance(regulon_genes, list):
        if not isinstance(regulon_genes[0], list):
            regulon_genes = [regulon_genes]
    elif isinstance(regulon_genes, dict):
        reg_names = list(regulon_genes.keys())
        regulon_genes = [regulon_genes[r] for r in reg_names]

    if keys is None:
        keys = reg_names
    assert set(keys) - set(reg_names) == set(), "`keys` must be subset of `reg_names`"

    # compute vector product between connections and regulon vector
    def _reg_vec(rg):
        if isinstance(rg, list):
            return [1 if gene in rg else 0 for gene in connections.columns.tolist()]
        elif isinstance(rg, dict):
            return [
                rg[gene] if gene in rg else 0 for gene in connections.columns.tolist()
            ]
        else:
            raise ValueError("need list or dict as input")

    regulon_vec = np.array([_reg_vec(rg) for rg in regulon_genes])
    vec_prod = np.dot(
        regulon_vec / np.linalg.norm(regulon_vec, axis=1).reshape(-1, 1),
        (connections.T - connections.mean(axis=1))
        / np.linalg.norm(connections, axis=1).reshape(1, -1),
    )  # product between unit vectors, adjust mean to avoid regulon biases

    # construct transition matrix / matrices
    transition_mat = [
        pd.Series(vp, index=connections.index).unstack(1).to_numpy() for vp in vec_prod
    ]
    for tm in transition_mat:
        tm[np.isnan(tm)] = 0

    if reg_names:
        transition_mat = dict(zip(reg_names, transition_mat))

    trans_mat = transition_mat[0] if len(transition_mat) == 1 else transition_mat

    if inplace:
        if isinstance(connections, sc.AnnData):
            connections.uns["crdb"]["transition_mat"] = trans_mat
            if isinstance(trans_mat, dict):
                for key in keys:
                    tmat = transition_mat[key]
                    tmat[tmat < 0] = np.nan
                    connections.uns["paga"][
                        f"{key}_transitions"
                    ] = sp.sparse.csr_matrix(np.matrix(tmat).T)
            elif isinstance(trans_mat, pd.DataFrame) and len(keys) > 0:
                assert (
                    len(keys) == 1
                ), "only one key allowed if `regulon_genes` is a list"
                trans_mat[trans_mat < 0] = np.nan
                connections.uns["paga"][
                    f"{keys[0]}_transitions"
                ] = sp.sparse.csr_matrix(np.matrix(trans_mat).T)
        else:
            raise ValueError("inplace is only available for sc.AnnData")
    else:
        return trans_mat


def get_regulon_transitions(
    ad: sc.AnnData,
    group: str,
    adj: Union[np.ndarray, sp.sparse.csr_matrix, pd.DataFrame],
    regulon_genes: Union[list, dict],
) -> Union[pd.DataFrame, dict]:
    """
    Get regulon transitions.

    Args:
        ad (sc.AnnData): AnnData object.
        group (str): Column in `ad.obs` that contains the group assignment.
        adj (Union[np.ndarray, sp.sparse.csr_matrix, pd.DataFrame]): Adjacency matrix.
        regulon_genes (Union[list, dict]): List or dictionary of regulon genes.

    Returns:
        Union[pd.DataFrame, dict]: A DataFrame or dictionary containing the regulon transitions.
    """
    conn_vec = get_transition_genes(ad, group, adj)

    transition_mat = get_regulon_match(conn_vec, regulon_genes)

    return transition_mat


def plot_transitions(ad: sc.AnnData, key: str, **kwargs) -> None:
    """
    Plot the transitions on PAGA.

    Args:
        ad (sc.AnnData): AnnData object.
        key (str): Key for the transitions.
        **kwargs: Additional keyword arguments to pass to `sc.pl.paga`.
    """
    paga_kwargs = {
        "transitions": f"{key}_transitions",
        "node_size_scale": 2,
        "edge_width_scale": 3,
        "threshold": 0.001,
        "color": "cell_type",
        "cmap": "Reds",
        "labels": None,
        "pos": ad.uns["paga"]["pos"],
        "frameon": True,
        "fontoutline": 3,
        "fontsize": 12,
    }
    paga_kwargs.update(kwargs)

    sc.pl.paga(
        ad,
        **paga_kwargs,
    )
    plt.tight_layout()
