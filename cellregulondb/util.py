import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def query_gprofiler(
    query_genes, use_src: list = None, plot: bool = True, organism="hsapiens"
) -> pd.DataFrame:
    """
    Queries g:Profiler with a list of genes.

    This method uses the `enrich` function from Scanpy's queries module to query g:Profiler with a list of genes.
    The results are filtered by the sources specified in `use_src` and sorted by the negative logarithm of the p-value.
    If `plot` is set to True, it will also plot the enrichment using the `plot_enrichment` method.

    Args:
        query_genes (list): A list of genes to query.
        use_src (list, optional): A list of sources to filter the results by. Defaults to ["GO:BP", "WP", "KEGG"].
        plot (bool, optional): Whether to plot the enrichment. Defaults to True.
        organism (str, optional): string describing the organism for `sc.queries.enrich`. Defaults to "hsapiens".

    Returns:
        pd.DataFrame: A DataFrame containing the filtered and sorted results from g:Profiler.
    """
    if use_src is None:
        use_src = ["GO:BP", "WP", "KEGG"]

    enr_df = sc.queries.enrich(
        query_genes,
        org=organism,
        gprofiler_kwargs=dict(all_results=True, no_evidences=False),
    )

    sub_df = enr_df.query("source in @use_src", engine="python")
    sub_df = sub_df.assign(neg_log_p=-np.log(sub_df["p_value"])).sort_values(
        "neg_log_p"
    )

    if plot:
        plot_enrichment(sub_df)

    return sub_df


def plot_enrichment(enr_df, n_terms: int = 10) -> None:
    """
    Plots the enrichment results from g:Profiler.

    This method uses the `matplotlib` package to create a horizontal bar plot of the enrichment results.
    The results are sorted by the negative logarithm of the p-value and the top `n_terms` results are plotted.
    The bars are colored red if the result is significant and blue otherwise.

    Args:
        enr_df (pd.DataFrame): A DataFrame containing the enrichment results from g:Profiler.
        n_terms (int, optional): The number of top results to plot. Defaults to 10.
    """
    with plt.rc_context({"figure.figsize": (5, 10)}):
        enr_plt = enr_df[-n_terms:]
        enr_plt.set_index("name").neg_log_p.plot.barh(
            color=["red" if x else "blue" for x in enr_plt.significant],
            legend=False,
        )
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
        plt.xlabel("-log(pval)")
        plt.show()
