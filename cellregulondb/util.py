import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import networkx as nx
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


def plot_enrichment(
    enr_df, n_terms: int = 10, title: str = "", show: bool = False
) -> plt.Figure:
    """
    Plots the enrichment results from g:Profiler.

    This method uses the `matplotlib` package to create a horizontal bar plot of the enrichment results.
    The results are sorted by the negative logarithm of the p-value and the top `n_terms` results are plotted.
    The bars are colored red if the result is significant and blue otherwise.

    Args:
        enr_df (pd.DataFrame): A DataFrame containing the enrichment results from g:Profiler.
        n_terms (int, optional): The number of top results to plot. Defaults to 10.
        title (str, optional): The title of the plot. Defaults to "".
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    with plt.rc_context({"figure.figsize": (5, 10)}):
        enr_plt = enr_df[-n_terms:]
        enr_plt.set_index("name").neg_log_p.plot.barh(
            color=["red" if x else "blue" for x in enr_plt.significant],
            legend=False,
        )
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
        plt.title(title)
        plt.xlabel("-log(pval)")
        if show:
            plt.show()
        else:
            return plt.gcf()


def plot_networkx(
    nx_graph: nx.DiGraph,
    title: str = "",
    show: bool = False,
    layout: str = "kamada-kawai",
    figsize=(10, 10),
    edge_label_kwargs: dict = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots a networkx graph.

    This method uses the `matplotlib` package to create a plot of the networkx graph.
    The nodes are colored red if they are transcription factors and blue otherwise.

    Args:
        nx_graph (nx.DiGraph): A networkx DiGraph object to plot.
        title (str, optional): The title of the plot. Defaults to "".
        show (bool, optional): Whether to display the plot. Defaults to False.
        layout (str, optional): The layout of the plot.
            Options: ['kamada-kawai', 'circular', 'random', 'spring', 'shell', 'spectral'].
            Defaults to "kamada-kawai".
        figsize (tuple, optional): The size of the figure. Defaults to (10, 10).
        edge_label_kwargs (dict, optional): Additional keyword arguments to pass to `nx.draw_networkx_edge_labels`.
        **kwargs: Additional keyword arguments to pass to `nx.draw`.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    layouts = {
        "kamada-kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "spring": nx.spring_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
    }
    if layout not in layouts:
        raise ValueError(
            f"Invalid layout: {layout}. Choose from {list(layouts.keys())}"
        )

    draw_kwargs = {
        "node_color": [
            "bisque"
            if "transcription_factor" in nx_graph.nodes[node]
            and nx_graph.nodes[node]["transcription_factor"]
            else "lightsteelblue"
            for node in nx_graph.nodes
        ],
        "with_labels": True,
    }
    draw_kwargs.update(kwargs)

    with plt.rc_context({"figure.figsize": figsize}):
        pos = layouts[layout](nx_graph)
        nx.draw(
            nx_graph,
            pos,
            **draw_kwargs,
        )
        if edge_label_kwargs is not None:
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                **edge_label_kwargs,
            )
        plt.title(title)
        if show:
            plt.show()
        else:
            return plt.gcf()


def scale(series, min_val: float = 0, max_val: float = 1) -> pd.Series:
    """
    Scales a pandas Series to a specified range.

    This method scales a Series to a specified range using the formula:
    scaled = (x - min_x) / (max_x - min_x) * (max_val - min_val) + min_val

    Args:
        series (pd.Series): The series to scale.
        min_val (int, optional): The minimum value of the range. Defaults to 0.
        max_val (int, optional): The maximum value of the range. Defaults to 1.

    Returns:
        pd.Series: The scaled series.
    """
    return (series - series.min()) / (series.max() - series.min()) * (
        max_val - min_val
    ) + min_val


def z_score(series):
    """
    Computes the z-score of a pandas Series.

    This method computes the z-score of a Series using the formula:
    z = (x - mean) / std

    Args:
        series (pd.Series): The series to compute the z-score for.

    Returns:
        pd.Series: The z-scored series.
    """
    return (series - series.mean()) / series.std()


def robust_z_score(series):
    """
    Computes the robust z-score of a pandas Series.

    This method computes the robust z-score of a Series using the formula:
    z = 0.6745 * (x - median) / mad

    Args:
        series (pd.Series): The series to compute the robust z-score for.

    Returns:
        pd.Series: The robust z-scored series.
    """
    mad = sp.stats.median_abs_deviation(series, scale=1.0)
    return 0.6745 * (series - series.median()) / mad
