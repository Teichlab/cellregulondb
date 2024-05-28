__version__ = "0.1.2"
__author__ = ["J.P.Pett", "M.Prete"]

import multiprocessing as mp

from .cellregulondb import CellRegulonDB, download_db
from .regulonatlas import RegulonAtlas
from .util import query_gprofiler, plot_enrichment, plot_networkx
import cellregulondb.anndata as ad

NCPU: int = mp.cpu_count()


def get_num_cpu() -> int:
    """
    Get the number of CPUs set globally.

    Returns:
        int: Number of CPUs.
    """
    return NCPU


def set_num_cpu(n: int) -> None:
    """
    Globally set the number of CPUs.

    Args:
        n (int): Number of CPUs.
    """
    global NCPU
    NCPU = n
