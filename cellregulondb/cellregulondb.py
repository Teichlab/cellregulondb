from typing import Union, Tuple
import os
import json
import pickle
import difflib
from tqdm import tqdm
import logging
import sqlite3
import pandas as pd
import numpy as np
import networkx as nx

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")

sqlite3.register_adapter(np.int64, lambda val: int(val))


class CellRegulonDB:
    """
    A class wrapping the CellRegulon database.

    Args:
        db_path (str): Path to the SQLite database file.

    Attributes:
        db (sqlite3.Connection): Connection to the SQLite database.
        genes (pandas.DataFrame): DataFrame containing information about genes in the database.
        lineages (pandas.DataFrame): DataFrame containing information about lineages in the database.
        tissues (pandas.DataFrame): DataFrame containing information about tissues in the database.
        cell_types (pandas.DataFrame): DataFrame containing information about cell_types in the database.
    """

    def __init__(self, db_path: str):
        """
        Initializes a CellRegulonDB object.

        Args:
            db_path (str): Path to the SQLite database file.

        Raises:
            Exception: If the provided database file does not exist.

        """
        if not os.path.isfile(db_path):
            raise SystemExit(f"Invalid database file '{db_path}'")
        self.db = sqlite3.connect(db_path)

        self.lineages = pd.read_sql("select * from lineages", self.db)
        self.tissues = pd.read_sql("select * from tissues", self.db)
        self.cell_types = pd.read_sql("select * from cell_types", self.db)
        self.genes = pd.read_sql("select * from nodes", self.db)

    def get_regulons(
        self,
        genes: list = None,
        cell_types: list = None,
        tissues: list = None,
        genes_are_transcription_factors: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves a pandas `DataFrame` of regulons based on several criteria.

        Args:
            genes (list, optional): List of gene names. Defaults to None.
            cell_types (list, optional): List of cell types to filter the regulon. Defaults to None.
            tissues (list, optional): List of tissues to filter the regulon. Defaults to None.
            genes_are_transcription_factors (bool, optional): Whether the provided genes are transcription factors. Defaults to False.

        Returns:
            pandas.DataFrame: DataFrame containing the regulon information.

        """

        # match to database ids
        if tissues is None:
            tissue_ids = []
        else:
            tissue_ids = self.tissues[self.tissues["label"].isin(tissues)]["id"].values

        if cell_types is None:
            cell_type_ids = []
        else:
            cell_type_ids = self.cell_types[self.cell_types["label"].isin(cell_types)][
                "id"
            ].values

        if genes is None:
            gene_ids = []
        else:
            gene_ids = self.genes[self.genes["name"].isin(genes)]["id"].values

        # assemble SQL query
        sql_details = f"""
        SELECT  source.name as 'transcription_factor',
                p.regulation,
                target.name as 'target_gene',
                t.label as tissue,
                c.label as cell_type,
                p.author_cell_type,
                p.coexpression,
                p.rss
        FROM    nodes as source, nodes as target,
                edges AS e, properties AS p,
                tissues as t, cell_types as c
        WHERE   1=1
                AND source.id = e.source_id
                AND target.id = e.target_id
                AND e.id = p.edge_id 
                AND p.tissue_id = t.id
                AND p.cell_type_id = c.id
        """

        binds = []
        where = " "

        if len(cell_type_ids) > 0:
            conditions = []
            for _id in cell_type_ids:
                conditions.append(f" p.cell_type_id = ? ")
                binds.append(_id)
            where += f" AND ({ ' OR '.join(conditions) })"

        if len(tissue_ids) > 0:
            conditions = []
            for _id in tissue_ids:
                conditions.append(f" p.tissue_id = ? ")
                binds.append(_id)
            where += f" AND ({ ' OR '.join(conditions) })"

        if len(gene_ids) > 0:
            conditions = []
            for _id in gene_ids:
                if genes_are_transcription_factors:
                    conditions.append(f" e.source_id = ? ")
                    binds.append(_id)
                else:
                    conditions.append(f" ( e.source_id = ? OR e.target_id = ?) ")
                    binds.extend([_id, _id])
            where += f" AND ({ ' OR '.join(conditions) })"

        # read SQL database and return as DataFrame
        df = pd.read_sql(sql_details + where, self.db, params=binds)

        return df

    def get_shared_genes(
        self,
        regulons: Union[list, pd.DataFrame],
        transcription_factor_column: str = "transcription_factor",
        target_gene_column: str = "target_gene",
    ) -> list:
        """
        Retrieves the shared genes amongst multiple regulons.

        Args:
            data (Union[list, pd.DataFrame]): List of transcription factors or a DataFrame containing regulon information.

        Returns:
            list: A list containing the shared genes.

        """
        if isinstance(regulons, list):
            regulons_df = self.get_regulons(
                genes=regulons, genes_are_transcription_factors=True
            )
        else:
            regulons_df = regulons

        regulons_df = regulons_df.groupby(transcription_factor_column)[
            target_gene_column
        ].apply(list)

        intersection = set.intersection(
            *[set(genes) for _, genes in regulons_df.items()]
        )

        return self.genes[self.genes["name"].isin(intersection)]

    def to_networkx(
        self,
        df: pd.DataFrame = None,
        source_name: str = "transcription_factor",
        target_name: str = "target_gene",
        filename: str = None,
    ) -> nx.MultiDiGraph:
        """
        Converts the database to a NetworkX MultiDiGraph.

        Returns:
            nx.MultiDiGraph: NetworkX MultiDiGraph representing the database.

        """
        # create graph
        G = nx.MultiGraph()

        # add all gemes to the graph
        logging.info("Collecting nodes")
        G.add_nodes_from(
            [(g.name, g) for g in tqdm(self.genes.itertuples(index=False))]
        )

        # read all edges
        sql = """
            SELECT  source.name as 'transcription_factor',
                    p.regulation,
                    target.name as 'target_gene',
                    t.label as tissue,
                    c.label as cell_type,
                    p.author_cell_type,
                    p.coexpression,
                    p.rss
            FROM    nodes as source, nodes as target,
                    edges AS e, properties AS p,
                    tissues as t, cell_types as c
            WHERE   1=1
                    AND source.id = e.source_id
                    AND target.id = e.target_id
                    AND e.id = p.edge_id 
                    AND p.tissue_id = t.id
                    AND p.cell_type_id = c.id
        """
        df = pd.read_sql(sql, self.db)
        # add them to the graph
        logging.info("Collecting edges")
        G.add_edges_from(
            [
                (e.transcription_factor, e.target_gene, e)
                for e in tqdm(df.itertuples(index=False))
            ]
        )
        return G

    def save_networkx(G, path: str = "cellregulon.nx") -> None:
        # store as pickled object
        with open(path, "wb") as f:
            pickle.dump(G, f)

    def load_networkx(path: str = "cellregulon.nx") -> nx.MultiDiGraph:
        # load pickled object
        with open(path, "rb") as f:
            G = pickle.load(f)
        return G

    # def to_cytosscape(self, df: pd.DataFrame, name: str = "cytoscape.txt") -> None:
    #     """
    #     Exports the database to a Cytoscape-readable file.

    #     Args:
    #         name (str, optional): Name of the output file. Defaults to "cytoscape.txt".

    #     """
    #     # import networkx
    #     # self.G.cytoscape_data
    #     pass

    # def as_pyscenic(self, df: pd.DataFrame) -> "pySCENIC":
    #     try:
    #         import pyscenic
    #     except ModuleNotFoundError:
    #         raise Exception("Module 'pyscenic' is not installed.")
    #     """
    #     Converts the database to pySCENIC format.

    #     Returns:
    #         pySCENIC?: pySCENIC representation of the database.

    #     """
    #     pass


def download_db(db_path: str = None, version: str = "latest") -> str:
    """
    Downloads a CellRegulon database file.

    Args:
        db_path (str, optional): Path to save the downloaded database file. If not provided, the file will be saved as "cellregulon-vXXX.db" in the current directory. Defaults to None.
        version (str, optional): Version of the CellRegulon database to download. Defaults to "latest".

    Returns:
        str: Path to the downloaded database file.

    Raises:
        Exception: If the specified version is not available.

    """
    import requests
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress_columns = (
        "[progress.description]{task.description}",
        BarColumn(),
        TimeRemainingColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )

    # get avaiable versions
    versions_url = "https://cellregulondb.cog.sanger.ac.uk/db/versions.json"
    data = requests.get(versions_url).json()
    if version == "latest":
        version = data["latest"]
    required_version = [db for db in data["versions"] if version == db["version"]]

    # sanity check the version exists
    if not required_version:
        raise Exception(
            f"Wrong verision number. Avaiable versions: {[db['version'] for db in data['versions']]}"
        )

    selected = required_version[0]
    if not db_path:
        db_path = f"cellregulon-v{selected['version']}.db"

    if os.path.isfile(db_path):
        logging.warning(
            f"Database file '{db_path}' already exists. Will be overwritten."
        )

    with requests.get(selected["url"], allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        content_length = int(r.headers.get("Content-Length"))
        logging.info(f"Downloading database verssion {selected['version']}")
        with Progress(*progress_columns) as progress:
            task = progress.add_task(
                f"downloading v{selected['version']}", total=content_length
            )
            with open(db_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=5 * 1024**2):
                    progress.advance(task, f.write(chunk))
            logging.info(f"Downloaded {db_path}")
            return db_path
