from typing import Union
import os
import json
import pickle
import logging
import sqlite3
import pandas as pd
import networkx as nx

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")


class CellRegulonDB:
    """
    A class wrapping the CellRegulon database.

    Args:
        db_path (str): Path to the SQLite database file.

    Attributes:
        db (sqlite3.Connection): Connection to the SQLite database.
        TF (pandas.DataFrame): DataFrame containing information about transcription factors (TFs).
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
            raise Exception(f"Invalid file '{db_path}'")
        self.db = sqlite3.connect(db_path)
        self.TFs = self._build_TF_dataframe()
        self.lineages = self._get_lineages()

    def _get_lineages(self) -> pd.DataFrame:
        rows = self.db.execute(
            "select distinct properties->>'cell_label' from edges"
        ).fetchall()
        return pd.DataFrame([row[0] for row in rows], columns=["lineage"])

    def _build_TF_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame containing information about transcription factors (TFs) from the database.

        Returns:
            pandas.DataFrame: DataFrame containing TF information.

        """
        rows = self.db.execute(
            """
            select name, properties 
            from nodes where properties->>'is_TF'=1
            """
        ).fetchall()

        return pd.DataFrame.from_records(
            [
                self._get_symbol_info(row[0], json.loads(row[1])).values()
                for row in rows
            ],
            columns=["TF", "ENS", "chromosome", "start", "end"],
        )

    def location_search(
        self,
        chromosome: str,
        location: int = None,
        start: int = None,
        end: int = None,
        delta: int = 10000,
    ) -> pd.DataFrame:
        """
        Searches for transcription factors based on location information.

        Args:
            chromosome (str): Chromosome name.
            location (int, optional): Specific location to search for the start of the transcription factors. Defaults to None.
            start (int, optional): Start position of the range to search for transcription factors. Defaults to None.
            end (int, optional): End position of the range to search for transcription factors. Defaults to None.
            delta (int, optional): Delta value used for range search. Defaults to 10000 (+/-10kb).

        Returns:
            pandas.DataFrame: DataFrame containing the matching transcription factors.

        Raises:
            Exception: If neither 'location' nor 'start' and 'end' are provided.

        """
        chromosome = str(chromosome)

        if location is not None:
            return self.TFs[
                (location >= self.TFs["start"] - delta)
                & (location <= self.TFs["start"] + delta)
                & (self.TFs["chromosome"] == chromosome)
            ]
        elif start is not None and end is not None:
            return self.TFs[
                (end >= self.TFs["start"] - delta)
                & (start <= self.TFs["start"] + delta)
                & (self.TFs["chromosome"] == chromosome)
            ]
        else:
            raise Exception("Required 'location' or 'start' and 'end'")

    def _get_regulon(self, TF: str, lineage: list = None) -> pd.DataFrame:
        """
        Retrieves the regulon for a given transcription factor.

        Args:
            TF (str): Name of the transcription factors.
            lineage (list, optional): List of cell lineages to filter the regulon. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the regulon information.

        """
        regulon = []
        rows = self.db.execute(
            """
            select s.name,t.name,e.properties
            from edges as e, nodes as s, nodes as t
            where s.id=e.source_id and t.id=e.target_id and s.name = ?
            """,
            [TF],
        ).fetchall()
        for row in rows:
            data = json.loads(row[2])
            if lineage is None or data["cell_label"] in lineage:
                regulon.append(
                    [
                        row[0],
                        row[1],
                        data["regulation"],
                        data["coexpresion"],
                        data["motif_enrichment"],
                        data["cell_label"],
                        data["dataset"],
                    ]
                )
        return pd.DataFrame(
            regulon,
            columns=[
                "TF",
                "gene",
                "regulation",
                "coexpresion",
                "motif_enrichment",
                "cell_label",
                "dataset_source",
            ],
        )

    def get_regulon(self, TF: Union[str, list], lineage: list = None) -> pd.DataFrame:
        """
        Retrieves the regulon for one or more TFs.

        Args:
            TF (Union[str, list]): Name of the TF or list of TF names.
            lineage (list, optional): List of cell lineages to filter the regulon. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the regulon information.

        """
        if isinstance(TF, list):
            regulons = pd.concat([self._get_regulon(tf, lineage) for tf in set(TF)])
        else:
            regulons = self._get_regulon(str(TF), lineage)
        return regulons

    def get_shared_genes(self, data: Union[list, pd.DataFrame]) -> list:
        """
        Retrieves the shared genes among multiple regulons.

        Args:
            data (Union[list, pd.DataFrame]): List of transcription factors or a DataFrame containing regulon information.

        Returns:
            list: A list containing the shared genes.

        """
        if isinstance(data, list):
            data = self.get_regulon(data)
        regulons = data.groupby("TF")["gene"].apply(list)
        return list(set.intersection(*[set(genes) for _, genes in regulons.items()]))

    def _get_symbol_info(self, symbol: str, attr: dict) -> dict:
        """
        Extracts relevant information from a node's attributes.

        Args:
            symbol (str): Name of the gene.
            attr (dict): Dictionary containing symbol attributes.

        Returns:
            dict: Dictionary with extracted symbol information.

        """
        info = {"TF": symbol, "ENS": 0, "chromosome": 0, "start": 0, "end": 0}
        if "symbol_information" in attr and attr["symbol_information"]:
            _id = attr["symbol_information"]["id"]
            _start = attr["symbol_information"]["start"]
            _end = attr["symbol_information"]["end"]
            _strand = attr["symbol_information"]["strand"]
            _chromosome = attr["symbol_information"]["seq_region_name"]
            info["ENS"] = _id
            info["chromosome"] = _chromosome
            info["start"] = _start if _strand == 1 else _end
            info["end"] = _end if _strand == 1 else _start

        return info

    def to_networkx(
        self, df: pd.DataFrame = None, source: str = "TF", target: str = "gene"
    ) -> nx.MultiDiGraph:
        """
        Converts the database to a NetworkX MultiDiGraph.

        Returns:
            nx.MultiDiGraph: NetworkX MultiDiGraph representing the database.

        """
        if df is None:
            logging.warning(
                "No df provided. Converting whole database to NetworkX MultiDiGraph. This operation will take a long time."
            )
        # create graph
        G = nx.MultiGraph()
        if df is None:
            # read all nodes
            nodes = self.db.execute("select id, name, properties from nodes").fetchall()
            # add them to the graph
            G.add_nodes_from([(n[1], json.loads(n[2])) for n in nodes])

            # read all edges
            edges = self.db.execute(
                """
                select s.name,t.name,e.properties
                from edges e, nodes t, nodes s
                where s.id=e.source_id and t.id=e.target_id
                """
            ).fetchall()
            # add them to the graph
            G.add_edges_from([(e[0], e[1], json.loads(e[2])) for e in edges])
        else:
            G.add_edges_from(
                [
                    (data[source], data[target], data.to_json(orient="columns"))
                    for _, data in df.iterrows()
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

    def to_cytosscape(self, df: pd.DataFrame, name: str = "cytoscape.txt") -> None:
        """
        Exports the database to a Cytoscape-readable file.

        Args:
            name (str, optional): Name of the output file. Defaults to "cytoscape.txt".

        """
        # import networkx
        # self.G.cytoscape_data
        pass

    def as_pyscenic(self, df: pd.DataFrame) -> "pySCENIC":
        try:
            import pyscenic
        except ModuleNotFoundError:
            raise Exception("Module 'pyscenic' is not installed.")
        """
        Converts the database to pySCENIC format.

        Returns:
            pySCENIC?: pySCENIC representation of the database.

        """
        pass


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
