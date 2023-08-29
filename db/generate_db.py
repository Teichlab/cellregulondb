#!/usr/bin/env python3
#
# pip install requests pandas pyscenic
#
import warnings

warnings.filterwarnings("ignore")
import os
import gc
import json
import time
import requests
import sqlite3
import logging
import datetime
import numpy as np
import pandas as pd
from pyscenic.cli.utils import load_signatures

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")

sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.float64, lambda val: float(val))

###########################################################
# Config section
###########################################################

# DB name (uses 'cellregulon_<current date>.db' by default)
TIMESTAMP = datetime.datetime.now().isoformat(sep="T", timespec="seconds").replace(":", "-")

DB_PATH = f"cellregulon_{TIMESTAMP}.db"


# mapping file for harmonization
# see: https://docs.google.com/spreadsheets/d/1cD_4v3-Eqlchf2eQC2oiWOAgzlra_KAppJGwN45J2Ts/edit?usp=sharing
# (a) expected results sheet fromat
# | dataset | lineage | cell_type | tissue |   aucell   | grnboost2 | cistarget | subreg   | dif_exp               | barcode_celltype_map     |
# |   str   |   str   |    str    |  str   | file.loom  |  adj.csv  | reg.csv   | reg.yaml | DE_result_global.csv  | barcode_celltype_map.csv |
# (b) # expected tissue, lineage and cell_type sheets fromat
# | original_term  |   ontology    | ontology_label  |
# |   str          |  str (obo id) |     str         |
MAPPING_FILE = "mapping.xlsx"
mappings = {}
# load mappings for harmonizing tissues/lineages/celltypes
for m in ["tissue", "lineage", "cell_type"]:
    mappings[m] = (
        pd.read_excel(MAPPING_FILE, sheet_name=m)
        .fillna("")
        .set_index("original_term")
        .to_dict(orient="index")
    )
mappings["results"] = pd.read_excel(MAPPING_FILE, sheet_name="results", index_col=False)
mappings["datasets"] = pd.read_excel(
    MAPPING_FILE, sheet_name="datasets", index_col=False
)


###########################################################
# DB schema definition
###########################################################

# querys to create database structure
# - nodes table
create_nodes_table = """
CREATE TABLE IF NOT EXISTS nodes (
        id              INTEGER PRIMARY KEY,
        name            TEXT,
        ensembl_id      TEXT,
        chromosome      TEXT,
        strand          INTEGER,
        start           INTEGER,
        end             INTEGER,
        assembly        TEXT,
        description     TEXT,
        transcription_factor BOOLEAN
);
"""
# - node scores
create_node_scores_table = """
CREATE TABLE IF NOT EXISTS node_scores (
        id              INTEGER PRIMARY KEY,
        node_id         INTEGER,
        dataset_id      INTEGER,
        tissue_id       INTEGER,
        cell_type_id    INTEGER,
        dif_exp_score   REAL,
        dif_exp_pval    REAL,
        FOREIGN KEY(node_id) REFERENCES nodes (id)
        FOREIGN KEY(dataset_id) REFERENCES datasets (id), 
        FOREIGN KEY(tissue_id) REFERENCES tissues (id), 
        FOREIGN KEY(cell_type_id) REFERENCES cell_types (id)
);
"""
# - edges table
create_edges_table = """
CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY, 
        source_id INTEGER, 
        target_id INTEGER, 
        FOREIGN KEY(source_id) REFERENCES nodes (id), 
        FOREIGN KEY(target_id) REFERENCES nodes (id)
);
"""
# - datasets table
create_datasets_table = """
CREATE TABLE IF NOT EXISTS datasets (
        id          INTEGER PRIMARY KEY, 
        name        INTEGER, 
        doi         TEXT,
        year        INTEGER,
        description TEXT,
        data        TEXT
);
"""
# - tissues table
create_tissues_table = """
CREATE TABLE IF NOT EXISTS tissues (
        id      INTEGER PRIMARY KEY,
        label   TEXT,
        obo_id  TEXT
);
"""
# - lineages table
create_lineages_table = """
CREATE TABLE IF NOT EXISTS lineages (
        id      INTEGER PRIMARY KEY, 
        label   TEXT,
        obo_id  TEXT
);
"""
# - cell_types table
create_cell_types_table = """
CREATE TABLE IF NOT EXISTS cell_types (
        id      INTEGER PRIMARY KEY, 
        label   TEXT,
        obo_id  TEXT
);
"""
# - properties table
create_properties_table = """
CREATE TABLE IF NOT EXISTS properties (
        id               INTEGER PRIMARY KEY, 
        edge_id          INTEGER,
        dataset_id       INTEGER,
        tissue_id        INTEGER,
        lineage_id       INTEGER,
        cell_type_id     INTEGER,
        author_cell_type TEXT,
        regulation       TEXT,
        coexpression     REAL,
        rss              REAL,
        FOREIGN KEY(edge_id) REFERENCES edges (id), 
        FOREIGN KEY(dataset_id) REFERENCES datasets (id), 
        FOREIGN KEY(tissue_id) REFERENCES tissues (id), 
        FOREIGN KEY(lineage_id) REFERENCES lineages (id), 
        FOREIGN KEY(cell_type_id) REFERENCES cell_types (id)
);
"""
# - vestion table
create_version_table = """
CREATE TABLE IF NOT EXISTS version (
        version TEXT,
        notes TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);
"""


DATASETS = {}
TISSUES = {}
LINEAGES = {}
CELL_TYPES = {}


###########################################################
# Init DB and create tables and indices
###########################################################
def create_db():
    # connect to db file and create DB and indices
    logging.info(f"Opening database '{DB_PATH}'")
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    logging.info(f"Creating database schema '{DB_PATH}'")
    cursor.execute(create_datasets_table)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name ASC);"
    )
    cursor.execute(create_tissues_table)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tissues_label ON tissues (label ASC);"
    )
    cursor.execute(create_lineages_table)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_lineages_label ON lineages (label ASC);"
    )
    cursor.execute(create_cell_types_table)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cell_types_label ON cell_types (label ASC);"
    )
    cursor.execute(create_nodes_table)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes ON nodes (id ASC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name ASC);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_tf ON nodes (transcription_factor ASC);"
    )
    cursor.execute(create_node_scores_table)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores ON node_scores (id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_node ON node_scores (node_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_dataset ON node_scores (dataset_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_cell_type ON node_scores (cell_type_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_tissue ON node_scores (tissue_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_dif_exp ON node_scores (dif_exp_score ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_scores_full ON node_scores (dataset_id ASC, cell_type_id ASC, tissue_id ASC, dif_exp_score ASC);"
    )
    cursor.execute(create_edges_table)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges ON edges (id ASC);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id ASC);"
    )
    cursor.execute(create_properties_table)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_properties ON properties (id ASC);")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_edge ON properties (edge_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_dataset ON properties (dataset_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_lineage ON properties (lineage_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_cell_type ON properties (cell_type_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_tissue ON properties (tissue_id ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_coexpression ON properties (coexpression);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_regulation ON properties (regulation ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_rss ON properties (rss ASC);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_properties_full ON properties (edge_id ASC, dataset_id ASC, cell_type_id ASC, tissue_id ASC, regulation ASC);"
    )
    cursor.execute(create_version_table)
    logging.info(f"Done creating database schema '{DB_PATH}'")
    connection.commit()
    cursor.close()
    connection.close()


def insert_dataset_lineage_cell_type_tissue():
    # connect to db file and create DB and indices
    logging.info(f"Opening database '{DB_PATH}'")
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    logging.info(f"==Inserting datasets list")
    df = pd.read_excel(MAPPING_FILE, sheet_name="datasets")
    for d in df["dataset"].unique():
        if pd.isna(d):
            continue
        cursor.execute("INSERT INTO datasets (name) VALUES (?)", [d])
        DATASETS[d] = cursor.lastrowid
    logging.info(f"==Inserting lineages list")
    df = pd.read_excel(MAPPING_FILE, sheet_name="lineage", index_col=None)
    for label in df.ontology_label.unique():
        if pd.isna(label):
            continue
        obo = df[df.ontology_label == label].ontology.values[0]
        cursor.execute("INSERT INTO lineages (label,obo_id) VALUES (?,?)", [label, obo])
        LINEAGES[label] = cursor.lastrowid
    logging.info(f"==Inserting tissue list")
    df = pd.read_excel(MAPPING_FILE, sheet_name="tissue", index_col=None)
    for label in df.ontology_label.unique():
        if pd.isna(label):
            continue
        obo = df[df.ontology_label == label].ontology.values[0]
        cursor.execute("INSERT INTO tissues (label,obo_id) VALUES (?,?)", [label, obo])
        TISSUES[label] = cursor.lastrowid
    logging.info(f"==Inserting cell type list")
    df = pd.read_excel(MAPPING_FILE, sheet_name="cell_type")
    for label in df.ontology_label.unique():
        if pd.isna(label):
            continue
        obo = df[df.ontology_label == label].ontology.values[0]
        cursor.execute(
            "INSERT INTO cell_types (label,obo_id) VALUES (?,?)", [label, obo]
        )
        CELL_TYPES[label] = cursor.lastrowid

    connection.commit()
    cursor.close()
    connection.close()


###########################################################
# Manifest processing and inserts
###########################################################
def insert_aucell_and_subset():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    whole_thing = time.time()

    # make everything fast — what? that it may lead to corruption in the DB? oh well...
    connection.execute("PRAGMA synchronous  = OFF")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA temp_store   = memory")
    connection.execute("PRAGMA mmap_size    = 30000000000")
    connection.execute("PRAGMA mmap_size    = 30000000000")
    connection.execute("PRAGMA foreign_keys = OFF")

    rss = None
    NODES = {}
    EDGES = {}
    dataset = ""

    # keep a log of what went wrong
    errors = []

    # process manifest file
    df = mappings["results"]

    time_process = time.time()
    for item in df.to_dict(orient="records"):
        # skip any records with "doublet" annotation on cell types
        if "doublet" in item["cell_type"]:
            logging.warning(
                f"Skipping doublet {item['dataset']} :: {item['cell_type']} :: {item['tissue']}"
            )
            continue

        # check if we've change dataset?
        if dataset != item["dataset"]:
            if dataset:
                # if new dataset commit previous
                logging.info(f"Commiting {dataset}")
                connection.commit()
                gc.collect()
            dataset = item["dataset"]

            # load new RSS data
            if os.path.isfile(f"RSS/{dataset}_rss.csv"):
                logging.info(f"Loading {dataset} RSS")
                rss = pd.read_csv(f"RSS/{dataset}_rss.csv")
                rss = rss.set_index(rss.columns[0])
            else:
                logging.info(f"Missing file 'RSS/{dataset}_rss.csv'")
                errors.append(f"missing rss file for dataset {dataset}")
                rss = None

        try:
            # translate to harmonized annotations
            tissue = mappings["tissue"].get(item["tissue"])
            lineage = mappings["lineage"].get(item["lineage"])
            cell_type = mappings["cell_type"].get(item["cell_type"])

            # celltype tissue key
            cell_type__tissue = item["cell_type"] + "__" + item["tissue"]

            # start processing regulons for this record
            regulons = load_signatures(item["subreg"])
            logging.info(
                f"{dataset} :: {item['lineage']} :: {item['cell_type']} :: {item['tissue']} :: {len(regulons)} regulons"
            )
            if len(regulons) == 0:
                errors.append(f"empty regulon: {json.dumps(item)}")

            # process each regulon for this subsetted celltype/tissue
            for regulon in regulons:
                tf = regulon.name[:-3]
                # insert TF if not inserted before
                if tf not in NODES:
                    cursor.execute(
                        "INSERT INTO nodes (name, transcription_factor) VALUES (?,1)",
                        [tf],
                    )
                    NODES[tf] = {"tf": True, "id": cursor.lastrowid}
                # update to transcription_factor if gene has been inserted as a target gene before
                if not NODES[tf]["tf"]:
                    cursor.execute(
                        "UPDATE nodes SET transcription_factor = 1 WHERE id = :id",
                        NODES[tf],
                    )

                # get regulon name, regulation and score
                regulation = regulon.name[-2:-1]
                try:
                    rss_score = rss[regulon.name][cell_type__tissue]
                except:
                    errors.append(
                        f"missing rss: {dataset} :: {regulon.name} :: {cell_type__tissue} :: using 'rss = 0'"
                    )
                    rss_score = 0

                # process each target gene for this regulon
                for gene, weight in regulon.gene2weight.items():
                    # insert gene if not inserted before
                    if gene not in NODES:
                        cursor.execute(
                            "INSERT INTO nodes (name, transcription_factor) VALUES (?,0)",
                            [gene],
                        )
                        NODES[gene] = {"tf": False, "id": cursor.lastrowid}

                    # create edge key
                    edge_key = (
                        NODES[regulon.transcription_factor]["id"],
                        NODES[gene]["id"],
                    )
                    # insert edge if not inserted
                    if edge_key not in EDGES:
                        cursor.execute(
                            "INSERT INTO edges (source_id, target_id) VALUES (?,?)",
                            [edge_key[0], edge_key[1]],
                        )
                        EDGES[edge_key] = cursor.lastrowid

                    # values for this insertion
                    bind_values = {
                        "edge_id": EDGES[edge_key],
                        "dataset_id": DATASETS[item["dataset"]],
                        "tissue_id": TISSUES[tissue["ontology_label"]],
                        "lineage_id": LINEAGES[lineage["ontology_label"]],
                        "cell_type_id": CELL_TYPES[cell_type["ontology_label"]],
                        "author_cell_type": item["cell_type"],
                        "regulation": regulation,
                        "coexpression": weight,
                        "rss": rss_score,
                    }
                    # insert stuff
                    cursor.execute(
                        """
                        INSERT INTO properties (
                            edge_id,
                            dataset_id,
                            tissue_id,
                            lineage_id,
                            cell_type_id,
                            author_cell_type,
                            regulation,
                            coexpression,
                            rss)
                        VALUES (
                            :edge_id,
                            :dataset_id,
                            :tissue_id,
                            :lineage_id,
                            :cell_type_id,
                            :author_cell_type,
                            :regulation,
                            :coexpression,
                            :rss)
                        """,
                        bind_values,
                    )

        except Exception as e:
            logging.error(f"exception: {repr(e)}")
            errors.append(f"exception: {repr(e)} :: {json.dumps(item)}")

    # commit this dataset regulons
    connection.commit()

    cursor.close()
    connection.close()

    logging.info(
        f"Done properties {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-time_process))}"
    )

    logging.info(
        f"Finish {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-whole_thing))}"
    )

    with open(f"errors_insert_aucell_and_subset_{TIMESTAMP}.txt", "wt") as f:
        for e in errors:
            f.write(e)
            f.write("\n")


###########################################################
# DE scores for each gene
###########################################################
def insert_de_scores():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    whole_thing = time.time()

    # make everything fast — what? that it may lead to corruption in the DB? oh well...
    connection.execute("PRAGMA synchronous  = OFF")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA temp_store   = memory")
    connection.execute("PRAGMA mmap_size    = 30000000000")
    connection.execute("PRAGMA mmap_size    = 30000000000")
    connection.execute("PRAGMA foreign_keys = OFF")

    NODES = pd.read_sql("SELECT * FROM nodes", connection).set_index("name")
    DATASETS = pd.read_sql("SELECT * FROM datasets", connection).set_index("name")
    CELL_TYPES = pd.read_sql("SELECT * FROM cell_types", connection).set_index("label")
    TISSUES = pd.read_sql("SELECT * FROM tissues", connection).set_index("label")

    # keep a log of what went wrong
    errors = []

    for item in mappings["datasets"].to_dict(orient="records"):
        time_process = time.time()
        dataset_id = DATASETS.loc[item["dataset"]].id

        logging.info(f"Reading DE for {item['dataset']}")
        dif_exp = pd.read_csv(item["dif_exp"])
        # filter out genes not in the DB
        logging.info(f" ...filtering")
        dif_exp = dif_exp[dif_exp['names'].isin(NODES.index)]
        # remove all 'doublets' cell types
        logging.info(f" ...removing doublets")
        dif_exp = dif_exp[~dif_exp['group'].str.contains('doublet')]
        # add column ids and translate to harmonized annotations
        logging.info(f" ...hamonizing cell types")
        dif_exp['cell_type'] = [g.split("__")[0] for g in dif_exp['group'].values]
        dif_exp['cell_type_id'] = dif_exp.apply(lambda x: CELL_TYPES.loc[mappings["cell_type"].get(x.cell_type)['ontology_label']].id if mappings["cell_type"].get(x.cell_type) else 0, axis = 1)
        logging.info(f" ...hamonizing tissues")
        # have to do manual filtering because fetal lung doesn't have tissue property added
        if item["dataset"] != "fetal_lung":
            dif_exp['tissue'] = [g.split("__")[1] for g in dif_exp['group'].values]
        else:
            dif_exp['tissue'] = "lung"
        dif_exp['tissue_id'] = dif_exp.apply(lambda x: TISSUES.loc[mappings["tissue"].get(x.tissue)['ontology_label']].id if mappings["tissue"].get(x.tissue) else 0, axis = 1)
        logging.info(f" ...matching ids")
        dif_exp['dataset_id'] = dataset_id
        dif_exp['node_id'] = dif_exp.apply(lambda x: NODES.loc[x.names].id, axis = 1)
        logging.info(f" ...building inserts")
        dif_exp['params'] = dif_exp.apply(lambda x: {
                "node_id": x.node_id,
                "dataset_id": x.dataset_id,
                "cell_type_id": x.cell_type_id,
                "tissue_id": x.tissue_id,
                "score": x.scores,
                "pval": x.pvals,
            }, axis = 1)
    
        logging.info(f" ...executing inserts")
        cursor.executemany(
            """
            INSERT INTO node_scores (node_id, dataset_id, tissue_id, cell_type_id, dif_exp_score, dif_exp_pval)
            VALUES (:node_id, :dataset_id, :tissue_id, :cell_type_id, :score, :pval)
            """,
            dif_exp['params'].values
        )
        
        # commit this dataset
        connection.commit()
        logging.info(
            f"Done DE for {item['dataset']} {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-time_process))}"
        )

    cursor.close()
    connection.close()

    logging.info(
        f"Finish {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time()-whole_thing))}"
    )

    with open(f"errors_insert_de_scores_{TIMESTAMP}.txt", "wt") as f:
        for e in errors:
            f.write(e)
            f.write("\n")


###########################################################
# Lookup gene symbol information in
###########################################################
def lookup_symbols():
    # open db connection
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # make everything fast — what? that it may lead to corruption in the DB? oh well...
    connection.execute("PRAGMA synchronous  = OFF")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA temp_store   = memory")
    connection.execute("PRAGMA mmap_size    = 30000000000")

    # get all symbols from the DB
    all_symbols = cursor.execute("SELECT id, name FROM nodes").fetchall()
    match_results = []

    # setup both GRCh38 and GRCh37 endpoints and headers
    GRCH38_LOOK_SYMBOL_ENSEMBL_API = (
        "https://rest.ensembl.org/lookup/symbol/homo_sapiens"
    )
    GRCH37_LOOK_SYMBOL_ENSEMBL_API = (
        "https://grch37.rest.ensembl.org/lookup/symbol/homo_sapiens"
    )
    GRCH38_LOOK_ID_ENSEMBL_API = "https://rest.ensembl.org/lookup/id"
    GRCH37_LOOK_ID_ENSEMBL_API = "https://grch37.rest.ensembl.org/lookup/id"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    # start fetching symbols and update db
    # first pass will try to match symbol names to GRCh38
    total = len(all_symbols)
    good = 0
    bad = 0
    failed1 = []
    logging.info(f"Running all symbols {len(all_symbols)} with GRCh38 lookup/symbol")
    # batch genes 1000 at the time
    for i in range(0, len(all_symbols), 999):
        batch_symbols = all_symbols[i : i + 999]
        response = requests.post(
            GRCH38_LOOK_SYMBOL_ENSEMBL_API,
            headers=HEADERS,
            json={"symbols": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for id, name in batch_symbols:
                if name in data.keys() and data[name] is not None:
                    symbol_information = data[name]
                    cursor.execute(
                        "UPDATE nodes SET ensembl_id = ?, chromosome = ?, strand = ?, start = ?, end = ?, assembly = ?, description = ? WHERE id = ?",
                        [
                            symbol_information.get("id", None),
                            symbol_information.get("seq_region_name", None),
                            symbol_information.get("strand", None),
                            symbol_information.get("start", None),
                            symbol_information.get("end", None),
                            symbol_information.get("assembly_name", None),
                            symbol_information.get("description", None),
                            id,
                        ],
                    )
                    good += 1
                    match_results.append({"gene_in_db": name, **data[name]})
                else:
                    bad += 1
                    failed1.append([id, name])
            logging.info(
                f"[GRCh38 symbol] match={good} :: unmatched={bad} :: total={total}"
            )
        else:
            logging.error("[GRCh38 symbol] POST failed", response.content)
        connection.commit()

    # second pass will try to match unmatched symbol names to GRCh37
    logging.info(f"Running failed symbols {len(failed1)} with GRCh37 lookup/symbol")
    total = len(failed1)
    good = 0
    bad = 0
    failed2 = []
    for i in range(0, len(failed1), 999):
        batch_symbols = failed1[i : i + 999]
        response = requests.post(
            GRCH37_LOOK_SYMBOL_ENSEMBL_API,
            headers=HEADERS,
            json={"symbols": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for id, name in batch_symbols:
                if name in data.keys() and data[name] is not None:
                    symbol_information = data[name]
                    cursor.execute(
                        "UPDATE nodes SET ensembl_id = ?, chromosome = ?, strand = ?, start = ?, end = ?, assembly = ?, description = ? WHERE id = ?",
                        [
                            symbol_information.get("id", None),
                            symbol_information.get("seq_region_name", None),
                            symbol_information.get("strand", None),
                            symbol_information.get("start", None),
                            symbol_information.get("end", None),
                            symbol_information.get("assembly_name", None),
                            symbol_information.get("description", None),
                            id,
                        ],
                    )
                    good += 1
                    match_results.append({"gene_in_db": name, **data[name]})
                else:
                    bad += 1
                    failed2.append([id, name])
            logging.info(
                f"[GRCh37 symbol] match={good} :: unmatched={bad} :: total={total}"
            )
        else:
            logging.error("[GRCh37 symbol] POST fail", response.content)

    # third pass we'll asume they are not symbols but ENS ids
    logging.info(f"Running failed symbols {len(failed2)} with GRCh38 lookup/id")
    total = len(failed2)
    good = 0
    bad = 0
    failed3 = []
    for i in range(0, len(failed2), 999):
        batch_symbols = failed2[i : i + 999]
        response = requests.post(
            GRCH38_LOOK_ID_ENSEMBL_API,
            headers=HEADERS,
            json={"ids": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for id, name in batch_symbols:
                if name in data.keys() and data[name] is not None:
                    symbol_information = data[name]
                    cursor.execute(
                        "UPDATE nodes SET ensembl_id = ?, chromosome = ?, strand = ?, start = ?, end = ?, assembly = ?, description = ? WHERE id = ?",
                        [
                            symbol_information.get("id", None),
                            symbol_information.get("seq_region_name", None),
                            symbol_information.get("strand", None),
                            symbol_information.get("start", None),
                            symbol_information.get("end", None),
                            symbol_information.get("assembly_name", None),
                            symbol_information.get("description", None),
                            id,
                        ],
                    )
                    good += 1
                    match_results.append({"gene_in_db": name, **data[name]})
                else:
                    bad += 1
                    failed3.append([id, name])
            logging.info(f"[GRCh38 id] Match={good}. Unmatched={bad}. Total={total}")
        else:
            logging.error("[GRCh38 id] POST fail", response.content)

    # third pass we'll asume they are not symbols but ENS ids
    logging.info(f"Running failed symbols {len(failed3)} with GRCh37 lookup/id")
    total = len(failed3)
    good = 0
    bad = 0
    for i in range(0, len(failed3), 999):
        batch_symbols = failed3[i : i + 999]
        response = requests.post(
            GRCH37_LOOK_ID_ENSEMBL_API,
            headers=HEADERS,
            json={"ids": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for id, name in batch_symbols:
                if name in data.keys() and data[name] is not None:
                    symbol_information = data[name]
                    cursor.execute(
                        "UPDATE nodes SET ensembl_id = ?, chromosome = ?, strand = ?, start = ?, end = ?, assembly = ?, description = ? WHERE id = ?",
                        [
                            symbol_information.get("id", None),
                            symbol_information.get("seq_region_name", None),
                            symbol_information.get("strand", None),
                            symbol_information.get("start", None),
                            symbol_information.get("end", None),
                            symbol_information.get("assembly_name", None),
                            symbol_information.get("description", None),
                            id,
                        ],
                    )
                    good += 1
                    match_results.append({"gene_in_db": name, **data[name]})
                else:
                    bad += 1
                    match_results.append({"gene_in_db": name})
            logging.info(f"[GRCh37 id] Match={good}. Unmatched={bad}. Total={total}")
        else:
            logging.error("[GRCh37 id] POST fail", response.content)
        connection.commit()

    # gracefuly close everything
    cursor.close()
    connection.close()
    logging.info(f"Finished looking up symbols")

    # dataframe with all match results
    result = pd.DataFrame.from_records(match_results)

    logging.info(f"Saving symbol lookup logs")
    result.to_csv(f"lookup_symbol_summary_{TIMESTAMP}.csv", index=False)

    # summary
    total = result.shape[0]
    matched = result[~pd.isnull(result["id"])].shape[0]
    unmatched = total - matched

    logging.info(f"Symbol lookup total.......{total}")
    logging.info(f"Symbol lookup matched.....{matched} ({round(matched*100/total,2)}%)")
    logging.info(
        f"Symbol lookup unmatched...{unmatched} ({round(unmatched*100/total,2)}%)"
    )

    logging.info(f"Done")


if __name__ == "__main__":
    create_db()
    insert_dataset_lineage_cell_type_tissue()
    insert_aucell_and_subset()
    insert_de_scores()
    lookup_symbols()

    logging.info(f"THE END")
