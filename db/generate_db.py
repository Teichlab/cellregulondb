#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")
import os
import json
import time
import sqlite3
import logging
import datetime
import loompy as lp
import pandas as pd
from pyscenic.cli.utils import load_signatures

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")


###########################################################
# Config section
###########################################################

# path to the results manifests
# expected manifest fromat
# | dataset | dataset_lineage | ontology_label |      AUCell      |    GRNBoost   |   cisTarget   |
# |   name  |     lineage     |   onto label   |  path/file.loom  |  path/adj.csv | path/reg.csv  |
RESULTS_MANIFEST = "results.csv"

# DB name (uses 'cellregulon_<current date>.db' by default)
DB_PATH = f"cellregulon_{datetime.date.today()}.db"

# get coexpresion from GRNboost?
FETCH_COEXPRESION = True

# get motif expression from cisTarget?
FETHC_MOTIF_ENRICHMENT = True


###########################################################
# DB schema definition
###########################################################

# querys to create database structure
# - nodes table
create_nodes_table = """
CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY,
        name TEXT,
        properties JSON
);
"""
# - edges table
create_edges_table = """
CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY, 
        source_id INTEGER, 
        target_id INTEGER, 
        properties JSON,
        FOREIGN KEY(source_id) REFERENCES nodes (id), 
        FOREIGN KEY(target_id) REFERENCES nodes (id)
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


###########################################################
# Init DB and create tables and indices
###########################################################
def create_db():
    # connect to db file and create DB and indices
    logging.info(f"Opening database '{DB_PATH}'")
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    logging.info(f"Creating database schema '{DB_PATH}'")
    cursor.execute(create_nodes_table)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes ON nodes (id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name);")
    cursor.execute(create_edges_table)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges ON edges (id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id);")
    cursor.execute(create_version_table)
    logging.info(f"Done creating database '{DB_PATH}'")
    connection.commit()
    cursor.close()
    connection.close()


###########################################################
# Manifest processing and inserts
###########################################################
def main():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # make everything fast — what? that it may lead to corruption in the DB? oh well...
    connection.execute("PRAGMA synchronous  = OFF")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA temp_store   = memory")
    connection.execute("PRAGMA mmap_size    = 30000000000")

    # keep a references of all the inserted nodes
    nodes = {}

    # process manifest file
    df = pd.read_csv(RESULTS_MANIFEST, index_col=False)
    time_process = time.time()
    for _, row in df.iterrows():
        time_row = time.time()
        dataset = row["dataset"]
        lineage = row["dataset_lineage"]
        DOI = row["DOI"]
        CL = row["ontology_label"]
        logging.info(f"Dataset: '{dataset}'. Lineage '{lineage}'. Ontology '{CL}'")

        # read loom output from AUCell
        auc_outpout = row["AUCell"]
        logging.info(f"Reading AUCell output: '{auc_outpout}'")
        lf = lp.connect(auc_outpout, mode="r", validate=False)

        # create dataframe
        # some examples at https://github.com/aertslab/pySCENIC/issues/352
        df = pd.DataFrame(lf.ra.Regulons, index=lf.ra.Gene)
        logging.info(f"Total Regulons in file: {df.shape[1]}")
        logging.info(f"Total Genes in file: {df.shape[0]}")

        # remove genes that don't match any regulon — why are those here tho?
        df = df.loc[~(df == 0).all(axis=1)]
        logging.info(f"Genes after filtering: {df.shape[0]}")

        # insert all nodes first
        logging.info(f"Inserting nodes (TF and genes)")
        # - all transcription factors
        for TF in sorted(
            list(set([k[:-3] for k in lf.ra.Regulons.dtype.fields.keys()]))
        ):
            if TF not in nodes:
                props = {"is_TF": True}
                cursor.execute(
                    "INSERT INTO nodes (name, properties) VALUES (?,?)",
                    [TF, json.dumps(props)],
                )
                nodes[TF] = cursor.lastrowid
            else:
                # if symbol was already inserted but as 'gene' instead of TF update record to be TF (is_TF = True)
                node = cursor.execute(
                    "SELECT id, properties FROM nodes WHERE id = ? AND properties->>'is_TF' = 0",
                    [nodes[TF]],
                ).fetchone()
                if node:
                    props = json.loads(node[1])
                    props["is_TF"] = True
                    cursor.execute(
                        "UPDATE nodes SET properties = ? WHERE id = ?",
                        [json.dumps(props), node[0]],
                    )

        # - all genes
        for gene in lf.ra.Gene:
            if gene not in nodes:
                props = {"is_TF": False}
                cursor.execute(
                    "INSERT INTO nodes (name, properties) VALUES (?,?)",
                    [gene, json.dumps(props)],
                )
                nodes[gene] = cursor.lastrowid
        logging.info(f"Done inserting nodes")

        # read additional analysis for this dataset/lineage
        if FETCH_COEXPRESION:
            GRNBoost_path = row["GRNBoost"]
            logging.info(f"Reading GRNBoost from file: '{GRNBoost_path}'")
            GRNBoost = pd.read_csv(GRNBoost_path)

        # preprocess cisTopic horrible output where motifs is an array of tuples per row in the file
        # but it's a string so it's converted to list eval(motifs) and then fattened into a single list
        if FETHC_MOTIF_ENRICHMENT:
            cisTarget_path = row["cisTarget"]
            logging.info(f"Reading cisTarget from file: '{cisTarget_path}'")
            cisTarget = load_signatures(cisTarget_path)

        logging.info(f"Inserting edges")
        # transverse the columns (tf) and the matching rows (genes) to insert edges
        for label, content in df.items():
            genes = content[content == 1].index.values
            logging.info(f"Inserting TF '{label}' ({len(genes)} genes)")
            rname = label[:-3]
            regulation = label[-2:-1]

            # filter GRNBoost and cisTarget using the TF
            if FETCH_COEXPRESION:
                TF_GRNBoost = GRNBoost.query(f"TF=='{rname}'")
            if FETHC_MOTIF_ENRICHMENT:
                TF_cisTarget = [tf for tf in cisTarget if tf.name == label]
                if len(TF_cisTarget) > 1:
                    logging.warning(
                        f"Regulon {label} is present more than once in cisTarget result?"
                    )

            for gene in genes:
                # the following two lookups (GRNboosst and cisTarget) take too long and should be optimized?
                # - get GRNboost coexpresion for gene
                coexpresion = 0
                if FETCH_COEXPRESION and not TF_GRNBoost.empty:
                    gene_GRNBoost = TF_GRNBoost.query(f"target=='{gene}'")
                    if not gene_GRNBoost.empty:
                        coexpresion = gene_GRNBoost["importance"].values[0]
                # - get cisTarget motif enrichment for gene
                motif_enrichment = 0
                if FETHC_MOTIF_ENRICHMENT and len(TF_cisTarget):
                    motif_enrichment = cisTarget[0].gene2weight.get(gene, 0)

                # build edge properties json
                props = {
                    "regulation": regulation,
                    "coexpresion": coexpresion,
                    "motif_enrichment": motif_enrichment,
                    "dataset": dataset,
                    "doi": DOI,
                    "lineage": lineage,
                    "cell_label": CL,
                }
                cursor.execute(
                    "INSERT INTO edges (source_id, target_id, properties) VALUES (?,?,?)",
                    [nodes[rname], nodes[gene], json.dumps(props)],
                )

        logging.info(f"Committing transaction")
        connection.commit()

        logging.info(
            f"Done in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time()-time_row))}"
        )

    cursor.close()
    connection.close()
    logging.info(
        f"Done in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.time()-time_process))}"
    )


def is_valid_manifest():
    if not os.path.isfile(RESULTS_MANIFEST):
        logging.error(f"Missing manifest file '{RESULTS_MANIFEST}'")
        return False

    df = pd.read_csv(RESULTS_MANIFEST, index_col=False)

    logging.info(f"Validating columns")
    result = True
    for col in [
        "dataset",
        "doi",
        "dataset_lineage",
        "ontology_label",
        "AUCell",
        "GRNBoost",
        "cisTarget",
    ]:
        if col not in df.columns:
            result = False
            logging.error(f"Missing column '{col}' in manifest")
    if not result:
        return False

    result = True
    logging.info(f"Validating manifestfile references")
    for _, row in df.iterrows():
        if not os.path.isfile(row["AUCell"]):
            result = False
            logging.error(f"Missing file for AUCell {row['AUCell']}")
        if not os.path.isfile(row["GRNBoost"]):
            result = False
            logging.error(f"Missing file for GRNBoost {row['GRNBoost']}")
        if not os.path.isfile(row["cisTarget"]):
            result = False
            logging.error(f"Missing file for cisTarget {row['cisTarget']}")
    return result


if __name__ == "__main__":
    assert is_valid_manifest() == True, "Found missing input files. Abort."
    create_db()
    main()
