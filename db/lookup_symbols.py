#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")

import json
import sqlite3
import logging
import requests
import pandas as pd

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")


###########################################################
# Config section
###########################################################

# DB name
DB_PATH = f"cellregulon_2023-06-22.db"

# csv output path
CSV_PATH = f"gene-match-cellregulon.csv"

# update database or only generate output csv report
UPDATE_DATABASE = True

# fetch symbols and update db
# TODO: split this into 2 functions to make it cleaner?
def fetch_symbols():
    # open db connection
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # make everything fast — what? that it may lead to corruption in the DB? oh well...
    connection.execute("PRAGMA synchronous  = OFF")
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA temp_store   = memory")
    connection.execute("PRAGMA mmap_size    = 30000000000")

    # get all symbols from the DB
    all_symbols = cursor.execute(
        "SELECT id, name, properties->>'is_TF' FROM nodes"
    ).fetchall()
    match_results = []

    if not UPDATE_DATABASE:
        cursor.close()
        connection.close()

    # setup both GRCh38 and GRCh37 endpoints and headers
    GRCH38_ENSEMBL_API = "https://rest.ensembl.org/lookup/symbol/homo_sapiens"
    GRCH37_ENSEMBL_API = "https://grch37.rest.ensembl.org/lookup/symbol/homo_sapiens"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    # first pass will try to match symbol names to GRCh38
    total = len(all_symbols)
    good = 0
    bad = 0
    failed = []
    # batch genes 1000 at the time
    for i in range(0, len(all_symbols), 999):
        batch_symbols = all_symbols[i : i + 999]
        response = requests.post(
            GRCH38_ENSEMBL_API,
            headers=HEADERS,
            json={"symbols": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for symbol in batch_symbols:
                if symbol[1] in data.keys():
                    if UPDATE_DATABASE:
                        node = cursor.execute(
                            "SELECT id, properties FROM nodes WHERE id = ?", [symbol[0]]
                        ).fetchone()
                        props = json.loads(node[1])
                        props["symbol_information"] = data[symbol[1]]
                        cursor.execute(
                            "UPDATE nodes SET properties = ? WHERE id = ?",
                            [json.dumps(props), node[0]],
                        )
                    good += 1
                    match_results.append(
                        {"gene_in_db": symbol[1], "is_TF": symbol[2], **data[symbol[1]]}
                    )
                else:
                    bad += 1
                    failed.append(symbol)
            logging.info(f"[Grch38] Match={good}. Unmatched={bad}. Total={total}.")
        else:
            logging.error("[Grch38] POST failed", response.content)
        if UPDATE_DATABASE:
            connection.commit()
    logging.info(f"[Grch38] Finished matching symbols")

    # second pass will try to match unmatched symbol names to GRCh37
    logging.info(f"Running failed symbols {len(failed)} with Grch37.")
    total = len(failed)
    good = 0
    bad = 0
    for i in range(0, len(failed), 999):
        batch_symbols = failed[i : i + 999]
        response = requests.post(
            GRCH37_ENSEMBL_API,
            headers=HEADERS,
            json={"symbols": [s[1] for s in batch_symbols]},
        )
        if response.ok:
            data = response.json()
            for symbol in batch_symbols:
                if symbol[1] in data.keys():
                    if UPDATE_DATABASE:
                        node = cursor.execute(
                            "SELECT id, properties FROM nodes WHERE id = ?", [symbol[0]]
                        ).fetchone()
                        props = json.loads(node[1])
                        props["symbol_information"] = data[symbol[1]]
                        cursor.execute(
                            "UPDATE nodes SET properties = ? WHERE id = ?",
                            [json.dumps(props), node[0]],
                        )
                    good += 1
                    match_results.append(
                        {"gene_in_db": symbol[1], "is_TF": symbol[2], **data[symbol[1]]}
                    )
                else:
                    bad += 1
                    match_results.append({"gene_in_db": symbol[1], "is_TF": symbol[2]})
            logging.info(f"[GRCh37] Match={good}. Unmatched={bad}. Total={total}.")
        else:
            logging.error("[GRCh37] POST fail", response.content)

        if UPDATE_DATABASE:
            connection.commit()

    # gracefuly close everything
    if UPDATE_DATABASE:
        cursor.close()
        connection.close()

    # return dataframe with all match results
    return pd.DataFrame.from_records(match_results)


if __name__ == "__main__":
    result = fetch_symbols()
    logging.info(f"Finished looking up symbols")

    df = pd.DataFrame.from_records(result)
    total = df.shape[0]
    matched = df[pd.isnull(df["id"])].shape[0]
    unmatched = df[~pd.isnull(df["id"])].shape[0]
    tf_total = df[df["is_TF"] == 1].shape[0]
    tf_matched = df[~pd.isnull(df["id"]) & df["is_TF"] == 1].shape[0]
    tf_unmatched = df[pd.isnull(df["id"]) & df["is_TF"] == 1].shape[0]

    logging.info(f"[TF] matched={tf_matched}")
    logging.info(f"[TF] unmatched={tf_unmatched}")
    logging.info(f"[TF] total={tf_total}")

    logging.info(f"[all] matched={matched} ({round(matched*100/total,2)}%)")
    logging.info(f"[all] unmatched={unmatched} ({round(unmatched*100/total,2)}%)")
    logging.info(f"[all] total={total}")

    logging.info(f"Saving results as '{CSV_PATH}'")
    df.to_csv(CSV_PATH, index=False)

    logging.info(f"Done")
