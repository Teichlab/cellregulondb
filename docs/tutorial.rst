Tutorial
============

Getting started
---------------

Download the latest database
++++++++++++++++++++++++++++

Use dafault naming when dowloading the database (``cellregulon-v[VERSION].db)``:
    .. code-block:: python

        import cellregulondb as crdb
        db_file = crdb.download_db()
        print(db_file)

Save with specific name:
    .. code-block:: python

        import cellregulondb as crdb
        crdb.download_db(db_path="cellregulon.db")
        print(db_file)


Loading the database
++++++++++++++++++++

After downloading the database, load it using the ``CellRegulonDB`` class and print all the transcription factors.

    .. code-block:: python

        import cellregulondb as crdb
        db = crdb.CellRegulonDB("cellregulon.db")
        print(db.TFs)


Querying the databse
++++++++++++++++++++

After downloading the database, load it using the ``CellRegulonDB`` class and print all the transcription factors.

    .. code-block:: python

        import cellregulondb as crdb
        db = crdb.CellRegulonDB("cellregulon.db")
        regulons = db.get_regulon(["SRY","NFIB"])
        print(regulons)

