import scipy as sp
import pandas as pd
import scanpy as sc


class RegulonAtlas:
    def __init__(self, adata: sc.AnnData = None):
        self.adata: sc.AnnData = adata

    def load_from_df(self, df: pd.DataFrame):
        df = df.copy()
        df["regulon"] = (
            df["transcription_factor"].astype(str)
            + " - ("
            + df["regulation"]
            + ") - "
            + df["author_cell_type"]
            + " - "
            + df["tissue"]
        )

        # Create an anndata object from the DataFrame
        val_df = pd.DataFrame(
            [1] * df.shape[0],
            index=pd.MultiIndex.from_frame(df[["regulon", "target_gene"]]),
        ).unstack(fill_value=0)

        adata = sc.AnnData(
            X=sp.sparse.csr_matrix(val_df),
            obs=pd.DataFrame(index=val_df.index.get_level_values("regulon")),
            var=pd.DataFrame(index=val_df.columns.get_level_values("target_gene")),
        )

        # Add obs information to adata
        adata.obs = pd.concat(
            [
                adata.obs,
                df.drop(columns=["target_gene", "coexpression"])
                .drop_duplicates("regulon")
                .set_index("regulon"),
            ],
            axis=1,
        )

        # TODO: add gene information from the database to adata.var
        # TODO: add obsm information to adata for coexpression values

        self.adata = adata

    def get_df(self):
        ad_tf = pd.concat([self.adata.to_df(), self.adata.obs], axis=1)
        ad_tf = ad_tf.melt(
            id_vars=ad_tf.columns[-self.adata.obs.shape[1] :], var_name="target_gene"
        )
        ad_tf = ad_tf[ad_tf["value"] == 1].drop(columns="value")
        return ad_tf

    def embedding(self):
        pass

    def score_gene_set(self, gene_set: list):
        pass

    def perturbation_direction(self, gene_set: list):
        pass
