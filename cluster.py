import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition

def kmeans(df):
    """
    Scatterplot coloured by average colour

    Parameters
    ----------
    df : dataframe
    size: float

    Returns
    -------
    None
    Produces a seaborn plot
    """    
    rgb = df[["R_mean", "G_mean", "B_mean"]].to_numpy()

    # optional: normalise if values are 0–255
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto")
    df["clusters_kmeans"] = pd.Categorical(kmeans.fit_predict(rgb))
    return df


def banksy(df, k_geom=10, lambda_list=[0.3], resolution=0.2):
    # df has columns: x, y, R_mean, G_mean, B_mean

    # extract RGB matrix (observations × variables)
    X = df[["R_mean", "G_mean", "B_mean"]].to_numpy(dtype=float)

    # optional: make sparse (often unnecessary for RGB, but AnnData-compatible)
    X = csr_matrix(X)

    # create AnnData
    adata_banksy = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=["R", "G", "B"])
    )

    # store spatial coordinates (x, y)
    adata_banksy.obsm["spatial"] = df[["x", "y"]].to_numpy(dtype=float)
    
    banksy_dict = initialize_banksy(
        adata_banksy,
        ('xcoord', 'ycoord', 'spatial'),
        k_geom,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False, #turn these on to enable diagnostic plotting - some of these opts take a long time though
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    
    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata_banksy,
        banksy_dict,
        lambda_list,
        1
    )
    
    pca_umap(
        banksy_dict,
        pca_dims=[9],
        add_umap=True,
        plt_remaining_var=False,
    )
    
    results_df, max_num_labels = run_Leiden_partition(
        banksy_dict,
        [resolution],
        num_nn=20,
        num_iterations=-1,
        partition_seed=1234,
        match_labels=True,
    )
    
    df['clusters_banksy'] = results_df.relabeled[0].dense
    df['clusters_banksy'] = df['clusters_banksy'].astype("category")
    
    return df
