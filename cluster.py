import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
