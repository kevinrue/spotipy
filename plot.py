import seaborn as sns
import matplotlib.pyplot as plt


def xy(df, size=5):
    """
    Scatterplot

    Parameters
    ----------
    df : dataframe
    size: float

    Returns
    -------
    None
    Produces a seaborn plot
    """
    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        s=size,
        ax=ax,
        legend=True,
    )

    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.show()


def rgb(df, size=5):
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
    # build Nx3 RGB array and normalise to [0, 1] if needed
    rgb = df[["R_mean", "G_mean", "B_mean"]].to_numpy()

    # if values are 0–255, normalise
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        c=rgb,
        s=size,
        ax=ax,
        legend=True,
    )

    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.show()


def cluster(df, label, size=5, palette="tab20", exclude_labels=[]):
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
    df_sub = df[~df[label].isin(exclude_labels)]

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df_sub,
        x="x",
        y="y",
        hue=label,
        palette=palette,
        s=size,
        ax=ax,
        legend=False,
    )

    ax.set_aspect("equal")
    # ax.legend(
    #     title="label",
    #     bbox_to_anchor=(1.05, 1),
    #     loc="upper left",
    #     borderaxespad=0
    # )
    plt.gca().invert_yaxis()
    plt.show()
    