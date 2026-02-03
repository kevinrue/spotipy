import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

def staggered_grid(img_df, spacing,
                   vertical_spacing=None,
                   include_right_endpoint=True, include_top_endpoint=True):
    """
    Create a staggered grid that fills the bounding box [xmin, xmax] x [ymin, ymax]
    using a horizontal spacing `spacing` between consecutive points.
    Every other row is shifted by half the horizontal spacing.

    Parameters
    ----------
    img_df : dataframe
        Contains pixel coordinates in columns 'row' and 'col'.
    spacing : float
        Horizontal spacing between consecutive points in a full row.
    vertical_spacing : float | None
        Distance between consecutive rows. If None, uses hexagonal packing:
            vertical_spacing = spacing * sqrt(3)/2
    include_right_endpoint, include_top_endpoint : bool
        Whether to include points exactly at xmax / ymax when they fall on the grid.
        (slight numerical tolerance is used.)

    Returns
    -------
    pd.DataFrame
        Columns: x (float), y (float)
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    xmin=min(img_df['col'])
    xmax=max(img_df['col'])
    ymin=min(img_df['row'])
    ymax=max(img_df['row'])
        
    eps = 1e-9
    if vertical_spacing is None:
        vertical_spacing = spacing * np.sqrt(3) / 2.0

    # generate list of y coordinates starting from ymin up to ymax
    if vertical_spacing == 0:
        ys = np.array([(ymin + ymax) / 2.0])
    else:
        if include_top_endpoint:
            ys = np.arange(ymin, ymax + eps, vertical_spacing)
        else:
            ys = np.arange(ymin, ymax - eps, vertical_spacing)

    rows = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            # full row: start at xmin, step spacing
            if include_right_endpoint:
                xs = np.arange(xmin, xmax + eps, spacing)
            else:
                xs = np.arange(xmin, xmax - eps, spacing)
        else:
            # shifted row: start at xmin + spacing/2
            start = xmin + spacing / 2.0
            if include_right_endpoint:
                xs = np.arange(start, xmax + eps, spacing)
            else:
                xs = np.arange(start, xmax - eps, spacing)

        # clip any numerical roundoff
        xs = xs[(xs >= xmin - eps) & (xs <= xmax + eps)]
        # if no points in this shifted row, skip
        if xs.size == 0:
            continue

        for j, x in enumerate(xs):
            rows.append({"x": float(x), "y": float(y)})

    return pd.DataFrame(rows)


def average_rgb_within_radius(centroid_df, img_df, radius,
                              pixel_row_col_names=("row", "col"),
                              centroid_xy_names=("x", "y"),
                              rgb_cols=("R", "G", "B"),
                              min_count=1,
                              n_jobs=1):
    """
    For each centroid in centroid_df compute the mean RGB of img_df pixels
    whose Euclidean distance <= radius (in same coordinate units).
    
    Parameters
    ----------
    centroid_df : pd.DataFrame
        DataFrame with centroid coordinates (columns centroid_xy_names).
    img_df : pd.DataFrame
        DataFrame with pixel coordinates and RGB columns (names pixel_row_col_names + rgb_cols).
        pixel coords are (row, col) which correspond to (y, x).
    radius : float
        Search radius (same coordinate units as coords in dataframes; typically pixels).
    pixel_row_col_names : tuple
        Column names in img_df for row and col, default ("row", "col").
    centroid_xy_names : tuple
        Column names in centroid_df for x and y, default ("x", "y").
    rgb_cols : tuple
        Names of the RGB columns in img_df.
    min_count : int
        Minimum number of pixels required to compute an average; if fewer, result is NaN.
    n_jobs : int
        Number of worker threads passed to cKDTree.query_ball_point (workers param).
    
    Returns
    -------
    pd.DataFrame
        A copy of centroid_df with additional columns: R_mean, G_mean, B_mean, n_pixels.
    """
    # Build pixel coordinate array (y,x) because img_df stores row, col
    row_name, col_name = pixel_row_col_names
    px_coords = np.column_stack((img_df[row_name].to_numpy(dtype=float),
                                 img_df[col_name].to_numpy(dtype=float)))
    # KD-tree on pixels
    tree = cKDTree(px_coords)
    
    # Prepare centroids coordinates as (y,x) to match pixels
    cx_name, cy_name = centroid_xy_names[0], centroid_xy_names[1]
    cent_coords = np.column_stack((centroid_df[cy_name].to_numpy(dtype=float),
                                   centroid_df[cx_name].to_numpy(dtype=float)))
    
    # Query: indices of points within radius for each centroid
    # cKDTree.query_ball_point supports workers for parallelism (SciPy >= 1.6)
    neighbours_idx = tree.query_ball_point(cent_coords, r=radius, workers=n_jobs)
    
    # Prepare arrays for results
    R_mean = np.full(len(centroid_df), np.nan, dtype=float)
    G_mean = np.full(len(centroid_df), np.nan, dtype=float)
    B_mean = np.full(len(centroid_df), np.nan, dtype=float)
    n_pixels = np.zeros(len(centroid_df), dtype=int)
    
    # Extract RGB arrays once
    R_arr = img_df[rgb_cols[0]].to_numpy(dtype=float)
    G_arr = img_df[rgb_cols[1]].to_numpy(dtype=float)
    B_arr = img_df[rgb_cols[2]].to_numpy(dtype=float)
    
    for i, inds in enumerate(neighbours_idx):
        if len(inds) >= min_count:
            n_pixels[i] = len(inds)
            R_mean[i] = R_arr[inds].mean()
            G_mean[i] = G_arr[inds].mean()
            B_mean[i] = B_arr[inds].mean()
        else:
            # leave as NaN and n_pixels = 0 (or set to len(inds) if you prefer)
            n_pixels[i] = len(inds)
    
    out = centroid_df.copy().reset_index(drop=True)
    out["R_mean"] = R_mean
    out["G_mean"] = G_mean
    out["B_mean"] = B_mean
    out["n_pixels"] = n_pixels
    return out
