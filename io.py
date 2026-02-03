from PIL import Image
import numpy as np
import pandas as pd

def load_image_data(file):
    """
    Import image data into a long dataframe.
    Returns a DataFrame with columns: 'row', 'col', 'r', 'g', 'b'.
    """
    img = Image.open(file)
    
    arr = np.array(img)
    
    arr = np.transpose(arr, (2, 0, 1)) # cyx
    
    # reshape to (n_pixels, 3)
    rgb = arr.reshape(3, -1).T

    # pixel coordinates
    y, x = np.indices(arr.shape[1:])

    img_df = pd.DataFrame({
        "row": y.ravel(),
        "col": x.ravel(),
        "R": rgb[:, 0],
        "G": rgb[:, 1],
        "B": rgb[:, 2],
    })
    
    return img_df
