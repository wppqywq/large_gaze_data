from math import atan2, dist
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image

from constants import *


def gender_str_convert(x: str) -> str:
    if 'FEMALE' in x:
        return 'FEMALE'
    elif 'MALE' in x:
        return 'MALE'
    else:
        return 'OTHER'


def px_to_dva(px: Union[float, np.ndarray, List[Any]]) -> Union[float, np.ndarray]:
    """
    Converts pixel distance to degrees of visual angle
    """

    return 2 * np.degrees(np.arctan((px * PX_PER_CM) / (2 * SCREENDIST)))


def age_to_bin(age: Union[int, float]) -> str:
    for binrange, binstr in zip(BINRANGES, BINSTRINGS):
        if binrange[0] <= age <= binrange[1]:
            return binstr

    return 'None'


def heatmap_from_df(df: pd.DataFrame, blur: bool) -> np.ndarray:
    x = np.asarray(df['avg_x']).round().astype(int)
    y = np.asarray(df['avg_y']).round().astype(int)

    # Clamp coordinates to resolution
    x = np.where(x >= RESOLUTION[0], RESOLUTION[0] - 1, x)
    y = np.where(y >= RESOLUTION[1], RESOLUTION[1] - 1, y)
    x = np.where(x < 0, 0, x)
    y = np.where(y < 0, 0, y)

    # Make heatmap
    heatmap = np.zeros(shape=(RESOLUTION[1], RESOLUTION[0]), dtype=float)
    for x_coord, y_coord in zip(x, y):
        heatmap[y_coord, x_coord] += 1

    if blur:
        heatmap = gaussian_filter(heatmap, sigma=(44.8, 44.8))

    return heatmap


def get_aois_as_single_matrix() -> np.ndarray:
    """
    Similar to get_aois_as_matrices_dict() below, but returns a single array with the AOI NUMBER (0-14) at each pixel.

    To compute AOI stats, loop over gaze coordinates and retrieve the associated value to know on which AOI it was.
    """

    path = ROOT_DIR / 'AOIs'
    im_paths = sorted(list(path.rglob('AOI_*.png')))

    arr = np.zeros(shape=(RESOLUTION[1], RESOLUTION[0]))

    for ip in im_paths:
        # Get the number
        aoi_num = int(ip.stem.split('_')[1])

        # Open image with PIL, convert to grayscale
        im = Image.open(ip).convert('L')

        # Convert to array
        im = np.asarray(im, dtype=np.uint8)

        assert im.shape == (1080, 1920)

        # Set everything to 0 or 1.
        # 1 -> AOI, 0 -> not AOI
        im = np.where(im < 128, aoi_num, 0).astype(np.uint8)

        # Add to arr, but only where there isn't already a value besides 0, otherwise we get annoying overlap issues.
        # These overlaps only occur along some edges, shouldn't matter much.
        arr = np.where(arr == 0, im, arr)

    # Visualize values within the array
    # import seaborn as sns
    # plt.figure(figsize=(8, 4.5))
    # sns.heatmap(arr, square=True, cbar=False)
    # plt.ylim(1080, 0)
    # plt.axis('off')
    # plt.tight_layout(pad=.05)
    # plt.savefig(ROOT_DIR / 'AOIs' / 'all_heatmap.png', dpi=300)
    # plt.show()

    return arr


def get_aois_as_matrices_dict() -> dict[str : np.ndarray]:
    """
    Loads black and white images from the AOI directory as boolean numpy arrays
    (True -> coordinate falls within this AOI; False -> outside of this AOI).

    These arrays are returned within a python dictionary,
    in which the dictionary key is the AOI name based on the filename.

    To compute AOI stats, simply loop over gaze coordinates or fixations,
    and then find in which array the value is True at that location.

    Note that numpy array values should be retrieved using arr[y, x] when using gaze coordinates.
    """
    path = ROOT_DIR / 'AOIs'
    im_paths = sorted(list(path.rglob('AOI_*.png')))

    arrs = {}

    for ip in im_paths:
        # Get the name
        aoi_name = ip.stem.split('_')[-1]

        # Open image with PIL, convert to grayscale
        im = Image.open(ip).convert('L')

        # Convert to array
        im = np.asarray(im, dtype=np.uint8)

        assert im.shape == (1080, 1920)

        # Set everything to 0 or 1.
        # 1 -> AOI, 0 -> not AOI
        im = np.where(im < 128, 1, 0).astype(np.bool)

        # Add to dict
        arrs[aoi_name] = im

    # Define the 'None' AOI, which is everything outside of the defined AOIs
    arr = np.zeros(shape=im.shape, dtype=int)
    for k, im in arrs.items():
        arr += im.astype(int)

    # Invert the array, such that this has True for everywhere there is no existing AOI
    arr = np.where(arr == 0, 1, 0)

    # Add to dict
    arrs['none'] = arr.astype(np.bool)

    return arrs


