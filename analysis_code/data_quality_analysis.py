from typing import List, Union, Any, Literal
import numpy as np
from pathlib import Path
from math import dist

import pandas as pd

from constants import *


def px_to_dva(px: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts pixel distance to degrees of visual angle
    """
    return px / 44.8


def rms_s2s(x: np.ndarray, y: np.ndarray) -> float:
    # Computes the sample-to-sample root median (instead of mean) squared displacement.
    # Using median instead of means accounts for saccades, instead of having to filter those out.
    # https://link.springer.com/article/10.3758/s13428-020-01400-9

    # Returns n-1 sample-to-sample diffs (x_i+1 - x_i)
    xdiff = np.diff(x) ** 2
    ydiff = np.diff(y) ** 2

    # Sum the two arrays (length of disp is still n-1)
    disp = xdiff + ydiff

    # Square root of the mean
    rms = np.sqrt(np.nanmedian(disp))

    return rms


def compute_rms(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rms = rms_s2s(x, y)
    rms_dva = px_to_dva(rms)

    return rms, rms_dva


def main(task: Literal['search', 'freeviewing'] = 'search'):
    pp_info = pd.read_csv(INTERM_DATA_DIR / task / 'participant_info.csv')
    valid_IDs = list(pp_info.loc[pp_info['Valid task'] == True]['ID'])

    df = pd.read_csv(INTERM_DATA_DIR / task / 'compiled_gaze.csv')
    df = df.loc[df['ID'].isin(valid_IDs)]

    # NEMO
    results = {
        'Dataset': [],
        'ID': [],
        'trial': [],
        'RMS_px': [],
        'RMS_dva': [],
    }

    for ID in list(df['ID'].unique()):
        dfid = df.loc[df['ID'] == ID]

        t, x, y = np.asarray(dfid['time_from_start']), np.asarray(dfid['x']), np.asarray(dfid['y'])
        rms, rms_dva = compute_rms(x, y)

        results['Dataset'].append('NEMO')
        results['ID'].append(ID)
        results['trial'].append(np.nan)
        results['RMS_px'].append(rms)
        results['RMS_dva'].append(rms_dva)

    results = pd.DataFrame(results)
    rg = results.agg({'RMS_px': [np.nanmean, np.nanstd], 'RMS_dva': [np.nanmean, np.nanstd]})
    print(f'Task: {task}\n', rg)


if __name__ == '__main__':
    main(task='freeviewing')
    main(task='search')
