from typing import Literal
import numpy as np
import pandas as pd
from I2MC import I2MC
from constants import *
from analysis_code.dataloader_helpers import filter_data_valid


def _detect_fixations_I2MC_helper(
    ID: str, skip_existing=True, task: Literal['search', 'freeviewing'] = 'search'
) -> bool:
    new_path = FIX_DATA_DIR / task / f'{ID}.csv'

    if new_path.exists() and skip_existing:
        print(f'Skipping fixation detection for {ID}')
        return None

    df = pd.read_csv(PROC_DATA_DIR / task / f'{ID}.csv')

    if len(df) == 0:
        return None

    df = df.loc[(df['label'] == 'p') & (df['time_from_start'] < 10)]
    df = df.rename({'time_from_start': 'time', 'x': 'average_X', 'y': 'average_Y'}, axis=1)
    df['time'] = df['time'] * 1000
    df = df.drop([x for x in list(df.columns) if x not in ['time', 'average_X', 'average_Y']], axis=1)

    options = {
        'xres': RESOLUTION[0],
        'yres': RESOLUTION[1],
        'freq': 60,
        'missingx': np.nan,
        'missingy': np.nan,
        'scrSz': DISPSIZE,
        'disttoscreen': SCREENDIST,
        'maxMergeDist': 44.8,  # approx 1 deg
        'minFixDur': 60,
        'downsampFilter': False,
        'maxerrors': 200,
        'downsamples': [],
    }

    try:
        i2mc_res = I2MC(gazeData=df, options=options, logging=False)

        fixations = pd.DataFrame(i2mc_res[0])

        fixations = fixations.drop(
            ['cutoff', 'start', 'end', 'flankdataloss', 'fracinterped', 'RMSxy', 'BCEA', 'fixRangeX', 'fixRangeY'],
            axis=1,
        )

        fixations = fixations.rename(
            {'startT': 'onset', 'endT': 'offset', 'dur': 'duration', 'xpos': 'avg_x', 'ypos': 'avg_y'}, axis=1
        )

        fixations['onset'] = fixations['onset'] / 1000
        fixations['offset'] = fixations['offset'] / 1000
        fixations['duration'] = fixations['duration'] / 1000

        fixations['ID'] = [ID] * len(fixations)
        fixations['label'] = ['FIXA'] * len(fixations)

        fixations.to_csv(new_path)

        return True

    except Exception as e:
        print(ID, e)
        return False


def detect_fixations_I2MC(
    concatenate_after=True, skip_existing=True, task: Literal['search', 'freeviewing'] = 'search'
):
    path = PROC_DATA_DIR / task
    files = sorted(list(path.rglob('*.csv')))

    files = filter_data_valid(files=files, filter_on='task', task=task)

    IDs = [f.stem for f in files]

    if N_JOBS == 1:
        success = []
        for i, ID in enumerate(IDs):
            # print(f'Detecting fixations for {i + 1} of {len(IDs)} ({ID})')
            success.append(_detect_fixations_I2MC_helper(ID, skip_existing, task))

    else:
        from joblib import Parallel, delayed

        success = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(
            delayed(_detect_fixations_I2MC_helper)(ID, skip_existing, task) for ID in IDs
        )

    # Retrieve how often True or False were returned
    success = np.asarray(success)
    success = success[success != None]
    print(f'Fixation detection succeeded for {np.sum(success)} of {len(success)} datasets')

    if concatenate_after:
        # We skipped existing files we should reload all data, otherwise we'll miss part of it in the concatenated file
        path = FIX_DATA_DIR / task
        files = sorted(list(path.rglob('*.csv')))
        fixations = [pd.read_csv(f) for f in files]
        fixations = pd.concat(fixations, ignore_index=True)
        fixations.to_csv(INTERM_DATA_DIR / task / 'compiled_fixations.csv')

        # Fixation summary statistics
        fg = fixations.groupby('ID').size()
        print(f'{round(np.nanmean(fg) / 10, 2)} fixations per second, averaged across participants')


if __name__ == '__main__':
    detect_fixations_I2MC(concatenate_after=True, skip_existing=False, task='search')
    detect_fixations_I2MC(concatenate_after=True, skip_existing=False, task='freeviewing')
