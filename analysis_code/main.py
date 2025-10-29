from pathlib import Path
from typing import List, Tuple, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis_code.helperfunctions import age_to_bin
from constants import *
from analysis_code.plots import plot_timeseries
from analysis_code.dataloader_helpers import (
    check_for_valid_gaze_task,
    check_for_valid_gaze_overall,
    format_single_df,
    get_filepaths_csv,
    get_filepaths_txt,
    get_participant_info,
    load_file_as_df,
    pp_info_add_valid,
    filter_data_valid,
)
from analysis_code.fixation_detection import detect_fixations_I2MC


def summarize_data(task: Literal['search', 'freeviewing']):
    pp_info = pd.read_csv(INTERM_DATA_DIR / task / 'participant_info.csv')
    # pp_info = pp_info.loc[pp_info['Task'] == task]

    valid_task = pp_info.loc[pp_info['Valid task'] == True]
    valid_demo = pp_info.loc[(pp_info['Valid demographics'] == True) & (pp_info['Valid task'] == True)]

    # Data exclusion
    print(
        'Dataset exclusion\n'
        f'Total:              {len(pp_info)}\n'
        f'Valid task:         {len(valid_task)} ({round((len(valid_task) / len(pp_info)) * 100, 2)}%)\n'
        f'Valid demographics: {len(valid_demo)} ({round((len(valid_demo) / len(pp_info)) * 100, 2)}%)\n'
    )

    # Summary
    gender = valid_demo.groupby('Gender').size()
    gender_perc = (gender / gender.sum()) * 100
    print(
        f'Summary of data with valid demographics\n'
        f'N:         {len(valid_demo)}\n'
        f'Age range: {np.min(valid_demo["Age"])}-{np.max(valid_demo["Age"])}\n'
        f'Age mean:  {np.nanmean(valid_demo["Age"]).round(2)}\n'
        f'Age SD:    {np.nanstd(valid_demo["Age"]).round(2)}\n'
        f'Female:    {gender["FEMALE"]} ({gender_perc["FEMALE"].round(2)}%)\n'
        f'Male:      {gender["MALE"]} ({gender_perc["MALE"].round(2)}%)\n'
        f'Other:     {gender["OTHER"]} ({gender_perc["OTHER"].round(2)}%)\n'
    )

    # Age bins
    valid_demo['Bin'] = valid_demo['Age'].apply(age_to_bin)
    valid_demo = valid_demo.loc[valid_demo['Bin'] != 'None']
    agebin = valid_demo.groupby('Bin').size()
    agebin_perc = (agebin / agebin.sum()) * 100

    print(f'Summary of age bins (6-59, N = {agebin.sum()})')
    for bs in BINSTRINGS:
        print(f'{bs} N: {agebin[bs]} ({agebin_perc[bs].round(2)}%)')


def check_isi(task: Literal['search', 'freeviewing']):
    # Check inter-sample interval
    dfs, files = load_data(pre_process=False, task=task)

    isi = []
    for df in dfs:
        ts = np.array(df['time_from_start'])
        diffs = np.diff(ts)
        diffs = diffs[diffs < 0.03]  # Remove large gaps, because those are not ISIs but just missing data
        isi.append(np.nanmean(diffs))

    print('\n' '-----------------------' '\n' 'INTER-SAMPLE INTERVALS:')

    isi = np.array(isi)
    print(np.mean(isi), 'mean')
    print(np.min(isi), 'min')
    print(np.max(isi), 'max')

    print('-----------------------\n')


def process_plots(dfs: List[pd.DataFrame], files: List[Path]) -> None:
    print('Making event detection plots')

    # DEBUG
    dfs = dfs[:50]
    files = files[:50]

    # Change folder and filetype in the filename specification
    if '/raw_data' in str(files[0]):
        files = [str(f).replace('/raw_data', '/processed_data') for f in files]
        files = [str(f).replace('.txt', '.csv') for f in files]
    elif '/fixation_data' in str(files[0]):
        files = [str(f).replace('/fixation_data', '/processed_data') for f in files]

    # Make timeseries plots. Only show in IDE if it's < 30 files
    for df, file in zip(dfs, files):
        plot_timeseries(df, file, show=True if len(dfs) < 30 else False)


def multiproc_dataloader(files: List[Path]) -> List[pd.DataFrame]:
    print('Loading data')
    dfs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(load_file_as_df)(f) for f in files)
    return dfs


def multiproc_format_df(
    dfs: List[pd.DataFrame], files: List[Path], save_to_csv: bool, task: Literal['search', 'freeviewing']
) -> List[pd.DataFrame]:
    print('Formatting dataframes')

    # saveto = [save_to_csv] * len(dfs)
    dfs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(
        delayed(format_single_df)(df, f, save_to_csv, task) for df, f in zip(dfs, files)
    )

    return dfs


def multiproc_valid_gaze(dfs: List[pd.DataFrame]) -> List[bool]:
    print('Checking for valid gaze')
    valids = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(
        delayed(check_for_valid_gaze_overall)(df) for df in dfs
    )
    return valids


def multiproc_valid_gaze_fv(dfs: List[pd.DataFrame], task: Literal['search', 'freeviewing']) -> List[bool]:
    print('Checking for valid gaze during task')
    valids = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(
        delayed(check_for_valid_gaze_task)(df, task) for df in dfs
    )
    return valids


def load_data(
    pre_process=True, task: Literal['search', 'freeviewing'] = 'search'
) -> Tuple[List[pd.DataFrame], List[Path]]:
    if pre_process:
        files = get_filepaths_txt(RAW_DATA_DIR / task)

        # DEBUG
        # files = files[:10]

        # Load data and retrieve participant info
        dfs = multiproc_dataloader(files)
        pp_info = get_participant_info(dfs, files)

        # Format files and save immediately
        dfs = multiproc_format_df(dfs, files, save_to_csv=True, task=task)

        # Check whether each dataset is valid and add it to the pp_info dataframe
        valid_fv_list = multiproc_valid_gaze_fv(dfs, task=task)
        valid_list = multiproc_valid_gaze(dfs)
        pp_info_add_valid(pp_info, valid_list, valid_fv_list, task=task)

    else:
        # Load already pre-processed data
        files = get_filepaths_csv(PROC_DATA_DIR / task)

        dfs = multiproc_dataloader(files)

    return dfs, files


def main(task: Literal['search', 'freeviewing'] = 'search') -> None:
    print(f'Running code for {task}...\n' '----------------------------')

    # Load and process. Set pre_process=False if preprocessing has already been done (saves time)
    dfs, files = load_data(pre_process=True, task=task)
    # dfs, files = load_data(pre_process=False, task=task)

    summarize_data(task=task)

    # Filter data
    dfs, files = filter_data_valid(dfs, files, filter_on='task', task=task)

    # Concatenate files
    for df, file in zip(dfs, files):
        df['ID'] = [file.stem] * len(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[(df['label'] == 'p') & (df['time_from_start'] <= 10)]
    df = df.drop([x for x in list(df.columns) if x not in ['ID', 'x', 'y', 'time_from_start']], axis=1)
    print('Saving compiled data')
    df.to_csv(INTERM_DATA_DIR / task / 'compiled_gaze.csv')

    # Check the inter-sample intervals, just in case
    check_isi(task)

    # Detect and save fixations
    detect_fixations_I2MC(concatenate_after=True, skip_existing=True, task=task)

    # Plot (samples of) event detection
    # process_plots(dfs, files)


if __name__ == '__main__':
    main(task='search')
    main(task='freeviewing')
