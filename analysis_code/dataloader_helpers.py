import time
from datetime import datetime
from pathlib import Path
from random import sample
from typing import Any, List, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from I2MC import I2MC

import constants
from constants import *
from helperfunctions import gender_str_convert


def get_filepaths_txt(data_path: Path) -> List[Path]:
    files = sorted(list(data_path.glob('*.txt')))
    print(f'{len(files)} .txt files found')

    return files


def get_filepaths_csv(data_path: Path) -> List[Path]:
    files = sorted(list(data_path.glob('*.csv')))
    print(f'{len(files)} .csv files found')

    return files


def load_file_as_df(file: Path) -> pd.DataFrame:
    # Load file as .txt or as .csv
    try:
        if file.suffix == '.txt':
            with open(file, 'r') as f:
                text = f.read()

            text = text.split('\n')
            df = pd.DataFrame(text)

        elif file.suffix == '.csv':
            df = pd.read_csv(file)

        else:
            df = None
            print(f'Cannot load {file.suffix} filetype')

        return df

    except Exception as e:
        print(f'Error loading {str(file)}: {e}')


def save_dataframes_as_csv(dfs: List[pd.DataFrame], filenames: List[Path], suffix: str = '') -> None:
    for df, f in zip(dfs, filenames):
        df.to_csv(f'{str(f).replace(".txt", "").replace("raw_data", "processed_data")}{suffix}.csv')


def _split_column(x: str) -> List[Any]:
    """
    p,(144.9, 39.1),20230309153847840,172840254586,True,1387.806,630.7996,True,2.817505,True,1557.035,635.3011,True,2.592422
    Explanation: 0 [identifier],
                1([gaze position x],
                2[gaze position y]),
                3[timestamp],
                4[DeviceTimeStamp],
                5[LeftEye_GazePoint_Validity],
                6[LeftEye_GazePoint_PositionOnDisplayArea_X],
                7[LeftEye_GazePoint_PositionOnDisplayArea_Y],
                8[LeftEye_Pupil_Validity],
                9[LeftEye_Pupil_PupilDiameter],
                10[RightEye_GazePoint_Validity],
                11[RightEye_GazePoint_PositionOnDisplayArea_X],
                12[RightEye_GazePoint_PositionOnDisplayArea_Y],
                13[RightEye_Pupil_Validity],
                14[RightEye_Pupil_PupilDiameter]
    :param x:
    :return: new row:
            0 label
            1 'x_old',
            2 'y_old',
            3 'x',
            4 'y',
            5 'timestamp',
            6 'event',
            7 'message',
            8 'valid_gaze',
            9 'pupil_size'
    """

    x = str(x)

    x = x.replace('(', '')
    x = x.replace(')', '')
    row = x.split(',')

    # Some rows are short or empty
    if len(row) < 3:
        return [np.nan] * 10

    # If row contains event, keep x/y columns open
    # (i.e., [e, nan, nan, nan, nan timestamp, message, nan, nan])
    if row[0] == 'e':
        timestamp = row[2]
        split_message = row[1].split(':')
        return ['e', np.nan, np.nan, np.nan, np.nan, timestamp, split_message[0], split_message[1], np.nan, np.nan]
    else:
        # Check whether gaze point is valid, otherwise assign nan's
        timestamp = row[3]

        old_x = float(row[1])
        old_y = float(row[2])

        xgaze = int(row[6])
        ygaze = constants.RESOLUTION[1] - int(row[7])  # Flip the y-origin for the raw signal
        pupil = float(row[9])

        # Gaze is out of bounds, data is invalid
        if ygaze < 0 or ygaze > RESOLUTION[1] or xgaze < 0 or xgaze > RESOLUTION[0]:
            return ['p', old_x, old_y, np.nan, np.nan, timestamp, '', '', False, np.nan]

        # Signal is good (column 5 indicates data loss). We already caught the gaze out of bounds condition above
        if row[5] == 'True':
            return ['p', old_x, old_y, xgaze, ygaze, timestamp, '', '', True, pupil]

        # Data loss
        else:
            return [row[0], old_x, old_y, np.nan, np.nan, timestamp, '', '', False, np.nan]


def format_single_df(df: pd.DataFrame, filename, save_to_csv, task: Literal['search', 'freeviewing']) -> pd.DataFrame:
    if task == 'search':
        df = df.iloc[7:]
    else:
        df = df.iloc[5:]

    # Now separate by commas
    new_columns = [_split_column(x) for x in list(df.iloc[:, 0])]
    new_df = pd.DataFrame(
        new_columns,
        columns=['label', 'x_old', 'y_old', 'x', 'y', 'timestamp', 'event', 'message', 'valid_gaze', 'pupil_size'],
    )

    # Add more convenient timestamps
    new_df['datetime'] = new_df['timestamp'].apply(convert_timestamp, args=(False,))
    new_df['unix_time'] = new_df['timestamp'].apply(convert_timestamp)

    start_time = list(new_df['unix_time'])[0]
    new_df['time_from_start'] = new_df['unix_time'].apply(lambda x: x - start_time)

    # Drop the filtered gaze coordinates
    new_df = new_df.drop(['x_old', 'y_old'], axis=1)

    if save_to_csv:
        assert filename is not None, 'format_single_df(): Cannot save to csv if no filenames are supplied'
        save_dataframes_as_csv([new_df], [filename])

    return new_df


def get_participant_info(dfs: List[pd.DataFrame], files: [List[Path]]) -> pd.DataFrame:
    def resp_to_shape(x) -> str:
        if int(x) == 0:
            return 'plus'
        elif int(x) == 1:
            return 'x'
        else:
            return 'Nothing'

    info = {
        'ID': [],
        'Gender': [],
        'DoB': [],
        'Age': [],
        'Task': [],
        'Image': [],
        'Target location': [],
        'Shape': [],
        'Response': [],
        'Response outcome': [],
        'Hit': [],
        'False Alarm': [],
        'Nothing found': [],
    }

    for df, file in zip(dfs, files):
        ID = str(Path(file).stem)
        dob = int(df.iloc[1, 0])

        info['ID'].append(ID)
        info['Gender'].append(gender_str_convert(df.iloc[0, 0]))
        info['DoB'].append(dob)

        curr_year = int(ID[:4])
        curr_month = int(ID[4:6])
        curr_day = int(ID[6:8])

        # Because we don't know pp's birth month (only year), we just assume that everyone's birthday is
        # halfway through the year: July 1st. If before July, set the current year one year back
        if curr_month < 7:
            info['Age'].append((curr_year - 1) - dob)
        else:
            info['Age'].append(curr_year - dob)

        if int(ID[:8]) > 20240824:
            # Task changed from freeviewing to search after the 24th of august 2024.
            # File format in the first rows also changed after that.
            # 0: Gender, 1: DoB, 2: image (contains location and shape), 3: response, 4-6: ROIs
            task = 'search'
            im = df.iloc[2, 0]
            loc = int(im.split('_')[0][-1])
            shape = im.split('_')[-1]
            resp = resp_to_shape(df.iloc[3, 0])

            if resp == 'Nothing':
                hit, fa, nothing, resp_outcome = False, False, True, 'Nothing found'
            elif resp == shape:
                hit, fa, nothing, resp_outcome = True, False, False, 'Hit'
            elif resp != shape:
                hit, fa, nothing, resp_outcome = False, True, False, 'False Alarm'
            else:
                hit, fa, nothing, resp_outcome = np.nan, np.nan, np.nan, np.nan

            info['Task'].append(task)
            info['Image'].append(im)
            info['Target location'].append(loc)
            info['Shape'].append(shape)
            info['Response'].append(resp)
            info['Response outcome'].append(resp_outcome)

            info['Hit'].append(hit)
            info['False Alarm'].append(fa)
            info['Nothing found'].append(nothing)

        else:
            # For freeviewing, append nan's to all search-related variables
            # 0: Gender, 1: DoB, 2-4: ROIs
            task = 'freeviewing'
            info['Task'].append(task)
            info['Image'].append(np.nan)
            info['Target location'].append(np.nan)
            info['Shape'].append(np.nan)
            info['Response'].append(np.nan)
            info['Response outcome'].append(np.nan)
            info['Hit'].append(np.nan)
            info['False Alarm'].append(np.nan)
            info['Nothing found'].append(np.nan)

    info = pd.DataFrame(info)

    info.to_csv(INTERM_DATA_DIR / task / 'participant_info.csv')

    return info


def pp_info_add_valid(info: pd.DataFrame, valid, valid_fv, task: Literal['search', 'freeviewing']) -> None:
    info['Valid demographics'] = valid
    info['Valid task'] = valid_fv

    info.to_csv(INTERM_DATA_DIR / task / 'participant_info.csv')


def convert_timestamp(timestamp: str, as_unix: bool = True) -> Union[datetime, float]:
    # Timestamps should be strings, e.g., '20220129125024936' -> '2022/01/29 12:50:24 936ms'
    timestamp = str(timestamp)

    if timestamp == 'nan' or timestamp == 'None':
        return np.nan

    year, month, day = timestamp[0:4], timestamp[4:6], timestamp[6:8]
    hour, minute, second, ms = timestamp[8:10], timestamp[10:12], timestamp[12:14], timestamp[14:17]

    # datetime timestamp
    date_time = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), int(ms) * 1000)

    # Unix timestamp
    unix_time = date_time.timestamp()

    if as_unix:
        return unix_time
    else:
        return date_time


def check_for_valid_gaze_overall(df: pd.DataFrame):
    # Check whether there is a period of invalid gaze the size of INVALID_WINDOW_OVERALL.
    # This means someone likely left the eyetracker and we can't trust the demographics
    valid = list(df['valid_gaze'])

    messages = np.array(df['message'])

    # Define a sample window. E.g., 5 seconds, which translates to 300 gaze samples (5000ms / 16.6666ms per sample)
    sample_window = int(INVALID_WINDOW_OVERALL / (1000 / SAMPLING_RATE))

    i = 0
    while i < len(valid) - sample_window:
        window_start = i
        window_end = i + sample_window

        # Mark the end of the experiment
        if 'ScreenVideoFeedback' in messages[window_start:window_end]:
            break

        # There is a window of size sample_window which only contains False; so someone likely left the tracker
        if True not in valid[window_start:window_end]:
            return False

        i += 1

    return True


def check_for_valid_gaze_task(df: pd.DataFrame, task: Literal['search', 'freeviewing']):
    # Check whether there is a period of invalid gaze the size of INVALID_WINDOW_TASK.
    # This means someone likely left the eyetracker, so we can't trust the search task data
    start = 0

    # Check which row marks the end of search task
    events = np.array(df['event'])
    messages = np.array(df['message'])

    if task == 'search':
        end = np.argwhere((events == 'ScreenStart') & (messages == 'ScreenQuestionX')).ravel()[0]
    else:
        try:
            end = np.argwhere((events == 'ScreenStart') & (messages == 'ScreenInstructionPostGame')).ravel()[0]
        except IndexError:
            return False

    # Column with True/False
    valid = list(df['valid_gaze'].astype(bool))[start:end]

    # If there's more TOTAL missing rows of data than INVALID_SUM_TASK allows, mark as False.
    # 600 is the default number of rows (10 seconds, 60Hz)
    if np.sum(valid) < 600 - (INVALID_SUM_TASK / (1000 / SAMPLING_RATE)):
        return False

    # Check for window of continuous loss.
    # Define a sample window. E.g., 1 second, which translates to 60 gaze samples (1000ms / 16.6666ms per sample)
    sample_window = int(INVALID_WINDOW_TASK / (1000 / SAMPLING_RATE))
    i = 0
    while i < len(valid) - sample_window:
        window_start = i
        window_end = i + sample_window

        # There is a window of size sample_window which only contains False; so someone likely left the tracker
        if True not in valid[window_start:window_end]:
            return False

        i += 1

    return True


def filter_data_valid(
    dfs: List[pd.DataFrame] = None,
    files: List[Path] = None,
    filter_on: Literal['task', 'demographics'] = 'task',
    task: Literal['search', 'freeviewing'] = 'search',
):
    pp_info = pd.read_csv(INTERM_DATA_DIR / task / 'participant_info.csv')

    len_before = len(pp_info)
    pp_info = pp_info.loc[pp_info[f'Valid {filter_on}'] == True]
    len_after = len(pp_info)
    print(
        f'Remaining: {len_after} of {len_before} datasets after filtering for invalid {filter_on} '
        f'({round(((len_before - len_after) / len_before) * 100, 1)}%)'
    )

    pp_info['ID'] = pp_info['ID'].astype(str)
    valid_IDs = list(pp_info['ID'].unique())

    # A list of dataframes was supplied
    if isinstance(dfs, list):
        new_dfs = []
        new_files = []

        if files is None:
            print('filter_data_valid: if supplying dfs as list, a list of filenames should be supplied')
            exit()

        for df, file in zip(dfs, files):
            ID = Path(file).stem
            if ID in valid_IDs:
                new_dfs.append(df)
                new_files.append(file)

        return new_dfs, new_files

    # A single dataframe was supplied
    elif isinstance(dfs, pd.DataFrame):
        assert 'ID' in list(dfs.columns), 'filter_data_valid: dataframe should contain an "ID" column'
        new_df = dfs.loc[dfs['ID'].isin(valid_IDs)]
        return new_df

    # A list of filenames was supplied, without a list of dataframes
    elif dfs is None:
        if files is not None:
            new_files = []
            for file in files:
                ID = Path(file).stem
                if ID in valid_IDs:
                    new_files.append(file)

            return new_files

        else:
            print('filter_data_valid: if not supplying a (list of) dfs, supply a list of filenames')

    else:
        print('filter_data_valid: dfs should be pd.DataFrame or a list thereof')
        exit()


def load_pp_info(task: Literal['search', 'freeviewing'] = 'search', filter_fv=True, filter_demo=True):
    pp_info = pd.read_csv(INTERM_DATA_DIR / task / 'participant_info.csv')

    if filter_fv or filter_demo:
        pp_info = pp_info.loc[pp_info['Valid task'] == True]

    if filter_demo:
        pp_info = pp_info.loc[pp_info['Valid demographics'] == True]

    return pp_info


def id_to_age_from_ppinfo(x, task: Literal['search', 'freeviewing'] = 'search'):
    ppinfo = load_pp_info(task=task)
    ppinfo['ID'] = ppinfo['ID'].astype(str)

    if isinstance(x, str) or isinstance(x, int):
        pinf = ppinfo.loc[ppinfo['ID'] == str(x)]

        if len(pinf) > 0:
            return pinf.iloc[0]['Age']
        else:
            return np.nan

    elif isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, pd.Series):
        ages = []
        for ID in x:
            pinf = ppinfo.loc[ppinfo['ID'] == str(ID)]
            if len(pinf) > 0:
                ages.append(pinf.iloc[0]['Age'])
            else:
                ages.append(np.nan)

        return ages
