from pathlib import Path
from typing import List, Literal
from math import dist
import time
from itertools import combinations, permutations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from joblib import Parallel, delayed

from constants import *
from analysis_code.helperfunctions import (
    age_to_bin,
    gender_str_convert,
    px_to_dva,
    heatmap_from_df,
)
from stimulus_code.create_stimulus import get_stimulus_loc_dict, get_stimulus_locs_px
from analysis_code.dataloader_helpers import load_pp_info, id_to_age_from_ppinfo


############
# DATA QUALITY ETC.
############
def plot_timeseries(df: pd.DataFrame, filename: Path, show=True):
    """
    Plot a timeseries with periods of fixation overlaid, if possible.
    Comment/uncomment to plot either x/y or displacement
    :param df: pre-processed data
    :param filename:
    :param show: whether to show plots as output (True), or save only (False)
    :return:
    """

    x = np.array(df['x']).ravel()
    y = np.array(df['y']).ravel()
    t = np.array(df['time_from_start']).ravel()

    try:
        fixations = pd.read_csv(str(filename).replace('/processed_data', '/fixation_data'))
        fixations = fixations.loc[fixations['label'] == 'FIXA']
        starts = np.array(fixations['onset'])
        ends = np.array(fixations['offset'])

        if len(starts) == 0:
            starts, ends = [0], [0]

    except Exception as e:
        print(e)
        starts, ends = [0], [0]

    if len(x) > 0:
        try:
            labels = ['x', 'y', 'Fixation']
            palette = list(sns.color_palette('deep', 10))

            f = plt.figure(figsize=(10, 5))

            # Make subplots
            ax = []
            ax.append(f.add_subplot(1, 1, 1))

            # X/Y
            h0 = ax[0].plot(t, x, label=labels[0], linewidth=1, color=palette[0], marker='o', markersize=1.5)
            h1 = ax[0].plot(
                t, y, label=labels[2], color=palette[2], linestyle='--', linewidth=1, marker='o', markersize=1.5
            )

            ax[0].set_ylim((0, 2000))
            ax[0].set_yticks([0, 540, 1080, 1920])
            ax[0].set_ylabel('Gaze position (x/y)')
            ax[0].set_xlabel('Time (s)')
            ax[0].set_xlim((0, 10))

            # Displacement
            # displacement = get_displacement(xe, ye)
            # displacement = px_to_dva(displacement) * SAMPLING_RATE
            # h2 = ax[1].plot(t[1:], displacement, label=labels[4],
            #                 color='gray',
            #                 alpha=.5,
            #                 # linewidth=1,
            #                 linewidth=0,
            #                 zorder=-20
            #                 )
            #
            # ax[1].set_ylim((0, 300))  # np.max(displacement) * 2
            # ax[1].set_yticks([0, 50, 100, 150, 200, 250])
            # ax[1].set_ylabel('Velocity (°/s)')
            #
            # ax[1].yaxis.set_label_position("right")
            # ax[1].yaxis.tick_right()
            # ax[1].set_yticks([])
            #
            # ax[1].set_xticks([])
            # ax[1].set_xlim((0, 10))
            # ax[1].set_xlabel('')

            # Boxes which indicate periods of fixation
            for i, (s, e) in enumerate(zip(starts, ends)):
                h3 = ax[0].axvspan(s, e, color='gray', alpha=0.05, label=labels[2], zorder=-30)

            ax[0].legend([h0[0], h1[0], h3], labels, loc='upper right')

            plt.title(Path(filename).stem)
            plt.tight_layout()

            plt.savefig(PLOT_DIR / 'fixation_detection' / f'{filename.stem}.png', dpi=200)

            if show:
                plt.show()

            plt.close()

        except ValueError:
            pass


def plot_single_trace(n: int = 3, task: Literal['search', 'freeviewing'] = 'freeviewing', skip_existing=True):
    im_path = ROOT_DIR / 'stimulus_code' / 'image_hd.png'
    im = Image.open(im_path)

    gaze_data_path = FIX_DATA_DIR / task
    gaze_paths = sorted(list(gaze_data_path.rglob('*.csv')))

    i = 0
    for p in gaze_paths:
        i += 1

        if i >= n:
            break
        if skip_existing and Path(PLOT_DIR / 'fixation_single_traces' / f'{p.stem}.png').exists():
            print(f'Skipping {i} of {n}, already exists')
            continue

        try:
            df = pd.read_csv(p)
            df = df.rename({'avg_x': 'x', 'avg_y': 'y', 'onset': 'time_from_start'}, axis=1)
            df = df.loc[df['time_from_start'] < 10]  # Only take freeviewing part

            x = np.asarray(df['x'])
            y = np.asarray(df['y'])
            s = np.asarray(df['duration'])

            # Scale duration markers
            s = (s - np.min(s)) / (np.max(s) - np.min(s)) * 2000

            # Calculate the midpoints of each line segment
            x_mid = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
            y_mid = [(y[i] + y[i + 1]) / 2 for i in range(len(y) - 1)]

            # Calculate the direction of each segment for rotation
            angles = np.arctan2(np.diff(y), np.diff(x))

            # Make and format fig
            f = plt.figure(figsize=(7.5, 4.22), facecolor='black')
            ax = f.add_subplot(1, 1, 1)
            ax.axis('off')
            ax.set_xlim(0, 1920)
            ax.set_ylim(0, 1080)

            # Show image
            ax.imshow(im, extent=(0, 1920, 0, 1080), alpha=0.5)

            # Add fixation markers
            ax.scatter(
                x,
                y,
                marker='o',
                s=s,
                c='white',
                alpha=0.5,
            )

            # Conncect fixations
            ax.plot(
                x,
                y,
                ms=0,
                c='white',
                alpha=0.5,
            )

            # Add arrow heads in center
            for xm, ym, angle in zip(x_mid, y_mid, angles):
                plt.plot(
                    xm,
                    ym,
                    marker=(3, 0, np.degrees(angle) - 90),  # Triangle marker rotated to match the line direction
                    color='white',
                    markersize=6,
                    alpha=0.5,
                    lw=0,
                )

            f.tight_layout(pad=0.05)
            plt.savefig(PLOT_DIR / 'fixation_single_traces' / f'{p.stem}.png', dpi=300)
            plt.show()

        except Exception as e:
            print(p.name, e)


def plot_demographics(task: Literal['search', 'freeviewing'] = 'search'):
    pp_info = pd.read_csv(INTERM_DATA_DIR / task / 'participant_info.csv')
    valid_demo = pp_info.loc[(pp_info['Valid demographics'] == True) & (pp_info['Valid task'] == True)]
    valid_demo['Gender'] = valid_demo['Gender'].apply(lambda x: x.title())
    valid_demo['Bin'] = valid_demo['Age'].apply(age_to_bin)

    f = plt.figure(figsize=(5.5, 3))
    ax = f.add_subplot(1, 1, 1)

    sns.kdeplot(
        data=valid_demo,
        x='Age',
        hue='Gender',
        hue_order=['Female', 'Male', 'Other'],
        palette='colorblind',
        linewidth=3,
        fill=True,
        alpha=0.4,
        clip=[1, 85],
        ax=ax,
    )

    ytext = 0.019 if task == 'freeviewing' else 0.023
    ax.text(x=-1, y=ytext, s='n = ', c='gray', ha='center', fontsize=10)

    agebin = valid_demo.groupby('Bin').size()
    for br, bs in zip(BINRANGES, BINSTRINGS):
        x = br[0]

        if x != 1:
            ax.axvline(x, ls=':', c='gray', lw=0.5, zorder=-500)

        x = np.mean([br[0], br[1]]) + 0.5
        try:
            s = agebin[bs]
        except:
            s = 0

        ax.text(x=x, y=ytext, s=s, c='gray', ha='center', fontsize=10)

    sns.move_legend(ax, loc='center left', bbox_to_anchor=(0.75, 0.5), title=None, fontsize=10)

    ax.set_xlim(0, 95)
    ax.set_xlabel('Age', fontsize=14)
    ax.set_xticks([x[0] for x in BINRANGES] + [BINRANGES[-1][-1]])

    ax.set_ylabel('Number of visitors', fontsize=14)
    ax.set_yticks([])

    f.tight_layout(pad=0.5)
    f.savefig(PLOT_DIR / f'demographics_{task}.png', dpi=300)
    plt.show()


############
# HEATMAPS
############
def freeviewing_heatmap(plot=True, overlay=True, blur=True) -> np.ndarray:
    pp_info = load_pp_info('freeviewing', filter_fv=True, filter_demo=False)

    gaze = pd.read_csv(INTERM_DATA_DIR / 'freeviewing' / 'compiled_fixations.csv')
    gaze = gaze.loc[gaze['ID'].isin(list(pp_info['ID'].unique()))]

    heatmap = heatmap_from_df(gaze, blur=blur)

    if plot:
        im = Image.open(ROOT_DIR / 'stimulus_code' / 'image_hd.png')
        f = plt.figure(figsize=(7.5, 4.22), facecolor='black')
        ax = f.add_subplot(1, 1, 1)

        ax.axis('off')
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)

        if overlay:
            ax.imshow(im, extent=(0, 1920, 1080, 0), alpha=1)

        ax.imshow(heatmap, cmap='inferno', alpha=0.85, extent=(0, 1920, 0, 1080))

        f.tight_layout(h_pad=0.05, w_pad=0.5, pad=0.05)
        f.savefig(PLOT_DIR / f'freeviewing_heatmap{"_overlay" if overlay else ""}.png', dpi=300)
        plt.show()

    return heatmap


############
# COMPUTE ACCURACY & RT METRICS
############
def compute_accuracy_response_based(filter_demo=True) -> pd.DataFrame:
    df = load_pp_info('search', filter_fv=True, filter_demo=filter_demo)
    print(f'Computing accuracy of {len(df)} participants')

    df['Target location'] = df['Target location'].apply(lambda x: TARGET_LOC_DICT[x])
    dfg = df.groupby(['Target location', 'Response outcome']).agg({'ID': 'count'})
    dfg = dfg.groupby(['Target location'])['ID'].transform(lambda x: x / np.sum(x)).reset_index()
    dfg = dfg.rename({'ID': 'Proportion'}, axis=1)

    return dfg


def compute_accuracy_gaze_based(max_dist: float = 1.5, min_dur: float = 0.2, filter_demo=True) -> pd.DataFrame:
    def combined_sdt(r, f):
        """
        :param r: Response outcome (hit/false alarm/nothing found)
        :param f: Target fixated or not
        :return:
        """
        if r == 'Hit' and f:
            # Responded and fixated; Full hit
            return 'Full hit'

        if r == 'False Alarm' and f:
            # False alarm but fixated; Misremembered, or could be a guess after LBFTS)
            return 'Misremembered/LBFTS'

        if (r == 'False Alarm' or r == 'Hit') and not f:
            # Responded but not fixated; FA --> regardless of whether response was correct
            return 'Guessed (not fixated)'

        if r == 'Nothing found' and f:
            # Not responded but fixated (Looked But Failed To See)
            return 'LBFTS'

        if r == 'Nothing found' and not f:
            # Not responded, not fixated; Full miss
            return 'Full miss'

    acc = load_pp_info('search', filter_fv=True, filter_demo=filter_demo)
    acc['Target location px'] = acc['Target location'].apply(lambda x: get_stimulus_locs_px()[x - 1])
    acc['Target location name'] = acc['Target location'].apply(lambda x: TARGET_LOC_DICT[x])

    gaze = pd.read_csv(INTERM_DATA_DIR / 'search' / 'compiled_fixations.csv')
    gaze = gaze.loc[gaze['ID'].isin(list(acc['ID']))]

    results = {
        'ID': [],
        'Target location': [],
        'Target location px': [],
        'Response outcome': [],
        'Response hit': [],
        'Gaze hit': [],
        'Combined hit': [],
        'Combined SDT': [],
        'Gaze hit onset': [],
        'Gaze hit duration': [],
        'Refixation count': [],
        'Nearest gaze': [],
    }
    for ID in list(acc['ID'].unique()):
        accid = acc.loc[acc['ID'] == ID]
        gid = gaze.loc[gaze['ID'] == ID]

        targ_px = accid['Target location px'].values[0]
        targ_name = accid['Target location name'].values[0]
        resp_hit = accid['Hit'].astype(bool).values[0]
        resp_outc = accid['Response outcome'].values[0]

        gaze_hit = False
        gaze_onset = np.nan
        gaze_dur = np.nan
        refix_count = 0
        nearest = np.inf

        # Loop through fixation coordinates and compute whether they were on the target
        for i in range(len(gid)):
            gidi = gid.iloc[i]
            x, y, onset, dur = gidi['avg_x'], gidi['avg_y'], gidi['onset'], gidi['duration']

            # Flip target y coordinates to set origin the same as gaze
            tx, ty = targ_px[0], RESOLUTION[1] - targ_px[1]

            # Get euclidean distance in degrees
            euc_ = px_to_dva(dist((x, y), (tx, ty)))

            if euc_ <= max_dist and dur >= min_dur:
                if gaze_hit:
                    refix_count += 1

                gaze_hit = True
                gaze_onset = onset
                gaze_dur = dur

            if euc_ < nearest:
                nearest = euc_

        results['ID'].append(ID)
        results['Target location'].append(targ_name)
        results['Target location px'].append(targ_px)
        results['Response outcome'].append(resp_outc)
        results['Response hit'].append(resp_hit)
        results['Gaze hit'].append(gaze_hit)
        results['Combined hit'].append(resp_hit & gaze_hit)
        results['Combined SDT'].append(combined_sdt(resp_outc, gaze_hit))
        results['Gaze hit onset'].append(gaze_onset)
        results['Gaze hit duration'].append(gaze_dur)
        results['Refixation count'].append(refix_count)
        results['Nearest gaze'].append(nearest)

    results = pd.DataFrame(results)
    results.to_csv(INTERM_DATA_DIR / 'search' / 'accuracy_with_gaze.csv')

    return results


############
# PLOT ACCURACY & RT METRICS
############
def plot_sdt_overlay(max_dist: float = 1.5, min_dur: float = 0.2):
    df = compute_accuracy_gaze_based(max_dist, min_dur, filter_demo=False)

    # "Looked but failed to report"
    df['Combined SDT'] = df['Combined SDT'].apply(lambda x: 'LBFTR' if 'LBFTS' in x else x)
    outcome_opts = ['Full hit', 'LBFTR', 'Guessed (not fixated)', 'Full miss']

    stim_locs = get_stimulus_loc_dict()

    # Make and format fig
    f = plt.figure(figsize=(7.5, 4.22), facecolor='black')
    ax = f.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)

    # Show image
    im = Image.open(ROOT_DIR / 'stimulus_code' / 'image_hd.png')
    ax.imshow(im, extent=(0, 1920, 1080, 0), alpha=0.5)

    for i, loc in enumerate(list(stim_locs.keys())):
        acc_df = df.loc[(df['Target location'] == loc)]

        dfg = acc_df.groupby(['Target location', 'Combined SDT']).agg({'ID': 'count'})
        dfg = dfg.groupby(['Target location'])['ID'].transform(lambda x: x / np.sum(x)).reset_index()
        dfg = dfg.rename({'ID': 'Proportion'}, axis=1)

        outcomes = []
        for oco in outcome_opts:
            try:
                oc = dfg.loc[dfg['Combined SDT'] == oco]['Proportion'].values[0]
                outcomes.append(oc)
            except:
                outcomes.append(0)

        colours = list(sns.color_palette('colorblind'))

        local_ax = f.add_subplot(2, 3, i + 1)
        local_ax.axis('off')
        patches, _, _ = local_ax.pie(
            outcomes,
            colors=colours,
            radius=0.5,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white', 'alpha': 0.5},
            counterclock=False,
            startangle=90,
            autopct=lambda p: round(p, 1),
            pctdistance=1.35,
            textprops={'color': 'white', 'fontsize': 10},
        )

        if i == 0:
            ax.legend(
                handles=patches,
                labels=outcome_opts,
                loc='upper left',
                labelcolor='white',
                ncols=4,
                fontsize=10,
            )

    f.tight_layout(pad=0.05)
    f.savefig(PLOT_DIR / 'search_sdt_overlay.png', dpi=600)
    plt.show()


def plot_rt_overlay(max_dist: float = 1.5, min_dur: float = 0.2):
    df = compute_accuracy_gaze_based(max_dist, min_dur, filter_demo=False)

    # Looked but failed to report
    df['Combined SDT'] = df['Combined SDT'].apply(lambda x: 'LBFTR' if 'LBFTS' in x else x)

    colours = list(sns.color_palette('colorblind'))

    stim_locs = get_stimulus_loc_dict()

    # Make and format fig
    f = plt.figure(figsize=(7.5, 4.22), facecolor='black')
    ax = f.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)

    # Show image
    im = Image.open(ROOT_DIR / 'stimulus_code' / 'image_hd.png')
    ax.imshow(im, extent=(0, 1920, 1080, 0), alpha=0.25)

    for i, loc in enumerate(list(stim_locs.keys())):
        acc_df = df.loc[(df['Target location'] == loc)]

        local_ax = f.add_subplot(2, 3, i + 1, facecolor='none')

        sns.violinplot(
            data=acc_df,
            x='Gaze hit onset',
            orient='h',
            hue='Combined SDT',
            hue_order=['Full hit', 'LBFTR'],
            palette=colours,
            fill=True,
            alpha=0.75,
            linewidth=2,
            linecolor='white',
            split=True,
            gap=0.1,
            common_norm=False,
            cut=0,
            legend=True if i == 2 else False,
            inner_kws={'solid_capstyle': 'butt', 'color': 'gray'},
            ax=local_ax,
        )

        local_ax.set_xlim(-0.1, 10)
        local_ax.set_xlabel('')
        local_ax.set_ylabel('')
        local_ax.set_xticks([])
        local_ax.set_yticks([])
        local_ax.spines[['right', 'top']].set_visible(False)
        local_ax.spines[['left', 'bottom']].set_color('white')

        if i == 2:
            sns.move_legend(local_ax, 'upper left', labelcolor='white', title=None, fontsize=10)

        if i == 4:
            local_ax.set_xlabel('Time until first fixation (s)', color='white', fontsize=11)

        if i >= 3:
            local_ax.set_xticks([0, 5, 10], [0, 5, 10], color='white', fontsize=10)

    f.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.5)
    f.savefig(PLOT_DIR / 'search_rt_overlay.png', dpi=600)
    plt.show()


def plot_search_accuracy_overlay():
    df = compute_accuracy_response_based(filter_demo=False)

    stim_locs = get_stimulus_loc_dict()

    # Make and format fig
    f = plt.figure(figsize=(7.5, 4.22), facecolor='black')
    ax = f.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)

    # Show image
    im = Image.open(ROOT_DIR / 'stimulus_code' / 'image_hd.png')
    ax.imshow(im, extent=(0, 1920, 1080, 0), alpha=0.5)

    for i, loc in enumerate(list(stim_locs.keys())):
        acc_df = df.loc[(df['Target location'] == loc)]

        try:
            hits = acc_df.loc[acc_df['Response outcome'] == 'Hit']['Proportion'].values[0]
        except:
            hits = 0
        try:
            fas = acc_df.loc[acc_df['Response outcome'] == 'False Alarm']['Proportion'].values[0]
        except:
            fas = 0
        try:
            nfs = acc_df.loc[acc_df['Response outcome'] == 'Nothing found']['Proportion'].values[0]
        except:
            nfs = 0

        colours = list(sns.color_palette('colorblind'))

        local_ax = f.add_subplot(2, 3, i + 1)
        local_ax.axis('off')
        patches, _, _ = local_ax.pie(
            [hits, fas, nfs],
            colors=[colours[8], colours[9], 'none'],
            radius=0.5,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white', 'alpha': 0.5},
            counterclock=False,
            startangle=90,
            normalize=False,
            autopct=lambda p: round(p, 1),
            pctdistance=1.35,
            textprops={'color': 'white', 'fontsize': 10},
        )

        if i == 0:
            ax.legend(
                handles=patches,
                labels=['Correct', 'Incorrect', 'Nothing found'],
                loc='upper left',
                labelcolor='white',
                ncols=3,
            )

    f.tight_layout(pad=0.05)
    f.savefig(PLOT_DIR / 'search_accuracy_overlay.png', dpi=300)
    plt.show()


def make_all_plots():
    plot_single_trace(n=5, task='freeviewing')
    plot_single_trace(n=5, task='search')

    plot_demographics(task='freeviewing')
    plot_demographics(task='search')

    freeviewing_heatmap(overlay=True)

    plot_search_accuracy_overlay()

    # plot_sdt_overlay()
    # plot_rt_overlay()


if __name__ == '__main__':
    make_all_plots()
