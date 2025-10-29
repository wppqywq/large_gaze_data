from pathlib import Path
from matplotlib import font_manager, rcParams

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw_data'
PROC_DATA_DIR = DATA_DIR / 'processed_data'
FIX_DATA_DIR = DATA_DIR / 'fixation_data'
INTERM_DATA_DIR = DATA_DIR / 'intermediate_data'

RESULT_DIR = ROOT_DIR / 'analysis_results'
PLOT_DIR = RESULT_DIR / 'plots'


N_JOBS = 4  # Number of cores/threads for multiprocessing

SAMPLING_RATE = 60  # Eye tracker sampling rate
INVALID_WINDOW_OVERALL = 5000   # Max. allowed window (ms) of data loss in order to reject participant demographics
INVALID_WINDOW_TASK = 1000      # Max. allowed window (ms) of data loss in order to reject participant
INVALID_SUM_TASK = 2000         # Max. allowed total data loss during task in order to reject participant

RESOLUTION = (1920, 1080)  # 16:9
DISPSIZE = (598, 336)  # mm
SCREENDIST = 800  # mm
PX_PER_CM = DISPSIZE[0] / RESOLUTION[0]

TARGET_LOC_DICT = {1: 'Top L', 2: 'Top C', 3: 'Top R', 4: 'Bot L', 5: 'Bot C', 6: 'Bot R'}

BINRANGES = [
    (1, 5),
    (6, 10),
    (11, 15),
    (16, 20),
    (21, 25),
    (26, 30),
    (31, 35),
    (36, 40),
    (41, 45),
    (46, 50),
    (51, 55),
    (56, 60),
    (61, 65),
    (66, 70),
    (71, 75),
    (76, 80),
    (81, 85),
    (86, 90),
    (91, 95)
]

BINSTRINGS = []
for a, b in BINRANGES:
    if a < 10:
        a = f'0{a}'
    if b < 10:
        b = f'0{b}'
    BINSTRINGS.append(f'{a}-{b}')

# Data visualisation stuff. Use fira sans as font if possible, otherwise fall back to helvetica
try:
    fe = font_manager.FontEntry(fname='/Users/4137477/Library/Fonts/FiraSansRegular.ttf', name='Fira Sans Regular')
    font_manager.fontManager.ttflist.insert(0, fe)
    rcParams['font.family'] = 'Fira Sans Regular'

    fe = font_manager.FontEntry(fname='/Users/4137477/Library/Fonts/FiraSansBold.ttf', name='Fira Sans Bold')
    font_manager.fontManager.ttflist.insert(0, fe)

except Exception as e:
    print(e, 'falling back to Helvetica')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Helvetica'

rcParams['font.size'] = 12
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.frameon'] = False
