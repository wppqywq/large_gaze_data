# A very large eye tracking dataset of freeviewing and visual search

This project processes and analyzes eye-tracking data from a visual search and free-viewing experiment.

## Data Structure

All data organized by task: `search/` and `freeviewing/`.
- `data/raw_data/`

  Original eye-tracker output files from the experiment device.
  - Gaze samples: position (x, y), timestamp, pupil size, validity flag
  - Event markers: task transitions, screen changes, response events.

- `data/processed_data/`

  Cleaned and formatted gaze data. One .csv file per participant.

  Columns: time_from_start (seconds), x, y (pixel coordinates), timestamp, datetime, unix_time, valid_gaze (bool), pupil_size, event(scren change), message.
  Data is parsed from raw .txt files and time-aligned to task start.

- `data/fixation_data/`

  Detected fixation points computed using I2MC algorithm. One .csv file per participant.
  
  Columns: onset (s), offset (s), duration (=offset-onset), avg_x, avg_y (pixel coordinates), ID, label='FIXA'.
  
  Only includes fixations with duration >= 60ms within first 10 seconds of task.

- `data/intermediate_data/`

  Aggregated data and participant metadata. Contains:
  - participant_info.csv: [ID, Gender, DoB, Age, Task,	Image,	Target location,	Shape,	Response,	Response outcome(check shape=response, Hit/False Alarm/Nothing found),Valid demographics, Valid task]
  - compiled_gaze.csv: all valid gaze samples concatenated (first 10s only), [x, y, time_from_start, ID]
    
    $\downarrow$ I2MC 
  - compiled_fixations.csv: all valid fixations concatenated, columns same as `/fixation_data/`

  - for search task, also `accuracy_with_gaze.csv`: including basic [ID, Target location(str & px), Response outcome(Hit/False Alarm/Nothing found)], also joint responce and gaze data together:
    - Response hit: whether the response is correct or not
    - Gaze hit: whether eye movement saw the target (within 1.5 degrees and lasting ≥0.2 seconds)
    - Combined hit: Response is correct AND eye movement is seen
    - Combined SDT:
      - Full hit: Response correct + Detection correct
      - Guess: detection incorrect + response correct
      - Full miss: No detection + response Incorrect
      - LBFTS (Looked But Failed To See): response incorrect + detection correct (Response outcome = 'Nothing found')
      - Misremembered/LBFTS: same as LBFTS, but Response outcome = 'False Alarm'
    - Gaze hit onset/duration: first gaze at the target, start time and duration
    - Refixation count: count that the target is refixed after seeing it (0 = only seen once).
    - Nearest gaze: (visual angle) between the nearest gaze point and the target.

Free viewing and searching are completely **independent** samples of participants (zero overlaped participant).

## Analysis Code

`analysis_code/main.py`: handles both 'search' and 'freeviewing' tasks.
Runs the complete preprocessing workflow:
- Load raw .txt files, parse and format to .csv 
- Quality check, compile all valid data.
  - Reject if >1s continuous loss or >2s total loss during task
- I2MC fixation detection algorithm
  - Input(x,y,t) -> velocity -> K-mean cluster to separate fixations from saccades.
- Data stability metrics
  RMS of gaze position using saccade coord median.
- Plotters:

  - `plot_timeseries()`: Gaze trajectory with fixation periods overlaid
  - `plot_single_trace()`: Fixation sequence visualization on task image
  - `plot_demographics()`: Age and gender distribution
  - `freeviewing_heatmap()`: Aggregate attention map across all participants
  - `compute_accuracy_response_based()`: Button response accuracy by target location
  - `compute_accuracy_gaze_based()`: Gaze-based accuracy, classifies as:
    - Full hit (responded correctly + fixated target)
    - LBFTS (Looked But Failed To See: fixated + wrong response)
    - Guessed (responded + no fixation on target)
    - Full miss (no response + no fixation)
  - `plot_sdt_overlay()`: Accuracy classifications visualized on task image


### `create_stimulus.py`
Generates search task stimulus images.

Output: 12 stimulus images saved to `stimulus_code/stimuli/` (target symbol 6 locations × 2 shapes)

### `constants.py`

Configures all paths, parameters, and experiment settings.

- `SAMPLING_RATE = 60` Hz (eye-tracker samples at 60 per second)
- `RESOLUTION = (1920, 1080)` pixels
- `DISPSIZE = (598, 336)` mm (physical screen size)
- `SCREENDIST = 800` mm (eye-to-screen distance)
---
- `INVALID_WINDOW_OVERALL = 5000` ms: max continuous data loss in demographics phase
- `INVALID_WINDOW_TASK = 1000` ms: max continuous data loss during task
- `INVALID_SUM_TASK = 2000` ms: max total data loss during task
---
- `N_JOBS = 4`: number of CPU cores for parallel processing
---
- `minFixDur = 60` ms: minimum fixation duration to count as fixation
- `maxMergeDist = 44.8` pixels: ≈1 degree visual angle, fixations closer than this are merged

- Default `max_dist = 1.5` degrees: fixation must be within this distance of target to count as "fixated"
- Default `min_dur = 0.2` seconds: fixation must exceed this duration to count as deliberate
