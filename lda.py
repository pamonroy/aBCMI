#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:41:18 2024
Modified on Thu Mar 26 2026

@author: Pablo Andrés Monroy D'Croz

================================================================================
DISCLAIMER
================================================================================
This software is provided for research purposes only. The authors and
institutions are not liable for any direct or indirect damages arising from its
use. It is not intended for clinical or diagnostic applications.

================================================================================
MIT LICENSE
================================================================================
Copyright (c) 2024-2026 Pablo Andrés Monroy D'Croz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================================================
INSTITUTIONS
================================================================================
- Universitat Pompeu Fabra (Barcelona, Spain)
- Universidad Icesi (Cali, Colombia)

================================================================================
LINEAR DISCRIMINANT ANALYSIS (LDA) - 2×2 Affective Music Generation Experiment
================================================================================
Real‑time affective music generation based on EEG frontal asymmetry (AF7/AF8)
using a Linear Discriminant Analysis (LDA) classifier. The system extracts power
spectral features, trains an LDA model during calibration, and generates adaptive
MIDI music based on real-time emotion predictions under a 2×2 factorial design.

The experiment includes:
- Real‑time EEG acquisition via BITalino or offline simulation from XDF files
- Power spectral feature extraction (theta, alpha, low/high beta, gamma bands)
- LDA classifier training on individual participant data
- LSL streaming for recording and synchronisation
- Emotion prediction (arousal with short and long-term smoothing)
- Adaptive music generation with mode, loudness, tempo, and rhythm controlled by emotion
- 2×2 factorial design (sad/happy × music/no-music conditions)
- Self-report mechanism for emotional state validation (1-9 scale)

-------------------------------------------------------------------------------
DESCRIPTION
-------------------------------------------------------------------------------
This software implements a real-time affective Brain-Computer Musical Interface
(BCMI) using a Linear Discriminant Analysis (LDA) classifier to predict
emotional states from EEG signals under a 2×2 experimental design.

The system:
    1. Acquires EEG signals from prefrontal electrodes (AF7, AF8) using BITalino
    2. Computes spectral features (theta, alpha, low/high beta, gamma bands)
    3. Calibration phase:
        - Collects baseline (eyes closed, no music)
        - Records emotional states (sad, neutral, happy)
        - Trains LDA classifier with 10 spectral features
    4. Online classification phase:
        - Extracts features in real-time
        - Predicts valence probability (happy vs. sad)
        - Computes smoothed emotion values (10-second and 5-second buffers)
    5. Generates adaptive algorithmic music in real-time via MIDI with:
        - Mode selection based on valence (7 modes from Lydian to Locrian)
        - Tempo and rhythmic density controlled by arousal
        - Loudness and brightness controlled by emotional intensity
        - Harmonic progression following I-IV-V-I pattern
    6. Sends synchronized markers via Lab Streaming Layer (LSL)
    7. Collects self-report ratings (1-9) for emotional state validation

-------------------------------------------------------------------------------
EXPERIMENTAL FRAMEWORK (2×2 FACTORIAL DESIGN)
-------------------------------------------------------------------------------
Independent Variables:
    - Emotion Induction: Sad vs. Happy
    - Auditory Condition: Music vs. No Music

Dependent Variables:
    - EEG spectral power features (10 features: theta, alpha, low/high beta, gamma
      from AF7 and AF8)
    - Self-reported emotional state (1-9 Likert scale)
    - Classifier prediction accuracy and response times

Hypothesis:
    EEG spectral features from prefrontal electrodes (AF7, AF8) can predict
    emotional states (happy vs. sad) using Linear Discriminant Analysis.
    The combination of emotion induction and music presentation modulates
    both neural and subjective emotional responses.

Feature Set (10 features per window):
    - Theta power: 4-7.9 Hz (both AF7 and AF8)
    - Alpha power: 8-13.9 Hz (both AF7 and AF8)
    - Low Beta power: 14-21.9 Hz (both AF7 and AF8)
    - High Beta power: 22-29.9 Hz (both AF7 and AF8)
    - Gamma power: 30-47 Hz (both AF7 and AF8)

Musical Mapping:
    - Valence → Musical mode (7 Greek modes: Lydian/happy to Locrian/sad)
    - Arousal → Tempo, rhythm density, loudness, harmonic complexity
    - Smoothed valence (5-second buffer) → Stable mode selection per harmonic bar

-------------------------------------------------------------------------------
EXPERIMENTAL PROTOCOL
-------------------------------------------------------------------------------
Phase 0: Preparation
    - Participant code entry
    - LSL outlet initialization
    - EEG hardware connection check

Phase 1: Volume Setting (optional)
    - Alternating sad/happy music to set comfortable listening level

Phase 2: Meditation (3 minutes)
    - Eyes closed rest period to establish baseline physiological state

Phase 3: Calibration (Training Data Collection)
    - 6 trials: idle (eyes closed), sad, happy, neutral (counterbalanced order)
    - Each trial: 10s baseline silence + 20s emotional condition music
    - Self-report before and after calibration block

Phase 4: Model Training
    - Baseline subtraction and standardization
    - LDA training on sad vs. happy conditions

Phase 5: Online Phase (2×2 Factorial)
    - 20 trials: 5 repetitions of each condition (sad-music, sad-noMusic,
      happy-music, happy-noMusic)
    - Random order with constraint: no more than two consecutive same
      emotion or action
    - Each trial: 13s silence + 4s cue + 33-40s experimental condition
    - Self-report after each trial
    - Real-time emotion prediction and adaptive music generation (music conditions)

Phase 6: Debriefing
    - Experiment completion signals
    - Data saving and cleanup

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
Core Dependencies:
    - pylsl: Lab Streaming Layer for data synchronization
    - numpy: Numerical computations
    - scipy: Signal processing (Welch's method, Simpson integration)
    - mido: MIDI communication for music generation
    - bitalino: BITalino hardware interface
    - pyxdf: XDF file loading for offline simulation (optional)
    - pynput: Keyboard input for self-report collection

Machine Learning:
    - scikit-learn: LinearDiscriminantAnalysis, StandardScaler

-------------------------------------------------------------------------------
HARDWARE REQUIREMENTS
-------------------------------------------------------------------------------
- BITalino (r)evolution with EEG sensor kit
- Electrodes: AF7 (left prefrontal), AF8 (right prefrontal)
- MIDI synthesizer or DAW (e.g., Ableton Live) for audio output
- MIDI controller/interface for self-report (keyboard 1-9 keys)

-------------------------------------------------------------------------------
FILE OUTPUTS
-------------------------------------------------------------------------------
- {participant}_training.csv: Spectral features with class labels (training data)
- {participant}_model.pickle: Trained LDA classifier
- {participant}_standardScaler.pickle: Fitted StandardScaler for online use

-------------------------------------------------------------------------------
MARKER CODES
-------------------------------------------------------------------------------
Experiment Markers:
    6: setVolume      7: online       8: modeling
    9: music         10: pause       11: askForReport
    12: stopMusic    13: startNoMusic 14: stopNoMusic
    -1: min_power    0: silence      1: sad
    2: neutral       3: happy

-------------------------------------------------------------------------------
"""

from pylsl import StreamInfo, StreamOutlet
import time
from collections import deque
import numpy as np
import random as rn
from threading import Thread, Event
import sys
import pandas as pd
import mido
from mido import Message
import os
import pickle
from pynput import keyboard

# Change working directory to project location
os.chdir("/Volumes/Academico/2026-TIC phd/software/participants/")

# MIDI output port for music generation
outport = mido.open_output()

# Electrode impedance thresholds (BITalino raw ADC values)
THRESHOLD_LOW = 10      # Values below indicate poor electrode contact
THRESHOLD_HIGH = 1014   # Values above indicate saturation

# ============================================================================
# Global Parameters
# ============================================================================
samplingRate = 1000          # EEG sampling frequency in Hz
windowSize = 4               # Analysis window length in seconds
buffersize = samplingRate * windowSize  # Buffer size in samples
update_rate = 0.5            # Emotion update rate in seconds

# Threading events for experiment control
event = Event()              # Signal to stop all threads
is_memorizing = Event()      # Flag to indicate data collection for training
is_online = Event()          # Flag to indicate online classification phase
bitalino_connected = False    # Flag for device connection status

# LSL outlets (initialized later in main)
outletMarkers = None          # For experiment markers
outletHarmony = None          # For chord progression markers
outletSelfReport = None       # For self-report values
outletEmotionalMarkers = None # For predicted emotion values
outletPredictedEmotion = None # For emotion vector (current, sequence, bar)


# ============================================================================
# Signal Processing Functions
# ============================================================================

def select_lsl_stream(stream_name, streams):
    """
    Select a specific LSL stream from loaded XDF data.
    
    Parameters
    ----------
    stream_name : str
        Name of the stream to select (e.g., 'eeg-bitalino')
    streams : list
        List of streams from pyxdf.load_xdf()
    
    Returns
    -------
    array_like or False
        Time series data if stream found, False otherwise
    """
    stream_list = streams[0]
    for stream in stream_list:
        if stream['info']['name'][0] == stream_name:
            return stream['time_series']
    return False


def RawEeg2uVolt(data_segment):
    """
    Convert raw BITalino ADC values to microvolts.
    
    BITalino uses 10-bit ADC (0-1023) with 3.3V reference and specific gain.
    
    https://support.pluxbiosignals.com/knowledge-base/bitalino-documentation/
    
    Parameters
    ----------
    data_segment : array_like
        Raw ADC values (0-1023)
    
    Returns
    -------
    array_like
        Microvolt values (µV)
    """
    return (data_segment / 2**10 - 0.5) * 3.3 / 41782 * 1e6


def bandpower_welch(data, sf, band, window_sec=None, relative=False):
    """
    # taken from https://raphaelvallat.com/bandpower.html
    Compute band power using Welch's method.
    
    Parameters
    ----------
    data : array_like
        Input signal in time domain
    sf : float
        Sampling frequency in Hz
    band : list
        [low, high] frequency band in Hz
    window_sec : float, optional
        Length of each Welch window in seconds
    relative : bool, default=False
        If True, return relative power (normalized by total power)
    
    Returns
    -------
    float
        Band power (absolute or relative)
    """
    from scipy.signal import welch
    from scipy.integrate import simpson
    band = np.asarray(band)
    low, high = band
    
    # Determine window length: at least 2 cycles of the lowest frequency
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    
    # Compute power spectral density using Welch's method
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # Integrate power in the frequency band using Simpson's rule
    bp = simpson(psd[idx_band], dx=freq_res)
    
    if relative:
        bp /= simpson(psd, dx=freq_res)
    
    return bp


def power_spectral_density(electrode_signal, band):
    """
    Compute power in specific frequency band for an EEG signal.
    
    Parameters
    ----------
    electrode_signal : array_like
        EEG signal in raw ADC units
    band : str
        Frequency band: 'theta', 'alpha', 'low_beta', 'high_beta', 'beta', 'gamma'
    
    Returns
    -------
    float
        Band power in microvolts squared (µV²)
    """
    global samplingRate
    
    # Define frequency ranges for each band
    if band == 'theta':
        low, high = 4, 7.9
    elif band == 'alpha':
        low, high = 8, 13.9
    elif band == 'low_beta':
        low, high = 14, 21.9
    elif band == 'high_beta':
        low, high = 22, 29.9
    elif band == 'beta':
        low, high = 14, 29.9
    else:  # gamma
        low, high = 30, 47
    
    # Convert to microvolts and compute band power
    electrode_signal = RawEeg2uVolt(electrode_signal)
    power = bandpower_welch(electrode_signal, samplingRate, [low, high], 
                           window_sec=1, relative=False)
    
    if power <= 0:
        sendMarker("min_power")
        print('NEGATIVE POWER', power, band)
    
    return power


def get_power(left_electrode, right_electrode):
    """
    Extract power features from both EEG channels.
    
    Computes power in 5 frequency bands (theta, alpha, low_beta, high_beta, gamma)
    for both left (AF7) and right (AF8) prefrontal channels.
    
    Parameters
    ----------
    left_electrode : array_like
        AF7 signal (left prefrontal)
    right_electrode : array_like
        AF8 signal (right prefrontal)
    
    Returns
    -------
    list
        Feature vector with 10 values: 
        [theta_AF7, theta_AF8, alpha_AF7, alpha_AF8,
         low_beta_AF7, low_beta_AF8, high_beta_AF7,
         high_beta_AF8, gamma_AF7, gamma_AF8]
    """
    AF7 = np.asarray(left_electrode, dtype=np.float64)
    AF8 = np.asarray(right_electrode, dtype=np.float64)
    
    # Compute powers for AF7 (left)
    theta_AF7 = power_spectral_density(AF7, band='theta')
    alpha_AF7 = power_spectral_density(AF7, band='alpha')
    low_beta_AF7 = power_spectral_density(AF7, band='low_beta')
    high_beta_AF7 = power_spectral_density(AF7, band='high_beta')
    gamma_AF7 = power_spectral_density(AF7, band='gamma')
    
    # Compute powers for AF8 (right)
    theta_AF8 = power_spectral_density(AF8, band='theta')
    alpha_AF8 = power_spectral_density(AF8, band='alpha')
    low_beta_AF8 = power_spectral_density(AF8, band='low_beta')
    high_beta_AF8 = power_spectral_density(AF8, band='high_beta')
    gamma_AF8 = power_spectral_density(AF8, band='gamma')
    
    return [theta_AF7, theta_AF8, alpha_AF7, alpha_AF8, low_beta_AF7, 
            low_beta_AF8, high_beta_AF7, high_beta_AF8, gamma_AF7, gamma_AF8]


# ============================================================================
# Data Acquisition Functions
# ============================================================================

def affectiveEstimator(emo_vector):
    """
    Real-time EEG acquisition from BITalino device with feature extraction.
    
    This function runs in a separate thread and continuously:
    1. Reads raw EEG data from BITalino
    2. Updates rolling buffers for AF7 and AF8
    3. Computes power spectral features every update_rate seconds
    4. Performs online classification when is_online flag is set
    5. Stores data for training when is_memorizing flag is set
    6. Maintains smoothed emotion values (10-second and 5-second buffers)
    
    Parameters
    ----------
    emo_vector : list
        Shared list to store emotion values:
        [0] = current arousal
        [1] = sequence-smoothed arousal (10-second average)
        [2] = bar-smoothed arousal (5-second average)
    """
    global samplingRate, buffersize, update_rate, segmento, baseline, bitalino_connected
    
    from bitalino import BITalino
    macAddress = "/dev/tty.BITalino-8B-B1-DevB"
    acqChannels = [0, 1]
    
    # Initialize smoothing buffers
    sequence_buffer_size = int(10 / update_rate)  # 10-second buffer
    sequence_buffer = deque([0.5] * sequence_buffer_size, maxlen=sequence_buffer_size)
    
    bar_buffer_size = int(5 / update_rate)  # 5-second buffer (one harmonic bar)
    bar_buffer = deque([0.5] * bar_buffer_size, maxlen=bar_buffer_size)
    
    # Initialize EEG buffers with neutral values (512 = ~2.5V)
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)
    
    nSamples = int(samplingRate / 2)  # Read 500 samples (0.5 seconds) each iteration
    
    # Create LSL outlet for streaming raw EEG
    info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32', 'eeg-data')
    outlet = StreamOutlet(info)
    print("Created Signal stream: %s" % info.name())
    
    # Connect to BITalino hardware
    connection = False
    while not connection:
        try:
            print("Connecting to BITalino(%s)..." % macAddress)
            device = BITalino(macAddress)
            connection = True
        except (TypeError, ValueError) as e:
            print("Connection failed:", e)
            time.sleep(1)
    
    print('\a', end='')
    print(device.version())
    print("Connected to BITalino(%s)" % macAddress)
    device.start(samplingRate, acqChannels)
    bitalino_connected = True
    
    deltaTime = time.time()  # Timer for electrode alarm
    
    try:
        while not event.is_set():
            # Read a chunk of data from BITalino
            data_bitalino = device.read(nSamples)
            for rowData in data_bitalino:
                dataAF7 = rowData[6]   # AF7 (left) on channel 1
                dataAF8 = rowData[5]   # AF8 (right) on channel 0
                localAF7.append(dataAF7)
                localAF8.append(dataAF8)
                outlet.push_sample([dataAF7, dataAF8])
            
            # Compute power features from current buffers
            signal_AF7 = list(localAF7)
            signal_AF8 = list(localAF8)
            vector = get_power(signal_AF7, signal_AF8)
            
            # Store data for training if in memorizing mode
            if is_memorizing.is_set():
                segmento.append(vector)
            
            # Perform online classification if in online mode
            if is_online.is_set():
                current_power = np.asarray(vector).reshape(1, 10)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)
                arousal = clf.predict_proba(current_power_std)[0, 0]
                
                # Update smoothing buffers
                sequence_buffer.append(arousal)
                bar_buffer.append(arousal)
                
                # Update shared emotion vector
                emo_vector[0] = arousal
                emo_vector[1] = np.mean(sequence_buffer)
                emo_vector[2] = np.mean(bar_buffer)
                
                # Send to LSL
                outletPredictedEmotion.push_sample([emo_vector[0], emo_vector[1], emo_vector[2]])
            
            # Check electrode impedance (values outside threshold indicate poor contact)
            if time.time() - deltaTime > 0.05:
                deltaTime = time.time()
                if not (THRESHOLD_LOW < dataAF7 < THRESHOLD_HIGH):
                    print(">> ELECTRODE ALARM AF7 <<")
                    print('\a', end='')
                if not (THRESHOLD_LOW < dataAF8 < THRESHOLD_HIGH):
                    print(">> ELECTRODE ALARM AF8 <<")
                    print('\a', end='')
    
    finally:
        print('Stop capturing')
        device.stop()
        device.close()
        sys.exit(0)


def affectiveEstimatorFromFile(emo_vector):
    """
    Simulate real-time EEG capture from pre-recorded XDF files.
    
    This function is used for offline testing and development. It reads
    pre-recorded XDF files (both AF7 and AF8 data) and simulates the real-time
    data stream at the appropriate timing intervals.
    
    Parameters
    ----------
    emo_vector : list
        Shared list to store emotion values:
        [0] = current arousal
        [1] = sequence-smoothed arousal (10-second average)
        [2] = bar-smoothed arousal (5-second average)
    """
    import pyxdf as xdf
    
    global samplingRate, buffersize, update_rate, segmento, baseline, bitalino_connected, participant
    
    # Initialize smoothing buffers
    sequence_buffer_size = int(10 / update_rate)  # 10-second buffer
    sequence_buffer = deque([0.5] * sequence_buffer_size, maxlen=sequence_buffer_size)
    
    bar_buffer_size = int(5 / update_rate)  # 5-second buffer
    bar_buffer = deque([0.5] * bar_buffer_size, maxlen=bar_buffer_size)
    
    # Create simulated LSL outlet
    simulated_info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32',
                                'eeg-simulated-data')
    simulated_outlet = StreamOutlet(simulated_info)
    
    # Initialize EEG buffers
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)
    
    nSamples = int(samplingRate / 2)  # Process 500 samples per update
    
    # Load XDF files (AF7 from -afah experiment, ML from -ml experiment)
    xdf_file1 = '../participants/' + participant + '-afah.xdf'
    streams1 = xdf.load_xdf(xdf_file1)
    xdf_series1 = select_lsl_stream('eeg-bitalino', streams1)
    xdf_series1 = xdf_series1[50000:]  # Remove first 50 seconds (connection noise)
    
    xdf_file2 = '../participants/' + participant + '-ml.xdf'
    streams2 = xdf.load_xdf(xdf_file2)
    xdf_series2 = select_lsl_stream('eeg-bitalino', streams2)
    xdf_series2 = xdf_series2[50000:]  # Remove first 50 seconds
    
    # Concatenate data from both files
    xdf_series = np.concatenate((xdf_series2, xdf_series1, xdf_series2))
    
    # Extract signals (order: AF7, AF8)
    signal_AF7 = xdf_series[:, 0].tolist()
    signal_AF8 = xdf_series[:, 1].tolist()
    signal_full = list(zip(signal_AF7, signal_AF8))
    
    deltaTime = time.time()
    
    while signal_full and not event.is_set():
        # Process data at the specified update rate
        if time.time() - deltaTime >= update_rate:
            deltaTime = time.time()
            
            # Process next chunk of data
            sample = signal_full[:nSamples]
            signal_full = signal_full[nSamples:]
            
            for data in sample:
                localAF7.append(data[0])
                localAF8.append(data[1])
                simulated_outlet.push_sample([data[0], data[1]])
            
            # Compute power features
            signal_AF7_list = list(localAF7)
            signal_AF8_list = list(localAF8)
            vector = get_power(signal_AF7_list, signal_AF8_list)
            
            # Store data for training if in memorizing mode
            if is_memorizing.is_set():
                segmento.append(vector)
            
            # Perform online classification if in online mode
            if is_online.is_set():
                current_power = np.asarray(vector).reshape(1, 10)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)
                arousal = clf.predict_proba(current_power_std)[0, 0]
                
                # Update smoothing buffers
                sequence_buffer.append(arousal)
                bar_buffer.append(arousal)
                
                # Update shared emotion vector
                emo_vector[0] = arousal
                emo_vector[1] = np.mean(sequence_buffer)
                emo_vector[2] = np.mean(bar_buffer)
                
                # Send to LSL
                outletPredictedEmotion.push_sample([emo_vector[0], emo_vector[1], emo_vector[2]])
            
            # Simulate electrode alarms
            if data[0] < THRESHOLD_LOW or data[1] < THRESHOLD_LOW:
                print(">> ELECTRODE ALARM <<")
                print('\a', end='')
    
    print('Stop capturing')
    print('\a')
    sys.exit(0)


# ============================================================================
# Music Generation Functions
# ============================================================================

def getChord(mode, bar):
    """
    Get chord number for current mode and bar position.
    
    Implements the harmonic sequence I - IV - V - I across 4 bars.
    Modes are spaced by 5th intervals following the circle of fifths.
    
    Parameters
    ----------
    mode : int
        Musical mode (0=Lydian to 6=Locrian)
    bar : int
        Bar position (0-3)
    
    Returns
    -------
    int
        Chord number (1-7 for diatonic chords, 0 for silence)
    """
    # Base chord determined by mode (iterating by 5th intervals)
    base = (3 + (mode * 5) - mode) % 7
    
    # Bar mapping: I - IV - V - I progression
    delta_map = {0: 0, 1: 3, 2: 4, 3: 0}
    delta = delta_map.get(bar, 0)
    
    # Return chord number (1-7, 0 is silence marker)
    return (base + delta) % 7 + 1


def affectiveMusicGenerator(isRealTime=True, delay=40, in_arousal=None, in_valence=None):
    """
    Generate affective music based on emotional state.
    
    This function creates MIDI music that adapts to the predicted emotion in real-time.
    Musical parameters controlled by emotion include:
    - Mode (scale) determined by valence
    - Tempo and rhythm density controlled by arousal
    - Loudness and brightness controlled by arousal and valence
    - Harmony and melodic patterns based on Greek modes
    
    The music is generated using a 4-bar chord progression with 7 different modes
    (Lydian to Locrian) mapped to valence values.
    
    Parameters
    ----------
    isRealTime : bool, default=True
        Use real-time emotion prediction (True) or pre-recorded values (False)
    delay : float, default=40
        Duration of music generation in seconds
    in_arousal : list, optional
        Pre-recorded arousal values for offline playback
    in_valence : list, optional
        Pre-recorded valence values for offline playback
    """
    global event
    
    # Chord progressions for different modes (MIDI note numbers)
    chordlist = np.array([
        [60, 64, 55, 59],  # C major voicing
        [62, 65, 57, 60],  # D major voicing
        [64, 55, 59, 62],  # E major voicing
        [60, 65, 57, 64],  # F major voicing
        [55, 59, 62, 65],  # G major voicing
        [57, 60, 64, 55],  # A major voicing
        [59, 62, 65, 57]   # B major voicing
    ])
    
    # Mode definitions: each mode has a unique 4-bar chord progression
    modeset = np.zeros((4, 4, 7))  # [bar, chord_note, mode]
    mode_names = ['lydian', 'ionian', 'mixolydian', 'dorian', 
                  'aeolian', 'phrygian', 'locrian']
    
    mode_map = [
        [3, 6, 0, 3],  # Lydian: F, B, C, F
        [0, 3, 4, 0],  # Ionian: C, F, G, C
        [4, 0, 1, 4],  # Mixolydian: G, C, D, G
        [1, 4, 5, 1],  # Dorian: D, G, A, D
        [5, 1, 2, 5],  # Aeolian: A, D, E, A
        [2, 5, 6, 2],  # Phrygian: E, A, B, E
        [6, 2, 3, 6]   # Locrian: B, E, F, B
    ]
    
    # Build mode-specific chord sets
    for mode_idx, mode_indices in enumerate(mode_map):
        for bar_idx in range(4):
            modeset[bar_idx, :, mode_idx] = chordlist[mode_indices[bar_idx], :]
    
    modeset -= 3  # Transpose down by 3 semitones for better range
    
    LOW_LOUDNESS = 50  # Minimum velocity
    tick = beat = bar = 0  # Musical timing counters
    start = time.time()
    
    try:
        while (time.time() - start < delay) and not event.is_set():
            # Update harmony at bar boundaries (every 4 beats)
            if beat % 4 == 0 and tick % 2 == 0:
                # Get current emotional state
                if isRealTime:
                    arousal = emotional_vector[2]  # Use bar-smoothed arousal
                    valence = arousal
                else:
                    if not in_valence or not in_arousal:
                        break
                    valence = in_valence.pop(0)
                    arousal = in_arousal.pop(0)
                
                # Select musical mode based on valence
                if bar == 0:
                    if isRealTime:
                        valence = emotional_vector[2]  # 5-second smoothed valence
                        arousal = valence
                    
                    # Map valence to mode (0=lydian/happy, 6=locrian/sad)
                    mode = max(0, min(6, 7 - round(valence * 6) - 1))
                    print("Mode:", mode_names[int(mode)], int(mode))
                
                # Log current emotional state
                if isRealTime:
                    print('EMO: %.2f - current: %.2f - sequence: %.2f - bar: %.2f' % 
                          (arousal, emotional_vector[0], emotional_vector[1], emotional_vector[2]))
                    outletEmotionalMarkers.push_sample([arousal])
                
                # Map emotion to musical parameters
                roughness = 1 - arousal      # Higher arousal = less repetition
                velocity = arousal           # Higher arousal = faster tempo
                voicing = valence            # Higher valence = brighter harmony
                loudness = int(round(arousal * 10) / 10 * 40 + 60)
                
                # Generate rhythmic patterns based on roughness
                activate1 = np.random.rand(8)
                activate1[activate1 < roughness] = 0
                activate1[activate1 >= roughness] = 1
                
                activate2 = np.random.rand(8)
                activate2[activate2 < roughness] = 0
                activate2[activate2 >= roughness] = 1
                
                # Generate brightness offsets for octave transposition
                bright = np.random.rand(6)
                if voicing < 0.5:
                    bright[bright > voicing * 2] = -1
                    bright[bright <= voicing * 2] = 0
                else:
                    bright[bright < (voicing - 0.5) * 2] = 1
                    bright[bright >= (voicing - 0.5) * 2] = 0
                
                # Send "All Notes Off" to all channels
                for channel in range(4):
                    msg = Message('control_change', control=123, value=0, channel=channel)
                    outport.send(msg)
                
                # Send chord marker to LSL
                sendChordMarker(getChord(mode, bar))
                
                # Play chord on channels 0 (piano) and 1 (cello)
                channels = [0, 1]
                for channel in channels:
                    for note_index in range(3):
                        note = int(modeset[bar, note_index, mode] + bright[note_index] * 12)
                        vel = rn.randint(LOW_LOUDNESS, loudness)
                        msg = Message('note_on', channel=channel, note=note, velocity=vel)
                        outport.send(msg)
                
                # Play bass on channel 2
                bass_note = modeset[bar, 0, mode] - (24 if voicing <= 0.5 else 12)
                msg = Message('note_on', channel=2, note=int(bass_note),
                              velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
                
                bar = (bar + 1) % 4  # Advance to next bar
            
            # Update beat counter (4 beats per bar)
            if tick % 2 == 0:
                beat = (beat + 1) % 4
            
            # Play melody on channel 3 (lead instrument)
            msg = Message('control_change', control=123, value=0, channel=3)
            outport.send(msg)
            
            if activate1[tick] == 1:
                note = int(modeset[bar-1, 0, mode] + bright[4] * 12)
                msg = Message('note_on', channel=3, note=note,
                              velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
            
            if activate2[tick] == 1:
                idx = rn.randint(1, 2)
                note = int(modeset[bar-1, idx, mode] + bright[5] * 12)
                msg = Message('note_on', channel=3, note=note,
                              velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
            
            # Tempo control: higher arousal = shorter sleep time (faster tempo)
            time.sleep(0.3 - velocity * 0.15)
            tick = (tick + 1) % 8  # 8 ticks per bar
    
    except KeyboardInterrupt:
        event.set()
    
    finally:
        # Ensure all notes are turned off
        for channel in range(4):
            msg = Message('control_change', control=123, value=0, channel=channel)
            outport.send(msg)
        
        if event.is_set():
            sys.exit(0)


# ============================================================================
# Self-Report Functions
# ============================================================================

class KeyPressThread(Thread):
    """
    Thread to capture keyboard input for self-report ratings.
    
    Listens for numeric key presses (1-9) with a timeout mechanism.
    """
    def __init__(self):
        super().__init__()
        self.key_pressed = False
        self.pressed_key = None
        self.timer_duration = 15  # Seconds to wait for response
        self.valid_keys = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        self.listener = keyboard.Listener(on_press=self.on_press)
    
    def run(self):
        self.listener.start()
        while not self.key_pressed:
            print("Asking for self report")
            msg = Message('control_change', control=84, value=127, channel=5)
            outport.send(msg)
            self.wait_for_keypress(self.timer_duration)
            if not self.key_pressed:
                print("Asking again!")
        self.listener.stop()
    
    def wait_for_keypress(self, timeout):
        """Wait for a keypress or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.key_pressed:
                return
            time.sleep(0.1)
    
    def on_press(self, key):
        """Callback for keyboard press events."""
        try:
            if key.char in self.valid_keys:
                self.key_pressed = True
                self.pressed_key = key.char
        except AttributeError:
            pass  # Ignore special keys


def self_report():
    """
    Collect self-report emotional rating from participant.
    
    Sends a MIDI message requesting a report and waits for numeric key input (1-9).
    The rating is sent to LSL for synchronization with EEG data.
    """
    time.sleep(2)
    sendMarker("askForReport")
    
    # Simulate response for testing (set to False for real use)
    key_press_simulation = False
    
    if key_press_simulation:
        print("Asking for self report")
        msg = Message('control_change', control=84, value=127, channel=5)
        outport.send(msg)
        time.sleep(4)
        selfReportValue = rn.randint(1, 9)  # Simulated response
    else:
        keypress_thread = KeyPressThread()
        keypress_thread.start()
        keypress_thread.join()
        selfReportValue = int(keypress_thread.pressed_key)
    
    # Send report to LSL
    outletSelfReport.push_sample([selfReportValue])
    print("Self report:", selfReportValue)


def close_eyes():
    """Ask participant to close eyes and relax."""
    time.sleep(1)
    print("Close eyes and relax")
    msg = Message('control_change', control=83, value=127, channel=5)
    outport.send(msg)
    time.sleep(6)


# ============================================================================
# Experiment Control Functions
# ============================================================================

def sendMarker(marker):
    """
    Send an integer marker to LSL stream for experiment synchronization.
    
    Parameters
    ----------
    marker : str
        Marker name from dictionary
    """
    lsl_markers = {
        "setVolume": 6, "silence": 0, "neutral": 2, "sad": 1,
        "happy": 3, "modeling": 8, "online": 7, "music": 9,
        "min_power": -1, "pause": 10, "askForReport": 11,
        "stopMusic": 12, "startNoMusic": 13, "stopNoMusic": 14
    }
    outletMarkers.push_sample([lsl_markers[marker]])


def sendChordMarker(chord):
    """
    Send chord marker to LSL for harmonic analysis.
    
    Parameters
    ----------
    chord : int
        Chord number (0=silence, 1-7=diatonic chords)
    """
    outletHarmony.push_sample([chord])


def targetMessage(emotion):
    """
    Send target emotion message to Ableton Live via MIDI.
    
    Parameters
    ----------
    emotion : str
        Target emotion: 'sad', 'neutral', or 'happy'
    """
    control_map = {'sad': 82, 'neutral': 81, 'happy': 80}
    sendMarker(emotion)
    msg = Message('control_change', control=control_map[emotion], value=127, channel=5)
    outport.send(msg)


def play_silence(sleepTime):
    """
    Play silence (no music) for specified duration.
    
    Parameters
    ----------
    sleepTime : float
        Duration of silence in seconds
    """
    sendMarker("silence")
    sendChordMarker(0)  # Send silence to harmony markers
    time.sleep(sleepTime)


def play_volume_setting():
    """Play volume setting routine (alternating sad/happy)."""
    input_valence = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    input_arousal = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_sad_class():
    """Play sad emotion example (low arousal and valence)."""
    input_valence = [0] * 8
    input_arousal = [0] * 8
    sendMarker('sad')
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_neutral_class():
    """Play neutral emotion example (mid arousal and valence)."""
    input_valence = [0.5] * 12
    input_arousal = [0.5] * 12
    sendMarker('neutral')
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_happy_class():
    """Play happy emotion example (high arousal and valence)."""
    input_valence = [1] * 16
    input_arousal = [1] * 16
    sendMarker('happy')
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_neutral_online():
    """
    Play neutral harmonic sequence at beginning of each online trial.
    
    Provides a consistent starting point before adaptive music begins.
    """
    input_valence = [0.5] * 4
    input_arousal = [0.5] * 4
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == "__main__":
    
    experiment_start_time = time.time()
    
    # ------------------------------------------------------------------------
    # Participant Setup
    # ------------------------------------------------------------------------
    participant = input("Write the participant CODE and press Enter: ")
    
    # ------------------------------------------------------------------------
    # Initialize LSL Outlets
    # ------------------------------------------------------------------------
    infoMarkers = StreamInfo('markers', 'Markers', 1, 0, 'int32', 'eegloop-markers')
    outletMarkers = StreamOutlet(infoMarkers)
    
    infoHarmonyMarkers = StreamInfo('harmony', 'Markers', 1, 0, 'int32', 'harmony-sequence')
    outletHarmony = StreamOutlet(infoHarmonyMarkers)
    
    infoSelfReportMarkers = StreamInfo('self-report', 'Markers', 1, 0, 'int32', 'emotional-self-report')
    outletSelfReport = StreamOutlet(infoSelfReportMarkers)
    
    infoEmotionalMarkers = StreamInfo('emotion', 'Markers', 1, 0, 'float32', 'music-generator-emotion-data')
    outletEmotionalMarkers = StreamOutlet(infoEmotionalMarkers)
    
    infoPredictedEmotion = StreamInfo('predicted-emotion', 'Emotion', 3, 2, 'float32', 'emotion-data')
    outletPredictedEmotion = StreamOutlet(infoPredictedEmotion)
    
    # Shared emotion vector: [current arousal, sequence-smoothed, bar-smoothed]
    emotional_vector = [0.5, 0.5, 0.5]
    
    # Start acquisition thread
    if bitalino_connected:
        t = Thread(target=affectiveEstimatorFromFile, args=(emotional_vector,))
    else:
        t = Thread(target=affectiveEstimator, args=(emotional_vector,))
    t.start()
    
    # Wait for BITalino connection
    while not bitalino_connected:
        time.sleep(0.1)
        
    input("Record on LabRecorder (adding _lda2x2 suffix: " + participant + "_lda2x2.xdf) and Press Enter to start...")
    
    # ------------------------------------------------------------------------
    # Volume Setting Phase (Optional)
    # ------------------------------------------------------------------------
    askVolume = input("Set volume (y/n)? ")
    if askVolume == 'y':
        sendMarker("setVolume")
        msg = Message('control_change', control=13, value=127, channel=5)
        outport.send(msg)
        time.sleep(10)
        play_volume_setting()
    
    sendMarker("pause")
    input("Ask to close eyes and Press Enter to continue...")
    
    # ------------------------------------------------------------------------
    # Meditation Phase (3 minutes)
    # ------------------------------------------------------------------------
    print("\n--------------------> [MEDITATION]")
    sendMarker("pause")
    msg = Message('control_change', control=16, value=127, channel=5)
    outport.send(msg)
    time.sleep(180)
    
    # ------------------------------------------------------------------------
    # Calibration Phase - Collect Training Data
    # ------------------------------------------------------------------------
    print("\n--------------------> [CALIBRATING]")
    msg = Message('control_change', control=14, value=127, channel=5)
    outport.send(msg)
    time.sleep(16)
    sendMarker("pause")
    
    # Initial self-report and eyes closed
    self_report()
    close_eyes()
    
    # Pseudorandomize calibration sequence
    max_attempts = 10
    class_sequence = ['sad', 'happy', 'neutral'] * 4
    shuffled_sequence = []
    n = rn.randint(0, len(class_sequence) - 1)
    shuffled_sequence.append(class_sequence.pop(n))
    attempts = 0
    
    while class_sequence:
        n = rn.randint(0, len(class_sequence) - 1)
    
        if shuffled_sequence[-1] != class_sequence[n]:
            shuffled_sequence.append(class_sequence.pop(n))
            attempts = 0
        else:
            attempts += 1
            if attempts >= max_attempts:
                # Reset if too many attempts
                class_sequence = ['sad', 'happy', 'neutral'] * 4
                shuffled_sequence = []
                n = rn.randint(0, len(class_sequence) - 1)
                shuffled_sequence.append(class_sequence.pop(n))
                attempts = 0
            continue
    
    # Collect training data
    df = pd.DataFrame()
    
    # Define column names for the 10 power features
    feature_columns = ['theta_AF7', 'theta_AF8', 'alpha_AF7', 'alpha_AF8', 
                   'low_beta_AF7', 'low_beta_AF8', 'high_beta_AF7', 
                   'high_beta_AF8', 'gamma_AF7', 'gamma_AF8']
    
    
    for i, val in enumerate(shuffled_sequence):
        if event.is_set():
            sys.exit(0)
        
        print('\n---> trial %d/%d: %s' % (i+1, len(shuffled_sequence), val))
        
        # Record idle baseline (10 seconds of silence)
        segmento = []
        is_memorizing.set()
        play_silence(10)
        is_memorizing.clear()
        
        df_temp = pd.DataFrame(segmento, columns=feature_columns)
        df_temp['class'] = 'idle'
        print('\nclass idle:', len(df_temp), 'samples\n')
        df = pd.concat([df, df_temp])
        
        # Record emotional condition (20 seconds of fixed music)
        segmento = []
        is_memorizing.set()
        sendMarker("music")
        eval("play_" + val + "_class()")
        is_memorizing.clear()
        
        df_temp = pd.DataFrame(segmento, columns=feature_columns)
        df_temp['class'] = val
        print('class', val, ':', len(df_temp), 'samples')
        df = pd.concat([df, df_temp])
    
    play_silence(2)
    
    # Save training data
    df = df.reset_index(drop=True)
    df.to_csv(participant + '_training.csv', sep='\t', index=False)
    
    # ------------------------------------------------------------------------
    # Model Training (LDA)
    # ------------------------------------------------------------------------
    print("\n--------------------> [MODELING]")
    sendMarker('modeling')
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Compute baseline from idle condition
    baseline = df.loc[df['class'] == 'idle'].iloc[:, :-1].mean().to_numpy()
    
    # Prepare training data (sad and happy conditions only)
    X = df.loc[df['class'].isin(['sad', 'happy'])].iloc[:, :-1].to_numpy()
    y = df.loc[df['class'].isin(['sad', 'happy'])].iloc[:, -1:].values.ravel()
    
    # Baseline subtraction and standardization
    X_cleared = X - baseline
    sc = StandardScaler()
    X_std = sc.fit_transform(X_cleared)
    
    # Train LDA classifier
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(X_std, y)
    
    # Save model and scaler
    with open(participant + '_model.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(participant + '_standardScaler.pickle', 'wb') as f:
        pickle.dump(sc, f)
    
    # ------------------------------------------------------------------------
    # Generate Random Trial Sequence (2×2 Design)
    # ------------------------------------------------------------------------
    # Conditions: sad-music, sad-noMusic, happy-music, happy-noMusic
    conditions = ["sad-music", "sad-noMusic", "happy-music", "happy-noMusic"]
    trials_per_condition = 5
    max_attempts = 100
    
    emotions_seed_list = conditions * trials_per_condition
    emotion_buffer = deque(["empty", "empty"], maxlen=2)
    action_buffer = deque(["empty", "empty"], maxlen=2)
    emotions_list = []
    attempts = 0
    
    # Ensure no more than two consecutive same emotions or actions
    while emotions_seed_list:
        blocked_emotion = emotion_buffer[0] if emotion_buffer[0] == emotion_buffer[1] else ""
        blocked_action = action_buffer[0] if action_buffer[0] == action_buffer[1] else ""
        
        n = rn.randint(0, len(emotions_seed_list) - 1)
        element = emotions_seed_list[n]
        target_emotion, trial_action = element.split('-')
        
        if target_emotion == blocked_emotion or trial_action == blocked_action:
            attempts += 1
            if attempts >= max_attempts:
                # Reset if too many attempts
                emotions_seed_list = conditions * trials_per_condition
                emotion_buffer = deque(["empty", "empty"], maxlen=2)
                action_buffer = deque(["empty", "empty"], maxlen=2)
                emotions_list = []
                attempts = 0
            continue
        
        emotions_list.append(element)
        emotions_seed_list.pop(n)
        emotion_buffer.append(target_emotion)
        action_buffer.append(trial_action)
        attempts = 0
    
    # ------------------------------------------------------------------------
    # Online Phase
    # ------------------------------------------------------------------------
    print("\n--------------------> [ONLINE]")
    sendMarker("pause")
    
    # Trigger online phase in Ableton Live
    msg = Message('control_change', control=15, value=127, channel=5)
    outport.send(msg)
    time.sleep(30)
    
    is_online.set()
    sendMarker('online')
    
    # Initial self-report
    self_report()
    
    # Run trials
    for i, condition in enumerate(emotions_list):
        if event.is_set():
            sys.exit(0)
        
        target_emotion = condition.split('-')[0]
        trial_action = condition.split('-')[1]
        
        close_eyes()
        play_silence(13)
        targetMessage(target_emotion)
        print('\n---> trial %d: towards %s' % (i+1, condition))
        time.sleep(4)  # Silence without marker
        
        if trial_action == "music":
            sendMarker("music")
            play_neutral_online()
            affectiveMusicGenerator(True, 33)  # 33 seconds for ~40-second trial
            sendMarker("stopMusic")
        else:
            sendMarker("startNoMusic")
            msg = Message('control_change', control=85, value=127, channel=5)
            outport.send(msg)
            time.sleep(40)  # 40 seconds of silence
            sendMarker("stopNoMusic")
            msg = Message('control_change', control=85, value=127, channel=5)
            outport.send(msg)
        
        # Collect self-report after each trial
        self_report()
    
    # ------------------------------------------------------------------------
    # Experiment End
    # ------------------------------------------------------------------------
    play_silence(10)
    
    # Send experiment completion signals
    msg = Message('control_change', control=17, value=127, channel=5)
    outport.send(msg)
    time.sleep(8)
    
    msg = Message('control_change', control=18, value=127, channel=5)
    outport.send(msg)
    time.sleep(1)
    outport.send(msg)
    
    # Stop all threads
    event.set()
    t.join()
    
    print("Execution time: %s minutes" % int((time.time() - experiment_start_time) / 60))