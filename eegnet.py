#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Aug 25 20:05:09 2024
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
EEGNet - DEEP LEARNING FOR AFFECTIVE MUSIC GENERATION
================================================================================
Real-time affective music generation based on EEG frontal asymmetry (AF7/AF8)
using a Convolutional Neural Network (EEGNet architecture). The system extracts
raw EEG signals, trains a deep learning model during calibration, and generates
adaptive MIDI music based on real-time emotion predictions under a 2×2 factorial
design.

The experiment includes:
- Real-time EEG acquisition via BITalino or offline simulation from XDF files
- Raw EEG signal processing (100 Hz sampling, 10-second windows)
- EEGNet deep learning model for binary emotion classification (happy vs. sad)
- LSL streaming for recording and synchronization
- Emotion prediction (current arousal and smoothed valence)
- Adaptive music generation with mode, loudness, tempo, and rhythm controlled by emotion
- 2×2 factorial design (sad/happy × music/no-music conditions)
- Self-report mechanism for emotional state validation (1-9 scale)

-------------------------------------------------------------------------------
DESCRIPTION
-------------------------------------------------------------------------------
This software implements a real-time affective Brain-Computer Musical Interface
(BCMI) using a Convolutional Neural Network (EEGNet architecture) to classify
emotional states from raw EEG signals under a 2×2 experimental design.

The system:
    1. Acquires raw EEG signals from prefrontal electrodes (AF7, AF8) using BITalino
    2. Preprocesses signals (conversion to microvolts, standardization)
    3. Calibration phase:
        - Collects baseline and emotional state data (sad, neutral, happy)
        - Creates sequences of 10-second windows (1000 samples per window)
        - Trains EEGNet model on raw EEG signals (no hand-crafted features)
    4. Online classification phase:
        - Processes raw EEG in real-time (0.5-second updates, 10-second windows)
        - Predicts arousal probability using trained EEGNet model
        - Computes smoothed valence (10-second moving average)
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
    - Raw EEG signals (2 channels: AF7, AF8)
    - Self-reported emotional state (1-9 Likert scale)
    - EEGNet classification accuracy and confidence scores

Deep Learning Architecture (EEGNet):
    - Input shape: (samples, 2 channels, 1000 timesteps, 1)
    - Conv2D layers for spatial and temporal feature extraction
    - Separable convolutions for efficient EEG processing
    - Dense layers for binary classification (happy vs. sad)
    - Output: Probability of happiness (0-1)

Feature Learning Approach:
    - No hand-crafted spectral features (theta, alpha, beta, gamma)
    - End-to-end learning from raw EEG signals
    - Automatic feature extraction through convolutional layers
    - Temporal and spatial pattern recognition

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
    - 6-12 trials depending on configuration (sad, happy, neutral)
    - Each trial: 10s silence + 20s emotional condition music
    - Self-report before and after calibration block

Phase 4: Model Training (EEGNet)
    - Sequence creation (10-second windows, 100 Hz = 1000 samples)
    - Data standardization
    - Model training using TensorFlow/Keras
    - Model saving for online inference

Phase 5: Online Phase (2×2 Factorial)
    - 24 trials: 6 repetitions of each condition (sad-music, sad-noMusic,
      happy-music, happy-noMusic)
    - Random order with constraint: no more than two consecutive same
      emotion or action
    - Each trial: 13s silence + 4s cue + 30s experimental condition
    - Self-report after each trial
    - Real-time emotion prediction using EEGNet
    - Adaptive music generation (music conditions only)

Phase 6: Debriefing
    - Experiment completion signals
    - Data saving and cleanup

-------------------------------------------------------------------------------
DATA FLOW
-------------------------------------------------------------------------------
                      CALIBRATION PHASE
EEG Acquisition (BITalino) → Raw Signals (AF7, AF8) → Sequence Creation (10s windows)
                                                                 ↓
                                                    EEGNet Training → Model Export
                                                                        
                      ONLINE PHASE                                    
EEG Acquisition (BITalino) → Raw Signals (AF7, AF8) → Real-time Windows (10s)
                                                                 ↓
                                                    EEGNet Inference → Emotion Prediction
                                                                 ↓
                                                    MIDI Music Generation

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
Core Dependencies:
    - pylsl: Lab Streaming Layer for data synchronization
    - numpy: Numerical computations and array handling
    - tensorflow.keras: Deep learning framework (EEGNet model)
    - mido: MIDI communication for music generation
    - bitalino: BITalino hardware interface (optional)
    - pyxdf: XDF file loading for offline simulation (optional)
    - pynput: Keyboard input for self-report collection

Machine Learning:
    - scikit-learn: StandardScaler, LabelEncoder
    - tensorflow.keras.models: load_model for trained model loading

External Training Script:
    - eegnet_train.ipynb: Jupyter notebook for EEGNet model training
    - Called via subprocess to leverage GPU acceleration

-------------------------------------------------------------------------------
HARDWARE REQUIREMENTS
-------------------------------------------------------------------------------
- BITalino (r)evolution with EEG sensor kit
- Electrodes: AF7 (left prefrontal), AF8 (right prefrontal)
- MIDI synthesizer or DAW (e.g., Ableton Live) for audio output
- Keyboard for self-report responses (keys 1-9)

-------------------------------------------------------------------------------
FILE OUTPUTS
-------------------------------------------------------------------------------
- {participant}_training_raw.csv: Raw EEG data with class labels
- {participant}_X_train.npy: Preprocessed training sequences
- {participant}_y_train.npy: Training labels
- {participant}_scaler.pickle: Fitted StandardScaler for online use
- {participant}_eegnet_model.h5: Trained EEGNet model

-------------------------------------------------------------------------------
MARKER CODES
-------------------------------------------------------------------------------
Experiment Markers:
    6: setVolume      7: online       8: modeling
    9: music         10: pause       11: askForReport
    12: stopMusic    13: startNoMusic 14: stopNoMusic
    -1: min_power    0: silence      1: sad
    2: neutral       3: happy

Harmony Markers:
    0: silence
    1-7: Diatonic chords (I, II, III, IV, V, VI, VII)

-------------------------------------------------------------------------------
NOTES
-------------------------------------------------------------------------------
- Sampling rate: 100 Hz (downsampled from 1000 Hz for deep learning efficiency)
- Window size: 10 seconds (1000 samples) for optimal EEGNet performance
- Update rate: 0.5 seconds (50% overlap between consecutive windows)
- Threshold for electrode alarm: -38.72 mV (RAW value = 10)
- EEGNet model expects input shape: (1, 2 channels, 1000 timesteps, 1)

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
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle
from pynput import keyboard

# Change working directory to project location
os.chdir("/Volumes/Academico/2026-TIC phd/software/participants/")

# MIDI output port for music generation
outport = mido.open_output()


# Electrode impedance threshold in microvolts (RAW=10 corresponds to -38.72 mV)
THRESHOLD = -38.71938723613039

# ============================================================================
# Global Parameters
# ============================================================================
samplingRate = 100           # EEG sampling frequency in Hz (downsampled)
windowSize = 10              # Analysis window length in seconds
buffersize = samplingRate * windowSize  # Buffer size in samples (1000)
update_rate = 0.5            # Emotion update rate in seconds

# Signal buffers for EEG channels
AF7 = deque([], buffersize)  # Left prefrontal channel buffer
AF8 = deque([], buffersize)  # Right prefrontal channel buffer

# Threading events for experiment control
event = Event()              # Signal to stop all threads
is_memorizing = Event()      # Flag to indicate data collection for training
is_online = Event()          # Flag to indicate online classification phase
bitalino_connected = False    # Flag for device connection status

# LSL outlets (initialized later in main)
outletMarkers = None
outletHarmony = None
outletSelfReport = None
outletEmotionalMarkers = None
outletPredictedEmotion = None


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


def create_sequences(data, labels, timesteps, stride=0):
    """
    Create sequences for deep learning model training.
    
    Parameters
    ----------
    data : array_like
        Input data matrix
    labels : array_like
        Labels corresponding to data
    timesteps : int
        Number of timesteps per sequence
    stride : int, default=0
        Stride for sequence creation (0 = overlapping sequences)
    
    Returns
    -------
    X : array_like
        Sequences of input data
    y : array_like
        Labels for each sequence
    """
    X, y = [], []
    if stride > 0:
        for i in np.arange(0, len(data) - timesteps, stride):
            X.append(data[i:i + timesteps, :-1])
            y.append(labels[i + timesteps, -1])
    else:
        for i in range(len(data) - timesteps + 1):
            X.append(data[i:i + timesteps, :-1])
            y.append(labels[i + timesteps - 1, -1])
    return np.array(X), np.array(y)


# ============================================================================
# Data Acquisition Functions
# ============================================================================

def affectiveEstimator(emotion_vector):
    """
    Real-time EEG acquisition from BITalino device with deep learning inference.
    
    This function runs in a separate thread and continuously:
    1. Reads raw EEG data from BITalino
    2. Converts to microvolts
    3. Updates rolling buffers for AF7 and AF8
    4. Performs online classification using EEGNet model when is_online flag is set
    5. Stores raw data for training when is_memorizing flag is set
    
    Parameters
    ----------
    emotion_vector : list
        Shared list to store emotion values:
        [0] = current arousal prediction
        [1] = smoothed arousal (10-second average)
    """
    global samplingRate, buffersize, update_rate, segmento, bitalino_connected
    
    # Initialize smoothed valence buffer (10-second moving average)
    smvalence_buffer_size = int(10 / update_rate)
    local_smoothed_valence = deque([0.5] * smvalence_buffer_size, maxlen=smvalence_buffer_size)
    
    from bitalino import BITalino
    macAddress = "/dev/tty.BITalino-8B-B1-DevB"
    
    # Initialize EEG buffers with neutral values (512 = ~2.5V)
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)
    
    nSamples = int(samplingRate / 2)  # Read 50 samples (0.5 seconds) each iteration
    
    acqChannels = [0, 1]
    
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
            data_bitalino = device.read(nSamples)
            
            for rowData in data_bitalino:
                dataAF7 = rowData[6]  # AF7 (left) on channel 1
                dataAF7_uv = RawEeg2uVolt(dataAF7)  # Convert to microvolts
                localAF7.append(dataAF7_uv)
                
                dataAF8 = rowData[5]  # AF8 (right) on channel 0
                dataAF8_uv = RawEeg2uVolt(dataAF8)  # Convert to microvolts
                localAF8.append(dataAF8_uv)
                
                outlet.push_sample([rowData[6], rowData[5]])
            
            # Store data for training if in memorizing mode
            if is_memorizing.is_set():
                for rowData in data_bitalino:
                    segmento.append([rowData[6], rowData[5]])
            
            # Perform online classification if in online mode
            if is_online.is_set():
                # Prepare input for EEGNet
                sample_from_queues = np.column_stack((localAF7, localAF8))
                current_sample = np.asarray(sample_from_queues).reshape(1, buffersize, 2)
                
                # Standardize data
                current_sample_std = scaler.fit_transform(
                    current_sample.reshape(-1, current_sample.shape[-1])
                ).reshape(current_sample.shape)
                
                # Reshape for Conv2D layer (samples, channels, timesteps, 1)
                current_sample = np.asarray(current_sample_std).reshape(1, 2, buffersize, 1)
                
                # Predict arousal probability
                arousal = modelo.predict(current_sample, verbose=0)[0, 0]
                
                # Update smoothing buffer
                local_smoothed_valence.append(arousal)
                emotion_vector[0] = arousal
                emotion_vector[1] = np.mean(local_smoothed_valence)
                outletPredictedEmotion.push_sample([emotion_vector[0], emotion_vector[1]])
            
            # Check electrode impedance (values below threshold indicate poor contact)
            if time.time() - deltaTime > 0.05:
                deltaTime = time.time()
                if dataAF7_uv < THRESHOLD:
                    print(">> ELECTRODE ALARM AF7 <<")
                    print('\a', end='')
                if dataAF8_uv < THRESHOLD:
                    print(">> ELECTRODE ALARM AF8 <<")
                    print('\a', end='')
    
    finally:
        print('Stop capturing')
        device.stop()
        device.close()
        sys.exit(0)


def affectiveEstimatorFromFile(emotion_vector):
    """
    Simulate real-time EEG capture from pre-recorded XDF files.
    
    This function is used for offline testing and development. It reads
    pre-recorded XDF files and simulates the real-time data stream.
    
    Parameters
    ----------
    emotion_vector : list
        Shared list to store emotion values:
        [0] = current arousal prediction
        [1] = smoothed arousal (10-second average)
    """
    global samplingRate, buffersize, update_rate, segmento, baseline, bitalino_connected, participant
    
    import pyxdf as xdf
    
    # Initialize smoothed valence buffer
    smvalence_buffer_size = int(10 / update_rate)
    local_smoothed_valence = deque([0.5] * smvalence_buffer_size, maxlen=smvalence_buffer_size)
    
    # Create simulated LSL outlet
    simulated_info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32',
                                'eeg-simulated-data')
    simulated_outlet = StreamOutlet(simulated_info)
    
    # Initialize EEG buffers
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)
    
    nSamples = int(samplingRate / 2)  # Process 50 samples per update
    
    # Load XDF files
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
    
    # Downsample from 1000Hz to 100Hz
    xdf_series = xdf_series[::int(1000 / 100)]
    
    # Extract signals (order: AF7, AF8)
    signal_AF7 = xdf_series[:, 0].tolist()
    signal_AF8 = xdf_series[:, 1].tolist()
    signal_full = list(zip(signal_AF7, signal_AF8))
    
    deltaTime = time.time()
    
    while signal_full and not event.is_set():
        if time.time() - deltaTime >= update_rate:
            deltaTime = time.time()
            
            # Process next chunk
            sample = signal_full[:nSamples]
            sample = np.array(sample)  # Convert to array
            sample = RawEeg2uVolt(sample)  # Convert to microvolts
            signal_full = signal_full[nSamples:]
            
            for data in sample:
                localAF7.append(data[0])
                localAF8.append(data[1])
                simulated_outlet.push_sample([data[0], data[1]])
            
            # Store data for training
            if is_memorizing.is_set():
                for data in sample:
                    segmento.append([data[0], data[1]])
            
            # Perform online classification
            if is_online.is_set():
                sample_from_queues = np.column_stack((localAF7, localAF8))
                current_sample = np.asarray(sample_from_queues).reshape(1, buffersize, 2)
                
                current_sample_std = scaler.fit_transform(
                    current_sample.reshape(-1, current_sample.shape[-1])
                ).reshape(current_sample.shape)
                
                current_sample = np.asarray(current_sample_std).reshape(1, 2, buffersize, 1)
                arousal = modelo.predict(current_sample, verbose=0)[0, 0]
                
                local_smoothed_valence.append(arousal)
                emotion_vector[0] = arousal
                emotion_vector[1] = np.mean(local_smoothed_valence)
                outletPredictedEmotion.push_sample([emotion_vector[0], emotion_vector[1]])
            
            # Simulate electrode alarms
            if data[0] < THRESHOLD or data[1] < THRESHOLD:
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
    base = (3 + (mode * 5) - mode) % 7
    delta_map = {0: 0, 1: 3, 2: 4, 3: 0}
    delta = delta_map.get(bar, 0)
    return (base + delta) % 7 + 1


def affectiveMusicGenerator(isRealTime=True, delay=40, in_arousal=None, in_valence=None):
    """
    Generate affective music based on emotional state using EEGNet predictions.
    
    This function creates MIDI music that adapts to the predicted emotion in real-time.
    Musical parameters controlled by emotion include:
    - Mode (scale) determined by valence
    - Tempo and rhythm density controlled by arousal
    - Loudness and brightness controlled by arousal and valence
    
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
    
    chordlist = np.array([
        [60, 64, 55, 59], [62, 65, 57, 60], [64, 55, 59, 62],
        [60, 65, 57, 64], [55, 59, 62, 65], [57, 60, 64, 55],
        [59, 62, 65, 57]
    ])
    
    modeset = np.zeros((4, 4, 7))
    modes = ['lydian', 'ionian', 'mixolydian', 'dorian', 'aeolian', 'phrygian', 'locrian']
    
    mode_map = [
        [3, 6, 0, 3], [0, 3, 4, 0], [4, 0, 1, 4], [1, 4, 5, 1],
        [5, 1, 2, 5], [2, 5, 6, 2], [6, 2, 3, 6]
    ]
    
    # Build mode-specific chord sets
    for mode_idx, mode_indices in enumerate(mode_map):
        for bar_idx in range(4):
            modeset[bar_idx, :, mode_idx] = chordlist[mode_indices[bar_idx], :]
    
    modeset -= 3  # Transpose down by 3 semitones
    
    LOW_LOUDNESS = 50
    tick = beat = bar = 0
    start = time.time()
    
    try:
        while (time.time() - start < delay) and not event.is_set():
            if beat % 4 == 0 and tick % 2 == 0:
                if isRealTime:
                    arousal = power[0]  # Current arousal prediction
                    valence = arousal
                else:
                    if not in_valence or not in_arousal:
                        break
                    valence = in_valence.pop(0)
                    arousal = in_arousal.pop(0)
                
                if bar == 0:
                    if isRealTime:
                        valence = power[1]  # Smoothed valence
                        arousal = valence
                    
                    mode = max(0, min(6, 7 - round(valence * 6) - 1))
                    print("Mode:", modes[int(mode)], int(mode))
                
                if isRealTime:
                    print('ARO: %.2f - VAL: %.2f - sVAL: %.2f' % 
                          (arousal, valence, power[1]))
                
                outletEmotionalMarkers.push_sample([arousal])
                
                # Map emotion to musical parameters
                roughness = 1 - arousal
                velocity = arousal
                voicing = valence
                loudness = int(round(arousal * 10) / 10 * 40 + 60)
                
                # Generate rhythmic patterns
                activate1 = np.random.rand(8)
                activate1[activate1 < roughness] = 0
                activate1[activate1 >= roughness] = 1
                
                activate2 = np.random.rand(8)
                activate2[activate2 < roughness] = 0
                activate2[activate2 >= roughness] = 1
                
                # Generate brightness offsets
                bright = np.random.rand(6)
                if voicing < 0.5:
                    bright[bright > voicing * 2] = -1
                    bright[bright <= voicing * 2] = 0
                else:
                    bright[bright < (voicing - 0.5) * 2] = 1
                    bright[bright >= (voicing - 0.5) * 2] = 0
                
                # Send all notes off
                for channel in range(4):
                    msg = Message('control_change', control=123, value=0, channel=channel)
                    outport.send(msg)
                
                # Send chord marker
                sendChordMarker(getChord(mode, bar))
                
                # Play chords on piano and cello
                channels = [0, 1]
                for channel in channels:
                    for note_index in range(3):
                        note = int(modeset[bar, note_index, mode] + bright[note_index] * 12)
                        vel = rn.randint(LOW_LOUDNESS, loudness)
                        msg = Message('note_on', channel=channel, note=note, velocity=vel)
                        outport.send(msg)
                
                # Play bass
                bass_note = modeset[bar, 0, mode] - (24 if voicing <= 0.5 else 12)
                msg = Message('note_on', channel=2, note=int(bass_note),
                              velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
                
                bar = (bar + 1) % 4
            
            # Update beat counter
            if tick % 2 == 0:
                beat = (beat + 1) % 4
            
            # Play melody
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
            
            # Tempo control
            time.sleep(0.3 - velocity * 0.15)
            tick = (tick + 1) % 8
    
    except KeyboardInterrupt:
        event.set()
    
    finally:
        # All notes off
        for channel in range(4):
            msg = Message('control_change', control=123, value=0, channel=channel)
            outport.send(msg)
        
        if event.is_set():
            sys.exit(0)


# ============================================================================
# Self-Report Functions
# ============================================================================

class KeyPressThread(Thread):
    """Thread to capture keyboard input for self-report ratings."""
    
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
            pass


def self_report():
    """Collect self-report emotional rating from participant."""
    time.sleep(2)
    sendMarker("askForReport")
    
    key_press_simulation = False  # Set to True for testing
    
    if key_press_simulation:
        print("Asking for self report")
        msg = Message('control_change', control=84, value=127, channel=5)
        outport.send(msg)
        time.sleep(4)
        time.sleep(rn.randint(3, 5))
        selfReportValue = rn.randint(1, 9)
    else:
        keypress_thread = KeyPressThread()
        keypress_thread.start()
        keypress_thread.join()
        selfReportValue = int(keypress_thread.pressed_key)
    
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
    """Send an integer marker to LSL stream for experiment synchronization."""
    lsl_markers = {
        "setVolume": 6, "silence": 0, "neutral": 2, "sad": 1,
        "happy": 3, "modeling": 8, "online": 7, "music": 9,
        "min_power": -1, "pause": 10, "askForReport": 11,
        "stopMusic": 12, "startNoMusic": 13, "stopNoMusic": 14
    }
    outletMarkers.push_sample([lsl_markers[marker]])


def sendChordMarker(chord):
    """Send chord marker to LSL for harmonic analysis."""
    outletHarmony.push_sample([chord])


def targetMessage(emotion):
    """Send target emotion message to Ableton Live via MIDI."""
    control_map = {'sad': 82, 'neutral': 81, 'happy': 80}
    sendMarker(emotion)
    msg = Message('control_change', control=control_map[emotion], value=127, channel=5)
    outport.send(msg)


def play_silence(sleepTime):
    """Play silence (no music) for specified duration."""
    sendMarker("silence")
    sendChordMarker(0)
    time.sleep(sleepTime)


def play_volume_setting():
    """Play volume setting routine."""
    input_valence = [0]*4 + [1]*8 + [0]*4 + [1]*8
    input_arousal = input_valence[:]
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_sad_class():
    """Play sad emotion example."""
    affectiveMusicGenerator(False, 60, [0]*8, [0]*8)
    sendMarker('sad')


def play_neutral_class():
    """Play neutral emotion example."""
    affectiveMusicGenerator(False, 60, [0.5]*12, [0.5]*12)
    sendMarker('neutral')


def play_happy_class():
    """Play happy emotion example."""
    affectiveMusicGenerator(False, 60, [1]*16, [1]*16)
    sendMarker('happy')


def play_neutral_online():
    """Play neutral harmonic sequence at beginning of each online trial."""
    affectiveMusicGenerator(False, 60, [0.5]*4, [0.5]*4)


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
    
    infoPredictedEmotion = StreamInfo('predicted-emotion', 'Emotion', 2, 2, 'float32', 'emotion-data')
    outletPredictedEmotion = StreamOutlet(infoPredictedEmotion)
    
    # Shared emotion vector: [current arousal, smoothed valence]
    power = [0.5, 0.5]
    
    # Start EEG acquisition thread (use XDF simulation if specified)
    if bitalino_connected:
        t = Thread(target=affectiveEstimatorFromFile, args=(power,))
    else:
        t = Thread(target=affectiveEstimator, args=(power,))
    t.start()
    
    # Wait for BITalino connection
    while not bitalino_connected:
        time.sleep(0.1)
    
    input("Record on LabRecorder (adding _eegnet suffix: " + participant + "_eegnet.xdf) and Press Enter to start...")
    
    # ------------------------------------------------------------------------
    # Volume Setting Phase
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
    # Calibration Phase
    # ------------------------------------------------------------------------
    msg = Message('control_change', control=14, value=127, channel=5)
    outport.send(msg)
    time.sleep(16)
    
    print("\n--------------------> [CALIBRATING]")
    sendMarker("pause")
    
    # Initial self-report
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
    length_sequence = len(shuffled_sequence)
    
    for i, val in enumerate(shuffled_sequence):
        if event.is_set():
            sys.exit(0)
        
        print('\n---> trial %d/%d: %s' % (i+1, length_sequence, val))
        
        # Record idle baseline
        segmento = []
        is_memorizing.set()
        play_silence(10)
        is_memorizing.clear()
        
        df_temp = pd.DataFrame(segmento, columns=['AF8', 'AF7'])  # Note: order is AF8, AF7
        df_temp['class'] = 'idle'
        print('\nclass idle:', len(df_temp), 'samples\n')
        df = pd.concat([df, df_temp])
        
        # Record emotional condition
        segmento = []
        is_memorizing.set()
        sendMarker("music")
        eval("play_" + val + "_class()")
        is_memorizing.clear()
        
        df_temp = pd.DataFrame(segmento, columns=['AF8', 'AF7'])
        df_temp['class'] = val
        print('class', val, ':', len(df_temp), 'samples')
        df = pd.concat([df, df_temp])
    
    play_silence(2)
    
    # Post-calibration self-report
    self_report()
    
    # Save training data
    df = df.reset_index(drop=True)
    df.to_csv(participant + '_training_raw.csv', sep='\t', index=False)
    
    # ------------------------------------------------------------------------
    # Model Training (EEGNet)
    # ------------------------------------------------------------------------
    print("\n--------------------> [MODELING]")
    sendMarker('modeling')
    
    # Announce training start
    msg = Message('control_change', control=19, value=127, channel=5)
    outport.send(msg)
    
    # Prepare data for training
    X = df.loc[df['class'].isin(['sad', 'happy'])].iloc[:, :-1].to_numpy()
    y = df.loc[df['class'].isin(['sad', 'happy'])].iloc[:, -1:].values.ravel()
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = y.reshape(-1, 1)
    
    dataset = np.hstack((X, y))
    
    # Create sequences for deep learning
    X_seq, y_seq = create_sequences(dataset, dataset[:, -1:], timesteps=buffersize, stride=0)
    
    from sklearn.utils import shuffle
    X_shuffled, y_shuffled = shuffle(X_seq, y_seq)
    
    # Standardize data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_shuffled.reshape(-1, X_shuffled.shape[-1])).reshape(X_shuffled.shape)
    
    # Reshape for Conv2D (samples, channels, timesteps, 1)
    x_train = X_std.reshape(X_std.shape[0], X_std.shape[2], X_std.shape[1], 1)
    
    # Save scaler
    with open(participant + '_scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file)
    
    # Save training data
    np.save(participant + '_X_train.npy', x_train)
    np.save(participant + '_y_train.npy', y_shuffled)
    
    # Train model using external script
    import subprocess
    subprocess.run(["ipython", "../eegnet_train.ipynb", participant])
    
    # Load trained model
    modelo = load_model(participant + '_eegnet_model.h5')
    
    # ------------------------------------------------------------------------
    # Generate Random Trial Sequence (2×2 Design)
    # ------------------------------------------------------------------------
    conditions = ["sad-music", "sad-noMusic", "happy-music", "happy-noMusic"]
    trials_per_condition = 6
    max_attempts = 100
    
    emotions_seed_list = conditions * trials_per_condition
    emotion_buffer = deque(["empty", "empty"], maxlen=2)
    action_buffer = deque(["empty", "empty"], maxlen=2)
    emotions_list = []
    attempts = 0
    
    while emotions_seed_list:
        blocked_emotion = emotion_buffer[0] if emotion_buffer[0] == emotion_buffer[1] else ""
        blocked_action = action_buffer[0] if action_buffer[0] == action_buffer[1] else ""
        
        n = rn.randint(0, len(emotions_seed_list) - 1)
        element = emotions_seed_list[n]
        target_emotion, trial_action = element.split('-')
        
        if target_emotion == blocked_emotion or trial_action == blocked_action:
            attempts += 1
            if attempts >= max_attempts:
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
    sendMarker("pause")
    msg = Message('control_change', control=15, value=127, channel=5)
    outport.send(msg)
    time.sleep(30)
    
    print("\n--------------------> [ONLINE]")
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
        time.sleep(4)
        
        if trial_action == "music":
            sendMarker("music")
            play_neutral_online()
            affectiveMusicGenerator(True, 23)
            sendMarker("stopMusic")
        else:
            sendMarker("startNoMusic")
            msg = Message('control_change', control=85, value=127, channel=5)
            outport.send(msg)
            time.sleep(30)
            sendMarker("stopNoMusic")
            msg = Message('control_change', control=85, value=127, channel=5)
            outport.send(msg)
        
        # Self-report after each trial
        self_report()
    
    # ------------------------------------------------------------------------
    # Experiment End
    # ------------------------------------------------------------------------
    play_silence(10)
    
    msg = Message('control_change', control=17, value=127, channel=5)
    outport.send(msg)
    time.sleep(8)
    
    msg = Message('control_change', control=18, value=127, channel=5)
    outport.send(msg)
    time.sleep(1)
    outport.send(msg)
    
    event.set()
    t.join()
    
    print("Execution time: %s minutes" % int((time.time() - experiment_start_time) / 60))