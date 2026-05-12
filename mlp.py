#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 01:23:48 2023

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
Copyright (c) 2023-2026 Pablo Andrés Monroy D'Croz

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
MULTILAYER PERCEPTRON (MLP) - EEG-based Affective Music Generation Experiment
================================================================================
Real‑time affective music generation based on EEG frontal brain activity (AF7/AF8)
using a Multi-Layer Perceptron (MLP) classifier. The system extracts power
spectral features, trains a neural network during calibration, and generates
adaptive MIDI music based on real-time emotion predictions.

The experiment includes:
- Real‑time EEG acquisition via BITalino or offline simulation from XDF files
- Power spectral feature extraction (theta, alpha, low/high beta, gamma bands)
- MLP classifier training on individual participant data
- LSL streaming for recording and synchronisation
- Emotion prediction (arousal and smoothed valence)
- Adaptive music generation with mode, loudness, tempo, and rhythm controlled by emotion
- Protocol with randomized happy/sad target trials

-------------------------------------------------------------------------------
DESCRIPTION
-------------------------------------------------------------------------------
This software implements a real-time affective Brain-Computer Musical Interface
(BCMI) using a Multi-Layer Perceptron (MLP) classifier to predict emotional
states from EEG signals.

The system:
    1. Acquires EEG signals from prefrontal electrodes (AF7, AF8) using BITalino
    2. Computes spectral features (theta, alpha, low/high beta, gamma bands)
    3. Calibration phase:
        - Collects baseline (eyes closed, no music)
        - Records emotional states (sad, neutral, happy)
        - Trains MLP classifier with 10 spectral features
    4. Online classification phase:
        - Extracts features in real-time
        - Predicts arousal probability (happy vs. sad)
        - Computes smoothed valence (10-second moving average)
    5. Generates adaptive algorithmic music in real-time via MIDI with:
        - Mode selection based on valence (7 modes from Lydian to Locrian)
        - Tempo and rhythmic density controlled by arousal
        - Loudness and brightness controlled by emotional intensity
    6. Sends synchronized markers via Lab Streaming Layer (LSL)
    
-------------------------------------------------------------------------------
EXPERIMENTAL FRAMEWORK
-------------------------------------------------------------------------------
Hypothesis:
    EEG spectral features from prefrontal electrodes (AF7, AF8) can predict
    emotional states (happy vs. sad) using a Multi-Layer Perceptron classifier.

Feature Set (10 features per window):
    - Theta power: 4-7.9 Hz (both AF7 and AF8)
    - Alpha power: 8-13.9 Hz (both AF7 and AF8)
    - Low Beta power: 14-21.9 Hz (both AF7 and AF8)
    - High Beta power: 22-29.9 Hz (both AF7 and AF8)
    - Gamma power: 30-47 Hz (both AF7 and AF8)

Musical Mapping:
    - Valence → Musical mode (7 modes from Lydian/happy to Locrian/sad)
    - Arousal → Tempo, rhythm density, loudness
    - Smoothed valence → Stable mode selection

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
Core Dependencies:
- pylsl: Lab Streaming Layer for data synchronization
- numpy: Numerical computations
- scipy: Signal processing (Welch's method, integration)
- mido: MIDI communication for music generation
- bitalino: BITalino hardware interface (optional)
- pyxdf: XDF file loading for offline simulation (optional)

Machine Learning:
- scikit-learn: MLPClassifier, StandardScaler

-------------------------------------------------------------------------------
HARDWARE REQUIREMENTS
-------------------------------------------------------------------------------
- BITalino (r)evolution with EEG sensor kit
- Electrodes: AF7 (left prefrontal), AF8 (right prefrontal)
- MIDI synthesizer or DAW (e.g., Ableton Live) for audio output

-------------------------------------------------------------------------------
EXPERIMENT PROTOCOL
-------------------------------------------------------------------------------
Phase 1: Volume Setting (optional)
    - Alternating sad/happy music to set comfortable listening level

Phase 2: Calibration (Training Data Collection)
    - 3 trials: idle (eyes closed), sad, happy, neutral (randomized order)
    - Each trial: 10s baseline + emotional condition music

Phase 3: Model Training
    - Baseline subtraction and standardization
    - MLP training with 30-30 hidden layers, ReLU activation, Adam optimizer

Phase 4: Online Phase
    - 4 trials: 2 happy + 2 sad (randomized order, no consecutive duplicates)
    - Real-time emotion prediction and adaptive music generation

-------------------------------------------------------------------------------
FILE OUTPUTS
-------------------------------------------------------------------------------
- {participant}_training_data.csv: Spectral features with class labels
- {participant}_model.pickle: Trained MLP classifier
- {participant}_standardScaler.pickle: Fitted StandardScaler for online use
- {participant}-mlp.xdf: EEG raw signal, markers, and predicted-emotion streams

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
from scipy import signal, integrate

# Change working directory to project location
os.chdir("/Volumes/Academico/2026-TIC phd/software/participants/")

# MIDI output port
outport = mido.open_output()

# ============================================================================
# Global Parameters
# ============================================================================
samplingRate = 1000          # Hz
windowSize = 4               # Analysis window in seconds
buffersize = samplingRate * windowSize
update_rate = 0.5            # Emotion update rate in seconds

# Signal buffers (initialized later)
AF7 = deque([], buffersize)  # Left prefrontal channel
AF8 = deque([], buffersize)  # Right prefrontal channel

# Threading events
event = Event()
is_memorizing = Event()
is_online = Event()
bitalino_connected = False

# LSL outlets (initialized later)
outletMarkers = None
outletEmotionalMarkers = None
outletPredictedEmotion = None


# ============================================================================
# Signal Processing Functions
# ============================================================================

def select_lsl_stream(stream_name, streams):
    """Select a specific LSL stream from loaded XDF data.
    
    Parameters
    ----------
    stream_name : str
        Name of the stream to select
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


def bandpower_welch(data, sf, band, window_sec=None, relative=False):
    """Compute band power using Welch's method.
    
    Parameters
    ----------
    data : array_like
        Input signal
    sf : float
        Sampling frequency
    band : list
        [low, high] frequency band in Hz
    window_sec : float, optional
        Window length in seconds
    relative : bool, default=False
        If True, return relative power (normalized by total power)
    
    Returns
    -------
    float
        Band power
    """
    band = np.asarray(band)
    low, high = band
    
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = integrate.simpson(psd[idx_band], dx=freq_res)
    
    if relative:
        bp /= integrate.simpson(psd, dx=freq_res)
    
    return bp


def RawEeg2uVolt(data_segment):
    """Convert raw BITalino ADC values to microvolts.
    
    Parameters
    ----------
    data_segment : array_like
        Raw ADC values (0-1023)
    
    Returns
    -------
    array_like
        Microvolt values
    """
    return (data_segment / 2**10 - 0.5) * 3.3 / 41782 * 1e6


def power_spectral_density(electrode_signal, band):
    """Compute power in specific frequency band for an EEG signal.
    
    Parameters
    ----------
    electrode_signal : array_like
        EEG signal in raw ADC units
    band : str
        Frequency band: 'theta', 'alpha', 'low_beta', 'high_beta', 'beta', 'gamma'
    
    Returns
    -------
    float
        Band power in microvolts squared
    """
    global samplingRate
    
    # Define frequency ranges
    band_ranges = {
        'theta': (4, 7.9),
        'alpha': (8, 13.9),
        'low_beta': (14, 21.9),
        'high_beta': (22, 29.9),
        'beta': (14, 29.9),
        'gamma': (30, 47)
    }
    
    low, high = band_ranges.get(band, (30, 47))
    
    # Convert to microvolts and compute power
    electrode_signal = RawEeg2uVolt(electrode_signal)
    power = bandpower_welch(electrode_signal, samplingRate, [low, high], 
                           window_sec=1, relative=False)
    
    return power


def get_power(left_electrode, right_electrode):
    """Extract power features from both EEG channels.
    
    Parameters
    ----------
    left_electrode : array_like
        AF7 signal (left prefrontal)
    right_electrode : array_like
        AF8 signal (right prefrontal)
    
    Returns
    -------
    list
        Feature vector with 10 values: [theta_AF7, theta_AF8, alpha_AF7, alpha_AF8,
                                       low_beta_AF7, low_beta_AF8, high_beta_AF7,
                                       high_beta_AF8, gamma_AF7, gamma_AF8]
    """
    AF7_signal = np.asarray(left_electrode, dtype=np.float64)
    AF8_signal = np.asarray(right_electrode, dtype=np.float64)
    
    # Compute powers for each band and channel
    theta_AF7 = power_spectral_density(AF7_signal, 'theta')
    alpha_AF7 = power_spectral_density(AF7_signal, 'alpha')
    low_beta_AF7 = power_spectral_density(AF7_signal, 'low_beta')
    high_beta_AF7 = power_spectral_density(AF7_signal, 'high_beta')
    gamma_AF7 = power_spectral_density(AF7_signal, 'gamma')
    
    theta_AF8 = power_spectral_density(AF8_signal, 'theta')
    alpha_AF8 = power_spectral_density(AF8_signal, 'alpha')
    low_beta_AF8 = power_spectral_density(AF8_signal, 'low_beta')
    high_beta_AF8 = power_spectral_density(AF8_signal, 'high_beta')
    gamma_AF8 = power_spectral_density(AF8_signal, 'gamma')
    
    return [theta_AF7, theta_AF8, alpha_AF7, alpha_AF8, low_beta_AF7, 
            low_beta_AF8, high_beta_AF7, high_beta_AF8, gamma_AF7, gamma_AF8]


# ============================================================================
# Data Acquisition Functions
# ============================================================================

def affectiveEstimator(power_vector, smooth_valence):
    """Capture real-time EEG from BITalino device and compute features.
    
    Parameters
    ----------
    power_vector : list
        Shared list to store current power features (10 values)
    smooth_valence : list
        Shared list to store smoothed valence value
    """
    global samplingRate, buffersize, update_rate, segmento, bitalino_connected, participant
    
    # Initialize smoothed valence buffer (10 seconds)
    smvalence_buffer_size = int(10 / update_rate)
    local_smoothed_valence = deque([0.5] * smvalence_buffer_size, maxlen=smvalence_buffer_size)
    
    from bitalino import BITalino
    macAddress = "/dev/tty.BITalino-8B-B1-DevB"
    
    # Initialize EEG buffers
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)
    
    nSamples = int(samplingRate / 2)  # Read every 0.5 seconds
    
    # Create LSL outlet for raw EEG
    info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32', 'eeg-data')
    outlet = StreamOutlet(info)
    print("Created Signal stream: %s" % info.name())
    
    # Connect to BITalino
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
    device.start(samplingRate, [0, 1])
    bitalino_connected = True
    
    deltaTime = time.time()
    
    try:
        while not event.is_set():
            data_bitalino = device.read(nSamples)
            for rowData in data_bitalino:
                dataAF7 = rowData[6]   # Left prefrontal (channel 1)
                dataAF8 = rowData[5]   # Right prefrontal (channel 0)
                localAF7.append(dataAF7)
                localAF8.append(dataAF8)
                outlet.push_sample([dataAF7, dataAF8])
            
            # Compute power features
            signal_AF7 = list(localAF7)
            signal_AF8 = list(localAF8)
            vector = get_power(signal_AF7, signal_AF8)
            
            # Update shared power vector
            for idx, val in enumerate(vector):
                power_vector[idx] = val
            
            # Store segment if memorizing
            if is_memorizing.is_set():
                segmento.append(vector)
            
            # Online classification
            if is_online.is_set():
                current_power = np.asarray(power_vector).reshape(1, -1)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)
                arousal = clf.predict_proba(current_power_std)[0, 0]
                
                local_smoothed_valence.append(arousal)
                smooth_valence[0] = np.mean(local_smoothed_valence)
                outletPredictedEmotion.push_sample([arousal, smooth_valence[0]])
            
            # Electrode impedance check
            if time.time() - deltaTime > 0.1:
                deltaTime = time.time()
                if dataAF7 < 10:
                    print(">> ELECTRODE ALARM AF7 <<")
                    print('\a', end='')
                if dataAF8 < 10:
                    print(">> ELECTRODE ALARM AF8 <<")
                    print('\a', end='')
    
    finally:
        print('Stop capturing')
        device.stop()
        device.close()
        sys.exit(0)


def affectiveEstimatorFromFile(power_vector, smooth_valence):
    """Simulate real-time capture from pre-recorded XDF file.
    
    Parameters
    ----------
    power_vector : list
        Shared list to store current power features
    smooth_valence : list
        Shared list to store smoothed valence value
    """
    import pyxdf as xdf
    
    global samplingRate, buffersize, update_rate, segmento, baseline
    
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
    
    nSamples = int(samplingRate / 2)
    
    # Load XDF file
    xdf_file = '/Volumes/Academico/2021-TIC phd/eeg/XDF/markers.xdf'
    streams = xdf.load_xdf(xdf_file)
    xdf_series = select_lsl_stream('eeg-bitalino', streams)
    
    if xdf_series is False:
        raise ValueError("eeg-bitalino stream not found in XDF file")
    
    # Extract signals (order: AF7, AF8)
    signal_AF7 = xdf_series[:, 0].tolist()
    signal_AF8 = xdf_series[:, 1].tolist()
    signal_full = list(zip(signal_AF7, signal_AF8))
    
    deltaTime = time.time()
    
    while signal_full and not event.is_set():
        if time.time() - deltaTime >= update_rate:
            deltaTime = time.time()
            
            # Process chunk
            sample = signal_full[:nSamples]
            signal_full = signal_full[nSamples:]
            
            for data in sample:
                localAF7.append(data[0])
                localAF8.append(data[1])
                simulated_outlet.push_sample([data[0], data[1]])
            
            # Compute features
            signal_AF7_list = list(localAF7)
            signal_AF8_list = list(localAF8)
            vector = get_power(signal_AF7_list, signal_AF8_list)
            
            # Update shared vector
            for idx, val in enumerate(vector):
                power_vector[idx] = val
            
            # Store segment if memorizing
            if is_memorizing.is_set():
                segmento.append(vector)
            
            # Online classification
            if is_online.is_set():
                current_power = np.asarray(power_vector).reshape(1, -1)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)
                arousal = clf.predict_proba(current_power_std)[0, 0]
                
                local_smoothed_valence.append(arousal)
                smooth_valence[0] = np.mean(local_smoothed_valence)
                outletPredictedEmotion.push_sample([arousal, smooth_valence[0]])
            
            # Electrode check
            if data[0] < 10 or data[1] < 10:
                print(">> ELECTRODE ALARM <<")
                print('\a', end='')
    
    print('Stop capturing')
    print('\a')
    sys.exit(0)


# ============================================================================
# Music Generation Functions
# ============================================================================

def affectiveMusicGenerator(isRealTime=True, delay=40, in_arousal=None, in_valence=None):
    """Generate affective music based on emotional state.
    
    Parameters
    ----------
    isRealTime : bool, default=True
        Use real-time emotion prediction
    delay : float, default=40
        Duration in seconds
    in_arousal : list, optional
        Pre-recorded arousal values for offline playback
    in_valence : list, optional
        Pre-recorded valence values for offline playback
    """
    global event, power, baseline, sc, clf, svalence, outletEmotionalMarkers
    
    # Chord progressions for different modes (MIDI note numbers)
    chordlist = np.array([
        [60, 64, 55, 59], [62, 65, 57, 60], [64, 55, 59, 62],
        [60, 65, 57, 64], [55, 59, 62, 65], [57, 60, 64, 55],
        [59, 62, 65, 57]
    ])
    
    # Mode definitions (4 bars × 4 chords × 7 modes)
    modeset = np.zeros((4, 4, 7))
    mode_map = [
        [3, 6, 0, 3], [0, 3, 4, 0], [4, 0, 1, 4], [1, 4, 5, 1],
        [5, 1, 2, 5], [2, 5, 6, 2], [6, 2, 3, 6]
    ]
    
    for mode_idx, indices in enumerate(mode_map):
        for bar_idx in range(4):
            modeset[bar_idx, :, mode_idx] = chordlist[indices[bar_idx], :]
    
    modeset -= 3  # Transpose down by 3 semitones
    
    mode_names = ['lydian', 'ionian', 'mixolydian', 'dorian', 'aeolian', 'phrygian', 'locrian']
    low_loudness = 50
    tick = beat = bar = 0
    start = time.time()
    
    try:
        while (time.time() - start < delay) and not event.is_set():
            # Update harmony at bar boundaries (every 4 beats)
            if beat % 4 == 0 and tick % 2 == 0:
                if isRealTime:
                    # Get real-time prediction
                    current_power = np.asarray(power).reshape(1, -1)
                    current_power = current_power - baseline
                    current_power_std = sc.transform(current_power)
                    arousal = clf.predict_proba(current_power_std)[0, 0]
                    valence = arousal  # Simplified: use arousal as valence
                else:
                    # Use pre-recorded values
                    if not in_valence or not in_arousal:
                        break
                    valence = in_valence.pop(0)
                    arousal = in_arousal.pop(0)
                
                # Select mode based on valence (0-6 mapping)
                if bar == 0:  
                    if isRealTime:
                        valence = svalence[0]  ## 10 sec average smoothed valence
                        arousal = valence
                
                    mode = 7-round(valence*6)-1; # set harmonic mode based on valence
                    print("Mode:", mode_names[mode], mode)
                    
                
                if isRealTime:
                    print('ARO: %.2f - VAL: %.2f - sVAL: %.2f' % 
                          (arousal, valence, svalence[0]))
                    outletEmotionalMarkers.push_sample([arousal])
                
                # Map emotion to musical parameters
                roughness = 1 - arousal
                velocity = arousal
                voicing = valence
                loudness = int(round(arousal * 10) / 10 * 40 + 60)
                
                # Generate rhythmic patterns
                activate1 = (np.random.rand(8) >= roughness).astype(int)
                activate2 = (np.random.rand(8) >= roughness).astype(int)
                
                # Generate brightness offsets
                bright = np.random.rand(6)
                if voicing < 0.5:
                    bright = np.where(bright > voicing * 2, -1, 0)
                else:
                    bright = np.where(bright < (voicing - 0.5) * 2, 1, 0)
                
                # Send all notes off
                for ch in range(4):
                    outport.send(Message('control_change', control=123, value=0, channel=ch))
                
                # Play chord on channels 0 and 1
                for ch in [0, 1]:
                    for i in range(3):
                        note = int(modeset[bar, i, mode] + bright[i] * 12)
                        vel = rn.randint(low_loudness, loudness)
                        outport.send(Message('note_on', channel=ch, note=note, velocity=vel))
                
                # Play bass on channel 2
                bass_note = int(modeset[bar, 0, mode] - (12 if voicing > 0.5 else 24))
                outport.send(Message('note_on', channel=2, note=bass_note,
                                    velocity=rn.randint(low_loudness, loudness)))
                
                bar = (bar + 1) % 4
            
            # Update beat counter
            if tick % 2 == 0:
                beat = (beat + 1) % 4
            
            # Play melody on channel 3
            outport.send(Message('control_change', control=123, value=0, channel=3))
            if activate1[tick]:
                note = int(modeset[bar-1, 0, mode] + bright[4] * 12)
                outport.send(Message('note_on', channel=3, note=note,
                                    velocity=rn.randint(low_loudness, loudness)))
            if activate2[tick]:
                idx = rn.randint(1, 2)
                note = int(modeset[bar-1, idx, mode] + bright[5] * 12)
                outport.send(Message('note_on', channel=3, note=note,
                                    velocity=rn.randint(low_loudness, loudness)))
            
            # Tempo control
            time.sleep(0.3 - velocity * 0.15)
            tick = (tick + 1) % 8
    
    except KeyboardInterrupt:
        event.set()
    
    finally:
        # All notes off
        for ch in range(4):
            outport.send(Message('control_change', control=123, value=0, channel=ch))
        
        if event.is_set():
            sys.exit(0)


# ============================================================================
# Experiment Control Functions
# ============================================================================

def sendMarker(marker):
    """Send an integer marker to LSL stream.
    
    Parameters
    ----------
    marker : str
        Marker name from dictionary
    """
    lsl_markers = {
        "setVolume": 6, "silence": 0, "neutral": 2, "sad": 1,
        "happy": 3, "modeling": 8, "online": 7, "music": 9,
        "min_power": -1, "pause": 10
    }
    outletMarkers.push_sample([lsl_markers[marker]])


def targetMessage(emotion):
    """Send target emotion message to Ableton Live.
    
    Parameters
    ----------
    emotion : str
        'sad', 'neutral', or 'happy'
    """
    control_map = {'sad': 82, 'neutral': 81, 'happy': 80}
    sendMarker(emotion)
    outport.send(Message('control_change', control=control_map[emotion], 
                        value=127, channel=5))


def play_silence(sleepTime):
    """Play silence for specified duration.
    
    Parameters
    ----------
    sleepTime : float
        Duration in seconds
    """
    sendMarker("silence")
    time.sleep(sleepTime)


def play_volume_setting():
    """Play volume setting routine."""
    input_valence = [0]*4 + [1]*8 + [0]*4 + [1]*8
    affectiveMusicGenerator(False, 60, input_valence, input_valence)


def play_sad_class():
    """Play sad emotion example."""
    affectiveMusicGenerator(False, 60, [0]*8, [0]*8)


def play_neutral_class():
    """Play neutral emotion example."""
    affectiveMusicGenerator(False, 60, [0.5]*12, [0.5]*12)


def play_happy_class():
    """Play happy emotion example."""
    affectiveMusicGenerator(False, 60, [1]*16, [1]*16)


# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # Initialize LSL outlets
    # ------------------------------------------------------------------------
    indice = rn.randint(3, 7)  # Avoid stream name conflicts
    infoMarkers = StreamInfo('markers', 'Markers', 1, 0, 'int32', 
                             'eegloop-markers' + str(indice))
    outletMarkers = StreamOutlet(infoMarkers)
    
    outletEmotionalMarkers = StreamOutlet(
        StreamInfo('emotion', 'Markers', 1, 0, 'float32', 'music-generator-emotion-data')
    )
    
    outletPredictedEmotion = StreamOutlet(
        StreamInfo('predicted-emotion', 'Emotion', 2, 2, 'float32', 'emotion-data')
    )
    
    # Shared data structures
    power = [1.0] * 10  # Power features (10 values)
    svalence = [0.5]    # Smoothed valence
    
    # Start acquisition thread
    if bitalino_connected:
        t = Thread(target=affectiveEstimatorFromFile, args=(power, svalence))
    else:
        t = Thread(target=affectiveEstimator, args=(power, svalence))
    t.start()
    
    # Wait for connection
    while not bitalino_connected:
        time.sleep(0.1)
    
    # ------------------------------------------------------------------------
    # Participant setup
    # ------------------------------------------------------------------------
    participant = input("\nWrite the participant CODE and press Enter: ")
    input("\nRecord on LabRecorder (adding -mlp suffix) and Press Enter to start...")
    
    # Volume setting
    if input("Set volume (y/n)? ").lower() == 'y':
        sendMarker("setVolume")
        outport.send(Message('control_change', control=13, value=127, channel=5))
        time.sleep(10)
        play_volume_setting()
    
    # ------------------------------------------------------------------------
    # Calibration phase
    # ------------------------------------------------------------------------
    sendMarker("pause")
    input("Ask participant to close eyes and press Enter to continue...")
    
    print("\n--------------------> [CALIBRATING]")
    outport.send(Message('control_change', control=14, value=127, channel=5))
    time.sleep(17)
    sendMarker("pause")
    
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
        
        # Record idle baseline
        segmento = []
        is_memorizing.set()
        play_silence(10)
        is_memorizing.clear()
        
        if segmento:  # Only create DataFrame if there's data
            df_temp = pd.DataFrame(segmento, columns=feature_columns)
            df_temp['class'] = 'idle'
            print('idle:', len(df_temp))
            df = pd.concat([df, df_temp], ignore_index=True)
        
        # Record emotional condition
        segmento = []
        is_memorizing.set()
        sendMarker("music")
        eval("play_" + val + "_class()")
        is_memorizing.clear()
        
        if segmento:  # Only create DataFrame if there's data
            df_temp = pd.DataFrame(segmento, columns=feature_columns)
            df_temp['class'] = val
            print(val + ':', len(df_temp))
            df = pd.concat([df, df_temp], ignore_index=True)
    
    # Check if we have data
    if df.empty:
        print("ERROR: No data collected during calibration!")
        sys.exit(1)
    
    # Save training data
    df.to_csv(participant + '_training_data.csv', sep='\t', index=False)
    print(f"\nTraining data saved to {participant}_training_data.csv")
    
    # ------------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------------
    print("\n--------------------> [MODELING]")
    sendMarker('modeling')
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    
    # Prepare data
    X = df[feature_columns].values  # All features
    y = df['class'].values  # Class labels
    
    # Compute baseline from idle condition
    idle_mask = y == 'idle'
    if not np.any(idle_mask):
        print("ERROR: No idle data found for baseline calculation!")
        sys.exit(1)
    
    baseline = X[idle_mask].mean(axis=0)
    print(f"Baseline vector shape: {baseline.shape} (should be 10)")
    
    # Filter sad and happy conditions for training
    train_mask = np.isin(y, ['sad', 'happy'])
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    if len(X_train) == 0:
        print("ERROR: No sad/happy data found for training!")
        sys.exit(1)
    
    # Subtract baseline and standardize
    X_cleared = X_train - baseline
    sc = StandardScaler()
    X_std = sc.fit_transform(X_cleared)
    
    # Train neural network
    clf = MLPClassifier(hidden_layer_sizes=(30, 30), activation='relu', 
                        solver='adam', alpha=0.01, max_iter=650, random_state=42)
    clf.fit(X_std, y_train)
    
    print(f"Model trained on {len(X_train)} samples")
    
    # Save model and scaler
    with open(participant + '_model.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(participant + '_standardScaler.pickle', 'wb') as f:
        pickle.dump(sc, f)
    
    print(f"Model saved to {participant}_model.pickle")
    
    sendMarker("pause")
    time.sleep(10)
    
    # ------------------------------------------------------------------------
    # Online phase
    # ------------------------------------------------------------------------
    print("\n--------------------> [ONLINE]")
    is_online.set()
    
    outport.send(Message('control_change', control=15, value=127, channel=5))
    time.sleep(13)
    sendMarker('online')
    
    # Pseudorandomize trial order (no consecutive duplicates)
    trials_per_condition = 8
    emotions_seed_list = ['happy'] * trials_per_condition + ['sad'] * trials_per_condition
    emotions_list = []
    if emotions_seed_list:
        idx = rn.randint(0, len(emotions_seed_list) - 1)
        emotions_list.append(emotions_seed_list.pop(idx))
    
    while emotions_seed_list:
        idx = rn.randint(0, len(emotions_seed_list) - 1)
        candidate = emotions_seed_list[idx]
        if (len(emotions_list) >= 2 and 
            emotions_list[-2] == emotions_list[-1] == candidate):
            continue
        emotions_list.append(candidate)
        emotions_seed_list.pop(idx)
    
    print(f"\nRunning {len(emotions_list)} trials...")
    
    # Run trials
    for i, val in enumerate(emotions_list):
        if event.is_set():
            sys.exit(0)
        
        play_silence(13)
        targetMessage(val)
        print('\ntrial %d: towards %s' % (i+1, val))
        time.sleep(4)
        sendMarker("music")
        affectiveMusicGenerator(True, 30)
    
    # ------------------------------------------------------------------------
    # Experiment end
    # ------------------------------------------------------------------------
    play_silence(20)
    outport.send(Message('control_change', control=17, value=127, channel=5))
    time.sleep(8)
    outport.send(Message('control_change', control=18, value=127, channel=5))
    time.sleep(1)
    
    event.set()
    t.join()
    
    print("\nExperiment completed successfully!")