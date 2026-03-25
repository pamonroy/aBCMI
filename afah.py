#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 01:27:58 2026

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
Copyright (c) 2026 Pablo Andrés Monroy D'Croz

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
ASYMMETRIC FRONTAL ACTIVITY HYPOTHESIS (AFAH) Experiment
================================================================================
Real‑time affective music generation based on EEG frontal asymmetry (AF7/AF8).
The system estimates arousal and valence from EEG power spectra, normalises them,
and drives a generative MIDI music engine. The experiment includes:
- Real‑time EEG acquisition via BITalino
- LSL streaming for recording and synchronisation
- Emotion prediction (1D and smoothed valence)
- Music generation with mode, loudness, and rhythm controlled by emotion
- Protocol with random‑order happy/sad target trials

-------------------------------------------------------------------------------
DESCRIPTION
-------------------------------------------------------------------------------
This software implements a real-time affective Brain-Computer Musical Interface
(BCMI) designed to investigate the Asymmetric Frontal Activity Hypothesis (AFAH).

The system:
    1. Acquires EEG signals from prefrontal electrodes (AF7, AF8) using BITalino
    2. Computes spectral features (alpha and beta band power)
    3. Estimates emotional dimensions:
        - Valence (approach/withdrawal asymmetry)
        - Arousal (activation level)
    4. Normalizes and maps emotional state into:
        - 2D emotional space (valence, arousal)
        - 1D control signal
    5. Generates adaptive algorithmic music in real time via MIDI
    6. Sends synchronized markers via Lab Streaming Layer (LSL)
-------------------------------------------------------------------------------
EXPERIMENTAL FRAMEWORK
-------------------------------------------------------------------------------
Hypothesis:
    Asymmetric frontal alpha activity reflects emotional valence:
        - Left dominance → Positive affect
        - Right dominance → Negative affect

Derived Metrics:
    Valence  = (AF8_alpha / AF8_beta) - (AF7_alpha / AF7_beta)
    Arousal  = (AF7_beta + AF8_beta) / (AF7_alpha + AF8_alpha)

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
- pylsl
- numpy
- scipy
- mido
- bitalino
- pyxdf (optional, for offline simulation)

-------------------------------------------------------------------------------
"""

from pylsl import StreamInfo, StreamOutlet
from threading import Thread, Event
import time
from collections import deque
import numpy as np
import random
import mido
from mido import Message
import os

# Change working directory to the project location (customise as needed)
os.chdir("/Volumes/Academico/2026-TIC phd/software/participants/")

# MIDI output port (default to the first available output)
outport = mido.open_output()

# =============================================================================
# Global parameters
# =============================================================================
samplingRate = 1000          # Hz
windowSize = 4               # seconds (analysis window)
buffersize = samplingRate * windowSize   # number of samples per window
update_rate = 0.5            # seconds between emotion updates

# Threading events for controlled shutdown and reset of normalisation bounds
event = Event()
reset_maxmin = Event()

# LSL outlets (initialised later)
outletMarkers = None
outletEmotionalMarkers = None
outletPredictedEmotion = None


# =============================================================================
# Signal Processing Functions
# =============================================================================

def bandpower_welch(data, sf, band, window_sec=None, relative=False):
    """
    Compute average power in a frequency band using Welch's method.

    Parameters
    ----------
    data : array_like
        Input signal.
    sf : float
        Sampling frequency (Hz).
    band : [low, high]
        Frequency band of interest (Hz).
    window_sec : float, optional
        Length of the Welch window in seconds. If None, it is set to 2/low.
    relative : bool, default=False
        If True, return power relative to total power.

    Returns
    -------
    bp : float
        Band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        # Heuristic: use a window that captures at least two cycles of the lowest frequency
        nperseg = (2 / low) * sf

    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def RawEeg2uVolt(data_segment):
    """
    Convert raw BITalino ADC values to microvolts.

    BITalino uses a 10‑bit ADC (0‑1023) with a 3.3 V reference.
    The conversion factor is derived from the device's internal gain.

    Parameters
    ----------
    data_segment : array_like
        Raw ADC values (integers).

    Returns
    -------
    array_like
        Microvolt values.
    """
    return (data_segment / 2**10 - 0.5) * 3.3 / 41782 * 1e6


def power_spectral_density_window(electrode_signal):
    """
    Compute alpha and beta band powers for a given EEG signal.

    Parameters
    ----------
    electrode_signal : array_like
        EEG time series (in µV).

    Returns
    -------
    alpha_power : float
        Power in alpha band (8‑13.9 Hz).
    beta_power : float
        Power in beta band (14‑29.9 Hz).
    """
    global samplingRate
    alpha_range = [8, 13.9]
    beta_range = [14, 29.9]
    alpha_power = bandpower_welch(electrode_signal, samplingRate, alpha_range, relative=False)
    beta_power = bandpower_welch(electrode_signal, samplingRate, beta_range, relative=False)
    return alpha_power, beta_power


def normalize(window, baseline_time=5):
    """
    Normalise a sliding window of emotion values using a baseline period.

    Parameters
    ----------
    window : list or array
        Sequence of values (e.g., arousal or valence) over time.
    baseline_time : float, default=5
        Length (in seconds) of the baseline used to compute mean and range.

    Returns
    -------
    norm : float
        Normalised value in [0,1].
    """
    global update_rate
    max_val = np.max(window)
    min_val = np.min(window)
    last = len(window)
    first = last - int(baseline_time / update_rate)
    mean = np.mean(window[first:last])
    if max_val - min_val <= 0:
        return 0.5
    norm = (mean - min_val) / (max_val - min_val)
    return min(max(norm, 0.0), 1.0)


def long_term_normalization(current_value, max_val, min_val):
    """
    Update global min/max and normalise a value using running bounds.

    Parameters
    ----------
    current_value : float
        New value to normalise.
    max_val : float
        Current maximum observed value.
    min_val : float
        Current minimum observed value.

    Returns
    -------
    normalized : float
        Normalised value in [0,1].
    max_val : float
        Updated maximum.
    min_val : float
        Updated minimum.
    """
    if current_value > max_val:
        max_val = current_value
    if current_value < min_val:
        min_val = current_value
    if max_val - min_val <= 0:
        normalized = 0.5
    else:
        normalized = (current_value - min_val) / (max_val - min_val)
    return min(max(normalized, 0.0), 1.0), max_val, min_val


def emotion_from_2D_to_1D(a, v):
    """
    Convert a 2‑D emotion vector (arousal, valence) into a 1‑D scalar.

    The mapping projects the point onto the diagonal arousal = valence,
    preserving a single dimension of emotional intensity.

    Parameters
    ----------
    a : float
        Arousal (normalised).
    v : float
        Valence (normalised).

    Returns
    -------
    float
        1‑D emotion value.
    """
    import math
    return math.sqrt((a**2 + v**2) / 2) * math.cos(math.atan(v / a) - math.pi / 4)


def calculate_emotion(windowed_signal):
    """
    Compute arousal and valence from a 2‑channel EEG window (AF7, AF8).

    Arousal is defined as (beta_AF7 + beta_AF8) / (alpha_AF7 + alpha_AF8).
    Valence is defined as (alpha_AF8 / beta_AF8) - (alpha_AF7 / beta_AF7).

    Parameters
    ----------
    windowed_signal : list of tuples
        Each tuple contains (AF7_raw, AF8_raw) for one sample.

    Returns
    -------
    arousal : float
        Estimated arousal.
    valence : float
        Estimated valence.
    """
    AF7 = RawEeg2uVolt(np.asarray(list(zip(*windowed_signal))[0], dtype=np.float64))
    AF8 = RawEeg2uVolt(np.asarray(list(zip(*windowed_signal))[1], dtype=np.float64))

    AF7_alpha, AF7_beta = power_spectral_density_window(AF7)
    AF8_alpha, AF8_beta = power_spectral_density_window(AF8)
    valence = AF8_alpha / AF8_beta - AF7_alpha / AF7_beta
    arousal = (AF7_beta + AF8_beta) / (AF7_alpha + AF8_alpha)
    return arousal, valence


# =============================================================================
# Real‑time EEG Acquisition and Emotion Estimation (BITalino)
# =============================================================================

def affectiveEstimator(emotion_vector):
    """
    Thread function: reads raw EEG from BITalino, computes emotion,
    pushes samples to LSL, and updates a shared emotion vector.

    Parameters
    ----------
    emotion_vector : list of two floats
        Shared list that will be updated with the current
        [1D emotion, smoothed valence] estimates.
    """
    global samplingRate, buffersize, update_rate, bitalino_connected, participant, outletPredictedEmotion, event, reset_maxmin

    from bitalino import BITalino
    import numpy as np

    macAddress = "/dev/tty.BITalino-8B-B1-DevB"  # Bluetooth device address
    acqChannels = [0, 1]  # AF8 (right) on channel 0, AF7 (left) on channel 1

    # Buffers for normalisation windows
    norm_window_size = int(20 / update_rate)  # 20‑second baseline window
    norm_window = deque([[0.5, 0.5]] * norm_window_size, maxlen=norm_window_size)
    smvalence_buffer_size = int(10 / update_rate)  # 10‑second smoothed valence window
    local_smoothed_valence = deque([0.5] * smvalence_buffer_size, maxlen=smvalence_buffer_size)

    # EEG signal buffers (raw ADC values)
    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)

    nSamples = int(samplingRate / 2)  # Read half a second of data each time

    # Create LSL outlet for raw EEG
    info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32', 'eeg-data')
    outlet = StreamOutlet(info)
    print("Created Signal stream:", info.name())

    # Connect to BITalino
    device = None
    while device is None:
        try:
            print("Connecting to BITalino(%s)..." % macAddress)
            device = BITalino(macAddress)
        except (TypeError, ValueError) as e:
            print("Connection failed:", e)
            time.sleep(1)

    print('\a', end='')
    print(device.version())
    print("Connected to BITalino(%s)" % macAddress)
    device.start(samplingRate, acqChannels)
    bitalino_connected = True

    # Initialise running normalisation bounds
    min_value = 1.0
    max_value = 0.0
    min_valence = 1.0
    max_valence = 0.0

    deltaTime = time.time()  # For electrode alarm timers

    try:
        while not event.is_set():
            data_bitalino = device.read(nSamples)  # Read nSamples from all channels
            for rowData in data_bitalino:
                dataAF7 = rowData[6]  # AF7 (left) is channel 1 (index 6 in returned tuple)
                dataAF8 = rowData[5]  # AF8 (right) is channel 0 (index 5)
                localAF7.append(dataAF7)
                localAF8.append(dataAF8)
                outlet.push_sample([dataAF7, dataAF8])  # Stream raw values

            # Compute emotion from the current window
            windowed_signal = list(zip(localAF7, localAF8))
            arousal, valence = calculate_emotion(windowed_signal)

            # Update normalisation buffers
            norm_window.append([arousal, valence])
            arr = np.array(norm_window)

            # Convert 2D emotion to 1D and normalise globally
            current_1d = emotion_from_2D_to_1D(normalize(arr[:, 0]), normalize(arr[:, 1]))
            normalized, max_value, min_value = long_term_normalization(current_1d, max_value, min_value)

            # Smoothed valence
            local_smoothed_valence.append(normalize(arr[:, 1]))
            current_smoothed_valence = np.mean(local_smoothed_valence)
            normalized_smoothed_valence, max_valence, min_valence = long_term_normalization(
                current_smoothed_valence, max_valence, min_valence
            )

            # Update shared emotion vector
            emotion_vector[0] = normalized
            emotion_vector[1] = normalized_smoothed_valence
            outletPredictedEmotion.push_sample([normalized, normalized_smoothed_valence])

            # Reset normalisation bounds if requested
            if reset_maxmin.is_set():
                min_value = max_value = min_valence = max_valence = 0.5
                reset_maxmin.clear()
                print('Reset max-min normalization')

            # Electrode impedance check: values <10 indicate poor contact
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


# =============================================================================
# Simulation from a Recorded XDF File
# =============================================================================

def affectiveEstimatorFromFile(emotion_vector):
    """
    Thread function: reads prerecorded EEG data from an XDF file and
    simulates real‑time emotion estimation. Useful for offline testing.

    Parameters
    ----------
    emotion_vector : list of two floats
        Shared list that will be updated with the current
        [1D emotion, smoothed valence] estimates.
    """
    import pyxdf

    norm_window_size = int(20 / update_rate)
    norm_window = deque([[0.5, 0.5]] * norm_window_size, maxlen=norm_window_size)
    smvalence_buffer_size = int(10 / update_rate)
    local_smoothed_valence = deque([0.5] * smvalence_buffer_size, maxlen=smvalence_buffer_size)

    localAF7 = deque([512] * buffersize, maxlen=buffersize)
    localAF8 = deque([512] * buffersize, maxlen=buffersize)

    nSamples = int(samplingRate / 2)  # Simulate reading in chunks

    # Create an LSL outlet for the simulated EEG stream
    simulated_info = StreamInfo('eeg-bitalino', 'EEG', 2, samplingRate, 'float32', 'eeg-simulated-data')
    simulated_outlet = StreamOutlet(simulated_info)

    # Load XDF file (adjust path as needed)
    xdf_file = '/Volumes/Academico/2021-TIC phd/eeg/XDF/markers.xdf'
    streams = pyxdf.load_xdf(xdf_file)
    stream_list = streams[0]
    xdf_series = None
    for stream in stream_list:
        if stream['info']['name'][0] == 'eeg-bitalino':
            xdf_series = stream['time_series']
            break
    if xdf_series is None:
        raise ValueError("eeg-bitalino stream not found in XDF")

    # Extract the two channels (order: AF8, AF7)
    signal_AF7 = xdf_series[:, 0].tolist()
    signal_AF8 = xdf_series[:, 1].tolist()
    signal_full = list(zip(signal_AF7, signal_AF8))

    # Initialise normalisation bounds
    min_value = 1.0
    max_value = 0.0
    min_valence = 1.0
    max_valence = 0.0
    deltaTime = time.time()

    while signal_full and not event.is_set():
        if time.time() - deltaTime >= update_rate:
            deltaTime = time.time()
            sample = signal_full[:nSamples]
            signal_full = signal_full[nSamples:]

            for data in sample:
                # data[1] = AF7, data[0] = AF8
                localAF7.append(data[1])
                localAF8.append(data[0])
                simulated_outlet.push_sample([data[0], data[1]])

            windowed_signal = list(zip(localAF7, localAF8))
            arousal, valence = calculate_emotion(windowed_signal)

            norm_window.append([arousal, valence])
            arr = np.array(norm_window)
            current_1d = emotion_from_2D_to_1D(normalize(arr[:, 0]), normalize(arr[:, 1]))
            normalized, max_value, min_value = long_term_normalization(current_1d, max_value, min_value)

            local_smoothed_valence.append(normalize(arr[:, 1]))
            smoothed_valence = np.mean(local_smoothed_valence)
            norm_smoothed, max_valence, min_valence = long_term_normalization(smoothed_valence, max_valence, min_valence)

            emotion_vector[0] = normalized
            emotion_vector[1] = norm_smoothed
            outletPredictedEmotion.push_sample([normalized, norm_smoothed])

            if reset_maxmin.is_set():
                min_value = max_value = min_valence = max_valence = 0.5
                reset_maxmin.clear()

            # Simulate electrode alarms if values are low
            if data[0] < 10 or data[1] < 10:
                print(">> ELECTRODE ALARM <<")
                print('\a', end='')

    print('Stop capturing')
    print('\a')


# =============================================================================
# Music Generation and Marker Control
# =============================================================================

def affectiveMusicGenerator(isRealTime=True, delay=40, in_arousal=None, in_valence=None):
    """
    Generate MIDI music driven by emotional state.

    Parameters
    ----------
    isRealTime : bool, default=True
        If True, emotion is read from the global 'emotion' vector.
        If False, use provided lists of arousal and valence values.
    delay : float, default=40
        Duration of music generation in seconds.
    in_arousal : list, optional
        Sequence of arousal values (normalised) for offline playback.
    in_valence : list, optional
        Sequence of valence values (normalised) for offline playback.
    """
    global event, emotion

    # Chord progressions (MIDI note numbers) for different modes
    chordlist = np.array([
        [60, 64, 55, 59],
        [62, 65, 57, 60],
        [64, 55, 59, 62],
        [60, 65, 57, 64],
        [55, 59, 62, 65],
        [57, 60, 64, 55],
        [59, 62, 65, 57]
    ])

    # Build mode‑specific pitch sets (transposed by -3 to centre around middle C)
    modeset = np.zeros((4, 4, 7))
    mode_map = [
        [3, 6, 0, 3], [0, 3, 4, 0], [4, 0, 1, 4], [1, 4, 5, 1],
        [5, 1, 2, 5], [2, 5, 6, 2], [6, 2, 3, 6]
    ]
    for i, indices in enumerate(mode_map):
        for j in range(4):
            modeset[j, :, i] = chordlist[indices[j], :]
    modeset -= 3

    low_loudness = 50
    tick, beat, bar = 0, 0, 0
    start = time.time()
    modes = ['lydian', 'ionian', 'mixolydian', 'dorian', 'aeolian', 'phrygian', 'locrian']

    while (time.time() - start < delay) and not event.is_set():
        try:
            # At each bar (every 4 beats) update emotion and mode
            if beat % 4 == 0 and tick % 2 == 0:
                if isRealTime:
                    valence = (emotion[0] + emotion[1]) / 2
                    arousal = valence
                else:
                    if not in_valence or not in_arousal:
                        break
                    valence = in_valence.pop(0)
                    arousal = in_arousal.pop(0)

                # Select mode based on valence (0 = lydian, 6 = locrian)
                if bar == 0:
                    mode = int(7 - round(valence * 6) - 1)
                    print("mode:", modes[mode], mode)

                if isRealTime:
                    print('ARO: %.2f - VAL: %.2f' % (arousal, valence))
                    outletEmotionalMarkers.push_sample([arousal])

                # Map emotion to music parameters
                roughness = 1 - arousal       # higher arousal = less rhythmic repetition
                velocity = arousal            # higher arousal = louder notes
                voicing = valence             # higher valence = brighter harmony
                loudness = int(round(arousal * 10) / 10 * 40 + 60)

                # Activation patterns for rhythmic events
                activate1 = (np.random.rand(8) >= roughness).astype(int)
                activate2 = (np.random.rand(8) >= roughness).astype(int)

                # Brightness control: add/subtract an octave based on valence
                bright = np.random.rand(6)
                if voicing < 0.5:
                    bright = np.where(bright > voicing * 2, -1, 0)
                else:
                    bright = np.where(bright < (voicing - 0.5) * 2, 1, 0)

                # Send all‑notes‑off to all channels
                for ch in range(4):
                    outport.send(Message('control_change', control=123, value=0, channel=ch))

                # Play chord on channels 0 and 1
                for ch in [0, 1]:
                    for i in range(3):
                        note = int(modeset[bar, i, mode] + bright[i] * 12)
                        vel = random.randint(low_loudness, loudness)
                        outport.send(Message('note_on', channel=ch, note=note, velocity=vel))

                # Bass note on channel 2
                bass_note = int(modeset[bar, 0, mode] - (12 if voicing > 0.5 else 24))
                outport.send(Message('note_on', channel=2, note=bass_note,
                                    velocity=random.randint(low_loudness, loudness)))

                bar = (bar + 1) % 4

            # Update beat counter
            if tick % 2 == 0:
                beat = (beat + 1) % 4

            # Rhythmic layer on channel 3
            outport.send(Message('control_change', control=123, value=0, channel=3))
            if activate1[tick]:
                note = int(modeset[bar - 1, 0, mode] + bright[4] * 12)
                outport.send(Message('note_on', channel=3, note=note,
                                    velocity=random.randint(low_loudness, loudness)))
            if activate2[tick]:
                idx = random.randint(1, 2)
                note = int(modeset[bar - 1, idx, mode] + bright[5] * 12)
                outport.send(Message('note_on', channel=3, note=note,
                                    velocity=random.randint(low_loudness, loudness)))

            # Wait for the next tick (tempo controlled by arousal)
            time.sleep(max(0.05, 0.3 - velocity * 0.15))
            tick = (tick + 1) % 8

        except KeyboardInterrupt:
            event.set()
            break

    # All notes off at the end
    for ch in range(4):
        outport.send(Message('control_change', control=123, value=0, channel=ch))


def sendMarker(marker):
    """
    Send an integer marker to the LSL 'markers' stream.

    Parameters
    ----------
    marker : str
        Key in the markers dictionary.
    """
    markers = {
        "setVolume": 6,
        "silence": 0,
        "sad": 1,
        "happy": 3,
        "online": 7,
        "music": 9,
        "min_power": -1,
        "pause": 10
    }
    outletMarkers.push_sample([markers[marker]])


def targetMessage(emotion):
    """
    Send a marker and a MIDI control change to signal the target emotion.

    Parameters
    ----------
    emotion : str
        Either 'sad', 'neutral', or 'happy'.
    """
    control = {'sad': 82, 'neutral': 81, 'happy': 80}[emotion]
    sendMarker(emotion)
    outport.send(Message('control_change', control=control, value=127, channel=5))


def play_silence(sleepTime):
    """
    Play silence for a given duration by sending a 'silence' marker.

    Parameters
    ----------
    sleepTime : float
        Duration in seconds.
    """
    sendMarker("silence")
    time.sleep(sleepTime)


def play_volume_setting():
    """Play a short piece to set the listening volume."""
    input_valence = [0] * 4 + [1] * 8 + [0] * 4 + [1] * 8
    input_arousal = input_valence[:]
    affectiveMusicGenerator(False, 60, input_arousal, input_valence)


def play_sad_class():
    """Play a sad example (low arousal/valence)."""
    affectiveMusicGenerator(False, 60, [0] * 8, [0] * 8)


def play_happy_class():
    """Play a happy example (high arousal/valence)."""
    affectiveMusicGenerator(False, 60, [1] * 16, [1] * 16)


# =============================================================================
# Main Experiment
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Randomise trial order (no more than two consecutive same emotions)
    # -------------------------------------------------------------------------
    emotions_seed_list = ['happy'] * 8 + ['sad'] * 8
    emotions_list = []
    if emotions_seed_list:
        idx = random.randint(0, len(emotions_seed_list) - 1)
        emotions_list.append(emotions_seed_list.pop(idx))
    while emotions_seed_list:
        idx = random.randint(0, len(emotions_seed_list) - 1)
        candidate = emotions_seed_list[idx]
        if len(emotions_list) >= 2 and emotions_list[-2] == emotions_list[-1] == candidate:
            continue
        emotions_list.append(candidate)
        emotions_seed_list.pop(idx)

    # -------------------------------------------------------------------------
    # 2. Initialise LSL outlets
    # -------------------------------------------------------------------------
    infoMarkers = StreamInfo('markers', 'Markers', 1, 0, 'int32', 'eegloop-markers')
    outletMarkers = StreamOutlet(infoMarkers)
    outletEmotionalMarkers = StreamOutlet(StreamInfo('emotion', 'Markers', 1, 0, 'float32', 'music-generator-emotion-data'))
    outletPredictedEmotion = StreamOutlet(StreamInfo('predicted-emotion', 'Emotion', 2, 2, 'float32', 'emotion-data'))

    # -------------------------------------------------------------------------
    # 3. Start the EEG acquisition thread (real or simulated)
    # -------------------------------------------------------------------------
    emotion = [0.5, 0.5]
    # Uncomment the next line to use a prerecorded XDF file instead of real hardware
    # t = Thread(target=affectiveEstimatorFromFile, args=(emotion,))
    t = Thread(target=affectiveEstimator, args=(emotion,))
    t.start()

    # -------------------------------------------------------------------------
    # 4. Participant setup and baseline recording
    # -------------------------------------------------------------------------
    participant = input("\nWrite the participant CODE and press Enter: ")
    input("\nRecord on LabRecorder (adding -afah suffix) and Press Enter to start...")

    if input("Set volume (y/n)?") == 'y':
        sendMarker("setVolume")
        outport.send(Message('control_change', control=13, value=127, channel=5))
        time.sleep(10)
        play_volume_setting()

    sendMarker("pause")
    input("\nAsk participant to close eyes and wait for 30 sec. and Press Enter to continue...")
    outport.send(Message('control_change', control=16, value=127, channel=5))
    time.sleep(8)
    play_silence(15)
    sendMarker("min_power")
    reset_maxmin.set()   # Reset normalisation bounds after baseline
    play_silence(15)

    # -------------------------------------------------------------------------
    # 5. Online phase: run the trials
    # -------------------------------------------------------------------------
    print('>> ONLINE <<')
    outport.send(Message('control_change', control=15, value=127, channel=5))
    time.sleep(13)
    sendMarker('online')

    for i, val in enumerate(emotions_list):
        if event.is_set():
            break
        play_silence(13)
        targetMessage(val)
        print(f'\ntrial {i}: towards {val}')
        time.sleep(4)
        sendMarker("music")
        affectiveMusicGenerator(True, 30)

    # -------------------------------------------------------------------------
    # 6. End of experiment: send final markers and shutdown
    # -------------------------------------------------------------------------
    play_silence(20)
    outport.send(Message('control_change', control=17, value=127, channel=5))
    time.sleep(8)
    outport.send(Message('control_change', control=18, value=127, channel=5))
    time.sleep(1)

    event.set()   # Signal the acquisition thread to stop
    t.join()