# aBCMI: Affective Brain-Computer Musical Interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10.9](https://img.shields.io/badge/python-3.10.9-blue.svg)](https://www.python.org/downloads/release/python-3109/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19059357.svg)](https://doi.org/10.5281/zenodo.19059357)

Real-time affective Brain-Computer Musical Interface (BCMI) systems that generate adaptive music based on EEG frontal asymmetry (AF7/AF8 electrodes). This repository contains four complete experimental frameworks for investigating emotion recognition from EEG signals and generating music in real-time.

**This repository is a direct contribution supporting the doctoral thesis:**  
*"EEG-Driven Brain–Computer Musical Interfaces for Emotion Self-Induction: A Comparative Study of Deterministic and Machine Learning Approaches"*

**Repository URL:** [https://github.com/pamonroy/aBCMI/](https://github.com/pamonroy/aBCMI/)

## Overview

This project implements four different approaches to EEG-based emotion recognition and affective music generation:

1. **Asymmetric Frontal Activity Hypothesis (AFAH)** - Real-time valence/arousal estimation using alpha/beta band power ratios
2. **Multi-Layer Perceptron (MLP)** - Neural network classification using power spectral features
3. **Linear Discriminant Analysis (LDA)** - 2×2 factorial design with statistical classification
4. **EEGNet Deep Learning** - Convolutional neural network with LSTM for end-to-end learning

Each implementation provides a complete experimental pipeline including real-time EEG acquisition, signal processing, machine learning/deep learning, and adaptive MIDI music generation.

## Institutions

- **Universitat Pompeu Fabra** (Barcelona, Spain)
- **Universidad Icesi** (Cali, Colombia)

## Technical Specifications

### Software Environment
- **Python Version**: 3.10.9
- **LSL Recording Software**: Lab Recorder version 1.14.0
- **Operating System**: macOS / Linux (compatible)

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.23.2 | Numerical computations and array handling |
| pandas | 2.2.2 | Data management and CSV export |
| mido | 1.2.10 | MIDI communication for music generation |
| scikit-learn | 1.5.1 | MLP and LDA classifiers, StandardScaler |
| pynput | 1.7.7 | Keyboard input for self-report collection |
| pyxdf | 1.16.3 | XDF file loading for offline simulation |
| bitalino | 1.2.6 | BITalino hardware interface |
| pylsl | 1.15.0 | Lab Streaming Layer synchronization |
| tensorflow-macos | 2.9.2 | EEGNet deep learning framework (macOS) |

**Note for Linux/Windows users**: Replace `tensorflow-macos` with `tensorflow` version 2.9.2 or higher.

## Citation

If you use this software in your research, please cite:

```bibtex
@phdthesis{Monroy_2026_BCMI,
  author = {Pablo Andrés Monroy D'Croz},
  title = {EEG-Driven Brain–Computer Musical Interfaces for Emotion Self-Induction: A Comparative Study of Deterministic and Machine Learning Approaches},
  year = {2026},
  school = {Universitat Pompeu Fabra},
  address = {Barcelona, Spain},
  note = {Repository: https://github.com/pamonroy/aBCMI/}
}

@software{Monroy_aBCMI_2026,
  author = {Pablo Andrés Monroy D'Croz},
  title = {aBCMI: Affective Brain-Computer Musical Interface},
  year = {2026},
  url = {https://github.com/pamonroy/aBCMI/},
  version = {1.0},
  doi = {[Add DOI if available]}
}
```

## Public Datasets

This thesis contributes **four publicly available datasets** collected from **82 participants** across sequential BCMI experiments, totaling more than **50 hours of synchronized recordings**. All datasets are available on Zenodo:

[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-10.5281/zenodo.19059357-blue.svg)](https://zenodo.org/records/19059357)

### Dataset Overview

| Dataset | Paradigm | Participants | Duration (approx.) | EEG Sampling Rate | Key Features |
|---------|----------|--------------|-------------------|-------------------|--------------|
| **AFAH** | Asymmetric Frontal Activity Hypothesis | 23 | 14 min | 1000 Hz | Dual-channel EEG, valence-arousal predictions |
| **MLP** | Multi-Layer Perceptron | 23 | 20 min | 1000 Hz | Labeled training data, trained models |
| **EEGNet** | Deep Learning | 26 | 30 min | 100 Hz | 6 synchronized streams, real-time predictions, self-reports |
| **LDA** | Linear Discriminant Analysis | 33 | 30 min | 1000 Hz | Structured similar to EEGNet with higher sampling rate |
| **TOTAL** | **All Paradigms** | **82** | **50+ hours** | **100-1000 Hz** | **Synchronized recordings + questionnaires** |

### Included Materials
Each dataset includes:
- **Raw EEG recordings** (synchronized LSL streams)
- **Experimental markers** for trial segmentation
- **Pre- and post-experiment questionnaires**
- **Trained models** (where applicable)
- **Documentation** (data format specifications, experimental protocols)
- **Code snippets** for data loading and analysis

### Zenodo Access
The complete datasets are publicly available at:
**https://zenodo.org/records/19059357**

  
# Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation](#installation)
- [Experimental Methods](#experimental-methods)
  - [Method 1: AFAH (Asymmetric Frontal Activity Hypothesis)](#method-1-afah-asymmetric-frontal-activity-hypothesis)
  - [Method 2: MLP (Multi-Layer Perceptron)](#method-2-mlp-multi-layer-perceptron)
  - [Method 3: LDA (Linear Discriminant Analysis)](#method-3-lda-linear-discriminant-analysis)
  - [Method 4: EEGNet Deep Learning](#method-4-eegnet-deep-learning)
- [Common Protocol](#common-protocol)
- [File Outputs](#file-outputs)
- [MIDI Configuration](#midi-configuration)
- [Lab Streaming Layer (LSL) Configuration](#lab-streaming-layer-lsl-configuration)
- [Thesis Context](#thesis-context)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

# Features

- **Real-time EEG acquisition** from BITalino device (AF7, AF8 electrodes)
- **Multiple classification approaches:**
  - AFAH (Asymmetric Frontal Activity Hypothesis)
  - MLP (Multi-Layer Perceptron)
  - LDA (Linear Discriminant Analysis)
  - EEGNet Deep Learning
- **Offline simulation** using pre-recorded XDF files
- **Lab Streaming Layer (LSL)** integration for data synchronization
- **Adaptive MIDI music generation** with emotion-controlled parameters:
  - Musical mode (7 Greek modes from Lydian to Locrian)
  - Tempo and rhythm density (controlled by arousal)
  - Loudness and brightness (controlled by emotional intensity)
  - Harmonic progression (I-IV-V-I pattern)
- **2×2 factorial design** (sad/happy × music/no-music) for LDA and EEGNet
- **Self-report mechanism** (1–9 scale) for emotional state validation
- **Comprehensive data logging** for offline analysis

# Hardware Requirements

- **BITalino (r)evolution** with EEG sensor kit
- **Electrodes**: AF7 (left prefrontal), AF8 (right prefrontal)
- **Digital Audio Workstation** (e.g., Ableton Live) for audio output
- **Computer** with Bluetooth capability for BITalino connection
- **Keyboard** for self-report responses (keys 1–9)

# Software Dependencies

## Core Dependencies with Version Specifications

| Package | Version | Purpose |
|---|---|---|
| `numpy` | `1.23.2` | Numerical computations and array handling |
| `pandas` | `2.2.2` | Data management and CSV export |
| `mido` | `1.2.10` | MIDI communication for music generation |
| `scikit-learn` | `1.5.1` | MLP and LDA classifiers, `StandardScaler` |
| `pynput` | `1.7.7` | Keyboard input for self-report collection |
| `pyxdf` | `1.16.3` | XDF file loading for offline simulation |
| `bitalino` | `1.2.6` | BITalino hardware interface |
| `pylsl` | `1.15.0` | Lab Streaming Layer synchronization |
| `tensorflow-macos` | `2.9.2` | EEGNet deep learning framework (macOS) |
| `matplotlib` | `3.4.3` | Training curve visualization |
| `scipy` | `1.7.3` | Signal processing (Welch's method) |

## Important Notes

- **LSL Recording:** Lab Recorder version `1.14.0` was used for recording all LSL streams.
- **Python Version:** All code was developed and tested with Python `3.10.9`.
- **TensorFlow Compatibility:**
  - On macOS, use `tensorflow-macos`
  - On Linux/Windows, use `tensorflow==2.9.2`

# Installation

## Clone the Repository

```bash
git clone https://github.com/pamonroy/aBCMI.git
cd aBCMI
```

## Create a Virtual Environment (Recommended)

```bash
python3.10 -m venv eeg_env
source eeg_env/bin/activate
```

On Windows:

```bash
eeg_env\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import numpy, pandas, mido, sklearn, pynput, pyxdf, bitalino, pylsl, tensorflow; print('All dependencies installed successfully!')"
```

## Configure Working Directory

- Update the `os.chdir()` path in each script to point to your project directory.
- Ensure the BITalino MAC address is correct in the connection code.

---

# Experimental Methods

## Method 1: AFAH (Asymmetric Frontal Activity Hypothesis)

**File:** `afah.py`

Real-time affective music generation based on EEG frontal asymmetry without machine learning. The system estimates emotional dimensions directly from EEG power spectra.

### Key Features

- Real-time valence and arousal estimation
- No training required (works immediately)
- Normalization with baseline recording
- Smoothed valence (10-second moving average)

### Emotion Metrics

```text
Valence = (AF8_alpha / AF8_beta) - (AF7_alpha / AF7_beta)

Arousal = (AF7_beta + AF8_beta) / (AF7_alpha + AF8_alpha)
```

### Protocol

- 15 seconds baseline (eyes closed)
- 20 trials: 10 happy + 10 sad (semi-randomized order)
- 30 seconds music per trial

### Usage

```bash
python afah.py
```

---

## Method 2: MLP (Multi-Layer Perceptron)

**File:** `mlp.py`

EEG-based emotion classification using a Multi-Layer Perceptron neural network trained on power spectral features.

### Key Features

- 10 power spectral features (theta, alpha, low beta, high beta, gamma) extracted from AF7 and AF8
- MLP classifier with `30-30` hidden layers
- Calibration phase with 3 conditions (sad, happy, neutral)
- Online real-time classification

### Calibration Protocol

- 6 trials: sad, happy, neutral (counterbalanced order)
- Each trial: 10s silence + 20s music
- Feature extraction and model training

### Online Protocol

- 12 trials: 6 happy + 6 sad (randomized order)
- Real-time classification and adaptive music

### Usage

```bash
python mlp.py
```

---

## Method 3: LDA (Linear Discriminant Analysis)

**File:** `lda.py`

2×2 factorial design experiment using Linear Discriminant Analysis for emotion classification with music/no-music conditions.

### Key Features

- 2×2 factorial design (sad/happy × music/no-music)
- LDA classifier with 10 spectral features
- 3-minute meditation phase
- Self-report after each trial (1–9 scale)
- 4-second and 10-second emotion smoothing buffers

### Experimental Design
- Independent variables: Emotion (sad/happy) × Auditory Condition (music/no-music)
- Dependent variables: EEG features, self-report ratings, classifier accuracy

### Protocol
- 3 minutes meditation (baseline)
- 6 calibration trials (sad, happy, neutral)
- 20 online trials (5 per condition, randomized)
- 40-second trials with self-report after each

### Usage

```bash
python lda.py
```

---

## Method 4: EEGNet Deep Learning

### Files

- `eegnet.py` — Main experiment script
- `eegnet_train.ipynb` — Model training notebook

End-to-end deep learning using EEGNet architecture with LSTM for raw EEG classification.

### Key Features

- Raw EEG signal processing (no hand-crafted features)
- EEGNet architecture with convolutional and LSTM layers
- 10-second windows, 100 Hz sampling rate
- Binary classification: sad vs. happy

### Model Architecture

```text
Input (2 channels × 1000 timesteps)
→ Conv2D (8 filters, 1×50)
→ DepthwiseConv2D (depth multiplier 4)
→ SeparableConv2D (16 filters)
→ LSTM (16 units)
→ Dense (2 units)
→ Output (1 unit, sigmoid)
```

### Protocol

- 3 minutes meditation (baseline)
- 12 calibration trials (sad, happy, neutral)
- 24 online trials (6 per condition in 2×2 design)
- 30-second trials with self-report after each

### Usage

```bash
python eegnet.py
```

---

# Common Protocol

All four methods share a common experimental structure.

## Phase 1: Setup

- Participant code entry
- LSL outlet initialization
- EEG hardware connection check

## Phase 2: Volume Setting (Optional)

- Alternating sad/happy music
- Comfortable listening level adjustment

## Phase 3: Meditation / Baseline (Methods 3 & 4 Only)

- 3-minute eyes-closed rest period
- Baseline physiological state acquisition

## Phase 4: Calibration

- 10 seconds silence (idle baseline)
- 20 seconds fixed music (sad/happy/neutral)
- Counterbalanced order
- Self-report collection (Methods 3 & 4)

## Phase 5: Model Training (Methods 2–4)

- Feature extraction and preprocessing
- Model training (MLP, LDA, or EEGNet)
- Model serialization for online use

## Phase 6: Online Phase

- Real-time emotion prediction
- Adaptive music generation
- Randomized trial order (no consecutive duplicates)
- Self-report after each trial

## Phase 7: Debriefing

- Experiment completion signals
- Data saving and cleanup

---

# MIDI Configuration

## Channel Mapping

| Channel | Instrument | Purpose |
|---|---|---|
| `0` | Piano | Chord voicing (upper register) |
| `1` | Cello | Chord voicing (middle register) |
| `2` | Bass | Bass line |
| `3` | Piano | Melodic patterns |
| `5` | Control | MIDI CC messages (Auditory cues) |

---

## Control Change (CC) Messages

| CC | Purpose | Value |
|---|---|---|
| `13` | Volume setting request | `127` |
| `14` | Calibration start | `127` |
| `15` | Online phase start | `127` |
| `16` | Meditation start | `127` |
| `17` | Experiment end | `127` |
| `18` | Stop all | `127` |
| `19` | Model training start | `127` |
| `80–82` | Target emotion (`80=happy`, `81=neutral`, `82=sad`) | `127` |
| `83` | Close eyes instruction | `127` |
| `84` | Self-report request | `127` |
| `85` | No-music condition | `127` |
| `123` | All notes off | `0` |

---

# Lab Streaming Layer (LSL) Configuration

## LSL Streams
All methods create the following LSL streams (recorded with Lab Recorder 1.14.0):

- eeg-bitalino - Raw EEG data (2 channels, 1000Hz or 100Hz)
- markers - Experiment markers (integers)
- predicted-emotion - Predicted emotion values (2-3 channels, 2Hz)
- emotion - Predicted emotion fed to the music generator at the start of each bar
- self-report - Self-report ratings (EEGNet and LDA methods only)
- harmony - Chord progression markers (LDA method only)

## Marker Codes

| Marker | Value | Description |
|---|---|---|
| `setVolume` | `6` | Volume setting phase |
| `silence` | `0` | Silence period |
| `neutral` | `2` | Neutral emotion |
| `sad` | `1` | Sad emotion |
| `happy` | `3` | Happy emotion |
| `modeling` | `8` | Model training phase |
| `online` | `7` | Online phase start |
| `music` | `9` | Start music condition |
| `min_power` | `-1` | Minimum power alarm |
| `pause` | `10` | Pause between phases |
| `askForReport` | `11` | Self-report request |
| `stopMusic` | `12` | Stop music condition |
| `startNoMusic` | `13` | Start no-music condition |
| `stopNoMusic` | `14` | Stop no-music condition |
