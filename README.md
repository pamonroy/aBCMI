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
