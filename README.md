# aBCMI: Affective Brain-Computer Musical Interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
  doi = {[Add DOI if available]}
}

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
