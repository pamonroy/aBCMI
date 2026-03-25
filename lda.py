#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:41:18 2024

@author: pablo
"""

from pylsl import StreamInfo, StreamOutlet
import time


from collections import deque
import numpy as np
import random as rn
from threading import Thread, Event
import sys

import pandas as pd

import mido as mido
from mido import Message

import os
os.chdir("/Volumes/Academico/2021-TIC phd/eeg/participants_lda_2x2/")

from pynput import keyboard

outport = mido.open_output()

long_sequence = False

 # threshold for electrode disconection
THRESHOLD_LOW = 10
THRESHOLD_HIGH = 1014

samplingRate = 1000
windowSize = 4 # size of window in seconds
buffersize = samplingRate * windowSize
update_rate = .5 # updating rate in seconds


event = Event()
is_memorizing = Event()
is_online = Event()
bitalino_connected = False

def select_lsl_stream(stream_name, streams):
    stream_list = streams[0]  
    for stream in stream_list:
        if stream['info']['name'][0] ==  stream_name:  
            return stream['time_series']
    return False

def RawEeg2uVolt(data_segment):
    # Convert raw eeg signal to micro volt
    # Specifications: https://www.bitalino.com/storage/uploads/media/revolution-eeg-sensor-datasheet-revb.pdf
    return (data_segment/2**10 - 0.5)*3.3/41782*1e6

#%%
    
def bandpower_welch(data, sf, band, window_sec=None, relative=False):
    # taken from https://raphaelvallat.com/bandpower.html
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simpson
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp

def power_spectral_density(electrode_signal, band):
    global samplingRate
    
    
    if band=='theta':
        low, high = 4, 7.9
    elif band=='alpha':
        low, high = 8, 13.9
    elif band=='low_beta':
        low, high = 14, 21.9
    elif band=='high_beta':
        low, high = 22, 29.9
    elif band=='beta':
        low, high = 14, 29.9
    else:  #gamma
        low, high = 30, 47
        
    electrode_signal = RawEeg2uVolt(electrode_signal)
    power = bandpower_welch(electrode_signal, samplingRate, [low, high], window_sec=1, relative=False)
    
    if power <= 0 :
        sendMarker("min_power")
        print('NEGATIVE POWER',power, band)

    return power

def get_power(left_electrode, right_electrode):
    Fp1 = np.asarray(left_electrode, dtype=np.float64)
    Fp2 = np.asarray(right_electrode, dtype=np.float64) 
    
    
    theta_Fp1 = power_spectral_density(Fp1, band='theta')
    alpha_Fp1 = power_spectral_density(Fp1, band='alpha')
    low_beta_Fp1 = power_spectral_density(Fp1, band='low_beta')
    high_beta_Fp1 = power_spectral_density(Fp1, band='high_beta')
    gamma_Fp1 = power_spectral_density(Fp1, band='gamma')
    
    theta_Fp2 = power_spectral_density(Fp2, band='theta')
    alpha_Fp2 = power_spectral_density(Fp2, band='alpha')
    low_beta_Fp2 = power_spectral_density(Fp2, band='low_beta')
    high_beta_Fp2 = power_spectral_density(Fp2, band='high_beta')
    gamma_Fp2 = power_spectral_density(Fp2, band='gamma')
    
    band_vector = [theta_Fp1, theta_Fp2, alpha_Fp1, alpha_Fp2, low_beta_Fp1, low_beta_Fp2, high_beta_Fp1, high_beta_Fp2, gamma_Fp1, gamma_Fp2]
    
    return band_vector

#%%
def captureSignal(emo_vector):
    global samplingRate, buffersize, update_rate, segmento, baseline, bitalino_connected
    
    from bitalino import BITalino
    macAddress = "/dev/tty.BITalino-8B-B1-DevB"
    acqChannels = [0,1]

    
    sequence_buffer_size =int (10 / update_rate)  ## 10 seconds buffer
    sequence_buffer = deque([0.5] * sequence_buffer_size, maxlen=sequence_buffer_size) 
    
    bar_buffer_size = int (5 / update_rate)  ## 5 seconds buffer 
    bar_buffer = deque([0.5] * bar_buffer_size, maxlen=bar_buffer_size)    
            
    
    localFP1 = deque([512] * buffersize , maxlen=buffersize)  # window
    localFP2 = deque([512] * buffersize , maxlen=buffersize)  # window
    
    nSamples = int(samplingRate/2)  # fixed to update every .5 second
    
    
    print("creating Signal stream...")
    info = StreamInfo('eeg-bitalino','EEG',2,samplingRate,'float32','eeg-data')
    # next make an outlet
    outlet = StreamOutlet(info)
    print("created Signal stream : %s" %info.name())
    
    # Connect to BITalino
    connection=0
    while connection == 0:
        try:
            print("connecting to BITalino(%s)..." %macAddress)
            device = BITalino(macAddress)
            connection=1;
            break
        except TypeError:
            print("T MAC address (%s) is not defined..." %macAddress)
            print("connecting to BITalino(%s)...")
        except ValueError:
            print("V MAC address (%s) is not defined..." %macAddress)
            print("connecting to BITalino(%s)...")
    
    print('\a', end='')
    
    
    
    if connection:
        # Read BITalino version
        print(device.version())
        print("connected to BITalino(%s)" %macAddress)
        # Start Acquisition

        device.start(samplingRate, acqChannels)    
        bitalino_connected = True
    
        deltaTime = time.time()
        while True:
            
            data_bitalino = device.read(nSamples)

            for rowData in data_bitalino:
                #print(rowData)
                dataFP1 = rowData[6]  # left prefrontal 
                localFP1.append(dataFP1)
                
                dataFP2 = rowData[5]  # right prefrontal
                localFP2.append(dataFP2)
                
                outlet.push_sample([rowData[6], rowData[5]])  ## [left-AF7, right-AF8] 
                
            signal_FP1 = list(localFP1)
            signal_FP2 = list(localFP2)
    
            vector = get_power(signal_FP1, signal_FP2)  ## delays almost 20 ms
                
            
            # store data for the calibration stage
            if is_memorizing.is_set():
                segmento.append(vector)              
                
            if is_online.is_set():                       
                current_power = vector

                current_power = np.asarray(current_power).reshape(1, 10)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)  # standarization
                arousal = clf.predict_proba(current_power_std)[0,0]
                #print('.', end='')
                
                sequence_buffer.append(arousal)
                bar_buffer.append(arousal)
                
                emo_vector[0] = arousal
                emo_vector[1] = np.mean(sequence_buffer)
                emo_vector[2] = np.mean(bar_buffer)

                
                outletPredictedEmotion.push_sample([emo_vector[0], emo_vector[1], emo_vector[2]]) 
                
                # print((arousal > 0.5).astype("int32"))
                
            if time.time()-deltaTime > 0.05:
                # print(dataFP1, dataFP2)
                deltaTime = time.time()
                if not (THRESHOLD_LOW < dataFP1 < THRESHOLD_HIGH):
                    print(">> ELECTRODE ALARM AF7 <<")
                    print('\a', end='')
                if not (THRESHOLD_LOW < dataFP2 < THRESHOLD_HIGH):
                    print(">> ELECTRODE ALARM AF8 <<")
                    print('\a', end='')
                    
            if event.is_set():
                break
            
        print('Stop capturing')
        # Stop acquisition
        device.stop()
        # Close connection
        device.close()
        
        sys.exit(0)
    


def captureSignalFromXDFBuffer(emo_vector):
# def captureSignalFromXDFBuffer(power_vector, smooth_valence):
    global samplingRate, buffersize, update_rate, segmento, baseline, bitalino_connected, participant
    
    import pyxdf as xdf
    
    sequence_buffer_size =int (10 / update_rate)  ## 10 seconds buffer
    sequence_buffer = deque([0.5] * sequence_buffer_size, maxlen=sequence_buffer_size) 
    
    bar_buffer_size = int (4 / update_rate)  ## 4 seconds buffer 
    bar_buffer = deque([0.5] * bar_buffer_size, maxlen=bar_buffer_size)    
            
    
    #lsl out simulation
    simulatedInfo = StreamInfo('eeg-bitalino','EEG',2,samplingRate,'float32','eeg-simulated-data')
    simulatedOutlet = StreamOutlet(simulatedInfo)
    
    
    localFP1 = deque([], buffersize)  # window
    localFP2 = deque([], buffersize)  # window
    
    for j in range(buffersize):
        localFP1.append(512)
        localFP2.append(512)
    
    nSamples = int(samplingRate/2)  # fixed to update every .5 second
    
    

  

    xdf_file1 = '../participants/'+participant + '-afah.xdf'
    streams1 = xdf.load_xdf(xdf_file1)
    xdf_series1 =  select_lsl_stream('eeg-bitalino', streams1)
    xdf_series1 = xdf_series1[50000::]  # erase 50 seconds at the begining to clean connection noise


    xdf_file2  = '../participants/'+participant + '-ml.xdf'
    streams2 = xdf.load_xdf(xdf_file2)
    xdf_series2 =  select_lsl_stream('eeg-bitalino', streams2)
    xdf_series2 = xdf_series2[50000::]  # erase 50 seconds at the begining to clean connection noise

    xdf_series = np.concatenate((xdf_series2, xdf_series1, xdf_series2))
    
    #downsample the signal from 1000Hz to 100Hz, because it was recorded at 1000Hz
    #xdf_series = xdf_series[::int(1000/100)]
    

    

    signal_Fp1 = xdf_series[:,0].tolist()
    signal_Fp2 = xdf_series[:,1].tolist()
    signal_full = list(zip(signal_Fp1, signal_Fp2))

    deltaTime = time.time()
   
    while len(signal_full):
       if time.time()-deltaTime >= update_rate:
            deltaTime = time.time()
            
            sample = signal_full[0:nSamples]  # get current chunk
            
            #sample = np.array(sample) # convert from list to array
            #sample = RawEeg2uVolt(sample) # convert from RAW to uV
            
            signal_full = signal_full[nSamples:]
    
                
            for data in sample:  # delays almost 13 ms
                localFP1.append(data[0])
                localFP2.append(data[1])
                simulatedOutlet.push_sample([data[0], data[1]])

            signal_FP1 = list(localFP1)
            signal_FP2 = list(localFP2)
    
            vector = get_power(signal_FP1, signal_FP2)  ## delays almost 20 ms
    
            
            # store data for the calibration stage
            if is_memorizing.is_set():
                #print("calibration")
                segmento.append(vector)
            
            if is_online.is_set():
                #current_power = power_vector
                current_power = vector

                current_power = np.asarray(current_power).reshape(1, 10)
                current_power = current_power - baseline
                current_power_std = sc.transform(current_power)  # standarization
                arousal = clf.predict_proba(current_power_std)[0,0]
                #print('.', end='')
                

                sequence_buffer.append(arousal)
                bar_buffer.append(arousal)
                
                emo_vector[0] = arousal
                emo_vector[1] = np.mean(sequence_buffer)
                emo_vector[2] = np.mean(bar_buffer)
                

                outletPredictedEmotion.push_sample([emo_vector[0], emo_vector[1], emo_vector[2]]) 
                
                # print((arousal > 0.5).astype("int32"))

         
            if data[0] < THRESHOLD_LOW or data[1] < THRESHOLD_LOW:  # threshold for electrode disconection
                print(">> ELECTRODE ALARM <<")
                print('\a', end='')
                
            if event.is_set():
                break

    print('Stop capturing')
    print('\a')
    sys.exit(0)
 
#%%

 
def current_milli_time():
    return round(time.time() * 1000)

def getChord(mode, bar):
    ## the greek modes iterates by 5th interval
    # en modo Lidio, la base es el 4to grado (3)
    base = (3 + (mode * 5) - mode) % 7
    
    # Map bar values to deltas
    delta_map = {0: 0, 1: 3, 2: 4, 3: 0}  ## la secuencia armónica es I IV V I
    delta = delta_map.get(bar, 0)  # Default to 0 if bar is not in the map
    
    # added 1 to the output because 0 is a marker for silence
    return (base + delta) % 7 + 1
   
def composer(isRealTime = True, delay = 40, in_arousal = [], in_valence = []):
    global event
    
    chordlist = np.array([
        [60, 64, 55, 59],
        [62, 65, 57, 60],
        [64, 55, 59, 62],
        [60, 65, 57, 64],
        [55, 59, 62, 65],
        [57, 60, 64, 55],
        [59, 62, 65, 57]
    ])
    
    modeset = np.zeros(shape=(4,4,7))
    
    modes = ['lydian','ionian','mixolydian','dorian','aeolian','phrygian','locrian']
    
    mode_map = [
        [3, 6, 0, 3],    # Lydian
        [0, 3, 4, 0],    # Ionian
        [4, 0, 1, 4],    # Mixolydian
        [1, 4, 5, 1],    # Dorian
        [5, 1, 2, 5],    # Aeolian
        [2, 5, 6, 2],    # Phrygian
        [6, 2, 3, 6]     # Locrian
    ]
    
    for i, mode_indices in enumerate(mode_map):
        for j in range(4):
            modeset[j, :, i] = chordlist[mode_indices[j], :]
        
    modeset = modeset-3 # pitch-down the complete modeset with 3 half-tones
    
    LOW_LOUDNESS = 50         # set minimal loudnes
    tick, beat, bar = 0, 0, 0
    
    start = time.time()

    while (time.time() - start < delay) and not event.is_set() :
        try:

            if beat % 4 == 0 and tick % 2 == 0:  # este sirve para sincronizar y se puede incluir en el siguiente if
                #print('bar:',bar)  # aqui se cambia de armonia y se calcula la melodia / leo valence y arousal
                #print('mode:', mode)
                if isRealTime:
                    arousal = emotional_vector[2]
                    valence = arousal

                else: 
                    try:
                        valence = in_valence.pop(0)
                        arousal = in_arousal.pop(0)
                    except:
                        break
    
                if bar == 0:  ## calculo el modo  
                    if isRealTime:
                        valence = emotional_vector[2]  ## 4 sec average smoothed valence
                        arousal = valence
                
                    mode = 7-round(valence*6)-1; # set harmonic mode based on valence
                    print("mode:",modes[int(mode)], int(mode)) 
    
                if isRealTime:
                    print('EMO: %.2f' % arousal,' - current: %.2f' % emotional_vector[0], 'sequence: %.2f' % emotional_vector[1], 'bar: %.2f' % emotional_vector[2])
                    outletEmotionalMarkers.push_sample([arousal])
                
        
                # aqui calculo los demas parámetros del acorde
                roughness = 1-arousal      # set rhythmic roughness based on arousal
                velocity = arousal         # set tempo based on arousal
                voicing = valence          # set voicing based on valence
                loudness = (round(arousal*10))/10*40+60;   # set maximal loudness based on arousal
                                
                # create roughness
                activate1 = np.random.rand(8)
                activate1[activate1 < roughness] = 0
                activate1[activate1 >= roughness] = 1
                
                activate2 = np.random.rand(8)
                activate2[activate2 < roughness] = 0
                activate2[activate2 >= roughness] = 1

                bright = np.random.rand(6)
                if voicing < 0.5:
                    bright[bright > voicing * 2] = -1
                    bright[bright <= voicing * 2] = 0
                else:
                    bright[bright < (voicing - 0.5) * 2] = 1
                    bright[bright >= (voicing - 0.5) * 2] = 0
                
                # stop all channels
                for channel in range(4):
                    msg = Message('control_change', control=123, value=0, channel=channel)
                    outport.send(msg)
        
                # store the chord as a marker in a LSL stream
                sendChordMarker(getChord(mode, bar))
                
                # Playing chords on piano and chello
                channels = [0, 1] # 0:piano, 1:chello on Ableton Live
                for channel in channels:
                    for note_index in range(3):
                        msg = Message('note_on', channel=channel,
                                      note=int(modeset[bar, note_index, mode] + bright[note_index] * 12),
                                      velocity=rn.randint(LOW_LOUDNESS, loudness))
                        outport.send(msg)

                # Playing the bass
                bass_note = modeset[bar, 0, mode] - (24 if voicing <= 0.5 else 12)
                msg = Message('note_on', channel=2, note=int(bass_note),
                              velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
                
                bar += 1
                if bar == 4:
                    bar = 0  # aqui se reinicia la secuencia
            
            if tick % 2 == 0:
                #print('beat ', beat)
                beat += 1
                if beat == 4:
                    beat = 0
          
            # print('.')  # aqui se controla la melodia
            # play de la melodia en el canal 4
            msg = Message('control_change', control=123, value=0, channel=3)
            outport.send(msg)
            if activate1[tick] == 1:
                msg = Message('note_on',  channel=3, note=int(modeset[bar-1,0,mode]+bright[4]*12), velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg) 
        
            if activate2[tick] == 1:
                msg = Message('note_on',  channel=3, note=int(modeset[bar-1,rn.randint(1, 2),mode]+bright[5]*12), velocity=rn.randint(LOW_LOUDNESS, loudness))
                outport.send(msg)
            time.sleep(0.3-velocity*0.15)
            
            tick += 1
            if tick > 7:
                tick = 0
      
        except KeyboardInterrupt:
            event.set()
            break
    
    # stop all channels    
    for channel in range(4):
        msg = Message('control_change', control=123, value=0, channel=channel)
        outport.send(msg)

    if event.is_set():
        sys.exit(0)

#%%
class KeyPressThread(Thread):
    def __init__(self):
        super().__init__()
        self.key_pressed = False
        self.pressed_key = None
        self.timer_duration = 15 # if there is no answer after 15 seconds the system will ask for report again 
        self.valid_keys = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        self.listener = keyboard.Listener(on_press=self.on_press)

    def run(self):
        self.listener.start()
        while not self.key_pressed:
            print("Asking for self report")
            msg = Message('control_change', control=84, value=127, channel=5) # send midi CC messages on channel 5
            outport.send(msg)
            self.wait_for_keypress(self.timer_duration)  
            if not self.key_pressed:
                print("Asking again!")
        self.listener.stop()

    def wait_for_keypress(self, timeout):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.key_pressed:
                return
            time.sleep(0.1)  # Sleep briefly to avoid busy waiting

    def on_press(self, key):
        try:
            if key.char in self.valid_keys:
                self.key_pressed = True
                self.pressed_key = key.char
        except AttributeError:
            pass  # Special keys (e.g., shift) are ignored

def self_report():
    time.sleep(2)
    sendMarker("askForReport")
    
    key_press_simulation = False
    
    if key_press_simulation:
        ## key pressing simulation
        print("Asking for self report")
        msg = Message('control_change', control=84, value=127, channel=5) # send midi CC messages on channel 5
        outport.send(msg)
        time.sleep(4) 
        #time.sleep(rn.randint(3, 5)) ## delay simulation
        selfReportValue = rn.randint(1, 9)  ## response simulation
    else:
        keypress_thread = KeyPressThread()
        keypress_thread.start()
        keypress_thread.join()
        selfReportValue = int(keypress_thread.pressed_key)

    # registrar el reporte en LSL
    outletSelfReport.push_sample([selfReportValue])
    print("self report:",selfReportValue)
            
#%%    
def close_eyes():
    time.sleep(1)
    print("Close eyes and relax")
    msg = Message('control_change', control=83, value=127, channel=5) # send midi CC messages on channel 5
    outport.send(msg)
    time.sleep(6)        
#%%
def sendMarker(marker):
    lslMarkers = {
        "setVolume":    6,
        "silence":      0,
        "neutral":      2,
        "sad":          1,
        "happy":        3,
        "modeling":     8,
        "online":       7,
        "music":        9, #startMusic
        "min_power":    -1,
        "pause":        10,
        "askForReport": 11,
        "stopMusic":    12,
        "startNoMusic": 13,
        "stopNoMusic":  14
    }
    outletMarkers.push_sample([lslMarkers[marker]])
    
def sendChordMarker(chord):
    # 0 is a silence
    # > 0 is diatonical chord (1:I, 2:II, 3:III, 4:IV, etc)
    outletHarmony.push_sample([chord])

def targetSound(emotion):  # triggers 1 beep for sad emotion, 2 beeps for neutral, and 3 beeps for happy
    baseTime = 0.3
    indicatorNote = 72 # 72 is C4
    if emotion == 'sad':
        quantity = 1
    elif emotion == 'neutral':
        quantity = 2
    else:
        quantity = 3
    beepTime = baseTime / (2*quantity - 1) # to avoid the last silence in the loop

    # send lsl marker for the target emotion
    sendMarker(emotion)

    for k in range(quantity):
        msg = Message('note_on',  channel=4, note=indicatorNote, velocity=127)
        outport.send(msg) 
        time.sleep(beepTime)
        msg = Message('control_change', control=123, value=0, channel=4)
        outport.send(msg) 
        if k < quantity:
            time.sleep(beepTime) # to avoid the last silence in the loop
            
def targetMessage(emotion):  # triggers messages in Ableton Live

    if emotion == 'sad':
        control = 82
    elif emotion == 'neutral':
        control = 81
    else:  # happy
        control = 80
        
    # send lsl marker for the target emotion
    sendMarker(emotion)
    msg = Message('control_change', control=control, value=127, channel=5) # send midi CC messages on channel 5
    outport.send(msg)
         
 
def play_silence(sleepTime):
    sendMarker("silence")
    sendChordMarker(0)  # send a silence to the harmonyMarkers
    time.sleep(sleepTime)
            
def play_volume_setting():
    ## sad + happy + sad + happy routine
    input_valence = (np.linspace(1, 0, 8)).tolist() + (np.linspace(0, 1, 8)).tolist() 
    input_arousal = (np.linspace(0, 1, 8)).tolist() + (np.linspace(1, 0, 8)).tolist() 
    composer(False, 60, input_arousal, input_valence)
    
def play_sad_class():
    input_valence = (np.linspace(0, 0, 8)).tolist()
    input_arousal = (np.linspace(0, 0, 8)).tolist()
    sendMarker('sad')  # use this if there is no previous emotional target sound
    composer(False, 60, input_arousal, input_valence)
    
def play_neutral_class():
    input_valence = (np.linspace(0.5, 0.5, 12)).tolist()
    input_arousal = (np.linspace(0.5, 0.5, 12)).tolist()
    sendMarker('neutral')  # use this if there is no previous emotional target sound
    composer(False, 60, input_arousal, input_valence)
    
def play_happy_class():
    input_valence = (np.linspace(1, 1, 16)).tolist()
    input_arousal = (np.linspace(1, 1, 16)).tolist()
    sendMarker('happy')  # use this if there is no previous emotional target sound
    composer(False, 60, input_arousal, input_valence)

## play a neutral harmonic sequence at the begining of each online trial
def play_neutral_online():
    input_valence = (np.linspace(0.5, 0.5, 4)).tolist()
    input_arousal = (np.linspace(0.5, 0.5, 4)).tolist()
    composer(False, 60, input_arousal, input_valence)

#%%
# pseudorandomize calibration class sequence
if long_sequence:
    class_sequence = ['sad','happy','neutral','sad','happy','neutral','sad','happy','neutral','sad','happy','neutral']
else:
    class_sequence = ['sad','happy','neutral','sad','happy','neutral']
    #class_sequence = ['sad','happy','neutral']

shuffled_sequence = []
n = rn.randint(0, len(class_sequence)-1)
element = class_sequence[n]
shuffled_sequence.append(element)
class_sequence.remove(element)

i = 0
while i < len(class_sequence):
    n = rn.randint(0, len(class_sequence)-1)
    element = class_sequence[n]
    if shuffled_sequence[-1] != element:
        shuffled_sequence.append(element)
        class_sequence.remove(element)
        

#%%
## pseudorandomize emotions for the online stage

## generate trials based on emotions and music/silence
## it is not allowed to have three (3) consecutive conditions 
##  generate the emotion_list sequence, ensuring no emotion or action repeats consecutively more than twice.

conditions = ["sad-music","sad-noMusic","happy-music", "happy-noMusic"]
trials_per_condition = 5
max_attempts = 100  # Max attempts before doing again


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
            # do it again!  Initialize lists
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
    attempts = 0  # Reset attempts after successful append


#%%

experiment_start_time = time.time()

participant = input("Write the participant CODE and press Enter: ")

infoMarkers = StreamInfo('markers','Markers',1,0,'int32','eegloop-markers')
outletMarkers = StreamOutlet(infoMarkers)

infoHarmonyMarkers = StreamInfo('harmony','Markers',1,0,'int32','harmony-sequence')
outletHarmony = StreamOutlet(infoHarmonyMarkers)

infoSelfReporMarkers = StreamInfo('self-report','Markers',1,0,'int32','emotional-self-report')
outletSelfReport = StreamOutlet(infoSelfReporMarkers)

infoEmotionalMarkers = StreamInfo('emotion','Markers',1,0,'float32','music-generator-emotion-data')
outletEmotionalMarkers = StreamOutlet(infoEmotionalMarkers)

infoPredictedEmotion = StreamInfo('predicted-emotion','Emotion',3,2,'float32','emotion-data')  # 3 channel at 2Hz
outletPredictedEmotion = StreamOutlet(infoPredictedEmotion)


emotional_vector = [0.5, 0.5, 0.5]  ## [current , smooth for sequence, smooth for bar]
t = Thread(target=captureSignal, args=[emotional_vector, ])

t.start()

# the next loop waits for bitalino connection
while not bitalino_connected:
    continue

# participant = input("Write the participant CODE and press Enter: ")

input("Record on LabRecorder (adding _lda2x2 sufix: " + participant + "_lda2x2.xdf) and Press Enter to start...")

## VOLUME SETTING
askVolume = input("Set volume (y/n)?")

if askVolume=='y': 
    sendMarker("setVolume")
    # trigger Initial message asking for volume setting
    msg = Message('control_change', control=13, value=127, channel=5) # send midi CC messages on channel 5
    outport.send(msg)
    time.sleep(10)
    play_volume_setting()

sendMarker("pause")
input("Ask to close eyes and Press Enter to continue...")


## MEDITATION
print("\n--------------------> [MEDITATION]")
sendMarker("pause")
msg = Message('control_change', control=16, value=127, channel=5)
outport.send(msg)
time.sleep(180)

## CALIBRATION
print("\n--------------------> [CALIBRATING]")
# trigger  message asking for eyes closing and emotion modulation
msg = Message('control_change', control=14, value=127, channel=5) # send midi CC messages on channel 5
outport.send(msg)
time.sleep(16)

sendMarker("pause")

# this is an inital self report
self_report()
close_eyes()

df = pd.DataFrame() #initialize empty dataframe
length_sequence = len(shuffled_sequence)
for i, val in enumerate(shuffled_sequence):
    
    if event.is_set():
        sys.exit(0)
    print('\n---> trial',i+1,'/',length_sequence, ':', val)
    # close_eyes()
    
    # 10 seconds of silence
    segmento = []
    is_memorizing.set()
    play_silence(10)
    is_memorizing.clear()
    df_temp = pd.DataFrame(segmento, columns=['t1','t2','a1','a2','lb1','lb2','hb1','hb2','g1','g2'])
    df_temp['class'] = 'idle'
    print('\nclass idle:',len(df_temp), 'samples\n')
    df  = pd.concat([df, df_temp])

    # 20 seconds of fixed music
    segmento = []
    is_memorizing.set()
    sendMarker("music")
    eval("play_"+val+"_class()")
    is_memorizing.clear()
    df_temp = pd.DataFrame(segmento, columns=['t1','t2','a1','a2','lb1','lb2','hb1','hb2','g1','g2'])
    df_temp['class'] = val
    print('class', val,': ',len(df_temp), 'samples')
    df  = pd.concat([df, df_temp])
    

play_silence(2)

# post calibrating self report
# self_report()
  
df = df.reset_index(drop=True)

# record training data to do a later offline machine learning / deep learning model exploration
df.to_csv(participant + '_training.csv', sep='\t', index=False)  


## MODELING
print("\n--------------------> [MODELING]")               
sendMarker('modeling')

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# generate baseline-vector from the idle condition
baseline = df.loc[df['class'] == 'idle'].iloc[:,:-1].mean().to_numpy()


# create matrix from sad and happy condition
X = df.loc[df['class'].isin(['sad','happy'])].iloc[:,:-1].to_numpy()
y = df.loc[df['class'].isin(['sad','happy'])].iloc[:,-1:].values.ravel()

# substract baseline vector from observed data
X_cleared = X - baseline

#standardize
sc = StandardScaler()
X_std = sc.fit_transform(X_cleared)


# call LinearDiscriminantAnalysis Classifier model
clf = LinearDiscriminantAnalysis(solver='svd')

# train the model
clf.fit(X_std, y)

## save trained model to file
import pickle
with open(participant+'_model.pickle', 'wb') as f:
    pickle.dump(clf, f)
## save standard Scaler model to file   
with open(participant+'_standardScaler.pickle', 'wb') as file:
    pickle.dump(sc, file)



# ONLINE
print("\n--------------------> [ONLINE]")    
sendMarker("pause")
# trigger  message asking for emotion modulation
msg = Message('control_change', control=15, value=127, channel=5) # send midi CC messages on channel 5
outport.send(msg)
time.sleep(30)

   
is_online.set()
sendMarker('online')

# this is an inital self report
self_report()

for i, condition in enumerate(emotions_list):
    if event.is_set():
        sys.exit(0)
        
    target_emotion = condition.split('-')[0]
    trial_action = condition.split('-')[1]
    
    close_eyes()
    
    play_silence(13) ## original value is 13
    targetMessage(target_emotion)
    print('\n---> trial',i+1, ': towards', condition)
    time.sleep(4) # silence without marker
    if trial_action=="music":
        sendMarker("music")
        play_neutral_online()
        composer(True, 33)  # 23 for 30-sec trial, 33 for 40-sec trial
        sendMarker("stopMusic")
    else:
        sendMarker("startNoMusic")
        msg = Message('control_change', control=85, value=127, channel=5)
        outport.send(msg)
        time.sleep(40) ## original value is 30
        sendMarker("stopNoMusic")
        msg = Message('control_change', control=85, value=127, channel=5)
        outport.send(msg)
    
    # send self-report message
    self_report()
    
play_silence(10) # last marker

# trigger  message announcing that the experiment finished.
msg = Message('control_change', control=17, value=127, channel=5) # send midi CC messages on channel 5
outport.send(msg)
time.sleep(8)

# stop all on ableton live
msg = Message('control_change', control=18, value=127, channel=5) # send midi CC messages on channel 5
outport.send(msg)
time.sleep(1)
outport.send(msg)



event.set()
t.join()
print("execution time: %s minutes" % int((time.time() - experiment_start_time)/60))