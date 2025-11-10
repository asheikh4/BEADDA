from pyOpenBCI import OpenBCICyton  # or OpenBCIGanglion if that's your board
import numpy as np
from scipy.signal import butter, lfilter
import time

# ============ PARAMETERS ============
FS = 250  # Sampling frequency (Hz)
CHANNEL = 1  # Which EMG channel to use
MVC = 0.0006  # Replace with your subject’s measured max voluntary contraction (in Volts)
WINDOW_SIZE = 1  # seconds for RMS window
FATIGUE_THRESHOLD = 1  # proportion of MVC at which fatigue flag triggers
# ====================================

# Bandpass filter for EMG
def butter_bandpass(lowcut=20, highcut=450, fs=FS, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=20, highcut=450, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# RMS calculation
def compute_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

# ============ DATA CALLBACK ============
emg_window = []

def handle_sample(sample):
    global emg_window
    val = sample.channels_data[CHANNEL - 1] * (4.5 / (24 * (2**23 - 1)))  # Convert from ADC counts to volts (Cyton)
    emg_window.append(val)
    
    # keep rolling window
    if len(emg_window) > FS * WINDOW_SIZE:
        emg_window = emg_window[-int(FS * WINDOW_SIZE):]

    if len(emg_window) == int(FS * WINDOW_SIZE):
        filtered = bandpass_filter(emg_window)
        rms = compute_rms(filtered)
        percent_mvc = (rms / MVC) * 100

        fatigue_flag = percent_mvc > (FATIGUE_THRESHOLD * 100)

        print(f"%MVC: {percent_mvc:.2f}% | Fatigue: {'YES' if fatigue_flag else 'NO'}")

# ============ CONNECT TO BOARD ============
if __name__ == "__main__":
    print("Connecting to OpenBCI board...")
    board = OpenBCICyton(port='COM7', daisy=False)  # change COM port as needed
    board.start_stream(handle_sample)
