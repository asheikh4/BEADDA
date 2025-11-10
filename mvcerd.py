from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time

# -----------------------------------------------------------
# STEP 1: Connect to OpenBCI LSL stream
# -----------------------------------------------------------
print("Looking for an OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EMG')  # Same stream carries EMG + EEG channels
if len(streams) == 0:
    raise RuntimeError("No stream found! Make sure OpenBCI is streaming via LSL.")
inlet = StreamInlet(streams[0])
print("Connected!\n")

# -----------------------------------------------------------
# STEP 2: MVC + EEG Calibration
# -----------------------------------------------------------
print("Calibration: Relax completely.\n")
time.sleep(3)
print("Recording relaxed baseline for 5 seconds...")

rest_emg, rest_eeg = [], []
start_time = time.time()

while time.time() - start_time < 5:
    sample, _ = inlet.pull_sample()
    rest_emg.append(sample[0])             # EMG: channel 1
    rest_eeg.append(sample[1:4])           # EEG: channels 2–4

rest_emg_rms = np.sqrt(np.mean(np.square(rest_emg)))
rest_eeg_rms = np.sqrt(np.mean(np.square(rest_eeg)))
print(f"Relaxed EMG RMS: {rest_emg_rms:.3f} µV | Relaxed EEG RMS: {rest_eeg_rms:.3f} µV")

time.sleep(2)
print("\nNow contract maximally and focus intensely for 5 seconds!\n")

max_emg, max_eeg = [], []
start_time = time.time()

while time.time() - start_time < 5:
    sample, _ = inlet.pull_sample()
    max_emg.append(sample[0])
    max_eeg.append(sample[1:4])

max_emg_rms = np.sqrt(np.mean(np.square(max_emg)))
max_eeg_rms = np.sqrt(np.mean(np.square(max_eeg)))

print(f"Max EMG RMS: {max_emg_rms:.3f} µV | Max EEG RMS: {max_eeg_rms:.3f} µV")
print("\nCalibration complete.\n")

# -----------------------------------------------------------
# STEP 3: Live tracking loop
# -----------------------------------------------------------
emg_window, eeg_window = [], []
window_size = 200  # about 2 seconds at 100 Hz

print("Starting live EMG + EEG tracking...\n")

while True:
    sample, timestamp = inlet.pull_sample()
    emg_window.append(sample[0])
    eeg_window.append(sample[1:4])

    if len(emg_window) > window_size:
        emg_window.pop(0)
        eeg_window.pop(0)

        # EMG activation
        emg_rms = np.sqrt(np.mean(np.square(emg_window)))
        emg_percent = (emg_rms - rest_emg_rms) / (max_emg_rms - rest_emg_rms) * 100
        emg_percent = np.clip(emg_percent, 0, 100)

        # EEG activation
        eeg_rms = np.sqrt(np.mean(np.square(eeg_window)))
        eeg_percent = (eeg_rms - rest_eeg_rms) / (max_eeg_rms - rest_eeg_rms) * 100
        eeg_percent = np.clip(eeg_percent, 0, 100)

        print(f"EMG: {emg_percent:6.1f}% | EEG: {eeg_percent:6.1f}% | "
              f"RMS (µV): {emg_rms:6.3f} / {eeg_rms:6.3f}", end='\r')