from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from scipy.signal import butter, filtfilt, welch

# ------------------ CONFIG ------------------
WINDOW_SIZE = 0.25   # seconds for RMS window
SAMPLING_RATE = 200  # adjust for your setup (e.g., 200Hz for Ganglion)
CALIBRATION_DURATION = 5  # seconds to record max voluntary contraction
RMS_FATIGUE_THRESHOLD = 0.7  # 70% of max RMS sustained indicates fatigue
# --------------------------------------------

# ------------------ FILTER ------------------
def bandpass_filter(data, lowcut=20, highcut=450, fs=SAMPLING_RATE, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# ------------------ RMS ------------------
def compute_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

# ------------------ FATIGUE ------------------
def compute_mean_freq(signal, fs=SAMPLING_RATE):
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal))
    return np.sum(f * Pxx) / np.sum(Pxx)
# --------------------------------------------

print("Looking for OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EMG')  # Make sure you select 'EMG' stream in GUI
inlet = StreamInlet(streams[0])
print("Connected!")

# ------------------ CALIBRATION ------------------
print("\n=== Calibration phase: Contract muscle maximally for 5 seconds! ===")
calibration_data = []
start_time = time.time()
while time.time() - start_time < CALIBRATION_DURATION:
    sample, _ = inlet.pull_sample()
    calibration_data.append(sample[0])  # Assuming 1 EMG channel

calibration_data = np.array(calibration_data)
filtered_calibration = bandpass_filter(calibration_data)
max_rms = compute_rms(filtered_calibration)
print(f"Calibration complete. Max RMS = {max_rms:.4f}")

# ------------------ REAL-TIME STREAMING ------------------
print("\n=== Real-time monitoring started ===")
buffer = []

while True:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample[0])

    # keep buffer ~WINDOW_SIZE seconds long
    if len(buffer) > int(WINDOW_SIZE * SAMPLING_RATE):
        buffer.pop(0)

        # process buffer
        filtered = bandpass_filter(np.array(buffer))
        rms = compute_rms(filtered)
        percent_mvc = (rms / max_rms) * 100

        # compute mean frequency for fatigue indicator
        mean_freq = compute_mean_freq(filtered)
        fatigue_flag = percent_mvc > (RMS_FATIGUE_THRESHOLD * 100) or mean_freq < 50  # tune threshold

        print(f"%MVC: {percent_mvc:.1f}% | Mean Freq: {mean_freq:.1f} Hz | Fatigue: {fatigue_flag}")