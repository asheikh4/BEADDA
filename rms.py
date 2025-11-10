from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time

# -----------------------------------------------------------
# STEP 1: Connect to EMG LSL stream
# -----------------------------------------------------------
print("Looking for an OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EEG')  # or 'EEG' if using EEG electrodes
inlet = StreamInlet(streams[0])
print("Connected!\n")

# -----------------------------------------------------------
# STEP 2: Prepare for MVC calibration
# -----------------------------------------------------------
print("We’ll record your maximum voluntary contraction (MVC).")
print("Make sure electrodes are on and relax your muscle.\n")

# Wait before starting calibration
for i in range(5, 0, -1):
    print(f"Starting calibration in {i}...", end='\r')
    time.sleep(1)

print("\nBEGIN maximum contraction! Hold for 5 seconds...")

mvc_samples = []
start_time = time.time()

# Record for 5 seconds
while time.time() - start_time < 5:
    sample, timestamp = inlet.pull_sample()
    mvc_samples.append(sample[0])  # Use one channel

print("Stop contracting. Calculating MVC...")

# Compute RMS
mvc_rms = np.sqrt(np.mean(np.square(mvc_samples)))
print(f"MVC RMS: {mvc_rms:.3f} µV")
print("You can now relax.\n")

time.sleep(3)
print("Starting live %MVC tracking...\n")

# -----------------------------------------------------------
# STEP 3: Live tracking loop
# -----------------------------------------------------------
window = []
window_size = 500  # about 2 seconds if sampling at 100 Hz

while True:
    sample, timestamp = inlet.pull_sample()
    window.append(sample[0])

    if len(window) > window_size:
        window.pop(0)
        rms = np.sqrt(np.mean(np.square(window)))
        percent_mvc = (rms / mvc_rms) * 100
        print(f"%MVC: {percent_mvc:.1f}% | RMS: {rms:.3f} µV", end='\r')