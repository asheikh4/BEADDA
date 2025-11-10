from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time

# -----------------------------------------------------------
# STEP 1: Connect to EMG LSL stream
# -----------------------------------------------------------
print("Looking for an OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EEG')  # or 'EMG' if using EMG stream
inlet = StreamInlet(streams[0], max_buflen=1)  # Keep buffer short to reduce lag
print("Connected!\n")

# -----------------------------------------------------------
# STEP 2: MVC Calibration
# -----------------------------------------------------------
print("We’ll record your maximum voluntary contraction (MVC).")
print("Make sure electrodes are on and relax your muscle.\n")

for i in range(5, 0, -1):
    print(f"Starting calibration in {i}...", end='\r')
    time.sleep(1)

print("\nBEGIN maximum contraction! Hold for 5 seconds...")

mvc_samples = []
start_time = time.time()

# Collect 5 seconds of data
while time.time() - start_time < 5:
    chunk, timestamps = inlet.pull_chunk(timeout=1.0)
    if chunk:
        mvc_samples.extend([s[0] for s in chunk])  # Only take channel 0

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
window_size = 500  # ~2 seconds if sampling at 250 Hz (adjust for your rate)

while True:
    # Pull all new samples since last call
    chunk, timestamps = inlet.pull_chunk(timeout=0.1)
    if not chunk:
        continue  # skip if no new data
    
    # Flatten chunk into window
    for s in chunk:
        window.append(s[0])

    # Keep window size constant
    if len(window) > window_size:
        window = window[-window_size:]  # keep only latest samples

    # Compute RMS and %MVC if enough samples
    if len(window) >= window_size:
        rms = np.sqrt(np.mean(np.square(window)))
        percent_mvc = (rms / mvc_rms) * 100
        print(f"%MVC: {percent_mvc:.1f}% | RMS: {rms:.3f} µV", end='\r')
