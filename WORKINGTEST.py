from pylsl import StreamInlet, resolve_byprop
import numpy as np

print("Looking for an OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EEG')  # Could also be 'EMG'
inlet = StreamInlet(streams[0])
print("Connected!")

while True:
    sample, timestamp = inlet.pull_sample()
    print(f"{sample[0]}")