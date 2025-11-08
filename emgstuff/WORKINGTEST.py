from pylsl import StreamInlet, resolve_byprop
import numpy as np

print("Looking for an OpenBCI LSL stream...")
streams = resolve_byprop('type', 'EEG')  # Could also be 'EMG'
inlet1 = StreamInlet(streams[0])
# inlet2 = StreamInlet(streams[1])
print("Connected!")

while True:
    sample, timestamp = inlet1.pull_sample()
    # sample2, timestamp = inlet2.pull_sample()
    print(f"{sample[0]}")