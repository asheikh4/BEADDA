from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time

from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading

# SETTING UP FLASK STUFF

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def counter_thread():
    count = 0
    while True:
        socketio.emit('counter_update', {'count': count})
        count += 1
        time.sleep(1)  # Update every second


# Begin calibration when the calibration button is pressed
@socketio.on('button_pressed')
def begin_calibration(data):
    print(f"Button pressed! Message from client: {data}")
    # you can trigger any function here (e.g., start calibration, save data, etc.)


# Start the background thread
thread = threading.Thread(target=counter_thread)
thread.daemon = True
thread.start()

socketio.run(app, debug=True, port=8080)

gay = input("gay or not?")

if gay == "1":
    print("gay!!")




else:
    print("Looking for an OpenBCI LSL stream...")
    streams = resolve_byprop('type', 'EEG')  # Could also be 'EMG'
    inlet1 = StreamInlet(streams[0])
    # inlet2 = StreamInlet(streams[1])
    print("Connected!")
    print("Begin Flexing!")
    print("Calibration in 3:")

    time.sleep(1)
    print("Calibration in 2:")
    time.sleep(1)
    print("Calibration in 1:")
    time.sleep(1)

    calibration = []

    print("Calibrating with 100 samples.")

    while len(calibration) < 100:
        sample, timestamp = inlet1.pull_sample()
        calibration.append(sample)

    print(calibration)


    # Compute RMS
    mvc_rms = np.sqrt(np.mean(np.square(calibration)))
    print(f"MVC RMS: {mvc_rms:.3f} µV")
    print("You can now relax.\n")

    print("Starting live %MVC tracking...\n")
    time.sleep(3)

    # -----------------------------------------------------------
    # STEP 3: Live tracking loop
    # -----------------------------------------------------------
    window = []
    window_size = 200  # about 2 seconds if sampling at 100 Hz

    while True:
        sample, timestamp = inlet1.pull_sample()
        window.append(sample[0])

        if len(window) > window_size:
            window.pop(0)
            rms = np.sqrt(np.mean(np.square(window)))
            percent_mvc = (rms / mvc_rms) * 100
            print(f"%MVC: {percent_mvc:.1f}% | RMS: {rms:.3f} µV", end='\r')