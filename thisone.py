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

mvc_rms = 1
mvc_rms = 1

is_collecting_data = False

@app.route('/')
def index():
    return render_template('index.html')


def counter_thread():
    count = 0
    while True:
        # socketio.emit('counter_update', {'count': count})
        count += 1
        time.sleep(1)  # Update every second

def collecting_data():
    global rms, mvc_rms
    # -----------------------------------------------------------
    # STEP 3: Live tracking loop
    # -----------------------------------------------------------
    window = []
    window_size = 100  # about 2 seconds if sampling at 100 Hz
    print("asdas")
    while True:
        # latest_samples, timestamp = inlet1.pull_chunk()
        latest_samples, timestamp = inlet1.pull_chunk()
        
        # Only proceed if latest_samples is not empty
        if len(latest_samples) == 0:
            continue  # skip this iteration
        
        # Get the most recent sample
        # sample = latest_samples[-1]
        # print(latest_samples)
        sample = latest_samples[-1] 
        window.append(sample[0])
        # print("freak", sample[0])

        if len(window) > window_size:
            window.pop(0)
            rms = np.sqrt(np.mean(np.square(window)))
            percent_mvc = (rms / mvc_rms) * 100
            socketio.emit('counter_update', {'count': round(percent_mvc,2)})
            print(f"current sample: {sample[0]} %MVC: {percent_mvc:.1f}% | RMS: {rms:.3f} µV | MVC_RMS: {mvc_rms:.3f}", end='\r')
            # time.sleep(0.2)


# Begin calibration when the calibration button is pressed
@socketio.on('button_pressed')
def begin_calibration(data):
    global rms, mvc_rms, is_collecting_data



    print(f"Button pressed! Message from client: {data}")
    # you can trigger any function here (e.g., start calibration, save data, etc.)
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
        latest_samples, timestamp = inlet1.pull_chunk()
        
        # Only proceed if latest_samples is not empty
        if len(latest_samples) == 0:
            continue  # skip this iteration
        
        # Get the most recent sample
        sample = latest_samples[-1]
        print(sample, "poop")
        calibration.append(sample[0])


    print(calibration)

    # Compute RMS
    mvc_rms = np.sqrt(np.mean(np.square(calibration)))
    print(f"MVC RMS: {mvc_rms:.3f} µV")
    print("You can now relax.\n")

    if not is_collecting_data:
        thread2 = threading.Thread(target=collecting_data)
        thread2.daemon = True
        is_collecting_data = True
        thread2.start()

# MAIN
if __name__ == '__main__':
    
    socketio.run(app, debug=True, port=8080)
    print("Looking for an OpenBCI LSL stream...")
    streams = resolve_byprop('type', 'EEG')  # Could also be 'EMG'
    inlet1 = StreamInlet(streams[0])

    # Start the background thread
    thread = threading.Thread(target=counter_thread)
    thread.daemon = True
    thread.start()

    # thread2 = threading.Thread(target=collecting_data)
    # thread2.daemon = True
    # thread2.start()

    # socketio.run(app, debug=True, port=8080)


    # print("Looking for an OpenBCI LSL stream...")
    # streams = resolve_byprop('type', 'EEG')  # Could also be 'EMG'
    # inlet1 = StreamInlet(streams[0])
    # inlet2 = StreamInlet(streams[1])


    print("Starting live %MVC tracking...\n")
    time.sleep(1)



    # socketio.run(app, debug=True, port=8080)

