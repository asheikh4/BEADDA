from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
import sys
from pathlib import Path

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import threading

# CAMERA SETUP
import cv2

# Add src/CV to path for imports
current_dir = Path(__file__).parent
cv_dir = current_dir / "src" / "CV"
if str(cv_dir) not in sys.path:
    sys.path.insert(0, str(cv_dir))

# Import CV modules
import mediapipe as mp
from pose_utils import (put, draw_angle_with_arc, EMA, MedianFilter, HoldTimer)
from exercises import (squat_metrics, wallsit_metrics, curl_metrics, RepCounter, 
                       FormAwareRepCounter, LumbarExcursionTracker)

# SETTING UP FLASK STUFF
app = Flask(__name__)
socketio = SocketIO(app)

# Camera setup
CAM_INDEX = 0
FRAME_W, FRAME_H, FPS = 1280, 720, 30
cap = cv2.VideoCapture(CAM_INDEX)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Exercise settings
exercise = "squat"  # "squat" | "wallsit" | "curl"
curl_side = "LEFT"

# Exercise thresholds
SQUAT_BOTTOM_DEG = 95.0
SQUAT_TOP_DEG = 140.0
CURL_BOTTOM_DEG = 45.0
CURL_TOP_DEG = 130.0

# Smoothing filters
ema_knee, med_knee = EMA(0.25), MedianFilter(5)
ema_elbow, med_elbow = EMA(0.25), MedianFilter(5)

# Rep counters
squat_counter = RepCounter(low_thresh=SQUAT_BOTTOM_DEG, high_thresh=SQUAT_TOP_DEG)
curl_counter = FormAwareRepCounter(low_thresh=CURL_BOTTOM_DEG, high_thresh=CURL_TOP_DEG)
hold_timer = HoldTimer()
lumbar_tracker = LumbarExcursionTracker()

# EMG variables
mvc_rms = 1
is_collecting_data = False
inlet1 = None

@app.route('/')
def index():
    return render_template('index.html')

# WEBCAM DISPLAY
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


inlet = None
rest_emg_rms = 0
rest_eeg_rms = 0
max_emg_rms = 0
max_eeg_rms = 0



def start_live_tracking():
    global inlet, rest_emg_rms, rest_eeg_rms, max_emg_rms, max_eeg_rms
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

            # Emit live percent update to frontend
            socketio.emit('live_percent_update', {
                'emg_percent': float(emg_percent),
                'eeg_percent': float(eeg_percent)
            })




def calibrate_and_start_emg_stream():
    global inlet, rest_emg_rms, rest_eeg_rms, max_emg_rms, max_eeg_rms
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

    start_live_tracking()

 

# SocketIO event: start EMG calibration
@socketio.on('start_emg_calibration')
def handle_start_emg_calibration(data):
    print("Received EMG calibration request from frontend.")
    calibrate_and_start_emg_stream()

# Helper functions for rendering exercises
def get_squat_cues(m, lumbar_excursion_deg=None):
    """Extract cues for squat form feedback."""
    cues = []
    if m["depth_zone"] in ("acceptable", "poor"):
        cues.append("Go deeper")
    if m["torso_lean_zone"] in ("acceptable", "poor"):
        cues.append("Stay more upright")
    if m["trunk_tibia_zone"] in ("acceptable", "poor"):
        cues.append("Align torso with shins")
    if m["tibia_angle_zone"] in ("acceptable", "poor"):
        cues.append("Ankle mobility / knees back")
    if m["depth_symmetry_zone"] in ("acceptable", "poor"):
        cues.append("Even depth L/R")
    if lumbar_excursion_deg is not None and lumbar_excursion_deg > 15.0:
        cues.append("Keep chest up / brace")
    if m["back_zone"] in ("acceptable", "poor"):
        cues.append("Keep back straight")
    return cues

def get_wallsit_cues(m):
    """Extract cues for wall-sit form feedback."""
    cues = []
    if m["knee_zone"] in ("acceptable", "poor"):
        cues.append("Aim for 90° at knee")
    if m["back_zone"] in ("acceptable", "poor"):
        cues.append("Back flat against wall")
    if m.get("knee_over_toe_norm") is not None and m["knee_over_toe_zone"] in ("acceptable", "poor"):
        cues.append("Knees over ankles")
    return cues

def get_curl_cues(m):
    """Extract cues for curl form feedback."""
    cues = []
    if m["elbow_zone"] in ("acceptable", "poor"):
        cues.append("Squeeze at top / Full contraction")
    if m["upper_arm_zone"] in ("acceptable", "poor"):
        cues.append("Keep upper arm still")
    return cues

def render_squat(frame, m, knee, reps, phase, cues, lumbar_excursion_deg=None):
    """Render squat-specific UI elements."""
    jp = m["joints_px"]
    draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                        angle_deg=knee, color=m["depth_color"], show_text=True)
    cv2.line(frame, jp["shoulder"], jp["hip"], m["torso_lean_color"], 3)
    cv2.line(frame, jp["ankle"], jp["knee"], m["tibia_angle_color"], 2)
    
    y_offset = 30
    put(frame, f"[SQUAT] Reps: {reps} | Phase: {phase.upper()}", y_offset, (0, 255, 255))
    y_offset += 30
    put(frame, f"Depth: {int(knee)}° ({m['depth_zone']})", y_offset, m["depth_color"])
    y_offset += 30
    put(frame, f"Torso Lean: {int(m['torso_lean_deg'])}° ({m['torso_lean_zone']})", 
        y_offset, m["torso_lean_color"])
    y_offset += 30
    if lumbar_excursion_deg is not None:
        lumbar_color = (0, 255, 0) if lumbar_excursion_deg <= 15.0 else (
            (0, 165, 255) if lumbar_excursion_deg <= 20.0 else (0, 0, 255))
        put(frame, f"Lumbar Excursion: {lumbar_excursion_deg:.1f}°", 
            y_offset, lumbar_color)
        y_offset += 30
    put(frame, " | ".join(cues) if cues else "✓ Good form!", y_offset,
        (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

def render_wallsit(frame, m, seconds, cues):
    """Render wall-sit specific UI elements."""
    jp = m["joints_px"]
    draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                        angle_deg=m["knee_angle"], color=m["knee_color"], show_text=True)
    cv2.line(frame, jp["shoulder"], jp["hip"], m["back_color"], 3)
    
    y_offset = 30
    put(frame, f"[WALL-SIT] Hold: {seconds:.1f}s", y_offset, (0, 255, 255))
    y_offset += 30
    put(frame, f"Knee Angle: {int(m['knee_angle'])}° ({m['knee_zone']})", y_offset, m["knee_color"])
    y_offset += 30
    put(frame, f"Back Alignment: {int(m['torso_vertical_deg'])}° ({m['back_zone']})",
        y_offset, m["back_color"])
    y_offset += 30
    put(frame, " | ".join(cues) if cues else "✓ Solid hold!", y_offset,
        (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

def render_curl(frame, m, elbow, reps, phase, cues, side):
    """Render curl-specific UI elements."""
    jp = m["joints_px"]
    draw_angle_with_arc(frame, jp["shoulder"], jp["elbow"], jp["wrist"],
                        angle_deg=elbow, color=m["elbow_color"], show_text=True)
    cv2.line(frame, jp["shoulder"], jp["elbow"], m["upper_arm_color"], 3)
    
    y_offset = 30
    put(frame, f"[CURL - {side}] Reps: {reps} | Phase: {phase.upper()}", y_offset, (0, 255, 255))
    y_offset += 30
    put(frame, f"Elbow Angle: {int(elbow)}° ({m['elbow_zone']})", y_offset, m["elbow_color"])
    y_offset += 30
    put(frame, f"Upper Arm Alignment: {int(m['upper_arm_vertical_deg'])}° ({m['upper_arm_zone']})",
        y_offset, m["upper_arm_color"])
    y_offset += 30
    put(frame, " | ".join(cues) if cues else "✓ Clean rep!",
        y_offset, (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

# FRAMES FOR CAMERA with CV processing
def generate_frames():
    global cap, exercise, curl_side, pose
    global squat_counter, curl_counter, hold_timer, lumbar_tracker
    global ema_knee, med_knee, ema_elbow, med_elbow
    
    last_frame_time = time.time()
    frame_time_target = 1.0 / FPS
    frame_count = 0
    
    while True:
        # FPS limiting
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)
        last_frame_time = time.time()
        
        # Read camera frame
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to read frame from camera")
            break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        
        # Draw pose landmarks
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        
        # Process exercise metrics
        if lm:
            if exercise == "squat":
                m = squat_metrics(lm, w, h, mp_pose)
                if m:
                    knee = ema_knee(med_knee(m["knee_angle"]))
                    reps, phase = squat_counter.update(knee)
                    # Track lumbar excursion
                    if phase == "down" and not lumbar_tracker.rep_active:
                        lumbar_tracker.start_rep(m["torso_lean_deg"])
                    lumbar_tracker.update(m["torso_lean_deg"], phase)
                    lumbar_excursion = lumbar_tracker.get_excursion()
                    cues = get_squat_cues(m, lumbar_excursion)
                    render_squat(frame, m, knee, reps, phase, cues, lumbar_excursion)
                    # print(int(reps))
                    socketio.emit('rep_update', {'reps': reps})
            elif exercise == "wallsit":
                m = wallsit_metrics(lm, w, h, mp_pose)
                if m:
                    seconds = hold_timer.update(m["knee_90"] and m["back_vertical"])
                    cues = get_wallsit_cues(m)
                    render_wallsit(frame, m, seconds, cues)
            elif exercise == "curl":
                m = curl_metrics(lm, w, h, mp_pose, side=curl_side)
                if m:
                    elbow = ema_elbow(med_elbow(m["elbow_angle"]))
                    zones = {
                        "elbow_zone": m["elbow_zone"],
                        "upper_arm_zone": m["upper_arm_zone"]
                    }
                    reps, phase = curl_counter.update(elbow, zones=zones)
                    cues = get_curl_cues(m)
                    render_curl(frame, m, elbow, reps, phase, cues, curl_side)
                    socketio.emit('rep_update', {'reps': reps})
        else:
            # No pose detected
            put(frame, "No pose detected. Position yourself in frame.", 30, (0, 0, 255))
            put(frame, f"Exercise: {exercise.upper()}", 60, (255, 255, 255))
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Warning: Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames. Pose detected: {lm is not None}")





# TEST COUNTER
def counter_thread():
    count = 0
    while True:
        socketio.emit('counter_update', {'count': count})
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


# Handle exercise change requests from the web UI
@socketio.on('change_exercise')
def handle_change_exercise(data):
    """Change the global exercise mode based on client request.

    Expects data to be a dict with key 'exercise' and value one of: 'squat',
    'wallsit', 'curl'. Emits an 'exercise_changed' event back to clients.
    """
    global exercise
    try:
        ex = data.get('exercise')
    except Exception:
        ex = None

    if ex in ('squat', 'wallsit', 'curl'):
        exercise = ex
        print(f"Exercise changed to: {exercise}")
        # notify connected clients of the change
        socketio.emit('exercise_changed', {'exercise': exercise})
    else:
        print(f"Received invalid exercise change request: {data}")

# MAIN
if __name__ == '__main__':

    # Start the background thread
    thread = threading.Thread(target=counter_thread)
    thread.daemon = True
    thread.start()

    # Setup camera FIRST (before starting server)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAM_INDEX}")
        print("Please check that your camera is connected.")
        exit(1)
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized: {actual_w}x{actual_h}")
    print(f"Computer Vision: {exercise} mode")
    print(f"MediaPipe Pose Detection: Enabled")


    socketio.run(app, debug=True, port=8080, host='0.0.0.0')
    
    # # Try to initialize EMG stream (optional)
    # try:
    #     print("Looking for an OpenBCI LSL stream...")
    #     streams = resolve_byprop('type', 'EEG')
    #     if streams:
    #         inlet1 = StreamInlet(streams[0])
    #         print("EMG stream connected")
    #     else:
    #         print("No EMG stream found - continuing without EMG")
    # except Exception as e:
    #     print(f"EMG stream not available: {e}")
    #     print("Continuing without EMG functionality")
    
 
    
    # print(f"Starting web server on port 8080...")
    # print("Visit http://localhost:8080 to view the interface")
    
    # socketio.run(app, debug=True, port=8080, host='0.0.0.0')

