# src/CV/main.py
# Webcam app that uses MediaPipe Pose + your exercises metrics.
# 
# DATA COLLECTION OPTIMIZED:
# - Enhanced MediaPipe configuration (higher confidence thresholds)
# - Real-time data quality monitoring (visibility, lighting, scale)
# - Calibration mode for optimal setup
# - Camera settings optimization (exposure, white balance, focus)
# - Quality indicators displayed on-screen
# 
# Keys: [1]=Squat  [2]=Wall-sit  [3]=Curl  [R]=Reset  [L]=Left arm  [Q]=Right arm  [C]=Calibrate  [ESC]=Quit

import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import cv2
import numpy as np
import mediapipe as mp
from pose_utils import (put, draw_angle_with_arc, EMA, MedianFilter, HoldTimer,
                        check_pose_quality, check_lighting_quality, validate_person_scale,
                        get_landmark_visibility, AdaptiveEMA)
from exercises import (squat_metrics, wallsit_metrics, curl_metrics, RepCounter, 
                       FormAwareRepCounter, LumbarExcursionTracker)

# ---------------- Enhanced Config for Data Collection ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H, FPS = 1280, 720, 30

# MediaPipe settings (higher for data quality)
MODEL_COMPLEXITY = 2
MIN_DET_CONF = 0.75  # Increased from 0.70 for better data quality
MIN_TRK_CONF = 0.75  # Increased from 0.70 for better data quality
SMOOTH_LANDMARKS = True

# Data quality thresholds
MIN_VISIBILITY_SCORE = 0.7  # Minimum landmark visibility (70%)
MIN_LIGHTING_BRIGHTNESS = 50  # Minimum frame brightness
MAX_LIGHTING_BRIGHTNESS = 200  # Maximum frame brightness
MIN_CONTRAST = 30  # Minimum frame contrast
PERSON_SCALE_MIN = 0.3  # Minimum person height / frame height (30%)
PERSON_SCALE_MAX = 0.8  # Maximum person height / frame height (80%)
QUALITY_THRESHOLD = 0.7  # Minimum overall quality to process frame

# Smoothing (can be adjusted based on data quality)
EMA_ALPHA = 0.25
MEDIAN_FILTER_SIZE = 5

# Rep thresholds (for knee/elbow angles)
# Squat: based on depth thresholds (100° perfect, so 95° ensures full depth)
SQUAT_BOTTOM_DEG = 95.0
SQUAT_TOP_DEG    = 140.0
# Curl: based on research (45° perfect contraction, so 45° ensures full contraction)
# Top is when arm is extended (~130-160° typical)
CURL_BOTTOM_DEG  = 45.0  # Full contraction (research-based)
CURL_TOP_DEG     = 130.0  # Extended arm

cap = cv2.VideoCapture(CAM_INDEX)

global_frame = None

# ---------------- Setup ----------------
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---------------- Camera Setup ----------------
def setup_camera_optimal(cap, target_w=1280, target_h=720):
    """Configure camera for optimal pose detection and data collection."""
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    
    # Enable auto-exposure (helps with lighting changes)
    # Try to set auto-exposure, fallback if not supported
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = auto, 0.25 = manual
    except:
        pass  # Some cameras don't support this
    
    # Auto white balance (helps with color consistency)
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)  # 1.0 = auto, 0.0 = manual
    except:
        pass
    
    # Focus (if supported)
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto-focus
    except:
        pass
    
    # Frame rate
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Buffer size (reduce latency)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    
    return cap

# ---------------- Data Quality Display ----------------
def render_data_quality(frame, quality_score, is_valid, issues, lighting_ok, 
                       scale_ok, scale_factor, y_start=10):
    """Display data quality indicators for data collection."""
    if frame is None or frame.size == 0:
        return
    
    h, w = frame.shape[:2]
    bar_width = 200
    bar_height = 20
    bar_x = w - bar_width - 10
    
    # Quality indicator bar
    if is_valid and lighting_ok and scale_ok:
        bar_color = (0, 255, 0)  # Green
        status_text = "READY"
    elif quality_score > 0.5:
        bar_color = (0, 165, 255)  # Orange
        status_text = "MARGINAL"
    else:
        bar_color = (0, 0, 255)  # Red
        status_text = "POOR"
    
    # Draw quality bar background
    cv2.rectangle(frame, (bar_x - 2, y_start - 2),
                  (bar_x + bar_width + 2, y_start + bar_height + 2),
                  (0, 0, 0), -1)
    
    # Draw quality bar
    cv2.rectangle(frame, (bar_x, y_start),
                  (bar_x + int(bar_width * quality_score), y_start + bar_height),
                  bar_color, -1)
    cv2.rectangle(frame, (bar_x, y_start),
                  (bar_x + bar_width, y_start + bar_height),
                  (255, 255, 255), 2)
    
    # Status text
    put(frame, f"Data Quality: {status_text} ({quality_score:.0%})",
        y_start + bar_height + 20, bar_color, scale=0.6, x=bar_x)
    
    # Lighting status
    lighting_color = (0, 255, 0) if lighting_ok else (0, 0, 255)
    lighting_text = "Lighting: OK" if lighting_ok else "Lighting: ADJUST"
    put(frame, lighting_text, y_start + bar_height + 40,
        lighting_color, scale=0.5, x=bar_x)
    
    # Scale status
    scale_color = (0, 255, 0) if scale_ok else (0, 0, 255)
    scale_text = f"Distance: OK ({scale_factor:.0%})" if scale_ok else f"Distance: {scale_factor:.0%}"
    put(frame, scale_text, y_start + bar_height + 60,
        scale_color, scale=0.5, x=bar_x)
    
    # Show issues if any
    if issues:
        for i, issue in enumerate(issues[:2]):  # Show first 2 issues
            put(frame, f"⚠ {issue}", y_start + bar_height + 80 + (i * 18),
                (0, 0, 255), scale=0.5, x=bar_x)

# ---------------- Calibration Mode ----------------
def calibration_mode(cap, pose, mp_pose, mp_draw):
    """
    Calibration mode to help user set up camera properly.
    Guides user through optimal positioning for data collection.
    """
    print("=== CALIBRATION MODE ===")
    print("Position yourself in frame. Press SPACE when ready, ESC to skip.")
    
    calibration_ready = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        
        # Check lighting
        lighting_ok, brightness, lighting_issues = check_lighting_quality(frame)
        
        if lm:
            # Check pose quality
            pose_ok, quality_score, pose_issues = check_pose_quality(lm, mp_pose, "squat")
            
            # Check scale
            scale_ok, scale_factor, scale_issue = validate_person_scale(lm, mp_pose, w, h)
            
            # Draw calibration guides
            cv2.putText(frame, "CALIBRATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Adjust position and lighting", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Lighting status
            lighting_color = (0, 255, 0) if lighting_ok else (0, 0, 255)
            lighting_status = "OK" if lighting_ok else f"ADJUST - {', '.join(lighting_issues[:2])}"
            cv2.putText(frame, f"Lighting: {lighting_status}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lighting_color, 2)
            
            # Pose quality
            pose_color = (0, 255, 0) if pose_ok else (0, 0, 255)
            pose_status = "OK" if pose_ok else f"ADJUST - {', '.join(pose_issues[:2])}"
            cv2.putText(frame, f"Pose: {pose_status} ({quality_score:.0%})",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)
            
            # Scale
            scale_color = (0, 255, 0) if scale_ok else (0, 0, 255)
            scale_status = "OK" if scale_ok else scale_issue
            cv2.putText(frame, f"Distance: {scale_status} ({scale_factor:.1%})",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, scale_color, 2)
            
            # Draw pose
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
            
            # All checks passed
            if lighting_ok and pose_ok and scale_ok:
                cv2.putText(frame, "READY! Press SPACE to continue", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                calibration_ready = True
            else:
                cv2.putText(frame, "Adjust position and lighting", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No pose detected. Ensure good lighting.", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            lighting_color = (0, 255, 0) if lighting_ok else (0, 0, 255)
            lighting_status = "OK" if lighting_ok else f"ADJUST - {', '.join(lighting_issues[:2])}"
            cv2.putText(frame, f"Lighting: {lighting_status}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lighting_color, 2)
        
        cv2.putText(frame, "Press SPACE when ready, ESC to skip", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE to confirm
            break
        elif key == 27:  # ESC to skip
            calibration_ready = False
            break
    
    cv2.destroyWindow("Calibration")
    if calibration_ready:
        print("Calibration complete. Ready for data collection.")
    else:
        print("Calibration skipped.")
    return calibration_ready

def draw_legend(frame):
    """Draw the tolerance zone legend with adaptive positioning."""
    if frame is None or frame.size == 0:
        return
    h, w = frame.shape[:2]
    legend_width = min(210, w // 4)  # Adaptive width
    legend_x = max(10, w - legend_width - 10)
    legend_y = 30
    legend_height = 100
    
    # Ensure legend fits on screen
    if legend_y + legend_height > h:
        legend_y = max(10, h - legend_height - 10)
    
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20),
                  (min(w - 10, legend_x + legend_width), legend_y + legend_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20),
                  (min(w - 10, legend_x + legend_width), legend_y + legend_height), (255, 255, 255), 1)
    cv2.putText(frame, "Tolerance Zones:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "GREEN = Perfect",     (legend_x, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW = Good",       (legend_x, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "ORANGE = Acceptable", (legend_x, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "RED = Needs Work",    (legend_x, legend_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

def render_squat(frame, m, knee, reps, phase, cues, lumbar_excursion_deg=None):
    """Render squat-specific UI elements with evidence-based metrics."""
    jp = m["joints_px"]
    
    # Draw knee angle arc
    draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                        angle_deg=knee, color=m["depth_color"], show_text=True)
    
    # Draw torso line (shoulder-hip) in torso lean color
    cv2.line(frame, jp["shoulder"], jp["hip"], m["torso_lean_color"], 3)
    
    # Draw shin line (tibia) in tibia angle color
    cv2.line(frame, jp["ankle"], jp["knee"], m["tibia_angle_color"], 2)
    
    # HUD - Primary metrics (evidence-based)
    y_offset = 30
    put(frame, f"[SQUAT] Reps: {reps} | Phase: {phase.upper()}", y_offset, (0, 255, 255))
    y_offset += 30
    put(frame, f"Depth: {int(knee)}° ({m['depth_zone']})", y_offset, m["depth_color"])
    y_offset += 30
    put(frame, f"Torso Lean: {int(m['torso_lean_deg'])}° ({m['torso_lean_zone']})", 
        y_offset, m["torso_lean_color"])
    y_offset += 30
    put(frame, f"Trunk-Tibia: {int(m['trunk_tibia_diff'])}° ({m['trunk_tibia_zone']})", 
        y_offset, m["trunk_tibia_color"])
    y_offset += 30
    put(frame, f"Tibia Angle: {int(m['tibia_angle_deg'])}° ({m['tibia_angle_zone']})", 
        y_offset, m["tibia_angle_color"])
    y_offset += 30
    put(frame, f"Symmetry: {m['depth_symmetry_deg']:.1f}° ({m['depth_symmetry_zone']})", 
        y_offset, m["depth_symmetry_color"])
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
    """Render wall-sit specific UI elements with evidence-based metrics."""
    jp = m["joints_px"]
    
    # Draw knee angle arc
    draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                        angle_deg=m["knee_angle"], color=m["knee_color"], show_text=True)
    # Torso line (back alignment)
    cv2.line(frame, jp["shoulder"], jp["hip"], m["back_color"], 3)
    # Draw line from ankle to knee to visualize knee-over-toe
    if "heel" in jp and "toe" in jp:
        cv2.line(frame, jp["ankle"], jp["knee"], m["knee_over_toe_color"], 2)
    
    # HUD - Evidence-based metrics
    y_offset = 30
    put(frame, f"[WALL-SIT] Hold: {seconds:.1f}s", y_offset, (0, 255, 255))
    y_offset += 30
    put(frame, f"Knee Angle: {int(m['knee_angle'])}° ({m['knee_zone']})", y_offset, m["knee_color"])
    y_offset += 30
    put(frame, f"Back Alignment: {int(m['torso_vertical_deg'])}° ({m['back_zone']})",
        y_offset, m["back_color"])
    y_offset += 30
    if m["knee_over_toe_norm"] is not None:
        put(frame, f"Knee Over Toe: {m['knee_over_toe_norm']*100:.1f}% ({m['knee_over_toe_zone']})",
            y_offset, m["knee_over_toe_color"])
        y_offset += 30
    put(frame, " | ".join(cues) if cues else "✓ Solid hold!", y_offset,
        (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

def render_curl(frame, m, elbow, reps, phase, cues, side):
    """Render curl-specific UI elements with evidence-based metrics."""
    jp = m["joints_px"]
    
    # Draw elbow angle arc (contraction)
    draw_angle_with_arc(frame, jp["shoulder"], jp["elbow"], jp["wrist"],
                        angle_deg=elbow, color=m["elbow_color"], show_text=True)
    
    # Upper arm line (colored by vertical alignment zone)
    cv2.line(frame, jp["shoulder"], jp["elbow"], m["upper_arm_color"], 3)
    
    # HUD - Evidence-based metrics
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

def get_squat_cues(m, lumbar_excursion_deg=None):
    """Extract cues for squat form feedback using evidence-based metrics."""
    cues = []
    # Depth
    if m["depth_zone"] in ("acceptable", "poor"):
        cues.append("Go deeper")
    # Torso lean
    if m["torso_lean_zone"] in ("acceptable", "poor"):
        cues.append("Stay more upright")
    # Trunk-tibia harmony
    if m["trunk_tibia_zone"] in ("acceptable", "poor"):
        cues.append("Align torso with shins")
    # Tibia angle (knee travel)
    if m["tibia_angle_zone"] in ("acceptable", "poor"):
        cues.append("Ankle mobility / knees back")
    # Depth symmetry
    if m["depth_symmetry_zone"] in ("acceptable", "poor"):
        cues.append("Even depth L/R")
    # Lumbar excursion
    if lumbar_excursion_deg is not None and lumbar_excursion_deg > 15.0:
        cues.append("Keep chest up / brace")
    # Back straightness
    if m["back_zone"] in ("acceptable", "poor"):
        cues.append("Keep back straight")
    return cues

def get_wallsit_cues(m):
    """Extract cues for wall-sit form feedback using evidence-based metrics."""
    cues = []
    # Knee angle
    if m["knee_zone"] in ("acceptable", "poor"):
        cues.append("Aim for 90° at knee")
    # Back alignment
    if m["back_zone"] in ("acceptable", "poor"):
        cues.append("Back flat against wall")
    # Knee over toe
    if m.get("knee_over_toe_norm") is not None and m["knee_over_toe_zone"] in ("acceptable", "poor"):
        cues.append("Knees over ankles")
    return cues

def get_curl_cues(m):
    """Extract cues for curl form feedback using evidence-based metrics."""
    cues = []
    # Elbow contraction
    if m["elbow_zone"] in ("acceptable", "poor"):
        cues.append("Squeeze at top / Full contraction")
    # Upper arm stability (prevents cheating/swinging)
    if m["upper_arm_zone"] in ("acceptable", "poor"):
        cues.append("Keep upper arm still")
    return cues


def main():
    global cap, global_frame

    """Main application loop with enhanced data collection quality checks."""
    # Initialize camera with error handling
    # cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAM_INDEX}")
        print("Please check that your camera is connected and not being used by another application.")
        return
    
    # Setup camera with optimal settings for data collection
    cap = setup_camera_optimal(cap, FRAME_W, FRAME_H)
    
    # Verify camera works
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        cap.release()
        return
    
    global_frame = frame
    
    # Report actual camera resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized: {actual_w}x{actual_h}")
    print(f"MediaPipe settings: Model={MODEL_COMPLEXITY}, DetConf={MIN_DET_CONF}, TrkConf={MIN_TRK_CONF}")
    print(f"Data quality thresholds: Visibility>={MIN_VISIBILITY_SCORE:.0%}, Quality>={QUALITY_THRESHOLD:.0%}")

    exercise = "squat"   # "squat" | "wallsit" | "curl"
    curl_side = "LEFT"   # "LEFT" | "RIGHT"
    exercise_switch_time = None
    switch_message_duration = 2.0  # seconds

    # Smoothers for displayed angles
    ema_knee, med_knee   = EMA(0.25), MedianFilter(5)
    ema_elbow, med_elbow = EMA(0.25), MedianFilter(5)

    # Rep counters
    squat_counter = RepCounter(low_thresh=SQUAT_BOTTOM_DEG, high_thresh=SQUAT_TOP_DEG)
    # Use form-aware counter for curls - only counts reps when form is good
    curl_counter  = FormAwareRepCounter(low_thresh=CURL_BOTTOM_DEG, high_thresh=CURL_TOP_DEG)

    # Hold timer for wall-sit
    hold_timer = HoldTimer()
    
    # Lumbar excursion tracker for squats
    lumbar_tracker = LumbarExcursionTracker()
    
    # FPS limiting
    last_frame_time = time.time()
    frame_time_target = 1.0 / FPS

    try:
        with mp_pose.Pose(
            model_complexity=MODEL_COMPLEXITY,
            smooth_landmarks=SMOOTH_LANDMARKS,
            enable_segmentation=False,
            min_detection_confidence=MIN_DET_CONF,
            min_tracking_confidence=MIN_TRK_CONF
        ) as pose:
            
            # Optional calibration mode
            print("\nPress 'C' during session to enter calibration mode, or continue with current setup.")
            

            while True:
                # FPS limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time_target:
                    time.sleep(frame_time_target - elapsed)
                last_frame_time = time.time()
                
                ok, frame = cap.read()
                if not ok:
                    print("Warning: Failed to read frame from camera")
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                # MediaPipe normalized landmarks list
                lm = res.pose_landmarks.landmark if res.pose_landmarks else None
                
                # Data quality checks (for data collection optimization)
                lighting_ok, brightness_score, lighting_issues = check_lighting_quality(frame)
                scale_ok, scale_factor, scale_issue = (True, 0.5, None)  # Default
                pose_ok, quality_score, pose_issues = (False, 0.0, [])
                
                if lm:
                    # Check pose quality
                    pose_ok, quality_score, pose_issues = check_pose_quality(lm, mp_pose, exercise)
                    # Check scale
                    scale_ok, scale_factor, scale_issue = validate_person_scale(lm, mp_pose, w, h)
                    
                    # Draw landmarks (nice official style)
                    mp_draw.draw_landmarks(
                        frame,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
                else:
                    pose_issues = ["No pose detected"]

                # Render data quality indicators (top right corner)
                render_data_quality(frame, quality_score, pose_ok, pose_issues, 
                                   lighting_ok, scale_ok, scale_factor, y_start=10)

                # Process exercise metrics only if quality is acceptable
                if lm:
                    # Optionally skip processing if quality is too low (comment out for always process)
                    # if quality_score < QUALITY_THRESHOLD:
                    #     put(frame, f"⚠ Low quality frame - processing anyway", 250, (0, 0, 255))
                    
                    if exercise == "squat":
                        m = squat_metrics(lm, w, h, mp_pose)
                        if m:
                            # Angle smoothing for display
                            knee = ema_knee(med_knee(m["knee_angle"]))
                            
                            # Rep count on knee angle
                            reps, phase = squat_counter.update(knee)
                            
                            # Track lumbar excursion
                            if phase == "down" and not lumbar_tracker.rep_active:
                                lumbar_tracker.start_rep(m["torso_lean_deg"])
                            lumbar_tracker.update(m["torso_lean_deg"], phase)
                            lumbar_excursion = lumbar_tracker.get_excursion()
                            
                            # Get cues
                            cues = get_squat_cues(m, lumbar_excursion)
                            
                            # Render
                            render_squat(frame, m, knee, reps, phase, cues, lumbar_excursion)

                    elif exercise == "wallsit":
                        m = wallsit_metrics(lm, w, h, mp_pose)
                        if m:
                            seconds = hold_timer.update(m["knee_90"] and m["back_vertical"])
                            
                            # Get cues
                            cues = get_wallsit_cues(m)
                            
                            # Render
                            render_wallsit(frame, m, seconds, cues)

                    elif exercise == "curl":
                        m = curl_metrics(lm, w, h, mp_pose, side=curl_side)
                        if m:
                            elbow = ema_elbow(med_elbow(m["elbow_angle"]))
                            # Pass form zones to rep counter - only counts if all are good/acceptable
                            zones = {
                                "elbow_zone": m["elbow_zone"],
                                "upper_arm_zone": m["upper_arm_zone"]
                            }
                            reps, phase = curl_counter.update(elbow, zones=zones)
                            
                            # Get cues
                            cues = get_curl_cues(m)
                            
                            # Render
                            render_curl(frame, m, elbow, reps, phase, cues, curl_side)

                # Draw legend and controls
                draw_legend(frame)
                put(frame, "Keys: [1]=Squat  [2]=Wall-sit  [3]=Curl  [R]=Reset  [L]=Left  [Q]=Right  [C]=Calibrate  [ESC]=Quit",
                    frame.shape[0] - 10, (255, 255, 255), 0.6, 1)
                
                # Show exercise switch message
                if exercise_switch_time:
                    elapsed_switch = time.time() - exercise_switch_time
                    if elapsed_switch < switch_message_duration:
                        put(frame, f"Switched to: {exercise.upper()}", 210, (255, 255, 0), scale=0.8)
                    else:
                        exercise_switch_time = None

                cv2.imshow("AI Physio (Webcam)", frame)

                

                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                elif k == ord('c') or k == ord('C'):  # Calibration mode
                    calibration_mode(cap, pose, mp_pose, mp_draw)
                elif k == ord('1'):
                    exercise = "squat"
                    exercise_switch_time = time.time()
                    # Reset counters and smoothers
                    squat_counter = RepCounter(low_thresh=SQUAT_BOTTOM_DEG, high_thresh=SQUAT_TOP_DEG)
                    ema_knee, med_knee = EMA(0.25), MedianFilter(5)
                    lumbar_tracker.reset()
                elif k == ord('2'):
                    exercise = "wallsit"
                    exercise_switch_time = time.time()
                    # Reset timer
                    hold_timer = HoldTimer()
                elif k == ord('3'):
                    exercise = "curl"
                    exercise_switch_time = time.time()
                    # Reset counters and smoothers
                    curl_counter = FormAwareRepCounter(low_thresh=CURL_BOTTOM_DEG, high_thresh=CURL_TOP_DEG)
                    ema_elbow, med_elbow = EMA(0.25), MedianFilter(5)
                elif k == ord('r') or k == ord('R'):
                    # Manual reset for current exercise
                    if exercise == "squat":
                        squat_counter.reset()
                        ema_knee, med_knee = EMA(0.25), MedianFilter(5)
                    elif exercise == "curl":
                        curl_counter.reset()
                        ema_elbow, med_elbow = EMA(0.25), MedianFilter(5)
                    elif exercise == "wallsit":
                        hold_timer = HoldTimer()
                elif k == ord('l') or k == ord('L'):
                    # Switch curl to left side
                    if exercise == "curl":
                        curl_side = "LEFT"
                        curl_counter.reset()  # Reset when switching sides
                elif k == ord('q') or k == ord('Q'):
                    # Switch curl to right side (q for Right to avoid conflict with Reset)
                    if exercise == "curl":
                        curl_side = "RIGHT"
                        curl_counter.reset()  # Reset when switching sides

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()
