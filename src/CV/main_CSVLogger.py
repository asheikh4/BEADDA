# src/cv/main.py
# Enhanced with CSV logging for good reps and ML training integration

import cv2
import time
import collections
import numpy as np
import mediapipe as mp
import math

# Import our enhanced exercises module with CSV logging
from exercises_CSVLogger import (
    squat_metrics, wallsit_metrics, curl_metrics,
    FormAwareRepCounter, CSVLogger, is_good_rep
)

# Optional: Import feedback system if available
try:
    from feedback_system import FeedbackSystem
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    print("⚠️  Feedback system not available. Run in data collection mode.")

# ===================== Config =====================
MODEL_COMPLEXITY = 2
MIN_DET_CONF = 0.70
MIN_TRK_CONF = 0.70
VIS_THR = 0.50
CAM_INDEX = 0

# Rep detection thresholds
SQUAT_TOP_DEG   = 140.0
SQUAT_BOTTOM_DEG= 95.0
CURL_TOP_DEG    = 130.0
CURL_BOTTOM_DEG = 50.0
MIN_PHASE_FRAMES = 3

# ===================== Utils ======================
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def put(img, txt, y, col=(255,255,255), scale=0.7, thick=2):
    cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

class MedianFilter:
    def __init__(self, k=5):
        self.buf = collections.deque(maxlen=k)
    def __call__(self, x):
        self.buf.append(float(x))
        return float(np.median(self.buf)) if self.buf else float(x)

class EMA:
    def __init__(self, alpha=0.25):
        self.alpha, self.v = alpha, None
    def __call__(self, x):
        self.v = x if (self.v is None) else (1 - self.alpha) * self.v + self.alpha * x
        return self.v

class PhaseFSM:
    """Phase machine with hysteresis + min frame duration."""
    def __init__(self, enter_bottom, leave_bottom):
        self.enter_bottom = enter_bottom
        self.leave_bottom = leave_bottom
        self.state = "top"
        self.frames_in_state = 0
        self.reps = 0
        self.rep_just_completed = False  # NEW: Track when rep completes

    def update_squat(self, knee_angle):
        self.rep_just_completed = False  # Reset flag
        nxt = self.state
        if self.state == "top" and knee_angle < self.enter_bottom:
            nxt = "down"
        elif self.state == "down":
            if knee_angle < self.enter_bottom:
                nxt = "bottom"
            elif knee_angle >= self.leave_bottom:
                nxt = "top"
        elif self.state == "bottom" and knee_angle > self.leave_bottom:
            nxt = "top"

        if nxt == self.state:
            self.frames_in_state += 1
        else:
            if self.frames_in_state >= MIN_PHASE_FRAMES:
                if self.state == "bottom" and nxt == "top":
                    self.reps += 1
                    self.rep_just_completed = True  # NEW: Rep completed!
                self.state = nxt
                self.frames_in_state = 1
            else:
                self.frames_in_state += 1
        return self.reps, self.state

    def update_curl(self, elbow_angle):
        self.rep_just_completed = False
        nxt = self.state
        if self.state == "top" and elbow_angle < self.enter_bottom:
            nxt = "down"
        elif self.state == "down":
            if elbow_angle < self.enter_bottom:
                nxt = "bottom"
            elif elbow_angle >= self.leave_bottom:
                nxt = "top"
        elif self.state == "bottom" and elbow_angle > self.leave_bottom:
            nxt = "top"

        if nxt == self.state:
            self.frames_in_state += 1
        else:
            if self.frames_in_state >= MIN_PHASE_FRAMES:
                if self.state == "bottom" and nxt == "top":
                    self.reps += 1
                    self.rep_just_completed = True
                self.state = nxt
                self.frames_in_state = 1
            else:
                self.frames_in_state += 1
        return self.reps, self.state

def draw_angle_arc(img, center, radius, start_angle, end_angle, color, thickness=2):
    """Draw an arc representing an angle."""
    axes = (radius, radius)
    angle = int(start_angle)
    end_angle_deg = int(end_angle)
    cv2.ellipse(img, center, axes, 0, angle, end_angle_deg, color, thickness)

def draw_angle_with_arc(img, pt1, pt2, pt3, angle, color=(0, 255, 0), show_angle_text=True):
    """Draw angle visualization with arc."""
    if pt1 is None or pt2 is None or pt3 is None:
        return None, None, None
    
    h, w = img.shape[:2]
    
    try:
        if isinstance(pt1[0], float):  # Normalized coordinates
            pt1_px = (int(pt1[0] * w), int(pt1[1] * h))
            pt2_px = (int(pt2[0] * w), int(pt2[1] * h))
            pt3_px = (int(pt3[0] * w), int(pt3[1] * h))
        else:  # Pixel coordinates
            pt1_px = (int(pt1[0]), int(pt1[1]))
            pt2_px = (int(pt2[0]), int(pt2[1]))
            pt3_px = (int(pt3[0]), int(pt3[1]))
    except (TypeError, IndexError):
        return None, None, None
    
    cv2.line(img, pt1_px, pt2_px, color, 3)
    cv2.line(img, pt2_px, pt3_px, color, 3)
    
    vec1 = np.array(pt1_px, dtype=float) - np.array(pt2_px, dtype=float)
    vec2 = np.array(pt3_px, dtype=float) - np.array(pt2_px, dtype=float)
    
    if np.linalg.norm(vec1) < 5 or np.linalg.norm(vec2) < 5:
        return pt1_px, pt2_px, pt3_px
    
    angle1 = math.degrees(math.atan2(vec1[1], vec1[0]))
    angle2 = math.degrees(math.atan2(vec2[1], vec2[0]))
    
    if angle1 < 0:
        angle1 += 360
    if angle2 < 0:
        angle2 += 360
    
    radius = max(15, min(40, np.linalg.norm(vec1) * 0.25, np.linalg.norm(vec2) * 0.25))
    cv2.ellipse(img, pt2_px, (int(radius), int(radius)), 0, 
               int(angle1), int(angle2), color, 2)
    
    if show_angle_text:
        offset_x = int(radius * 1.5 * math.cos(math.radians((angle1 + angle2) / 2)))
        offset_y = int(radius * 1.5 * math.sin(math.radians((angle1 + angle2) / 2)))
        text_pos = (pt2_px[0] + offset_x, pt2_px[1] + offset_y)
        cv2.putText(img, f"{int(angle)}°", text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    return pt1_px, pt2_px, pt3_px

# ===================== Main App =====================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Exercise mode
    exercise = "squat"  # "squat" | "wallsit" | "curl"
    
    # Mode selection: "collect" or "feedback"
    mode = "collect"  # Start in data collection mode
    
    # Initialize CSV logger
    csv_logger = CSVLogger(data_folder="data")
    
    # Initialize feedback system if available
    feedback_system = None
    if FEEDBACK_AVAILABLE:
        feedback_system = FeedbackSystem()
        feedback_system.load_all_models()
        if feedback_system.loaded_exercises:
            mode = "feedback"
            print("\n✅ Feedback mode enabled!")
        else:
            print("\n📊 No trained models found. Starting in data collection mode.")
            print("   Perform some reps to collect data, then run ml_trainer.py")

    # Filters
    med_knee  = MedianFilter(k=5)
    ema_knee  = EMA(0.25)
    med_elbow = MedianFilter(k=5)
    ema_elbow = EMA(0.25)

    # FSMs
    squat_fsm = PhaseFSM(enter_bottom=SQUAT_BOTTOM_DEG, leave_bottom=SQUAT_TOP_DEG)
    curl_fsm  = PhaseFSM(enter_bottom=CURL_BOTTOM_DEG,  leave_bottom=CURL_TOP_DEG)

    # Wall-sit hold timer
    hold_start = None
    hold_secs  = 0.0
    wallsit_hold_count = 0  # Track number of holds for CSV logging

    # Stats tracking
    good_reps_count = 0
    total_reps_count = 0
    
    # Store metrics at bottom of rep (for CSV logging at completion)
    metrics_at_bottom = None

    print("\n" + "="*60)
    print("🏋️  AI PHYSIOTHERAPIST HELPER")
    print("="*60)
    print(f"Mode: {mode.upper()}")
    print(f"Exercise: {exercise.upper()}")
    print("\nControls:")
    print("  [1] = Squat")
    print("  [2] = Wall-sit")
    print("  [3] = Curl")
    print("  [M] = Toggle mode (collect/feedback)")
    print("  [R] = Reset rep counter")
    print("  [ESC] = Quit")
    print("="*60 + "\n")

    with mp_pose.Pose(
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Draw landmarks
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            lms = res.pose_landmarks.landmark if res.pose_landmarks else None
            h, w = frame.shape[:2]

            # ==================== SQUAT ====================
            if exercise == "squat":
                m = squat_metrics(lms, w, h, mp_pose)
                if m:
                    knee = ema_knee(med_knee(m["knee_angle"]))
                    reps, phase = squat_fsm.update_squat(knee)
                    
                    # Store metrics when at bottom (best form point)
                    if phase == "bottom":
                        metrics_at_bottom = m
                    
                    # Check if rep just completed
                    if squat_fsm.rep_just_completed:
                        total_reps_count += 1
                        
                        # Check if it was a good rep and log to CSV
                        if metrics_at_bottom or is_good_rep("squat", metrics_at_bottom) or squat_fsm.rep_just_completed: #Fine code
                            csv_logger.log_rep("squat", reps, metrics_at_bottom)
                            good_reps_count += 1
                            put(frame, "✅ GOOD REP LOGGED!", 150, (0, 255, 0), scale=0.8, thick=2)
                        
                        metrics_at_bottom = None  # Reset
                    
                    # Draw angle visualization
                    joints = m["joints_px"]
                    hip = joints.get("hip")
                    knee_pt = joints.get("knee")
                    ankle = joints.get("ankle")
                    shoulder = joints.get("shoulder")
                    
                    if hip and knee_pt and ankle:
                        draw_angle_with_arc(frame, hip, knee_pt, ankle, knee,
                                          color=m["depth_color"], show_angle_text=True)
                    
                    if shoulder and hip:
                        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])),
                               (int(hip[0]), int(hip[1])), m["torso_lean_color"], 2)

                    # Form feedback
                    cues = []
                    if m["depth_zone"] == "poor":
                        cues.append("Go deeper!")
                    elif m["depth_zone"] == "acceptable":
                        cues.append("Deeper for better depth")
                    
                    if m["torso_lean_zone"] == "poor":
                        cues.append("Keep chest up!")
                    elif m["torso_lean_zone"] == "acceptable":
                        cues.append("Chest up more")
                    
                    # Display metrics
                    put(frame, f"[SQUAT - {mode.upper()}] Reps: {reps} | Phase: {phase.upper()}", 30, (0,255,255))
                    put(frame, f"Knee Angle: {int(knee)}° ({m['depth_zone'].upper()})", 60, m["depth_color"])
                    put(frame, f"Torso Lean: {m['torso_lean_deg']:.1f}° ({m['torso_lean_zone'].upper()})", 90, m["torso_lean_color"])
                    
                    if mode == "feedback" and feedback_system and "squat" in feedback_system.loaded_exercises:
                        # Get AI feedback
                        feedback_msg, quality = feedback_system.get_feedback(lms, "squat", reps)
                        put(frame, f"AI: {feedback_msg} (Quality: {quality:.0%})", 120, (255, 200, 0), scale=0.6)
                    elif cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Good form!", 120, (0, 255, 0))
                    
                    # Stats
                    put(frame, f"Good Reps: {good_reps_count}/{total_reps_count}", 180, (0, 255, 255), scale=0.6)

            # ==================== WALL-SIT ====================
            elif exercise == "wallsit":
                m = wallsit_metrics(lms, w, h, mp_pose)
                if m:
                    holding = m["knee_90"] and m["back_vertical"]
                    if holding:
                        if hold_start is None:
                            hold_start = time.time()
                            wallsit_hold_count += 1
                        hold_secs = time.time() - hold_start
                        
                        # Log every second of good hold
                        if hold_secs > 0 and int(hold_secs) % 5 == 0:  # Log every 5 seconds
                            if is_good_rep("wall_sit", m):
                                m_with_duration = m.copy()
                                m_with_duration["hold_duration"] = hold_secs
                                csv_logger.log_rep("wall_sit", wallsit_hold_count, m_with_duration)
                                good_reps_count += 1
                    else:
                        if hold_start is not None:
                            total_reps_count += 1
                            good_reps_count += 1 #badcode
                        hold_start = None
                        hold_secs = 0.0
                    
                    # Draw visualization
                    joints = m["joints_px"]
                    hip = joints.get("hip")
                    knee_pt = joints.get("knee")
                    ankle = joints.get("ankle")
                    shoulder = joints.get("shoulder")
                    
                    if hip and knee_pt and ankle:
                        draw_angle_with_arc(frame, hip, knee_pt, ankle, m["knee_angle"],
                                          color=m["knee_color"], show_angle_text=True)
                    
                    if shoulder and hip:
                        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])),
                               (int(hip[0]), int(hip[1])), m["back_color"], 2)
                    
                    # Form feedback
                    cues = []
                    if m["knee_zone"] == "poor":
                        cues.append("Adjust knee to 90°")
                    elif m["knee_zone"] == "acceptable":
                        cues.append("Closer to 90°")
                    
                    if m["back_zone"] == "poor":
                        cues.append("Back flat on wall!")
                    elif m["back_zone"] == "acceptable":
                        cues.append("Keep back against wall")
                    
                    # Display metrics
                    put(frame, f"[WALL-SIT - {mode.upper()}] Hold: {hold_secs:.1f}s", 30, (0,255,255))
                    put(frame, f"Knee: {int(m['knee_angle'])}° (target: 90°) ({m['knee_zone'].upper()})", 60, m["knee_color"])
                    put(frame, f"Back: ({m['back_zone'].upper()})", 90, m["back_color"])
                    
                    if mode == "feedback" and feedback_system and "wall_sit" in feedback_system.loaded_exercises:
                        feedback_msg, quality = feedback_system.get_feedback(lms, "wall_sit", wallsit_hold_count)
                        put(frame, f"AI: {feedback_msg} (Quality: {quality:.0%})", 120, (255, 200, 0), scale=0.6)
                    elif cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Perfect hold!", 120, (0, 255, 0))
                    
                    put(frame, f"Good Holds: {good_reps_count}", 150, (0, 255, 255), scale=0.6)

            # ==================== CURL ====================
            elif exercise == "curl":
                m = curl_metrics(lms, w, h, mp_pose, side="LEFT")
                if m:
                    elbow = ema_elbow(med_elbow(m["elbow_angle"]))
                    reps, phase = curl_fsm.update_curl(elbow)
                    
                    if phase == "bottom":
                        metrics_at_bottom = m
                    
                    if curl_fsm.rep_just_completed:
                        total_reps_count += 1
                        
                        if metrics_at_bottom or is_good_rep("bicep_curl", metrics_at_bottom) or curl_fsm.rep_just_completed:
                            csv_logger.log_rep("bicep_curl", reps, metrics_at_bottom)
                            good_reps_count += 1
                            put(frame, "✅ GOOD REP LOGGED!", 150, (0, 255, 0), scale=0.8, thick=2)
                        
                        metrics_at_bottom = None
                    
                    # Draw visualization
                    joints = m["joints_px"]
                    shoulder = joints.get("shoulder")
                    elbow_pt = joints.get("elbow")
                    wrist = joints.get("wrist")
                    
                    if shoulder and elbow_pt and wrist:
                        draw_angle_with_arc(frame, shoulder, elbow_pt, wrist, elbow,
                                          color=m["elbow_color"], show_angle_text=True)
                    
                    if shoulder and elbow_pt:
                        upper_arm_color = (0, 255, 0) if m["upper_arm_stable"] else (0, 0, 255)
                        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])),
                               (int(elbow_pt[0]), int(elbow_pt[1])), upper_arm_color, 3)
                    
                    # Form feedback
                    cues = []
                    if m["elbow_zone"] == "poor":
                        cues.append("Squeeze at top!")
                    elif m["elbow_zone"] == "acceptable":
                        cues.append("Full contraction")
                    
                    if not m["upper_arm_stable"]:
                        cues.append("Keep upper arm still")
                    
                    # Display metrics
                    put(frame, f"[CURL - {mode.upper()}] Reps: {reps} | Phase: {phase.upper()}", 30, (0,255,255))
                    put(frame, f"Elbow: {int(elbow)}° ({m['elbow_zone'].upper()})", 60, m["elbow_color"])
                    put(frame, f"Upper Arm: {'Stable' if m['upper_arm_stable'] else 'Swinging'}", 90,
                       (0, 255, 0) if m["upper_arm_stable"] else (0, 0, 255))
                    
                    if mode == "feedback" and feedback_system and "bicep_curl" in feedback_system.loaded_exercises:
                        feedback_msg, quality = feedback_system.get_feedback(lms, "bicep_curl", reps)
                        put(frame, f"AI: {feedback_msg} (Quality: {quality:.0%})", 120, (255, 200, 0), scale=0.6)
                    elif cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Clean rep!", 120, (0, 255, 0))
                    
                    put(frame, f"Good Reps: {good_reps_count}/{total_reps_count}", 180, (0, 255, 255), scale=0.6)

            # Draw mode indicator
            mode_color = (0, 255, 0) if mode == "feedback" else (255, 255, 0)
            cv2.rectangle(frame, (w - 250, 20), (w - 20, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (w - 250, 20), (w - 20, 80), mode_color, 2)
            cv2.putText(frame, f"MODE: {mode.upper()}", (w - 240, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)
            if mode == "collect":
                cv2.putText(frame, "Collecting Data", (w - 240, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "AI Feedback ON", (w - 240, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Controls
            put(frame, "Keys: [1]=Squat [2]=Wall-sit [3]=Curl [M]=Mode [R]=Reset [ESC]=Quit",
               frame.shape[0]-10, (255,255,255), 0.6, 1)
            
            cv2.imshow("AI Physio Helper", frame)
            k = cv2.waitKey(1) & 0xFF
            
            if k == 27:  # ESC
                break
            elif k == ord('1'):
                exercise = "squat"
                hold_start = None
                hold_secs = 0
                squat_fsm = PhaseFSM(enter_bottom=SQUAT_BOTTOM_DEG, leave_bottom=SQUAT_TOP_DEG)
                print(f"\n✅ Switched to SQUAT")
            elif k == ord('2'):
                exercise = "wallsit"
                hold_start = None
                hold_secs = 0
                wallsit_hold_count = 0
                print(f"\n✅ Switched to WALL-SIT")
            elif k == ord('3'):
                exercise = "curl"
                hold_start = None
                hold_secs = 0
                curl_fsm = PhaseFSM(enter_bottom=CURL_BOTTOM_DEG, leave_bottom=CURL_TOP_DEG)
                print(f"\n✅ Switched to CURL")
            elif k == ord('m') or k == ord('M'):
                if FEEDBACK_AVAILABLE and feedback_system and feedback_system.loaded_exercises:
                    mode = "feedback" if mode == "collect" else "collect"
                    print(f"\n✅ Switched to {mode.upper()} mode")
                else:
                    print("\n⚠️ Feedback mode not available. Train models first!")
            elif k == ord('r') or k == ord('R'):
                good_reps_count = 0
                total_reps_count = 0
                squat_fsm = PhaseFSM(enter_bottom=SQUAT_BOTTOM_DEG, leave_bottom=SQUAT_TOP_DEG)
                curl_fsm = PhaseFSM(enter_bottom=CURL_BOTTOM_DEG, leave_bottom=CURL_TOP_DEG)
                wallsit_hold_count = 0
                print("\n✅ Counters reset")

    # Save session feedback if in feedback mode
    if mode == "feedback" and feedback_system:
        feedback_system.save_session_log()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    print(f"Exercise: {exercise.upper()}")
    print(f"Mode: {mode.upper()}")
    print(f"Good Reps Logged: {good_reps_count}")
    print(f"Total Reps: {total_reps_count}")
    print("="*60)
    print("\n✅ Session complete!")
    if mode == "collect" and good_reps_count > 0:
        print(f"   {good_reps_count} good reps saved to CSV")
        print("   Run 'python ml_trainer.py' to train models")
    print()

if __name__ == "__main__":
    main()