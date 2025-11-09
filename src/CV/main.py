# src/CV/main.py
# Webcam app that uses MediaPipe Pose + your exercises metrics.
# Keys: [1]=Squat  [2]=Wall-sit  [3]=Curl  [ESC]=Quit

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import cv2
import numpy as np
import mediapipe as mp
from pose_utils import put, draw_angle_with_arc, EMA, MedianFilter, HoldTimer
from exercises import squat_metrics, wallsit_metrics, curl_metrics, RepCounter

# ---------------- Config ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H, FPS = 1280, 720, 30
MODEL_COMPLEXITY = 2
MIN_DET_CONF = 0.70
MIN_TRK_CONF = 0.70

# Rep thresholds (for knee/elbow angles). Tune to your liking.
SQUAT_BOTTOM_DEG = 95.0
SQUAT_TOP_DEG    = 140.0
CURL_BOTTOM_DEG  = 50.0
CURL_TOP_DEG     = 130.0

# ---------------- Setup ----------------
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_legend(frame):
    legend_y = 30
    legend_x = frame.shape[1] - 210
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20),
                  (frame.shape[1] - 10, legend_y + 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20),
                  (frame.shape[1] - 10, legend_y + 100), (255, 255, 255), 1)
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

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    exercise = "squat"   # "squat" | "wallsit" | "curl"

    # Smoothers for displayed angles
    ema_knee, med_knee   = EMA(0.25), MedianFilter(5)
    ema_elbow, med_elbow = EMA(0.25), MedianFilter(5)

    # Rep counters
    squat_counter = RepCounter(low_thresh=SQUAT_BOTTOM_DEG, high_thresh=SQUAT_TOP_DEG)
    curl_counter  = RepCounter(low_thresh=CURL_BOTTOM_DEG,  high_thresh=CURL_TOP_DEG)

    # Hold timer for wall-sit
    hold_timer = HoldTimer()

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

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Draw landmarks (nice official style)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            # MediaPipe normalized landmarks list
            lm = res.pose_landmarks.landmark if res.pose_landmarks else None
            if lm:
                if exercise == "squat":
                    m = squat_metrics(lm, w, h, mp_pose)
                    if m:
                        # Angle smoothing for display
                        knee = ema_knee(med_knee(m["knee_angle"]))

                        # Rep count on knee angle
                        reps, phase = squat_counter.update(knee)

                        # Draw knee angle arc
                        jp = m["joints_px"]
                        draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                                            angle_deg=knee, color=m["depth_color"], show_text=True)

                        # Draw torso line (shoulder-hip) in back-zone color
                        cv2.line(frame, jp["shoulder"], jp["hip"], m["back_color"], 3)

                        # Optional: draw shin line in hinge color (helps visualize)
                        cv2.line(frame, jp["ankle"], jp["knee"], m["hinge_color"], 2)

                        # Cues
                        cues = []
                        if m["depth_zone"] in ("acceptable", "poor"):
                            cues.append("Go deeper")
                        if m["back_zone"] in ("acceptable", "poor"):
                            cues.append("Chest up")
                        if m["hinge_zone"] in ("acceptable", "poor"):
                            cues.append("Brace core")
                        if m["knee_forward_norm"] is not None and m["knee_toe_zone"] in ("acceptable", "poor"):
                            cues.append("Knees back")

                        # HUD
                        put(frame, f"[SQUAT] Reps: {reps} | Phase: {phase.upper()}", 30, (0, 255, 255))
                        put(frame, f"Knee: {int(knee)}° ({m['depth_zone']})", 60, m["depth_color"])
                        put(frame, f"Torso->Vertical: {int(m['torso_vertical_deg'])}° ({m['back_zone']})", 90, m["back_color"])
                        put(frame, f"Hinge: {int(m['hinge_deg'])}° ({m['hinge_zone']})", 120, m["hinge_color"])
                        if m["knee_forward_norm"] is not None:
                            put(frame, f"Knee over toe: {m['knee_forward_norm']:.3f} ({m['knee_toe_zone']})",
                                150, m["knee_toe_color"])
                        put(frame, " | ".join(cues) if cues else "✓ Good form!", 180,
                            (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

                elif exercise == "wallsit":
                    m = wallsit_metrics(lm, w, h, mp_pose)
                    if m:
                        seconds = hold_timer.update(m["knee_90"] and m["back_vertical"])

                        # Draw knee angle arc
                        jp = m["joints_px"]
                        draw_angle_with_arc(frame, jp["hip"], jp["knee"], jp["ankle"],
                                            angle_deg=m["knee_angle"], color=m["knee_color"], show_text=True)
                        # Torso line
                        cv2.line(frame, jp["shoulder"], jp["hip"], m["back_color"], 3)

                        cues = []
                        if m["knee_zone"] in ("acceptable", "poor"):
                            cues.append("Aim for 90° at knee")
                        if m["back_zone"] in ("acceptable", "poor"):
                            cues.append("Back against wall")

                        put(frame, f"[WALL-SIT] Hold: {seconds:.1f}s", 30, (0, 255, 255))
                        put(frame, f"Knee: {int(m['knee_angle'])}° ({m['knee_zone']})", 60, m["knee_color"])
                        put(frame, f"Torso->Vertical: {int(m['torso_vertical_deg'])}° ({m['back_zone']})",
                            90, m["back_color"])
                        put(frame, " | ".join(cues) if cues else "✓ Solid hold!", 120,
                            (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

                elif exercise == "curl":
                    # Auto-pick a side yourself if you’d like; here we show LEFT by default.
                    m = curl_metrics(lm, w, h, mp_pose, side="LEFT")
                    if m:
                        elbow = ema_elbow(med_elbow(m["elbow_angle"]))
                        reps, phase = curl_counter.update(elbow)

                        jp = m["joints_px"]
                        draw_angle_with_arc(frame, jp["shoulder"], jp["elbow"], jp["wrist"],
                                            angle_deg=elbow, color=m["contraction_color"], show_text=True)

                        # Upper arm line (green if stable)
                        ua_col = (0, 255, 0) if m["upper_arm_stable"] else (0, 0, 255)
                        cv2.line(frame, jp["shoulder"], jp["elbow"], ua_col, 3)

                        cues = []
                        if m["contraction_zone"] in ("acceptable", "poor"):
                            cues.append("Squeeze at top")
                        if not m["upper_arm_stable"]:
                            cues.append("Keep upper arm still")

                        put(frame, f"[CURL] Reps: {reps} | Phase: {phase.upper()}", 30, (0, 255, 255))
                        put(frame, f"Elbow: {int(elbow)}° ({m['contraction_zone']})", 60, m["contraction_color"])
                        put(frame, f"Upper arm->Vertical: {int(m['upper_arm_vertical_deg'])}°",
                            90, ua_col)
                        put(frame, " | ".join(cues) if cues else "✓ Clean rep!",
                            120, (0, 0, 255) if cues else (0, 255, 0), scale=0.65)

            draw_legend(frame)
            put(frame, "Keys: [1]=Squat  [2]=Wall-sit  [3]=Curl   [ESC]=Quit",
                frame.shape[0] - 10, (255, 255, 255), 0.6, 1)

            cv2.imshow("AI Physio (Webcam)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('1'):
                exercise = "squat"
                # reset counters/smoothers if you want a clean start
            elif k == ord('2'):
                exercise = "wallsit"
                hold_timer.update(False)
            elif k == ord('3'):
                exercise = "curl"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
