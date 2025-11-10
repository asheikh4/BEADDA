# src/CV/exercises.py
# Finalized metrics (2D pixel-space, camera-agnostic)
# Enhanced with CSV logging for good reps

from pose_utils import angle_3pt, to_xy, EMA, HoldTimer
import math
import numpy as np
import csv
import os
from datetime import datetime

# ----------------------------- CSV LOGGING -----------------------------

class CSVLogger:
    """
    Logs good exercise reps to CSV files for ML training.
    Each exercise has its own CSV file with specific columns.
    """
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        
        # Define CSV columns for each exercise
        self.csv_headers = {
            "bicep_curl": [
                "timestamp", "rep_number", 
                "left_elbow_angle", "right_elbow_angle",
                "left_shoulder_x", "left_shoulder_y",
                "left_elbow_x", "left_elbow_y",
                "left_wrist_x", "left_wrist_y",
                "right_shoulder_x", "right_shoulder_y",
                "right_elbow_x", "right_elbow_y",
                "right_wrist_x", "right_wrist_y",
                "upper_arm_vertical_deg_left", "upper_arm_vertical_deg_right"
            ],
            "squat": [
                "timestamp", "rep_number",
                "left_hip_angle", "right_hip_angle",
                "left_knee_angle", "right_knee_angle",
                "left_hip_x", "left_hip_y",
                "left_knee_x", "left_knee_y",
                "left_ankle_x", "left_ankle_y",
                "right_hip_x", "right_hip_y",
                "right_knee_x", "right_knee_y",
                "right_ankle_x", "right_ankle_y",
                "left_shoulder_x", "left_shoulder_y",
                "right_shoulder_x", "right_shoulder_y",
                "torso_lean_deg", "trunk_tibia_diff"
            ],
            "wall_sit": [
                "timestamp", "rep_number",
                "left_knee_angle", "right_knee_angle",
                "left_hip_angle", "right_hip_angle",
                "hold_duration",
                "left_hip_x", "left_hip_y",
                "left_knee_x", "left_knee_y",
                "left_ankle_x", "left_ankle_y",
                "right_hip_x", "right_hip_y",
                "right_knee_x", "right_knee_y",
                "right_ankle_x", "right_ankle_y",
                "torso_vertical_deg", "knee_over_toe_norm_left", "knee_over_toe_norm_right"
            ]
        }
        
        # Initialize CSV files with headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Create CSV files with headers if they don't exist."""
        for exercise, headers in self.csv_headers.items():
            filepath = os.path.join(self.data_folder, f"{exercise}_good_reps.csv")
            if not os.path.exists(filepath):
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                print(f"✅ Created CSV file: {filepath}")
    
    def log_rep(self, exercise_name, rep_number, metrics):
        """
        Log a good rep to the appropriate CSV file.
        
        Args:
            exercise_name: "bicep_curl", "squat", or "wall_sit"
            rep_number: Current rep count
            metrics: Dictionary containing all the metrics and joint positions
        """
        filepath = os.path.join(self.data_folder, f"{exercise_name}_good_reps.csv")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare row data based on exercise type
        if exercise_name == "bicep_curl":
            row = self._prepare_curl_row(timestamp, rep_number, metrics)
        elif exercise_name == "squat":
            row = self._prepare_squat_row(timestamp, rep_number, metrics)
        elif exercise_name == "wall_sit":
            row = self._prepare_wallsit_row(timestamp, rep_number, metrics)
        else:
            print(f"⚠️ Unknown exercise: {exercise_name}")
            return
        
        # Append to CSV
        try:
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"✅ Logged {exercise_name} rep #{rep_number} to CSV")
        except Exception as e:
            print(f"❌ Error logging to CSV: {e}")
    
    def _prepare_curl_row(self, timestamp, rep_number, metrics):
        """Prepare row data for bicep curl."""
        joints = metrics.get("joints_px", {})
        
        # Get left side
        left_shoulder = joints.get("left_shoulder", (0, 0))
        left_elbow = joints.get("left_elbow", (0, 0))
        left_wrist = joints.get("left_wrist", (0, 0))
        
        # Get right side
        right_shoulder = joints.get("right_shoulder", (0, 0))
        right_elbow = joints.get("right_elbow", (0, 0))
        right_wrist = joints.get("right_wrist", (0, 0))
        
        return [
            timestamp, rep_number,
            metrics.get("left_elbow_angle", 0),
            metrics.get("right_elbow_angle", 0),
            left_shoulder[0], left_shoulder[1],
            left_elbow[0], left_elbow[1],
            left_wrist[0], left_wrist[1],
            right_shoulder[0], right_shoulder[1],
            right_elbow[0], right_elbow[1],
            right_wrist[0], right_wrist[1],
            metrics.get("upper_arm_vertical_deg_left", 0),
            metrics.get("upper_arm_vertical_deg_right", 0)
        ]
    
    def _prepare_squat_row(self, timestamp, rep_number, metrics):
        """Prepare row data for squat."""
        joints = metrics.get("joints_px", {})
        
        # Get both sides
        left_hip = joints.get("left_hip", (0, 0))
        left_knee = joints.get("left_knee", (0, 0))
        left_ankle = joints.get("left_ankle", (0, 0))
        left_shoulder = joints.get("left_shoulder", (0, 0))
        
        right_hip = joints.get("right_hip", (0, 0))
        right_knee = joints.get("right_knee", (0, 0))
        right_ankle = joints.get("right_ankle", (0, 0))
        right_shoulder = joints.get("right_shoulder", (0, 0))
        
        return [
            timestamp, rep_number,
            metrics.get("left_hip_angle", 0),
            metrics.get("right_hip_angle", 0),
            metrics.get("left_knee_angle", 0),
            metrics.get("right_knee_angle", 0),
            left_hip[0], left_hip[1],
            left_knee[0], left_knee[1],
            left_ankle[0], left_ankle[1],
            right_hip[0], right_hip[1],
            right_knee[0], right_knee[1],
            right_ankle[0], right_ankle[1],
            left_shoulder[0], left_shoulder[1],
            right_shoulder[0], right_shoulder[1],
            metrics.get("torso_lean_deg", 0),
            metrics.get("trunk_tibia_diff", 0)
        ]
    
    def _prepare_wallsit_row(self, timestamp, rep_number, metrics):
        """Prepare row data for wall sit."""
        joints = metrics.get("joints_px", {})
        
        # Get both sides
        left_hip = joints.get("left_hip", (0, 0))
        left_knee = joints.get("left_knee", (0, 0))
        left_ankle = joints.get("left_ankle", (0, 0))
        
        right_hip = joints.get("right_hip", (0, 0))
        right_knee = joints.get("right_knee", (0, 0))
        right_ankle = joints.get("right_ankle", (0, 0))
        
        return [
            timestamp, rep_number,
            metrics.get("left_knee_angle", 0),
            metrics.get("right_knee_angle", 0),
            metrics.get("left_hip_angle", 0),
            metrics.get("right_hip_angle", 0),
            metrics.get("hold_duration", 0),
            left_hip[0], left_hip[1],
            left_knee[0], left_knee[1],
            left_ankle[0], left_ankle[1],
            right_hip[0], right_hip[1],
            right_knee[0], right_knee[1],
            right_ankle[0], right_ankle[1],
            metrics.get("torso_vertical_deg", 0),
            metrics.get("knee_over_toe_norm_left", 0),
            metrics.get("knee_over_toe_norm_right", 0)
        ]


def is_good_rep(exercise_name, metrics):
    """
    Determine if a rep meets quality thresholds for CSV logging.
    Only "perfect", "good", and "acceptable" reps are logged (not "poor").
    
    Args:
        exercise_name: "bicep_curl", "squat", or "wall_sit"
        metrics: Dictionary containing exercise metrics and zones
    
    Returns:
        Boolean indicating if rep should be logged
    """
    if metrics is None:
        return True
    
    if exercise_name == "bicep_curl":
        # Check elbow angle zone and upper arm stability
        elbow_zone = metrics.get("elbow_zone", "poor")
        upper_arm_zone = metrics.get("upper_arm_zone", "poor")
        
        # Both must be at least "acceptable" (not "poor")
        return elbow_zone in ["perfect", "good", "acceptable"] and \
               upper_arm_zone in ["perfect", "good", "acceptable"]
    
    elif exercise_name == "squat":
        # Check depth and back lean zones
        depth_zone = metrics.get("depth_zone", "poor")
        torso_lean_zone = metrics.get("torso_lean_zone", "poor")
        trunk_tibia_zone = metrics.get("trunk_tibia_zone", "poor")
        
        # All major metrics must be at least "acceptable"
        return depth_zone in ["perfect", "good", "acceptable"] and \
               torso_lean_zone in ["perfect", "good", "acceptable"] and \
               trunk_tibia_zone in ["perfect", "good", "acceptable"]
    
    elif exercise_name == "wall_sit":
        # Check knee angle and back alignment
        knee_zone = metrics.get("knee_zone", "poor")
        back_zone = metrics.get("back_zone", "poor")
        
        # Both must be at least "acceptable"
        return knee_zone in ["perfect", "good", "acceptable"] and \
               back_zone in ["perfect", "good", "acceptable"]
    
    return True


# ----------------------------- Tolerances -----------------------------
# (Your existing tolerance constants remain the same)

SQUAT_DEPTH_PERFECT     = 100.0
SQUAT_DEPTH_GOOD        = 110.0
SQUAT_DEPTH_ACCEPTABLE  = 120.0

TORSO_LEAN_PERFECT      = 25.0
TORSO_LEAN_GOOD         = 35.0
TORSO_LEAN_ACCEPTABLE   = 45.0

TRUNK_TIBIA_DIFF_PERFECT    = 10.0
TRUNK_TIBIA_DIFF_GOOD       = 15.0
TRUNK_TIBIA_DIFF_ACCEPTABLE = 20.0

VALGUS_PERFECT_MAX      = 5.0
VALGUS_ACCEPTABLE_MAX   = 10.0

TIBIA_ANGLE_GOOD_MAX        = 35.0
TIBIA_ANGLE_ACCEPTABLE_MAX  = 40.0

LUMBAR_EXCURSION_PERFECT_MAX = 10.0
LUMBAR_EXCURSION_GOOD_MAX    = 15.0
LUMBAR_EXCURSION_POOR_MIN    = 20.0

DEPTH_SYMMETRY_PERFECT      = 5.0
DEPTH_SYMMETRY_ACCEPTABLE   = 10.0

BACK_STRAIGHT_PERFECT       = 5.0
BACK_STRAIGHT_GOOD          = 10.0
BACK_STRAIGHT_ACCEPTABLE    = 15.0

WALL_KNEE_TARGET        = 90.0
WALL_KNEE_TOL_PERFECT   = 5.0
WALL_KNEE_TOL_GOOD      = 8.0
WALL_KNEE_TOL_ACCEPTABLE = 12.0

WALL_BACK_VERT_PERFECT    = 6.0
WALL_BACK_VERT_GOOD       = 10.0
WALL_BACK_VERT_ACCEPTABLE = 14.0

WALL_KNEE_OVERTOE_GOOD_NORM    = 0.05
WALL_KNEE_OVERTOE_OK_NORM      = 0.08
WALL_KNEE_OVERTOE_ACCEPTABLE   = 0.12

WALL_BACK_PERFECT = WALL_BACK_VERT_PERFECT
WALL_BACK_GOOD = WALL_BACK_VERT_GOOD
WALL_BACK_ACCEPTABLE = WALL_BACK_VERT_ACCEPTABLE

CURL_ELBOW_PERFECT      = 45.0
CURL_ELBOW_GOOD         = 50.0
CURL_ELBOW_ACCEPT       = 55.0

CURL_UPPERARM_VERT_PERFECT = 10.0
CURL_UPPERARM_VERT_GOOD    = 20.0
CURL_UPPERARM_VERT_ACCEPT  = 30.0

CURL_CONTRACTION_PERFECT   = CURL_ELBOW_PERFECT
CURL_CONTRACTION_GOOD      = CURL_ELBOW_GOOD
CURL_CONTRACTION_ACCEPTABLE = CURL_ELBOW_ACCEPT
CURL_UPPERARM_STABLE_MAX   = CURL_UPPERARM_VERT_ACCEPT


# ----------------------------- Helpers -----------------------------
# (Your existing helper functions remain the same)

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v) + 1e-8
    return v / n

def _angle_between(u, v):
    u, v = _unit(u), _unit(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def angle_to_vertical_2d(p_from, p_to):
    """
    Angle (degrees) between vector p_from->p_to and the image-space vertical axis.
    0° = perfectly vertical (pointing up or down). Uses pixel coordinates (x right, y down).
    
    Returns the absolute angle deviation from vertical (0-90° range).
    """
    vx = float(p_to[0] - p_from[0])
    vy = float(p_to[1] - p_from[1])
    
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    
    vector_length = math.sqrt(vx * vx + vy * vy)
    if vector_length < 1e-6:
        return 0.0
    
    horizontal_component = abs(vx)
    angle_from_vertical = math.degrees(math.asin(min(1.0, horizontal_component / vector_length)))
    
    return angle_from_vertical

def get_tolerance_zone(value, perfect, good, acceptable, lower_is_better=True):
    """
    Returns: (zone, color_bgr)
    Colors: GREEN, YELLOW, ORANGE, RED as BGR tuples for OpenCV.
    """
    if lower_is_better:
        if value <= perfect:
            return 'perfect', (0, 255, 0)
        elif value <= good:
            return 'good', (0, 255, 255)
        elif value <= acceptable:
            return 'acceptable', (0, 165, 255)
        else:
            return 'poor', (0, 0, 255)
    else:
        if value >= perfect:
            return 'perfect', (0, 255, 0)
        elif value >= good:
            return 'good', (0, 255, 255)
        elif value >= acceptable:
            return 'acceptable', (0, 165, 255)
        else:
            return 'poor', (0, 0, 255)

def get_tolerance_zone_deviation(value, target, tol_perfect, tol_good, tol_accept):
    """
    For metrics that should be close to a target (e.g., 90° knee at wall-sit).
    """
    dev = abs(value - target)
    if dev <= tol_perfect:
        return 'perfect', (0, 255, 0)
    elif dev <= tol_good:
        return 'good', (0, 255, 255)
    elif dev <= tol_accept:
        return 'acceptable', (0, 165, 255)
    else:
        return 'poor', (0, 0, 255)

def calculate_valgus_fppa_simplified(l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle, w, h):
    """
    Calculate approximate FPPA (valgus) using both legs in 2D.
    For side view, true valgus is hard to detect, so we use a simplified metric.
    Returns None to indicate it can't be measured accurately from side view.
    """
    return None

class LumbarExcursionTracker:
    """
    Tracks lumbar excursion (change in back angle) during a rep.
    """
    def __init__(self):
        self.neutral_angle = None
        self.max_excursion = 0.0
        self.current_angle = None
        self.rep_active = False
    
    def start_rep(self, current_torso_angle):
        if not self.rep_active:
            self.neutral_angle = current_torso_angle
            self.max_excursion = 0.0
            self.rep_active = True
    
    def update(self, current_torso_angle, rep_phase):
        self.current_angle = current_torso_angle
        
        if self.rep_active and self.neutral_angle is not None:
            excursion = abs(current_torso_angle - self.neutral_angle)
            self.max_excursion = max(self.max_excursion, excursion)
        
        if rep_phase == "top" and self.rep_active:
            self.rep_active = False
    
    def get_excursion(self):
        return self.max_excursion if self.max_excursion > 0 else 0.0
    
    def reset(self):
        self.neutral_angle = None
        self.max_excursion = 0.0
        self.current_angle = None
        self.rep_active = False


# ----------------------------- Rep Counter -----------------------------
# (Your existing rep counter classes remain the same)

class RepCounter:
    """
    Simple finite-state rep counter using a single monotonically varying angle.
    """
    def __init__(self, low_thresh, high_thresh):
        self.low, self.high = float(low_thresh), float(high_thresh)
        self.state, self.reps = "top", 0
    
    def reset(self):
        self.state, self.reps = "top", 0

    def update(self, angle):
        a = float(angle)
        if self.state == "top" and a < self.high:
            self.state = "down"
        elif self.state == "down":
            if a < self.low:
                self.state = "bottom"
            elif a >= self.high:
                self.state = "top"
        elif self.state == "bottom" and a > self.high:
            self.state = "top"
            self.reps += 1
        return self.reps, self.state

class FormAwareRepCounter:
    """
    Rep counter that only counts reps when form metrics are acceptable.
    """
    def __init__(self, low_thresh, high_thresh):
        self.low, self.high = float(low_thresh), float(high_thresh)
        self.state, self.reps = "top", 0
        self.form_good_at_bottom = False
    
    def reset(self):
        self.state, self.reps = "top", 0
        self.form_good_at_bottom = False
    
    def update(self, angle, zones=None):
        """
        Update rep counter with angle and optional zone information.
        Only counts rep if all zones are acceptable when reaching bottom.
        """
        a = float(angle)
        
        form_acceptable = True
        if zones:
            for zone_name, zone_value in zones.items():
                if zone_value == "poor":
                    form_acceptable = False
                    break
        
        if self.state == "top" and a < self.high:
            self.state = "down"
            self.form_good_at_bottom = form_acceptable
        elif self.state == "down":
            if a < self.low:
                self.state = "bottom"
                self.form_good_at_bottom = form_acceptable
            elif a >= self.high:
                self.state = "top"
                self.form_good_at_bottom = False
        elif self.state == "bottom":
            if form_acceptable:
                pass
            else:
                self.form_good_at_bottom = False
            if a > self.high:
                self.state = "top"
                if self.form_good_at_bottom:
                    self.reps += 1
                self.form_good_at_bottom = False
        
        return self.reps, self.state


# ----------------------------- Metrics -----------------------------
# (Modified to include joint positions for both sides for CSV logging)

def get_landmark_coords(lm, landmark_name, w, h, mp_pose):
    """Helper to get landmark coordinates by name."""
    P = mp_pose.PoseLandmark
    idx = getattr(P, landmark_name).value
    return to_xy(lm, idx, w, h)

def squat_metrics(lm, w, h, mp_pose):
    """
    Squat analysis (2D pixels) with CSV logging support.
    Returns metrics including both sides' joint positions.
    """
    P = mp_pose.PoseLandmark
    LHIP, LKNEE, LANK  = P.LEFT_HIP.value, P.LEFT_KNEE.value, P.LEFT_ANKLE.value
    LSH, LHEEL, LTOE   = P.LEFT_SHOULDER.value, P.LEFT_HEEL.value, P.LEFT_FOOT_INDEX.value
    RHIP, RKNEE, RANK  = P.RIGHT_HIP.value, P.RIGHT_KNEE.value, P.RIGHT_ANKLE.value
    RSH, RHEEL, RTOE   = P.RIGHT_SHOULDER.value, P.RIGHT_HEEL.value, P.RIGHT_FOOT_INDEX.value

    def side_pack(hip_i, knee_i, ank_i, sh_i, heel_i, toe_i):
        hip  = to_xy(lm, hip_i,  w, h)
        knee = to_xy(lm, knee_i, w, h)
        ank  = to_xy(lm, ank_i,  w, h)
        sh   = to_xy(lm, sh_i,   w, h)
        heel = to_xy(lm, heel_i, w, h)
        toe  = to_xy(lm, toe_i,  w, h)
        visible = sum(p[0] > 0 and p[1] > 0 for p in (hip, knee, ank, sh))
        return dict(hip=hip, knee=knee, ank=ank, sh=sh, heel=heel, toe=toe, visible=visible)

    L = side_pack(LHIP, LKNEE, LANK, LSH, LHEEL, LTOE)
    R = side_pack(RHIP, RKNEE, RANK, RSH, RHEEL, RTOE)

    S = L if L["visible"] >= R["visible"] else R
    side_used = "LEFT" if S is L else "RIGHT"

    hip, knee, ank, sh, heel, toe = S["hip"], S["knee"], S["ank"], S["sh"], S["heel"], S["toe"]

    if min(hip + knee + ank + sh) == 0:
        return None

    # Get both sides for CSV logging
    l_hip = to_xy(lm, LHIP, w, h)
    l_knee = to_xy(lm, LKNEE, w, h)
    l_ank = to_xy(lm, LANK, w, h)
    l_sh = to_xy(lm, LSH, w, h)
    r_hip = to_xy(lm, RHIP, w, h)
    r_knee = to_xy(lm, RKNEE, w, h)
    r_ank = to_xy(lm, RANK, w, h)
    r_sh = to_xy(lm, RSH, w, h)

    # Calculate angles for both sides
    knee_ang = angle_3pt(hip, knee, ank)
    l_knee_ang = angle_3pt(l_hip, l_knee, l_ank)
    r_knee_ang = angle_3pt(r_hip, r_knee, r_ank)
    
    # Calculate hip angles for both sides
    l_hip_ang = angle_3pt(l_sh, l_hip, l_knee)
    r_hip_ang = angle_3pt(r_sh, r_hip, r_knee)

    # Torso lean and other metrics
    torso_lean_deg = angle_to_vertical_2d(hip, sh)
    tibia_angle_deg = angle_to_vertical_2d(ank, knee)
    trunk_tibia_diff = abs(torso_lean_deg - tibia_angle_deg)
    
    depth_symmetry_deg = abs(l_knee_ang - r_knee_ang) if (l_knee_ang > 0 and r_knee_ang > 0) else 0.0
    
    # Back straightness (keeping for compatibility)
    back_straightness_deg = BACK_STRAIGHT_ACCEPTABLE
    if (min(l_hip + l_sh) > 0 and min(r_hip + r_sh) > 0):
        left_torso_angle = math.degrees(math.atan2(l_sh[1] - l_hip[1], l_sh[0] - l_hip[0]))
        right_torso_angle = math.degrees(math.atan2(r_sh[1] - r_hip[1], r_sh[0] - r_hip[0]))
        left_torso_angle = left_torso_angle % 360
        right_torso_angle = right_torso_angle % 360
        angle_diff = abs(left_torso_angle - right_torso_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        back_straightness_deg = angle_diff

    # Zones
    depth_zone, depth_color = get_tolerance_zone(
        knee_ang, SQUAT_DEPTH_PERFECT, SQUAT_DEPTH_GOOD, SQUAT_DEPTH_ACCEPTABLE, lower_is_better=True
    )
    
    torso_lean_zone, torso_lean_color = get_tolerance_zone(
        torso_lean_deg, TORSO_LEAN_PERFECT, TORSO_LEAN_GOOD, TORSO_LEAN_ACCEPTABLE, lower_is_better=True
    )
    
    trunk_tibia_zone, trunk_tibia_color = get_tolerance_zone(
        trunk_tibia_diff, TRUNK_TIBIA_DIFF_PERFECT, TRUNK_TIBIA_DIFF_GOOD, 
        TRUNK_TIBIA_DIFF_ACCEPTABLE, lower_is_better=True
    )
    
    tibia_angle_zone, tibia_angle_color = ("perfect", (0, 255, 0))
    if tibia_angle_deg <= TIBIA_ANGLE_GOOD_MAX:
        tibia_angle_zone, tibia_angle_color = ("good", (0, 255, 255))
    elif tibia_angle_deg <= TIBIA_ANGLE_ACCEPTABLE_MAX:
        tibia_angle_zone, tibia_angle_color = ("acceptable", (0, 165, 255))
    else:
        tibia_angle_zone, tibia_angle_color = ("poor", (0, 0, 255))
    
    depth_symmetry_zone, depth_symmetry_color = ("perfect", (0, 255, 0))
    if depth_symmetry_deg <= DEPTH_SYMMETRY_PERFECT:
        depth_symmetry_zone, depth_symmetry_color = ("perfect", (0, 255, 0))
    elif depth_symmetry_deg <= DEPTH_SYMMETRY_ACCEPTABLE:
        depth_symmetry_zone, depth_symmetry_color = ("acceptable", (0, 165, 255))
    else:
        depth_symmetry_zone, depth_symmetry_color = ("poor", (0, 0, 255))
    
    back_zone, back_color = get_tolerance_zone(
        back_straightness_deg, BACK_STRAIGHT_PERFECT, BACK_STRAIGHT_GOOD, 
        BACK_STRAIGHT_ACCEPTABLE, lower_is_better=True
    )

    return {
        "side_used": side_used,
        # Core metrics
        "knee_angle": knee_ang,
        "left_knee_angle": l_knee_ang,
        "right_knee_angle": r_knee_ang,
        "left_hip_angle": l_hip_ang,
        "right_hip_angle": r_hip_ang,
        "torso_lean_deg": torso_lean_deg,
        "tibia_angle_deg": tibia_angle_deg,
        "trunk_tibia_diff": trunk_tibia_diff,
        "depth_symmetry_deg": depth_symmetry_deg,
        "valgus_deg": None,
        "torso_vertical_deg": torso_lean_deg,
        "back_straightness_deg": back_straightness_deg,
        "tibia_vertical_deg": tibia_angle_deg,
        # Zones
        "depth_zone": depth_zone, "depth_color": depth_color,
        "torso_lean_zone": torso_lean_zone, "torso_lean_color": torso_lean_color,
        "trunk_tibia_zone": trunk_tibia_zone, "trunk_tibia_color": trunk_tibia_color,
        "tibia_angle_zone": tibia_angle_zone, "tibia_angle_color": tibia_angle_color,
        "depth_symmetry_zone": depth_symmetry_zone, "depth_symmetry_color": depth_symmetry_color,
        "back_zone": back_zone, "back_color": back_color,
        # Joint positions (for CSV logging)
        "joints_px": {
            "hip": hip, "knee": knee, "ankle": ank, "shoulder": sh,
            "left_hip": l_hip, "left_knee": l_knee, "left_ankle": l_ank, "left_shoulder": l_sh,
            "right_hip": r_hip, "right_knee": r_knee, "right_ankle": r_ank, "right_shoulder": r_sh,
        },
    }


def wallsit_metrics(lm, w, h, mp_pose):
    """
    Wall-sit analysis with CSV logging support.
    """
    P = mp_pose.PoseLandmark
    LHIP, LKNEE, LANK, LSH = P.LEFT_HIP.value, P.LEFT_KNEE.value, P.LEFT_ANKLE.value, P.LEFT_SHOULDER.value
    RHIP, RKNEE, RANK, RSH = P.RIGHT_HIP.value, P.RIGHT_KNEE.value, P.RIGHT_ANKLE.value, P.RIGHT_SHOULDER.value

    def side_pack(hip_i, knee_i, ank_i, sh_i):
        hip  = to_xy(lm, hip_i,  w, h)
        knee = to_xy(lm, knee_i, w, h)
        ank  = to_xy(lm, ank_i,  w, h)
        sh   = to_xy(lm, sh_i,   w, h)
        visible = sum(p[0] > 0 and p[1] > 0 for p in (hip, knee, ank, sh))
        return dict(hip=hip, knee=knee, ank=ank, sh=sh, visible=visible)

    L = side_pack(LHIP, LKNEE, LANK, LSH)
    R = side_pack(RHIP, RKNEE, RANK, RSH)
    S = L if L["visible"] >= R["visible"] else R

    hip, knee, ank, sh = S["hip"], S["knee"], S["ank"], S["sh"]
    if min(hip + knee + ank + sh) == 0:
        return None

    # Get both sides for CSV
    l_hip = to_xy(lm, LHIP, w, h)
    l_knee = to_xy(lm, LKNEE, w, h)
    l_ank = to_xy(lm, LANK, w, h)
    l_sh = to_xy(lm, LSH, w, h)
    r_hip = to_xy(lm, RHIP, w, h)
    r_knee = to_xy(lm, RKNEE, w, h)
    r_ank = to_xy(lm, RANK, w, h)
    r_sh = to_xy(lm, RSH, w, h)

    # Calculate angles
    knee_ang = angle_3pt(hip, knee, ank)
    l_knee_ang = angle_3pt(l_hip, l_knee, l_ank)
    r_knee_ang = angle_3pt(r_hip, r_knee, r_ank)
    l_hip_ang = angle_3pt(l_sh, l_hip, l_knee)
    r_hip_ang = angle_3pt(r_sh, r_hip, r_knee)
    
    torso_vertical_deg = angle_to_vertical_2d(hip, sh)
    
    # Knee over toe for both sides
    knee_over_toe_norm_left = None
    knee_over_toe_norm_right = None
    
    if all(v > 0 for v in l_knee + l_ank) and w > 0:
        knee_forward_pixels = float(l_knee[0] - l_ank[0])
        if knee_forward_pixels > 0:
            knee_over_toe_norm_left = knee_forward_pixels / float(w)
        else:
            knee_over_toe_norm_left = 0.0
    
    if all(v > 0 for v in r_knee + r_ank) and w > 0:
        knee_forward_pixels = float(r_knee[0] - r_ank[0])
        if knee_forward_pixels > 0:
            knee_over_toe_norm_right = knee_forward_pixels / float(w)
        else:
            knee_over_toe_norm_right = 0.0
    
    # Use primary side's knee_over_toe for zone calculation
    knee_over_toe_norm = knee_over_toe_norm_left if S is L else knee_over_toe_norm_right
    
    # Zones
    knee_zone, knee_color = get_tolerance_zone_deviation(
        knee_ang, WALL_KNEE_TARGET, WALL_KNEE_TOL_PERFECT, WALL_KNEE_TOL_GOOD, WALL_KNEE_TOL_ACCEPTABLE
    )
    
    back_zone, back_color = get_tolerance_zone(
        torso_vertical_deg, WALL_BACK_VERT_PERFECT, WALL_BACK_VERT_GOOD, 
        WALL_BACK_VERT_ACCEPTABLE, lower_is_better=True
    )
    
    knee_over_toe_zone, knee_over_toe_color = ("perfect", (0, 255, 0))
    if knee_over_toe_norm is not None:
        if knee_over_toe_norm <= WALL_KNEE_OVERTOE_GOOD_NORM:
            knee_over_toe_zone, knee_over_toe_color = ("perfect", (0, 255, 0))
        elif knee_over_toe_norm <= WALL_KNEE_OVERTOE_OK_NORM:
            knee_over_toe_zone, knee_over_toe_color = ("good", (0, 255, 255))
        elif knee_over_toe_norm <= WALL_KNEE_OVERTOE_ACCEPTABLE:
            knee_over_toe_zone, knee_over_toe_color = ("acceptable", (0, 165, 255))
        else:
            knee_over_toe_zone, knee_over_toe_color = ("poor", (0, 0, 255))

    return {
        # Core metrics
        "knee_angle": knee_ang,
        "left_knee_angle": l_knee_ang,
        "right_knee_angle": r_knee_ang,
        "left_hip_angle": l_hip_ang,
        "right_hip_angle": r_hip_ang,
        "torso_vertical_deg": torso_vertical_deg,
        "knee_over_toe_norm": knee_over_toe_norm,
        "knee_over_toe_norm_left": knee_over_toe_norm_left if knee_over_toe_norm_left is not None else 0,
        "knee_over_toe_norm_right": knee_over_toe_norm_right if knee_over_toe_norm_right is not None else 0,
        # Zones
        "knee_zone": knee_zone,
        "knee_color": knee_color,
        "back_zone": back_zone,
        "back_color": back_color,
        "knee_over_toe_zone": knee_over_toe_zone,
        "knee_over_toe_color": knee_over_toe_color,
        # Legacy
        "knee_90": abs(knee_ang - WALL_KNEE_TARGET) <= WALL_KNEE_TOL_ACCEPTABLE,
        "back_vertical": torso_vertical_deg <= WALL_BACK_VERT_ACCEPTABLE,
        # Joint positions
        "joints_px": {
            "hip": hip, "knee": knee, "ankle": ank, "shoulder": sh,
            "left_hip": l_hip, "left_knee": l_knee, "left_ankle": l_ank,
            "right_hip": r_hip, "right_knee": r_knee, "right_ankle": r_ank,
        },
    }


def curl_metrics(lm, w, h, mp_pose, side="LEFT"):
    """
    Curl analysis with CSV logging support (both arms).
    """
    P = mp_pose.PoseLandmark
    
    # Get both sides
    l_sh = to_xy(lm, P.LEFT_SHOULDER.value, w, h)
    l_el = to_xy(lm, P.LEFT_ELBOW.value, w, h)
    l_wr = to_xy(lm, P.LEFT_WRIST.value, w, h)
    
    r_sh = to_xy(lm, P.RIGHT_SHOULDER.value, w, h)
    r_el = to_xy(lm, P.RIGHT_ELBOW.value, w, h)
    r_wr = to_xy(lm, P.RIGHT_WRIST.value, w, h)
    
    # Use specified side or auto-detect
    S = side.upper()
    if S == "LEFT":
        sh, el, wr = l_sh, l_el, l_wr
    else:
        sh, el, wr = r_sh, r_el, r_wr

    if min(sh + el + wr) == 0:
        return None

    # Calculate angles for both sides
    elbow_angle = angle_3pt(sh, el, wr)
    left_elbow_angle = angle_3pt(l_sh, l_el, l_wr) if min(l_sh + l_el + l_wr) > 0 else 0
    right_elbow_angle = angle_3pt(r_sh, r_el, r_wr) if min(r_sh + r_el + r_wr) > 0 else 0
    
    # Upper arm vertical alignment for both sides
    upper_arm_vert_deg = angle_to_vertical_2d(sh, el)
    upper_arm_vert_deg_left = angle_to_vertical_2d(l_sh, l_el) if min(l_sh + l_el) > 0 else 0
    upper_arm_vert_deg_right = angle_to_vertical_2d(r_sh, r_el) if min(r_sh + r_el) > 0 else 0
    
    # Zones
    elbow_zone, elbow_color = get_tolerance_zone(
        elbow_angle, CURL_ELBOW_PERFECT, CURL_ELBOW_GOOD, CURL_ELBOW_ACCEPT, lower_is_better=True
    )
    
    upper_arm_zone, upper_arm_color = get_tolerance_zone(
        upper_arm_vert_deg, CURL_UPPERARM_VERT_PERFECT, CURL_UPPERARM_VERT_GOOD, 
        CURL_UPPERARM_VERT_ACCEPT, lower_is_better=True
    )
    
    upper_arm_stable = upper_arm_vert_deg < CURL_UPPERARM_VERT_ACCEPT

    return {
        # Core metrics
        "elbow_angle": elbow_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "upper_arm_vertical_deg": upper_arm_vert_deg,
        "upper_arm_vertical_deg_left": upper_arm_vert_deg_left,
        "upper_arm_vertical_deg_right": upper_arm_vert_deg_right,
        # Zones
        "elbow_zone": elbow_zone,
        "elbow_color": elbow_color,
        "upper_arm_zone": upper_arm_zone,
        "upper_arm_color": upper_arm_color,
        # Legacy
        "contraction_zone": elbow_zone,
        "contraction_color": elbow_color,
        "upper_arm_stable": upper_arm_stable,
        # Joint positions
        "joints_px": {
            "shoulder": sh, "elbow": el, "wrist": wr,
            "left_shoulder": l_sh, "left_elbow": l_el, "left_wrist": l_wr,
            "right_shoulder": r_sh, "right_elbow": r_el, "right_wrist": r_wr,
        },
    }