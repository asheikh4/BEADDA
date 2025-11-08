# src/cv/main.py
import cv2
import time
import collections
import numpy as np
import mediapipe as mp
import math

# ===================== Config =====================
MODEL_COMPLEXITY = 2     # 0 (fast), 1 (default high-accuracy), 2 (highest)
MIN_DET_CONF = 0.70
MIN_TRK_CONF = 0.70
VIS_THR = 0.50           # landmark visibility threshold
CAM_INDEX = 0            # laptop webcam

# Rep detection thresholds (with hysteresis)
SQUAT_TOP_DEG   = 140.0  # leave bottom when knee angle > this
SQUAT_BOTTOM_DEG= 95.0   # enter bottom when knee angle < this

CURL_TOP_DEG    = 130.0  # leave bottom when elbow angle > this
CURL_BOTTOM_DEG = 50.0   # enter bottom when elbow angle < this

MIN_PHASE_FRAMES = 3     # require N frames before switching phase (de-bounce)

# ===================== Tolerance Zones =====================
# Format: (perfect, good, acceptable) - values are thresholds
# Colors: GREEN (perfect), YELLOW (good), ORANGE (acceptable), RED (poor)

# Squat tolerances
SQUAT_DEPTH_PERFECT = 100.0    # knee angle < this is perfect depth
SQUAT_DEPTH_GOOD = 110.0       # knee angle < this is good
SQUAT_DEPTH_ACCEPTABLE = 120.0 # knee angle < this is acceptable

SQUAT_BACK_LEAN_PERFECT = 0.04    # back lean < this is perfect
SQUAT_BACK_LEAN_GOOD = 0.06       # back lean < this is good
SQUAT_BACK_LEAN_ACCEPTABLE = 0.08 # back lean < this is acceptable

# Wall-sit tolerances
WALLSIT_KNEE_TOLERANCE_PERFECT = 5.0   # within 5° of 90° is perfect
WALLSIT_KNEE_TOLERANCE_GOOD = 8.0      # within 8° of 90° is good
WALLSIT_KNEE_TOLERANCE_ACCEPTABLE = 12.0 # within 12° of 90° is acceptable

WALLSIT_BACK_TOLERANCE_PERFECT = 0.02  # back alignment < this is perfect
WALLSIT_BACK_TOLERANCE_GOOD = 0.03     # back alignment < this is good
WALLSIT_BACK_TOLERANCE_ACCEPTABLE = 0.04 # back alignment < this is acceptable

# Curl tolerances
CURL_CONTRACTION_PERFECT = 40.0   # elbow angle < this is perfect contraction
CURL_CONTRACTION_GOOD = 45.0      # elbow angle < this is good
CURL_CONTRACTION_ACCEPTABLE = 50.0 # elbow angle < this is acceptable

# ===================== Utils ======================
mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def put(img, txt, y, col=(255,255,255), scale=0.7, thick=2):
    cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def get_tolerance_zone(value, perfect, good, acceptable, lower_is_better=True):
    """
    Returns: (zone, color) where zone is 'perfect', 'good', 'acceptable', or 'poor'
    Colors: GREEN, YELLOW, ORANGE, RED
    """
    if lower_is_better:
        if value <= perfect:
            return 'perfect', (0, 255, 0)  # GREEN
        elif value <= good:
            return 'good', (0, 255, 255)  # YELLOW
        elif value <= acceptable:
            return 'acceptable', (0, 165, 255)  # ORANGE
        else:
            return 'poor', (0, 0, 255)  # RED
    else:
        # For values where higher is better or we check deviation from target
        if value >= perfect:
            return 'perfect', (0, 255, 0)  # GREEN
        elif value >= good:
            return 'good', (0, 255, 255)  # YELLOW
        elif value >= acceptable:
            return 'acceptable', (0, 165, 255)  # ORANGE
        else:
            return 'poor', (0, 0, 255)  # RED

def get_tolerance_zone_deviation(value, target, perfect_tol, good_tol, acceptable_tol):
    """
    For metrics where we want to be close to a target value.
    Returns: (zone, color) based on deviation from target.
    """
    deviation = abs(value - target)
    if deviation <= perfect_tol:
        return 'perfect', (0, 255, 0)  # GREEN
    elif deviation <= good_tol:
        return 'good', (0, 255, 255)  # YELLOW
    elif deviation <= acceptable_tol:
        return 'acceptable', (0, 165, 255)  # ORANGE
    else:
        return 'poor', (0, 0, 255)  # RED

def draw_angle_arc(img, center, radius, start_angle, end_angle, color, thickness=2):
    """Draw an arc representing an angle."""
    axes = (radius, radius)
    angle = int(start_angle)
    end_angle_deg = int(end_angle)
    cv2.ellipse(img, center, axes, 0, angle, end_angle_deg, color, thickness)

def draw_angle_with_arc(img, pt1, pt2, pt3, angle, color=(0, 255, 0), show_angle_text=True):
    """
    Draw angle visualization: lines connecting points and arc showing the angle.
    pt2 is the vertex (middle point).
    Expects normalized coordinates (0-1 range) from MediaPipe.
    """
    if pt1 is None or pt2 is None or pt3 is None:
        return None, None, None
    
    h, w = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    try:
        pt1_px = (int(pt1[0] * w), int(pt1[1] * h))
        pt2_px = (int(pt2[0] * w), int(pt2[1] * h))
        pt3_px = (int(pt3[0] * w), int(pt3[1] * h))
    except (TypeError, IndexError):
        return None, None, None
    
    # Draw lines (thicker for visibility)
    cv2.line(img, pt1_px, pt2_px, color, 3)
    cv2.line(img, pt2_px, pt3_px, color, 3)
    
    # Calculate vectors for arc
    vec1 = np.array(pt1_px, dtype=float) - np.array(pt2_px, dtype=float)
    vec2 = np.array(pt3_px, dtype=float) - np.array(pt2_px, dtype=float)
    
    # Skip if vectors are too short
    if np.linalg.norm(vec1) < 5 or np.linalg.norm(vec2) < 5:
        return pt1_px, pt2_px, pt3_px
    
    # Calculate angles for arc
    angle1 = math.degrees(math.atan2(vec1[1], vec1[0]))
    angle2 = math.degrees(math.atan2(vec2[1], vec2[0]))
    
    # Normalize angles to 0-360 range
    if angle1 < 0:
        angle1 += 360
    if angle2 < 0:
        angle2 += 360
    
    # Draw arc
    radius = max(15, min(40, np.linalg.norm(vec1) * 0.25, np.linalg.norm(vec2) * 0.25))
    cv2.ellipse(img, pt2_px, (int(radius), int(radius)), 0, 
               int(angle1), int(angle2), color, 2)
    
    # Draw angle text near vertex (offset to avoid overlap)
    if show_angle_text:
        # Place text away from the vertex
        offset_x = int(radius * 1.5 * math.cos(math.radians((angle1 + angle2) / 2)))
        offset_y = int(radius * 1.5 * math.sin(math.radians((angle1 + angle2) / 2)))
        text_pos = (pt2_px[0] + offset_x, pt2_px[1] + offset_y)
        cv2.putText(img, f"{int(angle)}°", text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    return pt1_px, pt2_px, pt3_px

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

    def update_squat(self, knee_angle):
        nxt = self.state
        if self.state == "top" and knee_angle < self.enter_bottom:
            nxt = "down"
        elif self.state == "down" and knee_angle < self.enter_bottom:
            nxt = "bottom"
        elif self.state == "bottom" and knee_angle > self.leave_bottom:
            nxt = "top"

        if nxt == self.state:
            self.frames_in_state += 1
        else:
            # change state only if held long enough
            if self.frames_in_state >= MIN_PHASE_FRAMES:
                if self.state == "bottom" and nxt == "top":
                    self.reps += 1
                self.state = nxt
                self.frames_in_state = 1
            else:
                self.frames_in_state += 1
        return self.reps, self.state

    def update_curl(self, elbow_angle):
        nxt = self.state
        if self.state == "top" and elbow_angle < self.enter_bottom:
            nxt = "down"
        elif self.state == "down" and elbow_angle < self.enter_bottom:
            nxt = "bottom"
        elif self.state == "bottom" and elbow_angle > self.leave_bottom:
            nxt = "top"

        if nxt == self.state:
            self.frames_in_state += 1
        else:
            if self.frames_in_state >= MIN_PHASE_FRAMES:
                if self.state == "bottom" and nxt == "top":
                    self.reps += 1
                self.state = nxt
                self.frames_in_state = 1
            else:
                self.frames_in_state += 1
        return self.reps, self.state

def angle3d(a, b, c):
    """Angle ABC in degrees using 3D vectors."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def get_landmark_sets(results):
    """Return (lms_world, lms_2d) or (None, None) if not available."""
    l2d = results.pose_landmarks.landmark if results.pose_landmarks else None
    lw  = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
    return lw, l2d

def get_point(idx, lms_world, lms_2d):
    """Get (x,y,z) from world if available; also return visibility from 2D."""
    if lms_world is not None and lms_2d is not None:
        w = lms_world[idx]; v = lms_2d[idx].visibility
        return (w.x, w.y, w.z), float(v), (lms_2d[idx].x, lms_2d[idx].y)  # world, visibility, 2d
    if lms_2d is not None:
        p = lms_2d[idx]
        return (p.x, p.y, 0.0), float(p.visibility), (p.x, p.y)  # 2d only
    return None, 0.0, None

def get_point_2d_only(lms_2d, idx, img_width, img_height):
    """Get 2D pixel coordinates directly."""
    if lms_2d is None:
        return None
    p = lms_2d[idx]
    return (int(p.x * img_width), int(p.y * img_height)), float(p.visibility)

# ================== Metrics (World-Space, Side-View Optimized) ==================
def squat_metrics_world(lms_world, lms_2d, img_width=None, img_height=None):
    """
    Enhanced squat metrics optimized for side-view tracking.
    Tracks all joints specifically and provides detailed form analysis.
    """
    P = mp_pose.PoseLandmark
    
    # Get all relevant joints with visibility checks
    (l_hip, v_l_hip, l_hip_2d)   = get_point(P.LEFT_HIP.value,   lms_world, lms_2d)
    (l_knee, v_l_knee, l_knee_2d)  = get_point(P.LEFT_KNEE.value,  lms_world, lms_2d)
    (l_ank, v_l_ank, l_ank_2d)   = get_point(P.LEFT_ANKLE.value, lms_world, lms_2d)
    (l_sh,  v_l_sh, l_sh_2d)   = get_point(P.LEFT_SHOULDER.value, lms_world, lms_2d)
    
    # Also get right side for comparison and better tracking
    (r_hip, v_r_hip, r_hip_2d)   = get_point(P.RIGHT_HIP.value,   lms_world, lms_2d)
    (r_knee, v_r_knee, r_knee_2d)  = get_point(P.RIGHT_KNEE.value,  lms_world, lms_2d)
    (r_ank, v_r_ank, r_ank_2d)   = get_point(P.RIGHT_ANKLE.value, lms_world, lms_2d)
    (r_sh,  v_r_sh, r_sh_2d)   = get_point(P.RIGHT_SHOULDER.value, lms_world, lms_2d)
    
    # Use the side with better visibility (for side view, usually one side is better)
    use_left = (v_l_hip + v_l_knee + v_l_ank + v_l_sh) >= (v_r_hip + v_r_knee + v_r_ank + v_r_sh)
    
    if use_left:
        hip, knee, ank, sh = l_hip, l_knee, l_ank, l_sh
        hip_2d, knee_2d, ank_2d, sh_2d = l_hip_2d, l_knee_2d, l_ank_2d, l_sh_2d
        v_hip, v_knee, v_ank, v_sh = v_l_hip, v_l_knee, v_l_ank, v_l_sh
        side_used = "LEFT"
    else:
        hip, knee, ank, sh = r_hip, r_knee, r_ank, r_sh
        hip_2d, knee_2d, ank_2d, sh_2d = r_hip_2d, r_knee_2d, r_ank_2d, r_sh_2d
        v_hip, v_knee, v_ank, v_sh = v_r_hip, v_r_knee, v_r_ank, v_r_sh
        side_used = "RIGHT"
    
    # Check visibility thresholds
    if min(v_hip, v_knee, v_ank, v_sh) < VIS_THR or None in (hip, knee, ank, sh):
        return None
    
    # Calculate knee angle (hip-knee-ankle)
    knee_ang = angle3d(hip, knee, ank)
    
    # Back lean: For side view, calculate torso angle from vertical
    # In MediaPipe world coords: X=left/right, Y=up/down, Z=forward/back
    # For side view, we analyze the sagittal plane (Y-Z plane)
    back_vec = np.array(sh) - np.array(hip)     # 3D vector from hip to shoulder
    
    # Project to sagittal plane (Y-Z plane) - ignore X (left/right) for side view
    sagittal_back_vec = np.array([0, back_vec[1], back_vec[2]])
    
    # Calculate angle from vertical (vertical is negative Y in MediaPipe)
    vertical_vec = np.array([0, -1, 0])
    
    # Normalize vectors
    sagittal_back_norm = sagittal_back_vec / (np.linalg.norm(sagittal_back_vec) + 1e-8)
    
    # Calculate angle between torso and vertical
    cos_angle = np.clip(np.dot(sagittal_back_norm, vertical_vec), -1.0, 1.0)
    back_lean_angle = np.degrees(np.arccos(cos_angle))
    
    # Back lean metric: deviation from 0° (perfect vertical)
    # Also use Z component as a simpler metric (forward lean = positive Z)
    back_lean_z = abs(back_vec[2]) * 15  # Scale for better sensitivity (tuned for side view)
    
    # Combined metric: use both angle and Z deviation
    # For side view, Z deviation is a good indicator of forward/backward lean
    back_lean_combined = back_lean_z + (back_lean_angle * 0.1)
    
    # Knee valgus: For side view, check if knees collapse inward
    # Use difference between knee and ankle positions in X-axis (left/right)
    if lms_2d and img_width and img_height:
        # Use 2D coordinates for valgus detection (easier in side view)
        knee_valgus = abs(l_knee_2d[0] - l_ank_2d[0]) if use_left else abs(r_knee_2d[0] - r_ank_2d[0])
        knee_valgus_normalized = knee_valgus / img_width if img_width > 0 else 0
        # In side view, valgus is less relevant, but we can detect if knees are too far from ankles
        knee_valgus_detected = knee_valgus_normalized > 0.05  # 5% of image width
    else:
        # Fallback to world coords
        knee_valgus_detected = abs(knee[0] - ank[0]) > 0.03
    
    # Depth check with tolerance zones
    depth_zone, depth_color = get_tolerance_zone(
        knee_ang, 
        SQUAT_DEPTH_PERFECT, 
        SQUAT_DEPTH_GOOD, 
        SQUAT_DEPTH_ACCEPTABLE, 
        lower_is_better=True
    )
    
    # Back lean tolerance (using combined metric)
    back_lean_zone, back_lean_color = get_tolerance_zone(
        back_lean_combined,
        SQUAT_BACK_LEAN_PERFECT,
        SQUAT_BACK_LEAN_GOOD,
        SQUAT_BACK_LEAN_ACCEPTABLE,
        lower_is_better=True
    )
    
    return {
        "knee_angle": knee_ang,
        "back_lean": back_lean_combined,
        "back_lean_angle": back_lean_angle,
        "back_lean_z": back_lean_z,
        "knee_valgus": knee_valgus_detected,
        "depth_zone": depth_zone,
        "depth_color": depth_color,
        "back_lean_zone": back_lean_zone,
        "back_lean_color": back_lean_color,
        "joints": {
            "hip": (hip, hip_2d, v_hip),
            "knee": (knee, knee_2d, v_knee),
            "ankle": (ank, ank_2d, v_ank),
            "shoulder": (sh, sh_2d, v_sh)
        },
        "side_used": side_used
    }

def wallsit_metrics_world(lms_world, lms_2d, img_width=None, img_height=None):
    """
    Enhanced wall-sit metrics optimized for side-view tracking.
    """
    P = mp_pose.PoseLandmark
    
    # Get joints with visibility
    (l_hip, v_l_hip, l_hip_2d)   = get_point(P.LEFT_HIP.value,   lms_world, lms_2d)
    (l_knee, v_l_knee, l_knee_2d)  = get_point(P.LEFT_KNEE.value,  lms_world, lms_2d)
    (l_ank, v_l_ank, l_ank_2d)   = get_point(P.LEFT_ANKLE.value, lms_world, lms_2d)
    (l_sh,  v_l_sh, l_sh_2d)   = get_point(P.LEFT_SHOULDER.value, lms_world, lms_2d)
    
    (r_hip, v_r_hip, r_hip_2d)   = get_point(P.RIGHT_HIP.value,   lms_world, lms_2d)
    (r_knee, v_r_knee, r_knee_2d)  = get_point(P.RIGHT_KNEE.value,  lms_world, lms_2d)
    (r_ank, v_r_ank, r_ank_2d)   = get_point(P.RIGHT_ANKLE.value, lms_world, lms_2d)
    (r_sh,  v_r_sh, r_sh_2d)   = get_point(P.RIGHT_SHOULDER.value, lms_world, lms_2d)
    
    # Use side with better visibility
    use_left = (v_l_hip + v_l_knee + v_l_ank + v_l_sh) >= (v_r_hip + v_r_knee + v_r_ank + v_r_sh)
    
    if use_left:
        hip, knee, ank, sh = l_hip, l_knee, l_ank, l_sh
        hip_2d, knee_2d, ank_2d, sh_2d = l_hip_2d, l_knee_2d, l_ank_2d, l_sh_2d
        v_hip, v_knee, v_ank, v_sh = v_l_hip, v_l_knee, v_l_ank, v_l_sh
        side_used = "LEFT"
    else:
        hip, knee, ank, sh = r_hip, r_knee, r_ank, r_sh
        hip_2d, knee_2d, ank_2d, sh_2d = r_hip_2d, r_knee_2d, r_ank_2d, r_sh_2d
        v_hip, v_knee, v_ank, v_sh = v_r_hip, v_r_knee, v_r_ank, v_r_sh
        side_used = "RIGHT"
    
    if min(v_hip, v_knee, v_ank, v_sh) < VIS_THR or None in (hip, knee, ank, sh):
        return None
    
    knee_ang = angle3d(hip, knee, ank)
    
    # For side view, check back alignment using Z-axis (depth)
    back_vec = np.array(sh) - np.array(hip)
    # In side view, back should be against wall (minimal Z deviation)
    back_alignment_z = abs(back_vec[2]) * 10  # Scale for sensitivity
    
    # Knee angle tolerance (should be ~90°)
    knee_zone, knee_color = get_tolerance_zone_deviation(
        knee_ang, 
        90.0,
        WALLSIT_KNEE_TOLERANCE_PERFECT,
        WALLSIT_KNEE_TOLERANCE_GOOD,
        WALLSIT_KNEE_TOLERANCE_ACCEPTABLE
    )
    
    # Back alignment tolerance
    back_zone, back_color = get_tolerance_zone(
        back_alignment_z,
        WALLSIT_BACK_TOLERANCE_PERFECT,
        WALLSIT_BACK_TOLERANCE_GOOD,
        WALLSIT_BACK_TOLERANCE_ACCEPTABLE,
        lower_is_better=True
    )
    
    return {
        "knee_angle": knee_ang,
        "knee_90": abs(knee_ang - 90.0) <= WALLSIT_KNEE_TOLERANCE_ACCEPTABLE,
        "back_vertical": back_alignment_z < WALLSIT_BACK_TOLERANCE_ACCEPTABLE,
        "knee_zone": knee_zone,
        "knee_color": knee_color,
        "back_zone": back_zone,
        "back_color": back_color,
        "joints": {
            "hip": (hip, hip_2d, v_hip),
            "knee": (knee, knee_2d, v_knee),
            "ankle": (ank, ank_2d, v_ank),
            "shoulder": (sh, sh_2d, v_sh)
        },
        "side_used": side_used
    }

def curl_metrics_world(lms_world, lms_2d, side="LEFT", img_width=None, img_height=None):
    """
    Enhanced bicep curl metrics optimized for side-view tracking.
    Automatically selects best side if side="AUTO".
    """
    P = mp_pose.PoseLandmark
    
    # Auto-detect best side if requested
    if side.upper() == "AUTO":
        (l_sh, v_l_sh, l_sh_2d) = get_point(P.LEFT_SHOULDER.value, lms_world, lms_2d)
        (l_el, v_l_el, l_el_2d) = get_point(P.LEFT_ELBOW.value, lms_world, lms_2d)
        (l_wr, v_l_wr, l_wr_2d) = get_point(P.LEFT_WRIST.value, lms_world, lms_2d)
        
        (r_sh, v_r_sh, r_sh_2d) = get_point(P.RIGHT_SHOULDER.value, lms_world, lms_2d)
        (r_el, v_r_el, r_el_2d) = get_point(P.RIGHT_ELBOW.value, lms_world, lms_2d)
        (r_wr, v_r_wr, r_wr_2d) = get_point(P.RIGHT_WRIST.value, lms_world, lms_2d)
        
        # Choose side with better visibility
        left_score = v_l_sh + v_l_el + v_l_wr
        right_score = v_r_sh + v_r_el + v_r_wr
        side = "LEFT" if left_score >= right_score else "RIGHT"
    
    S = side.upper()
    SHO = getattr(P, f"{S}_SHOULDER").value
    ELB = getattr(P, f"{S}_ELBOW").value
    WRI = getattr(P, f"{S}_WRIST").value
    
    (sh, vs_sh, sh_2d) = get_point(SHO, lms_world, lms_2d)
    (el, vs_el, el_2d) = get_point(ELB, lms_world, lms_2d)
    (wr, vs_wr, wr_2d) = get_point(WRI, lms_world, lms_2d)
    
    if min(vs_sh, vs_el, vs_wr) < VIS_THR or None in (sh, el, wr):
        return None
    
    elbow_ang = angle3d(sh, el, wr)
    
    # Contraction tolerance (lower angle = better contraction)
    contraction_zone, contraction_color = get_tolerance_zone(
        elbow_ang,
        CURL_CONTRACTION_PERFECT,
        CURL_CONTRACTION_GOOD,
        CURL_CONTRACTION_ACCEPTABLE,
        lower_is_better=True
    )
    
    # Check if upper arm is stable (shoulder-elbow should be relatively vertical)
    # This helps detect swinging/cheating
    upper_arm_vec = np.array(el) - np.array(sh)
    vertical_vec = np.array([0, -1, 0])
    upper_arm_normalized = upper_arm_vec / (np.linalg.norm(upper_arm_vec) + 1e-8)
    upper_arm_angle = np.degrees(np.arccos(np.clip(np.dot(upper_arm_normalized, vertical_vec), -1.0, 1.0)))
    upper_arm_stable = upper_arm_angle < 30.0  # Within 30° of vertical is good
    
    return {
        "elbow_angle": elbow_ang,
        "full_top": elbow_ang < CURL_CONTRACTION_ACCEPTABLE,
        "contraction_zone": contraction_zone,
        "contraction_color": contraction_color,
        "upper_arm_stable": upper_arm_stable,
        "upper_arm_angle": upper_arm_angle,
        "joints": {
            "shoulder": (sh, sh_2d, vs_sh),
            "elbow": (el, el_2d, vs_el),
            "wrist": (wr, wr_2d, vs_wr)
        },
        "side_used": side
    }

# ===================== Main App =====================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    exercise = "squat"  # "squat" | "wallsit" | "curl"

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

            # Draw landmarks w/ official style (clearer visuals)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            lms_world, lms_2d = get_landmark_sets(res)
            h, w = frame.shape[:2]

            if exercise == "squat":
                m = squat_metrics_world(lms_world, lms_2d, img_width=w, img_height=h)
                if m:
                    # smooth angle
                    knee = ema_knee(med_knee(m["knee_angle"]))
                    reps, phase = squat_fsm.update_squat(knee)

                    # Draw angle visualization
                    joints = m["joints"]
                    hip_world, hip_2d, v_hip = joints["hip"]
                    knee_world, knee_2d, v_knee = joints["knee"]
                    ankle_world, ankle_2d, v_ankle = joints["ankle"]
                    shoulder_world, shoulder_2d, v_shoulder = joints["shoulder"]
                    
                    # Draw knee angle
                    if hip_2d is not None and knee_2d is not None and ankle_2d is not None:
                        draw_angle_with_arc(
                            frame, 
                            hip_2d, 
                            knee_2d, 
                            ankle_2d, 
                            knee,
                            color=m["depth_color"],
                            show_angle_text=True
                        )
                    
                    # Draw back alignment line (shoulder to hip)
                    if shoulder_2d is not None and hip_2d is not None:
                        cv2.line(frame, 
                                (int(shoulder_2d[0] * w), int(shoulder_2d[1] * h)),
                                (int(hip_2d[0] * w), int(hip_2d[1] * h)),
                                m["back_lean_color"], 2)

                    # Form feedback with tolerance zones
                    cues = []
                    if m["depth_zone"] == "poor":
                        cues.append("Go deeper!")
                    elif m["depth_zone"] == "acceptable":
                        cues.append("Deeper for better depth")
                    
                    if m["knee_valgus"]:
                        cues.append("Push knees out")
                    
                    if m["back_lean_zone"] == "poor":
                        cues.append("Keep chest up!")
                    elif m["back_lean_zone"] == "acceptable":
                        cues.append("Chest up more")
                    
                    # Display metrics with color coding
                    put(frame, f"[SQUAT] Reps: {reps} | Phase: {phase.upper()}", 30, (0,255,255))
                    put(frame, f"Knee Angle: {int(knee)}° ({m['depth_zone'].upper()})", 60, m["depth_color"])
                    put(frame, f"Back Lean: {m['back_lean']:.2f} ({m['back_lean_zone'].upper()})", 90, m["back_lean_color"])
                    
                    if cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Good form!", 120, (0, 255, 0))

            elif exercise == "wallsit":
                m = wallsit_metrics_world(lms_world, lms_2d, img_width=w, img_height=h)
                if m:
                    holding = m["knee_90"] and m["back_vertical"]
                    if holding:
                        if hold_start is None: hold_start = time.time()
                        hold_secs = time.time() - hold_start
                    else:
                        hold_start = None
                        hold_secs = 0.0
                    
                    # Draw angle visualization
                    joints = m["joints"]
                    hip_world, hip_2d, v_hip = joints["hip"]
                    knee_world, knee_2d, v_knee = joints["knee"]
                    ankle_world, ankle_2d, v_ankle = joints["ankle"]
                    shoulder_world, shoulder_2d, v_shoulder = joints["shoulder"]
                    
                    # Draw knee angle
                    if hip_2d is not None and knee_2d is not None and ankle_2d is not None:
                        draw_angle_with_arc(
                            frame,
                            hip_2d,
                            knee_2d,
                            ankle_2d,
                            m["knee_angle"],
                            color=m["knee_color"],
                            show_angle_text=True
                        )
                    
                    # Draw back alignment
                    if shoulder_2d is not None and hip_2d is not None:
                        cv2.line(frame,
                                (int(shoulder_2d[0] * w), int(shoulder_2d[1] * h)),
                                (int(hip_2d[0] * w), int(hip_2d[1] * h)),
                                m["back_color"], 2)
                    
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
                    put(frame, f"[WALL-SIT] Hold: {hold_secs:.1f}s (target: 10s)", 30, (0,255,255))
                    put(frame, f"Knee Angle: {int(m['knee_angle'])}° (target: 90°) ({m['knee_zone'].upper()})", 60, m["knee_color"])
                    put(frame, f"Back Alignment: ({m['back_zone'].upper()})", 90, m["back_color"])
                    
                    if cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Perfect hold!", 120, (0, 255, 0))

            elif exercise == "curl":
                m = curl_metrics_world(lms_world, lms_2d, side="AUTO", img_width=w, img_height=h)
                if m:
                    elbow = ema_elbow(med_elbow(m["elbow_angle"]))
                    reps, phase = curl_fsm.update_curl(elbow)
                    
                    # Draw angle visualization
                    joints = m["joints"]
                    shoulder_world, shoulder_2d, v_shoulder = joints["shoulder"]
                    elbow_world, elbow_2d, v_elbow = joints["elbow"]
                    wrist_world, wrist_2d, v_wrist = joints["wrist"]
                    
                    # Draw elbow angle
                    if shoulder_2d is not None and elbow_2d is not None and wrist_2d is not None:
                        draw_angle_with_arc(
                            frame,
                            shoulder_2d,
                            elbow_2d,
                            wrist_2d,
                            elbow,
                            color=m["contraction_color"],
                            show_angle_text=True
                        )
                    
                    # Draw upper arm stability indicator
                    if shoulder_2d is not None and elbow_2d is not None:
                        upper_arm_color = (0, 255, 0) if m["upper_arm_stable"] else (0, 0, 255)
                        cv2.line(frame,
                                (int(shoulder_2d[0] * w), int(shoulder_2d[1] * h)),
                                (int(elbow_2d[0] * w), int(elbow_2d[1] * h)),
                                upper_arm_color, 3)
                    
                    cues = []
                    if m["contraction_zone"] == "poor":
                        cues.append("Squeeze at top!")
                    elif m["contraction_zone"] == "acceptable":
                        cues.append("Full contraction")
                    
                    if not m["upper_arm_stable"]:
                        cues.append("Keep upper arm still")
                    
                    # Display metrics
                    put(frame, f"[CURL] Reps: {reps} | Phase: {phase.upper()} | Side: {m['side_used']}", 30, (0,255,255))
                    put(frame, f"Elbow Angle: {int(elbow)}° ({m['contraction_zone'].upper()})", 60, m["contraction_color"])
                    put(frame, f"Upper Arm: {'Stable' if m['upper_arm_stable'] else 'Swinging'} ({int(m['upper_arm_angle'])}°)", 90, (0, 255, 0) if m["upper_arm_stable"] else (0, 0, 255))
                    
                    if cues:
                        put(frame, " | ".join(cues), 120, (0, 0, 255), scale=0.6)
                    else:
                        put(frame, "✓ Clean rep!", 120, (0, 255, 0))

            # Draw tolerance zone legend (top right corner)
            legend_y = 30
            legend_x = frame.shape[1] - 200
            cv2.rectangle(frame, (legend_x - 10, legend_y - 20), (frame.shape[1] - 10, legend_y + 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (legend_x - 10, legend_y - 20), (frame.shape[1] - 10, legend_y + 100), (255, 255, 255), 1)
            cv2.putText(frame, "Tolerance Zones:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "GREEN = Perfect", (legend_x, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "YELLOW = Good", (legend_x, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "ORANGE = Acceptable", (legend_x, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "RED = Needs Work", (legend_x, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Controls & footer
            put(frame, "Keys: [1]=Squat [2]=Wall-sit [3]=Curl  [ESC]=Quit", frame.shape[0]-10, (255,255,255), 0.6, 1)
            cv2.imshow("AI Physio (Webcam)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('1'):
                exercise = "squat"
                hold_start = None
                hold_secs = 0
                squat_fsm = PhaseFSM(enter_bottom=SQUAT_BOTTOM_DEG, leave_bottom=SQUAT_TOP_DEG)  # reset FSM
            elif k == ord('2'):
                exercise = "wallsit"
                hold_start = None
                hold_secs = 0
            elif k == ord('3'):
                exercise = "curl"
                hold_start = None
                hold_secs = 0
                curl_fsm = PhaseFSM(enter_bottom=CURL_BOTTOM_DEG, leave_bottom=CURL_TOP_DEG)  # reset FSM

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
