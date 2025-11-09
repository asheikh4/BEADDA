# src/CV/exercises.py
# Finalized metrics (2D pixel-space, camera-agnostic)
# - Squat: knee angle, torso vs vertical, tibia vs vertical, hip hinge, knee-over-toe
# - Wall-sit: knee near 90°, torso vs vertical
# - Curl: elbow angle, upper-arm stability (vs vertical)
#
# These functions only need MediaPipe 2D landmarks and image width/height.
# They return numerics plus "zones" you can map to colors in your UI.

from pose_utils import angle_3pt, to_xy, EMA, HoldTimer
import math
import numpy as np

# ----------------------------- Tolerances -----------------------------
# Feel free to tune these from your main config if you prefer one source of truth.

# Squat depth (knee angle — smaller = deeper)
SQUAT_DEPTH_PERFECT     = 100.0
SQUAT_DEPTH_GOOD        = 110.0
SQUAT_DEPTH_ACCEPTABLE  = 120.0

# Back straightness: torso vs vertical (degrees) — smaller is better
BACK_VERT_PERFECT       = 8.0
BACK_VERT_GOOD          = 12.0
BACK_VERT_ACCEPTABLE    = 18.0

# Hip hinge / rounding: angle between torso (shoulder->hip) and thigh (knee->hip)
HINGE_PERFECT           = 25.0
HINGE_GOOD              = 35.0
HINGE_ACCEPTABLE        = 45.0

# Knee over toe (forward along foot axis), normalized by image width
# <= 3% is good, <= 6% acceptable, else poor
KNEE_OVERTOE_GOOD_NORM  = 0.03
KNEE_OVERTOE_OK_NORM    = 0.06

# Wall-sit knee target
WALL_KNEE_TARGET        = 90.0
WALL_KNEE_TOL_PERFECT   = 5.0
WALL_KNEE_TOL_GOOD      = 8.0
WALL_KNEE_TOL_ACCEPT    = 12.0

# Wall-sit back against wall (proxy in 2D: torso near vertical)
WALL_BACK_PERFECT       = 6.0
WALL_BACK_GOOD          = 10.0
WALL_BACK_ACCEPTABLE    = 14.0

# Curl contraction (elbow angle — smaller = better contraction)
CURL_CONTRACTION_PERFECT   = 40.0
CURL_CONTRACTION_GOOD      = 45.0
CURL_CONTRACTION_ACCEPTABLE= 50.0

# Curl upper-arm stability (vs vertical)
CURL_UPPERARM_STABLE_MAX   = 30.0


# ----------------------------- Helpers -----------------------------

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
    0° = perfectly vertical. Uses pixel coordinates (x right, y down).
    """
    vx = float(p_to[0] - p_from[0])
    vy = float(p_to[1] - p_from[1])
    # vertical axis is pointing "up" in math coords, but image y grows down.
    # Using atan2 with (vx, -vy) gives 0° when aligned with vertical.
    return abs(math.degrees(math.atan2(vx, -vy)))

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


# ----------------------------- Rep Counter -----------------------------

class RepCounter:
    """
    Simple finite-state rep counter using a single monotonically varying angle.
    'low' is the bottom threshold, 'high' is the top threshold (with hysteresis).
    """
    def __init__(self, low_thresh, high_thresh):
        self.low, self.high = float(low_thresh), float(high_thresh)
        self.state, self.reps = "top", 0

    def update(self, angle):
        a = float(angle)
        if self.state == "top" and a < self.high:
            self.state = "down"
        elif self.state == "down" and a < self.low:
            self.state = "bottom"
        elif self.state == "bottom" and a > self.high:
            self.state = "top"
            self.reps += 1
        return self.reps, self.state


# ----------------------------- Metrics -----------------------------

def squat_metrics(lm, w, h, mp_pose):
    """
    Squat analysis (2D pixels).
    Returns a dict with angles, normalized distances, zones, and key joint pixels.
    """
    P = mp_pose.PoseLandmark
    LHIP, LKNEE, LANK  = P.LEFT_HIP.value, P.LEFT_KNEE.value, P.LEFT_ANKLE.value
    LSH, LHEEL, LTOE   = P.LEFT_SHOULDER.value, P.LEFT_HEEL.value, P.LEFT_FOOT_INDEX.value
    RHIP, RKNEE, RANK  = P.RIGHT_HIP.value, P.RIGHT_KNEE.value, P.RIGHT_ANKLE.value
    RSH, RHEEL, RTOE   = P.RIGHT_SHOULDER.value, P.RIGHT_HEEL.value, P.RIGHT_FOOT_INDEX.value

    # Build both sides and choose whichever has more visible pixels (simple heuristic)
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

    # Guard against missing points (all zeros sometimes)
    if min(hip + knee + ank + sh) == 0:
        return None

    # Knee angle (hip-knee-ankle)
    knee_ang = angle_3pt(hip, knee, ank)

    # Torso vs vertical (back straightness)
    torso_vertical = angle_to_vertical_2d(hip, sh)

    # Tibia vs vertical (shin angle)
    tibia_vertical = angle_to_vertical_2d(ank, knee)

    # Hip hinge / rounding: angle between torso (sh<-hip) and thigh (knee<-hip)
    torso_vec = (sh[0] - hip[0],  sh[1] - hip[1])
    thigh_vec = (knee[0] - hip[0], knee[1] - hip[1])
    hinge_deg = _angle_between(torso_vec, thigh_vec)

    # Knee-over-toe along foot axis (2D)
    knee_forward_norm = None
    if all(v > 0 for v in heel + toe + ank + knee) and w > 0:
        foot_axis = _unit(np.array(toe) - np.array(heel))
        rel = np.array(knee) - np.array(ank)
        forward_pixels = float(np.dot(rel, foot_axis))
        knee_forward_norm = forward_pixels / float(w)  # normalize by image width

    # Zones
    depth_zone, depth_color = get_tolerance_zone(
        knee_ang, SQUAT_DEPTH_PERFECT, SQUAT_DEPTH_GOOD, SQUAT_DEPTH_ACCEPTABLE, lower_is_better=True
    )
    back_zone, back_color = get_tolerance_zone(
        torso_vertical, BACK_VERT_PERFECT, BACK_VERT_GOOD, BACK_VERT_ACCEPTABLE, lower_is_better=True
    )
    hinge_zone, hinge_color = get_tolerance_zone(
        hinge_deg, HINGE_PERFECT, HINGE_GOOD, HINGE_ACCEPTABLE, lower_is_better=True
    )

    knee_toe_zone, knee_toe_color = ("perfect", (0, 255, 0))
    if knee_forward_norm is not None:
        if knee_forward_norm <= KNEE_OVERTOE_GOOD_NORM:
            knee_toe_zone, knee_toe_color = ("good", (0, 255, 255) if knee_forward_norm > 0 else (0, 255, 0))
        elif knee_forward_norm <= KNEE_OVERTOE_OK_NORM:
            knee_toe_zone, knee_toe_color = ("acceptable", (0, 165, 255))
        else:
            knee_toe_zone, knee_toe_color = ("poor", (0, 0, 255))

    return {
        "side_used": side_used,
        "knee_angle": knee_ang,
        "torso_vertical_deg": torso_vertical,
        "tibia_vertical_deg": tibia_vertical,
        "hinge_deg": hinge_deg,
        "knee_forward_norm": knee_forward_norm,   # ~0..0.1 typical
        "depth_zone": depth_zone,     "depth_color": depth_color,
        "back_zone": back_zone,       "back_color": back_color,
        "hinge_zone": hinge_zone,     "hinge_color": hinge_color,
        "knee_toe_zone": knee_toe_zone, "knee_toe_color": knee_toe_color,
        "joints_px": {"hip": hip, "knee": knee, "ankle": ank, "shoulder": sh, "heel": heel, "toe": toe},
    }


def wallsit_metrics(lm, w, h, mp_pose):
    """
    Wall-sit analysis (2D pixels): knee should be ~90°, torso near vertical.
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

    knee_ang = angle_3pt(hip, knee, ank)
    torso_vertical = angle_to_vertical_2d(hip, sh)

    knee_zone, knee_color = get_tolerance_zone_deviation(
        knee_ang, WALL_KNEE_TARGET, WALL_KNEE_TOL_PERFECT, WALL_KNEE_TOL_GOOD, WALL_KNEE_TOL_ACCEPT
    )
    back_zone, back_color = get_tolerance_zone(
        torso_vertical, WALL_BACK_PERFECT, WALL_BACK_GOOD, WALL_BACK_ACCEPTABLE, lower_is_better=True
    )

    return {
        "knee_angle": knee_ang,
        "torso_vertical_deg": torso_vertical,
        "knee_zone": knee_zone, "knee_color": knee_color,
        "back_zone": back_zone, "back_color": back_color,
        "knee_90": abs(knee_ang - WALL_KNEE_TARGET) <= WALL_KNEE_TOL_ACCEPT,
        "back_vertical": torso_vertical <= WALL_BACK_ACCEPTABLE,
        "joints_px": {"hip": hip, "knee": knee, "ankle": ank, "shoulder": sh},
    }


def curl_metrics(lm, w, h, mp_pose, side="LEFT"):
    """
    Curl analysis (2D pixels): elbow angle + upper-arm stability (vs vertical).
    side: "LEFT" | "RIGHT"
    """
    S = side.upper()
    P = mp_pose.PoseLandmark
    SHO = getattr(P, f"{S}_SHOULDER").value
    ELB = getattr(P, f"{S}_ELBOW").value
    WRI = getattr(P, f"{S}_WRIST").value

    sh = to_xy(lm, SHO, w, h)
    el = to_xy(lm, ELB, w, h)
    wr = to_xy(lm, WRI, w, h)

    if min(sh + el + wr) == 0:
        return None

    elbow = angle_3pt(sh, el, wr)

    # Contraction zone
    contraction_zone, contraction_color = get_tolerance_zone(
        elbow, CURL_CONTRACTION_PERFECT, CURL_CONTRACTION_GOOD, CURL_CONTRACTION_ACCEPTABLE, lower_is_better=True
    )

    # Upper-arm vs vertical (should be relatively stable)
    upper_arm_vert = angle_to_vertical_2d(sh, el)
    upper_arm_stable = upper_arm_vert < CURL_UPPERARM_STABLE_MAX

    return {
        "elbow_angle": elbow,
        "contraction_zone": contraction_zone, "contraction_color": contraction_color,
        "upper_arm_vertical_deg": upper_arm_vert,
        "upper_arm_stable": upper_arm_stable,
        "joints_px": {"shoulder": sh, "elbow": el, "wrist": wr},
    }
