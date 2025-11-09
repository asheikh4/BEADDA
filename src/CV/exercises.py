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
# Evidence-based biomechanics thresholds from peer-reviewed research
# Sources: Schoenfeld (2010), Swinton et al. (2012), Lorenzetti et al. (2018), 
# Hartmann et al. (2013), Escamilla series

# Depth (knee angle; smaller = deeper)
# Parallel is ≈90-100°; deeper increases patellofemoral forces but safe with good technique
SQUAT_DEPTH_PERFECT     = 100.0
SQUAT_DEPTH_GOOD        = 110.0
SQUAT_DEPTH_ACCEPTABLE  = 120.0

# Torso lean vs vertical (degrees)
# High-bar/front squats more upright; low-bar allows greater forward lean
TORSO_LEAN_PERFECT      = 25.0
TORSO_LEAN_GOOD         = 35.0
TORSO_LEAN_ACCEPTABLE   = 45.0

# Trunk-tibia harmony (|torso angle - shank angle| in degrees)
# Helps distribute load between hip and knee, reduces lumbar shear
TRUNK_TIBIA_DIFF_PERFECT    = 10.0
TRUNK_TIBIA_DIFF_GOOD       = 15.0
TRUNK_TIBIA_DIFF_ACCEPTABLE = 20.0

# Dynamic knee valgus (FPPA - Frontal Plane Projection Angle, degrees)
# >10° is considered excessive in clinical literature
VALGUS_PERFECT_MAX      = 5.0
VALGUS_ACCEPTABLE_MAX   = 10.0  # >10 = poor

# Tibia angle vs vertical (degrees)
# "Knees past toes" increases knee moments but often necessary for depth
TIBIA_ANGLE_GOOD_MAX        = 35.0
TIBIA_ANGLE_ACCEPTABLE_MAX  = 40.0

# Lumbar excursion from neutral (degrees change during rep)
# Trained lifters show ~≤18° flex/extend ROM; avoid extreme flexion under load
LUMBAR_EXCURSION_PERFECT_MAX = 10.0
LUMBAR_EXCURSION_GOOD_MAX    = 15.0
LUMBAR_EXCURSION_POOR_MIN    = 20.0  # flag if exceeded

# Depth symmetry L vs R (knee angles compared at bottom, degrees difference)
DEPTH_SYMMETRY_PERFECT      = 5.0
DEPTH_SYMMETRY_ACCEPTABLE   = 10.0

# Back straightness: deviation between left and right torso angles (degrees)
# Measures if back is a straight line (allows forward tilt, penalizes rounding/curving)
BACK_STRAIGHT_PERFECT       = 5.0
BACK_STRAIGHT_GOOD          = 10.0
BACK_STRAIGHT_ACCEPTABLE    = 15.0

# Wall-sit thresholds - Evidence-based biomechanics
# Sources: MDPI biomechanical analysis, Verywell Fit, Cleveland Clinic
# Key points: thighs parallel to ground (≈90° knees), back flat against wall,
# weight on heels, knees over ankles (not forward past) for safety

# Knee angle target (hip-knee-ankle)
# Thighs should be parallel to ground, knees bent about 90°
WALL_KNEE_TARGET        = 90.0
WALL_KNEE_TOL_PERFECT   = 5.0   # ±5° → 85-95°
WALL_KNEE_TOL_GOOD      = 8.0   # ±8° → 82-98°
WALL_KNEE_TOL_ACCEPTABLE = 12.0  # ±12° → 78-102°

# Back vs vertical (shoulder->hip vs vertical)
# Back flat against wall = vertical alignment
WALL_BACK_VERT_PERFECT    = 6.0
WALL_BACK_VERT_GOOD       = 10.0
WALL_BACK_VERT_ACCEPTABLE = 14.0

# Knee over toe (forward travel, normalized by image width)
# Avoid excessive forward knee travel past ankle for safety
WALL_KNEE_OVERTOE_GOOD_NORM    = 0.05   # 5% image width
WALL_KNEE_OVERTOE_OK_NORM      = 0.08   # 8%
WALL_KNEE_OVERTOE_ACCEPTABLE   = 0.12   # 12%

# Legacy constants (for backward compatibility)
WALL_BACK_PERFECT = WALL_BACK_VERT_PERFECT
WALL_BACK_GOOD = WALL_BACK_VERT_GOOD
WALL_BACK_ACCEPTABLE = WALL_BACK_VERT_ACCEPTABLE

# Curl thresholds - Evidence-based biomechanics
# Sources: PMC studies on biceps activation, MDPI on forearm rotation, ResearchGate biomechanical analysis
# Highest biceps activation occurs at ~56° elbow flexion; targeting ≤45° ensures full contraction

# Elbow angle (contraction - smaller = better contraction)
# Research found 56° gave highest activation; ≤45° is stricter for full contraction
CURL_ELBOW_PERFECT      = 45.0
CURL_ELBOW_GOOD         = 50.0
CURL_ELBOW_ACCEPT       = 55.0

# Upper arm vertical alignment (shoulder->elbow vs vertical)
# Prevents shoulder/torso movement to compensate (swinging/cheating)
CURL_UPPERARM_VERT_PERFECT = 10.0
CURL_UPPERARM_VERT_GOOD    = 20.0
CURL_UPPERARM_VERT_ACCEPT  = 30.0

# Legacy constants (for backward compatibility)
CURL_CONTRACTION_PERFECT   = CURL_ELBOW_PERFECT
CURL_CONTRACTION_GOOD      = CURL_ELBOW_GOOD
CURL_CONTRACTION_ACCEPTABLE = CURL_ELBOW_ACCEPT
CURL_UPPERARM_STABLE_MAX   = CURL_UPPERARM_VERT_ACCEPT


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
    0° = perfectly vertical (pointing up or down). Uses pixel coordinates (x right, y down).
    
    Returns the absolute angle deviation from vertical (0-90° range).
    """
    vx = float(p_to[0] - p_from[0])
    vy = float(p_to[1] - p_from[1])
    
    # Handle zero vector
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    
    # Calculate the angle of the vector in image coordinates
    # atan2(vx, vy) gives angle from positive x-axis, with y increasing downward
    vector_angle = math.degrees(math.atan2(vx, vy))
    
    # Normalize to 0-180 range (since we only care about deviation from vertical)
    # Vertical vectors (pointing up or down) should give 0° or 180°
    # We want the deviation from vertical, so:
    # - If vector is vertical (vx ≈ 0), angle is 0° or 180°, deviation = 0°
    # - If vector is horizontal (vy ≈ 0), angle is 90° or -90°, deviation = 90°
    
    # The angle from vertical is: min(|vector_angle|, |vector_angle - 180|, |vector_angle + 180|)
    # But simpler: use the horizontal component relative to the vector length
    vector_length = math.sqrt(vx * vx + vy * vy)
    if vector_length < 1e-6:
        return 0.0
    
    # The angle from vertical is the arcsin of the horizontal component
    # divided by the vector length
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
    For side view, we approximate valgus by comparing left and right knee positions.
    In side view, true valgus is hard to detect, so we use a simplified metric:
    deviation of knee from expected alignment with hip and ankle.
    
    Returns valgus angle in degrees (positive = valgus tendency).
    Note: This is an approximation; true FPPA requires frontal view.
    """
    # For side view, we can approximate by looking at knee alignment
    # Calculate for both sides and take the maximum deviation
    valgus_angles = []
    
    for hip, knee, ankle in [(l_hip, l_knee, l_ankle), (r_hip, r_knee, r_ankle)]:
        if min(hip + knee + ankle) == 0:
            continue
        
        # Calculate angle at knee (hip-knee-ankle)
        knee_angle = angle_3pt(hip, knee, ankle)
        
        # Expected angle for straight alignment is ~180° (straight line)
        # Deviation from this indicates misalignment
        # But knee angle in squat is naturally < 180°, so we need a different approach
        
        # Instead, calculate the angle between thigh and shank vectors
        thigh_vec = np.array([knee[0] - hip[0], knee[1] - hip[1]], dtype=float)
        shank_vec = np.array([ankle[0] - knee[0], ankle[1] - knee[1]], dtype=float)
        
        thigh_norm = np.linalg.norm(thigh_vec)
        shank_norm = np.linalg.norm(shank_vec)
        
        if thigh_norm < 1e-6 or shank_norm < 1e-6:
            continue
        
        # Calculate the angle between vectors
        dot_product = np.clip(np.dot(thigh_vec / thigh_norm, shank_vec / shank_norm), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))
        
        # In a perfectly aligned leg (no valgus), thigh and shank form a smooth arc
        # For side view, we approximate valgus as deviation from expected alignment
        # Use the horizontal component of the knee deviation
        # Simplified: use the angle deviation as a proxy
        # A more accurate measure would require frontal view
        
        # For now, return a conservative estimate based on knee angle consistency
        # This is a placeholder - true valgus detection needs frontal view
        valgus_estimate = 0.0  # Conservative: assume no valgus in side view
    
    # Return average or max - for side view, this is limited
    # In practice, valgus detection from side view is not reliable
    # Return None to indicate it can't be measured accurately
    return None  # Valgus requires frontal view for accurate measurement

class LumbarExcursionTracker:
    """
    Tracks lumbar excursion (change in back angle) during a rep.
    Records neutral angle at rep start and tracks maximum excursion.
    """
    def __init__(self):
        self.neutral_angle = None
        self.max_excursion = 0.0
        self.current_angle = None
        self.rep_active = False
    
    def start_rep(self, current_torso_angle):
        """Called when rep starts (entering descent phase)."""
        if not self.rep_active:
            self.neutral_angle = current_torso_angle
            self.max_excursion = 0.0
            self.rep_active = True
    
    def update(self, current_torso_angle, rep_phase):
        """Update with current torso angle and rep phase."""
        self.current_angle = current_torso_angle
        
        if self.rep_active and self.neutral_angle is not None:
            excursion = abs(current_torso_angle - self.neutral_angle)
            self.max_excursion = max(self.max_excursion, excursion)
        
        # Reset when rep completes (back to top)
        if rep_phase == "top" and self.rep_active:
            self.rep_active = False
    
    def get_excursion(self):
        """Get current maximum excursion."""
        return self.max_excursion if self.max_excursion > 0 else 0.0
    
    def reset(self):
        """Reset tracker."""
        self.neutral_angle = None
        self.max_excursion = 0.0
        self.current_angle = None
        self.rep_active = False


# ----------------------------- Rep Counter -----------------------------

class RepCounter:
    """
    Simple finite-state rep counter using a single monotonically varying angle.
    'low' is the bottom threshold, 'high' is the top threshold (with hysteresis).
    """
    def __init__(self, low_thresh, high_thresh):
        self.low, self.high = float(low_thresh), float(high_thresh)
        self.state, self.reps = "top", 0
    
    def reset(self):
        """Reset the counter to initial state."""
        self.state, self.reps = "top", 0

    def update(self, angle):
        a = float(angle)
        if self.state == "top" and a < self.high:
            self.state = "down"
        elif self.state == "down":
            if a < self.low:
                self.state = "bottom"
            elif a >= self.high:  # Allow recovery back to top without counting
                self.state = "top"
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

    # Get both sides for back straightness check
    l_hip = to_xy(lm, LHIP, w, h)
    l_sh = to_xy(lm, LSH, w, h)
    r_hip = to_xy(lm, RHIP, w, h)
    r_sh = to_xy(lm, RSH, w, h)

    # Knee angle (hip-knee-ankle)
    knee_ang = angle_3pt(hip, knee, ank)

    # Back straightness: Check if left and right torso have similar angles
    # This allows forward tilt (normal in squats) but detects rounding/curving
    # A straight back means both sides should have similar torso angles
    back_straightness_deg = None
    if (min(l_hip + l_sh) > 0 and min(r_hip + r_sh) > 0):
        # Calculate the angle of each torso line (from hip to shoulder)
        # We use atan2 to get the angle, which works regardless of camera view
        left_torso_angle = math.degrees(math.atan2(l_sh[1] - l_hip[1], l_sh[0] - l_hip[0]))
        right_torso_angle = math.degrees(math.atan2(r_sh[1] - r_hip[1], r_sh[0] - r_hip[0]))
        
        # Normalize angles to 0-360 range
        left_torso_angle = left_torso_angle % 360
        right_torso_angle = right_torso_angle % 360
        
        # Calculate the difference (handle wrap-around)
        angle_diff = abs(left_torso_angle - right_torso_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # The difference indicates back curvature: small = straight, large = curved
        back_straightness_deg = angle_diff
    else:
        # Fallback: if we can't measure both sides, use a conservative estimate
        back_straightness_deg = BACK_STRAIGHT_ACCEPTABLE

    # ========== EVIDENCE-BASED BIOMECHANICS METRICS ==========
    
    # 1. Depth (knee angle - already computed)
    # knee_ang is computed above
    
    # 2. Torso lean vs vertical (shoulder->hip vs vertical)
    torso_lean_deg = angle_to_vertical_2d(hip, sh)
    
    # 3. Tibia angle vs vertical (ankle->knee vs vertical)
    tibia_angle_deg = angle_to_vertical_2d(ank, knee)
    
    # 4. Trunk-tibia harmony (|torso angle - shank angle|)
    trunk_tibia_diff = abs(torso_lean_deg - tibia_angle_deg)
    
    # 5. Depth symmetry (L vs R knee angles)
    l_knee_ang = angle_3pt(to_xy(lm, LHIP, w, h), to_xy(lm, LKNEE, w, h), to_xy(lm, LANK, w, h))
    r_knee_ang = angle_3pt(to_xy(lm, RHIP, w, h), to_xy(lm, RKNEE, w, h), to_xy(lm, RANK, w, h))
    depth_symmetry_deg = abs(l_knee_ang - r_knee_ang) if (l_knee_ang > 0 and r_knee_ang > 0) else 0.0
    
    # 6. Valgus (FPPA) - Note: requires frontal view for accurate measurement
    # For side view, we return None as it cannot be accurately measured
    valgus_deg = None  # calculate_valgus_fppa_simplified(...) - not reliable in side view
    
    # 7. Lumbar excursion - tracked externally via LumbarExcursionTracker class
    # This will be computed in main.py using the tracker
    
    # Hip hinge (angle between torso and thigh) - keeping for reference
    torso_vec = (sh[0] - hip[0],  sh[1] - hip[1])
    thigh_vec = (knee[0] - hip[0], knee[1] - hip[1])
    hinge_deg = _angle_between(torso_vec, thigh_vec)
    
    # Knee-over-toe along foot axis (2D) - keeping for reference
    knee_forward_norm = None
    if all(v > 0 for v in heel + toe + ank + knee) and w > 0:
        foot_axis = _unit(np.array(toe) - np.array(heel))
        rel = np.array(knee) - np.array(ank)
        forward_pixels = float(np.dot(rel, foot_axis))
        knee_forward_norm = forward_pixels / float(w)

    # ========== ZONES (Evidence-based thresholds) ==========
    
    # Depth zone
    depth_zone, depth_color = get_tolerance_zone(
        knee_ang, SQUAT_DEPTH_PERFECT, SQUAT_DEPTH_GOOD, SQUAT_DEPTH_ACCEPTABLE, lower_is_better=True
    )
    
    # Torso lean zone
    torso_lean_zone, torso_lean_color = get_tolerance_zone(
        torso_lean_deg, TORSO_LEAN_PERFECT, TORSO_LEAN_GOOD, TORSO_LEAN_ACCEPTABLE, lower_is_better=True
    )
    
    # Trunk-tibia harmony zone
    trunk_tibia_zone, trunk_tibia_color = get_tolerance_zone(
        trunk_tibia_diff, TRUNK_TIBIA_DIFF_PERFECT, TRUNK_TIBIA_DIFF_GOOD, 
        TRUNK_TIBIA_DIFF_ACCEPTABLE, lower_is_better=True
    )
    
    # Tibia angle zone (knee travel)
    tibia_angle_zone, tibia_angle_color = ("perfect", (0, 255, 0))
    if tibia_angle_deg <= TIBIA_ANGLE_GOOD_MAX:
        tibia_angle_zone, tibia_angle_color = ("good", (0, 255, 255))
    elif tibia_angle_deg <= TIBIA_ANGLE_ACCEPTABLE_MAX:
        tibia_angle_zone, tibia_angle_color = ("acceptable", (0, 165, 255))
    else:
        tibia_angle_zone, tibia_angle_color = ("poor", (0, 0, 255))
    
    # Depth symmetry zone
    depth_symmetry_zone, depth_symmetry_color = ("perfect", (0, 255, 0))
    if depth_symmetry_deg <= DEPTH_SYMMETRY_PERFECT:
        depth_symmetry_zone, depth_symmetry_color = ("perfect", (0, 255, 0))
    elif depth_symmetry_deg <= DEPTH_SYMMETRY_ACCEPTABLE:
        depth_symmetry_zone, depth_symmetry_color = ("acceptable", (0, 165, 255))
    else:
        depth_symmetry_zone, depth_symmetry_color = ("poor", (0, 0, 255))
    
    # Valgus zone (if measurable)
    valgus_zone, valgus_color = ("unknown", (128, 128, 128))
    if valgus_deg is not None:
        valgus_abs = abs(valgus_deg)
        if valgus_abs <= VALGUS_PERFECT_MAX:
            valgus_zone, valgus_color = ("perfect", (0, 255, 0))
        elif valgus_abs <= VALGUS_ACCEPTABLE_MAX:
            valgus_zone, valgus_color = ("acceptable", (0, 165, 255))
        else:
            valgus_zone, valgus_color = ("poor", (0, 0, 255))
    
    # Back straightness zone (legacy - keeping for compatibility)
    if back_straightness_deg is not None:
        back_zone, back_color = get_tolerance_zone(
            back_straightness_deg, BACK_STRAIGHT_PERFECT, BACK_STRAIGHT_GOOD, 
            BACK_STRAIGHT_ACCEPTABLE, lower_is_better=True
        )
    else:
        back_zone, back_color = "unknown", (128, 128, 128)
    
    # Hinge zone (legacy - keeping for reference)
    hinge_zone, hinge_color = get_tolerance_zone(
        hinge_deg, 25.0, 35.0, 45.0, lower_is_better=True
    )
    
    # Knee-over-toe zone (legacy - keeping for reference)
    knee_toe_zone, knee_toe_color = ("perfect", (0, 255, 0))
    if knee_forward_norm is not None:
        # Using simplified thresholds
        if knee_forward_norm <= 0.03:
            knee_toe_zone, knee_toe_color = ("good", (0, 255, 255))
        elif knee_forward_norm <= 0.06:
            knee_toe_zone, knee_toe_color = ("acceptable", (0, 165, 255))
        else:
            knee_toe_zone, knee_toe_color = ("poor", (0, 0, 255))

    return {
        "side_used": side_used,
        # Core metrics
        "knee_angle": knee_ang,
        "torso_lean_deg": torso_lean_deg,
        "tibia_angle_deg": tibia_angle_deg,
        "trunk_tibia_diff": trunk_tibia_diff,
        "depth_symmetry_deg": depth_symmetry_deg,
        "valgus_deg": valgus_deg,
        # Legacy metrics (for compatibility)
        "torso_vertical_deg": torso_lean_deg,  # Alias for torso_lean_deg
        "back_straightness_deg": back_straightness_deg if back_straightness_deg is not None else 0.0,
        "tibia_vertical_deg": tibia_angle_deg,  # Alias for tibia_angle_deg
        "hinge_deg": hinge_deg,
        "knee_forward_norm": knee_forward_norm,
        # Zones
        "depth_zone": depth_zone, "depth_color": depth_color,
        "torso_lean_zone": torso_lean_zone, "torso_lean_color": torso_lean_color,
        "trunk_tibia_zone": trunk_tibia_zone, "trunk_tibia_color": trunk_tibia_color,
        "tibia_angle_zone": tibia_angle_zone, "tibia_angle_color": tibia_angle_color,
        "depth_symmetry_zone": depth_symmetry_zone, "depth_symmetry_color": depth_symmetry_color,
        "valgus_zone": valgus_zone, "valgus_color": valgus_color,
        "back_zone": back_zone, "back_color": back_color,  # Legacy
        "hinge_zone": hinge_zone, "hinge_color": hinge_color,  # Legacy
        "knee_toe_zone": knee_toe_zone, "knee_toe_color": knee_toe_color,  # Legacy
        # Joint positions
        "joints_px": {"hip": hip, "knee": knee, "ankle": ank, "shoulder": sh, "heel": heel, "toe": toe},
    }


def wallsit_metrics(lm, w, h, mp_pose):
    """
    Wall-sit analysis (2D pixels) with evidence-based biomechanics.
    Metrics: knee angle (~90°), back vertical alignment, knee-over-toe position.
    
    Based on research:
    - Thighs parallel to ground (≈90° knees) for proper activation
    - Back flat against wall (vertical alignment)
    - Knees over ankles (not forward past) for safety
    """
    P = mp_pose.PoseLandmark
    LHIP, LKNEE, LANK, LSH = P.LEFT_HIP.value, P.LEFT_KNEE.value, P.LEFT_ANKLE.value, P.LEFT_SHOULDER.value
    LHEEL, LTOE = P.LEFT_HEEL.value, P.LEFT_FOOT_INDEX.value
    RHIP, RKNEE, RANK, RSH = P.RIGHT_HIP.value, P.RIGHT_KNEE.value, P.RIGHT_ANKLE.value, P.RIGHT_SHOULDER.value
    RHEEL, RTOE = P.RIGHT_HEEL.value, P.RIGHT_FOOT_INDEX.value

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

    hip, knee, ank, sh, heel, toe = S["hip"], S["knee"], S["ank"], S["sh"], S["heel"], S["toe"]
    if min(hip + knee + ank + sh) == 0:
        return None

    # ========== EVIDENCE-BASED BIOMECHANICS METRICS ==========
    
    # 1. Knee angle (hip-knee-ankle)
    # Thighs should be parallel to ground, knees bent about 90°
    knee_ang = angle_3pt(hip, knee, ank)
    
    # 2. Back vs vertical (shoulder->hip vs vertical)
    # Back flat against wall = vertical alignment
    torso_vertical_deg = angle_to_vertical_2d(hip, sh)
    
    # 3. Knee over toe (forward travel, normalized by image width)
    # Avoid excessive forward knee travel past ankle for safety
    # In wall-sit, knees should be over ankles (aligned), not forward
    knee_over_toe_norm = None
    if all(v > 0 for v in knee + ank) and w > 0:
        # Calculate forward knee position relative to ankle (in pixels)
        # Positive = knee forward of ankle (bad - excessive forward travel)
        # Negative/zero = knee aligned with or behind ankle (good)
        knee_forward_pixels = float(knee[0] - ank[0])
        # Normalize by image width - only count forward displacement
        if knee_forward_pixels > 0:
            knee_over_toe_norm = knee_forward_pixels / float(w)  # Normalized by image width
        else:
            knee_over_toe_norm = 0.0  # Knee is aligned with or behind ankle, which is good
    
    # ========== ZONES (Evidence-based thresholds) ==========
    
    # Knee angle zone (deviation from 90° target)
    knee_zone, knee_color = get_tolerance_zone_deviation(
        knee_ang, WALL_KNEE_TARGET, WALL_KNEE_TOL_PERFECT, WALL_KNEE_TOL_GOOD, WALL_KNEE_TOL_ACCEPTABLE
    )
    
    # Back vertical alignment zone
    back_zone, back_color = get_tolerance_zone(
        torso_vertical_deg, WALL_BACK_VERT_PERFECT, WALL_BACK_VERT_GOOD, 
        WALL_BACK_VERT_ACCEPTABLE, lower_is_better=True
    )
    
    # Knee-over-toe zone
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
        "torso_vertical_deg": torso_vertical_deg,
        "knee_over_toe_norm": knee_over_toe_norm,
        # Zones
        "knee_zone": knee_zone,
        "knee_color": knee_color,
        "back_zone": back_zone,
        "back_color": back_color,
        "knee_over_toe_zone": knee_over_toe_zone,
        "knee_over_toe_color": knee_over_toe_color,
        # Legacy fields (for backward compatibility)
        "knee_90": abs(knee_ang - WALL_KNEE_TARGET) <= WALL_KNEE_TOL_ACCEPTABLE,
        "back_vertical": torso_vertical_deg <= WALL_BACK_VERT_ACCEPTABLE,
        # Joint positions
        "joints_px": {"hip": hip, "knee": knee, "ankle": ank, "shoulder": sh, "heel": heel, "toe": toe},
    }


def curl_metrics(lm, w, h, mp_pose, side="LEFT"):
    """
    Curl analysis (2D pixels) with evidence-based biomechanics.
    Metrics: elbow angle (contraction) + upper-arm vertical alignment (prevents cheating).
    side: "LEFT" | "RIGHT"
    
    Based on research:
    - Highest biceps activation at ~56° elbow flexion
    - Upper arm stability prevents shoulder/torso compensation
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

    # ========== EVIDENCE-BASED BIOMECHANICS METRICS ==========
    
    # 1. Elbow angle (contraction - smaller = better contraction)
    # Research found 56° gave highest activation; ≤45° ensures full contraction
    elbow_angle = angle_3pt(sh, el, wr)

    # 2. Upper-arm vertical alignment (shoulder->elbow vs vertical)
    # Prevents swinging/cheating by keeping upper arm stable
    upper_arm_vert_deg = angle_to_vertical_2d(sh, el)
    
    # ========== ZONES (Evidence-based thresholds) ==========
    
    # Elbow contraction zone
    elbow_zone, elbow_color = get_tolerance_zone(
        elbow_angle, CURL_ELBOW_PERFECT, CURL_ELBOW_GOOD, CURL_ELBOW_ACCEPT, lower_is_better=True
    )
    
    # Upper arm vertical alignment zone
    upper_arm_zone, upper_arm_color = get_tolerance_zone(
        upper_arm_vert_deg, CURL_UPPERARM_VERT_PERFECT, CURL_UPPERARM_VERT_GOOD, 
        CURL_UPPERARM_VERT_ACCEPT, lower_is_better=True
    )
    
    # Legacy: upper_arm_stable (boolean) for backward compatibility
    upper_arm_stable = upper_arm_vert_deg < CURL_UPPERARM_VERT_ACCEPT

    return {
        # Core metrics
        "elbow_angle": elbow_angle,
        "upper_arm_vertical_deg": upper_arm_vert_deg,
        # Zones
        "elbow_zone": elbow_zone,
        "elbow_color": elbow_color,
        "upper_arm_zone": upper_arm_zone,
        "upper_arm_color": upper_arm_color,
        # Legacy fields (for backward compatibility)
        "contraction_zone": elbow_zone,
        "contraction_color": elbow_color,
        "upper_arm_stable": upper_arm_stable,
        # Joint positions
        "joints_px": {"shoulder": sh, "elbow": el, "wrist": wr},
    }
