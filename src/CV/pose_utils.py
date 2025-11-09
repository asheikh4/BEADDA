# src/CV/pose_utils.py
# Generic utilities: geometry, smoothing, timers, and drawing helpers.

from typing import Tuple, Optional, Any
import numpy as np
import cv2
import time
import math

# ---------------- Geometry ----------------

def angle_3pt(a: Tuple[float, float], b: Tuple[float, float], 
              c: Tuple[float, float]) -> float:
    """
    Angle ABC in degrees using 2D pixel coordinates.
    a, b, c = (x, y) pixels; b is the vertex.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = float(np.dot(ba, bc) / den)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def to_xy(lm: Any, idx: int, w: int, h: int) -> Tuple[int, int]:
    """
    Convert MediaPipe normalized landmark at index idx to pixel (x, y).
    If landmark list is missing or index invalid, returns (0, 0).
    """
    try:
        if lm is None or idx < 0 or idx >= len(lm):
            return 0, 0
        p = lm[idx]
        if not hasattr(p, 'x') or not hasattr(p, 'y'):
            return 0, 0
        return int(p.x * w), int(p.y * h)
    except (IndexError, AttributeError, TypeError):
        return 0, 0

# ---------------- Smoothing & timing ----------------

class EMA:
    """Exponential moving average for scalar streams."""
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.v = None
    def __call__(self, x):
        x = float(x)
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
        return self.v

class MedianFilter:
    """Median filter for scalar streams (odd window preferred)."""
    def __init__(self, k=5):
        self.buf = []
        self.k = int(k)
    def __call__(self, x):
        x = float(x)
        self.buf.append(x)
        if len(self.buf) > self.k:
            self.buf.pop(0)
        return float(np.median(self.buf))

class HoldTimer:
    """
    Simple hold timer. Call update(cond) every frame; returns seconds while cond True,
    otherwise resets and returns 0.0
    """
    def __init__(self) -> None:
        self.t0: Optional[float] = None
    
    def update(self, cond: bool) -> float:
        if cond:
            if self.t0 is None:
                self.t0 = time.time()
            return time.time() - self.t0
        self.t0 = None
        return 0.0

# ---------------- Quality checking utilities ----------------

def get_landmark_visibility(lm: Any, mp_pose: Any) -> Tuple[float, list]:
    """
    Calculate overall landmark visibility and quality.
    Returns visibility score (0-1) and list of missing landmarks.
    """
    if not lm:
        return 0.0, []
    
    P = mp_pose.PoseLandmark
    # Key landmarks for exercises
    key_landmarks = [
        P.LEFT_SHOULDER, P.RIGHT_SHOULDER,
        P.LEFT_ELBOW, P.RIGHT_ELBOW,
        P.LEFT_WRIST, P.RIGHT_WRIST,
        P.LEFT_HIP, P.RIGHT_HIP,
        P.LEFT_KNEE, P.RIGHT_KNEE,
        P.LEFT_ANKLE, P.RIGHT_ANKLE,
    ]
    
    visible_count = 0
    missing = []
    
    for landmark in key_landmarks:
        idx = landmark.value
        if idx < len(lm):
            visibility = lm[idx].visibility
            if visibility > 0.5:  # MediaPipe visibility threshold
                visible_count += 1
            else:
                missing.append(landmark.name)
    
    visibility_score = visible_count / len(key_landmarks) if key_landmarks else 0.0
    return visibility_score, missing

def check_pose_quality(lm: Any, mp_pose: Any, exercise_type: str = "squat") -> Tuple[bool, float, list]:
    """
    Check if pose is suitable for data collection.
    Returns (is_valid, quality_score, issues_list)
    """
    if not lm:
        return False, 0.0, ["No pose detected"]
    
    visibility_score, missing = get_landmark_visibility(lm, mp_pose)
    issues = []
    
    # Minimum visibility threshold
    if visibility_score < 0.7:  # At least 70% of key landmarks visible
        issues.append(f"Low visibility: {visibility_score:.1%}")
        if missing:
            issues.append(f"Missing: {', '.join(missing[:3])}")  # Show first 3
    
    # Exercise-specific checks
    if exercise_type == "squat":
        # Check side visibility (need hip, knee, ankle on one side)
        P = mp_pose.PoseLandmark
        left_visible = all([
            lm[P.LEFT_HIP.value].visibility > 0.5,
            lm[P.LEFT_KNEE.value].visibility > 0.5,
            lm[P.LEFT_ANKLE.value].visibility > 0.5
        ])
        right_visible = all([
            lm[P.RIGHT_HIP.value].visibility > 0.5,
            lm[P.RIGHT_KNEE.value].visibility > 0.5,
            lm[P.RIGHT_ANKLE.value].visibility > 0.5
        ])
        if not (left_visible or right_visible):
            issues.append("Side view not clear (need hip-knee-ankle)")
    elif exercise_type == "curl":
        # Check arm visibility
        P = mp_pose.PoseLandmark
        # This will be checked per side in the exercise logic
        pass
    elif exercise_type == "wallsit":
        # Check side visibility similar to squat
        P = mp_pose.PoseLandmark
        left_visible = all([
            lm[P.LEFT_HIP.value].visibility > 0.5,
            lm[P.LEFT_KNEE.value].visibility > 0.5,
            lm[P.LEFT_ANKLE.value].visibility > 0.5
        ])
        right_visible = all([
            lm[P.RIGHT_HIP.value].visibility > 0.5,
            lm[P.RIGHT_KNEE.value].visibility > 0.5,
            lm[P.RIGHT_ANKLE.value].visibility > 0.5
        ])
        if not (left_visible or right_visible):
            issues.append("Side view not clear (need hip-knee-ankle)")
    
    is_valid = len(issues) == 0 and visibility_score >= 0.7
    quality_score = visibility_score
    
    return is_valid, quality_score, issues

def check_lighting_quality(frame: np.ndarray) -> Tuple[bool, float, list]:
    """
    Check if lighting is adequate for pose detection.
    Returns (is_good, brightness_score, issues)
    """
    if frame is None or frame.size == 0:
        return False, 0.0, ["Invalid frame"]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness (mean pixel value)
    mean_brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    issues = []
    
    # Brightness check (0-255 range, ideal: 50-200)
    if mean_brightness < 50:
        issues.append("Too dark")
        is_good = False
    elif mean_brightness > 200:
        issues.append("Too bright")
        is_good = False
    else:
        is_good = True
    
    # Contrast check (low contrast = poor detection)
    if contrast < 30:
        issues.append("Low contrast")
        is_good = False
    
    brightness_score = min(1.0, mean_brightness / 128.0)  # Normalize to 0-1
    
    return is_good, brightness_score, issues

def validate_person_scale(lm: Any, mp_pose: Any, frame_w: int, frame_h: int) -> Tuple[bool, float, Optional[str]]:
    """
    Check if person is at appropriate distance from camera.
    Returns (is_valid, scale_factor, issue)
    """
    if not lm:
        return False, 0.0, "No pose detected"
    
    P = mp_pose.PoseLandmark
    
    # Calculate person height in pixels (shoulder to ankle)
    # Use the more visible side
    left_visible = lm[P.LEFT_SHOULDER.value].visibility > 0.5
    right_visible = lm[P.RIGHT_SHOULDER.value].visibility > 0.5
    
    if left_visible and lm[P.LEFT_ANKLE.value].visibility > 0.5:
        shoulder_y = lm[P.LEFT_SHOULDER.value].y * frame_h
        ankle_y = lm[P.LEFT_ANKLE.value].y * frame_h
    elif right_visible and lm[P.RIGHT_ANKLE.value].visibility > 0.5:
        shoulder_y = lm[P.RIGHT_SHOULDER.value].y * frame_h
        ankle_y = lm[P.RIGHT_ANKLE.value].y * frame_h
    else:
        return False, 0.0, "Shoulders/ankles not visible"
    
    person_height_px = abs(ankle_y - shoulder_y)
    frame_height = frame_h
    
    # Ideal: person should occupy 40-70% of frame height
    scale_factor = person_height_px / frame_height if frame_height > 0 else 0.0
    
    if scale_factor < 0.3:
        return False, scale_factor, "Too far from camera"
    elif scale_factor > 0.8:
        return False, scale_factor, "Too close to camera"
    else:
        return True, scale_factor, None

class AdaptiveEMA:
    """EMA that adjusts based on landmark confidence."""
    def __init__(self, alpha_base: float = 0.25):
        self.alpha_base = float(alpha_base)
        self.v: Optional[float] = None
    
    def __call__(self, x: float, confidence: float = 1.0) -> float:
        """
        x: value to smooth
        confidence: landmark confidence (0-1)
        Lower confidence = more smoothing
        """
        x = float(x)
        # Adjust alpha based on confidence
        # High confidence (1.0) → less smoothing (alpha = 0.4)
        # Low confidence (0.5) → more smoothing (alpha = 0.1)
        alpha = self.alpha_base * (0.5 + confidence * 0.5)
        
        self.v = x if self.v is None else (1 - alpha) * self.v + alpha * x
        return self.v

# ---------------- Drawing helpers ----------------

def put(img: np.ndarray, txt: str, y: int, col: Tuple[int, int, int] = (255, 255, 255), 
        scale: float = 0.7, thick: int = 2, x: int = 10) -> None:
    """Draw text on image with bounds checking."""
    if img is None or img.size == 0:
        return
    h, w = img.shape[:2]
    if y < 0 or y > h or x < 0 or x > w:
        return  # Skip drawing if out of bounds
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def draw_angle_with_arc(img: np.ndarray, a: Tuple[int, int], b: Tuple[int, int], 
                        c: Tuple[int, int], angle_deg: Optional[float] = None, 
                        color: Tuple[int, int, int] = (0, 255, 0), 
                        show_text: bool = True) -> None:
    """
    Draw the angle at vertex b for pixel points a, b, c.
    If angle_deg is None we compute it via angle_3pt.
    """
    if a is None or b is None or c is None:
        return
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    cx, cy = int(c[0]), int(c[1])

    # lines
    cv2.line(img, (bx, by), (ax, ay), color, 3)
    cv2.line(img, (bx, by), (cx, cy), color, 3)

    # vectors for arc
    v1 = np.array([ax - bx, ay - by], float)
    v2 = np.array([cx - bx, cy - by], float)
    if np.linalg.norm(v1) < 5 or np.linalg.norm(v2) < 5:
        return

    a1 = math.degrees(math.atan2(v1[1], v1[0]))
    a2 = math.degrees(math.atan2(v2[1], v2[0]))
    if a1 < 0: a1 += 360
    if a2 < 0: a2 += 360

    r = int(max(15, min(40, 0.25 * min(np.linalg.norm(v1), np.linalg.norm(v2)))))
    cv2.ellipse(img, (bx, by), (r, r), 0, int(a1), int(a2), color, 2)

    if show_text:
        theta = angle_deg if angle_deg is not None else angle_3pt((ax, ay), (bx, by), (cx, cy))
        mid = (a1 + a2) / 2.0
        offx = int(1.4 * r * math.cos(math.radians(mid)))
        offy = int(1.4 * r * math.sin(math.radians(mid)))
        cv2.putText(img, f"{int(theta)}°", (bx + offx, by + offy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
