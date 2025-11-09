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
