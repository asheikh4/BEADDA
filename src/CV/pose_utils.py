import numpy as np
import time

def angle_3pt(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def to_xy(lm, idx, w, h):
    p = lm[idx]
    return (int(p.x * w), int(p.y * h))

class EMA:
    def __init__(self, alpha=0.25):
        self.alpha, self.v = alpha, None
    def __call__(self, x):
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
        return self.v

class HoldTimer:
    def __init__(self): self.t0 = None
    def update(self, cond):
        if cond:
            if self.t0 is None: self.t0 = time.time()
            return time.time() - self.t0
        self.t0 = None
        return 0.0
