from pose_utils import angle_3pt, to_xy, EMA, HoldTimer

class RepCounter:
    def __init__(self, low_thresh, high_thresh):
        self.low, self.high = low_thresh, high_thresh
        self.state, self.reps = "top", 0
    def update(self, angle):
        if self.state == "top" and angle < self.high:
            self.state = "down"
        elif self.state == "down" and angle < self.low:
            self.state = "bottom"
        elif self.state == "bottom" and angle > self.high:
            self.state = "top"; self.reps += 1
        return self.reps, self.state

def squat_metrics(lm, w, h, mp_pose):
    LHIP=mp_pose.PoseLandmark.LEFT_HIP.value
    LKNEE=mp_pose.PoseLandmark.LEFT_KNEE.value
    LANK=mp_pose.PoseLandmark.LEFT_ANKLE.value
    LSH =mp_pose.PoseLandmark.LEFT_SHOULDER.value

    hip = to_xy(lm, LHIP, w, h)
    knee= to_xy(lm, LKNEE, w, h)
    ank = to_xy(lm, LANK, w, h)
    sh  = to_xy(lm, LSH,  w, h)

    knee_ang = angle_3pt(hip, knee, ank)
    # back angle vs vertical: shoulder-hip vector
    back_vec = (sh[0]-hip[0], sh[1]-hip[1])
    # if x deviation large, leaning forward/back; rough heuristic
    back_lean = abs(back_vec[0])

    depth_ok   = knee_ang < 105
    knee_valgus = (knee[0] < ank[0] - 10)  # tweak per camera

    return {"knee_angle":knee_ang, "depth_ok":depth_ok, "knee_valgus":knee_valgus, "back_lean_px":back_lean}

def wallsit_metrics(lm, w, h, mp_pose):
    LHIP=mp_pose.PoseLandmark.LEFT_HIP.value
    LKNEE=mp_pose.PoseLandmark.LEFT_KNEE.value
    LANK=mp_pose.PoseLandmark.LEFT_ANKLE.value
    LSH =mp_pose.PoseLandmark.LEFT_SHOULDER.value

    hip = to_xy(lm, LHIP, w, h)
    knee= to_xy(lm, LKNEE, w, h)
    ank = to_xy(lm, LANK, w, h)
    sh  = to_xy(lm, LSH,  w, h)

    knee_ang = angle_3pt(hip, knee, ank)
    knee_90 = abs(knee_ang - 90) <= 12
    back_vertical = abs(sh[0] - hip[0]) < 20
    return {"knee_angle":knee_ang, "knee_90":knee_90, "back_vertical":back_vertical}

def curl_metrics(lm, w, h, mp_pose, side="LEFT"):
    S = side.upper()
    SHO=getattr(mp_pose.PoseLandmark, f"{S}_SHOULDER").value
    ELB=getattr(mp_pose.PoseLandmark, f"{S}_ELBOW").value
    WRI=getattr(mp_pose.PoseLandmark, f"{S}_WRIST").value

    sh = to_xy(lm, SHO, w, h)
    el = to_xy(lm, ELB, w, h)
    wr = to_xy(lm, WRI, w, h)
    elbow = angle_3pt(sh, el, wr)
    full_top = elbow < 45
    return {"elbow_angle":elbow, "full_top":full_top}
