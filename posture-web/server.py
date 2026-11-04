"""Posture page: YOLO pose keypoints scored against a captured pose standard.

Runs locally or on a host such as Render. The browser owns the camera; this
process only scores uploaded frames.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

# OpenCV is only required for POSTURE_INFER=server (decode uploaded JPEGs).
try:
    import cv2
except ImportError:  # Free / client deploys omit opencv
    cv2 = None

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
SCORE_JS = ROOT / "score.js"
HOST = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))
# client = browser MediaPipe. server = YOLO pose on this process (Render quality path).
INFER_MODE = os.environ.get("POSTURE_INFER", "client").strip().lower()
if INFER_MODE not in {"client", "server"}:
    INFER_MODE = "client"
MODEL_NAME = os.environ.get("POSTURE_MODEL", "yolo26n-pose.pt").strip() or "yolo26n-pose.pt"
IMGSZ = max(256, min(640, int(os.environ.get("POSTURE_IMGSZ", "640" if INFER_MODE == "server" else "480"))))
MAX_UPLOAD = min(8 * 1024 * 1024, max(256 * 1024, int(os.environ.get("POSTURE_MAX_UPLOAD", str(4 * 1024 * 1024)))))


def _thread_count():
    """CPU threads for one pose forward pass. One request at a time, so use the whole instance."""
    raw = os.environ.get("POSTURE_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, os.cpu_count() or 1)


THREADS = _thread_count() if INFER_MODE == "server" else 1
if INFER_MODE == "server":
    # Set before torch is imported so OpenMP does not oversubscribe the instance.
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
TRACKED = (
    NOSE,
    L_EYE,
    R_EYE,
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
)
ARMS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}
HAND_SWITCH_MARGIN = 0.6
HAND_SWITCH_FRAMES = 8
HOLD_S = 0.3
ARM_HOLD_S = 0.45
GAP_S = 1.0
JUMP_WIDTH = 0.45
ARM_JUMP = 0.15
BONE_LO = 0.7
BONE_HI = 1.35
BONE_WINDOW = 15
BONE_READY = 5
ARM_DEAD_XY = 0.04
ARM_DEAD_DEG = 6.0
MIN_BONE = 0.18
MIN_EYE = 0.08
# Upper chest sits just below the shoulder midpoint, in shoulder-widths.
CHEST_DROP = 0.42

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
model = None
model_error = None
infer_lock = threading.Lock()
model_lock = threading.Lock()


class _Euro:
    """One Euro filter for one scalar. Still poses stay near 1 Hz; fast motion catches up."""

    def __init__(self):
        self.min_cutoff = 1.0
        self.beta = 0.02
        self.d_cutoff = 1.0
        self.hat = None
        self.speed = 0.0
        self.t = None

    def reset(self):
        self.hat = None
        self.speed = 0.0
        self.t = None

    def __call__(self, t, value):
        if self.hat is None or self.t is None:
            self.hat = value
            self.speed = 0.0
            self.t = t
            return value
        dt = t - self.t
        if dt <= 1e-6:
            return self.hat
        speed = (value - self.hat) / dt
        self.speed = self._blend(speed, self.speed, dt, self.d_cutoff)
        cutoff = self.min_cutoff + self.beta * abs(self.speed)
        self.hat = self._blend(value, self.hat, dt, cutoff)
        self.t = t
        return self.hat

    @staticmethod
    def _blend(value, prev, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        alpha = 1.0 / (1.0 + tau / dt)
        return alpha * value + (1.0 - alpha) * prev


class _PointTrack:
    """Smooth one keypoint, hold it briefly, and drop a single-frame jump."""

    def __init__(self, jump_width=JUMP_WIDTH):
        self.jump_width = jump_width
        self.fx = _Euro()
        self.fy = _Euro()
        self.xy = None
        self.conf = 0.0
        self.seen = None
        self.spike = None

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.xy = None
        self.conf = 0.0
        self.seen = None
        self.spike = None

    def snapshot(self):
        """Copy the accepted point so a bad arm frame can be rewound."""
        return {
            "xy": None if self.xy is None else (self.xy[0], self.xy[1]),
            "conf": self.conf,
            "seen": self.seen,
            "spike": None if self.spike is None else (self.spike[0], self.spike[1]),
            "fx": (self.fx.hat, self.fx.speed, self.fx.t),
            "fy": (self.fy.hat, self.fy.speed, self.fy.t),
        }

    def restore(self, snap):
        """Put back a snapshot taken before this frame was accepted."""
        self.xy = snap["xy"]
        self.conf = snap["conf"]
        self.seen = snap["seen"]
        self.spike = snap["spike"]
        self.fx.hat, self.fx.speed, self.fx.t = snap["fx"]
        self.fy.hat, self.fy.speed, self.fy.t = snap["fy"]

    def _holding(self, t):
        return self.xy is not None and self.seen is not None and t - self.seen <= HOLD_S

    def _jumped(self, x, y, width):
        if self.xy is None or width < 1e-6:
            return False
        return math.hypot(x - self.xy[0], y - self.xy[1]) > self.jump_width * width

    def _confirms(self, x, y, width):
        if self.spike is None or width < 1e-6:
            return False
        return math.hypot(x - self.spike[0], y - self.spike[1]) <= self.jump_width * width

    def update(self, t, x, y, conf, min_conf, width):
        if conf < min_conf:
            self.spike = None
            if self._holding(t):
                return self.xy[0], self.xy[1], self.conf
            return None
        if self._jumped(x, y, width) and not self._confirms(x, y, width):
            self.spike = (x, y)
            if self._holding(t):
                return self.xy[0], self.xy[1], self.conf
        self.spike = None
        self.xy = (self.fx(t, x), self.fy(t, y))
        self.conf = conf
        self.seen = t
        return self.xy[0], self.xy[1], self.conf


def _median(values):
    """Middle value of a short bone-length history. Empty history has no median."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bone_near(length, median):
    """True when this frame's bone still matches the recent trusted length."""
    if median is None:
        return True
    return BONE_LO * median <= length <= BONE_HI * median


def _push_bone(history, length):
    """Remember one trusted bone length and drop the oldest past the window."""
    history.append(length)
    if len(history) > BONE_WINDOW:
        del history[:-BONE_WINDOW]


class PoseTrack:
    """Single-user keypoint smoother. A gap longer than a second starts over."""

    def __init__(self):
        self.points = {
            index: _PointTrack(ARM_JUMP if index in {L_ELBOW, R_ELBOW, L_WRIST, R_WRIST} else JUMP_WIDTH)
            for index in TRACKED
        }
        self.t = None
        self.width = 0.0
        self.pending = None
        self.streak = 0
        self.bones = {name: {"upper": [], "fore": []} for name in ARMS}
        self.arm_hold = None

    def reset(self):
        for track in self.points.values():
            track.reset()
        self.t = None
        self.width = 0.0
        self.pending = None
        self.streak = 0
        self.bones = {name: {"upper": [], "fore": []} for name in ARMS}
        self.arm_hold = None

    def apply(self, kpts, min_conf, now):
        """Return keypoints with smoothed x, y. Unreliable points get confidence 0."""
        if self.t is not None and now - self.t > GAP_S:
            self.reset()
        self.t = now
        out = np.array(kpts, dtype=float, copy=True)
        width = self.width
        snaps = {index: self.points[index].snapshot() for index in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)}
        for index, track in self.points.items():
            x, y, conf = (float(v) for v in kpts[index])
            smoothed = track.update(now, x, y, conf, min_conf, width)
            if smoothed is None:
                out[index, 2] = 0.0
            else:
                out[index, 0], out[index, 1], out[index, 2] = smoothed
        self._gate_bones(out, min_conf, snaps)
        left = point(out, L_SHOULDER, min_conf)
        right = point(out, R_SHOULDER, min_conf)
        if left and right:
            dist = math.hypot(right[0] - left[0], right[1] - left[1])
            if dist >= 1e-6:
                self.width = dist
        return out

    def _gate_bones(self, out, min_conf, snaps):
        """Drop an elbow and wrist whose bone lengths leave the recent trusted range."""
        for name, (shoulder_i, elbow_i, wrist_i) in ARMS.items():
            shoulder = point(out, shoulder_i, min_conf)
            elbow = point(out, elbow_i, min_conf)
            wrist = point(out, wrist_i, min_conf)
            if shoulder is None or elbow is None or wrist is None:
                continue
            upper = segment(shoulder, elbow)
            fore = segment(elbow, wrist)
            hist = self.bones[name]
            ready = len(hist["upper"]) >= BONE_READY and len(hist["fore"]) >= BONE_READY
            bad = ready and (
                not _bone_near(upper, _median(hist["upper"])) or not _bone_near(fore, _median(hist["fore"]))
            )
            if bad and snaps[elbow_i]["xy"] is not None and snaps[wrist_i]["xy"] is not None:
                self.points[elbow_i].restore(snaps[elbow_i])
                self.points[wrist_i].restore(snaps[wrist_i])
                for index in (elbow_i, wrist_i):
                    xy = snaps[index]["xy"]
                    out[index, 0], out[index, 1], out[index, 2] = xy[0], xy[1], snaps[index]["conf"]
                continue
            _push_bone(hist["upper"], upper)
            _push_bone(hist["fore"], fore)

    def remember_arm(self, hand, pts):
        """Keep the last complete writing arm so one missing frame does not switch hands."""
        self.arm_hold = {"hand": hand, "pts": [tuple(p) for p in pts], "t": self.t}

    def held_arm(self, hand):
        """Return the remembered arm while it is still inside the short hold window."""
        hold = self.arm_hold
        if hold is None or hold["hand"] != hand or self.t is None or hold["t"] is None:
            return None
        if self.t - hold["t"] > ARM_HOLD_S:
            return None
        return hold["pts"]

    def lock_hand(self, prev_hand, best_name, best_score, prev_score):
        """Keep the current writing hand until another hand wins eight frames in a row."""
        if not prev_hand or best_name == prev_hand or prev_score is None:
            self.pending = None
            self.streak = 0
            return best_name
        if best_score < prev_score + HAND_SWITCH_MARGIN:
            self.pending = None
            self.streak = 0
            return prev_hand
        if self.pending == best_name:
            self.streak += 1
        else:
            self.pending = best_name
            self.streak = 1
        if self.streak >= HAND_SWITCH_FRAMES:
            self.pending = None
            self.streak = 0
            return best_name
        return prev_hand


TRACK = PoseTrack()


def reset_pose_track():
    """Clear smoother and hand-switch state. Used when a session gap is not enough."""
    TRACK.reset()


def load_model():
    """Load the YOLO pose model once and warm it. Only used when ``POSTURE_INFER=server``."""
    global model, model_error
    if INFER_MODE != "server":
        model_error = "server infer disabled"
        raise RuntimeError(model_error)
    import torch
    from ultralytics import YOLO

    if cv2 is None:
        raise RuntimeError("opencv is required for POSTURE_INFER=server")
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(1)
    try:
        loaded = YOLO(MODEL_NAME)
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        with torch.inference_mode():
            loaded.predict(dummy, verbose=False, conf=0.25, device="cpu", imgsz=IMGSZ)
        model = loaded
        model_error = None
        app.logger.info("pose model ready: %s imgsz=%s threads=%s", MODEL_NAME, IMGSZ, THREADS)
    except Exception as exc:
        model = None
        model_error = str(exc)
        raise


def ensure_model():
    """Load the YOLO model on first use so gunicorn workers do not need ``__main__``."""
    if INFER_MODE != "server":
        return False
    if model is not None:
        return True
    with model_lock:
        if model is not None:
            return True
        try:
            load_model()
        except Exception:
            app.logger.exception("model load failed")
            return False
    return model is not None


def form_float(name: str, default: float, lo: float, hi: float) -> float:
    """Read a clamped float from the multipart form."""
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(hi, max(lo, value))


def point(kpts: np.ndarray, index: int, min_conf: float):
    """Return (x, y, conf) when the keypoint clears the threshold."""
    x, y, conf = (float(v) for v in kpts[index])
    if conf < min_conf:
        return None
    return x, y, conf


def joint_angle(a, b, c):
    """Return the angle in degrees at point b, or None when a segment has no length."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    na = math.hypot(*ba)
    nc = math.hypot(*bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)))
    return math.degrees(math.acos(cos))


def fold_deg(angle: float) -> float:
    """Fold a degree value into -90..90 so a line has one signed tilt."""
    if angle > 90:
        return angle - 180
    if angle < -90:
        return angle + 180
    return angle


def shoulder_tilt(left, right) -> float:
    """Signed screen tilt of the shoulders. Positive means the screen-right shoulder is lower."""
    return fold_deg(math.degrees(math.atan2(right[1] - left[1], left[0] - right[0])))


def torso_lean(shoulder_mid, hip_mid) -> float:
    """Signed screen lean from vertical. Positive means the shoulders are screen-right of the hips."""
    return fold_deg(math.degrees(math.atan2(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])))


def body_axes(left, right):
    """Screen-right axis along a left-right segment, plus the downward perpendicular.

    Screen-right matches the mirrored preview: toward the person's right joint, which is
    decreasing image x when they face the camera. The second axis is the perpendicular that
    points downward, so a tilted shoulder line does not leak into lateral measurements.
    """
    span = math.hypot(right[0] - left[0], right[1] - left[1])
    if span < 1e-6:
        return None
    x_axis = ((right[0] - left[0]) / span, (right[1] - left[1]) / span)
    down_a = (-x_axis[1], x_axis[0])
    down_b = (x_axis[1], -x_axis[0])
    y_axis = down_a if down_a[1] >= down_b[1] else down_b
    return x_axis, y_axis, span


def project(origin, pt, x_axis, y_axis, scale):
    """Map a point into a shoulder frame. X is screen-right, Y is down, in units of scale."""
    dx = pt[0] - origin[0]
    dy = pt[1] - origin[1]
    return (
        (dx * x_axis[0] + dy * x_axis[1]) / scale,
        (dx * y_axis[0] + dy * y_axis[1]) / scale,
    )


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def segment(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def head_center(left_ear, right_ear, left_eye, right_eye):
    """Ear midpoint, or the eye midpoint when either ear is missing."""
    if left_ear and right_ear:
        return midpoint(left_ear, right_ear)
    if left_eye and right_eye:
        return midpoint(left_eye, right_eye)
    return None


def head_pitch(nose, left_eye, right_eye, shoulder_width):
    """Nose distance below the eye line, in eye-widths. Positive means the face is lower."""
    if shoulder_width < 1e-6:
        return None
    built = body_axes(left_eye, right_eye)
    if built is None or built[2] < MIN_EYE * shoulder_width:
        return None
    _x_axis, y_axis, eye_dist = built
    return project(midpoint(left_eye, right_eye), nose, built[0], y_axis, eye_dist)[1]


def hips_reliable(left_hip, right_hip, shoulder_mid, x_axis, y_axis, shoulder_width, frame_w, frame_h):
    """Accept hips only when both are inside the frame and shaped like a seated pelvis."""
    if frame_w <= 0 or frame_h <= 0 or shoulder_width < 1e-6:
        return False
    margin = max(2.0, 0.02 * min(frame_w, frame_h))
    for hip in (left_hip, right_hip):
        if hip[0] < margin or hip[1] < margin or hip[0] > frame_w - margin or hip[1] > frame_h - margin:
            return False
    ratio = segment(left_hip, right_hip) / shoulder_width
    if ratio < 0.55 or ratio > 1.35:
        return False
    below = project(shoulder_mid, midpoint(left_hip, right_hip), x_axis, y_axis, shoulder_width)[1]
    return below >= 0.3


def forearm_angle(elbow, wrist, x_axis, y_axis) -> float | None:
    """Signed forearm tilt in the shoulder frame, in -180..180.

    Zero is horizontal toward screen-right. Positive means the wrist is lower than the elbow.
    """
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    fx = dx * x_axis[0] + dy * x_axis[1]
    fy = dx * y_axis[0] + dy * y_axis[1]
    if math.hypot(fx, fy) < 1e-6:
        return None
    return math.degrees(math.atan2(fy, fx))


def angle_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two degrees, at most 180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def flatness(angle: float) -> float:
    """How close a forearm angle is to horizontal. 0 is level, 90 is vertical."""
    flat = abs(angle)
    if flat > 90:
        flat = 180 - flat
    return flat


def writing_score(shoulder, elbow, wrist, x_axis, y_axis) -> float | None:
    """Score a desk-writing arm. Horizontal forearm and an elbow near 100 degrees rank higher."""
    elbow_deg = joint_angle(shoulder, elbow, wrist)
    lean = forearm_angle(elbow, wrist, x_axis, y_axis)
    if elbow_deg is None or lean is None:
        return None
    horizontal = 1.0 - flatness(lean) / 90.0
    bent = max(0.0, 1.0 - abs(elbow_deg - 100.0) / 80.0)
    return horizontal + bent


def cross_penalty(name: str, wrist_x: float, left_x: float, right_x: float) -> float:
    """Penalize a wrist that has reached past the opposite shoulder.

    A writing hand normally stays on its own side of the body, including out past its own
    shoulder. Image x increases toward screen-left.
    """
    span = abs(right_x - left_x)
    if span < 1e-6:
        return 0.0
    if name == "right":
        crossed = (wrist_x - max(left_x, right_x)) / span
    else:
        crossed = (min(left_x, right_x) - wrist_x) / span
    return max(0.0, crossed - 0.15)


def choose_arm(kpts: np.ndarray, hand: str, min_conf: float, prev_hand: str | None):
    """Pick the writing arm. Auto keeps the previous hand until another wins eight frames."""
    found = {}
    for name, indices in ARMS.items():
        pts = [point(kpts, i, min_conf) for i in indices]
        if all(pts):
            found[name] = pts
    if hand in ARMS:
        TRACK.pending = None
        TRACK.streak = 0
        if hand in found:
            TRACK.remember_arm(hand, found[hand])
            return hand, found[hand]
        held = TRACK.held_arm(hand)
        return (hand, held) if held else (None, None)
    if not found:
        held = TRACK.held_arm(prev_hand) if prev_hand else None
        return (prev_hand, held) if held else (None, None)

    left_s = point(kpts, L_SHOULDER, min_conf)
    right_s = point(kpts, R_SHOULDER, min_conf)
    x_axis, y_axis = (-1.0, 0.0), (0.0, 1.0)
    if left_s and right_s:
        built = body_axes(left_s, right_s)
        if built:
            x_axis, y_axis = built[0], built[1]

    scored = []
    for name, pts in found.items():
        score = writing_score(*pts, x_axis, y_axis)
        if score is None:
            continue
        if left_s and right_s:
            score -= cross_penalty(name, pts[2][0], left_s[0], right_s[0])
        if pts[2][1] > pts[0][1]:
            score += 0.15
        scored.append({"name": name, "pts": pts, "score": score})
    if not scored:
        return None, None

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    if not prev_hand:
        TRACK.pending = None
        TRACK.streak = 0
        TRACK.remember_arm(best["name"], best["pts"])
        return best["name"], best["pts"]
    current = next((item for item in scored if item["name"] == prev_hand), None)
    if current is None:
        held = TRACK.held_arm(prev_hand)
        if held:
            return prev_hand, held
        TRACK.pending = None
        TRACK.streak = 0
        TRACK.remember_arm(best["name"], best["pts"])
        return best["name"], best["pts"]
    chosen = TRACK.lock_hand(prev_hand, best["name"], best["score"], current["score"])
    picked = next(item for item in scored if item["name"] == chosen)
    TRACK.remember_arm(picked["name"], picked["pts"])
    return picked["name"], picked["pts"]


def toward_midline(hand: str | None, wrist_x: float, target_x: float) -> bool:
    """True when the writing wrist has moved at least a little toward the chest.

    Positive X is the person's right. The right hand reaches the chest by moving left,
    and the left hand reaches it by moving right.
    """
    if hand == "right":
        return target_x - wrist_x >= 0.02
    if hand == "left":
        return wrist_x - target_x >= 0.02
    return False


def empty_sit() -> dict:
    """Sitting verdict before any reliable torso points are available."""
    return {
        "status": "unknown",
        "shoulder_deg": None,
        "torso_deg": None,
        "head_ratio": None,
        "head_drop": None,
        "chest_gap": None,
    }


def empty_pen() -> dict:
    """Arm verdict before a writing arm is available."""
    return {"status": "unknown", "scope": "arm", "hand": None, "elbow_deg": None, "forearm_deg": None}


def empty_levels() -> dict:
    """Per-part closeness. Unknown stays neutral instead of a failure color."""
    return {"shoulder": "unknown", "torso": "unknown", "head": "unknown", "chest": "unknown", "arm": "unknown"}


def _soften(offset: float, dead: float) -> float:
    """Ignore arm motion smaller than the detector noise before applying a limit."""
    return max(0.0, offset - dead)


def band_tol(offset: float, limit: float) -> str:
    """Inside the tolerance, within 20% past it, or clearly past it."""
    if offset <= limit:
        return "ok"
    if offset <= limit * 1.2:
        return "near"
    return "bad"


def worst_level(*levels: str) -> str:
    """The least correct known level. Unknown is ignored unless every input is unknown."""
    rank = {"ok": 0, "near": 1, "bad": 2}
    known = [level for level in levels if level in rank]
    if not known:
        return "unknown"
    return max(known, key=rank.get)


def judge(kpts: np.ndarray, limits: dict, prev_hand: str | None = None, frame_w: int = 0, frame_h: int = 0) -> dict:
    """Score sitting posture and the writing arm against the posted pose standard."""
    min_conf = limits["kpt_conf"]
    nose = point(kpts, NOSE, min_conf)
    left_eye = point(kpts, L_EYE, min_conf)
    right_eye = point(kpts, R_EYE, min_conf)
    left_ear = point(kpts, L_EAR, min_conf)
    right_ear = point(kpts, R_EAR, min_conf)
    left_shoulder = point(kpts, L_SHOULDER, min_conf)
    right_shoulder = point(kpts, R_SHOULDER, min_conf)
    left_hip = point(kpts, L_HIP, min_conf)
    right_hip = point(kpts, R_HIP, min_conf)
    sit = empty_sit()
    levels = empty_levels()
    shoulder_width = 0.0
    x_axis, y_axis = (-1.0, 0.0), (0.0, 1.0)
    if left_shoulder and right_shoulder:
        built = body_axes(left_shoulder, right_shoulder)
        if built:
            x_axis, y_axis, shoulder_width = built
            mid_s = midpoint(left_shoulder, right_shoulder)
            shoulder_deg = shoulder_tilt(left_shoulder, right_shoulder)
            center = head_center(left_ear, right_ear, left_eye, right_eye)
            head_x = project(mid_s, center, x_axis, y_axis, shoulder_width)[0] if center else None
            pitch = None
            if nose and left_eye and right_eye:
                pitch = head_pitch(nose, left_eye, right_eye, shoulder_width)
            levels["shoulder"] = band_tol(abs(shoulder_deg - limits["shoulder_target_deg"]), limits["shoulder_max_deg"])
            head_levels = []
            if head_x is not None:
                head_levels.append(band_tol(abs(head_x - limits["head_target_x"]), limits["head_max_ratio"]))
            if pitch is not None:
                head_levels.append(band_tol(abs(pitch - limits["head_target_drop"]), limits["head_drop_tol"]))
            if head_levels:
                levels["head"] = worst_level(*head_levels)
            chest_gap = None
            if nose and shoulder_width >= 1e-6:
                nose_y = project(mid_s, nose, x_axis, y_axis, shoulder_width)[1]
                chest_gap = CHEST_DROP - nose_y
                levels["chest"] = band_tol(max(0.0, limits["chest_gap"] - chest_gap), limits["chest_gap_tol"])
            torso_deg = None
            if left_hip and right_hip and hips_reliable(
                left_hip, right_hip, mid_s, x_axis, y_axis, shoulder_width, frame_w, frame_h
            ):
                torso_deg = torso_lean(mid_s, midpoint(left_hip, right_hip))
                levels["torso"] = band_tol(abs(torso_deg - limits["torso_target_deg"]), limits["torso_max_deg"])
            sit = {
                "status": worst_level(levels["shoulder"], levels["torso"], levels["head"], levels["chest"]),
                "shoulder_deg": round(shoulder_deg, 1),
                "torso_deg": None if torso_deg is None else round(torso_deg, 1),
                "head_ratio": None if head_x is None else round(head_x, 2),
                "head_drop": None if pitch is None else round(pitch, 2),
                "chest_gap": None if chest_gap is None else round(chest_gap, 2),
            }

    hand, arm = choose_arm(kpts, limits["hand"], min_conf, prev_hand)
    pen = empty_pen()
    pen["hand"] = hand
    if arm is not None:
        shoulder, elbow, wrist = arm
        elbow_deg = joint_angle(shoulder, elbow, wrist)
        lean = forearm_angle(elbow, wrist, x_axis, y_axis)
        wrist_x = wrist_y = None
        if shoulder_width >= 1e-6:
            if segment(shoulder, elbow) < MIN_BONE * shoulder_width or segment(elbow, wrist) < MIN_BONE * shoulder_width:
                elbow_deg = None
                lean = None
            wrist_x, wrist_y = project(shoulder, wrist, x_axis, y_axis, shoulder_width)
        inward = wrist_x is not None and toward_midline(hand, wrist_x, limits["wrist_target_x"])
        if inward:
            levels["arm"] = "ok"
        else:
            checks = []
            if wrist_x is not None:
                dx = wrist_x - limits["wrist_target_x"]
                dy = wrist_y - limits["wrist_target_y"]
                checks = [
                    band_tol(_soften(max(0.0, -dy), ARM_DEAD_XY), limits["wrist_up"]),
                    band_tol(_soften(max(0.0, dy), ARM_DEAD_XY), limits["wrist_down"]),
                    band_tol(_soften(max(0.0, -dx), ARM_DEAD_XY), limits["wrist_left"]),
                    band_tol(_soften(max(0.0, dx), ARM_DEAD_XY), limits["wrist_right"]),
                ]
            if elbow_deg is not None:
                checks.append(
                    band_tol(_soften(abs(elbow_deg - limits["elbow_target_deg"]), ARM_DEAD_DEG), limits["elbow_tol_deg"])
                )
            if lean is not None:
                checks.append(
                    band_tol(
                        _soften(angle_delta(lean, limits["forearm_target_deg"]), ARM_DEAD_DEG),
                        limits["forearm_tol_deg"],
                    )
                )
            if checks:
                levels["arm"] = worst_level(*checks)
        pen = {
            "status": levels["arm"],
            "scope": "arm",
            "hand": hand,
            "elbow_deg": None if elbow_deg is None else round(elbow_deg, 1),
            "forearm_deg": None if lean is None else round(lean, 1),
            "wrist_x": None if wrist_x is None else round(wrist_x, 2),
            "wrist_y": None if wrist_y is None else round(wrist_y, 2),
        }
    return {"sit": sit, "pen": pen, "levels": levels}


def largest_person(result) -> np.ndarray | None:
    """Return the (17, 3) keypoints of the largest detected person."""
    keypoints = result.keypoints
    if keypoints is None or keypoints.conf is None:
        return None
    xy = keypoints.xy
    conf = keypoints.conf
    if xy is None or len(xy) == 0:
        return None
    xy = xy.cpu().numpy()
    conf = conf.cpu().numpy()
    index = 0
    boxes = result.boxes
    if boxes is not None and len(boxes) == len(xy):
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        index = int(np.argmax(areas))
    return np.concatenate([xy[index], conf[index, :, None]], axis=1)


def limits_from_form() -> dict:
    """Standards and tolerances posted with each frame."""
    hand = request.form.get("hand", "auto")
    if hand not in {"left", "right", "auto"}:
        hand = "auto"
    return {
        "kpt_conf": form_float("kpt_conf", 0.5, 0.05, 0.95),
        "shoulder_max_deg": form_float("shoulder_max_deg", 12, 1, 45),
        "torso_max_deg": form_float("torso_max_deg", 15, 1, 45),
        "head_max_ratio": form_float("head_max_ratio", 0.35, 0.05, 1),
        "head_drop_tol": form_float("head_drop_tol", 0.35, 0.02, 1),
        "elbow_target_deg": form_float("elbow_target_deg", 105, 0, 180),
        "elbow_tol_deg": form_float("elbow_tol_deg", 35, 1, 90),
        "forearm_target_deg": form_float("forearm_target_deg", 0, -180, 180),
        "forearm_tol_deg": form_float("forearm_tol_deg", 25, 1, 90),
        "shoulder_target_deg": form_float("shoulder_target_deg", 0, -80, 80),
        "torso_target_deg": form_float("torso_target_deg", 0, -80, 80),
        "head_target_x": form_float("head_target_x", 0, -1, 1),
        "head_target_drop": form_float("head_target_drop", 0.7, -0.2, 2),
        "chest_gap": form_float("chest_gap", 0.9, 0.2, 1.8),
        "chest_gap_tol": form_float("chest_gap_tol", 0.12, 0.02, 0.6),
        "wrist_target_x": form_float("wrist_target_x", 0, -2, 2.5),
        "wrist_target_y": form_float("wrist_target_y", 0, -2, 2.5),
        "wrist_up": form_float("wrist_up", 0, 0, 2.5),
        "wrist_down": form_float("wrist_down", 1.5, 0, 2.5),
        "wrist_left": form_float("wrist_left", 1, 0, 2.5),
        "wrist_right": form_float("wrist_right", 1, 0, 2.5),
        "hand": hand,
        "prev_hand": request.form.get("prev_hand") if request.form.get("prev_hand") in {"left", "right"} else None,
    }


def blank(width: int, height: int) -> dict:
    """Response used when nobody is detected."""
    return {
        "person": False,
        "width": width,
        "height": height,
        "keypoints": [],
        "sit": empty_sit(),
        "pen": empty_pen(),
        "levels": empty_levels(),
    }


@app.get("/")
def index():
    """Serve the camera page. Avoid a stale page after the local file changes."""
    response = send_file(INDEX)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/score.js")
def score_js():
    """Browser-side scoring used when ``POSTURE_INFER=client``."""
    response = send_file(SCORE_JS, mimetype="application/javascript; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/config")
def config():
    """Tell the page whether to score in-browser or POST frames to ``/infer``."""
    return jsonify({"infer": INFER_MODE, "imgsz": IMGSZ, "model": MODEL_NAME if INFER_MODE == "server" else ""})


@app.get("/healthz")
def healthz():
    """Liveness for Render. Client mode is ready immediately; server waits for weights."""
    ready = True if INFER_MODE == "client" else model is not None
    body = {"ok": True, "ready": ready, "infer": INFER_MODE, "imgsz": IMGSZ}
    if model_error and INFER_MODE == "server" and not ready:
        body["error"] = model_error
    return jsonify(body), 200 if ready else 503


@app.post("/infer")
def infer():
    """Run pose inference and return sitting plus arm-angle verdicts."""
    if INFER_MODE != "server":
        return jsonify({"error": "server infer disabled; use client MediaPipe", "skipped": True}), 400
    if not ensure_model():
        return jsonify({"error": "model unavailable", "skipped": True}), 503
    if cv2 is None:
        return jsonify({"error": "opencv unavailable", "skipped": True}), 503
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "image required"}), 400
    raw = upload.read()
    if not raw:
        return jsonify({"error": "empty or oversized image"}), 400
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        skipped = blank(0, 0)
        skipped["skipped"] = True
        return jsonify(skipped)

    height, width = image.shape[:2]
    if not infer_lock.acquire(blocking=False):
        skipped = blank(width, height)
        skipped["skipped"] = True
        return jsonify(skipped)
    try:
        import torch

        with torch.inference_mode():
            result = model.predict(image, verbose=False, conf=0.25, device="cpu", imgsz=IMGSZ)[0]
    except Exception:
        app.logger.exception("infer failed")
        skipped = blank(width, height)
        skipped["skipped"] = True
        return jsonify(skipped)
    finally:
        infer_lock.release()
    try:
        kpts = largest_person(result)
    except Exception:
        app.logger.exception("infer failed")
        skipped = blank(width, height)
        skipped["skipped"] = True
        return jsonify(skipped)
    if kpts is None:
        return jsonify(blank(width, height))

    payload = blank(width, height)
    payload["person"] = True
    try:
        limits = limits_from_form()
        kpts = TRACK.apply(kpts, limits["kpt_conf"], time.monotonic())
        payload["keypoints"] = [[round(float(v), 1) for v in row] for row in kpts]
        payload.update(judge(kpts, limits, limits["prev_hand"], width, height))
    except Exception:
        app.logger.exception("infer failed")
        skipped = blank(width, height)
        skipped["skipped"] = True
        return jsonify(skipped)
    return jsonify(payload)


def _quiet_access(record: logging.LogRecord) -> bool:
    """Hide per-request access lines so a selected Windows console cannot stall inference."""
    return "HTTP/" not in record.getMessage()


def preload_model():
    """Warm YOLO weights when gunicorn imports this module in server mode."""
    if INFER_MODE != "server" or os.environ.get("POSTURE_PRELOAD", "1") == "0":
        return
    try:
        ensure_model()
    except Exception:
        app.logger.exception("model preload failed")


if __name__ != "__main__":
    preload_model()


if __name__ == "__main__":
    logging.getLogger("werkzeug").addFilter(_quiet_access)
    if INFER_MODE == "server":
        load_model()
    else:
        app.logger.info("POSTURE_INFER=client; browser MediaPipe scores poses")
    app.run(host=HOST, port=PORT, threaded=True)
