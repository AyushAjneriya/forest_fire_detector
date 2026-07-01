import sys
import subprocess
import os

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-INSTALL
# ─────────────────────────────────────────────────────────────────────────────
_PACKAGES = [
    ("ultralytics",  "ultralytics"),
    ("opencv-python","cv2"),
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("numpy",        "numpy"),
    ("pillow",       "PIL"),
    ("matplotlib",   "matplotlib"),
    ("scipy",        "scipy"),
]

print("\n" + "=" * 66)
print("  FOREST FIRE EARLY DETECTION SYSTEM  |  Dependency Check")
print("=" * 66)
for pkg, imp in _PACKAGES:
    try:
        __import__(imp)
        print(f"  [OK]       {pkg}")
    except ImportError:
        print(f"  [INSTALL]  {pkg} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q",
             "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"  [DONE]     {pkg}")
print("=" * 66 + "\n")

import time
import datetime
import math
import warnings
import collections
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

os.makedirs("screenshots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_SIMPLEX
OUTPUT_PATH = "output_fire_ai.mp4"
MAX_W       = 1280

# Fire & smoke COCO-proxy classes (YOLO detects persons/vehicles → we remap visuals;
# true fire/smoke needs a fire model — we layer color-analysis on top)
FIRE_COCO_PROXY = {"fire", "smoke"}
PERSON_CLASSES  = {"person"}
VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle"}

# Fire color ranges in HSV (BGR→HSV)
FIRE_LOWER1 = np.array([0,   150,  150], dtype=np.uint8)
FIRE_UPPER1 = np.array([18,  255,  255], dtype=np.uint8)
FIRE_LOWER2 = np.array([160, 150,  150], dtype=np.uint8)
FIRE_UPPER2 = np.array([180, 255,  255], dtype=np.uint8)

# Smoke color range
SMOKE_LOWER = np.array([0,   0,   140], dtype=np.uint8)
SMOKE_UPPER = np.array([180, 40,  220], dtype=np.uint8)

# Alert thresholds
FIRE_PIXEL_THRESH  = 800    # px² before triggering fire alert
SMOKE_PIXEL_THRESH = 2000

# Colors (BGR)
C_RED    = (30,  30, 220)
C_ORANGE = (20, 140, 255)
C_YELLOW = ( 0, 220, 255)
C_GREEN  = (50, 220,  50)
C_CYAN   = (220,220,  30)
C_WHITE  = (240,240, 240)
C_GRAY   = (140,140, 140)
C_DARK   = ( 12, 12,  12)
C_TEAL   = (200,200,  30)

RISK_COLORS = {
    "CRITICAL": C_RED,
    "HIGH":     C_ORANGE,
    "MODERATE": C_YELLOW,
    "LOW":      C_GREEN,
    "CLEAR":    C_CYAN,
}

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"

# ─────────────────────────────────────────────────────────────────────────────
#  FIRE & SMOKE COLOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_fire_smoke(frame):
    """
    Returns:
        fire_mask   (H,W) uint8
        smoke_mask  (H,W) uint8
        fire_px     int
        smoke_px    int
        fire_zones  list of (cx,cy,radius)
        thermal_map (H,W) float32  0-1
    """
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire detection (red/orange hues + high value)
    f1 = cv2.inRange(hsv, FIRE_LOWER1, FIRE_UPPER1)
    f2 = cv2.inRange(hsv, FIRE_LOWER2, FIRE_UPPER2)
    fire_raw  = cv2.bitwise_or(f1, f2)

    # Also require high brightness for true fire
    bright    = (hsv[:,:,2] > 160).astype(np.uint8) * 255
    fire_mask = cv2.bitwise_and(fire_raw, bright)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,
                                  np.ones((5,5), np.uint8))
    fire_mask = cv2.dilate(fire_mask, np.ones((7,7), np.uint8), iterations=2)

    # Smoke detection (gray/white desaturated regions in upper frame)
    smoke_raw  = cv2.inRange(hsv, SMOKE_LOWER, SMOKE_UPPER)
    # Smoke tends to be in upper half + has texture
    upper_mask = np.zeros((H,W), np.uint8)
    upper_mask[:H*2//3, :] = 255
    smoke_mask = cv2.bitwise_and(smoke_raw, upper_mask)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN,
                                   np.ones((9,9), np.uint8))

    fire_px  = int(np.sum(fire_mask  > 0))
    smoke_px = int(np.sum(smoke_mask > 0))

    # Find fire zone contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    fire_zones  = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 200:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            fire_zones.append((int(cx), int(cy), int(r)))

    # Thermal heatmap: orange/red channel dominance
    b, g, r_ch = frame[:,:,0].astype(np.float32), \
                 frame[:,:,1].astype(np.float32), \
                 frame[:,:,2].astype(np.float32)
    thermal = np.clip((r_ch * 0.6 + g * 0.3 - b * 0.5) / 255.0, 0, 1)
    thermal = gaussian_filter(thermal, sigma=12)

    return fire_mask, smoke_mask, fire_px, smoke_px, fire_zones, thermal


def fire_risk_level(fire_px, smoke_px, fire_zones):
    n_zones = len(fire_zones)
    if fire_px > 8000 or n_zones > 3:
        return "CRITICAL"
    if fire_px > 3000 or n_zones > 1:
        return "HIGH"
    if fire_px > FIRE_PIXEL_THRESH or smoke_px > SMOKE_PIXEL_THRESH:
        return "MODERATE"
    if fire_px > 200 or smoke_px > 500:
        return "LOW"
    return "CLEAR"

# ─────────────────────────────────────────────────────────────────────────────
#  MOTION-BASED SMOKE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class MotionSmokeTracker:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self.smoke_trail = collections.deque(maxlen=60)

    def update(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                               np.ones((5,5), np.uint8))
        # Upper region motion = likely smoke
        H = frame.shape[0]
        upper_fg = fg.copy()
        upper_fg[H//2:, :] = 0
        smoke_motion = int(np.sum(upper_fg > 0))
        self.smoke_trail.append(smoke_motion)
        avg_motion = np.mean(self.smoke_trail)
        return fg, smoke_motion, avg_motion

# ─────────────────────────────────────────────────────────────────────────────
#  THERMAL OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def apply_thermal_overlay(frame, thermal_map, alpha=0.38):
    thermal_u8 = (thermal_map * 255).astype(np.uint8)
    thermal_colored = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(frame, 1.0 - alpha, thermal_colored, alpha, 0)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  FIRE ZONE DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_fire_zones(frame, fire_zones, smoke_mask, risk_level):
    H, W = frame.shape[:2]

    # Smoke overlay (blue-gray tint)
    if np.any(smoke_mask > 0):
        smoke_colored = np.zeros_like(frame)
        smoke_colored[smoke_mask > 0] = [180, 180, 160]
        frame = cv2.addWeighted(frame, 1.0, smoke_colored, 0.28, 0)
        # Smoke contours
        sc, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for c in sc:
            if cv2.contourArea(c) > 500:
                cv2.drawContours(frame, [c], -1, (160, 160, 140), 1)

    # Fire zone circles with expanding rings
    for i, (cx, cy, r) in enumerate(fire_zones):
        # Outer pulsing ring
        outer_r = r + 20 + (i * 5)
        cv2.circle(frame, (cx, cy), outer_r, C_RED,    1)
        cv2.circle(frame, (cx, cy), outer_r+8, C_ORANGE, 1)
        # Core
        cv2.circle(frame, (cx, cy), r,     C_ORANGE, 2)
        cv2.circle(frame, (cx, cy), r//2,  C_YELLOW,  2)
        # Crosshair
        cv2.line(frame, (cx-outer_r-10, cy), (cx-r-4, cy), C_RED, 1)
        cv2.line(frame, (cx+r+4, cy), (cx+outer_r+10, cy), C_RED, 1)
        cv2.line(frame, (cx, cy-outer_r-10), (cx, cy-r-4), C_RED, 1)
        cv2.line(frame, (cx, cy+r+4), (cx, cy+outer_r+10), C_RED, 1)
        # Label
        lbl = f"FIRE ZONE {i+1}  r={r}px"
        cv2.putText(frame, lbl, (cx+r+6, cy-8), FONT_MONO, 0.38,
                    C_ORANGE, 1, cv2.LINE_AA)
        # Spread arrow (upward — fire/smoke rises)
        cv2.arrowedLine(frame, (cx, cy-r),
                        (cx, max(cy-r-40, 4)),
                        C_YELLOW, 2, tipLength=0.35)
        cv2.putText(frame, "SPREAD", (cx+4, max(cy-r-20, 14)),
                    FONT_MONO, 0.30, C_YELLOW, 1, cv2.LINE_AA)

    return frame

# ─────────────────────────────────────────────────────────────────────────────
#  HEATMAP ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class HeatmapAccumulator:
    def __init__(self, H, W):
        self.map = np.zeros((H, W), dtype=np.float32)

    def update(self, fire_mask, decay=0.97):
        self.map *= decay
        self.map += (fire_mask > 0).astype(np.float32) * 0.8
        self.map  = np.clip(self.map, 0, 1)

    def render(self, frame, alpha=0.42):
        h_u8     = (self.map * 255).astype(np.uint8)
        h_color  = cv2.applyColorMap(h_u8, cv2.COLORMAP_HOT)
        mask3    = np.stack([self.map > 0.05]*3, axis=2)
        blended  = frame.copy().astype(np.float32)
        blended[mask3] = (frame.astype(np.float32)[mask3] * (1-alpha)
                          + h_color.astype(np.float32)[mask3] * alpha)
        return blended.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  RISK SCORE MINI-PANEL
# ─────────────────────────────────────────────────────────────────────────────
class RiskScorePanel:
    def __init__(self):
        self.history = collections.deque(maxlen=80)

    def update(self, score):
        self.history.append(score)

    def render(self, frame, risk_level, fire_px, smoke_px, motion_avg):
        H, W = frame.shape[:2]
        pw = 230; ph = 148
        px = W - pw - 14; py = 62

        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px+pw, py+ph), C_DARK, -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

        rc = RISK_COLORS.get(risk_level, C_CYAN)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), rc, 1)

        def t(txt, pos, sc, col):
            x,y = int(pos[0]), int(pos[1])
            cv2.putText(frame, txt, (x+1,y+1), FONT_MONO, sc, (0,0,0),
                        2, cv2.LINE_AA)
            cv2.putText(frame, txt, (x,y),   FONT_MONO, sc, col,
                        1, cv2.LINE_AA)

        t(f"FIRE RISK: {risk_level}", (px+6, py+18), 0.46, rc)
        t(f"Fire pixels : {fire_px:6d}", (px+6, py+36), 0.38, C_WHITE)
        t(f"Smoke pixels: {smoke_px:6d}", (px+6, py+52), 0.38, C_WHITE)
        t(f"Motion (avg): {motion_avg:6.0f}", (px+6, py+68), 0.38, C_WHITE)

        # Mini sparkline
        if len(self.history) > 1:
            hist = list(self.history)
            max_v = max(max(hist), 1)
            pts   = []
            for i, v in enumerate(hist):
                gx = px + 6 + int(i / len(hist) * (pw-12))
                gy = py + ph - 6 - int(v / max_v * 40)
                pts.append((gx, gy))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], rc, 1)

        t("RISK TREND", (px+6, py+ph-8), 0.30, C_GRAY)

# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────
def _t(img, text, pos, scale, color, thickness=1, font=FONT_MONO):
    x,y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x+1,y+1), font, scale, (0,0,0),
                thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),   font, scale, color,
                thickness, cv2.LINE_AA)


def draw_top_hud(frame, fps, frame_idx, total, inf_ms, device_lbl, risk_level, ts):
    H, W = frame.shape[:2]
    bh = 54
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,bh), C_DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    # Accent
    rc = RISK_COLORS.get(risk_level, C_CYAN)
    cv2.line(frame, (0,bh), (W,bh), rc, 2)

    _t(frame, "FOREST FIRE EARLY DETECTION SYSTEM", (14,32), 0.66,
       (30, 200, 255), 1)

    r = W - 500
    _t(frame, f"FPS {fps:5.1f}", (r,    30), 0.50, C_GREEN)
    _t(frame, f"|  {inf_ms:.0f}ms",    (r+110, 30), 0.50, C_GRAY)
    _t(frame, f"|  {device_lbl}",       (r+210, 30), 0.50, C_YELLOW)
    _t(frame, f"|  {ts}",               (r+360, 30), 0.50, C_GRAY)


def draw_bottom_hud(frame, frame_idx, total, risk_level, n_zones, smoke_active):
    H, W = frame.shape[:2]
    bh = 42; y0 = H - bh
    ov = frame.copy()
    cv2.rectangle(ov, (0,y0), (W,H), C_DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    rc = RISK_COLORS.get(risk_level, C_CYAN)
    cv2.line(frame, (0,y0), (W,y0), rc, 1)

    if total > 0:
        pw = int(frame_idx/max(total,1)*(W-28))
        cv2.rectangle(frame, (14,y0+7), (14+pw,y0+12), rc, -1)
    cv2.rectangle(frame, (14,y0+7), (W-14,y0+12), (50,50,50), 1)

    _t(frame, f"FRAME {frame_idx:05d}/{total:05d}", (14, y0+30), 0.40, C_GRAY)
    _t(frame, f"FIRE ZONES: {n_zones}", (W//2-80, y0+30), 0.40,
       C_ORANGE if n_zones else C_GREEN)
    _t(frame, f"SMOKE: {'ACTIVE' if smoke_active else 'CLEAR'}",
       (W//2+80, y0+30), 0.40,
       C_YELLOW if smoke_active else C_GREEN)
    rc2 = RISK_COLORS.get(risk_level, C_CYAN)
    _t(frame, f"RISK: {risk_level}", (W-180, y0+30), 0.40, rc2)

    tags = ["Fire Analysis  ", "Thermal Vision  ",
            "Smoke Detection  ", "Motion Tracking  ", "Hazard AI"]
    tx = 14
    for tag in tags:
        cv2.putText(frame, tag, (tx, y0-6), FONT_MONO, 0.28,
                    (0, 160, 160), 1, cv2.LINE_AA)
        tw,_ = cv2.getTextSize(tag, FONT_MONO, 0.28, 1)
        tx += tw[0]+6


def draw_alert_banner(frame, risk_level):
    if risk_level not in ("CRITICAL","HIGH"):
        return
    H, W = frame.shape[:2]
    rc = RISK_COLORS[risk_level]
    text = ("!! CRITICAL FIRE DETECTED — EMERGENCY ALERT !!"
            if risk_level == "CRITICAL"
            else "!! HIGH FIRE RISK — MONITOR IMMEDIATELY !!")
    # Flashing border
    cv2.rectangle(frame, (2,2), (W-2, H-2), rc, 3)
    # Banner
    bw = W - 28; bh = 34; bx = 14; by = 58
    ov = frame.copy()
    cv2.rectangle(ov, (bx,by), (bx+bw,by+bh), rc, -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    tw,_ = cv2.getTextSize(text, FONT_MONO, 0.55, 1)
    tx = bx + (bw - tw[0])//2
    cv2.putText(frame, text, (tx+1,by+22), FONT_MONO, 0.55,
                (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,by+22),   FONT_MONO, 0.55,
                C_WHITE, 1, cv2.LINE_AA)


def draw_legend(frame):
    H, W = frame.shape[:2]
    items = [
        (C_RED,    "Critical fire zone"),
        (C_ORANGE, "Active fire"),
        (C_YELLOW, "Fire spread"),
        ((160,160,140), "Smoke region"),
        (C_CYAN,   "Clear / Monitoring"),
    ]
    px = 14; py = 62
    pw = 195; ph = len(items)*22 + 12
    ov = frame.copy()
    cv2.rectangle(ov, (px,py), (px+pw,py+ph), C_DARK, -1)
    cv2.addWeighted(ov, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (px,py), (px+pw,py+ph), (0,160,160), 1)
    _t(frame, "LEGEND", (px+6,py+12), 0.34, C_CYAN)
    for i,(col,lbl) in enumerate(items):
        cy2 = py + 20 + i*22
        cv2.rectangle(frame, (px+6, cy2), (px+20, cy2+12), col, -1)
        _t(frame, lbl, (px+26, cy2+11), 0.33, C_WHITE)


def vignette(frame, strength=0.30):
    H,W = frame.shape[:2]
    cx,cy = W/2,H/2
    Y,X   = np.ogrid[:H,:W]
    d     = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    v     = np.clip(1.0 - d*strength, 0.55, 1.0).astype(np.float32)
    return np.clip(frame.astype(np.float32)*v[:,:,np.newaxis],0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  MINI THERMAL INSET
# ─────────────────────────────────────────────────────────────────────────────
def embed_thermal_inset(frame, thermal_map):
    H,W = frame.shape[:2]
    iw = 240; ih = 135
    t_u8 = (thermal_map*255).astype(np.uint8)
    t_col = cv2.applyColorMap(t_u8, cv2.COLORMAP_INFERNO)
    t_small = cv2.resize(t_col, (iw, ih))
    x0 = W - iw - 14; y0 = 62
    ov = frame.copy()
    cv2.rectangle(ov, (x0-2,y0-2),(x0+iw+2,y0+ih+2), C_DARK, -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    frame[y0:y0+ih, x0:x0+iw] = cv2.addWeighted(
        frame[y0:y0+ih, x0:x0+iw], 0.20, t_small, 0.80, 0
    )
    cv2.rectangle(frame, (x0-2,y0-2),(x0+iw+2,y0+ih+2), C_ORANGE, 1)
    _t(frame, "THERMAL VIEW", (x0+4, y0+ih+14), 0.32, C_ORANGE)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 66)
    print("  FOREST FIRE EARLY DETECTION SYSTEM  |  Production v1.0")
    print("  Fire Analysis + Thermal Vision + Smoke Tracking + Hazard AI")
    print("=" * 66)

    if len(sys.argv) < 2:
        print("\n  Usage:")
        print("    python main.py input.mp4")
        print("    python main.py 0          (webcam)")
        sys.exit(0)

    source  = sys.argv[1]
    webcam  = str(source) in ("0","1","2")
    cap     = cv2.VideoCapture(int(source) if webcam else source)
    if not cap.isOpened():
        print(f"  [ERROR]  Cannot open: {source}")
        sys.exit(1)

    W_raw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_raw   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = 0 if webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale   = min(1.0, MAX_W / max(W_raw,1))
    W       = int(W_raw*scale); H = int(H_raw*scale)

    device, device_lbl = get_device()
    print(f"  [DEVICE]    {device_lbl}")

    # YOLO for person/vehicle context detection
    print(f"  [MODEL]     Loading YOLOv8n ...")
    model = YOLO("yolov8n.pt")
    if device == "cuda": model.to("cuda")
    print(f"  [MODEL]     Ready")

    writer  = cv2.VideoWriter(OUTPUT_PATH,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps_in, (W,H))
    smoke_tracker = MotionSmokeTracker()
    heatmap_acc   = HeatmapAccumulator(H,W)
    risk_panel    = RiskScorePanel()

    win = "Forest Fire Early Detection System"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    print(f"\n  [INFO]  {W}x{H}  |  {fps_in:.0f}fps  |  {total or 'live'} frames")
    print(f"  [INFO]  Output: {OUTPUT_PATH}")
    print(f"  Controls: Q=quit  P=pause  S=screenshot\n")

    from tqdm import tqdm
    bar        = tqdm(total=total or None, desc="  Processing",
                      unit="frame", dynamic_ncols=True)
    fps_smooth = 0.0; t_prev = time.time()
    frame_idx  = 0;   inf_ms = 0.0
    paused     = False

    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"),27): break
            if key == ord("p"):     paused = False
            continue

        ret, raw = cap.read()
        if not ret: break
        frame = cv2.resize(raw,(W,H)) if scale<1.0 else raw.copy()
        t0    = time.perf_counter()

        # ── Fire & smoke color analysis ──────────────────────────────────
        fire_mask, smoke_mask, fire_px, smoke_px, fire_zones, thermal = \
            analyze_fire_smoke(frame)

        # ── Motion smoke tracker ──────────────────────────────────────────
        _, smoke_motion, smoke_avg = smoke_tracker.update(frame)

        # ── Heatmap ───────────────────────────────────────────────────────
        heatmap_acc.update(fire_mask)

        # ── Risk level ────────────────────────────────────────────────────
        risk  = fire_risk_level(fire_px, smoke_px, fire_zones)
        score = min(fire_px / 100.0, 100.0)
        risk_panel.update(score)

        # ── YOLO context detection ────────────────────────────────────────
        inf_ms_yolo = 0.0
        results = model(frame, verbose=False, imgsz=640,
                        conf=0.30,
                        **({"half": True} if device=="cuda" else {}))[0]
        inf_ms = (time.perf_counter() - t0) * 1000

        # ── Thermal overlay (light) ───────────────────────────────────────
        frame = apply_thermal_overlay(frame, thermal, alpha=0.20)

        # ── Heatmap blend ─────────────────────────────────────────────────
        frame = heatmap_acc.render(frame, alpha=0.35)

        # ── Fire zones & smoke ────────────────────────────────────────────
        frame = draw_fire_zones(frame, fire_zones, smoke_mask, risk)

        # ── YOLO boxes (people/vehicles near fire) ────────────────────────
        if results.boxes is not None:
            for box in results.boxes:
                cls_name = results.names.get(int(box.cls[0]),"")
                if cls_name not in (PERSON_CLASSES | VEHICLE_CLASSES):
                    continue
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                col  = C_RED if cls_name=="person" else C_ORANGE
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                lbl = f"{cls_name} {conf:.2f}"
                _t(frame,lbl,(x1,y1-6),0.38,col)

        # ── Thermal inset ─────────────────────────────────────────────────
        embed_thermal_inset(frame, thermal)

        # ── HUD ───────────────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        t_now = time.time()
        dt = max(t_now-t_prev,1e-6); t_prev=t_now
        fps_smooth = 0.88*fps_smooth + 0.12/dt

        draw_top_hud(frame, fps_smooth, frame_idx, total,
                     inf_ms, device_lbl, risk, ts)
        draw_bottom_hud(frame, frame_idx, total, risk,
                        len(fire_zones), smoke_px>SMOKE_PIXEL_THRESH)
        draw_alert_banner(frame, risk)
        risk_panel.render(frame, risk, fire_px, smoke_px, smoke_avg)
        draw_legend(frame)
        frame = vignette(frame)

        writer.write(frame)
        cv2.imshow(win, frame)
        frame_idx += 1; bar.update(1)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"),27):
            print("\n  [QUIT]"); break
        elif key == ord("p"):
            paused = True; print("  [PAUSED]")
        elif key == ord("s"):
            sp = os.path.join("screenshots",
                 f"fire_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(sp, frame)
            print(f"\n  [SCREENSHOT]  {sp}")

    bar.close(); cap.release(); writer.release()
    cv2.destroyAllWindows()
    print(f"\n  [DONE]  {frame_idx} frames  ->  {OUTPUT_PATH}")
    print("=" * 66 + "\n")

if __name__ == "__main__":
    main()
