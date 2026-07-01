import sys
import subprocess
import os
import time
import datetime
import math
import warnings
import collections
import threading

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-INSTALL DEPENDENCIES
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
    ("flask",        "flask")
]

print("\n" + "=" * 66)
print("  FOREST FIRE WEB APP  |  Dependency Check")
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

from flask import Flask, Response, render_template, jsonify, request

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')
os.makedirs("uploads", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CV PIPELINE CONFIG & STATE
# ─────────────────────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_SIMPLEX
MAX_W       = 800

# Color definitions (BGR)
C_RED    = (30,  30, 220)
C_ORANGE = (20, 140, 255)
C_YELLOW = ( 0, 220, 255)
C_GREEN  = (50, 220,  50)
C_CYAN   = (220,220,  30)
C_WHITE  = (240,240, 240)
C_GRAY   = (140,140, 140)
C_DARK   = ( 12, 12,  12)

RISK_COLORS = {
    "CRITICAL": C_RED,
    "HIGH":     C_ORANGE,
    "MODERATE": C_YELLOW,
    "LOW":      C_GREEN,
    "CLEAR":    C_CYAN,
}

# Global Application State
app_state = {
    # Sources: "0" (webcam) or file path
    "source_type": "webcam",
    "video_path": None,
    "current_source": "0",
    
    # Flags / Settings
    "enable_thermal": True,
    "enable_heatmap": True,
    "enable_yolo": True,
    "enable_hud": True,
    "enable_vignette": True,
    
    # Thresholds & Parameters
    "thermal_alpha": 0.20,
    "heatmap_decay": 0.97,
    "yolo_conf": 0.30,
    "fire_pixel_thresh": 800,
    "smoke_pixel_thresh": 2000,
    
    # Real-time metrics
    "stats": {
        "risk_level": "CLEAR",
        "fire_pixels": 0,
        "smoke_pixels": 0,
        "motion_avg": 0.0,
        "n_zones": 0,
        "fps": 0.0,
        "device": "CPU",
        "inf_ms": 0.0,
        "timestamp": "",
        "status_log": [],
        "alerts": [],
        "max_confidence": 0.0
    }
}

state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGER UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def add_status_log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with state_lock:
        log_list = app_state["stats"]["status_log"]
        log_list.append(f"[{timestamp}] {message}")
        if len(log_list) > 50:
            log_list.pop(0)

def trigger_alert(level, message, ts):
    with state_lock:
        alerts = app_state["stats"]["alerts"]
        if not alerts or alerts[-1]["message"] != message:
            alerts.append({
                "timestamp": ts,
                "message": message,
                "level": level.lower()
            })
            if len(alerts) > 30:
                alerts.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTER VISION CLASSES & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"

class MotionSmokeTracker:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self.smoke_trail = collections.deque(maxlen=60)

    def update(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        H = frame.shape[0]
        upper_fg = fg.copy()
        upper_fg[H//2:, :] = 0
        smoke_motion = int(np.sum(upper_fg > 0))
        self.smoke_trail.append(smoke_motion)
        avg_motion = np.mean(self.smoke_trail)
        return fg, smoke_motion, avg_motion

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

def analyze_fire_smoke(frame, fire_px_thresh, smoke_px_thresh):
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire Lower/Upper 1 & 2
    f1 = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([18, 255, 255]))
    f2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))
    fire_raw  = cv2.bitwise_or(f1, f2)

    bright    = (hsv[:,:,2] > 160).astype(np.uint8) * 255
    fire_mask = cv2.bitwise_and(fire_raw, bright)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    fire_mask = cv2.dilate(fire_mask, np.ones((7,7), np.uint8), iterations=2)

    smoke_raw  = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 40, 220]))
    upper_mask = np.zeros((H,W), np.uint8)
    upper_mask[:H*2//3, :] = 255
    smoke_mask = cv2.bitwise_and(smoke_raw, upper_mask)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))

    fire_px  = int(np.sum(fire_mask  > 0))
    smoke_px = int(np.sum(smoke_mask > 0))

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_zones  = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 200:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            fire_zones.append((int(cx), int(cy), int(r)))

    # Thermal Map
    b, g, r_ch = frame[:,:,0].astype(np.float32), frame[:,:,1].astype(np.float32), frame[:,:,2].astype(np.float32)
    thermal = np.clip((r_ch * 0.6 + g * 0.3 - b * 0.5) / 255.0, 0, 1)
    thermal = gaussian_filter(thermal, sigma=12)

    return fire_mask, smoke_mask, fire_px, smoke_px, fire_zones, thermal

def fire_risk_level(fire_px, smoke_px, fire_zones, fire_px_thresh, smoke_px_thresh):
    n_zones = len(fire_zones)
    if fire_px > 8000 or n_zones > 3:
        return "CRITICAL"
    if fire_px > 3000 or n_zones > 1:
        return "HIGH"
    if fire_px > fire_px_thresh or smoke_px > smoke_px_thresh:
        return "MODERATE"
    if fire_px > 200 or smoke_px > 500:
        return "LOW"
    return "CLEAR"

def apply_thermal_overlay(frame, thermal_map, alpha=0.38):
    thermal_u8 = (thermal_map * 255).astype(np.uint8)
    thermal_colored = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(frame, 1.0 - alpha, thermal_colored, alpha, 0)

def draw_fire_zones(frame, fire_zones, smoke_mask, risk_level):
    if np.any(smoke_mask > 0):
        smoke_colored = np.zeros_like(frame)
        smoke_colored[smoke_mask > 0] = [180, 180, 160]
        frame = cv2.addWeighted(frame, 1.0, smoke_colored, 0.28, 0)
        sc, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sc:
            if cv2.contourArea(c) > 500:
                cv2.drawContours(frame, [c], -1, (160, 160, 140), 1)

    for i, (cx, cy, r) in enumerate(fire_zones):
        outer_r = r + 20 + (i * 5)
        cv2.circle(frame, (cx, cy), outer_r, C_RED,    1)
        cv2.circle(frame, (cx, cy), outer_r+8, C_ORANGE, 1)
        cv2.circle(frame, (cx, cy), r,     C_ORANGE, 2)
        cv2.circle(frame, (cx, cy), r//2,  C_YELLOW,  2)
        
        cv2.line(frame, (cx-outer_r-10, cy), (cx-r-4, cy), C_RED, 1)
        cv2.line(frame, (cx+r+4, cy), (cx+outer_r+10, cy), C_RED, 1)
        cv2.line(frame, (cx, cy-outer_r-10), (cx, cy-r-4), C_RED, 1)
        cv2.line(frame, (cx, cy+r+4), (cx, cy+outer_r+10), C_RED, 1)
        
        lbl = f"FIRE ZONE {i+1} r={r}px"
        cv2.putText(frame, lbl, (cx+r+6, cy-8), FONT_MONO, 0.38, C_ORANGE, 1, cv2.LINE_AA)
        cv2.arrowedLine(frame, (cx, cy-r), (cx, max(cy-r-40, 4)), C_YELLOW, 2, tipLength=0.35)
        cv2.putText(frame, "SPREAD", (cx+4, max(cy-r-20, 14)), FONT_MONO, 0.30, C_YELLOW, 1, cv2.LINE_AA)

    return frame

def draw_top_hud(frame, fps, device_lbl, risk_level, ts):
    H, W = frame.shape[:2]
    bh = 54
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,bh), C_DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    rc = RISK_COLORS.get(risk_level, C_CYAN)
    cv2.line(frame, (0,bh), (W,bh), rc, 2)

    def _t(img, text, pos, scale, color):
        x,y = int(pos[0]), int(pos[1])
        cv2.putText(img, text, (x+1,y+1), FONT_MONO, scale, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x,y), FONT_MONO, scale, color, 1, cv2.LINE_AA)

    _t(frame, "FOREST FIRE DETECTION SYSTEM", (14,32), 0.52, (30, 200, 255))
    r = W - 320
    _t(frame, f"FPS {fps:4.1f}", (r,    30), 0.40, C_GREEN)
    _t(frame, f"| {device_lbl}", (r+80, 30), 0.40, C_YELLOW)
    _t(frame, f"| {ts}", (r+180, 30), 0.40, C_GRAY)

def draw_bottom_hud(frame, risk_level, n_zones, smoke_active):
    H, W = frame.shape[:2]
    bh = 42; y0 = H - bh
    ov = frame.copy()
    cv2.rectangle(ov, (0,y0), (W,H), C_DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    rc = RISK_COLORS.get(risk_level, C_CYAN)
    cv2.line(frame, (0,y0), (W,y0), rc, 1)

    def _t(img, text, pos, scale, color):
        x,y = int(pos[0]), int(pos[1])
        cv2.putText(img, text, (x+1,y+1), FONT_MONO, scale, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x,y), FONT_MONO, scale, color, 1, cv2.LINE_AA)

    _t(frame, f"SYSTEM ONLINE", (14, y0+26), 0.40, C_GRAY)
    _t(frame, f"SMOKE: {'ACTIVE' if smoke_active else 'CLEAR'}", (W//2-100, y0+26), 0.40, C_YELLOW if smoke_active else C_GREEN)
    _t(frame, f"FIRE ZONES: {n_zones}", (W//2+60, y0+26), 0.40, C_ORANGE if n_zones else C_GREEN)
    _t(frame, f"RISK: {risk_level}", (W-150, y0+26), 0.40, rc)

def draw_alert_banner(frame, risk_level):
    if risk_level not in ("CRITICAL","HIGH"):
        return
    H, W = frame.shape[:2]
    rc = RISK_COLORS[risk_level]
    text = ("!! CRITICAL FIRE DETECTED — EMERGENCY ALERT !!"
            if risk_level == "CRITICAL" else "!! HIGH FIRE RISK — MONITOR IMMEDIATELY !!")
    cv2.rectangle(frame, (2,2), (W-2, H-2), rc, 3)
    bw = W - 28; bh = 34; bx = 14; by = 58
    ov = frame.copy()
    cv2.rectangle(ov, (bx,by), (bx+bw,by+bh), rc, -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    tw,_ = cv2.getTextSize(text, FONT_MONO, 0.55, 1)
    tx = bx + (bw - tw[0])//2
    cv2.putText(frame, text, (tx+1,by+22), FONT_MONO, 0.55, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,by+22), FONT_MONO, 0.55, C_WHITE, 1, cv2.LINE_AA)

def vignette(frame, strength=0.30):
    H,W = frame.shape[:2]
    cx,cy = W/2,H/2
    Y,X   = np.ogrid[:H,:W]
    d     = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    v     = np.clip(1.0 - d*strength, 0.55, 1.0).astype(np.float32)
    return np.clip(frame.astype(np.float32)*v[:,:,np.newaxis],0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  STREAM GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_frames():
    device, device_lbl = get_device()
    add_status_log(f"Backend CV device initialized: {device_lbl}")
    
    # Load YOLO
    add_status_log("Loading YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")
        if device == "cuda":
            model.to("cuda")
        add_status_log("YOLOv8n model loaded successfully.")
    except Exception as e:
        add_status_log(f"Error loading YOLO: {str(e)}")
        model = None

    last_source = None
    cap = None
    smoke_tracker = None
    heatmap_acc = None
    
    fps_smooth = 0.0
    t_prev = time.time()
    last_risk = "CLEAR"
    
    frame_idx = 0
    cached_boxes = []
    last_n_zones = 0
    last_smoke_active = False

    while True:
        with state_lock:
            current_src = app_state["current_source"]
            enable_thermal = app_state["enable_thermal"]
            enable_heatmap = app_state["enable_heatmap"]
            enable_yolo = app_state["enable_yolo"]
            enable_hud = app_state["enable_hud"]
            enable_vignette = app_state["enable_vignette"]
            thermal_alpha = app_state["thermal_alpha"]
            heatmap_decay = app_state["heatmap_decay"]
            yolo_conf = app_state["yolo_conf"]
            fire_pixel_thresh = app_state["fire_pixel_thresh"]
            smoke_pixel_thresh = app_state["smoke_pixel_thresh"]

        # Switch source if it changed
        if current_src != last_source:
            add_status_log(f"Switching video source to: {current_src}")
            if cap is not None:
                cap.release()
            
            # If "0", cast to int for webcam
            if current_src.isdigit():
                src_param = int(current_src)
                # Try DirectShow first on Windows as it is much more stable
                cap = cv2.VideoCapture(src_param, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    # Fallback to MSMF
                    cap = cv2.VideoCapture(src_param)
            else:
                src_param = current_src
                cap = cv2.VideoCapture(src_param)
            
            if not cap.isOpened():
                add_status_log(f"Failed to open source: {current_src}")
                time.sleep(2)
                continue
            
            # Re-init trackers
            smoke_tracker = MotionSmokeTracker()
            # Get frame dims
            ret, tmp_frame = cap.read()
            if ret:
                W_raw = tmp_frame.shape[1]
                scale = min(1.0, MAX_W / max(W_raw,1))
                W_proc = int(W_raw * scale)
                H_proc = int(tmp_frame.shape[0] * scale)
                heatmap_acc = HeatmapAccumulator(H_proc, W_proc)
            else:
                heatmap_acc = None
                
            last_source = current_src

        ret, raw = cap.read()
        if not ret:
            # If it's a video file, loop it
            if not current_src.isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                add_status_log("Webcam feed lost. Retrying...")
                time.sleep(2)
                continue

        t0 = time.perf_counter()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Dimensions
        H_raw, W_raw = raw.shape[:2]
        scale = min(1.0, MAX_W / max(W_raw, 1))
        W = int(W_raw * scale)
        H = int(H_raw * scale)
        frame = cv2.resize(raw, (W, H)) if scale < 1.0 else raw.copy()

        if heatmap_acc is None:
            heatmap_acc = HeatmapAccumulator(H, W)

        # 1. Fire and Smoke color analysis
        fire_mask, smoke_mask, fire_px, smoke_px, fire_zones, thermal = \
            analyze_fire_smoke(frame, fire_pixel_thresh, smoke_pixel_thresh)

        # 2. Motion smoke tracking
        _, smoke_motion, smoke_avg = smoke_tracker.update(frame)

        # 3. Update heatmap
        heatmap_acc.update(fire_mask, decay=heatmap_decay)

        # 4. Calculate Risk
        risk = fire_risk_level(fire_px, smoke_px, fire_zones, fire_pixel_thresh, smoke_pixel_thresh)

        # Alert trigger checking
        if risk != last_risk:
            add_status_log(f"Risk level changed from {last_risk} to {risk}!")
            last_risk = risk
            if risk in ("MODERATE", "HIGH", "CRITICAL"):
                trigger_alert(risk, f"WARNING: Forest Fire Risk level is {risk}", ts)
            elif risk == "LOW":
                trigger_alert("LOW", "Advisory: Low level threat detected", ts)
            else:
                trigger_alert("CLEAR", "System advisory: Threat level returned to CLEAR", ts)

        if n_zones != last_n_zones:
            if n_zones > last_n_zones:
                trigger_alert("HIGH" if n_zones > 1 else "MODERATE", f"Active fire zone detected! Zones count: {n_zones}", ts)
            last_n_zones = n_zones

        smoke_active = smoke_px > smoke_pixel_thresh
        if smoke_active != last_smoke_active:
            if smoke_active:
                trigger_alert("MODERATE", "Smoke density alert: Significant smoke detected in canopy", ts)
            last_smoke_active = smoke_active

        # 5. YOLO Context Detections
        inf_ms = 0.0
        n_zones = len(fire_zones)
        max_conf = 0.0
        
        if enable_yolo and model is not None:
            # Run YOLO every 4 frames (or if cache is empty) to optimize CPU FPS
            if frame_idx % 4 == 0 or len(cached_boxes) == 0:
                t_yolo_start = time.perf_counter()
                results = model(frame, verbose=False, imgsz=320, conf=yolo_conf,
                                **({"half": True} if device=="cuda" else {}))[0]
                inf_ms = (time.perf_counter() - t_yolo_start) * 1000
                
                cached_boxes = []
                if results.boxes is not None and len(results.boxes) > 0:
                    max_conf = float(torch.max(results.boxes.conf).item())
                    for box in results.boxes:
                        cls_name = results.names.get(int(box.cls[0]), "")
                        if cls_name in {"person", "car", "truck", "bus", "motorcycle", "bicycle"}:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            conf = float(box.conf[0])
                            cached_boxes.append((x1, y1, x2, y2, cls_name, conf))
            else:
                if len(cached_boxes) > 0:
                    max_conf = max(box[5] for box in cached_boxes)
            
            # Draw boxes (either fresh or cached)
            for x1, y1, x2, y2, cls_name, conf in cached_boxes:
                col = C_RED if cls_name == "person" else C_ORANGE
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                lbl = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, lbl, (x1+1, y1-5), FONT_MONO, 0.38, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, lbl, (x1, y1-6), FONT_MONO, 0.38, col, 1, cv2.LINE_AA)

        # 6. Apply overlays based on options
        if enable_thermal:
            frame = apply_thermal_overlay(frame, thermal, alpha=thermal_alpha)

        if enable_heatmap:
            frame = heatmap_acc.render(frame, alpha=0.35)

        # Draw Fire Zones & Smoke contours
        frame = draw_fire_zones(frame, fire_zones, smoke_mask, risk)

        # 7. Apply Vignette
        if enable_vignette:
            frame = vignette(frame)

        # 8. HUD Info
        t_now = time.time()
        dt = max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_smooth = 0.88 * fps_smooth + 0.12 / dt

        if enable_hud:
            draw_top_hud(frame, fps_smooth, device_lbl, risk, ts)
            draw_bottom_hud(frame, risk, n_zones, smoke_px > smoke_pixel_thresh)
            draw_alert_banner(frame, risk)

        # Update stats
        with state_lock:
            app_state["stats"]["risk_level"] = risk
            app_state["stats"]["fire_pixels"] = fire_px
            app_state["stats"]["smoke_pixels"] = smoke_px
            app_state["stats"]["motion_avg"] = float(smoke_avg)
            app_state["stats"]["n_zones"] = n_zones
            app_state["stats"]["fps"] = float(fps_smooth)
            app_state["stats"]["device"] = device_lbl
            app_state["stats"]["inf_ms"] = float(inf_ms)
            app_state["stats"]["timestamp"] = ts
            app_state["stats"]["max_confidence"] = float(max_conf)

        frame_idx += 1

        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTING
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize session trackers for client upload frames
client_trackers = {
    "smoke_tracker": None,
    "heatmap_acc": None,
    "frame_idx": 0,
    "cached_boxes": [],
    "last_n_zones": 0,
    "last_smoke_active": False,
    "last_risk": "CLEAR",
    "t_prev": time.time(),
    "fps_smooth": 0.0,
    "model": None
}

@app.route("/api/upload_frame", methods=["POST"])
def upload_frame():
    # Load YOLO if not loaded
    if client_trackers["model"] is None:
        try:
            device, _ = get_device()
            client_trackers["model"] = YOLO("yolov8n.pt")
            if device == "cuda":
                client_trackers["model"].to("cuda")
        except Exception as e:
            add_status_log(f"Error loading client YOLO: {str(e)}")
            
    file_bytes = request.data
    nparr = np.frombuffer(file_bytes, np.uint8)
    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if raw is None:
        return jsonify({"status": "error", "message": "Invalid frame data"}), 400
        
    device, device_lbl = get_device()
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    
    with state_lock:
        enable_thermal = app_state["enable_thermal"]
        enable_heatmap = app_state["enable_heatmap"]
        enable_yolo = app_state["enable_yolo"]
        enable_hud = app_state["enable_hud"]
        enable_vignette = app_state["enable_vignette"]
        thermal_alpha = app_state["thermal_alpha"]
        heatmap_decay = app_state["heatmap_decay"]
        yolo_conf = app_state["yolo_conf"]
        fire_pixel_thresh = app_state["fire_pixel_thresh"]
        smoke_pixel_thresh = app_state["smoke_pixel_thresh"]
        
    # Resize raw frame
    H_raw, W_raw = raw.shape[:2]
    scale = min(1.0, MAX_W / max(W_raw, 1))
    W = int(W_raw * scale)
    H = int(H_raw * scale)
    frame = cv2.resize(raw, (W, H)) if scale < 1.0 else raw.copy()
    
    # Initialize trackers if needed
    if client_trackers["smoke_tracker"] is None:
        client_trackers["smoke_tracker"] = MotionSmokeTracker()
    if client_trackers["heatmap_acc"] is None or client_trackers["heatmap_acc"].map.shape != (H, W):
        client_trackers["heatmap_acc"] = HeatmapAccumulator(H, W)
        
    # 1. Fire and Smoke color analysis
    fire_mask, smoke_mask, fire_px, smoke_px, fire_zones, thermal = \
        analyze_fire_smoke(frame, fire_pixel_thresh, smoke_pixel_thresh)
        
    # 2. Motion smoke tracking
    _, smoke_motion, smoke_avg = client_trackers["smoke_tracker"].update(frame)
    
    # 3. Update heatmap
    client_trackers["heatmap_acc"].update(fire_mask, decay=heatmap_decay)
    
    # 4. Calculate Risk
    risk = fire_risk_level(fire_px, smoke_px, fire_zones, fire_pixel_thresh, smoke_pixel_thresh)
    
    # Alert checking
    if risk != client_trackers["last_risk"]:
        add_status_log(f"Risk level changed from {client_trackers['last_risk']} to {risk}!")
        client_trackers["last_risk"] = risk
        if risk in ("MODERATE", "HIGH", "CRITICAL"):
            trigger_alert(risk, f"WARNING: Forest Fire Risk level is {risk}", ts)
        elif risk == "LOW":
            trigger_alert("LOW", "Advisory: Low level threat detected", ts)
        else:
            trigger_alert("CLEAR", "System advisory: Threat level returned to CLEAR", ts)
            
    n_zones = len(fire_zones)
    if n_zones != client_trackers["last_n_zones"]:
        if n_zones > client_trackers["last_n_zones"]:
            trigger_alert("HIGH" if n_zones > 1 else "MODERATE", f"Active fire zone detected! Zones count: {n_zones}", ts)
        client_trackers["last_n_zones"] = n_zones
        
    smoke_active = smoke_px > smoke_pixel_thresh
    if smoke_active != client_trackers["last_smoke_active"]:
        if smoke_active:
            trigger_alert("MODERATE", "Smoke density alert: Significant smoke detected in canopy", ts)
        client_trackers["last_smoke_active"] = smoke_active
        
    # 5. YOLO Context Detections
    inf_ms = 0.0
    max_conf = 0.0
    model = client_trackers["model"]
    
    if enable_yolo and model is not None:
        if client_trackers["frame_idx"] % 4 == 0 or len(client_trackers["cached_boxes"]) == 0:
            t_yolo_start = time.perf_counter()
            results = model(frame, verbose=False, imgsz=320, conf=yolo_conf,
                            **({"half": True} if device=="cuda" else {}))[0]
            inf_ms = (time.perf_counter() - t_yolo_start) * 1000
            
            client_trackers["cached_boxes"] = []
            if results.boxes is not None and len(results.boxes) > 0:
                max_conf = float(torch.max(results.boxes.conf).item())
                for box in results.boxes:
                    cls_name = results.names.get(int(box.cls[0]), "")
                    if cls_name in {"person", "car", "truck", "bus", "motorcycle", "bicycle"}:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf = float(box.conf[0])
                        client_trackers["cached_boxes"].append((x1, y1, x2, y2, cls_name, conf))
        else:
            if len(client_trackers["cached_boxes"]) > 0:
                max_conf = max(box[5] for box in client_trackers["cached_boxes"])
                
        # Draw boxes
        for x1, y1, x2, y2, cls_name, conf in client_trackers["cached_boxes"]:
            col = C_RED if cls_name == "person" else C_ORANGE
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            lbl = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, lbl, (x1+1, y1-5), FONT_MONO, 0.38, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x1, y1-6), FONT_MONO, 0.38, col, 1, cv2.LINE_AA)
            
    # 6. Apply overlays
    if enable_thermal:
        frame = apply_thermal_overlay(frame, thermal, alpha=thermal_alpha)
    if enable_heatmap:
        frame = client_trackers["heatmap_acc"].render(frame, alpha=0.35)
        
    frame = draw_fire_zones(frame, fire_zones, smoke_mask, risk)
    
    if enable_vignette:
        frame = vignette(frame)
        
    # 8. HUD info
    t_now = time.time()
    dt = max(t_now - client_trackers["t_prev"], 1e-6)
    client_trackers["t_prev"] = t_now
    client_trackers["fps_smooth"] = 0.88 * client_trackers["fps_smooth"] + 0.12 / dt
    
    if enable_hud:
        draw_top_hud(frame, client_trackers["fps_smooth"], device_lbl, risk, ts)
        draw_bottom_hud(frame, risk, n_zones, smoke_px > smoke_pixel_thresh)
        draw_alert_banner(frame, risk)
        
    # Update stats
    with state_lock:
        app_state["stats"]["risk_level"] = risk
        app_state["stats"]["fire_pixels"] = fire_px
        app_state["stats"]["smoke_pixels"] = smoke_px
        app_state["stats"]["motion_avg"] = float(smoke_avg)
        app_state["stats"]["n_zones"] = n_zones
        app_state["stats"]["fps"] = float(client_trackers["fps_smooth"])
        app_state["stats"]["device"] = device_lbl
        app_state["stats"]["inf_ms"] = float(inf_ms)
        app_state["stats"]["timestamp"] = ts
        app_state["stats"]["max_confidence"] = float(max_conf)
        
    client_trackers["frame_idx"] += 1
    
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"status": "error", "message": "Failed to encode frame"}), 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/api/stats")
def get_stats():
    with state_lock:
        return jsonify(app_state["stats"])

@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    if request.method == "POST":
        data = request.json
        with state_lock:
            if "enable_thermal" in data: app_state["enable_thermal"] = bool(data["enable_thermal"])
            if "enable_heatmap" in data: app_state["enable_heatmap"] = bool(data["enable_heatmap"])
            if "enable_yolo" in data: app_state["enable_yolo"] = bool(data["enable_yolo"])
            if "enable_hud" in data: app_state["enable_hud"] = bool(data["enable_hud"])
            if "enable_vignette" in data: app_state["enable_vignette"] = bool(data["enable_vignette"])
            
            if "thermal_alpha" in data: app_state["thermal_alpha"] = float(data["thermal_alpha"])
            if "heatmap_decay" in data: app_state["heatmap_decay"] = float(data["heatmap_decay"])
            if "yolo_conf" in data: app_state["yolo_conf"] = float(data["yolo_conf"])
            if "fire_pixel_thresh" in data: app_state["fire_pixel_thresh"] = int(data["fire_pixel_thresh"])
            if "smoke_pixel_thresh" in data: app_state["smoke_pixel_thresh"] = int(data["smoke_pixel_thresh"])
            
            if "source" in data:
                src = str(data["source"])
                if src == "webcam":
                    app_state["current_source"] = "0"
                    app_state["source_type"] = "webcam"
                elif src == "uploaded" and app_state["video_path"]:
                    app_state["current_source"] = app_state["video_path"]
                    app_state["source_type"] = "uploaded"
                    
        return jsonify({"status": "success", "config": {k: v for k, v in app_state.items() if k != "stats"}})
    
    with state_lock:
        config = {k: v for k, v in app_state.items() if k != "stats"}
        return jsonify(config)

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400
        
    filename = secure_filename_local(file.filename)
    dest_path = os.path.join("uploads", filename)
    file.save(dest_path)
    
    with state_lock:
        app_state["video_path"] = dest_path
        app_state["current_source"] = dest_path
        app_state["source_type"] = "uploaded"
        
    add_status_log(f"Uploaded and switched to video: {filename}")
    return jsonify({"status": "success", "filename": filename, "path": dest_path})

@app.route("/api/screenshot", methods=["POST"])
def take_screenshot():
    # Simple trigger to capture current camera/frame frame - but since generator runs in separate thread,
    # we can grab a quick frame from current source or notify frontend to capture canvas.
    # To keep it backend-consistent, we write screen to screenshots folder when next frame is processed.
    # We can just check the uploads folder or screenshots folder.
    with state_lock:
        src = app_state["current_source"]
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    ret, frame = cap.read()
    cap.release()
    if ret:
        sp = os.path.join("screenshots", f"web_fire_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        cv2.imwrite(sp, frame)
        add_status_log(f"Saved screenshot: {sp}")
        return jsonify({"status": "success", "filepath": sp})
    return jsonify({"status": "error", "message": "Could not capture frame"})

def secure_filename_local(filename):
    import re
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

if __name__ == "__main__":
    print("\n" + "=" * 66)
    print("  Starting Forest Fire Early Detection Web Interface...")
    print("  Access dashboard at: http://127.0.0.1:5000")
    print("=" * 66 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
