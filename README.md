# Forest Fire Early Detection AI Dashboard

An AI-powered computer vision system designed for real-time monitoring and early detection of forest fires. This application runs on a Python Flask backend wrapping OpenCV and YOLOv8, and features a premium dark-mode dashboard with geospatial tracking, active alert logging, and parameter tuning.

---

## 🛠️ Technologies Used

- **Backend (Python)**:
  - **Flask**: Lightweight web framework serving HTTP routes, frame uploads, and config APIs.
  - **OpenCV (cv2)**: Multi-spectral color range filtering (HSV), frame scaling, thermal mapping, contour mapping, and graphics overlays.
  - **Ultralytics YOLOv8**: Contextual object detection (people, vehicles) to assess fire zone proximity hazards.
  - **SciPy**: Gaussian filters for thermal gradient smoothing.
  - **PyTorch**: Core machine learning library driving YOLO.
- **Frontend (Web Dashboard)**:
  - **HTML5 & Vanilla CSS**: Premium dark-mode theme utilizing glassmorphism, responsive grids, and subtle micro-animations.
  - **Vanilla JavaScript**: AJAX configuration updates, UI data binding, and telemetry polling.
  - **Leaflet.js**: Geospatial mapping API tracking camera coordinates and pulsing alert zones.
  - **HTML5 Canvas**: Custom real-time sparkline plotting risk score trends.
  - **Browser MediaDevices API**: High-speed browser webcam frame capture and streaming.
- **Deployment & Dev Ops**:
  - **Docker**: Container configuration to streamline local/cloud deployments.
  - **Git**: Local version control.

---

## 🌟 Core Features

- **Real-Time Video Stream**: Processed camera frames displaying detected fire zones, spread trajectories, and smoke contours.
- **Geospatial Camera Tracking (Map)**: Interactive Leaflet.js map plotting camera coordinates. Spawns and pulses a red threat circle matching the scale of the detected fire zone.
- **AI Max Confidence Indicator**: Real-time gauge tracking the model's peak detection confidence.
- **Active Threat Alerts Log**: Scrollable panel logging timestamped threat levels and warnings (e.g. `[16:20:05] WARNING: Forest Fire Risk level is HIGH`).
- **Telemetry Charts**: Canvas-based real-time line charts mapping risk score trends.
- **Dynamic Configuration**: Adjust thermal overlays, heatmap decay rates, YOLO confidence thresholds, and pixel size limits in real-time.
- **Dual Webcam Engine**: Streams frames client-side (browser native) to avoid backend driver blocks, with direct backend capture fallback.

---

## 🚀 Performance Optimizations (CPU-ready)

Running deep learning models like YOLO on standard CPUs can lead to low framerates. This project implements the following custom pipeline optimizations to achieve high FPS:
1. **YOLO Detection Caching**: YOLOv8 inference is skipped on 3 out of 4 frames, using cached bounding boxes for intermediate frames. This reduces YOLO CPU load by **75%**.
2. **Reduced Inference Resolution**: Downscales YOLO image input (`imgsz`) from **640 to 320**, yielding a **4x reduction** in processing operations.
3. **Global Frame Downscaling**: Constrains processing width to **800px**, speeding up core color range and motion calculations by **2.5x**.

---

## 💻 Local Setup & Run

### Prerequisites
- Python 3.10+
- Pip package manager

### Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   cd "forest fire"
   ```
2. Run the server:
   ```bash
   py web_app.py
   ```
   *Note: Missing dependencies (`flask`, `ultralytics`, `scipy`, `opencv-python`) will be checked and installed automatically on startup.*
3. Open your web browser and navigate to:
   **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## ☁️ Cloud Deployment (Option A - Render/Railway)

The application is containerized and ready for instant cloud deployment:

### Configuration Files
- **[Dockerfile](Dockerfile)**: Sets up Python, installs system dependencies for OpenCV headless, downloads the PyTorch CPU-only runtime, and configures external port binding.
- **[requirements.txt](requirements.txt)**: Core packages (`flask`, `ultralytics`, `scipy`, `numpy`, `torchvision`, `opencv-python-headless`).
- **[.gitignore](.gitignore)**: Prevents committing heavy model weights (`yolov8n.pt`), caches, screenshots, and temp uploads.

### Deployment on Render.com
1. Create a new repository on your GitHub account and push this directory's files.
2. Log in to [Render.com](https://render.com) and click **New +** -> **Web Service**.
3. Link your GitHub repository.
4. Select **Docker** as the Runtime and choose the **Free** instance tier.
5. Click **Deploy Web Service**.
