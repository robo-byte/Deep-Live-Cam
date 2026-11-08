#!/usr/bin/env python3
"""
Browser Web UI for Deep-Live-Cam (GTX 1050 Ti 4GB Edition + Face Swap + Clothes Try-On)
Runs at http://localhost:7860 — supports 3 real-time modes:
  1. FACE SWAP         (CUDA InsightFace + InSwapper)
  2. CLOTHES TRY-ON    (MediaPipe Pose + SelfieSegmentation + Articulated Garment Warp, real face kept)
  3. BOTH              (Face Swap + Virtual Clothes Try-On simultaneously)
All modes output live to the 'Face Swap Cam' DirectShow virtual camera for WhatsApp/Zoom/Chrome.
"""

import os
import sys
import io
import json
import time
import base64
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

if sys.platform == "win32":
    _sp = os.path.join(project_root, "venv", "Lib", "site-packages")
    _nvidia_dir = os.path.join(_sp, "nvidia")
    if os.path.isdir(_nvidia_dir):
        for _pkg in os.listdir(_nvidia_dir):
            _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
            if os.path.isdir(_bin_dir):
                os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(_bin_dir)
                except OSError:
                    pass

import onnxruntime as ort
try:
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
    ort.set_default_logger_severity(3)
except Exception:
    pass

import modules.globals
from modules import imread_unicode, imwrite_unicode
from modules.video_capture import VideoCapturer
from modules.face_analyser import (
    get_face_analyser,
    reset_face_analyser,
    get_one_face,
    detect_one_face_fast,
)
from modules.processors.frame.face_swapper import (
    get_face_swapper,
    swap_face,
)
from modules.clothes_tryon import (
    get_tryon_engine,
    GARMENTS_DIR,
    SCREENSHOTS_DIR,
    GARMENT_CATEGORIES,
    ANCHOR_NAMES,
    ensure_sample_garments,
)

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _GPU_HANDLE = None

PORT = 7860
FACES_DIR = os.path.join(project_root, "faces")
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(GARMENTS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


class LiveFaceSwapEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.mode = "face_swap"  # 'face_swap', 'clothes_tryon', or 'both'
        self.source_path = os.path.join(FACES_DIR, "default_character.jpg")
        self.source_face = None
        self.source_thumb_b64 = ""
        self.profile = "balanced"
        self.camera_index = 0
        self.mirror = True
        self.swap_enabled = True
        self.virtual_cam_enabled = True
        self.fps = 0.0
        self.neural_fps = 0.0
        self.status_msg = "Initializing CUDA & Pose models..."
        self.latest_jpeg: bytes = b""
        self.latest_bgr: np.ndarray | None = None
        self.latest_pre_tryon_bgr: np.ndarray | None = None
        self.frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self.tryon = get_tryon_engine()

    def init_models(self):
        ensure_sample_garments()
        modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        modules.globals.apply_profile(self.profile)
        self.status_msg = "Loading InsightFace & InSwapper (CUDA) + MediaPipe Pose..."
        get_face_analyser()
        get_face_swapper()
        if os.path.isfile(self.source_path):
            self.set_source_image(self.source_path)
        self.status_msg = "Ready — Live stream active"

    def set_mode(self, mode: str) -> tuple[bool, str]:
        m = mode.lower().strip()
        if m not in ("face_swap", "clothes_tryon", "both"):
            return False, f"Invalid mode: {mode}"
        with self.lock:
            self.mode = m
            modules.globals.tryon_mode = m
            labels = {
                "face_swap": "Mode 1: FACE SWAP (CUDA Face Swap only)",
                "clothes_tryon": "Mode 2: CLOTHES TRY-ON (Real Face + Virtual Garment)",
                "both": "Mode 3: BOTH (Face Swap + Virtual Clothes Try-On)",
            }
            self.status_msg = f"Switched to {labels[m]}"
        return True, self.status_msg

    def set_source_image(self, img_path: str) -> tuple[bool, str]:
        img = imread_unicode(img_path)
        if img is None:
            return False, "Could not decode uploaded image file."
        prev_det = modules.globals.det_size
        modules.globals.det_size = 640
        reset_face_analyser()
        face = get_one_face(img)
        modules.globals.det_size = prev_det
        reset_face_analyser()

        if face is None:
            return False, "No clear face detected in that image! Try a clearer front-facing photo."

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = img.shape[:2]
        pad = int(max(x2 - x1, y2 - y1) * 0.25)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = cv2.resize(img[cy1:cy2, cx1:cx2], (180, 180))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        thumb_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

        with self.lock:
            self.source_path = img_path
            self.source_face = face
            self.source_thumb_b64 = thumb_b64
            if hasattr(swap_face, "_skip_state") and swap_face._skip_state:
                swap_face._skip_state["bgr_fake"] = None
            self.status_msg = f"Loaded source face: {os.path.basename(img_path)}"
        return True, self.status_msg

    def set_profile(self, prof: str):
        with self.lock:
            self.profile = prof
            modules.globals.apply_profile(prof)
            reset_face_analyser()
            self.status_msg = (
                f"Switched to {prof.upper()} profile "
                f"({modules.globals.capture_resolution[0]}x{modules.globals.capture_resolution[1]}, "
                f"det_size={modules.globals.det_size}, skip={modules.globals.frame_skip})"
            )

    def save_screenshot(self) -> dict:
        with self.lock:
            frame = None if self.latest_bgr is None else self.latest_bgr.copy()
            mode = self.mode
        if frame is None:
            return {"ok": False, "message": "No live frame available yet."}
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"screenshot_{mode}_{ts}.png"
        out_path = os.path.join(SCREENSHOTS_DIR, fn)
        imwrite_unicode(out_path, frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        self.status_msg = f"Saved screenshot: screenshots/{fn}"
        return {
            "ok": True,
            "filename": fn,
            "path": out_path,
            "preview_b64": b64,
            "message": self.status_msg,
        }

    def capture_snapshot_realistic(self) -> dict:
        with self.lock:
            base_frame = (
                self.latest_pre_tryon_bgr.copy()
                if self.latest_pre_tryon_bgr is not None
                else (self.latest_bgr.copy() if self.latest_bgr is not None else None)
            )
        if base_frame is None:
            return {"ok": False, "message": "No live frame available yet."}

        enhanced_bgr, report = self.tryon.process_snapshot_realistic(base_frame)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"snapshot_realistic_{ts}.png"
        out_path = os.path.join(SCREENSHOTS_DIR, fn)
        imwrite_unicode(out_path, enhanced_bgr)
        ok, buf = cv2.imencode(".jpg", enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        self.status_msg = f"Saved Snapshot Realistic: screenshots/{fn} ({report['elapsed_ms']} ms)"
        return {
            "ok": True,
            "filename": fn,
            "path": out_path,
            "preview_b64": b64,
            "report": report,
            "message": self.status_msg,
        }

    def start_stream(self, camera_index: int = 0):
        self.stop_stream()
        self.camera_index = camera_index
        self._stop_event.clear()
        self.running = True
        self._worker_thread = threading.Thread(target=self._loop, daemon=True)
        self._worker_thread.start()

    def stop_stream(self):
        self.running = False
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self._worker_thread = None

    def get_vram_str(self) -> str:
        if _GPU_HANDLE is None:
            return "N/A"
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            return f"{info.used // (1024*1024)} / {info.total // (1024*1024)} MiB"
        except Exception:
            return "N/A"

    def _loop(self):
        req_w, req_h = modules.globals.capture_resolution
        cap = VideoCapturer(self.camera_index)
        if not cap.start(req_w, req_h, 30):
            cap = VideoCapturer(-1)
            cap.start(req_w, req_h, 30)

        vcam = None
        frame_count = 0
        t_prev = time.perf_counter()
        last_neural_count = 0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                target_w, target_h = modules.globals.capture_resolution
                if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                with self.lock:
                    src_face = self.source_face
                    do_swap = self.swap_enabled
                    cur_mode = self.mode

                run_face_swap = do_swap and (cur_mode in ("face_swap", "both"))
                run_clothes_tryon = cur_mode in ("clothes_tryon", "both")

                if run_face_swap and src_face is not None:
                    t_face = detect_one_face_fast(frame)
                    if t_face is not None:
                        frame = swap_face(src_face, t_face, frame)

                with self.lock:
                    self.latest_pre_tryon_bgr = frame.copy()

                if run_clothes_tryon:
                    frame = self.tryon.process_frame(frame)

                with self.lock:
                    self.latest_bgr = frame

                # Send to Virtual Camera ('Face Swap Cam') if enabled
                if self.virtual_cam_enabled:
                    try:
                        fh, fw = frame.shape[:2]
                        if vcam is None or vcam.width != fw or vcam.height != fh:
                            if vcam is not None:
                                vcam.close()
                            import pyvirtualcam
                            vcam = pyvirtualcam.Camera(
                                width=fw,
                                height=fh,
                                fps=30,
                                fmt=pyvirtualcam.PixelFormat.BGR,
                                backend="unitycapture",
                            )
                        vcam.send(frame)
                    except Exception:
                        pass

                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.latest_jpeg = jpg.tobytes()
                    self.frame_event.set()

                frame_count += 1
                now = time.perf_counter()
                if now - t_prev >= 0.5:
                    self.fps = frame_count / (now - t_prev)
                    cur_neural = (
                        swap_face._skip_state["neural_count"]
                        if hasattr(swap_face, "_skip_state") and swap_face._skip_state
                        else 0
                    )
                    self.neural_fps = (cur_neural - last_neural_count) / (now - t_prev)
                    last_neural_count = cur_neural
                    frame_count = 0
                    t_prev = now
        finally:
            cap.release()
            if vcam is not None:
                try:
                    vcam.close()
                except Exception:
                    pass


ENGINE = LiveFaceSwapEngine()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Deep-Live-Cam Studio — Face Swap &amp; Virtual Clothes Try-On</title>
  <style>
    :root {
      --bg: #0b0e14;
      --card: #151923;
      --card-alt: #1b202e;
      --border: #283044;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --purple: #8b5cf6;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --green: #22c55e;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 12px 24px;
      background: #10141d;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    header h1 {
      margin: 0;
      font-size: 1.15rem;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      background: rgba(34, 197, 94, 0.15);
      color: var(--green);
      border: 1px solid rgba(34, 197, 94, 0.4);
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
    }
    .mode-bar {
      display: flex;
      gap: 6px;
      background: #0c0f17;
      padding: 5px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .mode-btn {
      background: transparent;
      color: var(--muted);
      border: none;
      padding: 8px 14px;
      border-radius: 7px;
      font-size: 0.86rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s;
    }
    .mode-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
    .mode-btn.active {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 2px 10px rgba(59, 130, 246, 0.35);
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 340px 1fr 370px;
      gap: 16px;
      padding: 16px 20px;
      max-width: 1680px;
      width: 100%;
      margin: 0 auto;
    }
    @media (max-width: 1280px) {
      main { grid-template-columns: 330px 1fr; }
      .right-col { grid-column: 1 / -1; }
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 13px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 13px;
    }
    .card h2 {
      margin: 0;
      font-size: 0.9rem;
      color: #cbd5e1;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .drop-zone {
      border: 2px dashed #3b82f677;
      border-radius: 11px;
      padding: 12px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: rgba(59, 130, 246, 0.05);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      font-size: 0.83rem;
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--accent);
      background: rgba(59, 130, 246, 0.14);
    }
    .face-thumb {
      width: 96px;
      height: 96px;
      border-radius: 50%;
      object-fit: cover;
      border: 3px solid var(--accent);
      background: #000;
    }
    .garment-thumb {
      width: 105px;
      height: 105px;
      border-radius: 10px;
      object-fit: contain;
      border: 2px solid var(--purple);
      background: #10131c;
      padding: 4px;
    }
    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 8px 13px;
      font-size: 0.86rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn-purple { background: var(--purple); }
    .btn-purple:hover { background: #7c3aed; }
    .btn-secondary {
      background: #232938;
      border: 1px solid var(--border);
      color: var(--text);
    }
    .btn-secondary:hover { background: #2e364a; }
    .btn-sm { padding: 6px 10px; font-size: 0.79rem; }
    .presets {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .control-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .slider-row {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .slider-header span:last-child {
      color: var(--text);
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
      margin: 2px 0;
    }
    label {
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 500;
    }
    select, input[type="text"] {
      background: #0f1219;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 7px 10px;
      font-size: 0.85rem;
    }
    .anywear-box {
      background: linear-gradient(145deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.15));
      border: 1px solid rgba(168, 85, 247, 0.35);
      border-radius: 10px;
      padding: 10px;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .anywear-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .anywear-badge {
      font-size: 0.78rem;
      font-weight: 700;
      color: #c084fc;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .bookmarklet-btn {
      background: rgba(168, 85, 247, 0.22);
      border: 1px dashed rgba(168, 85, 247, 0.65);
      color: #e9d5ff;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      text-decoration: none;
      cursor: grab;
      font-weight: 600;
      transition: all 0.2s;
    }
    .bookmarklet-btn:hover {
      background: rgba(168, 85, 247, 0.45);
      color: #ffffff;
    }
    .anywear-input-group {
      display: flex;
      gap: 6px;
    }
    .anywear-input-group input {
      flex: 1;
      font-size: 0.8rem;
      background: rgba(15, 18, 25, 0.85);
    }
    .anywear-subtext {
      font-size: 0.72rem;
      color: var(--muted);
      font-style: italic;
    }
    .toggle-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 3px 0;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .video-stage {
      position: relative;
      background: #06080c;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 440px;
    }
    .video-stage img {
      width: 100%;
      max-height: 620px;
      object-fit: contain;
      display: block;
    }
    .stats-bar {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.84rem;
      color: var(--muted);
      background: #10131c;
      padding: 9px 14px;
      border-radius: 9px;
      border: 1px solid var(--border);
      align-items: center;
    }
    .stats-bar strong { color: var(--text); }
    .action-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    #statusBanner {
      color: #60a5fa;
      font-size: 0.86rem;
      font-weight: 500;
    }
    .kbd-hint {
      font-size: 0.75rem;
      color: var(--muted);
      background: #10131c;
      padding: 6px 10px;
      border-radius: 7px;
      border: 1px solid var(--border);
    }
    .kbd {
      background: #23293a;
      border: 1px solid #384158;
      border-radius: 4px;
      padding: 1px 5px;
      font-family: monospace;
      color: #e2e8f0;
      font-size: 0.73rem;
    }
    /* Modal styles for 6-Point Anchor Calibration & Snapshot Realistic */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(5, 7, 12, 0.82);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
      padding: 16px;
    }
    .modal-backdrop.open { display: flex; }
    .modal-box {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      max-width: 780px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 92vh;
      overflow-y: auto;
    }
    .calib-layout {
      display: grid;
      grid-template-columns: 1fr 220px;
      gap: 14px;
      align-items: start;
    }
    #calibCanvas {
      background: repeating-conic-gradient(#181c26 0% 25%, #10131b 0% 50%) 50% / 20px 20px;
      border: 1px solid var(--border);
      border-radius: 10px;
      width: 100%;
      max-height: 460px;
      object-fit: contain;
      cursor: crosshair;
    }
    .anchor-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .anchor-item {
      background: #10131c;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 7px 9px;
      font-size: 0.78rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .anchor-item.selected {
      border-color: var(--accent);
      background: rgba(59, 130, 246, 0.16);
    }
  </style>
</head>
<body>
  <header>
    <h1>
      <span>Deep-Live-Cam Studio</span>
      <span class="badge">GTX 1050 Ti 4GB (CUDA + MediaPipe)</span>
    </h1>

    <!-- Mode Selector: 1. Face Swap | 2. Clothes Try-On | 3. Both -->
    <div class="mode-bar" role="tablist" aria-label="Pipeline Mode Selector">
      <button id="modeBtn_face_swap" class="mode-btn active" onclick="setMode('face_swap')">1. FACE SWAP</button>
      <button id="modeBtn_clothes_tryon" class="mode-btn" onclick="setMode('clothes_tryon')">2. CLOTHES TRY-ON</button>
      <button id="modeBtn_both" class="mode-btn" onclick="setMode('both')">3. BOTH (Face + Clothes)</button>
    </div>

    <div style="display:flex; align-items:center; gap:10px;">
      <a href="/camera_test.html" target="_blank" style="color:var(--muted); font-size:0.78rem; text-decoration:none; border:1px solid var(--border); padding:5px 9px; border-radius:7px; background:#10131c;">📹 Test Cam in Browser</a>
      <div id="statusBanner">Ready</div>
    </div>
  </header>

  <main>
    <!-- Left Column: Face Swap & Camera Controls -->
    <div class="card">
      <h2>1. Face Swap &amp; Camera</h2>
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <img id="faceThumb" class="face-thumb" src="" alt="Source Face" />
        <div><strong>Upload Face Photo</strong> or Drag &amp; Drop (`.jpg`, `.png`, `.webp`)</div>
        <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="uploadFile(this.files[0])" />
      </div>

      <div class="control-row">
        <label>Quick Sample Faces:</label>
        <div class="presets">
          <button class="btn btn-secondary btn-sm" onclick="selectPreset('default_character.jpg')">Sample Face 1</button>
          <button class="btn btn-secondary btn-sm" onclick="selectPreset('alt_character.jpg')">Sample Face 2</button>
        </div>
      </div>

      <div class="control-row">
        <label for="camSelect">Webcam Input:</label>
        <select id="camSelect" onchange="changeCamera(this.value)">
          <option value="0">A4tech FHD 1080P PC Camera (USB)</option>
          <option value="-1">Demo Video (Built-in Test Feed)</option>
        </select>
      </div>

      <div class="control-row">
        <label for="profSelect">GTX 1050 Ti Speed Profile:</label>
        <select id="profSelect" onchange="updateSettings()">
          <option value="fast">Fast — 480x360, det 160, skip 3 (~25 FPS)</option>
          <option value="balanced" selected>Balanced — 640x480, det 320, skip 2 (~20 FPS)</option>
          <option value="quality">Quality — 640x480, det 480, sync (~8 FPS)</option>
        </select>
      </div>

      <div class="toggle-row">
        <label for="chkSwap">Enable Face Swap</label>
        <input type="checkbox" id="chkSwap" checked onchange="updateSettings()" />
      </div>
      <div class="toggle-row">
        <label for="chkMirror">Mirror Webcam</label>
        <input type="checkbox" id="chkMirror" checked onchange="updateSettings()" />
      </div>
      <div class="toggle-row">
        <label for="chkVcam">Stream to "Face Swap Cam" Virtual Camera</label>
        <input type="checkbox" id="chkVcam" checked onchange="updateSettings()" />
      </div>

      <div class="kbd-hint">
        Shortcuts: <span class="kbd">M</span> Cycle Mode &nbsp;|&nbsp;
        <span class="kbd">1</span>-<span class="kbd">5</span> / <span class="kbd">[</span> <span class="kbd">]</span> Switch Garment &nbsp;|&nbsp;
        <span class="kbd">S</span> Save Screenshot
      </div>
    </div>

    <!-- Center Column: Live Stream & Snapshots -->
    <div class="card">
      <h2>
        <span id="stageTitle">Live Output Stream</span>
        <span id="activeModeBadge" class="badge">MODE: FACE SWAP</span>
      </h2>
      <div class="video-stage">
        <img id="liveStream" src="/api/stream.mjpg" alt="Live Webcam Stream" />
      </div>
      <div class="action-bar">
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button class="btn btn-secondary" onclick="saveScreenshot()">Save Screenshot (S)</button>
          <button class="btn btn-purple" onclick="captureSnapshotRealistic()">Snapshot Realistic (HD Drape)</button>
        </div>
        <div id="lastShotLink" style="font-size:0.8rem; color:var(--green);"></div>
      </div>
      <div class="stats-bar">
        <div>Stream FPS: <strong id="fpsVal">--</strong></div>
        <div>Mode: <strong id="modeVal">FACE SWAP</strong></div>
        <div>Profile: <strong id="profVal">BALANCED</strong></div>
        <div>GPU VRAM: <strong id="vramVal">--</strong></div>
        <div>Virtual Cam: <strong style="color:var(--green)">Face Swap Cam (Active)</strong></div>
      </div>
    </div>

    <!-- Right Column: Virtual Clothes Try-On Controls -->
    <div class="card right-col">
      <h2>2. Virtual Clothes Try-On</h2>

      <!-- Anywear E-Commerce Integration (URL / Bookmarklet / Zara / Shein / Amazon) -->
      <div class="anywear-box">
        <div class="anywear-header">
          <span class="anywear-badge">✨ Anywear Web Shopping</span>
          <a class="bookmarklet-btn" href="javascript:(function(){var s=document.createElement('script');s.src='http://localhost:7860/anywear_bookmarklet.js?t='+Date.now();document.body.appendChild(s);})();" onclick="alert('Drag this button to your Browser Bookmarks bar! While browsing Zara, Amazon, ASOS, or Shein, click the bookmark and then click any shirt to try it on live.');return false;" title="Drag to your Bookmarks Bar">🔖 Anywear Bookmarklet</a>
        </div>
        <div class="anywear-input-group">
          <input type="text" id="anywearUrlInput" placeholder="Paste Zara, Amazon, ASOS, Shein image URL..." onkeydown="if(event.key==='Enter')importAnywearUrl()" />
          <button class="btn btn-purple btn-sm" onclick="importAnywearUrl()">✨ Try On</button>
        </div>
        <div class="anywear-subtext">Tip: Drag images from shopping sites directly onto this window or your webcam stream!</div>
      </div>

      <div class="drop-zone" id="garmentDropZone" onclick="document.getElementById('garmentInput').click()">
        <img id="garmentThumb" class="garment-thumb" src="" alt="Active Garment" />
        <div><strong>Load Garment Image</strong> or Drop Shirt/Hoodie/Jacket (`.png`, `.jpg`, `.webp`)</div>
        <input type="file" id="garmentInput" accept="image/*" style="display:none" onchange="uploadGarment(this.files[0])" />
      </div>

      <div class="control-row">
        <label>Quick Garment Switcher (Keys 1-5):</label>
        <div id="garmentList" class="presets"></div>
      </div>

      <div class="control-row">
        <label for="catSelect">Garment Category:</label>
        <select id="catSelect" onchange="updateTryOnParams()">
          <option value="t-shirt">T-Shirt (Short Sleeve)</option>
          <option value="long-sleeve shirt">Long-Sleeve Shirt</option>
          <option value="jacket">Jacket / Blazer</option>
          <option value="hoodie">Hoodie / Sweatshirt</option>
        </select>
      </div>

      <!-- Fit & Alignment Sliders -->
      <div class="slider-row">
        <div class="slider-header"><span>Overall Scale</span><span id="val_scale">1.00</span></div>
        <input type="range" id="sl_scale" min="0.65" max="1.45" step="0.01" value="1.00" oninput="updateTryOnParams()" />
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Torso Width</span><span id="val_width">1.00</span></div>
        <input type="range" id="sl_width" min="0.65" max="1.45" step="0.01" value="1.00" oninput="updateTryOnParams()" />
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Torso Length</span><span id="val_length">1.00</span></div>
        <input type="range" id="sl_length" min="0.65" max="1.45" step="0.01" value="1.00" oninput="updateTryOnParams()" />
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Horizontal Offset (px)</span><span id="val_ox">0</span></div>
        <input type="range" id="sl_ox" min="-100" max="100" step="1" value="0" oninput="updateTryOnParams()" />
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Vertical Offset (px)</span><span id="val_oy">0</span></div>
        <input type="range" id="sl_oy" min="-100" max="100" step="1" value="0" oninput="updateTryOnParams()" />
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Garment Opacity</span><span id="val_opacity">1.00</span></div>
        <input type="range" id="sl_opacity" min="0.00" max="1.00" step="0.02" value="1.00" oninput="updateTryOnParams()" />
      </div>

      <div style="display:flex; gap:6px; flex-wrap:wrap;">
        <button class="btn btn-purple btn-sm" onclick="openCalibModal()">Calibrate Anchors (6-Pt)</button>
        <button class="btn btn-secondary btn-sm" onclick="resetTryOnSliders()">Reset Sliders</button>
      </div>

      <!-- Blending & Debug Toggles -->
      <div class="toggle-row">
        <label for="chkLight">Match Room Lighting &amp; Tone</label>
        <input type="checkbox" id="chkLight" checked onchange="updateTryOnParams()" />
      </div>
      <div class="toggle-row">
        <label for="chkCover">Cover Original Collar &amp; Neck</label>
        <input type="checkbox" id="chkCover" checked onchange="updateTryOnParams()" />
      </div>
      <div class="toggle-row">
        <label for="chkFgArms">Foreground Hands &amp; Chin Occlusion</label>
        <input type="checkbox" id="chkFgArms" checked onchange="updateTryOnParams()" />
      </div>
      <div class="toggle-row">
        <label for="chkSkel">Show Pose Skeleton (Debug)</label>
        <input type="checkbox" id="chkSkel" onchange="updateTryOnParams()" />
      </div>
      <div class="toggle-row">
        <label for="chkSeg">Show Segmentation Mask (Debug)</label>
        <input type="checkbox" id="chkSeg" onchange="updateTryOnParams()" />
      </div>

      <!-- Save / Load Garment Preset -->
      <div class="control-row" style="border-top:1px solid var(--border); padding-top:10px;">
        <label>Save / Load Garment Preset:</label>
        <div style="display:flex; gap:6px;">
          <input type="text" id="presetNameInput" placeholder="Preset name..." style="flex:1;" />
          <button class="btn btn-secondary btn-sm" onclick="saveGarmentPreset()">Save</button>
        </div>
        <div style="display:flex; gap:6px; margin-top:4px;">
          <select id="presetSelect" style="flex:1;"></select>
          <button class="btn btn-secondary btn-sm" onclick="loadGarmentPreset()">Load</button>
        </div>
      </div>
    </div>
  </main>

  <!-- 6-Point Garment Anchor Calibration Modal -->
  <div class="modal-backdrop" id="calibModal">
    <div class="modal-box">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0; font-size:1rem;">Calibrate Garment Anchors (Click or Drag 6 Points)</h2>
        <button class="btn btn-secondary btn-sm" onclick="closeCalibModal()">Close</button>
      </div>
      <div style="font-size:0.82rem; color:var(--muted);">
        Select an anchor on the right (or drag any circle directly on the garment cutout) to fine-tune how shoulders, hem, and sleeves map onto your body.
      </div>
      <div class="calib-layout">
        <canvas id="calibCanvas" width="640" height="640"></canvas>
        <div class="anchor-list">
          <div id="anchorItems"></div>
          <hr style="border-color:var(--border); width:100%;" />
          <button class="btn" onclick="saveCalibratedAnchors()">Apply &amp; Save Anchors</button>
          <button class="btn btn-secondary" onclick="resetAutoAnchors()">Reset Auto Anchors</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Snapshot Realistic Modal -->
  <div class="modal-backdrop" id="snapModal">
    <div class="modal-box">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0; font-size:1rem;">Snapshot Realistic — High-Detail Still Try-On</h2>
        <button class="btn btn-secondary btn-sm" onclick="document.getElementById('snapModal').classList.remove('open')">Close</button>
      </div>
      <img id="snapPreviewImg" src="" alt="Snapshot Realistic" style="width:100%; max-height:480px; object-fit:contain; border-radius:10px; background:#000;" />
      <div id="snapReportText" style="font-size:0.81rem; color:var(--muted); background:#10131c; padding:12px; border-radius:8px; border:1px solid var(--border); line-height:1.45;"></div>
    </div>
  </div>

  <script>
    const ANCHOR_META = [
      { key: 'left_shoulder',    label: '1. Left Shoulder',    color: '#22c55e' },
      { key: 'right_shoulder',   label: '2. Right Shoulder',   color: '#10b981' },
      { key: 'left_hem',         label: '3. Left Hem',         color: '#f59e0b' },
      { key: 'right_hem',        label: '4. Right Hem',        color: '#eab308' },
      { key: 'left_sleeve_end',  label: '5. Left Sleeve End',  color: '#3b82f6' },
      { key: 'right_sleeve_end', label: '6. Right Sleeve End', color: '#ec4899' }
    ];

    let currentMode = 'face_swap';
    let availableGarments = [];
    let currentGarmentIndex = 0;

    // Drag & Drop for Face
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });

    // Drag & Drop for Garment
    const gDrop = document.getElementById('garmentDropZone');
    gDrop.addEventListener('dragover', e => { e.preventDefault(); gDrop.classList.add('dragover'); });
    gDrop.addEventListener('dragleave', () => gDrop.classList.remove('dragover'));
    gDrop.addEventListener('drop', e => {
      e.preventDefault();
      gDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) uploadGarment(e.dataTransfer.files[0]);
    });

    function updateModeUI(mode) {
      currentMode = mode;
      ['face_swap', 'clothes_tryon', 'both'].forEach(m => {
        const btn = document.getElementById('modeBtn_' + m);
        if (btn) btn.classList.toggle('active', m === mode);
      });
      const labels = {
        'face_swap': 'FACE SWAP',
        'clothes_tryon': 'CLOTHES TRY-ON',
        'both': 'BOTH (FACE + CLOTHES)'
      };
      document.getElementById('modeVal').textContent = labels[mode] || mode.toUpperCase();
      document.getElementById('activeModeBadge').textContent = 'MODE: ' + (labels[mode] || mode.toUpperCase());
    }

    async function setMode(mode) {
      updateModeUI(mode);
      const res = await fetch('/api/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
    }

    async function uploadFile(file) {
      if (!file) return;
      document.getElementById('statusBanner').textContent = 'Uploading & detecting face...';
      const reader = new FileReader();
      reader.onload = async () => {
        const res = await fetch('/api/upload_face', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, data_url: reader.result })
        });
        const data = await res.json();
        document.getElementById('statusBanner').textContent = data.message;
        if (data.thumb) {
          document.getElementById('faceThumb').src = 'data:image/jpeg;base64,' + data.thumb;
        }
      };
      reader.readAsDataURL(file);
    }

    async function selectPreset(name) {
      const res = await fetch('/api/preset_face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.thumb) {
        document.getElementById('faceThumb').src = 'data:image/jpeg;base64,' + data.thumb;
      }
    }

    async function uploadGarment(file) {
      if (!file) return;
      document.getElementById('statusBanner').textContent = 'Removing background & detecting garment anchors...';
      const reader = new FileReader();
      reader.onload = async () => {
        const cat = document.getElementById('catSelect').value;
        const res = await fetch('/api/upload_garment', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, category: cat, data_url: reader.result })
        });
        const data = await res.json();
        document.getElementById('statusBanner').textContent = data.message;
        if (data.state) applyTryOnState(data.state);
        if (currentMode === 'face_swap') setMode('clothes_tryon');
      };
      reader.readAsDataURL(file);
    }

    async function selectGarment(filename, category, idx) {
      if (typeof idx === 'number') currentGarmentIndex = idx;
      const res = await fetch('/api/select_garment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, category })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.state) applyTryOnState(data.state);
    }

    async function updateTryOnParams() {
      const s = parseFloat(document.getElementById('sl_scale').value);
      const w = parseFloat(document.getElementById('sl_width').value);
      const l = parseFloat(document.getElementById('sl_length').value);
      const ox = parseFloat(document.getElementById('sl_ox').value);
      const oy = parseFloat(document.getElementById('sl_oy').value);
      const op = parseFloat(document.getElementById('sl_opacity').value);

      document.getElementById('val_scale').textContent = s.toFixed(2);
      document.getElementById('val_width').textContent = w.toFixed(2);
      document.getElementById('val_length').textContent = l.toFixed(2);
      document.getElementById('val_ox').textContent = Math.round(ox);
      document.getElementById('val_oy').textContent = Math.round(oy);
      document.getElementById('val_opacity').textContent = op.toFixed(2);

      const payload = {
        category: document.getElementById('catSelect').value,
        scale: s,
        width_scale: w,
        length_scale: l,
        offset_x: ox,
        offset_y: oy,
        opacity: op,
        lighting_match: document.getElementById('chkLight').checked,
        cover_original: document.getElementById('chkCover').checked,
        foreground_arms: document.getElementById('chkFgArms').checked,
        show_skeleton: document.getElementById('chkSkel').checked,
        show_seg_mask: document.getElementById('chkSeg').checked,
      };
      await fetch('/api/tryon_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    }

    function resetTryOnSliders() {
      document.getElementById('sl_scale').value = 1.00;
      document.getElementById('sl_width').value = 1.00;
      document.getElementById('sl_length').value = 1.00;
      document.getElementById('sl_ox').value = 0;
      document.getElementById('sl_oy').value = 0;
      document.getElementById('sl_opacity').value = 1.00;
      updateTryOnParams();
      document.getElementById('statusBanner').textContent = 'Reset garment sliders to defaults.';
    }

    function applyTryOnState(st) {
      if (!st) return;
      if (st.garment_thumb) {
        document.getElementById('garmentThumb').src = 'data:image/jpeg;base64,' + st.garment_thumb;
      }
      if (st.params) {
        document.getElementById('catSelect').value = st.params.category || 't-shirt';
        document.getElementById('sl_scale').value = st.params.scale ?? 1.0;
        document.getElementById('sl_width').value = st.params.width_scale ?? 1.0;
        document.getElementById('sl_length').value = st.params.length_scale ?? 1.0;
        document.getElementById('sl_ox').value = st.params.offset_x ?? 0;
        document.getElementById('sl_oy').value = st.params.offset_y ?? 0;
        document.getElementById('sl_opacity').value = st.params.opacity ?? 1.0;

        document.getElementById('val_scale').textContent = Number(st.params.scale ?? 1.0).toFixed(2);
        document.getElementById('val_width').textContent = Number(st.params.width_scale ?? 1.0).toFixed(2);
        document.getElementById('val_length').textContent = Number(st.params.length_scale ?? 1.0).toFixed(2);
        document.getElementById('val_ox').textContent = Math.round(st.params.offset_x ?? 0);
        document.getElementById('val_oy').textContent = Math.round(st.params.offset_y ?? 0);
        document.getElementById('val_opacity').textContent = Number(st.params.opacity ?? 1.0).toFixed(2);

        document.getElementById('chkLight').checked = !!st.params.lighting_match;
        document.getElementById('chkCover').checked = !!st.params.cover_original;
        document.getElementById('chkFgArms').checked = !!st.params.foreground_arms;
        document.getElementById('chkSkel').checked = !!st.params.show_skeleton;
        document.getElementById('chkSeg').checked = !!st.params.show_seg_mask;
      }
      if (Array.isArray(st.garments)) {
        availableGarments = st.garments;
        const container = document.getElementById('garmentList');
        container.innerHTML = '';
        st.garments.forEach((g, idx) => {
          const btn = document.createElement('button');
          btn.className = 'btn btn-secondary btn-sm';
          if (g.filename === st.garment_name) {
            btn.style.borderColor = '#8b5cf6';
            btn.style.background = 'rgba(139, 92, 246, 0.22)';
            currentGarmentIndex = idx;
          }
          const shortLabel = g.filename.replace(/\\.(png|jpg|jpeg|webp)$/i, '').replace(/_/g, ' ');
          btn.textContent = (idx + 1) + '. ' + shortLabel;
          btn.onclick = () => selectGarment(g.filename, g.category, idx);
          container.appendChild(btn);
        });
      }
      if (Array.isArray(st.presets)) {
        const sel = document.getElementById('presetSelect');
        const prev = sel.value;
        sel.innerHTML = '';
        st.presets.forEach(p => {
          const opt = document.createElement('option');
          opt.value = p;
          opt.textContent = p.replace(/\\.json$/i, '');
          sel.appendChild(opt);
        });
        if (prev && st.presets.includes(prev)) sel.value = prev;
      }
    }

    async function saveGarmentPreset() {
      const name = document.getElementById('presetNameInput').value.trim() || 'custom_preset';
      const res = await fetch('/api/save_preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.state) applyTryOnState(data.state);
    }

    async function loadGarmentPreset() {
      const filename = document.getElementById('presetSelect').value;
      if (!filename) return;
      const res = await fetch('/api/load_preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.state) applyTryOnState(data.state);
    }

    async function saveScreenshot() {
      const res = await fetch('/api/screenshot', { method: 'POST' });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.ok) {
        document.getElementById('lastShotLink').textContent = 'Saved: ' + data.filename;
      }
    }

    async function captureSnapshotRealistic() {
      document.getElementById('statusBanner').textContent = 'Generating Snapshot Realistic (cloth-drape luminance fold transfer)...';
      const res = await fetch('/api/snapshot_realistic', { method: 'POST' });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.ok) {
        document.getElementById('snapPreviewImg').src = 'data:image/jpeg;base64,' + data.preview_b64;
        document.getElementById('snapReportText').innerHTML =
          '<strong>' + data.report.mode + '</strong> (' + data.report.elapsed_ms + ' ms)<br/>' +
          '<strong>Saved file:</strong> <code>screenshots/' + data.filename + '</code><br/><br/>' +
          '<strong>GTX 1050 Ti 4GB Diffusion VTON Assessment:</strong> ' + data.report.diffusion_vton_feasibility;
        document.getElementById('snapModal').classList.add('open');
      }
    }

    // --- 6-Point Interactive Anchor Calibration Canvas ---
    let calibImg = new Image();
    let calibAnchors = {};
    let selectedAnchorKey = 'left_shoulder';
    let draggingAnchorKey = null;

    async function openCalibModal() {
      const res = await fetch('/api/garment_cutout');
      const data = await res.json();
      if (!data.ok) {
        document.getElementById('statusBanner').textContent = data.message;
        return;
      }
      calibAnchors = JSON.parse(JSON.stringify(data.anchors));
      const canvas = document.getElementById('calibCanvas');
      canvas.width = data.width;
      canvas.height = data.height;
      calibImg = new Image();
      calibImg.onload = () => {
        renderCalibCanvas();
        renderAnchorList();
        document.getElementById('calibModal').classList.add('open');
      };
      calibImg.src = 'data:image/png;base64,' + data.png_b64;
    }

    function closeCalibModal() {
      document.getElementById('calibModal').classList.remove('open');
    }

    function renderAnchorList() {
      const container = document.getElementById('anchorItems');
      container.innerHTML = '';
      ANCHOR_META.forEach(item => {
        const pt = calibAnchors[item.key] || [0, 0];
        const div = document.createElement('div');
        div.className = 'anchor-item' + (selectedAnchorKey === item.key ? ' selected' : '');
        div.innerHTML =
          '<span style="display:flex;align-items:center;gap:6px;">' +
          '<span style="width:10px;height:10px;border-radius:50%;background:' + item.color + ';display:inline-block;"></span>' +
          item.label + '</span>' +
          '<span>(' + Math.round(pt[0]) + ', ' + Math.round(pt[1]) + ')</span>';
        div.onclick = () => {
          selectedAnchorKey = item.key;
          renderAnchorList();
          renderCalibCanvas();
        };
        container.appendChild(div);
      });
    }

    function renderCalibCanvas() {
      const canvas = document.getElementById('calibCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (calibImg.complete && calibImg.naturalWidth > 0) {
        ctx.drawImage(calibImg, 0, 0);
      }

      // Draw torso quad & sleeve lines
      const ls = calibAnchors['left_shoulder'], rs = calibAnchors['right_shoulder'];
      const lh = calibAnchors['left_hem'], rh = calibAnchors['right_hem'];
      const lse = calibAnchors['left_sleeve_end'], rse = calibAnchors['right_sleeve_end'];

      if (ls && rs && lh && rh) {
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(ls[0], ls[1]);
        ctx.lineTo(rs[0], rs[1]);
        ctx.lineTo(rh[0], rh[1]);
        ctx.lineTo(lh[0], lh[1]);
        ctx.closePath();
        ctx.stroke();
      }
      if (ls && lse) {
        ctx.strokeStyle = '#3b82f6';
        ctx.beginPath();
        ctx.moveTo(ls[0], ls[1]);
        ctx.lineTo(lse[0], lse[1]);
        ctx.stroke();
      }
      if (rs && rse) {
        ctx.strokeStyle = '#ec4899';
        ctx.beginPath();
        ctx.moveTo(rs[0], rs[1]);
        ctx.lineTo(rse[0], rse[1]);
        ctx.stroke();
      }

      ANCHOR_META.forEach((item, idx) => {
        const pt = calibAnchors[item.key];
        if (!pt) return;
        const isSel = (item.key === selectedAnchorKey);
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], isSel ? 10 : 7, 0, Math.PI * 2);
        ctx.fillStyle = item.color;
        ctx.fill();
        ctx.lineWidth = isSel ? 3 : 2;
        ctx.strokeStyle = isSel ? '#ffffff' : '#000000';
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(String(idx + 1), pt[0] + 10, pt[1] + 4);
      });
    }

    (function initCalibEvents() {
      const canvas = document.getElementById('calibCanvas');
      function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return [
          Math.max(0, Math.min(canvas.width, (e.clientX - rect.left) * scaleX)),
          Math.max(0, Math.min(canvas.height, (e.clientY - rect.top) * scaleY))
        ];
      }
      canvas.addEventListener('mousedown', e => {
        const [x, y] = getCoords(e);
        let hitKey = null;
        let bestDist = 22;
        ANCHOR_META.forEach(item => {
          const pt = calibAnchors[item.key];
          if (pt) {
            const d = Math.hypot(pt[0] - x, pt[1] - y);
            if (d < bestDist) { bestDist = d; hitKey = item.key; }
          }
        });
        if (hitKey) {
          selectedAnchorKey = hitKey;
          draggingAnchorKey = hitKey;
        } else if (selectedAnchorKey) {
          calibAnchors[selectedAnchorKey] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
          draggingAnchorKey = selectedAnchorKey;
        }
        renderAnchorList();
        renderCalibCanvas();
      });
      window.addEventListener('mousemove', e => {
        if (!draggingAnchorKey) return;
        const [x, y] = getCoords(e);
        calibAnchors[draggingAnchorKey] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        renderAnchorList();
        renderCalibCanvas();
      });
      window.addEventListener('mouseup', () => { draggingAnchorKey = null; });
    })();

    async function saveCalibratedAnchors() {
      const res = await fetch('/api/tryon_anchors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ anchors: calibAnchors })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.state) applyTryOnState(data.state);
      closeCalibModal();
    }

    async function resetAutoAnchors() {
      const res = await fetch('/api/reset_anchors', { method: 'POST' });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
      if (data.state) {
        applyTryOnState(data.state);
        calibAnchors = JSON.parse(JSON.stringify(data.state.anchors));
        renderAnchorList();
        renderCalibCanvas();
      }
    }

    async function updateSettings() {
      const payload = {
        profile: document.getElementById('profSelect').value,
        swap_enabled: document.getElementById('chkSwap').checked,
        mirror: document.getElementById('chkMirror').checked,
        virtual_cam: document.getElementById('chkVcam').checked,
      };
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
    }

    async function changeCamera(camIdx) {
      document.getElementById('statusBanner').textContent = 'Switching camera...';
      const res = await fetch('/api/camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: parseInt(camIdx, 10) })
      });
      const data = await res.json();
      document.getElementById('statusBanner').textContent = data.message;
    }

    // Keyboard shortcuts: M = cycle mode, S = screenshot, 1-5 / [ ] = switch garment
    window.addEventListener('keydown', e => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;
      const k = e.key.toLowerCase();
      if (k === 'm') {
        const order = ['face_swap', 'clothes_tryon', 'both'];
        const next = order[(order.indexOf(currentMode) + 1) % order.length];
        setMode(next);
      } else if (k === 's') {
        saveScreenshot();
      } else if (k >= '1' && k <= '5') {
        const idx = parseInt(k, 10) - 1;
        if (availableGarments[idx]) {
          selectGarment(availableGarments[idx].filename, availableGarments[idx].category, idx);
        }
      } else if (e.key === '[' && availableGarments.length > 0) {
        const idx = (currentGarmentIndex - 1 + availableGarments.length) % availableGarments.length;
        selectGarment(availableGarments[idx].filename, availableGarments[idx].category, idx);
      } else if (e.key === ']' && availableGarments.length > 0) {
        const idx = (currentGarmentIndex + 1) % availableGarments.length;
        selectGarment(availableGarments[idx].filename, availableGarments[idx].category, idx);
      }
    });

    async function importAnywearUrl(customUrl) {
      const input = document.getElementById('anywearUrlInput');
      const url = customUrl || (input ? input.value.trim() : '');
      if (!url) return;
      document.getElementById('statusBanner').textContent = '✨ Anywear: Importing garment from URL & removing background...';
      try {
        const cat = document.getElementById('catSelect') ? document.getElementById('catSelect').value : 't-shirt';
        const res = await fetch('/api/import_url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, category: cat })
        });
        const data = await res.json();
        document.getElementById('statusBanner').textContent = data.message || 'Garment imported!';
        if (data.state) applyTryOnState(data.state);
        if (data.mode) currentMode = data.mode;
        if (currentMode === 'face_swap') setMode('clothes_tryon');
        if (input) input.value = '';
      } catch (err) {
        document.getElementById('statusBanner').textContent = 'Failed to import garment URL: ' + err;
      }
    }

    // Drag-and-drop listener for external web images or files
    window.addEventListener('dragover', e => { e.preventDefault(); });
    window.addEventListener('drop', e => {
      e.preventDefault();
      const dt = e.dataTransfer;
      if (!dt) return;
      if (dt.files && dt.files.length > 0) {
        uploadGarment(dt.files[0]);
      } else {
        const html = dt.getData('text/html');
        const text = dt.getData('text/plain') || dt.getData('text/uri-list');
        let imgUrl = '';
        if (html) {
          const m = html.match(/src=["']([^"']+)["']/i);
          if (m) imgUrl = m[1];
        }
        if (!imgUrl && text && (text.startsWith('http') || text.startsWith('data:image/'))) {
          imgUrl = text.trim();
        }
        if (imgUrl) importAnywearUrl(imgUrl);
      }
    });

    async function initTryOnUI() {
      try {
        const res = await fetch('/api/tryon_state');
        const st = await res.json();
        applyTryOnState(st);
      } catch (e) {}
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const d = await res.json();
        document.getElementById('fpsVal').textContent = d.fps.toFixed(1) + ' FPS';
        document.getElementById('profVal').textContent = d.profile.toUpperCase();
        document.getElementById('vramVal').textContent = d.vram;
        if (d.mode && d.mode !== currentMode) updateModeUI(d.mode);
        if (d.thumb && !document.getElementById('faceThumb').getAttribute('src')) {
          document.getElementById('faceThumb').src = 'data:image/jpeg;base64,' + d.thumb;
        }
      } catch (e) {}
    }
    initTryOnUI();
    setInterval(pollStatus, 1000);
    pollStatus();
  </script>
</body>
</html>
"""


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/anywear_bookmarklet.js":
            bk_path = os.path.join(project_root, "anywear_bookmarklet.js")
            if os.path.isfile(bk_path):
                with open(bk_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        elif parsed.path in ("/camera_test", "/camera_test.html"):
            cam_path = os.path.join(project_root, "camera_test.html")
            if os.path.isfile(cam_path):
                with open(cam_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        elif parsed.path == "/api/status":
            self._send_json({
                "fps": ENGINE.fps,
                "neural_fps": ENGINE.neural_fps,
                "profile": ENGINE.profile,
                "mode": ENGINE.mode,
                "vram": ENGINE.get_vram_str(),
                "status": ENGINE.status_msg,
                "thumb": ENGINE.source_thumb_b64,
            })
        elif parsed.path == "/api/tryon_state":
            self._send_json(ENGINE.tryon.get_state_dict())
        elif parsed.path == "/api/garment_cutout":
            self._send_json(ENGINE.tryon.get_cutout_png_b64())
        elif parsed.path == "/api/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ENGINE.frame_event.wait(timeout=1.0)
                    jpg = ENGINE.latest_jpeg
                    if not jpg:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        data = json.loads(raw.decode("utf-8")) if raw else {}

        if self.path == "/api/mode":
            ok, msg = ENGINE.set_mode(str(data.get("mode", "face_swap")))
            self._send_json({"ok": ok, "mode": ENGINE.mode, "message": msg})

        elif self.path == "/api/upload_face":
            filename = os.path.basename(data.get("filename", "uploaded_face.jpg"))
            data_url = data.get("data_url", "")
            if "," in data_url:
                b64_part = data_url.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_part)
                save_path = os.path.join(FACES_DIR, f"custom_{int(time.time())}_{filename}")
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                ok, msg = ENGINE.set_source_image(save_path)
                self._send_json({"ok": ok, "message": msg, "thumb": ENGINE.source_thumb_b64})
            else:
                self._send_json({"ok": False, "message": "Invalid image upload"}, 400)

        elif self.path == "/api/preset_face":
            name = os.path.basename(data.get("name", "default_character.jpg"))
            p = os.path.join(FACES_DIR, name)
            ok, msg = ENGINE.set_source_image(p)
            self._send_json({"ok": ok, "message": msg, "thumb": ENGINE.source_thumb_b64})

        elif self.path == "/api/import_url":
            url = str(data.get("url", "")).strip()
            category = str(data.get("category", "t-shirt"))
            if not url:
                self._send_json({"ok": False, "message": "No garment image URL provided"}, 400)
            else:
                try:
                    import urllib.request
                    import ssl
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    if url.startswith("data:image/"):
                        b64_part = url.split(",", 1)[1] if "," in url else url
                        img_bytes = base64.b64decode(b64_part)
                    else:
                        req = urllib.request.Request(
                            url,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                            }
                        )
                        with urllib.request.urlopen(req, timeout=12, context=ctx) as resp:
                            img_bytes = resp.read()

                    if len(img_bytes) < 100:
                        self._send_json({"ok": False, "message": "Downloaded image is empty or invalid"}, 400)
                    else:
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img_mat = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        if img_mat is None:
                            self._send_json({"ok": False, "message": "Unable to decode image from URL"}, 400)
                        else:
                            safe_ts = int(time.time())
                            save_path = os.path.join(GARMENTS_DIR, f"anywear_{safe_ts}.png")
                            cv2.imwrite(save_path, img_mat)
                            ok, msg = ENGINE.tryon.set_garment(save_path, category)
                            if ok:
                                if ENGINE.mode == "face_swap":
                                    ENGINE.set_mode("clothes_tryon")
                                ENGINE.status_msg = f"Anywear: Wearing '{os.path.basename(save_path)}' live"
                                self._send_json({
                                    "ok": True,
                                    "message": msg,
                                    "garment_name": os.path.basename(save_path),
                                    "state": ENGINE.tryon.get_state_dict(),
                                    "mode": ENGINE.mode,
                                })
                            else:
                                self._send_json({"ok": False, "message": msg}, 400)
                except Exception as e:
                    self._send_json({"ok": False, "message": f"Import failed: {str(e)}"}, 500)

        elif self.path == "/api/upload_garment":
            filename = os.path.basename(data.get("filename", "garment.png"))
            category = str(data.get("category", "t-shirt"))
            data_url = data.get("data_url", "")
            if "," in data_url:
                b64_part = data_url.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_part)
                save_path = os.path.join(GARMENTS_DIR, f"custom_{int(time.time())}_{filename}")
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                ok, msg = ENGINE.tryon.set_garment(save_path, category)
                ENGINE.status_msg = msg
                self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})
            else:
                self._send_json({"ok": False, "message": "Invalid garment upload"}, 400)

        elif self.path == "/api/select_garment":
            filename = os.path.basename(data.get("filename", "classic_navy_tshirt.png"))
            category = data.get("category")
            p = os.path.join(GARMENTS_DIR, filename)
            ok, msg = ENGINE.tryon.set_garment(p, category)
            ENGINE.status_msg = msg
            self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})

        elif self.path == "/api/tryon_settings":
            st = ENGINE.tryon.update_params(data)
            self._send_json({"ok": True, "state": st})

        elif self.path == "/api/tryon_anchors":
            anchors = data.get("anchors", {})
            ok, msg = ENGINE.tryon.set_custom_anchors(anchors)
            ENGINE.status_msg = msg
            self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})

        elif self.path == "/api/reset_anchors":
            ok, msg = ENGINE.tryon.reset_anchors()
            ENGINE.status_msg = msg
            self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})

        elif self.path == "/api/save_preset":
            name = str(data.get("name", "custom_preset"))
            ok, msg = ENGINE.tryon.save_preset(name)
            ENGINE.status_msg = msg
            self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})

        elif self.path == "/api/load_preset":
            filename = str(data.get("filename", ""))
            ok, msg = ENGINE.tryon.load_preset(filename)
            ENGINE.status_msg = msg
            self._send_json({"ok": ok, "message": msg, "state": ENGINE.tryon.get_state_dict()})

        elif self.path == "/api/screenshot":
            self._send_json(ENGINE.save_screenshot())

        elif self.path == "/api/snapshot_realistic":
            self._send_json(ENGINE.capture_snapshot_realistic())

        elif self.path == "/api/settings":
            prof = data.get("profile")
            if prof and prof != ENGINE.profile:
                ENGINE.set_profile(prof)
            ENGINE.swap_enabled = bool(data.get("swap_enabled", True))
            ENGINE.mirror = bool(data.get("mirror", True))
            ENGINE.virtual_cam_enabled = bool(data.get("virtual_cam", True))
            self._send_json({"ok": True, "message": ENGINE.status_msg})

        elif self.path == "/api/camera":
            cam_idx = int(data.get("camera_index", 0))
            ENGINE.start_stream(cam_idx)
            self._send_json({"ok": True, "message": f"Switched to camera index {cam_idx}"})
        else:
            self.send_error(404)


def main():
    print("============================================================================")
    print("  Deep-Live-Cam Studio (Face Swap + Clothes Try-On) — http://localhost:7860")
    print("============================================================================", flush=True)
    ENGINE.init_models()
    ENGINE.start_stream(0)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), WebHandler)
    url = f"http://localhost:{PORT}"
    print(f"[WebUI] Ready! Opening {url} in your browser...", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop_stream()
        server.server_close()


if __name__ == "__main__":
    main()
