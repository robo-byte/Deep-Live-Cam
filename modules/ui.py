"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode, imwrite_unicode
from modules.video_capture import VideoCapturer
from modules.clothes_tryon import (
    get_tryon_engine,
    GARMENTS_DIR,
    GARMENTS_PRESETS_DIR,
    SCREENSHOTS_DIR,
    GARMENT_CATEGORIES,
    ANCHOR_NAMES,
)

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 960
ROOT_WIDTH = 680
_LAST_LIVE_FRAME: Optional[np.ndarray] = None

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QMainWindow, QDialog { background-color: #1e1e1e; color: #e6e6e6; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 11pt; }

QGroupBox {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #3a3a3a;
}
QPushButton#secondary:hover { background-color: #4a4a4a; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 36px; height: 18px;
    border-radius: 9px;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #2a2a2a;
    border: 2px dashed #444;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #262626;
    border-radius: 10px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
# How long to wait quietly for a live-preview worker to notice the stop flag
# before saying so. Shutdown then keeps waiting: it has to finish.
WORKER_SHUTDOWN_GRACE_MS = 30000
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        indices: List[int] = []
        names: List[str] = []
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                # Exclude our own output virtual camera to avoid a feedback loop
                if "face swap cam" in name.lower() or "unity video capture" in name.lower():
                    continue
                indices.append(idx)
                names.append(name)
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
        # Always provide the built-in Demo Camera fallback so Live + Virtual Cam
        # can be tested even when no physical USB camera is plugged in.
        indices.append(-1)
        names.append("Demo Camera (Built-in Test Video)")
        return indices, names

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        cli_res = modules.globals.capture_resolution
        cli_det = modules.globals.det_size
        cli_skip = getattr(modules.globals, "frame_skip", 2)
        cli_prof = getattr(modules.globals, "profile", "balanced")
        cli_vcam = getattr(modules.globals, "virtual_cam", True)
        load_switch_states()
        # Preserve CLI/profile settings over stale switch_states.json
        modules.globals.capture_resolution = cli_res
        modules.globals.det_size = cli_det
        modules.globals.frame_skip = cli_skip
        modules.globals.profile = cli_prof
        modules.globals.virtual_cam = cli_vcam
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())

        # Action buttons
        layout.addLayout(self._build_action_row())

        # Camera selection
        layout.addWidget(self._build_camera_card())

        # Virtual Clothes Try-On & Mode Selector
        layout.addWidget(self._build_clothes_tryon_card())

        # Status & footer
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        footer = QLabel("Deep Live Cam (GTX 1050 Ti + Face Swap Cam Edition)")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        layout.addWidget(footer)

        # Auto-load CLI source_path or default character face
        if not modules.globals.source_path:
            default_face = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "faces",
                "default_character.jpg",
            )
            if os.path.isfile(default_face):
                modules.globals.source_path = default_face
        if modules.globals.source_path and os.path.isfile(modules.globals.source_path):
            try:
                self.source_label.setPixmap(
                    render_image_preview(modules.globals.source_path, (200, 200))
                )
                self.source_label.setText("")
            except Exception:
                pass
        if getattr(modules.globals, "auto_live", False):
            QTimer.singleShot(400, self._on_live)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown
        enhancer_label = QLabel(_("Face Enhancer:"))
        grid.addWidget(enhancer_label, len(items) // 2, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, len(items) // 2, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Row 0: camera selector + Refresh + Live button
        grid.addWidget(QLabel(_("Select Camera:")), 0, 0)
        self._camera_indices, self._camera_names = get_available_cameras()

        cam_row = QHBoxLayout()
        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
            if modules.globals.camera_index is not None and modules.globals.camera_index in self._camera_indices:
                self.cb_camera.setCurrentIndex(self._camera_indices.index(modules.globals.camera_index))
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        cam_row.addWidget(self.cb_camera, 1)

        self.btn_refresh_cam = QPushButton("🔄")
        self.btn_refresh_cam.setObjectName("secondary")
        self.btn_refresh_cam.setFixedWidth(38)
        self.btn_refresh_cam.setToolTip("Refresh connected USB cameras")
        self.btn_refresh_cam.clicked.connect(self._on_refresh_cameras)
        cam_row.addWidget(self.btn_refresh_cam)
        grid.addLayout(cam_row, 0, 1)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap and stream to Face Swap Cam"))
        self.btn_live.clicked.connect(self._on_live)
        grid.addWidget(self.btn_live, 0, 2)

        # Row 1: capture resolution
        grid.addWidget(QLabel(_("Resolution:")), 1, 0)
        self.cb_resolution = QComboBox()
        self._resolution_options = [
            ("480 x 360 (Fast)", (480, 360)),
            ("640 x 480 (Balanced)", (640, 480)),
            ("960 x 540 (qHD)", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 x 1080 (FHD)", (1920, 1080)),
        ]
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == cur), 1)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.cb_resolution.setToolTip(_("Requested webcam resolution (480x360 or 640x480 recommended for GTX 1050 Ti)."))
        grid.addWidget(self.cb_resolution, 1, 1, 1, 2)

        # Row 2: face detection size
        grid.addWidget(QLabel(_("Det size:")), 2, 0)
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 480, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v} x {v}")
        cur_det = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
        # Normalize to a valid option first so .index() can never raise: fall back
        # to the default size, then to the last (highest) option if even that is
        # somehow missing.
        if cur_det not in self._det_size_options:
            cur_det = (modules.globals.DEFAULT_DET_SIZE
                       if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                       else self._det_size_options[-1])
        idx = self._det_size_options.index(cur_det)
        self.cb_det_size.setCurrentIndex(idx)
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(_(
            "Face detection input resolution. Lower = faster, less accurate at "
            "distance. Changes take effect on next face analyser init."
        ))
        grid.addWidget(self.cb_det_size, 2, 1, 1, 2)

        # Row 3: GTX 1050 Ti Profile + Virtual Cam switch
        grid.addWidget(QLabel("GPU Profile:"), 3, 0)
        self.cb_profile = QComboBox()
        self._profile_options = [
            ("Fast (480x360, det 160, skip 3)", "fast"),
            ("Balanced (640x480, det 320, skip 2)", "balanced"),
            ("Quality (640x480, det 480, sync)", "quality"),
        ]
        for lbl, _pk in self._profile_options:
            self.cb_profile.addItem(lbl)
        cur_prof = getattr(modules.globals, "profile", "balanced")
        p_idx = next((i for i, (_, pk) in enumerate(self._profile_options) if pk == cur_prof), 1)
        self.cb_profile.setCurrentIndex(p_idx)
        self.cb_profile.currentIndexChanged.connect(self._on_profile_change)
        grid.addWidget(self.cb_profile, 3, 1)

        self.sw_vcam = _Switch("Face Swap Cam", getattr(modules.globals, "virtual_cam", True),
                               "Output swapped video to 'Face Swap Cam' Virtual Camera for WhatsApp/Chrome")
        self.sw_vcam.toggled.connect(lambda v: setattr(modules.globals, "virtual_cam", v))
        grid.addWidget(self.sw_vcam, 3, 2)

        grid.setColumnStretch(1, 1)
        return card

    # ── clothes try-on & mode selector card ──────────────────────────────

    def _build_clothes_tryon_card(self) -> QGroupBox:
        card = QGroupBox("Virtual Clothes Try-On & Mode Selector")
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        tryon = get_tryon_engine()
        if getattr(modules.globals, "garment_path", None) and os.path.isfile(modules.globals.garment_path):
            tryon.set_garment(modules.globals.garment_path)

        # Row 0: Mode selector + Load Garment button + Screenshot button
        grid.addWidget(QLabel("Live Mode:"), 0, 0)
        self.cb_tryon_mode = QComboBox()
        self._tryon_mode_options = [
            ("1. FACE SWAP (Face Only)", "face_swap"),
            ("2. CLOTHES TRY-ON (Keep Real Face)", "clothes_tryon"),
            ("3. BOTH (Face Swap + Clothes Try-On)", "both"),
        ]
        for lbl, _mk in self._tryon_mode_options:
            self.cb_tryon_mode.addItem(lbl)
        cur_m = getattr(modules.globals, "tryon_mode", "face_swap")
        m_idx = next((i for i, (_, mk) in enumerate(self._tryon_mode_options) if mk == cur_m), 0)
        self.cb_tryon_mode.setCurrentIndex(m_idx)
        self.cb_tryon_mode.currentIndexChanged.connect(self._on_tryon_mode_change)
        grid.addWidget(self.cb_tryon_mode, 0, 1)

        btn_load_g = QPushButton("Load Garment...")
        btn_load_g.clicked.connect(self._on_load_garment_file)
        grid.addWidget(btn_load_g, 0, 2)

        # Row 1: Quick Garment Switcher + Category Dropdown
        grid.addWidget(QLabel("Garment:"), 1, 0)
        self.cb_garment_select = QComboBox()
        self._refresh_garment_combo()
        self.cb_garment_select.currentIndexChanged.connect(self._on_garment_combo_change)
        grid.addWidget(self.cb_garment_select, 1, 1)

        self.cb_garment_cat = QComboBox()
        self.cb_garment_cat.addItems(list(GARMENT_CATEGORIES))
        self.cb_garment_cat.setCurrentText(tryon.params.category)
        self.cb_garment_cat.currentTextChanged.connect(self._on_garment_cat_change)
        grid.addWidget(self.cb_garment_cat, 1, 2)

        # Row 2 & 3: Compact Sliders (Scale, Width, Length, Offset X, Offset Y, Opacity)
        def _mk_slider(min_i, max_i, val_i, cb):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(min_i, max_i)
            s.setValue(val_i)
            s.valueChanged.connect(cb)
            return s

        s_grid = QGridLayout()
        s_grid.setHorizontalSpacing(8)
        s_grid.setVerticalSpacing(4)
        s_grid.addWidget(QLabel("Scale:"), 0, 0)
        self.sl_g_scale = _mk_slider(65, 145, 100, lambda v: tryon.update_params({"scale": v / 100.0}))
        s_grid.addWidget(self.sl_g_scale, 0, 1)

        s_grid.addWidget(QLabel("Width:"), 0, 2)
        self.sl_g_width = _mk_slider(65, 145, 100, lambda v: tryon.update_params({"width_scale": v / 100.0}))
        s_grid.addWidget(self.sl_g_width, 0, 3)

        s_grid.addWidget(QLabel("Length:"), 0, 4)
        self.sl_g_length = _mk_slider(65, 145, 100, lambda v: tryon.update_params({"length_scale": v / 100.0}))
        s_grid.addWidget(self.sl_g_length, 0, 5)

        s_grid.addWidget(QLabel("Off X:"), 1, 0)
        self.sl_g_ox = _mk_slider(-100, 100, 0, lambda v: tryon.update_params({"offset_x": float(v)}))
        s_grid.addWidget(self.sl_g_ox, 1, 1)

        s_grid.addWidget(QLabel("Off Y:"), 1, 2)
        self.sl_g_oy = _mk_slider(-100, 100, 0, lambda v: tryon.update_params({"offset_y": float(v)}))
        s_grid.addWidget(self.sl_g_oy, 1, 3)

        s_grid.addWidget(QLabel("Opacity:"), 1, 4)
        self.sl_g_op = _mk_slider(0, 100, 100, lambda v: tryon.update_params({"opacity": v / 100.0}))
        s_grid.addWidget(self.sl_g_op, 1, 5)

        grid.addLayout(s_grid, 2, 0, 1, 3)

        # Row 4: Calibrate Anchors, Save/Load Preset, Save Screenshot, Skeleton Toggle
        btn_row = QHBoxLayout()
        btn_calib = QPushButton("Calibrate Anchors")
        btn_calib.setObjectName("secondary")
        btn_calib.clicked.connect(self._on_calibrate_anchors)
        btn_row.addWidget(btn_calib)

        btn_save_p = QPushButton("Save Preset")
        btn_save_p.setObjectName("secondary")
        btn_save_p.clicked.connect(self._on_save_garment_preset)
        btn_row.addWidget(btn_save_p)

        btn_load_p = QPushButton("Load Preset")
        btn_load_p.setObjectName("secondary")
        btn_load_p.clicked.connect(self._on_load_garment_preset)
        btn_row.addWidget(btn_load_p)

        btn_shot = QPushButton("Save Screenshot")
        btn_shot.setObjectName("secondary")
        btn_shot.clicked.connect(self._on_save_screenshot)
        btn_row.addWidget(btn_shot)

        chk_skel = QCheckBox("Skeleton")
        chk_skel.toggled.connect(lambda v: tryon.update_params({"show_skeleton": v}))
        btn_row.addWidget(chk_skel)

        grid.addLayout(btn_row, 3, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        return card

    def _refresh_garment_combo(self) -> None:
        tryon = get_tryon_engine()
        self._garment_items = tryon.list_garments()
        self.cb_garment_select.blockSignals(True)
        self.cb_garment_select.clear()
        cur_name = tryon.current_garment.name if tryon.current_garment else ""
        sel_idx = 0
        for i, item in enumerate(self._garment_items):
            self.cb_garment_select.addItem(f"{item['filename']} ({item['category']})")
            if item["filename"] == cur_name:
                sel_idx = i
        self.cb_garment_select.setCurrentIndex(sel_idx)
        self.cb_garment_select.blockSignals(False)

    def _on_tryon_mode_change(self, idx: int) -> None:
        if 0 <= idx < len(self._tryon_mode_options):
            lbl, mk = self._tryon_mode_options[idx]
            modules.globals.tryon_mode = mk
            update_status(f"Switched live mode to: {lbl}")

    def _on_garment_combo_change(self, idx: int) -> None:
        if 0 <= idx < len(getattr(self, "_garment_items", [])):
            item = self._garment_items[idx]
            tryon = get_tryon_engine()
            ok, msg = tryon.set_garment(item["path"], item["category"])
            self.cb_garment_cat.blockSignals(True)
            self.cb_garment_cat.setCurrentText(tryon.params.category)
            self.cb_garment_cat.blockSignals(False)
            update_status(msg)

    def _on_garment_cat_change(self, cat: str) -> None:
        if cat in GARMENT_CATEGORIES:
            get_tryon_engine().update_params({"category": cat})
            update_status(f"Garment category set to: {cat}")

    def _on_load_garment_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Garment Image (PNG/JPG/WEBP)", GARMENTS_DIR, _IMAGE_FILE_FILTER
        )
        if path and os.path.isfile(path):
            import shutil
            filename = os.path.basename(path)
            dest_path = os.path.join(GARMENTS_DIR, filename)
            if os.path.abspath(path) != os.path.abspath(dest_path) and not os.path.exists(dest_path):
                try:
                    shutil.copy2(path, dest_path)
                    path = dest_path
                except Exception:
                    pass
            tryon = get_tryon_engine()
            ok, msg = tryon.set_garment(path, self.cb_garment_cat.currentText())
            self._refresh_garment_combo()
            update_status(msg)

    def _on_calibrate_anchors(self) -> None:
        dlg = GarmentAnchorCalibrationDialog(self)
        dlg.exec()

    def _on_save_garment_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Garment Preset JSON", GARMENTS_PRESETS_DIR, "JSON Preset (*.json)"
        )
        if path:
            name = os.path.splitext(os.path.basename(path))[0]
            ok, msg = get_tryon_engine().save_preset(name)
            update_status(msg)

    def _on_load_garment_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Garment Preset JSON", GARMENTS_PRESETS_DIR, "JSON Preset (*.json)"
        )
        if path and os.path.isfile(path):
            tryon = get_tryon_engine()
            ok, msg = tryon.load_preset(os.path.basename(path))
            p = tryon.params
            self.cb_garment_cat.setCurrentText(p.category)
            self.sl_g_scale.setValue(int(round(p.scale * 100)))
            self.sl_g_width.setValue(int(round(p.width_scale * 100)))
            self.sl_g_length.setValue(int(round(p.length_scale * 100)))
            self.sl_g_ox.setValue(int(round(p.offset_x)))
            self.sl_g_oy.setValue(int(round(p.offset_y)))
            self.sl_g_op.setValue(int(round(p.opacity * 100)))
            self._refresh_garment_combo()
            update_status(msg)

    def _on_save_screenshot(self) -> None:
        global _LAST_LIVE_FRAME
        if _LAST_LIVE_FRAME is None:
            update_status("Start Live preview first to capture a screenshot.")
            return
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"screenshot_{getattr(modules.globals, 'tryon_mode', 'live')}_{ts}.png"
        out_path = os.path.join(SCREENSHOTS_DIR, fn)
        imwrite_unicode(out_path, _LAST_LIVE_FRAME)
        update_status(f"Saved screenshot: screenshots/{fn}")

    def _on_refresh_cameras(self) -> None:
        self._camera_indices, self._camera_names = get_available_cameras()
        self.cb_camera.clear()
        self.cb_camera.addItems(self._camera_names)
        self.cb_camera.setEnabled(True)
        self.btn_live.setEnabled(True)
        update_status(f"Detected cameras: {', '.join(self._camera_names)}")

    def _on_profile_change(self, idx: int) -> None:
        if 0 <= idx < len(self._profile_options):
            _lbl, pk = self._profile_options[idx]
            modules.globals.apply_profile(pk)
            reset_face_analyser()
            # Sync resolution & det_size dropdowns
            r_idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == tuple(modules.globals.capture_resolution)), 1)
            self.cb_resolution.blockSignals(True)
            self.cb_resolution.setCurrentIndex(r_idx)
            self.cb_resolution.blockSignals(False)
            if modules.globals.det_size in self._det_size_options:
                self.cb_det_size.blockSignals(True)
                self.cb_det_size.setCurrentIndex(self._det_size_options.index(modules.globals.det_size))
                self.cb_det_size.blockSignals(False)
            update_status(
                f"Applied '{pk.upper()}' profile: {modules.globals.capture_resolution[0]}x{modules.globals.capture_resolution[1]}, "
                f"det_size={modules.globals.det_size}, frame_skip={modules.globals.frame_skip}"
            )

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            # Pre-translate the static template; append the dynamic size after
            # so the message still participates in localization (update_status
            # only translates whole-string keys).
            update_status(
                _("Capture resolution set (applies on next Live start):")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("Face detection size set (analyser re-inits on next call):")
                + f" {v}x{v}"
            )

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(path, (200, 200)))
            self.source_label.setText("")
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/random-person.jpeg",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"expected an image, got {content_type!r}")
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            staging_path = f"{temp_path}.part"
            # Download into a staging file and validate there. Writing straight
            # to temp_path would destroy the image currently in use whenever a
            # later fetch comes back bad, and content-type alone does not prove
            # the bytes decode.
            try:
                with open(staging_path, "wb") as f:
                    f.write(response.content)
                pixmap = render_image_preview(staging_path, (200, 200))
                if imread_unicode(staging_path) is None:
                    raise ValueError("downloaded image could not be decoded")
                os.replace(staging_path, temp_path)
            except Exception:
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
                raise
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(pixmap)
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (200, 200)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        cur_mode = getattr(modules.globals, "tryon_mode", "face_swap")
        if not modules.globals.map_faces:
            if cur_mode in ("face_swap", "both") and modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            if cur_mode in ("face_swap", "both"):
                from modules.face_analyser import get_face_analyser
                from modules.processors.frame.face_swapper import get_face_swapper
                get_face_analyser()
                get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()

    def keyPressEvent(self, event) -> None:
        text = event.text().lower()
        if text == "m":
            if hasattr(self, "cb_tryon_mode"):
                cur_idx = self.cb_tryon_mode.currentIndex()
                next_idx = (cur_idx + 1) % len(self._tryon_mode_options)
                self.cb_tryon_mode.setCurrentIndex(next_idx)
                return
        elif text == "s":
            self._on_save_screenshot()
            return
        elif text in ("1", "2", "3", "4", "5"):
            idx = int(text) - 1
            if hasattr(self, "cb_garment_select") and idx < self.cb_garment_select.count():
                self.cb_garment_select.setCurrentIndex(idx)
                return
        elif text == "[":
            if hasattr(self, "cb_garment_select"):
                cur = self.cb_garment_select.currentIndex()
                cnt = self.cb_garment_select.count()
                if cnt > 0:
                    self.cb_garment_select.setCurrentIndex((cur - 1 + cnt) % cnt)
                    return
        elif text == "]":
            if hasattr(self, "cb_garment_select"):
                cur = self.cb_garment_select.currentIndex()
                cnt = self.cb_garment_select.count()
                if cnt > 0:
                    self.cb_garment_select.setCurrentIndex((cur + 1) % cnt)
                    return
        super().keyPressEvent(event)


class GarmentAnchorCalibrationDialog(QDialog):
    """Interactive 6-point click-calibration dialog for the Desktop PyQt6 UI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate 6 Garment Anchors (Click Image to Set Selected Anchor)")
        self.resize(620, 660)
        self.tryon = get_tryon_engine()
        self.garment = self.tryon.current_garment
        self.anchors = (
            {k: list(v) for k, v in self.garment.anchors.items()}
            if self.garment is not None
            else {}
        )
        self.active_idx = 0

        layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Select Anchor Point to Place:"))
        self.cb_anchor = QComboBox()
        for i, name in enumerate(ANCHOR_NAMES):
            self.cb_anchor.addItem(f"{i+1}. {name}")
        self.cb_anchor.currentIndexChanged.connect(self._on_anchor_sel)
        top_row.addWidget(self.cb_anchor, 1)
        layout.addLayout(top_row)

        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setFixedSize(512, 512)
        self.canvas_label.mousePressEvent = self._on_canvas_click
        layout.addWidget(self.canvas_label, alignment=Qt.AlignmentFlag.AlignCenter)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset Auto Anchors")
        btn_reset.setObjectName("secondary")
        btn_reset.clicked.connect(self._on_reset)
        btn_row.addWidget(btn_reset)

        btn_apply = QPushButton("Apply & Save Anchors")
        btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(btn_apply)
        layout.addLayout(btn_row)
        self._redraw()

    def _on_anchor_sel(self, idx: int) -> None:
        self.active_idx = max(0, min(len(ANCHOR_NAMES) - 1, idx))
        self._redraw()

    def _on_canvas_click(self, event) -> None:
        if self.garment is None:
            return
        gh, gw = self.garment.shaded_rgba.shape[:2]
        pos = event.position()
        gx = float(np.clip(pos.x() * gw / 512.0, 0, gw - 1))
        gy = float(np.clip(pos.y() * gh / 512.0, 0, gh - 1))
        key = ANCHOR_NAMES[self.active_idx]
        self.anchors[key] = [round(gx, 1), round(gy, 1)]
        self.active_idx = (self.active_idx + 1) % len(ANCHOR_NAMES)
        self.cb_anchor.setCurrentIndex(self.active_idx)
        self._redraw()

    def _on_reset(self) -> None:
        if self.garment is not None:
            self.anchors = {k: list(v) for k, v in self.garment.auto_anchors.items()}
            self._redraw()

    def _on_apply(self) -> None:
        ok, msg = self.tryon.set_custom_anchors(self.anchors)
        update_status(msg)
        self.accept()

    def _redraw(self) -> None:
        if self.garment is None:
            return
        rgba = self.garment.shaded_rgba
        gh, gw = rgba.shape[:2]
        vis = np.full((gh, gw, 3), 36, dtype=np.uint8)
        a = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
        vis = np.clip(vis * (1.0 - a) + rgba[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)
        for i, name in enumerate(ANCHOR_NAMES):
            pt = self.anchors.get(name, [0, 0])
            col = (0, 255, 0) if i == self.active_idx else (0, 200, 255)
            p_i = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(vis, p_i, 9 if i == self.active_idx else 6, col, -1, cv2.LINE_AA)
            cv2.putText(vis, f"{i+1}.{name}", (p_i[0] + 8, p_i[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vis_resized = cv2.resize(vis, (512, 512), interpolation=cv2.INTER_AREA)
        self.canvas_label.setPixmap(_bgr_to_qpixmap(vis_resized))


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_frame = imread_unicode(modules.globals.source_path)
        if source_frame is None:
            update_status(
                f"Could not read source image: {modules.globals.source_path}"
            )
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(get_one_face(source_frame), temp_frame)
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance + Virtual Clothes Try-On, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        global _LAST_LIVE_FRAME
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        tryon_engine = get_tryon_engine()
        source_image = None
        last_source_path = None
        reported_source_error = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        det_interval = 1 if getattr(modules.globals, "frame_skip", 0) > 0 else max(1, round(self._fps * 0.08))
        vcam = None

        try:
          while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            cur_mode = getattr(modules.globals, "tryon_mode", "face_swap")
            run_face_swap = cur_mode in ("face_swap", "both")
            run_clothes_tryon = cur_mode in ("clothes_tryon", "both")

            if run_face_swap and not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    source_frame = imread_unicode(modules.globals.source_path)
                    if source_frame is None:
                        source_image = None
                        last_source_path = None
                        if reported_source_error != modules.globals.source_path:
                            reported_source_error = modules.globals.source_path
                            update_status(
                                f"Could not read source image: {modules.globals.source_path}"
                            )
                    else:
                        last_source_path = modules.globals.source_path
                        reported_source_error = None
                        source_image = get_one_face(source_frame)

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            elif run_face_swap:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            if run_clothes_tryon:
                temp_frame = tryon_engine.process_frame(temp_frame)

            _LAST_LIVE_FRAME = temp_frame

            # Send clean (un-watermarked) frame to Face Swap Cam DirectShow virtual camera
            if getattr(modules.globals, "virtual_cam", True):
                try:
                    fh, fw = temp_frame.shape[:2]
                    if vcam is None or vcam.width != fw or vcam.height != fh:
                        if vcam is not None:
                            vcam.close()
                        import pyvirtualcam
                        vcam = pyvirtualcam.Camera(
                            width=fw,
                            height=fh,
                            fps=int(getattr(modules.globals, "capture_fps", 30) or 30),
                            fmt=pyvirtualcam.PixelFormat.BGR,
                            backend="unitycapture",
                        )
                        print(f"[VirtualCam] Streaming {fw}x{fh} to '{vcam.device}'", flush=True)
                    vcam.send(temp_frame)
                except Exception as vcam_err:
                    if frame_count == 0:
                        print(f"[VirtualCam] Warning: {vcam_err}", flush=True)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            preview_frame = temp_frame
            if modules.globals.show_fps:
                preview_frame = temp_frame.copy()
                vcam_tag = "VCam:ON" if vcam is not None else "VCam:OFF"
                prof_tag = getattr(modules.globals, "profile", "balanced").upper()
                mode_tag = cur_mode.upper()
                cv2.putText(
                    preview_frame,
                    f"FPS: {fps:.1f} | {mode_tag} | {prof_tag} | {vcam_tag}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            try:
                self._pq.put_nowait(preview_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(preview_frame)
                except queue.Full:
                    pass
        finally:
            if vcam is not None:
                try:
                    vcam.close()
                except Exception:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        req_fps = int(getattr(modules.globals, "capture_fps", 30) or 30)
        if not self._cap.start(req_w, req_h, req_fps):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        # __init__ can bail out before these exist (e.g. the camera fails to open),
        # and closeEvent still runs.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        # Shutdown has to actually complete. Qt aborts the process when a
        # running QThread is destroyed, and holding a straggler in a module
        # global only moves that abort to interpreter exit. Both loops re-check
        # the stop flag between bounded operations (a camera read, a 50ms queue
        # get, one inference), so waiting terminates; a first-time model load
        # is simply slow, which is what the grace period absorbs quietly.
        for name in ("_capture_worker", "_processing_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            if not worker.wait(WORKER_SHUTDOWN_GRACE_MS):
                print(
                    f"[webcam] still waiting for {name[1:]} to stop...",
                    flush=True,
                )
                worker.wait()
        # Only now: releasing the capture while the capture thread could still
        # be inside cap.read() is not safe.
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()

    def keyPressEvent(self, event) -> None:
        global _MAIN
        if _MAIN is not None:
            _MAIN.keyPressEvent(event)
        else:
            super().keyPressEvent(event)


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    if _MAIN is not None:
        _WEBCAM_PREVIEW.move(max(40, _MAIN.x() + _MAIN.width() + 16), max(40, _MAIN.y()))
    _WEBCAM_PREVIEW.showNormal()
    _WEBCAM_PREVIEW.raise_()
    _WEBCAM_PREVIEW.activateWindow()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _close_live_windows() -> None:
    """Close every window that owns worker threads, so the threads stop."""
    for win in (_WEBCAM_PREVIEW, _LIVE_MAPPER):
        if win is None:
            continue
        try:
            win.close()
        except Exception:
            pass


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.showNormal()
        self._main.raise_()
        self._main.activateWindow()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    def _destroy_with_cleanup(*args, **kwargs):
        # core.destroy() ends in a bare quit(), which raises SystemExit and tears
        # the interpreter down without unwinding Qt. Destroying a running QThread
        # is fatal in Qt, so stop the live-preview workers before that happens.
        _close_live_windows()
        return destroy(*args, **kwargs)

    _MAIN = MainWindow(start, _destroy_with_cleanup)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
