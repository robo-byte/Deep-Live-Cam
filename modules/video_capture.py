import os
import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        self._is_demo_video = (device_index < 0)
        self._target_size: Optional[Tuple[int, int]] = None
        self._last_read_time: float = 0.0
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows" and not self._is_demo_video:
            self.graph = FilterGraph()
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                # Fall back to demo video if physical device is unplugged
                print(
                    f"[VideoCapturer] Device index {device_index} not connected "
                    f"(found {len(devices)} devices). Falling back to Demo Camera video."
                )
                self._is_demo_video = True

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """Initialize and start video capture"""
        try:
            self._target_size = (width, height)
            if self._is_demo_video:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                media_name = (
                    "test_webcam_480x360.mp4" if width <= 480 else "test_webcam_640x480.mp4"
                )
                video_path = os.path.join(root_dir, "media", media_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(root_dir, "media", "ludwig.gif")
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open demo video: {video_path}")
                self.actual_width = width
                self.actual_height = height
                self.actual_fps = float(fps or 30.0)
                print(
                    f"[VideoCapturer] Demo Camera ({os.path.basename(video_path)}) "
                    f"{self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f}fps",
                    flush=True,
                )
                self.is_running = True
                return True

            if platform.system() == "Windows":
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),
                    (self.device_index, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            elif platform.system() == "Linux":
                self.cap = cv2.VideoCapture(f"/dev/video{self.device_index}")
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = self._measure_fps(warmup=3, sample=8,
                                                fallback=reported_fps or fps)

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        if self._is_demo_video and self.actual_fps > 0:
            now = time.perf_counter()
            frame_period = 1.0 / self.actual_fps
            dt = now - self._last_read_time
            if dt < frame_period:
                time.sleep(max(0.0, frame_period - dt))
            self._last_read_time = time.perf_counter()

        ret, frame = self.cap.read()
        if not ret and self._is_demo_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret and frame is not None:
            if (
                self._target_size is not None
                and (frame.shape[1], frame.shape[0]) != self._target_size
            ):
                frame = cv2.resize(
                    frame, self._target_size, interpolation=cv2.INTER_AREA
                )
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS."""
        try:
            for _ in range(warmup):
                self.cap.read()
            t0 = time.perf_counter()
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret:
                    return fallback
            elapsed = time.perf_counter() - t0
            if elapsed <= 0:
                return fallback
            return sample / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
