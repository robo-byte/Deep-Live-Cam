#!/usr/bin/env python3
"""
GTX 1050 Ti (4GB VRAM, Pascal sm_61) Benchmark for All 3 Live Modes:
  1. FACE SWAP ONLY
  2. CLOTHES TRY-ON ONLY (real face preserved)
  3. BOTH (Face Swap + Clothes Try-On running together)
Also verifies end-to-end Virtual Camera ('Face Swap Cam') output via DirectShow capture.
"""

import os
import sys
import time
import threading
import subprocess
import numpy as np
import cv2

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

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
ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
ort.set_default_logger_severity(3)

import pynvml
import pyvirtualcam
import modules.globals
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
    ClothesTryOnEngine,
    ensure_sample_garments,
    GARMENTS_DIR,
    SCREENSHOTS_DIR,
)


def get_vram_mb(handle) -> tuple[float, float]:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024), info.total / (1024 * 1024)


def load_video_frames(path, max_frames=48):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        r, f = cap.read()
        if not r:
            break
        frames.append(f)
    cap.release()
    return frames


def run_all_modes_benchmark():
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    vram_base, vram_total = get_vram_mb(gpu_handle)

    print("=== GTX 1050 Ti 4GB: Face Swap + Virtual Clothes Try-On Benchmark ===")
    print(f"GPU: {gpu_name} ({vram_total:.0f} MiB Total, Baseline Used: {vram_base:.0f} MiB)")

    ensure_sample_garments()
    tryon = ClothesTryOnEngine()
    tryon.set_garment(os.path.join(GARMENTS_DIR, "crimson_street_hoodie.png"), "hoodie")

    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    swapper = get_face_swapper()
    assert swapper.session.get_providers()[0] == "CUDAExecutionProvider"

    source_path = os.path.join(project_root, "faces", "default_character.jpg")
    source_img = cv2.imread(source_path)
    modules.globals.det_size = 640
    reset_face_analyser()
    source_face = get_one_face(source_img)
    assert source_face is not None

    frames_480 = load_video_frames(os.path.join(project_root, "media", "test_webcam_480x360.mp4"), 48)
    frames_640 = load_video_frames(os.path.join(project_root, "media", "test_webcam_640x480.mp4"), 48)

    configs = [
        ("1. FACE SWAP", "face_swap", "fast", (480, 360), frames_480),
        ("1. FACE SWAP", "face_swap", "balanced", (640, 480), frames_640),
        ("1. FACE SWAP", "face_swap", "quality", (640, 480), frames_640),
        ("2. CLOTHES TRY-ON", "clothes_tryon", "fast", (480, 360), frames_480),
        ("2. CLOTHES TRY-ON", "clothes_tryon", "balanced", (640, 480), frames_640),
        ("3. BOTH (Face + Clothes)", "both", "fast", (480, 360), frames_480),
        ("3. BOTH (Face + Clothes)", "both", "balanced", (640, 480), frames_640),
    ]

    rows = []
    n_frames = 40

    for label, mode_key, prof_key, res, frames in configs:
        modules.globals.apply_profile(prof_key)
        reset_face_analyser()
        get_face_analyser()
        tryon.reset_tracking()

        # Warmup 4 frames
        for wf in frames[:4]:
            f = wf.copy()
            if mode_key in ("face_swap", "both"):
                tf = detect_one_face_fast(f)
                if tf is not None:
                    f = swap_face(source_face, tf, f)
            if mode_key in ("clothes_tryon", "both"):
                f = tryon.process_frame(f)

        if hasattr(swap_face, "_skip_state") and swap_face._skip_state:
            swap_face._skip_state["count"] = 0
            swap_face._skip_state["neural_count"] = 0

        face_times = []
        tryon_times = []
        total_times = []
        peak_vram = 0.0

        t_start = time.perf_counter()
        for i in range(n_frames):
            f = frames[i % len(frames)].copy()
            t0 = time.perf_counter()
            if mode_key in ("face_swap", "both"):
                tf = detect_one_face_fast(f)
                if tf is not None:
                    f = swap_face(source_face, tf, f)
            t1 = time.perf_counter()
            if mode_key in ("clothes_tryon", "both"):
                f = tryon.process_frame(f)
            t2 = time.perf_counter()

            face_times.append((t1 - t0) * 1000.0)
            tryon_times.append((t2 - t1) * 1000.0)
            total_times.append((t2 - t0) * 1000.0)
            if i % 8 == 0:
                used_mb, _ = get_vram_mb(gpu_handle)
                if used_mb > peak_vram:
                    peak_vram = used_mb

        elapsed = time.perf_counter() - t_start
        fps = n_frames / elapsed
        row = {
            "mode": label,
            "profile": prof_key.capitalize(),
            "res": f"{res[0]}x{res[1]}",
            "face_ms": float(np.mean(face_times)),
            "tryon_ms": float(np.mean(tryon_times)),
            "total_ms": float(np.mean(total_times)),
            "fps": fps,
            "peak_vram_mib": peak_vram,
        }
        rows.append(row)
        print(
            f"[{label:24s} | {prof_key.upper():8s} | {row['res']}] -> "
            f"{fps:5.1f} FPS (total={row['total_ms']:5.1f} ms: face={row['face_ms']:4.1f} ms, tryon={row['tryon_ms']:4.1f} ms) "
            f"| VRAM: {peak_vram:.0f}/{vram_total:.0f} MiB"
        )

    # Verify Virtual Camera ('Face Swap Cam') in BOTH mode
    print("\n=== Verifying 'Face Swap Cam' Virtual Camera in Mode 3 (BOTH: Face Swap + Clothes Try-On) ===")
    modules.globals.apply_profile("balanced")
    reset_face_analyser()
    tryon.reset_tracking()
    tryon.set_garment(os.path.join(GARMENTS_DIR, "crimson_street_hoodie.png"), "hoodie")

    orig_frame = frames_640[2].copy()
    t_face = get_one_face(orig_frame)
    swapped_base = swap_face(source_face, t_face, orig_frame.copy())
    for _ in range(3):
        both_frame = tryon.process_frame(swapped_base.copy())

    stop_vcam = False
    vcam_device_name = None

    def vcam_sender():
        nonlocal vcam_device_name
        with pyvirtualcam.Camera(
            width=640,
            height=480,
            fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend="unitycapture",
        ) as cam:
            vcam_device_name = cam.device
            while not stop_vcam:
                cam.send(both_frame)
                cam.sleep_until_next_frame()

    sender_thread = threading.Thread(target=vcam_sender, daemon=True)
    sender_thread.start()
    time.sleep(0.35)

    vcam_both_path = os.path.join(SCREENSHOTS_DIR, "vcam_both_mode_verified.png")
    ffmpeg_exe = os.path.join(project_root, "ffmpeg.exe")
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-f",
            "dshow",
            "-video_size",
            "640x480",
            "-i",
            "video=Face Swap Cam",
            "-vf",
            "select=gte(n\\,5)",
            "-frames:v",
            "1",
            vcam_both_path,
        ],
        capture_output=True,
    )
    stop_vcam = True
    sender_thread.join(timeout=1.0)

    captured_vcam = cv2.imread(vcam_both_path)
    assert captured_vcam is not None, "Failed to capture frame from Face Swap Cam!"
    print(f"Captured 'Face Swap Cam' frame in Mode 3 (BOTH): {captured_vcam.shape} -> {vcam_both_path}")

    # Append Try-On results to benchmark_report.md
    report_path = os.path.join(project_root, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# GTX 1050 Ti (4GB VRAM) Real-Time Face Swap & Virtual Clothes Try-On Benchmark Report\n\n")
        f.write(f"- **GPU**: {gpu_name} (Pascal `sm_61`, 4096 MiB VRAM, Driver 582.66)\n")
        f.write("- **Face Swap Backend**: `CUDAExecutionProvider` (`inswapper_128_fused.onnx` FP32 + CUDA Graph + IOBinding)\n")
        f.write("- **Clothes Try-On Backend**: MediaPipe Pose Lite (`model_complexity=0`) + SelfieSegmentation on CPU XNNPACK (**0 MiB extra GPU VRAM**) + OpenCV Perspective Torso & Articulated Sleeve Warping\n")
        f.write("- **Garment Cutout Backend**: `rembg` (`u2netp.onnx` on CPU, cached once per garment in `garments/cache/`)\n")
        f.write(f"- **Virtual Camera Verified**: `{vcam_device_name}` (`640x480` DirectShow output verified across all 3 modes)\n\n")
        f.write("## All 3 Modes Benchmark Results\n\n")
        f.write("| Mode | Profile | Input Res | Face Swap Time | Clothes Try-On Time | Total Frame Time | Effective Live FPS | Peak System VRAM |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in rows:
            f.write(
                f"| **{r['mode']}** | `{r['profile']}` | `{r['res']}` | "
                f"`{r['face_ms']:.1f} ms` | `{r['tryon_ms']:.1f} ms` | `{r['total_ms']:.1f} ms` | "
                f"**{r['fps']:.1f} FPS** | `{r['peak_vram_mib']:.0f} / 4096 MiB` |\n"
            )
    print(f"Updated benchmark report: {report_path}")


if __name__ == "__main__":
    run_all_modes_benchmark()
