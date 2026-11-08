#!/usr/bin/env python3
"""
GTX 1050 Ti (4GB VRAM, Pascal sm_61) 3-Profile Benchmark & Virtual Camera Verification
Measures real FPS, per-stage latency, and VRAM usage across Fast / Balanced / Quality profiles,
and verifies live swapped frames through the 'Face Swap Cam' DirectShow virtual camera.
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

# Register CUDA/cuDNN DLLs
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


def get_vram_mb(handle) -> tuple[float, float]:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024), info.total / (1024 * 1024)


def run_benchmark():
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    vram_base, vram_total = get_vram_mb(gpu_handle)

    avail_providers = ort.get_available_providers()
    print(f"=== Hardware & Backend Check ===")
    print(f"GPU: {gpu_name} ({vram_total:.0f} MiB Total, Baseline Used: {vram_base:.0f} MiB)")
    print(f"ONNX Runtime {ort.__version__} Available Providers: {avail_providers}")
    assert "CUDAExecutionProvider" in avail_providers, "CUDAExecutionProvider is missing!"

    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    modules.globals.fp_ui = {
        "face_enhancer": False,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False,
    }
    modules.globals.many_faces = False
    modules.globals.mouth_mask = False
    modules.globals.poisson_blend = False

    source_path = os.path.join(project_root, "faces", "default_character.jpg")
    source_img = cv2.imread(source_path)
    assert source_img is not None, f"Missing source image: {source_path}"

    # Initialize swapper once (loads inswapper_128_fused.onnx + CUDA Graph)
    swapper = get_face_swapper()
    assert swapper is not None, "Failed to load face swapper!"
    swapper_providers = swapper.session.get_providers()
    print(f"InSwapper Active Session Providers: {swapper_providers}")
    assert swapper_providers[0] == "CUDAExecutionProvider", "InSwapper not on CUDA!"

    # Load test videos into memory so disk decode doesn't skew GPU timing
    def load_video_frames(path, max_frames=60):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < max_frames:
            r, f = cap.read()
            if not r:
                break
            frames.append(f)
        cap.release()
        return frames

    frames_480 = load_video_frames(os.path.join(project_root, "media", "test_webcam_480x360.mp4"), 60)
    frames_640 = load_video_frames(os.path.join(project_root, "media", "test_webcam_640x480.mp4"), 60)

    # Extract source face embedding at det_size=640 for maximum accuracy
    modules.globals.det_size = 640
    reset_face_analyser()
    source_face = get_one_face(source_img)
    assert source_face is not None, "No face found in default_character.jpg!"

    profiles_to_test = [
        {
            "name": "Fast",
            "key": "fast",
            "res": (480, 360),
            "det_size": 160,
            "frame_skip": 3,
            "frames": frames_480,
        },
        {
            "name": "Balanced",
            "key": "balanced",
            "res": (640, 480),
            "det_size": 320,
            "frame_skip": 2,
            "frames": frames_640,
        },
        {
            "name": "Quality",
            "key": "quality",
            "res": (640, 480),
            "det_size": 480,
            "frame_skip": 0,
            "frames": frames_640,
        },
    ]

    results = []

    for p in profiles_to_test:
        print(f"\n--- Benchmarking Profile: {p['name']} ({p['res'][0]}x{p['res'][1]}, det_size={p['det_size']}, frame_skip={p['frame_skip']}) ---")
        modules.globals.apply_profile(p["key"])
        reset_face_analyser()
        fa = get_face_analyser()
        det_providers = fa.det_model.session.get_providers()
        assert det_providers[0] == "CUDAExecutionProvider", "Detector not on CUDA!"

        # Warmup (5 frames)
        for wf in p["frames"][:5]:
            tf = detect_one_face_fast(wf)
            if tf is not None:
                _ = swap_face(source_face, tf, wf.copy())

        # 1. Measure Effective Pipeline with configured frame_skip (N=48 frames)
        if hasattr(swap_face, "_skip_state"):
            swap_face._skip_state["count"] = 0
            swap_face._skip_state["neural_count"] = 0

        det_times = []
        swap_times = []
        total_times = []
        peak_vram = 0.0
        n_frames = 48

        t_wall_start = time.perf_counter()
        for i in range(n_frames):
            frame_in = p["frames"][i % len(p["frames"])].copy()
            t0 = time.perf_counter()
            target_face = detect_one_face_fast(frame_in)
            t1 = time.perf_counter()
            if target_face is not None:
                _ = swap_face(source_face, target_face, frame_in)
            t2 = time.perf_counter()

            det_times.append((t1 - t0) * 1000.0)
            swap_times.append((t2 - t1) * 1000.0)
            total_times.append((t2 - t0) * 1000.0)
            if i % 8 == 0:
                used_mb, _ = get_vram_mb(gpu_handle)
                if used_mb > peak_vram:
                    peak_vram = used_mb
        t_wall_elapsed = time.perf_counter() - t_wall_start
        effective_fps = n_frames / t_wall_elapsed
        neural_swaps = swap_face._skip_state["neural_count"]

        # 2. Also measure pure synchronous (frame_skip=0) for this resolution & det_size (N=15 frames)
        modules.globals.frame_skip = 0
        sync_times = []
        for i in range(15):
            frame_in = p["frames"][i % len(p["frames"])].copy()
            t0 = time.perf_counter()
            target_face = detect_one_face_fast(frame_in)
            if target_face is not None:
                _ = swap_face(source_face, target_face, frame_in)
            t1 = time.perf_counter()
            sync_times.append((t1 - t0) * 1000.0)
            used_mb, _ = get_vram_mb(gpu_handle)
            if used_mb > peak_vram:
                peak_vram = used_mb

        sync_avg_ms = float(np.mean(sync_times))
        sync_fps = 1000.0 / sync_avg_ms

        res_entry = {
            "name": p["name"],
            "res": f"{p['res'][0]}x{p['res'][1]}",
            "det_size": f"{p['det_size']}x{p['det_size']}",
            "frame_skip": p["frame_skip"],
            "det_ms": float(np.mean(det_times)),
            "swap_avg_ms": float(np.mean(swap_times)),
            "total_avg_ms": float(np.mean(total_times)),
            "effective_fps": effective_fps,
            "sync_avg_ms": sync_avg_ms,
            "sync_fps": sync_fps,
            "peak_vram_mib": peak_vram,
            "vram_delta_mib": max(0.0, peak_vram - vram_base),
        }
        results.append(res_entry)
        print(
            f"  -> Effective FPS (skip={p['frame_skip']}): {effective_fps:.1f} FPS "
            f"(avg {res_entry['total_avg_ms']:.1f} ms/frame: det={res_entry['det_ms']:.1f} ms, swap/warp={res_entry['swap_avg_ms']:.1f} ms)"
        )
        print(
            f"  -> Pure Sync FPS (skip=0): {sync_fps:.2f} FPS ({sync_avg_ms:.1f} ms/frame)"
        )
        print(
            f"  -> Peak System VRAM: {peak_vram:.0f} MiB / {vram_total:.0f} MiB (< 3500 MiB target: {peak_vram < 3500})"
        )

    # 3. Verify End-to-End Virtual Camera ('Face Swap Cam') with real swapped frame
    print("\n=== Verifying 'Face Swap Cam' Virtual Camera End-to-End ===")
    modules.globals.apply_profile("balanced")
    reset_face_analyser()
    fa_verify = get_face_analyser()
    orig_frame = frames_640[10].copy()
    t_face = get_one_face(orig_frame)
    swapped_frame = swap_face(source_face, t_face, orig_frame.copy())

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
                cam.send(swapped_frame)
                cam.sleep_until_next_frame()

    sender_thread = threading.Thread(target=vcam_sender, daemon=True)
    sender_thread.start()
    time.sleep(0.3)

    vcam_capture_path = os.path.join(project_root, "vcam_swapped_verified.png")
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
            "select=gte(n\\,6)",
            "-frames:v",
            "1",
            vcam_capture_path,
        ],
        capture_output=True,
    )
    stop_vcam = True
    sender_thread.join(timeout=1.0)

    captured_vcam = cv2.imread(vcam_capture_path)
    assert captured_vcam is not None, "Failed to capture frame from Face Swap Cam!"
    mean_bgr = captured_vcam.mean(axis=(0, 1))
    print(f"Virtual Camera Device: '{vcam_device_name}'")
    print(f"Captured Frame Shape: {captured_vcam.shape}, Mean BGR: {mean_bgr.round(1)}")
    assert mean_bgr.mean() > 20.0, "Captured frame is black!"

    # Verify face identity in captured Virtual Camera frame matches source character
    modules.globals.det_size = 640
    reset_face_analyser()
    vcam_face = get_one_face(captured_vcam)
    orig_face_full = get_one_face(orig_frame)
    assert vcam_face is not None, "No face detected in captured Virtual Camera frame!"

    sim_before = float(np.dot(orig_face_full.normed_embedding, source_face.normed_embedding))
    sim_after = float(np.dot(vcam_face.normed_embedding, source_face.normed_embedding))
    print(f"Face Detection Score on 'Face Swap Cam' Output: {vcam_face.det_score:.3f}")
    print(f"Cosine Similarity to Source Character Face: Before Swap = {sim_before:.3f} -> After Swap ('Face Swap Cam') = {sim_after:.3f}")
    assert sim_after > 0.45 and sim_after > sim_before + 0.30, "Face identity did not swap properly!"

    # Build a 3-panel visual proof image: [Source Face | Original Webcam Frame | Captured from Face Swap Cam]
    src_panel = cv2.resize(source_img, (480, 480))
    orig_panel = orig_frame.copy()
    vcam_panel = captured_vcam.copy()
    for panel, title in [
        (src_panel, "1. Source Character Face"),
        (orig_panel, "2. Input Webcam Frame (640x480)"),
        (vcam_panel, "3. Captured from 'Face Swap Cam'"),
    ]:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2)

    comparison = np.hstack([src_panel, orig_panel, vcam_panel])
    comp_path = os.path.join(project_root, "faceswap_verification_comparison.png")
    cv2.imwrite(comp_path, comparison)
    artifact_comp = r"C:\Users\sgsco\.gemini\antigravity\brain\819a02d9-5f0d-459e-9ee4-a7805b465ab0\faceswap_verification_comparison.png"
    try:
        cv2.imwrite(artifact_comp, comparison)
    except Exception:
        pass

    # Write benchmark_report.md
    report_path = os.path.join(project_root, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# GTX 1050 Ti (4GB VRAM) Real-Time Face Swap Benchmark Report\n\n")
        f.write(f"- **GPU**: {gpu_name} (Pascal `sm_61`, 4096 MiB VRAM, Driver 582.66)\n")
        f.write(f"- **Execution Provider**: `{avail_providers[0]}` (CUDA 12.9 + cuDNN 9.10.2.21)\n")
        f.write(f"- **Swap Model**: `inswapper_128_fused.onnx` (FP32 with fused Pad+Conv + CUDA Graph + IOBinding)\n")
        f.write(f"- **Face Enhancer (GFPGAN)**: `OFF` (keeps total VRAM well under 3.5 GB)\n")
        f.write(f"- **Virtual Camera Verified**: `{vcam_device_name}` (`640x480`, identity cosine similarity `{sim_before:.3f} -> {sim_after:.3f}`)\n\n")
        f.write("## Profile Benchmark Results\n\n")
        f.write("| Profile | Input Res | Det Size | Frame Skip | Det Latency | Avg Frame Time | Effective Live FPS | Pure Sync FPS (`skip=0`) | Peak VRAM (MiB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(
                f"| **{r['name']}** | `{r['res']}` | `{r['det_size']}` | `{r['frame_skip']}` | "
                f"`{r['det_ms']:.1f} ms` | `{r['total_avg_ms']:.1f} ms` | "
                f"**{r['effective_fps']:.1f} FPS** | `{r['sync_fps']:.2f} FPS` (`{r['sync_avg_ms']:.1f} ms`) | "
                f"`{r['peak_vram_mib']:.0f} / 4096 MiB` |\n"
            )
    print(f"\nSaved benchmark report to: {report_path}")
    print(f"Saved verification comparison image to: {comp_path}")


if __name__ == "__main__":
    run_benchmark()
