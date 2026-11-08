"""
Comprehensive Automated Verification Suite for Real-Time Virtual Clothes Try-On & 3-Mode Selector.
Tests:
1. ClothesTryOnEngine edge cases (empty frame, tracking loss hold & fade-out, opacity 0/0.5/1, custom anchors, presets, invalid path, snapshot realistic).
2. All 3 modes (Face Swap, Clothes Try-On, Both) output verification (checking face preservation in Mode 2, garment preservation in Mode 2 & 3).
3. Flask web_app.py REST API endpoints via Flask test_client().
"""

import os
import sys
import json
import cv2
import numpy as np

# Preload CUDA/cuDNN DLLs for ONNX Runtime GPU
venv_base = os.path.dirname(os.path.dirname(sys.executable))
site_packages = os.path.join(venv_base, "Lib", "site-packages")
nvidia_base = os.path.join(site_packages, "nvidia")
if os.path.isdir(nvidia_base):
    for pkg in os.listdir(nvidia_base):
        bin_dir = os.path.join(nvidia_base, pkg, "bin")
        if os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass

import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(directory="")
    except Exception:
        pass

from modules.clothes_tryon import ClothesTryOnEngine, ensure_sample_garments
import web_app


def test_engine_edge_cases():
    print("=== 1. Testing ClothesTryOnEngine Edge Cases ===")
    garments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garments")
    samples = ensure_sample_garments(garments_dir)
    assert len(samples) >= 4, f"Expected >=4 built-in sample garments, got {len(samples)}"

    engine = ClothesTryOnEngine()
    ok, msg = engine.load_garment(samples[0])
    assert ok, f"Failed to load first sample garment: {msg}"

    # Load a real person frame
    person_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces", "custom_1793644362_Photo0075.jpg")
    person_frame = cv2.imread(person_img_path)
    assert person_frame is not None, "Failed to read test person image"
    person_frame = cv2.resize(person_frame, (640, 480))

    # 1a. Normal frame processing
    out_normal = engine.process_frame(person_frame)
    assert out_normal.shape == person_frame.shape, "Output shape mismatch"
    diff_normal = float(np.mean(np.abs(out_normal.astype(np.float32) - person_frame.astype(np.float32))))
    print(f"  [PASS] Normal frame try-on mean pixel diff: {diff_normal:.2f}")
    assert diff_normal > 2.0, "Expected visible garment on normal frame"

    # 1b. Tracking loss hold (8 pose checks) + fade-out (10 pose checks) on empty black frames
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hold_diffs = []
    for i in range(42):
        out_black = engine.process_frame(black_frame)
        hold_diffs.append(float(np.mean(out_black)))
    # Early frames after person loss should still hold the garment (>0), and eventually fade out to 0.0
    assert hold_diffs[0] > 0.1, f"Expected hold on first lost frame, got {hold_diffs[0]}"
    assert hold_diffs[-1] == 0.0, f"Expected full fade-out to 0.0 after hold+fade window, got {hold_diffs[-1]}"
    print(f"  [PASS] Tracking-loss hold & smooth fade-out verified (frame[0]={hold_diffs[0]:.2f} -> frame[-1]={hold_diffs[-1]:.2f})")

    # Re-acquire person tracking
    _ = engine.process_frame(person_frame)
    _ = engine.process_frame(person_frame)

    # 1c. Opacity boundary values (0.0 vs 0.5 vs 1.0)
    engine.update_params(opacity=0.0)
    out_op0 = engine.process_frame(person_frame)
    diff_op0 = float(np.max(np.abs(out_op0.astype(np.int16) - person_frame.astype(np.int16))))
    assert diff_op0 == 0.0, f"Expected 0 diff when opacity=0.0, got {diff_op0}"

    engine.update_params(opacity=0.5)
    out_op50 = engine.process_frame(person_frame)
    diff_op50 = float(np.mean(np.abs(out_op50.astype(np.float32) - person_frame.astype(np.float32))))

    engine.update_params(opacity=1.0)
    out_op100 = engine.process_frame(person_frame)
    diff_op100 = float(np.mean(np.abs(out_op100.astype(np.float32) - person_frame.astype(np.float32))))
    assert 0.0 < diff_op50 < diff_op100, f"Expected 0 < diff(0.5)={diff_op50:.2f} < diff(1.0)={diff_op100:.2f}"
    print(f"  [PASS] Opacity boundary test (0.0 -> {diff_op0:.1f}, 0.5 -> {diff_op50:.2f}, 1.0 -> {diff_op100:.2f})")

    # 1d. Scale/width/length/offset extreme boundary clamping
    engine.update_params(scale=99.0, width_scale=0.01, length_scale=99.0, offset_x=-999, offset_y=999)
    assert engine.params.scale == 1.8
    assert engine.params.width_scale == 0.5
    assert engine.params.length_scale == 1.8
    assert engine.params.offset_x == -200.0
    assert engine.params.offset_y == 200.0
    out_extreme = engine.process_frame(person_frame)
    assert out_extreme.shape == person_frame.shape
    engine.update_params(scale=1.0, width_scale=1.0, length_scale=1.0, offset_x=0, offset_y=0, opacity=1.0)
    print("  [PASS] Extreme parameter clamping & off-screen ROI bounds safety verified")

    # 1e. Custom 6-point anchor calibration & reset
    custom_anchors = {
        "left_shoulder": [0.28, 0.14],
        "right_shoulder": [0.72, 0.14],
        "left_hem": [0.30, 0.88],
        "right_hem": [0.70, 0.88],
        "left_sleeve_end": [0.10, 0.42],
        "right_sleeve_end": [0.90, 0.42],
    }
    ok_c, _ = engine.set_custom_anchors(custom_anchors)
    assert ok_c
    state_after_custom = engine.get_state_dict()
    assert abs(state_after_custom["anchors"]["left_shoulder"][0] - 0.28) < 1e-4
    ok_r, _ = engine.reset_anchors()
    assert ok_r
    print("  [PASS] Custom 6-point anchor calibration & reset verified")

    # 1f. Save & load preset
    ok_s, _ = engine.save_preset("unit_test_preset")
    assert ok_s
    preset_path = os.path.join(garments_dir, "presets", "unit_test_preset.json")
    assert os.path.isfile(preset_path)
    ok_l, _ = engine.load_preset("unit_test_preset.json")
    assert ok_l
    try:
        os.remove(preset_path)
    except Exception:
        pass
    print("  [PASS] Preset save & load verified")

    # 1g. Invalid garment path handling
    ok_inv, _ = engine.load_garment("non_existent_garment_12345.png")
    assert ok_inv is False
    print("  [PASS] Invalid garment path handled gracefully")

    # 1h. High-Detail Snapshot Realistic Mode
    snap_hq, snap_report = engine.process_snapshot_realistic(person_frame)
    assert snap_hq.shape == person_frame.shape
    assert "elapsed_ms" in snap_report
    diff_hq = float(np.mean(np.abs(snap_hq.astype(np.float32) - out_op100.astype(np.float32))))
    assert diff_hq > 0.2, "Snapshot realistic should apply cloth drape fold lighting & edge refinement"
    print(f"  [PASS] Snapshot Realistic mode verified (drape refinement delta={diff_hq:.2f}, elapsed={snap_report['elapsed_ms']} ms)")


def test_all_3_modes_pipeline_verification():
    print("\n=== 2. Testing All 3 Modes Pixel Verification (Face Swap vs. Try-On vs. Both) ===")
    from modules.face_analyser import get_one_face, detect_one_face_fast, get_face_analyser
    from modules.processors.frame.face_swapper import swap_face, get_face_swapper
    from modules.clothes_tryon import ClothesTryOnEngine, GARMENTS_DIR
    import modules.globals

    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    get_face_analyser()
    get_face_swapper()

    person_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces", "custom_1793644362_Photo0075.jpg")
    person_frame = cv2.resize(cv2.imread(person_img_path), (640, 480))
    source_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces", "default_character.jpg")
    source_face = get_one_face(cv2.imread(source_img_path))
    assert source_face is not None, "Failed to load source face"

    target_face = detect_one_face_fast(person_frame)
    assert target_face is not None, "Failed to detect face on test person image"
    bx1, by1, bx2, by2 = [int(v) for v in target_face.bbox]

    tryon_engine = ClothesTryOnEngine()
    g_path = os.path.join(GARMENTS_DIR, "classic_navy_tshirt.png")
    ok, _ = tryon_engine.load_garment(g_path, "t-shirt")
    assert ok, "Failed to load sample garment for mode testing"

    # Torso region: below chin, central chest
    torso_y1, torso_y2 = by2 + 40, min(480, by2 + 220)
    torso_x1, torso_x2 = max(0, bx1 - 60), min(640, bx2 + 60)

    # --- Mode 1: Face Swap Only ---
    frame_m1 = swap_face(source_face, target_face, person_frame.copy())
    face_diff_m1 = float(np.mean(np.abs(frame_m1[by1:by2, bx1:bx2].astype(np.float32) - person_frame[by1:by2, bx1:bx2].astype(np.float32))))
    torso_diff_m1 = float(np.mean(np.abs(frame_m1[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32) - person_frame[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32))))
    assert face_diff_m1 > 10.0, f"Mode 1: expected face swapped, diff={face_diff_m1}"
    assert torso_diff_m1 == 0.0, f"Mode 1: expected torso unchanged (0.0), diff={torso_diff_m1}"
    print(f"  [PASS] Mode 1 (Face Swap): Face altered (diff={face_diff_m1:.2f}), Torso preserved (diff={torso_diff_m1:.2f})")

    # --- Mode 2: Clothes Try-On Only (Face preserved, Garment applied) ---
    frame_m2 = tryon_engine.process_frame(person_frame.copy())
    face_diff_m2 = float(np.mean(np.abs(frame_m2[by1:by2, bx1:bx2].astype(np.float32) - person_frame[by1:by2, bx1:bx2].astype(np.float32))))
    torso_diff_m2 = float(np.mean(np.abs(frame_m2[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32) - person_frame[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32))))
    assert face_diff_m2 == 0.0, f"Mode 2: expected real face completely untouched (0.0), diff={face_diff_m2}"
    assert torso_diff_m2 > 5.0, f"Mode 2: expected torso with virtual garment, diff={torso_diff_m2}"
    print(f"  [PASS] Mode 2 (Clothes Try-On): Real face untouched (diff={face_diff_m2:.2f}), Torso warped (diff={torso_diff_m2:.2f})")

    # --- Mode 3: Both (Face Swap + Clothes Try-On) ---
    swapped_m3 = swap_face(source_face, target_face, person_frame.copy())
    frame_m3 = tryon_engine.process_frame(swapped_m3)
    face_diff_m3 = float(np.mean(np.abs(frame_m3[by1:by2, bx1:bx2].astype(np.float32) - person_frame[by1:by2, bx1:bx2].astype(np.float32))))
    torso_diff_m3 = float(np.mean(np.abs(frame_m3[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32) - person_frame[torso_y1:torso_y2, torso_x1:torso_x2].astype(np.float32))))
    assert face_diff_m3 > 10.0, f"Mode 3: expected face swapped, diff={face_diff_m3}"
    assert torso_diff_m3 > 5.0, f"Mode 3: expected torso with virtual garment, diff={torso_diff_m3}"
    print(f"  [PASS] Mode 3 (Both): Face swapped (diff={face_diff_m3:.2f}), Torso warped (diff={torso_diff_m3:.2f})")


def test_web_app_endpoints():
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    print("\n=== 3. Testing web_app.py HTTP Server Endpoints ===")
    web_app.ENGINE.init_models()
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.WebHandler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    def http_get(path):
        with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as resp:
            return resp.status, resp.read()

    def http_post(path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    try:
        # GET /
        code, raw = http_get("/")
        assert code == 200
        html = raw.decode("utf-8")
        assert "1. Face Swap" in html and "2. Clothes Try-On" in html and "3. Both" in html
        assert "calibModal" in html or "calibCanvas" in html
        print("  [PASS] GET / HTML UI verified (contains 3-mode selector + 6-point calibration modal)")

        # GET /camera_test.html
        code_cam, raw_cam = http_get("/camera_test.html")
        assert code_cam == 200
        assert "Face Swap Cam" in raw_cam.decode("utf-8")
        print("  [PASS] GET /camera_test.html WebRTC test page served correctly")

        # GET /api/status
        code, raw = http_get("/api/status")
        assert code == 200
        st = json.loads(raw.decode("utf-8"))
        assert "mode" in st and "fps" in st
        print(f"  [PASS] GET /api/status (initial mode={st['mode']})")

        # POST /api/mode across all 3 modes
        for mode in ["face_swap", "clothes_tryon", "both"]:
            code, data = http_post("/api/mode", {"mode": mode})
            assert code == 200 and data["ok"] is True and data["mode"] == mode
        print("  [PASS] POST /api/mode verified for 'face_swap', 'clothes_tryon', 'both'")

        # GET /api/tryon_state
        code, raw = http_get("/api/tryon_state")
        assert code == 200
        ts = json.loads(raw.decode("utf-8"))
        assert len(ts["garments"]) >= 4
        print(f"  [PASS] GET /api/tryon_state ({len(ts['garments'])} garments available)")

        # POST /api/select_garment
        g_fn = ts["garments"][1]["filename"]
        code, data = http_post("/api/select_garment", {"filename": g_fn})
        assert code == 200 and data["ok"] is True
        print(f"  [PASS] POST /api/select_garment ({g_fn})")

        # POST /api/tryon_settings
        code, data = http_post("/api/tryon_settings", {
            "category": "hoodie",
            "scale": 1.08,
            "width_scale": 1.04,
            "length_scale": 1.02,
            "offset_x": 5,
            "offset_y": -4,
            "opacity": 0.95,
            "lighting_match": True,
            "foreground_arms": True,
            "cover_original": True,
        })
        assert code == 200 and data["ok"] is True
        params = data["state"]["params"]
        assert abs(params["scale"] - 1.08) < 1e-4 and params["category"] == "hoodie"
        print("  [PASS] POST /api/tryon_settings verified")

        # GET /api/garment_cutout
        code, raw = http_get("/api/garment_cutout")
        assert code == 200
        cut = json.loads(raw.decode("utf-8"))
        assert cut["ok"] is True and len(cut["png_b64"]) > 100
        assert "anchors" in cut and len(cut["anchors"]) == 6
        print("  [PASS] GET /api/garment_cutout verified")

        # POST /api/tryon_anchors & POST /api/reset_anchors
        code, data = http_post("/api/tryon_anchors", {"anchors": cut["anchors"]})
        assert code == 200 and data["ok"] is True
        code, data = http_post("/api/reset_anchors", {})
        assert code == 200 and data["ok"] is True
        print("  [PASS] POST /api/tryon_anchors & /api/reset_anchors verified")

        # POST /api/save_preset & POST /api/load_preset
        code, data = http_post("/api/save_preset", {"name": "api_test_preset"})
        assert code == 200 and data["ok"] is True
        code, data = http_post("/api/load_preset", {"filename": "api_test_preset.json"})
        assert code == 200 and data["ok"] is True
        test_preset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garments", "presets", "api_test_preset.json")
        if os.path.isfile(test_preset_file):
            os.remove(test_preset_file)
        print("  [PASS] POST /api/save_preset & /api/load_preset verified")

        # Reset settings back to clean defaults
        http_post("/api/tryon_settings", {
            "scale": 1.0, "width_scale": 1.0, "length_scale": 1.0,
            "offset_x": 0, "offset_y": 0, "opacity": 1.0
        })

        # Seed a test frame into ENGINE and verify /api/screenshot & /api/snapshot_realistic
        person_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces", "custom_1793644362_Photo0075.jpg")
        test_frame = cv2.resize(cv2.imread(person_img_path), (640, 480))
        with web_app.ENGINE.lock:
            web_app.ENGINE.latest_pre_tryon_bgr = test_frame.copy()
            web_app.ENGINE.latest_bgr = web_app.ENGINE.tryon.process_frame(test_frame)

        code, scr_data = http_post("/api/screenshot", {})
        assert code == 200 and scr_data["ok"] is True
        scr_path = scr_data["path"]
        assert os.path.isfile(scr_path)
        os.remove(scr_path)

        code, snap_data = http_post("/api/snapshot_realistic", {})
        assert code == 200 and snap_data["ok"] is True
        assert len(snap_data["preview_b64"]) > 100
        assert os.path.isfile(snap_data["path"])
        os.remove(snap_data["path"])
        print("  [PASS] POST /api/screenshot & /api/snapshot_realistic verified")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_engine_edge_cases()
    test_all_3_modes_pipeline_verification()
    test_web_app_endpoints()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
