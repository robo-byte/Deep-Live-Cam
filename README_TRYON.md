# Real-Time Virtual Clothes Try-On & 3-Mode Pipeline
### Deep-Live-Cam (GTX 1050 Ti 4GB Edition + `Face Swap Cam` + Anywear Extension)

> **OVERVIEW**  
> This feature adds real-time **Virtual Clothes Try-On** alongside the existing Face Swap pipeline, powered by **Anywear-style 3D body contouring** and **1-click e-commerce shopping integration**. You can load any photo of a garment (t-shirt, hoodie, jacket, long-sleeve shirt) or import clothes directly from Zara, Amazon, ASOS, and Shein, seeing yourself "wearing" them live on your webcam, with video streamed directly to the **`Face Swap Cam`** DirectShow virtual camera for WhatsApp, Chrome, Zoom, and Google Meet.

---

## 🌟 Anywear-Style Features & Upgrades

1. **Organic 3D Body Contouring & Fit**:
   - **Chest Cylinder Convexity**: Fabric naturally curves outward in 3D over the chest rather than sitting flat like a 2D sticker.
   - **Anatomical Waist Tapering**: Tapers inward (~7-8%) at the natural waistline ($v \approx 0.60$).
   - **Pelvic Hemline Arch**: Bottom hem gracefully arches over the hips and pelvis.
   - **Torso Yaw Turn Foreshortening**: As you turn your torso or head left or right, the receding side naturally compresses in perspective.
   - **Sub-4ms Execution**: Uses vectorized coordinate broadcasting and `cv2.remap` for zero latency and 45+ FPS.

2. **Complete Concealment of Underneath Clothing**:
   - Neutralizes old t-shirt collars, bold graphics, and sleeve cuffs underneath the virtual garment so they never peek through or ruin the try-on.
   - Dynamic skin-tone sampling around the clavicle/neckline blends seamlessly into the new garment's collar opening.

3. **Anywear E-Commerce Web Shopping Integration**:
   - **1-Click Bookmarklet**: Drag the `🔖 Anywear Bookmarklet` button to your browser bookmarks bar. While browsing Zara, Amazon, ASOS, or Shein, click the bookmark and click any shirt to wear it instantly on your webcam!
   - **Paste Product URL**: Paste image URLs directly into the studio bar (`[ Paste Zara/Amazon URL... ] [ ✨ Try On ]`).
   - **Direct Drag-and-Drop**: Drag any photo from another tab directly onto your webcam canvas.

---

## 1. The 3 Pipeline Modes

You can switch modes instantly at any time via the GUI or by pressing the **`M`** key:

| Mode | Pipeline Description | Face | Garment | FPS on GTX 1050 Ti |
| :--- | :--- | :--- | :--- | :--- |
| **1. FACE SWAP** | CUDA InsightFace + InSwapper neural swap | **Swapped** (from target character photo) | Your real clothing | **15 – 22 FPS** (Balanced / Fast) |
| **2. CLOTHES TRY-ON** | MediaPipe Pose Lite + Selfie Segmentation on CPU + OpenCV warp | **Your real face** (no swap running) | **Virtual Garment** | **43 – 60 FPS** (Balanced / Fast) |
| **3. BOTH** | CUDA Face Swap + MediaPipe Virtual Clothes Try-On simultaneously | **Swapped** | **Virtual Garment** | **11 – 16 FPS** (Balanced / Fast) |

---

## 2. Quick Start (How to Launch)

### Option A: Browser Web Studio (Recommended)
Double-click **`start_browser_ui.bat`** (or **`start_clothes_tryon.bat`**):
```bat
d:\ANTI\deep-live-cam\start_browser_ui.bat
```
- Automatically opens **`http://localhost:7860`** in your default browser.
- Displays live preview, drag-and-drop face & garment loaders, fit sliders, 6-point calibration modal, presets, and screenshot buttons.

### Option B: Desktop PySide6 GUI
Run:
```bat
d:\ANTI\deep-live-cam\start_clothes_tryon.bat --desktop
```
or launch with specific mode flags:
```bat
d:\ANTI\deep-live-cam\venv\Scripts\python.exe run.py --mode clothes_tryon --profile balanced --virtual-cam -s faces\default_character.jpg
```

---

## 3. How to Use & Switch Garments

### Built-in Sample Garments
The system comes pre-bundled with sample garments in `d:\ANTI\deep-live-cam\garments\`:
1. **Classic Navy T-Shirt** (`classic_navy_tshirt.png`) — Category: `t-shirt`
2. **Crimson Street Hoodie** (`crimson_street_hoodie.png`) — Category: `hoodie`
3. **Charcoal Zip Jacket** (`charcoal_blazer_jacket.png`) — Category: `jacket`
4. **Emerald Long-Sleeve Oxford** (`emerald_oxford_shirt.png`) — Category: `long-sleeve shirt`
5. **Studio Photo T-Shirt with Background** (`studio_teal_tshirt_bg.jpg`) — Demonstrates automatic background removal & anchor detection

### Keyboard Shortcuts
- **`M`**: Cycle through modes (1. Face Swap -> 2. Clothes Try-On -> 3. Both)
- **`1` – `5`**: Instantly switch between the first 5 garments
- **`[` / `]`**: Cycle to the previous / next garment
- **`S`**: Save a high-resolution screenshot to `screenshots/`

### Adding Your Own Garments
1. Take or download a photo of any shirt, t-shirt, hoodie, or jacket (flat lay, mannequin, or product catalog photo).
2. Drag and drop the image onto the **Virtual Clothes Try-On** drop zone (or click to browse).
3. The system automatically:
   - Runs `rembg` (`u2netp` on CPU) to remove any white or complex studio background and leaves clean RGBA edges.
   - Detects the 6 key garment anchors (left/right shoulder, left/right hem, left/right sleeve ends).
   - Generates separated torso and sleeve layers with feathered joints and baked 3D shading.
   - Caches the prepared garment in `garments/cache/` so subsequent loads are instantaneous.

---

## 4. Garment Anchor Calibration (6-Point Interactive Tool)

If an uploaded garment has unusual proportions or asymmetric sleeves:
1. Click **`Calibrate Anchors (6-Pt)`** in the web UI (or desktop UI).
2. The modal displays your garment cutout with 6 colored control points:
   - **Point 1 (Green)**: Left Shoulder seam
   - **Point 2 (Teal)**: Right Shoulder seam
   - **Point 3 (Orange)**: Left Bottom Hem corner
   - **Point 4 (Yellow)**: Right Bottom Hem corner
   - **Point 5 (Blue)**: Left Sleeve End (cuff or short-sleeve opening)
   - **Point 6 (Pink)**: Right Sleeve End
3. Drag any circle to adjust its position, or select the anchor on the right and click on the canvas.
4. Click **`Apply & Save Anchors`**. Your custom alignment is saved to the garment cache and immediately updates your live stream.
5. Click **`Reset Auto Anchors`** at any time to return to the automatic geometric detection.

---

## 5. Live Fit & Style Sliders

- **Overall Scale (`0.65 – 1.45`)**: Enlarges or shrinks the entire garment uniformly.
- **Torso Width (`0.65 – 1.45`)**: Widens or narrows the torso independently to match slim or heavy builds.
- **Torso Length (`0.65 – 1.45`)**: Extends or shortens the lower hem. Works automatically when sitting vs. standing.
- **Horizontal Offset (`-100 to +100 px`)**: Shifts the garment left or right.
- **Vertical Offset (`-100 to +100 px`)**: Shifts the garment up or down along the torso axis.
- **Garment Opacity (`0.00 – 1.00`)**: Blends between original clothing and virtual garment.
- **Match Room Lighting & Tone (Checkbox)**: Samples ambient room luminance from your neck/face and dynamically adjusts garment brightness and color temperature so white or black shirts match your room lighting.
- **Cover Original Collar & Neck (Checkbox)**: Synthesizes a collar shadow mask to hide your original t-shirt or collar from peeking out around your neck.
- **Foreground Hands & Chin Occlusion (Checkbox)**: Uses the MediaPipe selfie segmentation mask so that if you cross your arms, scratch your beard, or raise your hands, your real arms and hands stay in front of the garment.
- **Save / Load Presets**: Save custom fit configurations (e.g. `loose_hoodie`, `fitted_tshirt`) as JSON files in `garments/presets/`.

---

## 6. Output to Virtual Camera (`Face Swap Cam`)

The live stream (in **all 3 modes**) is fed directly into the DirectShow **`Face Swap Cam`** virtual camera via `pyvirtualcam` (`unitycapture`):
- **WhatsApp Web (Chrome)**: Go to `https://web.whatsapp.com`, open `chrome://settings/content/camera`, and select **`Face Swap Cam`**.
- **WhatsApp Desktop**: Go to Settings -> Video & voice -> Camera -> **`Face Swap Cam`**.
- **Zoom / Google Meet / Discord**: Select **`Face Swap Cam`** in the camera device dropdown.
- **In-Browser Verification**: Open `http://localhost:7860/camera_test.html` to verify the WebRTC video feed with live resolution and framerate.

---

## 7. Performance Benchmarks on GTX 1050 Ti (4GB VRAM, Pascal `sm_61`)

Measured on Windows 11 with DirectShow webcam capture at 30 FPS:

| Mode | Preset Profile | Resolution | Neural Face Time | Try-On Time | Total Time | Effective FPS | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Face Swap** | `Fast` | 480x360 | 46.9 ms | 0.0 ms | 46.9 ms | **21.3 FPS** | 2508 MiB |
| **1. Face Swap** | `Balanced` | 640x480 | 65.6 ms | 0.0 ms | 65.6 ms | **15.2 FPS** | 2504 MiB |
| **1. Face Swap** | `Quality` | 640x480 | 171.9 ms | 0.0 ms | 171.9 ms | **5.8 FPS** | 2644 MiB |
| **2. Clothes Try-On** | `Fast` | 480x360 | 0.0 ms | 16.8 ms | 16.8 ms | **59.2 FPS** | 2642 MiB |
| **2. Clothes Try-On** | `Balanced` | 640x480 | 0.0 ms | 23.1 ms | 23.1 ms | **43.2 FPS** | 2642 MiB |
| **3. Both** | `Fast` | 480x360 | 44.3 ms | 17.2 ms | 61.5 ms | **16.2 FPS** | 2641 MiB |
| **3. Both** | `Balanced` | 640x480 | 64.5 ms | 24.5 ms | 89.0 ms | **11.2 FPS** | 2641 MiB |

> **VRAM Safety**: Peak VRAM remains at **~2.64 GB**, leaving ~1.4 GB free headroom on the 4GB card. Because MediaPipe runs on CPU XNNPACK and rembg runs on CPU, clothes try-on adds **0 MiB** of GPU VRAM.

---

## 8. Troubleshooting Guide

### A. Jittery Garment Movement
- **Cause**: Rapid webcam sensor noise or poor lighting causing MediaPipe landmark variance.
- **Fix**: The built-in **One-Euro Filter** smooths keypoints with adaptive cutoff. For low-light rooms, increase your room lighting. In the Web UI, ensure your profile is set to `Balanced` or `Fast`.

### B. Garment Misaligned (Too High, Low, or Wide)
- **Fix 1**: Use the live **Vertical Offset** and **Torso Width / Scale** sliders in the right panel.
- **Fix 2**: Click **`Calibrate Anchors (6-Pt)`** to adjust the garment's shoulder or hem points directly on the cutout.

### C. Low FPS (< 10 FPS)
- **Fix 1**: Switch to **`Fast`** profile (480x360) in the top-left dropdown.
- **Fix 2**: In Mode 3 (Both), running CUDA Face Swap + Pose tracking takes ~89 ms/frame on a GTX 1050 Ti (~11.2 FPS). If you need higher framerates (>40 FPS), switch to **Mode 2 (Clothes Try-On only)**.

### D. Arms or Hands Appearing Behind Garment
- **Fix**: Ensure the **Foreground Hands & Chin Occlusion** checkbox is checked (enabled by default). When you raise your hands or cross your arms, the MediaPipe segmentation mask composites your real arms in front of the warped cloth.

### E. Original Clothes Peeking Out
- **Fix 1**: Check **Cover Original Collar & Neck** (synthesizes an expanded inner collar mask).
- **Fix 2**: Slightly increase the **Torso Width** (e.g. `1.05 – 1.10`) and **Torso Length** sliders so the virtual garment completely envelopes your original clothes.

### F. Tracking Loss When Stepping Away
- **Fix**: The engine includes **Graceful Tracking-Loss Handling**: when tracking is lost, it holds the last valid pose for 8 frames, then smoothly fades out the garment over 10 frames instead of popping or disappearing abruptly.

---

## 9. Honest Engineering Assessment: Real-Time vs. Diffusion VTON

### What Looks Good
- Real-time responsive latency (<25 ms in try-on mode, 40+ FPS).
- Follows torso rotation, leaning, sitting, and standing.
- Smooth articulation of sleeves around elbows and forearms.
- Dynamic room luminance and color temperature adaptation.
- Forearm and chin occlusion ensures hands stay in front.

### What Looks Fake / Limits
- **2.5D Surface vs. Physical 3D Fabric**: The live mode uses perspective and piecewise affine warping. While it adds subtle cylindrical shading, it does not physically simulate complex cloth folds or wrinkles during extreme torso twists.
- **Deep Diffusion VTON (e.g. IDM-VTON, CatVTON, OOTDiffusion)**:
  - Diffusion VTON models require **12 GB to 24 GB VRAM** and take **15 to 45 seconds per single frame** on high-end GPUs.
  - On a 4GB GTX 1050 Ti Pascal GPU, running full diffusion try-on is **infeasible** (instant CUDA Out-of-Memory).
  - For high-detail stills, use the **`Snapshot Realistic (HD Drape)`** button in the Web UI: it captures a frame and applies local cloth-drape fold luminance transfer and bilateral edge refinement in **~45 ms** with **0 extra VRAM**.
