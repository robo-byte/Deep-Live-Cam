# Deep-Live-Cam — Real-Time Webcam Face Swap (`GTX 1050 Ti 4GB` + `Face Swap Cam`)

> **ETHICAL USE & NON-IMPERSONATION NOTICE**
> This local setup is intended **strictly for personal entertainment, creative character experiments, and educational research**. It must **never** be used to impersonate real individuals without their explicit consent, deceive others in video calls, bypass identity verification, or generate non-consensual synthetic media.

---

## 1. Quick Start (How to Start)

### Option A: Standard GUI Launch (Recommended)
Double-click **`start_faceswap.bat`** in `d:\ANTI\deep-live-cam\` (or run it from a terminal):
```bat
d:\ANTI\deep-live-cam\start_faceswap.bat
```
1. The app opens with the **Balanced** profile (`640x480`, `det_size=320`, `frame_skip=2`), `CUDAExecutionProvider` active on your GTX 1050 Ti, `Face Swap Cam` Virtual Camera enabled, and `faces\default_character.jpg` pre-loaded.
2. In the **Camera** section at the bottom:
   - Make sure your USB webcam (`A4tech FHD 1080P PC Camera`) is plugged in. If you just plugged it in, click the **`🔄`** button next to the camera dropdown to refresh the camera list.
   - *(If your physical USB webcam is unplugged, select **`Demo Camera (Built-in Test Video)`** to test the live swap and virtual camera immediately.)*
3. Click **`Live`**.
   - The **Live Preview** window opens showing your real-time swapped video and FPS counter (`FPS: ... | BALANCED | VCam:ON`).
   - Simultaneously, the clean (un-watermarked) swapped video streams directly to the **`Face Swap Cam`** DirectShow Virtual Camera!

### Option B: One-Click Auto-Live Launch
Double-click **`start_faceswap_autolive.bat`**:
```bat
d:\ANTI\deep-live-cam\start_faceswap_autolive.bat
```
This opens the window and **automatically starts** the Live Webcam + `Face Swap Cam` stream after 0.4 seconds.

### Option C: Launch with a Specific Profile (`fast`, `balanced`, `quality`)
```bat
d:\ANTI\deep-live-cam\start_faceswap.bat fast
d:\ANTI\deep-live-cam\start_faceswap.bat balanced
d:\ANTI\deep-live-cam\start_faceswap.bat quality
```
You can also switch profiles on the fly inside the GUI using the **GPU Profile** dropdown.

---

## 2. How to Change the Source Character Face

You have **three ways** to use your own character image (`.jpg`, `.png`, `.webp`):

1. **In the GUI (even while Live mode is running!)**:
   - Click **`Select a face`** (top-left button under *Source face*) and choose any clear, front-facing character photo. The live webcam swap updates immediately!
2. **Pass the image path to `start_faceswap.bat`** (or drag-and-drop an image file onto `start_faceswap.bat` in Windows Explorer):
   ```bat
   d:\ANTI\deep-live-cam\start_faceswap.bat "C:\path\to\my_character.jpg"
   ```
3. **Replace the default face**:
   - Save your preferred character image as `d:\ANTI\deep-live-cam\faces\default_character.jpg` (an alternative sample face is also provided at `d:\ANTI\deep-live-cam\faces\alt_character.jpg`).

> **Tips for the Best Source Image**:
> - Use a clear, well-lit, front-facing face without heavy sunglasses or extreme head tilt.
> - Only **one** clear face should be in the source image.

---

## 3. How to Stop

- **Stop Live Webcam / Virtual Camera only**: Close the **Live Preview** window (`X` in the top-right corner).
- **Stop and exit the entire application**: Click the red **`Destroy`** button in the main window, close the main window (`X`), or press `Ctrl+C` in the launcher terminal window.

---

## 4. Using `Face Swap Cam` in WhatsApp Web & WhatsApp Desktop

> **Important Rule**: Always start **`start_faceswap.bat`** and click **`Live`** **BEFORE** starting your video call so `Face Swap Cam` is actively transmitting frames. Also make sure no other app is exclusively locking your physical USB webcam (`A4tech FHD 1080P PC Camera`) before `Deep-Live-Cam` opens it.

### A. WhatsApp Web in Google Chrome (`https://web.whatsapp.com`)
1. Start `start_faceswap.bat` and click **`Live`** (so `Deep-Live-Cam` captures your physical `A4tech FHD 1080P PC Camera` and outputs to `Face Swap Cam`).
2. Open Chrome and go to `https://web.whatsapp.com` (or open `d:\ANTI\deep-live-cam\camera_test.html` in Chrome first to verify).
3. Click the **Site Settings** icon (tune/lock icon on the left side of the Chrome address bar) -> **Site settings** -> **Camera** -> Ensure it is set to **Allow**.
4. To set **`Face Swap Cam`** as Chrome's default camera:
   - Open `chrome://settings/content/camera` in a Chrome tab.
   - Select **`Face Swap Cam`** from the dropdown menu at the top.
5. In WhatsApp Web, start a video call (or click the **`...` / Camera icon** in the call control bar) and select **`Face Swap Cam`**.

### B. WhatsApp Desktop (Windows Store App)
1. Start `start_faceswap.bat` and click **`Live`** first.
2. Open **WhatsApp Desktop** -> click the **Gear icon (`Settings`)** in the bottom-left corner -> click **Video & voice**.
3. Under **Camera**, open the dropdown and select **`Face Swap Cam`**.
4. During an active WhatsApp Desktop video call, you can also click the **arrow (`^`) next to the Camera button** to switch to **`Face Swap Cam`**.

#### What to do if WhatsApp Desktop does NOT offer camera selection (or defaults to the wrong camera):
- **Workaround 1 (Automatic Hardware Lock Trick)**: Because `Deep-Live-Cam` opens your physical USB webcam (`A4tech FHD 1080P PC Camera`) first via DirectShow, your physical USB camera is already busy/locked, which forces many versions of WhatsApp Desktop to automatically fall back to the next available camera (**`Face Swap Cam`**).
- **Workaround 2 (Windows 11 System Settings)**:
  - Open Windows **Settings (`Win + I`)** -> **Bluetooth & devices** -> **Cameras**.
  - Check that **`Face Swap Cam`** is listed under Connected cameras.
- **Workaround 3 (Guaranteed — Use WhatsApp Web in Chrome)**:
  - If your build of WhatsApp Desktop ignores camera selection, use **WhatsApp Web (`https://web.whatsapp.com`) in Chrome** with `chrome://settings/content/camera` set to **`Face Swap Cam`**, which always honors the selected virtual camera.

---

## 5. GTX 1050 Ti (4GB VRAM, Pascal `sm_61`) Performance & Honest Assessment

### Why the GTX 1050 Ti Requires Special Tuning
- Your **NVIDIA GeForce GTX 1050 Ti** uses the **Pascal `GP107` (`sm_61`)** architecture (768 CUDA cores, 2.1 TFLOPS FP32, **no Tensor Cores**, and 1:64 native FP16 throughput).
- NVIDIA dropped Pascal `sm_61` support in CUDA 13 and in `cuDNN >= 9.11.0`. We pinned **`onnxruntime-gpu==1.21.1` + `nvidia-cudnn-cu12==9.10.2.21` + CUDA 12.9**, which keeps `CUDAExecutionProvider` 100% active on your GPU.
- Furthermore, `inswapper_128_fp16.onnx` overflows `float16` (`65504`) inside its 12 `InstanceNormalization` blocks on Pascal GPUs (producing `NaN` black boxes) and runs slower due to 1:64 FP16 emulation. We therefore use **`inswapper_128_fused.onnx`** (FP32 with all 24 `Pad(reflect) + Conv` pairs fused into single `Conv` kernels + **CUDA Graph + GPU IOBinding**).

### Honest Real-Time Performance Breakdown
A single forward pass of `inswapper_128` requires **~272 GFLOPs** across twelve `1024x1024` 3x3 residual convolution layers:
- **Pure Synchronous Every-Frame Neural Swap (`frame_skip = 0`, Quality Profile)**:
  - Runs at **~6.0 – 6.6 FPS** (`~150–165 ms` per frame) on a GTX 1050 Ti.
  - **Honest Verdict**: Pure every-frame synchronous neural swapping cannot reach 15+ FPS on Pascal `sm_61` hardware due to the physical 2.1 TFLOPS FP32 ceiling.
- **Pose-Warped Frame Skipping (`frame_skip = 2` in Balanced, `frame_skip = 3` in Fast)**:
  - Because `RetinaFace` (`det_10g.onnx` at `160x160` or `320x320`) takes only **6.8–11.5 ms** on CUDA and our SIMD affine paste-back (`_fast_paste_back`) takes only **1.2 ms**, we run full neural `inswapper_128` every 3rd or 4th frame and on intermediate frames warp the cached `128x128` swapped face onto your **live current-frame 5-point facial keypoints (`target_face.kps`)**.
  - This achieves **16–22+ FPS effective throughput** with real-time head tracking while keeping total VRAM usage around **~2.7–2.9 GB** (safely below your 3.5 GB limit).

### Recommended GPU Upgrade for Native 30–60 FPS Synchronous Swapping
If you want **30–60 FPS synchronous every-frame neural swapping** (plus real-time `GFPGAN` / `GPEN-256` face enhancement) without frame skipping:
- **Best Budget Used Upgrade**: **NVIDIA RTX 2060 Super (8GB)** or **RTX 3060 (12GB)** — adds Tensor Cores + native FP16 execution (`inswapper_128_fp16.onnx` drops from `150 ms` to **`~12–16 ms` (`60+ FPS`)**).
- **Best Modern Low-Power Upgrade**: **NVIDIA RTX 4060 (8GB, 115W)** — ~7.5× faster FP32 and ~30× faster FP16 Tensor Core throughput than the GTX 1050 Ti; easily runs `inswapper_128_fp16` + `GFPGAN` at **30+ FPS at 720p/1080p**.

---

## 6. Troubleshooting

1. **Low FPS (`< 10 FPS`)**:
   - Switch the **GPU Profile** dropdown to **`Fast (480x360, det 160, skip 3)`** or run `start_faceswap.bat fast`.
   - Ensure **Face Enhancer** is set to **`None`** (GFPGAN takes ~800 ms/frame on Pascal and exceeds 3.5 GB VRAM).
   - Ensure **Many faces**, **Poisson Blend**, and **Mouth Mask** are toggled **OFF**.
2. **Black Screen or Color Pattern in `Face Swap Cam` Virtual Camera**:
   - `Face Swap Cam` only outputs video **while the `Live Preview` window is open and running**. Click **`Live`** in `Deep-Live-Cam` first, then open/refresh your camera in Chrome or WhatsApp.
   - Make sure the **`Face Swap Cam`** toggle switch in the Camera card is turned **ON**.
3. **Face Not Detected**:
   - Improve front lighting on your face and face the camera directly.
   - If you are sitting far from the webcam, increase **Det size** from `160 x 160` to `320 x 320` or `480 x 480`.
   - For the **Source face**, use a clear, front-facing portrait image where the face is unobstructed.
4. **Wrong Camera Selected / Physical USB Camera Missing**:
   - Plug in your USB webcam (`A4tech FHD 1080P PC Camera`), click the **`🔄`** button next to `Select Camera`, and pick your USB webcam from the dropdown.
   - Never close `Deep-Live-Cam` while in a call; make sure Chrome/WhatsApp isn't already holding your physical USB camera open before you click **`Live`**.

---

## 7. Official Sources & License Terms

All components and models were downloaded from their official upstream repositories and run **100% locally offline**:
- **Application**: [`hacksider/Deep-Live-Cam`](https://github.com/hacksider/Deep-Live-Cam) (v2.1.6) — open-source MIT-based license with non-commercial/ethical use provisions.
- **Face Detection & Recognition (`buffalo_l` pack: `det_10g.onnx`, `w600k_r50.onnx`, `2d106det.onnx`)**: Official [InsightFace](https://github.com/deepinsight/insightface) model zoo — licensed for **non-commercial research and educational use only**.
- **Face Swapper (`inswapper_128.onnx` / `inswapper_128_fp16.onnx`)**: Official InsightFace release mirrored on HuggingFace (`deepinsight/inswapper` / `hacksider/deep-live-cam`) — **non-commercial research and personal entertainment use only**.
- **Virtual Camera Driver (`UnityCaptureFilter64.dll`)**: Official [`schellingb/UnityCapture`](https://github.com/schellingb/UnityCapture) DirectShow filter — **MIT License**.
