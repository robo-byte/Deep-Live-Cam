# GTX 1050 Ti (4GB VRAM) Real-Time Face Swap & Virtual Clothes Try-On Benchmark Report

- **GPU**: NVIDIA GeForce GTX 1050 Ti (Pascal `sm_61`, 4096 MiB VRAM, Driver 582.66)
- **Face Swap Backend**: `CUDAExecutionProvider` (`inswapper_128_fused.onnx` FP32 + CUDA Graph + IOBinding)
- **Clothes Try-On Backend**: MediaPipe Pose Lite (`model_complexity=0`) + SelfieSegmentation on CPU XNNPACK (**0 MiB extra GPU VRAM**) + OpenCV Perspective Torso & Articulated Sleeve Warping
- **Garment Cutout Backend**: `rembg` (`u2netp.onnx` on CPU, cached once per garment in `garments/cache/`)
- **Virtual Camera Verified**: `Face Swap Cam` (`640x480` DirectShow output verified across all 3 modes)

## All 3 Modes Benchmark Results

| Mode | Profile | Input Res | Face Swap Time | Clothes Try-On Time | Total Frame Time | Effective Live FPS | Peak System VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. FACE SWAP** | `Fast` | `480x360` | `46.8 ms` | `0.0 ms` | `46.8 ms` | **21.3 FPS** | `3037 / 4096 MiB` |
| **1. FACE SWAP** | `Balanced` | `640x480` | `68.5 ms` | `0.0 ms` | `68.5 ms` | **14.6 FPS** | `3045 / 4096 MiB` |
| **1. FACE SWAP** | `Quality` | `640x480` | `185.5 ms` | `0.0 ms` | `185.5 ms` | **5.4 FPS** | `3180 / 4096 MiB` |
| **2. CLOTHES TRY-ON** | `Fast` | `480x360` | `0.0 ms` | `17.6 ms` | `17.6 ms` | **56.6 FPS** | `3173 / 4096 MiB` |
| **2. CLOTHES TRY-ON** | `Balanced` | `640x480` | `0.0 ms` | `23.3 ms` | `23.3 ms` | **42.7 FPS** | `3173 / 4096 MiB` |
| **3. BOTH (Face + Clothes)** | `Fast` | `480x360` | `46.4 ms` | `17.3 ms` | `63.8 ms` | **15.7 FPS** | `3171 / 4096 MiB` |
| **3. BOTH (Face + Clothes)** | `Balanced` | `640x480` | `68.7 ms` | `24.9 ms` | `93.7 ms` | **10.7 FPS** | `3171 / 4096 MiB` |
