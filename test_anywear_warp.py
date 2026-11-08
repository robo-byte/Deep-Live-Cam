#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np

def build_torso_curvilinear_map(
    garment_rgba: np.ndarray,
    anc: dict,
    dst_l_sh: np.ndarray,
    dst_r_sh: np.ndarray,
    dst_l_hp: np.ndarray,
    dst_r_hp: np.ndarray,
    sh_unit: np.ndarray,
    down_unit: np.ndarray,
    sh_w: float,
    yaw_factor: float,
    frame_size: tuple,
    grid_n: int = 7,
):
    fw, fh = frame_size
    gh, gw = garment_rgba.shape[:2]

    # Source corners on garment
    src_l_sh = np.array(anc["left_shoulder"], dtype=np.float32)
    src_r_sh = np.array(anc["right_shoulder"], dtype=np.float32)
    src_l_hp = np.array(anc["left_hem"], dtype=np.float32)
    src_r_hp = np.array(anc["right_hem"], dtype=np.float32)

    # 1. Compute Destination Mesh Grid Points (grid_n x grid_n) in camera frame space
    dst_grid = np.zeros((grid_n, grid_n, 2), dtype=np.float32)
    src_grid = np.zeros((grid_n, grid_n, 2), dtype=np.float32)

    sh_center = 0.5 * (dst_l_sh + dst_r_sh)
    hp_center = 0.5 * (dst_l_hp + dst_r_hp)
    torso_len = float(np.linalg.norm(hp_center - sh_center))

    for j in range(grid_n):
        v = j / float(grid_n - 1)  # 0.0 (shoulders) to 1.0 (hem)
        
        # Anatomical vertical spine line
        spine_pt = sh_center + down_unit * (v * torso_len)
        
        # Natural body width variation along torso:
        # v=0.0: shoulders
        # v=0.30: mid-chest / bust (full width + slight chest convexity)
        # v=0.62: natural waist (tapered ~92% width)
        # v=1.00: hip hem (curved pelvis band)
        width_profile = 1.0 - 0.08 * math.sin(math.pi * v) ** 2
        
        # 3D Depth / Cylindrical convexity forward displacement (droop curve)
        # Makes the center curve forward like an actual cylindrical torso
        convexity = math.sin(math.pi * v) * (0.05 * sh_w)
        
        # Left and right side endpoints along the body contour
        left_edge = dst_l_sh * (1.0 - v) + dst_l_hp * v
        right_edge = dst_r_sh * (1.0 - v) + dst_r_hp * v
        
        # Bottom hem arch (curving naturally down around hips)
        hem_arch = down_unit * (math.sin(math.pi * 0.5) * 0.04 * sh_w * (v ** 1.5))

        for i in range(grid_n):
            u = i / float(grid_n - 1)  # 0.0 (left seam) to 1.0 (right seam)
            
            # Yaw perspective compression (one side recedes when turning)
            u_warped = u + yaw_factor * math.sin(math.pi * u) * 0.22
            u_warped = np.clip(u_warped, 0.0, 1.0)
            
            # Destination camera point
            # Linear base
            base_pt = left_edge * (1.0 - u_warped) + right_edge * u_warped
            # Add cylindrical arch across chest/hem
            arch = down_unit * (math.sin(math.pi * u_warped) * (0.04 * sh_w + convexity))
            dst_pt = base_pt + arch + hem_arch
            dst_grid[j, i] = dst_pt

            # Source garment coordinate
            # Bilinear across garment anchor quad
            src_top = src_l_sh * (1.0 - u) + src_r_sh * u
            src_bot = src_l_hp * (1.0 - u) + src_r_hp * u
            src_pt = src_top * (1.0 - v) + src_bot * v
            src_grid[j, i] = src_pt

    # 2. Warp entire garment bounding box using cv2.remap on bounding box ROI
    # Find bounding box of dst_grid in camera frame
    min_x = max(0, int(np.floor(dst_grid[:, :, 0].min())) - 16)
    min_y = max(0, int(np.floor(dst_grid[:, :, 1].min())) - 16)
    max_x = min(fw, int(np.ceil(dst_grid[:, :, 0].max())) + 16)
    max_y = min(fh, int(np.ceil(dst_grid[:, :, 1].max())) + 16)

    roi_w = max_x - min_x
    roi_h = max_y - min_y
    if roi_w <= 10 or roi_h <= 10:
        return np.zeros((fh, fw, 4), dtype=np.uint8)

    # Invert the mapping: for each pixel in dst ROI, what is the (src_x, src_y) in garment?
    # Generate piecewise mesh or bicubic spline mapping:
    # Use cv2.remap with dense inverse grid
    # A fast, extremely robust method:
    # Build forward grid and interpolate dense map_x, map_y
    # In OpenCV, grid interpolation via Delaunay triangles / piecewise affine is exact:
    warped_rgba = np.zeros((fh, fw, 4), dtype=np.uint8)
    
    # Fast triangular mesh warp:
    for j in range(grid_n - 1):
        for i in range(grid_n - 1):
            # Two triangles per quad:
            # T1: (i, j), (i+1, j), (i, j+1)
            # T2: (i+1, j), (i+1, j+1), (i, j+1)
            for tri_idx in (0, 1):
                if tri_idx == 0:
                    src_tri = np.float32([src_grid[j, i], src_grid[j, i+1], src_grid[j+1, i]])
                    dst_tri = np.float32([dst_grid[j, i], dst_grid[j, i+1], dst_grid[j+1, i]])
                else:
                    src_tri = np.float32([src_grid[j, i+1], src_grid[j+1, i+1], src_grid[j+1, i]])
                    dst_tri = np.float32([dst_grid[j, i+1], dst_grid[j+1, i+1], dst_grid[j+1, i]])
                
                # Check bounding box
                bx0 = max(0, int(np.floor(dst_tri[:, 0].min())))
                by0 = max(0, int(np.floor(dst_tri[:, 1].min())))
                bx1 = min(fw, int(np.ceil(dst_tri[:, 0].max())) + 1)
                by1 = min(fh, int(np.ceil(dst_tri[:, 1].max())) + 1)
                bw, bh_t = bx1 - bx0, by1 - by0
                if bw <= 0 or bh_t <= 0:
                    continue
                
                # Shift triangles to ROI
                dst_tri_roi = dst_tri.copy()
                dst_tri_roi[:, 0] -= bx0
                dst_tri_roi[:, 1] -= by0
                
                mask_roi = np.zeros((bh_t, bw), dtype=np.uint8)
                cv2.fillConvexPoly(mask_roi, np.int32(dst_tri_roi), 255, lineType=cv2.LINE_AA)
                
                # Check src triangle bounding box
                sx0 = max(0, int(np.floor(src_tri[:, 0].min())))
                sy0 = max(0, int(np.floor(src_tri[:, 1].min())))
                sx1 = min(gw, int(np.ceil(src_tri[:, 0].max())) + 1)
                sy1 = min(gh, int(np.ceil(src_tri[:, 1].max())) + 1)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                
                src_tri_roi = src_tri.copy()
                src_tri_roi[:, 0] -= sx0
                src_tri_roi[:, 1] -= sy0
                
                src_patch = garment_rgba[sy0:sy1, sx0:sx1]
                M = cv2.getAffineTransform(src_tri_roi, dst_tri_roi)
                warped_patch = cv2.warpAffine(
                    src_patch, M, (bw, bh_t),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                
                # Composite into warped_rgba
                cur_target = warped_rgba[by0:by1, bx0:bx1]
                alpha_w = (warped_patch[:, :, 3].astype(np.float32) / 255.0) * (mask_roi.astype(np.float32) / 255.0)
                cur_a = cur_target[:, :, 3].astype(np.float32) / 255.0
                
                new_a = np.maximum(cur_a, alpha_w)
                # Where alpha_w > 0, blend
                mask_active = alpha_w > 0.05
                cur_target[mask_active] = warped_patch[mask_active]

    return warped_rgba

print("Torso Curvilinear Mesh module compiled successfully!")
