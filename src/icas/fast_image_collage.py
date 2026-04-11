import json
import cv2
import time
import numpy as np
import os
from src.config import *

# Global Cache: Allows different frames within the same process to share image memory,
# completely eliminating redundant cv2.imread calls.
_GLOBAL_IMG_CACHE = {}


def get_cached_image(image_folder, img_name, max_w, max_h):
    """Retrieves pre-scaled and cached images to significantly reduce I/O overhead."""
    cache_key = f"{img_name}_{max_w}_{max_h}"
    if cache_key in _GLOBAL_IMG_CACHE:
        return _GLOBAL_IMG_CACHE[cache_key]

    img_path = os.path.join(image_folder, img_name)
    # If the file does not exist, cv2.imread returns None instead of raising an error
    img = cv2.imread(img_path)

    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    # Pre-scaling logic
    if img_w > max_w or img_h > max_h:
        scale = min(max_w / img_w, max_h / img_h)
        img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)

    _GLOBAL_IMG_CACHE[cache_key] = img
    return img


def fast_render_collage(image_folder, json_dir, scaling_factor=1):
    """
    Extreme I/O Optimized Version:
    Incorporates a Global Cache to prevent multiple frames from repeatedly reading the same assets.
    """
    json_path = os.path.join(json_dir, 'slicing_result.json')
    # Removed os.path.exists check; use try-except directly to save one Disk I/O operation
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return

    # 1. Initialize canvas [Correction: Use 4-channel BGRA, default to 0 (fully transparent)]
    w = int(data['width'] * scaling_factor)
    h = int(data['height'] * scaling_factor)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # 2. Create image mapping table
    part_to_img = {}
    for img_info in data['images']:
        part_to_img[img_info['assigned_part']] = img_info['filename']

    # 3. Render each cell
    for i, part in enumerate(data['parts']):
        img_name = part_to_img.get(i)
        if not img_name:
            continue

        # Use global cache to retrieve image, passing max canvas dimensions for pre-scaling
        img = get_cached_image(image_folder, img_name, w, h)
        if img is None:
            continue

        # Get scaled coordinates
        coords = (np.array(part['coords']) * scaling_factor).astype(np.int32)

        # Get the bounding box of the polygon
        bx, by, bw, bh = cv2.boundingRect(coords)

        # Boundary safety checks
        bx = max(0, bx)
        by = max(0, by)
        bw = min(bw, w - bx)
        bh = min(bh, h - by)
        if bw <= 0 or bh <= 0:
            continue

        # Center cropping and asset scaling
        img_h, img_w = img.shape[:2]
        target_aspect = bw / bh if bh > 0 else 1
        img_aspect = img_w / img_h if img_h > 0 else 1

        if img_aspect > target_aspect:
            new_w = int(img_h * target_aspect)
            start_x = (img_w - new_w) // 2
            img_cropped = img[:, start_x:start_x + new_w]
        else:
            new_h = int(img_w / target_aspect)
            start_y = (img_h - new_h) // 2
            img_cropped = img[start_y:start_y + new_h, :]

        img_resized = cv2.resize(img_cropped, (bw, bh), interpolation=cv2.INTER_LINEAR)

        # Fix inverted image issue (double negatives cancel out)
        img_resized = cv2.flip(img_resized, 0)

        # [Correction: Convert image to BGRA (4-channel) to provide an opaque Alpha channel]
        img_resized_bgra = cv2.cvtColor(img_resized, cv2.COLOR_BGR2BGRA)

        # Local Mask
        local_coords = coords - [bx, by]
        mask_roi = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [local_coords], 255)

        # Capture ROI to accelerate bitwise operations (canvas_roi is also 4-channel here)
        canvas_roi = canvas[by:by + bh, bx:bx + bw]

        # Paste onto canvas (map the processed BGRA image onto the transparent canvas)
        np.copyto(canvas_roi, img_resized_bgra, where=(mask_roi == 255)[..., None])

    # Flip image (convert ICAS coordinate system to the correct visual coordinate system)
    canvas = cv2.flip(canvas, 0)

    save_path = os.path.join(json_dir, 'collage.png')
    cv2.imwrite(save_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])


if __name__ == '__main__':
    # Test execution time for the first run (Cache warming)
    start = time.perf_counter()
    fast_render_collage(ASSETS_IMAGES_DIR, os.path.join(JSON_DIR, "frame_0000"), scaling_factor=2)
    end = time.perf_counter()
    print(f"render_collage execution time 1: {end - start:.6f} seconds")

    # Test execution time for the second run (Hit cache)
    start = time.perf_counter()
    fast_render_collage(ASSETS_IMAGES_DIR, os.path.join(JSON_DIR, "frame_0000"), scaling_factor=2)
    end = time.perf_counter()
    print(f"render_collage execution time 2: {end - start:.6f} seconds")