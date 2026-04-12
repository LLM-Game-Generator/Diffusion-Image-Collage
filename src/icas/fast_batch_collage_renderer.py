import json
import cv2
import time
import numpy as np
import glob
from src.config import *


_GLOBAL_IMG_CACHE = {}


def get_cached_image(image_folder, img_name, max_w, max_h):
    """Retrieves pre-scaled and cached images to significantly reduce I/O overhead."""
    cache_key = f"{img_name}_{max_w}_{max_h}"
    if cache_key in _GLOBAL_IMG_CACHE:
        return _GLOBAL_IMG_CACHE[cache_key]

    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[Warning] Image not found: {img_path}")
        return None

    img_h, img_w = img.shape[:2]

    # Pre-scaling logic
    if img_w > max_w or img_h > max_h:
        scale = min(max_w / img_w, max_h / img_h)
        img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)

    _GLOBAL_IMG_CACHE[cache_key] = img
    return img


def render_single_frame(json_path, output_path, image_folder, scaling_factor=1):
    """
    渲染單一影格。從指定的 json_path 讀取座標，並輸出到 output_path。
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] JSON not found: {json_path}")
        return False

    w = int(data['width'] * scaling_factor)
    h = int(data['height'] * scaling_factor)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA 透明畫布

    # 建立照片對應表
    part_to_img = {img_info['assigned_part']: img_info['filename'] for img_info in data['images']}

    # 渲染每個細胞格子 (Cell)
    for i, part in enumerate(data['parts']):
        img_name = part_to_img.get(i)
        if not img_name:
            continue

        img = get_cached_image(image_folder, img_name, w, h)
        if img is None:
            continue

        coords = (np.array(part['coords']) * scaling_factor).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(coords)

        # 邊界安全檢查
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(bw, w - bx), min(bh, h - by)
        if bw <= 0 or bh <= 0:
            continue

        # 裁切與縮放 (Center Crop)
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
        img_resized = cv2.flip(img_resized, 0)
        img_resized_bgra = cv2.cvtColor(img_resized, cv2.COLOR_BGR2BGRA)

        # 建立局部 Mask 並貼上畫布
        local_coords = coords - [bx, by]
        mask_roi = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [local_coords], 255)

        canvas_roi = canvas[by:by + bh, bx:bx + bw]
        np.copyto(canvas_roi, img_resized_bgra, where=(mask_roi == 255)[..., None])

    # 翻轉回正確視覺座標
    canvas = cv2.flip(canvas, 0)
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True


def batch_render_video_frames(image_folder, json_sequence_dir, output_dir, scaling_factor=1):
    """
    批次處理資料夾內所有的 JSON 檔，生成連續的 PNG 拼貼動畫。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 抓取所有 json 檔案並照檔名排序 (例如 slicing_0000.json, slicing_0001.json...)
    json_files = sorted(glob.glob(os.path.join(json_sequence_dir, "*.json")))

    if not json_files:
        print(f"[Warning] No JSON files found in {json_sequence_dir}")
        return

    print(f"🚀 Starting batch render for {len(json_files)} frames...")
    start_time = time.perf_counter()

    for idx, json_path in enumerate(json_files):
        # 動態生成輸出檔名 collage_0000.png, collage_0001.png
        out_filename = f"collage_{idx:04d}.png"
        output_path = os.path.join(output_dir, out_filename)

        success = render_single_frame(json_path, output_path, image_folder, scaling_factor)
        if success:
            print(f"  [Rendered] {out_filename}")

    end_time = time.perf_counter()
    print(f"✅ Batch rendering complete! Total time: {end_time - start_time:.4f} seconds.")


if __name__ == '__main__':
    batch_render_video_frames(
        image_folder=ASSETS_IMAGES_DIR,
        json_sequence_dir=JSON_SEQUENCE_DIR,
        output_dir=COLLAGE_SEQUENCE_DIR,
        scaling_factor=2
    )