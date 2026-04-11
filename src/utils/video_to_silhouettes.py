import cv2
import os
import numpy as np
from rembg import remove, new_session
from PIL import Image
from src.config import VIDEO_PATH, FRAME_DIR, TARGET_SIZE, FRAME_INTERVAL


def video_to_silhouettes(video_path: str, output_dir: str, target_size: int = None, frame_interval: int = None):
    if not os.path.exists(video_path):
        print(f"❌ 錯誤：找不到影片檔案：{video_path}")
        return

    if target_size is None:
        target_size = TARGET_SIZE
    if frame_interval is None:
        frame_interval = FRAME_INTERVAL


    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = new_session("u2net", providers=providers)
        print("🚀 已啟動去背引擎 (優先使用 GPU)")
    except Exception as e:
        print(f"⚠️ GPU 初始化失敗: {e}, 切換至純 CPU 模式")
        session = new_session("u2net", providers=['CPUExecutionProvider'])

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    # 獲取原始影片資訊
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"原始解析度: {orig_w}x{orig_h} -> 目標解析度: {target_size}x{target_size}")

    frame_count = 0
    print(f"🎬 開始中心裁剪並處理影片...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # --- [功能新增：中心裁剪成正方形] ---
            # 1. 計算裁剪範圍
            min_side = min(orig_w, orig_h)
            x0 = (orig_w - min_side) // 2
            y0 = (orig_h - min_side) // 2

            # 2. 執行裁剪 (Crop)
            square_frame = frame[y0:y0 + min_side, x0:x0 + min_side]

            # 3. 縮放至目標大小 (Resize)
            # 使用 INTER_AREA 對於縮小影像有較好的抗鋸齒效果
            resized_frame = cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # --- 後續去背流程 ---
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # 執行去背
            output_rgba = remove(img_rgb, session=session)
            alpha = output_rgba[:, :, 3]

            # 二值化處理 (白底黑人)
            silhouette = np.where(alpha > 100, 0, 255).astype(np.uint8)

            # 轉回 RGB 儲存 (確保 ICAS 相容性)
            final_img = Image.fromarray(silhouette).convert("RGB")

            save_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            final_img.save(save_path)

            if frame_count % 30 == 0:
                print(f"⏳ 進度: 已處理第 {frame_count} 影格 (已縮放至 {target_size})")

        frame_count += 1

    cap.release()
    print(f"✨ 處理完成！所有影格已中心裁剪並縮放至 {target_size}x{target_size}")
    print(f"📂 檔案存放於: {output_dir}")


if __name__ == "__main__":
    print(VIDEO_PATH)
    video_to_silhouettes(VIDEO_PATH, FRAME_DIR, TARGET_SIZE, FRAME_INTERVAL)