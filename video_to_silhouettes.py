import cv2
import os
import numpy as np
from rembg import remove, new_session
from PIL import Image

# --- 設定參數 ---
VIDEO_PATH = "./videos/46012-448062061_medium.mp4"
OUTPUT_DIR = "D:/NCKU/CSIE-Project/Colla/input_silhouette_frames/"
TARGET_SIZE = 512  # 使用者想要的最終大小 (512x512)
FRAME_INTERVAL = 3  # 每 3 影格處理一次

# --- 強健 Session 初始化 ---
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    session = new_session("u2net", providers=providers)
    print("🚀 已啟動去背引擎 (優先使用 GPU)")
except Exception as e:
    print(f"⚠️ GPU 初始化失敗: {e}, 切換至純 CPU 模式")
    session = new_session("u2net", providers=['CPUExecutionProvider'])

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

# 獲取原始影片資訊
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"原始解析度: {orig_w}x{orig_h} -> 目標解析度: {TARGET_SIZE}x{TARGET_SIZE}")

frame_count = 0
print(f"🎬 開始中心裁剪並處理影片...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        # --- [功能新增：中心裁剪成正方形] ---
        # 1. 計算裁剪範圍
        min_side = min(orig_w, orig_h)
        x0 = (orig_w - min_side) // 2
        y0 = (orig_h - min_side) // 2

        # 2. 執行裁剪 (Crop)
        square_frame = frame[y0:y0 + min_side, x0:x0 + min_side]

        # 3. 縮放至目標大小 (Resize)
        # 使用 INTER_AREA 對於縮小影像有較好的抗鋸齒效果
        resized_frame = cv2.resize(square_frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # --- 後續去背流程 ---
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # 執行去背
        output_rgba = remove(img_rgb, session=session)
        alpha = output_rgba[:, :, 3]

        # 二值化處理 (白底黑人)
        silhouette = np.where(alpha > 100, 0, 255).astype(np.uint8)

        # 轉回 RGB 儲存 (確保 ICAS 相容性)
        final_img = Image.fromarray(silhouette).convert("RGB")

        save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.png")
        final_img.save(save_path)

        if frame_count % 30 == 0:
            print(f"⏳ 進度: 已處理第 {frame_count} 影格 (已縮放至 {TARGET_SIZE})")

    frame_count += 1

cap.release()
print(f"✨ 處理完成！所有影格已中心裁剪並縮放至 {TARGET_SIZE}x{TARGET_SIZE}")
print(f"📂 檔案存放於: {OUTPUT_DIR}")